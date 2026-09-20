"""Métricas del desafío.

Jerarquía explícita (plan §5):

1. **PR-AUC out-of-fold** es el número de selección de modelo. ROC-AUC de apoyo,
   Brier para calibración, accuracy nunca: el problema está desbalanceado por
   construcción y la accuracy premia al modelo que dice "sano" siempre.
2. **Curva de anticipación vs. falsas alarmas** es el número de portada del pitch.
3. **Bootstrap sobre los folds** para no confundir una diferencia con un ruido.
4. **Las mismas métricas a nivel vehículo** (`vehicle_scores` / `vehicle_metrics`):
   Ford marca autos, no cortes, así que la unidad de decisión es el vehículo —una
   bolsa de cortes, en el marco de Multiple-Instance Learning—. No reemplaza al
   PR-AUC por fila (cambia la tasa base, así que los dos números no se comparan
   entre sí: se compara el lift).

Definición operativa de alerta (la que evita inflar la anticipación): un vehículo
queda alertado cuando su score supera el umbral en `k_consecutive` cortes
seguidos. Sin esa condición, un pico de ruido aislado cuenta como acierto.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

REQUIRED_COLUMNS = ("vehicle_id", "cut_odo", "score", "event_observed", "time_to_event_km")


# --------------------------------------------------------------------------- #
# Métricas de clasificación
# --------------------------------------------------------------------------- #
def pr_auc(y_true: Sequence[int], y_score: Sequence[float]) -> float:
    """PR-AUC (average precision). NaN si hay una sola clase presente."""
    y_true = np.asarray(y_true, dtype=int)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, np.asarray(y_score, dtype=float)))


def roc_auc(y_true: Sequence[int], y_score: Sequence[float]) -> float:
    y_true = np.asarray(y_true, dtype=int)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, np.asarray(y_score, dtype=float)))


def brier(y_true: Sequence[int], y_score: Sequence[float]) -> float:
    return float(brier_score_loss(np.asarray(y_true, dtype=int), np.asarray(y_score, dtype=float)))


def classification_metrics(y_true: Sequence[int], y_score: Sequence[float]) -> dict[str, float]:
    """PR-AUC, ROC-AUC, Brier, tasa base y lift sobre la tasa base."""
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    base_rate = float(y_true.mean()) if len(y_true) else float("nan")
    ap = pr_auc(y_true, y_score)
    return {
        "pr_auc": ap,
        "roc_auc": roc_auc(y_true, y_score),
        "brier": brier(y_true, y_score),
        "base_rate": base_rate,
        "pr_auc_lift": float(ap / base_rate) if base_rate else float("nan"),
        "n": int(len(y_true)),
        "n_positive": int(y_true.sum()),
    }


# --------------------------------------------------------------------------- #
# Métricas de anticipación
# --------------------------------------------------------------------------- #
def _validate_predictions(predictions: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in predictions.columns]
    if missing:
        raise KeyError(
            f"Faltan columnas para las métricas de anticipación: {missing}. "
            f"Se esperan {list(REQUIRED_COLUMNS)}."
        )


def _first_sustained_index(flags: np.ndarray, k: int) -> int | None:
    """Índice del primer corte donde arranca una racha de `k` superaciones del umbral."""
    if k <= 1:
        hits = np.flatnonzero(flags)
        return int(hits[0]) if hits.size else None
    if flags.size < k:
        return None
    runs = np.convolve(flags.astype(int), np.ones(k, dtype=int), mode="valid")
    hits = np.flatnonzero(runs == k)
    return int(hits[0]) if hits.size else None


def _vehicle_sequences(predictions: pd.DataFrame) -> dict[str, dict[str, np.ndarray | int]]:
    """Agrupa por vehículo y ordena por odómetro (una sola vez, se reusa por umbral)."""
    _validate_predictions(predictions)

    # Un corte repetido rompe la regla de alerta sostenida: la misma fila contada dos
    # veces hace que un pico aislado parezca una racha de K, y la anticipación sale
    # inflada. Pasa si alguien serializa las R repeticiones de la CV repetida en
    # formato largo (`run_cv` guarda el promedio justamente para evitarlo) o si el
    # panel tiene varios horizontes por corte. Falla acá, no en el número del pitch.
    duplicated = predictions.duplicated(subset=["vehicle_id", "cut_odo"]).sum()
    if duplicated:
        raise ValueError(
            f"{duplicated} fila(s) con `(vehicle_id, cut_odo)` repetido. Las métricas de "
            "anticipación necesitan una fila por corte: promediá las repeticiones (ver "
            "`src/training/cv.py::run_cv`) o filtrá un solo horizonte antes de llamar."
        )

    ordered = predictions.sort_values(["vehicle_id", "cut_odo"])
    sequences: dict[str, dict[str, np.ndarray | int]] = {}
    for vehicle_id, group in ordered.groupby("vehicle_id", observed=True, sort=False):
        sequences[str(vehicle_id)] = {
            "score": group["score"].to_numpy(dtype=float),
            "time_to_event_km": group["time_to_event_km"].to_numpy(dtype=float),
            "event_observed": int(group["event_observed"].max()),
        }
    return sequences


def first_alert_lead_times(
    predictions: pd.DataFrame, threshold: float, *, k_consecutive: int = 2
) -> dict[str, Any]:
    """Para un umbral: quién se alerta y con cuánta anticipación.

    Devuelve los km de anticipación de la primera alerta sostenida en cada
    vehículo con evento y cuántos vehículos sanos se alertaron (falsas alarmas).
    """
    sequences = _vehicle_sequences(predictions)
    leads: list[float] = []
    n_event, n_healthy, n_false_alarm = 0, 0, 0

    for info in sequences.values():
        flags = info["score"] >= threshold
        idx = _first_sustained_index(flags, k_consecutive)
        if info["event_observed"] == 1:
            n_event += 1
            if idx is None:
                continue
            lead = float(info["time_to_event_km"][idx])
            # Una alerta posterior al evento (o sin km hasta el evento) no anticipa nada.
            if np.isfinite(lead) and lead > 0:
                leads.append(lead)
        else:
            n_healthy += 1
            if idx is not None:
                n_false_alarm += 1

    return {
        "threshold": float(threshold),
        "k_consecutive": int(k_consecutive),
        "lead_times_km": np.asarray(leads, dtype=float),
        "n_event_vehicles": n_event,
        "n_detected": len(leads),
        "n_healthy_vehicles": n_healthy,
        "n_false_alarm_vehicles": n_false_alarm,
    }


def false_alarm_rate(
    predictions: pd.DataFrame, threshold: float, *, k_consecutive: int = 2, per: int = 1000
) -> float:
    """Falsas alarmas por cada `per` vehículos sanos, con la regla de alerta sostenida."""
    stats = first_alert_lead_times(predictions, threshold, k_consecutive=k_consecutive)
    if stats["n_healthy_vehicles"] == 0:
        return float("nan")
    return float(per * stats["n_false_alarm_vehicles"] / stats["n_healthy_vehicles"])


def lead_time_curve(
    predictions: pd.DataFrame,
    *,
    thresholds: Iterable[float] | None = None,
    n_thresholds: int = 50,
    k_consecutive: int = 2,
    per: int = 1000,
) -> pd.DataFrame:
    """Curva de anticipación vs. falsas alarmas: la figura central del pitch.

    Eje x: falsas alarmas por cada `per` vehículos sanos.
    Eje y: mediana de km de anticipación de la primera alerta sostenida entre los
    vehículos con evento detectados (y `detection_rate`, que es la otra mitad de
    la frase: *a X falsas alarmas cada 1.000, detectamos el R% con mediana de N km*).
    """
    _validate_predictions(predictions)
    if thresholds is None:
        scores = predictions["score"].to_numpy(dtype=float)
        grid = np.quantile(scores, np.linspace(0.0, 1.0, n_thresholds))
        # Umbral por encima del máximo: es el extremo "no alerto nunca" de la curva
        # (0 falsas alarmas, 0 detección). Sin él, un modelo de score constante
        # devuelve una curva de un solo punto y el barrido parece roto.
        thresholds = np.unique(np.append(grid, np.nextafter(scores.max(), np.inf)))

    rows = []
    for threshold in thresholds:
        stats = first_alert_lead_times(predictions, threshold, k_consecutive=k_consecutive)
        leads = stats["lead_times_km"]
        n_event = stats["n_event_vehicles"]
        n_healthy = stats["n_healthy_vehicles"]
        rows.append(
            {
                "threshold": stats["threshold"],
                "k_consecutive": stats["k_consecutive"],
                "false_alarms_per_1000": (
                    float(per * stats["n_false_alarm_vehicles"] / n_healthy)
                    if n_healthy
                    else float("nan")
                ),
                "detection_rate": float(stats["n_detected"] / n_event) if n_event else float("nan"),
                "median_lead_km": float(np.median(leads)) if leads.size else float("nan"),
                "p25_lead_km": float(np.percentile(leads, 25)) if leads.size else float("nan"),
                "p75_lead_km": float(np.percentile(leads, 75)) if leads.size else float("nan"),
                "n_detected": stats["n_detected"],
                "n_event_vehicles": n_event,
                "n_false_alarm_vehicles": stats["n_false_alarm_vehicles"],
                "n_healthy_vehicles": n_healthy,
            }
        )
    return pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)


def operating_point(
    curve: pd.DataFrame, *, max_false_alarms_per_1000: float
) -> dict[str, Any] | None:
    """Mejor punto de operación bajo un presupuesto de falsas alarmas.

    "Mejor" = mayor `detection_rate`; a igualdad, mayor anticipación mediana.
    Devuelve None si ningún umbral respeta el presupuesto.
    """
    feasible = curve[curve["false_alarms_per_1000"] <= max_false_alarms_per_1000]
    feasible = feasible.dropna(subset=["detection_rate"])
    if feasible.empty:
        return None
    best = feasible.sort_values(
        ["detection_rate", "median_lead_km"], ascending=[False, False]
    ).iloc[0]
    return best.to_dict()


# --------------------------------------------------------------------------- #
# Agregación a nivel vehículo (Multiple-Instance Learning)
# --------------------------------------------------------------------------- #
VEHICLE_LABEL_COLUMN = "event_observed"
VEHICLE_AGGREGATIONS = ("max", "mean", "topk", "noisy_or")
DEFAULT_TOPK = 3


def _bag_score(scores: np.ndarray, how: str, k: int) -> float:
    """Pooling de una bolsa (Ilse, Tomczak & Welling, ICML 2018, §2).

    `topk` con menos de `k` instancias promedia las que hay: una bolsa corta no se
    castiga por ser corta.
    """
    if scores.size == 0:
        return float("nan")
    if how == "max":
        return float(np.max(scores))
    if how == "mean":
        return float(np.mean(scores))
    if how == "topk":
        k_eff = int(min(max(k, 1), scores.size))
        return float(np.mean(np.sort(scores)[-k_eff:]))
    if how == "noisy_or":
        return float(1.0 - np.prod(1.0 - scores))
    raise ValueError(
        f"Agregación de bolsa desconocida: {how!r}. Disponibles: {list(VEHICLE_AGGREGATIONS)}"
    )


def vehicle_scores(
    predictions: pd.DataFrame,
    how: str = "max",
    *,
    k: int = DEFAULT_TOPK,
    score_column: str = "score",
    label_column: str = VEHICLE_LABEL_COLUMN,
    fold_column: str | None = "fold",
) -> pd.DataFrame:
    """Colapsa los cortes de cada vehículo en un score único: una bolsa por vehículo.

    El panel tiene una fila por `(vehículo, corte)` y el PR-AUC de selección se mide
    ahí, pero **la decisión de negocio es por vehículo**: Ford marca autos, no cortes.
    Es el marco de Multiple-Instance Learning: cada vehículo es una bolsa, cada corte
    una instancia, y la etiqueta de la bolsa es `event_observed` —el vehículo tuvo
    evento— y no `label`, que es "el evento cae en el horizonte de *este* corte".
    Acá no se entrena nada: es una capa de decisión sobre los scores out-of-fold que
    ya existen (attention-MIL entrenado end-to-end sobreajusta con 53 bolsas).

    Agregaciones (`how`), todas sin parámetros que ajustar salvo `k`:

    * `max`: la regla operativa natural —alcanza un corte sospechoso para marcar el
      auto—. Es la más sensible a un pico de ruido aislado.
    * `mean`: el vehículo entero está degradado, no un corte. Diluye la señal de un
      vehículo con historia larga y sana antes de empeorar.
    * `topk`: media de los `k` cortes mayores. El punto medio entre las dos: exige
      más de un corte alto sin pedir que toda la historia lo sea.
    * `noisy_or`: `1 − Π(1 − p_i)`, el pooling probabilístico clásico ("al menos una
      instancia positiva"). **Necesita que los scores sean probabilidades**: si el
      modelo devuelve un margen, un riesgo relativo o cualquier cosa fuera de [0, 1],
      esto levanta `ValueError` en vez de calibrar por su cuenta —calibrar acá
      escondería el problema dentro de una métrica de selección—. Ojo además con que
      crece con el tamaño de la bolsa: en el panel v1 las bolsas van de 1 a 68 cortes,
      así que `noisy_or` mide en parte cuánta historia tiene el vehículo.

    Out-of-fold: el split es agrupado por vehículo (CLAUDE.md, regla 2), así que
    todas las filas de una bolsa salieron del mismo fold de validación y el score
    agregado sigue siendo estrictamente out-of-fold. Si algún día alguien splitea por
    fila, deja de serlo — por eso, si las predicciones traen la columna de fold, acá
    se verifica que cada vehículo tenga uno solo.

    Devuelve un DataFrame con `vehicle_id`, `score` (el de la bolsa), `label`
    (la etiqueta de la bolsa) y `n_cuts`, ordenado por `vehicle_id`.
    """
    for column in ("vehicle_id", score_column, label_column):
        if column not in predictions.columns:
            raise KeyError(
                f"Falta la columna `{column}` para agregar por vehículo. "
                f"Las predicciones traen {list(predictions.columns)}."
            )
    if how not in VEHICLE_AGGREGATIONS:
        raise ValueError(
            f"Agregación de bolsa desconocida: {how!r}. Disponibles: {list(VEHICLE_AGGREGATIONS)}"
        )

    scores = predictions[score_column].to_numpy(dtype=float)
    if not np.isfinite(scores).all():
        raise ValueError(
            f"La columna `{score_column}` tiene {int((~np.isfinite(scores)).sum())} valor(es) "
            "no finitos: una bolsa con un NaN contamina el score del vehículo entero."
        )
    if how == "noisy_or" and (scores.min() < 0.0 or scores.max() > 1.0):
        raise ValueError(
            f"`noisy_or` necesita probabilidades y `{score_column}` está en "
            f"[{scores.min():.4g}, {scores.max():.4g}]. Este modelo no devuelve "
            "probabilidades: sacá `noisy_or` de `eval.vehicle_aggregation` o calibralo "
            "explícitamente antes (calibrar acá adentro escondería el problema)."
        )

    if fold_column and fold_column in predictions.columns:
        folds_per_vehicle = predictions.groupby("vehicle_id", observed=True)[fold_column].nunique()
        leaky = folds_per_vehicle[folds_per_vehicle > 1]
        if len(leaky):
            raise ValueError(
                f"{len(leaky)} vehículo(s) con cortes en más de un fold "
                f"(`{fold_column}`): el score de la bolsa mezclaría folds y dejaría de ser "
                "out-of-fold. El split tiene que ser agrupado por vehículo (CLAUDE.md, regla 2)."
            )

    rows = []
    for vehicle_id, group in predictions.groupby("vehicle_id", observed=True, sort=True):
        bag = group[score_column].to_numpy(dtype=float)
        labels = group[label_column].to_numpy()
        rows.append(
            {
                "vehicle_id": str(vehicle_id),
                "score": _bag_score(bag, how, k),
                "label": int(np.max(labels)),
                "n_cuts": int(bag.size),
            }
        )
    return pd.DataFrame(rows, columns=["vehicle_id", "score", "label", "n_cuts"])


def vehicle_metrics(
    predictions: pd.DataFrame,
    how: str = "max",
    *,
    k: int = DEFAULT_TOPK,
    score_column: str = "score",
    label_column: str = VEHICLE_LABEL_COLUMN,
    fold_column: str | None = "fold",
) -> dict[str, Any]:
    """Las métricas de clasificación, pero con el vehículo como unidad de decisión.

    Mismas métricas que `classification_metrics()` (PR-AUC, ROC-AUC, Brier, tasa base
    y lift) sobre las bolsas de `vehicle_scores()`, más `how`, `k` y el tamaño de bolsa.

    **El PR-AUC por vehículo no se compara contra el PR-AUC por fila.** Son dos tasas
    base distintas: en dev del panel v1 hay 53 vehículos con evento sobre 171 con
    cortes (0,31) contra 254 filas positivas sobre 2.029 (0,125), y el PR-AUC arranca
    en la tasa base. Un PR-AUC por vehículo de 0,40 con tasa base 0,31 es *peor* que
    uno por fila de 0,165 con tasa base 0,125. Lo comparable es **el lift**
    (`pr_auc_lift`), y es lo que hay que reportar.

    Y ojo con la tasa base misma: 0,31 es la del panel, donde los sanos están
    emparejados por odómetro y mes (`sampling` en `configs/data/panel_v1.yaml`), no la
    de la flota de Ford. El lift es lo que sobrevive al cambio de prevalencia; el
    número pelado no.
    """
    bags = vehicle_scores(
        predictions,
        how,
        k=k,
        score_column=score_column,
        label_column=label_column,
        fold_column=fold_column,
    )
    metrics: dict[str, Any] = dict(classification_metrics(bags["label"], bags["score"]))
    metrics["how"] = how
    if how == "topk":
        metrics["k"] = int(k)
    metrics["bag_size_mean"] = float(bags["n_cuts"].mean())
    metrics["bag_size_median"] = float(bags["n_cuts"].median())
    return metrics


# --------------------------------------------------------------------------- #
# Significancia
# --------------------------------------------------------------------------- #
def bootstrap_ci(
    values: Sequence[float],
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
    statistic: str = "mean",
) -> dict[str, float]:
    """Intervalo por bootstrap sobre los valores por fold (plan §5, punto 3)."""
    clean = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if clean.size == 0:
        return {"point": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    func = {"mean": np.mean, "median": np.median}[statistic]
    rng = np.random.default_rng(seed)
    draws = rng.choice(clean, size=(n_boot, clean.size), replace=True)
    stats = func(draws, axis=1)
    return {
        "point": float(func(clean)),
        "lo": float(np.quantile(stats, alpha / 2)),
        "hi": float(np.quantile(stats, 1 - alpha / 2)),
        "n": int(clean.size),
    }


def dispersion(values: Sequence[float]) -> dict[str, float]:
    """Media, desvío y extremos de una métrica **entre repeticiones** de la CV.

    Es la otra mitad de la CV repetida: sin la dispersión, R pasadas son solo un
    número más estable y no se ve si la diferencia entre dos modelos entra dentro
    del ruido del sorteo. Desvío poblacional (ddof=0) porque las R repeticiones son
    las que hay, no una muestra de una población de repeticiones.
    """
    clean = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if clean.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "min": float("nan"),
                "max": float("nan"), "n": 0}
    return {
        "mean": float(clean.mean()),
        "std": float(clean.std(ddof=0)),
        "min": float(clean.min()),
        "max": float(clean.max()),
        "n": int(clean.size),
    }


def summarize_folds(fold_metrics: list[dict[str, float]], *, seed: int = 42) -> dict[str, float]:
    """Media e intervalo bootstrap por métrica, a partir de la lista por fold."""
    if not fold_metrics:
        return {}
    summary: dict[str, float] = {}
    for key in fold_metrics[0]:
        values = [m[key] for m in fold_metrics if isinstance(m.get(key), (int, float))]
        if not values:
            continue
        ci = bootstrap_ci(values, seed=seed)
        summary[f"{key}_mean"] = ci["point"]
        summary[f"{key}_lo"] = ci["lo"]
        summary[f"{key}_hi"] = ci["hi"]
    return summary
