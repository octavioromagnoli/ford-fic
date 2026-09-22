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
# Descomposición: cuánto del PR-AUC es "qué auto" y cuánto es "cuándo"
# --------------------------------------------------------------------------- #
def _vehicle_groups(vehicle_ids: Sequence[Any]) -> np.ndarray:
    """Los ids como array de strings, que es como se agrupa en todo el módulo."""
    return np.asarray(pd.Series(list(vehicle_ids)).astype(str))


def cohort_ceiling(
    y_true: Sequence[int], vehicle_ids: Sequence[Any]
) -> dict[str, float]:
    """El techo de un modelo que sabe **qué** vehículos fallan y nada del *cuándo*.

    Puntúa cada fila con 1 si su vehículo tiene al menos un corte positivo y 0 si no:
    el identificador de cohorte perfecto, con timing nulo. Como el score es constante
    dentro de cada grupo, el PR-AUC que sale es exactamente la precisión de marcar a
    todos los cortes de los vehículos fallados — en el dev del panel v1, 254/967.

    **Es el piso contra el que se lee un PR-AUC por fila, no la tasa base.** En este
    panel las 254 filas positivas están todas dentro de los 967 cortes de vehículos
    fallados, así que 0,263 de PR-AUC (lift 2,10×) se consigue sin anticipar nunca, y
    la meta de 1,6–2× de lift del plan cae **por debajo** de ese techo. Un modelo que
    no lo supera no demostró timing: demostró que reconoce la cohorte.

    No confundir con puntuar por la *tasa* de positivos del vehículo (`pr_auc_rate`):
    también es constante por vehículo, pero ordena a los fallados entre sí por cuánta
    de su historia cae en la ventana de riesgo, que ya es información del *cuándo*.
    Va como referencia secundaria —una cota superior laxa—, nunca como el techo.

    Devuelve `pr_auc`/`lift` (el indicador binario, el techo) más `base_rate`,
    `n_failed_rows` y la variante por tasa. NaN si hay una sola clase.
    """
    y = np.asarray(y_true, dtype=int)
    groups = _vehicle_groups(vehicle_ids)
    if len(y) != len(groups):
        raise ValueError(f"y_true tiene {len(y)} filas y vehicle_ids {len(groups)}")

    labels = pd.Series(y, index=pd.Index(groups, name="vehicle_id"))
    by_vehicle = labels.groupby(level=0, observed=True)
    indicator = by_vehicle.transform("max").to_numpy(dtype=float)
    rate = by_vehicle.transform("mean").to_numpy(dtype=float)

    base_rate = float(y.mean()) if len(y) else float("nan")
    ap = pr_auc(y, indicator)
    ap_rate = pr_auc(y, rate)
    return {
        "pr_auc": ap,
        "lift": float(ap / base_rate) if base_rate else float("nan"),
        "pr_auc_rate": ap_rate,
        "lift_rate": float(ap_rate / base_rate) if base_rate else float("nan"),
        "base_rate": base_rate,
        "n": int(len(y)),
        "n_positive": int(y.sum()),
        "n_failed_rows": int(indicator.sum()),
        "n_failed_vehicles": int(by_vehicle.max().sum()),
    }


def pr_auc_within_failed(
    y_true: Sequence[int], y_score: Sequence[float], vehicle_ids: Sequence[Any]
) -> dict[str, float]:
    """PR-AUC sobre las filas de vehículos que **sí** fallan: la pregunta del *cuándo*.

    Restringe la medición a los cortes de los vehículos con al menos un positivo, que
    es donde la pregunta "¿falla en los próximos H km?" tiene sentido. Ahí el nivel del
    vehículo ya no ordena nada —todas las filas son del mismo lado de la cohorte—, así
    que lo único que puede subir el PR-AUC es el orden de los cortes dentro del auto.

    El azar acá no es la tasa base del panel (0,125) sino la tasa dentro del subconjunto
    (0,263 en dev): `base_rate` y `lift` salen de esa tasa, no de la global. Por eso el
    PR-AUC de acá **no se compara** contra el PR-AUC por fila de la tabla; lo comparable
    es el lift, igual que con el eje de vehículo.

    Complementa a `cohort_ceiling()`: aquel dice cuánto se saca sin timing, este mide
    lo que queda cuando el timing es lo único que hay.
    """
    y = np.asarray(y_true, dtype=int)
    score = np.asarray(y_score, dtype=float)
    groups = _vehicle_groups(vehicle_ids)
    if not (len(y) == len(score) == len(groups)):
        raise ValueError(
            f"Longitudes distintas: y_true={len(y)}, y_score={len(score)}, "
            f"vehicle_ids={len(groups)}"
        )

    labels = pd.Series(y, index=pd.Index(groups, name="vehicle_id"))
    failed = labels.groupby(level=0, observed=True).transform("max").to_numpy() == 1
    y_f, score_f = y[failed], score[failed]
    base_rate = float(y_f.mean()) if len(y_f) else float("nan")
    ap = pr_auc(y_f, score_f)
    return {
        "pr_auc": ap,
        "base_rate": base_rate,
        "lift": float(ap / base_rate) if base_rate else float("nan"),
        "roc_auc": roc_auc(y_f, score_f),
        "n": int(len(y_f)),
        "n_positive": int(y_f.sum()),
        "n_vehicles": int(pd.unique(groups[failed]).size),
    }


def when_contribution(
    predictions: pd.DataFrame,
    *,
    score_column: str = "score",
    label_column: str = "label",
) -> dict[str, float]:
    """(a') El aporte del *cuándo*: cuánto PR-AUC se pierde al borrar el orden interno.

    Reemplaza el score de cada fila por el promedio de su vehículo —sin reentrenar, con
    todo lo demás fijo— y devuelve la caída de PR-AUC. Lo que se pierde es exactamente
    lo que el modelo sabía del *cuándo*: el colapso conserva intacto el orden **entre**
    vehículos y destruye el orden **dentro** de cada uno.

    Es la única de las auditorías que aísla el timing. La permutación intra-vehículo de
    `scripts/audit_model.py` no puede: deja en pie *qué* vehículos fallan, que es de
    donde sale casi todo el PR-AUC de este panel, así que no tiene nulo contra el cual
    leerse (ver `docs/memoria/decisiones.md`).

    `delta` positivo = el orden dentro del vehículo suma; negativo = el modelo ordena
    los cortes al revés y colapsarlo lo mejoraría.
    """
    for column in (score_column, label_column, "vehicle_id"):
        if column not in predictions:
            raise KeyError(f"Las predicciones no traen `{column}`")

    y = predictions[label_column].to_numpy(dtype=int)
    score = predictions[score_column].to_numpy(dtype=float)
    collapsed = (
        predictions.groupby("vehicle_id", observed=True)[score_column]
        .transform("mean")
        .to_numpy(dtype=float)
    )
    full, flat = pr_auc(y, score), pr_auc(y, collapsed)
    return {
        "pr_auc": full,
        "pr_auc_vehicle_mean": flat,
        "delta": float(full - flat),
        "n": int(len(y)),
        "n_positive": int(y.sum()),
    }


# --------------------------------------------------------------------------- #
# Costo de la decisión: matriz asimétrica y barrido de C_FN/C_FP
# --------------------------------------------------------------------------- #
def cost_matrix_score(
    y_true: Sequence[int], y_pred: Sequence[int], matrix: Sequence[Sequence[float]]
) -> float:
    """Costo total de las decisiones, con filas reales y columnas predichas.

    La escala y la asimetría pertenecen al experimento y por eso `matrix` llega
    desde el YAML. El retorno es la suma (no una accuracy disfrazada), igual que
    la definición del benchmark SCANIA Component X; quien necesite costo medio o
    por 1.000 filas lo deriva conservando este total auditable.
    """
    truth = np.asarray(y_true)
    predicted = np.asarray(y_pred)
    costs = np.asarray(matrix, dtype=float)
    if truth.ndim != 1 or predicted.ndim != 1 or len(truth) != len(predicted):
        raise ValueError("`y_true` e `y_pred` tienen que ser vectores del mismo largo")
    if costs.ndim != 2 or costs.shape[0] != costs.shape[1] or costs.shape[0] == 0:
        raise ValueError("La matriz de costos tiene que ser cuadrada y no vacía")
    if not np.isfinite(costs).all() or (costs < 0).any():
        raise ValueError("La matriz de costos solo puede contener valores finitos no negativos")

    truth_int = truth.astype(int)
    predicted_int = predicted.astype(int)
    if not np.array_equal(truth, truth_int) or not np.array_equal(predicted, predicted_int):
        raise ValueError("Las clases reales y predichas tienen que ser enteros")
    n_classes = costs.shape[0]
    if (
        (truth_int < 0).any()
        or (truth_int >= n_classes).any()
        or (predicted_int < 0).any()
        or (predicted_int >= n_classes).any()
    ):
        raise ValueError(f"Las clases tienen que estar entre 0 y {n_classes - 1}")
    return float(costs[truth_int, predicted_int].sum())


DEFAULT_COST_RATIOS = (2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 50.0)


def cost_ratio_sweep(
    y_true: Sequence[int],
    y_score: Sequence[float],
    ratios: Sequence[float] = DEFAULT_COST_RATIOS,
    *,
    cost_fp: float = 1.0,
) -> list[dict[str, float]]:
    """Barre `C_FN / C_FP` y mide cuánto ahorra alertar por score contra no modelar.

    Una matriz de costos única no dice nada a esta tasa base: con `p = 0,125`,
    alertar siempre le gana a no alertar nunca en cuanto `C_FN / C_FP > (1 - p) / p`
    (≈ 7). Elegir un ratio de 20–50, como el benchmark de camiones pesados, hace que
    la política óptima sea "revisar todo" **sin importar el modelo**, y el número
    resultante mide la matriz, no el clasificador. El barrido evita esa trampa:
    para cada ratio compara el mejor umbral contra las dos políticas triviales.

    Las tres políticas comparadas, con el mismo costo unitario de inspección:

    * **alertar siempre**: paga `n_negativos · C_FP` y nunca un falso negativo;
    * **no alertar nunca**: paga `n_positivos · C_FN`;
    * **umbral sobre el score**: paga `FP · C_FP + FN · C_FN`.

    El mínimo sobre umbrales incluye los dos extremos degenerados, así que el costo
    del modelo nunca supera al de la mejor política trivial y `savings_vs_trivial`
    es cero exactamente cuando el óptimo es no modelar. Ese cero es el **punto de
    quiebre** que se reporta.

    **El umbral se elige sobre las mismas predicciones que se evalúan**, así que el
    ahorro es una cota superior optimista del valor del modelo, no una estimación
    honesta de lo que rendiría en producción. Se usa así a propósito: si ni con el
    umbral oráculo el modelo le gana a alertar todo, la conclusión es firme.

    Devuelve una fila por ratio; es reporte, nunca criterio de selección (regla 5).
    """
    truth = np.asarray(y_true, dtype=int)
    score = np.asarray(y_score, dtype=float)
    if truth.ndim != 1 or score.ndim != 1 or len(truth) != len(score):
        raise ValueError("`y_true` e `y_score` tienen que ser vectores del mismo largo")
    if not len(truth):
        raise ValueError("No hay filas para barrer el costo")
    if not np.isin(truth, (0, 1)).all():
        raise ValueError("`cost_ratio_sweep` es una decisión binaria: `y_true` solo admite 0/1")
    if not np.isfinite(score).all():
        raise ValueError("`y_score` tiene valores no finitos")
    if cost_fp <= 0:
        raise ValueError("`cost_fp` tiene que ser positivo")
    ratios_arr = np.asarray(list(ratios), dtype=float)
    if ratios_arr.size == 0 or not np.isfinite(ratios_arr).all() or (ratios_arr <= 0).any():
        raise ValueError("`ratios` tiene que ser una lista no vacía de positivos finitos")

    n = len(truth)
    n_pos = int(truth.sum())
    n_neg = n - n_pos
    if not n_pos or not n_neg:
        raise ValueError("El barrido necesita positivos y negativos para tener sentido")

    # Umbrales candidatos: alertar las k filas de score más alto, para cada k que
    # cae en un cambio de score (los empates alertan juntos o no alertan).
    order = np.argsort(-score, kind="stable")
    ranked_truth = truth[order]
    ranked_score = score[order]
    edges = np.flatnonzero(np.diff(ranked_score)) if n > 1 else np.array([], dtype=int)
    cuts = np.append(edges, n - 1)
    tp = np.cumsum(ranked_truth)[cuts]
    fp = np.cumsum(1 - ranked_truth)[cuts]
    thresholds = ranked_score[cuts]
    n_alerts = cuts + 1

    always_cost = float(n_neg * cost_fp)
    rows: list[dict[str, float]] = []
    for ratio in ratios_arr:
        cost_fn = float(ratio) * cost_fp
        never_cost = float(n_pos * cost_fn)
        costs = fp * cost_fp + (n_pos - tp) * cost_fn
        best = int(np.argmin(costs))
        model_cost = float(costs[best])
        # "No alertar nunca" no está entre los cortes de arriba (todo corte alerta al
        # menos una fila), así que entra acá como candidato explícito.
        if never_cost <= model_cost:
            model_cost = never_cost
            threshold = float("inf")
            alerts, hits, misses = 0, 0, n_pos
        else:
            threshold = float(thresholds[best])
            alerts = int(n_alerts[best])
            hits = int(tp[best])
            misses = n_pos - hits
        trivial_cost = min(always_cost, never_cost)
        best_trivial = "always" if always_cost <= never_cost else "never"
        savings = trivial_cost - model_cost
        rows.append(
            {
                "ratio": float(ratio),
                "cost_fp": float(cost_fp),
                "cost_fn": cost_fn,
                "threshold": threshold,
                "n_alerts": alerts,
                "alert_rate": float(alerts / n),
                "detection_rate": float(hits / n_pos),
                "false_alarm_rate": float((alerts - hits) / n_neg),
                "n_false_negative": int(misses),
                "model_cost": model_cost,
                "always_cost": always_cost,
                "never_cost": never_cost,
                "trivial_cost": float(trivial_cost),
                "best_trivial": best_trivial,
                "savings_vs_trivial": float(savings),
                "savings_frac": float(savings / trivial_cost) if trivial_cost else 0.0,
                "beats_trivial": bool(savings > 0),
            }
        )
    return rows


MATERIAL_SAVINGS_FRAC = 0.05


def cost_ratio_breakeven(
    sweep: Sequence[dict[str, float]], *, material_frac: float = MATERIAL_SAVINGS_FRAC
) -> dict[str, Any]:
    """Resume el barrido: en qué tramo de `C_FN / C_FP` el modelo todavía paga.

    El ahorro no es monótono en el ratio, y por eso no alcanza con un solo número.
    Abajo del cruce de las dos políticas triviales —`always_beats_never_above_ratio`,
    que vale `(1 - p) / p`— no alertar nunca ya es barato y el modelo casi no tiene
    dónde ahorrar; muy por encima, un falso negativo es tan caro que la política
    óptima se corre hacia alertar todo y el ahorro se desvanece otra vez. El valor
    del modelo se concentra alrededor del cruce, y eso es lo que hay que reportar:

    * `best_ratio` / `best_savings_frac`: el pico del barrido;
    * `material_ratios`: los ratios donde el ahorro supera `material_frac`. Un ahorro
      del 2% sobre revisar todo es indistinguible de revisar todo y **no** cuenta como
      "el modelo paga", aunque `beats_trivial` sea cierto;
    * `saturates_at_grid_edge`: el último ratio de la grilla todavía ahorra, así que
      de esta grilla no se puede leer un punto de quiebre superior.

    Es un resumen de la grilla barrida, no una raíz interpolada: fuera de esa grilla
    no se afirma nada.
    """
    rows = list(sweep)
    if not rows:
        raise ValueError("El barrido está vacío")
    winners = [row for row in rows if row["beats_trivial"]]
    ratios = [row["ratio"] for row in winners]
    material = [row["ratio"] for row in rows if row["savings_frac"] > material_frac]
    best = max(winners, key=lambda row: row["savings_frac"], default=None)
    return {
        "ratios": [row["ratio"] for row in rows],
        "material_frac": float(material_frac),
        "n_ratios_with_savings": len(winners),
        "min_ratio_with_savings": min(ratios) if ratios else None,
        "max_ratio_with_savings": max(ratios) if ratios else None,
        "min_material_ratio": min(material) if material else None,
        "max_material_ratio": max(material) if material else None,
        "best_ratio": best["ratio"] if best else None,
        "best_savings_frac": best["savings_frac"] if best else 0.0,
        "best_savings_vs_trivial": best["savings_vs_trivial"] if best else 0.0,
        "saturates_at_grid_edge": bool(rows[-1]["savings_frac"] > material_frac),
        "always_beats_never_above_ratio": float(
            rows[0]["always_cost"] / (rows[0]["never_cost"] / rows[0]["ratio"])
        ),
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


# --------------------------------------------------------------------------- #
# Supervivencia: el orden, y el intervalo que corresponde al diseño
# --------------------------------------------------------------------------- #
FOLLOWUP_COLUMN = "aux_km_observed_after_cut"


def survival_durations(
    predictions: pd.DataFrame, *, gap_km: float | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """`(duración, evento)` por fila, con el origen corrido al final del gap.

    Misma definición que `src/training/targets.py`: con evento, `time_to_event_km − G`;
    censurado, `aux_km_observed_after_cut − G`. Se calcula acá otra vez —y no se importa
    del target— para que la métrica no dependa de con qué objetivo se entrenó: así el
    C-index de un LightGBM sobre `label` y el de un modelo de supervivencia miden lo
    mismo y son comparables.

    Las filas que no estuvieron en riesgo en ningún km visible (duración negativa: el
    evento cayó dentro del gap) salen con duración NaN, para que quien mida las filtre
    en vez de contarlas como una supervivencia de 0 km.
    """
    missing = [c for c in ("time_to_event_km", "event_observed") if c not in predictions]
    if missing:
        raise KeyError(f"Las predicciones no traen {missing}")
    if FOLLOWUP_COLUMN not in predictions:
        raise KeyError(
            f"Las predicciones no traen `{FOLLOWUP_COLUMN}`: el panel es anterior a la "
            "columna de seguimiento. Reconstruilo con scripts/build_dataset.py."
        )
    if gap_km is None:
        if "gap_km" not in predictions:
            raise KeyError("Pasá `gap_km`: las predicciones no lo traen")
        gap = float(predictions["gap_km"].iloc[0])
    else:
        gap = float(gap_km)

    event = predictions["event_observed"].to_numpy().astype(int)
    duration = (
        np.where(
            event == 1,
            predictions["time_to_event_km"].to_numpy(dtype=float),
            predictions[FOLLOWUP_COLUMN].to_numpy(dtype=float),
        )
        - gap
    )
    at_risk = np.isfinite(duration) & (duration >= 0)
    return np.where(at_risk, duration, np.nan), np.where(at_risk, event, 0)


def concordance_index_oof(
    predictions: pd.DataFrame, *, gap_km: float | None = None
) -> dict[str, float]:
    """C-index out-of-fold sobre la duración censurada. Métrica secundaria.

    Contesta algo que el PR-AUC no: de dos cortes, ¿el modelo pone más arriba al que
    está más cerca del evento? Usa a los sanos como censurados en vez de como ceros, así
    que dos modelos pueden empatar en PR-AUC y ordenar distinto.

    **No es independiente entre filas**: los ~5 cortes de un mismo vehículo entran como
    pares comparables aunque compartan el auto. Sirve para comparar modelos sobre los
    mismos folds; no para un test de hipótesis. Devuelve NaN si lifelines no está.
    """
    try:
        from lifelines.utils import concordance_index
    except ImportError:  # pragma: no cover - lifelines está en requirements.txt
        return {"c_index": float("nan"), "n": 0, "n_events": 0}

    duration, event = survival_durations(predictions, gap_km=gap_km)
    score = predictions["score"].to_numpy(dtype=float)
    usable = np.isfinite(score) & np.isfinite(duration)
    if int(event[usable].sum()) == 0:
        return {"c_index": float("nan"), "n": int(usable.sum()), "n_events": 0}
    # `concordance_index` espera que un valor más alto sea "sobrevive más": el score es
    # riesgo, así que va con el signo cambiado.
    value = concordance_index(duration[usable], -score[usable], event[usable])
    return {
        "c_index": float(value),
        "n": int(usable.sum()),
        "n_events": int(event[usable].sum()),
    }


def bootstrap_by_vehicle(
    predictions: pd.DataFrame,
    *,
    metrics: tuple[str, ...] = ("pr_auc", "roc_auc"),
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
    score_column: str = "score",
    label_column: str = "label",
) -> dict[str, dict[str, float]]:
    """Intervalo remuestreando **vehículos**, no filas.

    Las filas de un mismo vehículo no son independientes (mismo auto, misma ruta, ~5
    cortes correlacionados): un bootstrap por fila trata cada corte como una
    observación nueva y devuelve un intervalo mucho más angosto de lo que corresponde.
    La unidad de muestreo del estudio es el vehículo, así que el remuestreo también.

    Complementa `bootstrap_ci` sobre los folds: aquel mide la variabilidad del sorteo de
    folds (5 valores), este la de la muestra de vehículos.
    """
    for column in (score_column, label_column, "vehicle_id"):
        if column not in predictions:
            raise KeyError(f"Las predicciones no traen `{column}`")
    functions = {"pr_auc": pr_auc, "roc_auc": roc_auc, "brier": brier}
    unknown = [m for m in metrics if m not in functions]
    if unknown:
        raise KeyError(f"Métricas desconocidas para el bootstrap: {unknown}")

    y = predictions[label_column].to_numpy(dtype=int)
    score = predictions[score_column].to_numpy(dtype=float)
    groups = predictions["vehicle_id"].astype(str).to_numpy()
    order = np.argsort(groups, kind="stable")
    _, starts = np.unique(groups[order], return_index=True)
    per_vehicle = np.split(order, starts[1:])

    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {name: [] for name in metrics}
    for _ in range(n_boot):
        picked = rng.integers(0, len(per_vehicle), size=len(per_vehicle))
        idx = np.concatenate([per_vehicle[i] for i in picked])
        for name in metrics:
            draws[name].append(functions[name](y[idx], score[idx]))

    out: dict[str, dict[str, float]] = {}
    for name in metrics:
        values = np.asarray([v for v in draws[name] if np.isfinite(v)], dtype=float)
        out[name] = {
            "point": float(functions[name](y, score)),
            "lo": float(np.quantile(values, alpha / 2)) if values.size else float("nan"),
            "hi": float(np.quantile(values, 1 - alpha / 2)) if values.size else float("nan"),
            "n_boot": int(values.size),
            "n_vehicles": int(len(per_vehicle)),
        }
    return out
