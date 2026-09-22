"""Qué se le pasa al estimador como `y`, y cómo su salida vuelve a un score.

`src/training/cv.py` evalúa **siempre** contra la etiqueta dura `label` (CLAUDE.md,
regla 5). Lo que este archivo decide es otra cosa: con qué objetivo se *entrena*, y
cómo se traduce lo que el estimador devuelve. La etiqueta binaria de una ventana tira
información que el panel ya tiene —a qué distancia cae el evento, hasta dónde se
observó un vehículo sano, de qué vehículo es la fila— y cada modo de acá la recupera
de una forma distinta.

**El registro es el contrato.** `cv.py` no sabe qué modos existen: pide
`build_target(name, panel, train_mask, **params)`, `decode_predictions(...)` y
`target_report(...)`, y usa lo que vuelva. Agregar un modo es escribir una función con
`@register_target("...")` en este archivo —más un `@register_decoder` y un
`@register_reporter` si su salida no es P(evento en H) directa— y nombrarla en el
`target:` del YAML del experimento; no se toca `cv.py` ni `train.py`, y dos ramas
pueden agregar modos distintos sin pisarse. Bins, horizontes y costos vienen del YAML,
nunca de constantes del código.

**Reglas que todo modo tiene que respetar:**

1. El target se construye **solo con las filas de train del fold** (`train_mask`). El
   panel entero entra como argumento porque hay modos que necesitan agrupar por
   vehículo, pero cualquier estadístico se estima sobre el train (CLAUDE.md, regla 3).
2. `TargetSpec.y` está alineado 1:1 con `panel.loc[train_mask]`, en ese orden. Es lo
   que `Pipeline.fit` recibe como `y`, así que el estimador del `model:` del YAML y el
   modo del `target:` van de a pares.
3. Ningún modo puede mirar los km del gap de blanking (regla 1). El de supervivencia lo
   cumple corriendo el origen de la duración a `c + G`; el ordinal, mandando a la clase
   0 cualquier evento que caiga adentro del gap.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

GROUP_COLUMN = "vehicle_id"
FOLLOWUP_COLUMN = "aux_km_observed_after_cut"


@dataclass(frozen=True)
class TargetSpec:
    """Lo que un modo le entrega a `cv.py`: el `y` de train y con qué se construyó.

    `y` está alineado 1:1 con `panel.loc[train_mask]` y es lo único que `cv.py` usa;
    `name`, `params` e `info` viajan para que la corrida quede trazable en el log y en
    la metadata del experimento.
    """

    y: np.ndarray
    name: str
    params: dict[str, Any] = field(default_factory=dict)
    info: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TargetPredictions:
    """Score binario comparable y columnas auxiliares de una decodificación."""

    score: np.ndarray
    extras: dict[str, np.ndarray] = field(default_factory=dict)


TargetBuilder = Callable[..., TargetSpec]
TargetDecoder = Callable[..., TargetPredictions]
TargetReporter = Callable[..., dict[str, Any]]

_REGISTRY: dict[str, TargetBuilder] = {}
_DECODERS: dict[str, TargetDecoder] = {}
_REPORTERS: dict[str, TargetReporter] = {}


def register_target(name: str) -> Callable[[TargetBuilder], TargetBuilder]:
    def decorator(builder: TargetBuilder) -> TargetBuilder:
        if name in _REGISTRY:
            raise ValueError(f"El target `{name}` ya está registrado")
        _REGISTRY[name] = builder
        return builder

    return decorator


def register_decoder(name: str) -> Callable[[TargetDecoder], TargetDecoder]:
    def decorator(decoder: TargetDecoder) -> TargetDecoder:
        if name in _DECODERS:
            raise ValueError(f"El decoder del target `{name}` ya está registrado")
        _DECODERS[name] = decoder
        return decoder

    return decorator


def register_reporter(name: str) -> Callable[[TargetReporter], TargetReporter]:
    def decorator(reporter: TargetReporter) -> TargetReporter:
        if name in _REPORTERS:
            raise ValueError(f"El reporte del target `{name}` ya está registrado")
        _REPORTERS[name] = reporter
        return reporter

    return decorator


def available_targets() -> list[str]:
    return sorted(_REGISTRY)


def build_target(
    name: str, panel: pd.DataFrame, train_mask: np.ndarray, **params: Any
) -> TargetSpec:
    """Construye el target de train del modo `name`. Único punto de entrada."""
    if name not in _REGISTRY:
        raise KeyError(f"Target `{name}` no registrado. Disponibles: {available_targets()}")
    spec = _REGISTRY[name](panel, train_mask, **params)
    n_train = int(np.asarray(train_mask).sum())
    if len(spec.y) != n_train:
        raise RuntimeError(
            f"El target `{name}` devolvió {len(spec.y)} valores para {n_train} filas de "
            "train. Tiene que estar alineado 1:1 con `panel.loc[train_mask]`."
        )
    return spec


def decode_predictions(
    name: str | None, estimator: Any, features: pd.DataFrame, **params: Any
) -> TargetPredictions:
    """Decodifica cualquier target; sin decoder propio usa clasificación binaria.

    El fallback mantiene compatibles los targets cuyo estimador ya devuelve
    directamente P(evento en H), como el hook de supervivencia de la rama paralela.
    """
    decoded = (
        _DECODERS[name](estimator, features, **params)
        if name in _DECODERS
        else _binary_predictions(estimator, features)
    )
    score = np.asarray(decoded.score, dtype=float)
    if score.ndim != 1 or len(score) != len(features):
        raise RuntimeError(
            f"El decoder `{name or 'binary'}` devolvió score con forma {score.shape}; "
            f"se esperaban {len(features)} valores."
        )
    extras: dict[str, np.ndarray] = {}
    for column, values in decoded.extras.items():
        array = np.asarray(values, dtype=float)
        if array.ndim != 1 or len(array) != len(features):
            raise RuntimeError(
                f"El decoder `{name}` devolvió `{column}` con forma {array.shape}; "
                f"se esperaban {len(features)} valores."
            )
        extras[column] = array
    return TargetPredictions(score=score, extras=extras)


def target_report(
    name: str | None,
    predictions: pd.DataFrame,
    *,
    cost_matrix: Sequence[Sequence[float]] | None = None,
    **params: Any,
) -> dict[str, Any]:
    """Enriquece predicciones y devuelve el reporte propio del target, si existe."""
    if name not in _REPORTERS:
        return {}
    return _REPORTERS[name](predictions, cost_matrix=cost_matrix, **params)


def _binary_predictions(estimator: Any, features: pd.DataFrame) -> TargetPredictions:
    """Probabilidad de clase 1; `decision_function` como fallback compartido."""
    if hasattr(estimator, "predict_proba"):
        proba = np.asarray(estimator.predict_proba(features), dtype=float)
        classes = list(estimator.classes_)
        if 1 in classes:
            return TargetPredictions(proba[:, classes.index(1)])
        return TargetPredictions(proba[:, -1])
    return TargetPredictions(np.asarray(estimator.decision_function(features), dtype=float))


def _validated_bins(bins_km: Sequence[float] | None) -> np.ndarray:
    if bins_km is None:
        raise ValueError("`target.params.bins_km` es obligatorio y tiene que salir del YAML")
    bins = np.asarray(bins_km, dtype=float)
    if bins.ndim != 1 or bins.size == 0 or not np.isfinite(bins).all():
        raise ValueError("`bins_km` tiene que ser una lista no vacía de números finitos")
    if (bins <= 0).any() or (np.diff(bins) <= 0).any():
        raise ValueError("`bins_km` tiene que ser positivo y estrictamente creciente")
    return bins


def _score_class_min(bins: np.ndarray, score_max_tte_km: float | None) -> int:
    if score_max_tte_km is None:
        raise ValueError("`target.params.score_max_tte_km` es obligatorio")
    matches = np.flatnonzero(np.isclose(bins, float(score_max_tte_km)))
    if matches.size != 1:
        raise ValueError("`score_max_tte_km` tiene que coincidir con un corte de `bins_km`")
    # Clase K = evento más cercano; clase 1 = último bucket dentro del horizonte.
    return int(len(bins) - matches[0])


# --------------------------------------------------------------------------- #
# Supervivencia en tiempo discreto ("survival stacking")
# --------------------------------------------------------------------------- #
#: Campos del `y` estructurado que consume `src/models/survival_stacking.py`.
SURVIVAL_FIELDS = ("duration_km", "event", "at_risk", "group")


@register_target("discrete_survival")
def build_discrete_survival_target(
    panel: pd.DataFrame,
    train_mask: np.ndarray,
    *,
    gap_km: float | None = None,
    followup_column: str = FOLLOWUP_COLUMN,
    group_column: str = GROUP_COLUMN,
) -> TargetSpec:
    """`(duración, evento, vehículo)` por fila de train, con el origen corrido al gap.

    El panel ya es un dataset persona-período: una fila por `(vehículo, corte)`. Lo que
    falta para tratarlo como supervivencia es decir, para cada fila, **cuánto se la
    observó y si terminó en evento** (Craig, Zhong & Tibshirani 2021, arXiv:2107.13480).

    * **Origen.** La duración no se cuenta desde el corte `c` sino desde `c + G`: el
      modelo no puede ver los km del gap de blanking (CLAUDE.md, regla 1). Una fila que
      "sobrevivió 0 km" es una fila cuyo evento cae justo al terminar el blanking.
    * **Filas con evento** (`event_observed == 1`): duración `time_to_event_km − G`,
      evento 1. El panel las corta en `E − G`, así que la duración es ≥ 0 por
      construcción.
    * **Filas sanas** (`event_observed == 0`): duración `aux_km_observed_after_cut − G`,
      evento 0. No son ceros limpios: son "llegó hasta acá sin fallar". Con
      `censored_policy: drop` esa duración es ≥ H, así que dentro del horizonte se
      comportan igual que un negativo; la diferencia aparece cuando el estimador mira
      más allá de H (`max_horizon_km`), que es justamente lo que la etiqueta binaria no
      puede usar.
    * **Duración < 0** (`at_risk = False`): la fila no está en riesgo en ningún km
      visible, porque el evento cayó dentro del gap. El estimador la descarta al apilar;
      usarla sería mirar los km que la regla 1 prohíbe. El panel v1 no tiene ninguna,
      pero el dummy sí, porque genera cortes hasta el evento. Una duración de
      exactamente 0 **sí** está en riesgo: es el evento justo al terminar el blanking, y
      es un `label = 1`.

    La etiqueta dura sale de esto sin residuo: `label = 1 ⇔ at_risk y evento y duración
    ∈ [0, H]`. No es una reparametrización libre, es la misma definición escrita sobre
    el eje correcto — y `scripts/check_setup.py` lo verifica fila a fila.

    `group` viaja para el efecto aleatorio por vehículo (GPBoost): los ~5 cortes de un
    mismo vehículo no son observaciones independientes.
    """
    mask = np.asarray(train_mask, dtype=bool)
    train = panel.loc[mask]

    missing = [c for c in ("time_to_event_km", "event_observed", group_column) if c not in train]
    if missing:
        raise KeyError(f"El panel no tiene {missing}: no se puede armar el target de supervivencia")
    if followup_column not in train:
        raise KeyError(
            f"El panel no tiene `{followup_column}`: los censurados no traen hasta dónde se "
            "observaron. Reconstruí el panel con `python scripts/build_dataset.py "
            "--config configs/data/panel_v1.yaml` (src/data/panel.py lo emite)."
        )

    gap = float(train["gap_km"].iloc[0]) if gap_km is None else float(gap_km)
    if gap_km is None and train["gap_km"].nunique() != 1:
        raise ValueError(
            "El panel mezcla varios `gap_km`: pasá `target.params.gap_km` explícito."
        )

    event = train["event_observed"].to_numpy().astype(np.int8)
    tte = train["time_to_event_km"].to_numpy(dtype=float)
    followup = train[followup_column].to_numpy(dtype=float)
    # Con evento, el seguimiento termina en el evento; sin evento, en el último km visto.
    raw = np.where(event == 1, tte, followup) - gap
    if np.isnan(raw).any():
        n = int(np.isnan(raw).sum())
        raise ValueError(
            f"{n} filas de train sin duración: `time_to_event_km` nulo en filas con evento "
            f"o `{followup_column}` nulo en censuradas."
        )

    at_risk = raw >= 0
    duration = np.where(at_risk, raw, 0.0)
    event = np.where(at_risk, event, 0).astype(np.int8)

    groups = train[group_column].astype(str).to_numpy()
    width = max(1, max((len(g) for g in groups), default=1))
    y = np.empty(
        len(train),
        dtype=[("duration_km", "f8"), ("event", "i1"), ("at_risk", "?"), ("group", f"U{width}")],
    )
    y["duration_km"] = duration
    y["event"] = event
    y["at_risk"] = at_risk
    y["group"] = groups

    info = {
        "gap_km": gap,
        "n_rows": int(len(train)),
        "n_events": int(event.sum()),
        "n_not_at_risk": int((~at_risk).sum()),
        "duration_km_median": float(np.median(duration[at_risk])) if at_risk.any() else float("nan"),
    }
    logger.debug("target discrete_survival | %s", info)
    return TargetSpec(
        y=y,
        name="discrete_survival",
        params={"gap_km": gap, "followup_column": followup_column, "group_column": group_column},
        info=info,
    )


# --------------------------------------------------------------------------- #
# Supervivencia en días desde c + G, con el conjunto en riesgo de la ventana
# --------------------------------------------------------------------------- #
#: Campos del `y` estructurado que consume `src/models/window_stacking.py`.
WINDOW_SURVIVAL_FIELDS = ("entry", "exit", "event", "at_risk", "group")


@register_target("window_survival")
def build_window_survival_target(
    panel: pd.DataFrame,
    train_mask: np.ndarray,
    *,
    entry_column: str = "aux_risk_entry_days",
    exit_column: str = "aux_risk_exit_days",
    event_column: str = "aux_risk_event",
    group_column: str = GROUP_COLUMN,
    **_: Any,
) -> TargetSpec:
    """El tramo en riesgo de cada fila de train, en días desde que el odómetro llega a `c + G`.

    Es el objetivo de F5 §3.3 (`docs/memoria/f5-preregistro-ss-post-venta.md`). Las mismas
    filas del panel v1, con la censura que CLAUDE.md declara correcta desde el 22-09:

    * **Origen** `t0`: la fecha en que el odómetro llega a `c + G`. El gap sigue en km
      (regla 1), así que el riesgo arranca después de él.
    * `entry`: `max(0, inicio de la ventana − t0)`. Entrada tardía si el corte es anterior
      a la ventana del registro.
    * `exit`: `min(evento, fin de la ventana) − t0`. **Un sano sale en el fin de la
      ventana, no en su último viaje**: el registro habría anotado el evento igual.
    * `event`: el evento del vehículo cae en `(entry, exit]`.
    * `at_risk`: `exit > entry`. Una fila fuera de riesgo (un sano cuyo corte es posterior
      a la ventana) no informa el ajuste, pero sigue en el `y` para quedar alineada.

    Las columnas las deja `scripts/build_window_survival_panel.py`; acá no se recalcula
    nada, se valida que sean coherentes.
    """
    mask = np.asarray(train_mask, dtype=bool)
    train = panel.loc[mask]
    columns = (entry_column, exit_column, event_column, group_column, "event_observed")
    missing = [c for c in columns if c not in train]
    if missing:
        raise KeyError(
            f"El panel no tiene {missing}: el target `window_survival` es del panel en días "
            "(`python scripts/build_window_survival_panel.py --config configs/data/panel_survival_ps.yaml`)."
        )
    entry = train[entry_column].to_numpy(dtype=float)
    exit_ = train[exit_column].to_numpy(dtype=float)
    event = train[event_column].to_numpy(dtype=int)
    known = np.isfinite(entry) & np.isfinite(exit_)
    at_risk = known & (exit_ > entry)
    if (event.astype(bool) & ~at_risk).any():
        raise ValueError("Hay filas con evento y sin tramo en riesgo: el panel está mal armado")
    if (event.astype(bool) & (train["event_observed"].to_numpy(dtype=int) == 0)).any():
        raise ValueError("Hay filas con evento en la ventana y `event_observed == 0`")
    if (entry[known] < 0).any():
        raise ValueError("Hay entradas negativas: el tramo en riesgo empieza en `c + G`, no antes")

    groups = train[group_column].astype(str).to_numpy()
    width = max(1, max((len(g) for g in groups), default=1))
    y = np.empty(len(train), dtype=[
        ("entry", "f8"), ("exit", "f8"), ("event", "i1"), ("at_risk", "?"), ("group", f"U{width}"),
    ])
    y["entry"] = np.where(known, entry, 0.0)
    y["exit"] = np.where(known, exit_, 0.0)
    y["event"] = np.where(at_risk, event, 0)
    y["at_risk"] = at_risk
    y["group"] = groups
    info = {
        "n_rows": int(len(train)),
        "n_at_risk": int(at_risk.sum()),
        "n_events": int(y["event"].sum()),
        "n_origin_unknown": int((~known).sum()),
        "exposure_days_total": float(np.clip(exit_ - entry, 0, None)[at_risk].sum()),
    }
    logger.debug("target window_survival | %s", info)
    return TargetSpec(
        y=y,
        name="window_survival",
        params={"entry_column": entry_column, "exit_column": exit_column,
                "event_column": event_column, "group_column": group_column},
        info=info,
    )


# --------------------------------------------------------------------------- #
# La etiqueta de siempre, con el vehículo al lado
# --------------------------------------------------------------------------- #
@register_target("grouped_label")
def build_grouped_label_target(
    panel: pd.DataFrame,
    train_mask: np.ndarray,
    *,
    label_column: str = "label",
    group_column: str = GROUP_COLUMN,
) -> TargetSpec:
    """`(label, vehículo)` por fila de train: **el mismo objetivo**, con el vehículo al lado.

    No cambia con qué se entrena: el modelo interno recibe `label` tal cual. Existe para
    los envoltorios que necesitan saber de qué vehículo es cada fila y no lo pueden sacar
    de `X`, porque el `Pipeline` del fold se la pasa preprocesada y sin `vehicle_id`
    (`vehicle_bagging`, `src/models/bagging.py`, que remuestrea autos y no filas). La
    salida del modelo ya es P(evento en H), así que no necesita decoder propio.
    """
    mask = np.asarray(train_mask, dtype=bool)
    train = panel.loc[mask]
    missing = [c for c in (label_column, group_column) if c not in train]
    if missing:
        raise KeyError(f"El panel no tiene {missing}: no se puede armar `grouped_label`")

    groups = train[group_column].astype(str).to_numpy()
    width = max(1, max((len(g) for g in groups), default=1))
    y = np.empty(len(train), dtype=[("label", "i1"), ("group", f"U{width}")])
    y["label"] = train[label_column].astype(int).to_numpy()
    y["group"] = groups
    return TargetSpec(
        y=y,
        name="grouped_label",
        params={"label_column": label_column, "group_column": group_column},
        info={"n_rows": int(len(train)), "n_positive": int(y["label"].sum()),
              "n_vehicles": int(len(np.unique(groups)))},
    )


# --------------------------------------------------------------------------- #
# Multi-horizonte ordinal
# --------------------------------------------------------------------------- #
@register_target("ordinal_horizon")
def build_ordinal_target(
    panel: pd.DataFrame,
    train_mask: np.ndarray,
    bins_km: Sequence[float] | None = None,
    **_: Any,
) -> TargetSpec:
    """Convierte distancia al evento en clases ordenadas; los sanos son clase 0.

    Para K cortes crecientes hay K+1 clases. La clase 0 significa sano/no
    inminente: incluye censurados, eventos posteriores al último corte y cualquier
    evento dentro del gap de blanking. Dentro del horizonte visible, la clase 1 es
    el bucket más lejano y la clase K el más cercano. Por lo tanto, una clase mayor
    siempre significa un evento más próximo.

    **Dónde cae `bins_km[-1]` respecto de `G + H` decide qué mide el target.** Si el
    último corte coincide con `G + H`, todas las clases positivas viven adentro de la
    ventana que ya era `label = 1` y el target es un refinamiento estricto de la clase
    positiva: no dice nada de las filas negativas, que son el 87% del panel. Si el
    último corte lo supera, las clases intermedias se llenan con filas `label = 0` de
    vehículos que sí fallan —información nueva—, a cambio de parecerse más a
    "identificar la cohorte de muestreo", que en este panel *es* la etiqueta
    (CLAUDE.md). `info["class_0"]` deja medido ese riesgo: mientras la clase 0
    conserve filas de vehículos con evento, no es un indicador de cohorte.
    """
    bins = _validated_bins(bins_km)
    mask = np.asarray(train_mask, dtype=bool)
    if mask.ndim != 1 or len(mask) != len(panel):
        raise ValueError("`train_mask` tiene que ser booleano y medir lo mismo que el panel")
    train = panel.loc[mask]
    required = ("time_to_event_km", "event_observed", "gap_km")
    missing = [column for column in required if column not in train]
    if missing:
        raise KeyError(f"El panel no tiene {missing}: no se puede armar el target ordinal")

    observed = train["event_observed"].to_numpy(dtype=int) == 1
    tte = train["time_to_event_km"].to_numpy(dtype=float)
    if np.isnan(tte[observed]).any():
        raise ValueError("Hay filas con evento observado y `time_to_event_km` nulo")
    if (tte[observed] <= 0).any():
        raise ValueError("El target ordinal solo admite cortes anteriores al evento")

    y = np.zeros(len(train), dtype=np.int8)
    gap = train["gap_km"].to_numpy(dtype=float)
    imminent = observed & (tte >= gap) & (tte <= bins[-1])
    # searchsorted: 0 para <= primer bin y K-1 para el último bucket. Se invierte
    # para que la proximidad al evento crezca junto con el número de clase.
    y[imminent] = len(bins) - np.searchsorted(bins, tte[imminent], side="left")

    counts = np.bincount(y, minlength=len(bins) + 1)
    # Cuánto de cada clase viene de vehículos que fallan. En la clase 0 es la
    # medida directa del riesgo de cohorte: si llegara a cero, "clase 0" y
    # "vehículo sano" serían la misma variable y el target dejaría de ser un
    # horizonte para ser la etiqueta de muestreo.
    observed_per_class = np.bincount(y[observed], minlength=len(counts))
    info = {
        "n_rows": int(len(train)),
        "n_classes": int(len(counts)),
        "class_counts": {str(i): int(n) for i, n in enumerate(counts)},
        "class_event_counts": {str(i): int(n) for i, n in enumerate(observed_per_class)},
        "class_0": {
            "n_rows": int(counts[0]),
            "n_from_event_vehicles": int(observed_per_class[0]),
            "frac_from_event_vehicles": (
                float(observed_per_class[0] / counts[0]) if counts[0] else float("nan")
            ),
        },
    }
    logger.debug("target ordinal_horizon | %s", info)
    return TargetSpec(
        y=y,
        name="ordinal_horizon",
        params={"bins_km": bins.tolist()},
        info=info,
    )


@register_decoder("ordinal_horizon")
def decode_ordinal_predictions(
    estimator: Any,
    features: pd.DataFrame,
    *,
    bins_km: Sequence[float] | None = None,
    score_max_tte_km: float | None = None,
    **_: Any,
) -> TargetPredictions:
    """Reconstruye P(evento antes de H) desde probabilidades acumuladas ordinales."""
    bins = _validated_bins(bins_km)
    min_class = _score_class_min(bins, score_max_tte_km)
    if not hasattr(estimator, "predict_proba"):
        raise TypeError("El modelo ordinal tiene que exponer `predict_proba`")

    raw = np.asarray(estimator.predict_proba(features), dtype=float)
    classes = np.asarray(estimator.classes_, dtype=int)
    n_classes = len(bins) + 1
    probabilities = np.zeros((len(features), n_classes), dtype=float)
    for source, klass in enumerate(classes):
        if klass < 0 or klass >= n_classes:
            raise ValueError(f"El modelo devolvió la clase ordinal inesperada {klass}")
        probabilities[:, klass] = raw[:, source]

    extras = {
        f"ordinal_proba_{klass}": probabilities[:, klass]
        for klass in range(n_classes)
    }
    # P(Y >= k) es monótona por construcción porque se obtiene sumando las
    # probabilidades de clase. El score binario es la acumulada cuyo límite en km
    # coincide con el H declarado en el YAML.
    for klass in range(1, n_classes):
        extras[f"ordinal_cum_ge_{klass}"] = probabilities[:, klass:].sum(axis=1)
    score = extras[f"ordinal_cum_ge_{min_class}"]
    return TargetPredictions(score=score, extras=extras)


@register_reporter("ordinal_horizon")
def report_ordinal_predictions(
    predictions: pd.DataFrame,
    *,
    bins_km: Sequence[float] | None = None,
    score_max_tte_km: float | None = None,
    cost_matrix: Sequence[Sequence[float]] | None = None,
    **_: Any,
) -> dict[str, Any]:
    """Composición de las clases OOF y, si el YAML la declara, la matriz de costos.

    La matriz es **opcional**: a esta tasa base una matriz única con falsos negativos
    20–50 veces más caros vuelve degenerada la decisión (la política óptima es
    revisar todo, gane quien gane), así que el reporte de costo que sí discrimina es
    el barrido de `eval.cost_ratios` —`src/eval/metrics.py::cost_ratio_sweep()`—, que
    no depende del target y corre para cualquier corrida. La matriz se conserva para
    reproducir la variante de bins restringidos ya registrada.
    """
    from src.eval.metrics import cost_matrix_score

    bins = _validated_bins(bins_km)
    _score_class_min(bins, score_max_tte_km)  # valida trazabilidad del score binario
    n_classes = len(bins) + 1
    matrix = None
    if cost_matrix is not None:
        matrix = np.asarray(cost_matrix, dtype=float)
        if matrix.shape != (n_classes, n_classes):
            raise ValueError(
                f"La matriz de costos mide {matrix.shape}; se esperaba "
                f"{(n_classes, n_classes)} para {n_classes} clases"
            )

    probability_columns = [f"ordinal_proba_{klass}" for klass in range(n_classes)]
    missing = [column for column in probability_columns if column not in predictions]
    if missing:
        raise KeyError(f"Faltan probabilidades ordinales OOF: {missing}")
    probabilities = predictions[probability_columns].to_numpy(dtype=float)
    spec = build_ordinal_target(
        predictions, np.ones(len(predictions), dtype=bool), bins_km=bins
    )
    target = spec.y.astype(int)
    predictions["ordinal_target"] = target

    report: dict[str, Any] = {
        "name": "ordinal_horizon",
        "n_classes": n_classes,
        "bins_km": bins.tolist(),
        "score_max_tte_km": float(score_max_tte_km),
        "extends_beyond_horizon": bool(bins[-1] > float(score_max_tte_km)),
        "true_class_counts": {
            str(i): int(n)
            for i, n in enumerate(np.bincount(target, minlength=n_classes))
        },
        # El número que decide si el target es un horizonte o la cohorte disfrazada.
        "class_0": spec.info["class_0"],
        "class_event_counts": spec.info["class_event_counts"],
    }
    if matrix is None:
        return report

    expected_cost = probabilities @ matrix
    predicted = expected_cost.argmin(axis=1).astype(int)
    predictions["ordinal_prediction"] = predicted
    predictions["ordinal_expected_cost"] = expected_cost.min(axis=1)

    total = cost_matrix_score(target, predicted, matrix)
    argmax_total = cost_matrix_score(target, probabilities.argmax(axis=1), matrix)
    always_class_0 = cost_matrix_score(target, np.zeros(len(target), dtype=int), matrix)
    always_nearest = cost_matrix_score(
        target, np.full(len(target), n_classes - 1, dtype=int), matrix
    )
    report.update(
        {
            "decision_rule": "minimum_expected_cost",
            "cost_total": total,
            "cost_mean": float(total / len(predictions)) if len(predictions) else float("nan"),
            "cost_per_1000": (
                float(1000.0 * total / len(predictions)) if len(predictions) else float("nan")
            ),
            "argmax_cost_total": argmax_total,
            "always_class_0_cost_total": always_class_0,
            "always_nearest_cost_total": always_nearest,
            "savings_vs_always_class_0": float(always_class_0 - total),
            "savings_vs_always_nearest": float(always_nearest - total),
            "predicted_class_counts": {
                str(i): int(n)
                for i, n in enumerate(np.bincount(predicted, minlength=n_classes))
            },
        }
    )
    return report


# --------------------------------------------------------------------------- #
# Cure model por hito post-venta
# --------------------------------------------------------------------------- #
#: Campos del `y` estructurado que consume `src/models/cure.py` (y el normalizador de
#: flota, que usa `healthy` y `group`).
CURE_FIELDS = (
    "entry", "exit", "event", "at_risk", "healthy", "landmark", "score_start", "score_end", "group",
)


@register_target("cure_window")
def build_cure_window_target(
    panel: pd.DataFrame,
    train_mask: np.ndarray,
    *,
    entry_column: str = "aux_dss_entry",
    exit_column: str = "aux_dss_exit",
    event_column: str = "aux_event_in_window",
    landmark_column: str = "aux_landmark_day",
    gap_column: str = "aux_gap_days",
    horizon_column: str = "aux_horizon_days",
    group_column: str = GROUP_COLUMN,
    **_: Any,
) -> TargetSpec:
    """La exposición en la ventana del registro, por fila de train del panel de hitos.

    El reloj es días desde la venta. Para cada fila (vehículo, hito L):

    * `entry` y `exit`: el tramo `(e, x]` en que el registro podía anotar el evento,
      `e = max(L + G, inicio de la ventana)` y `x = min(evento, fin de la ventana)`. El
      gap de blanking (regla 1) ya está en `e`: el riesgo arranca en L + G;
    * `event`: δ, evento en `(e, x]`;
    * `at_risk`: `x > e`. Una fila fuera de riesgo **no informa la verosimilitud**, pero
      sigue en el `y` porque entra a la referencia de flota y se puntúa;
    * `healthy`: `event_observed == 0`, lo que usa la referencia de flota;
    * `landmark`, `score_start` = L + G y `score_end` = L + G + H: la ventana del score,
      que es la del reloj post-venta **sin** la del registro (lo que valdría en
      producción);
    * `group`: el vehículo.

    Todo sale de columnas que `scripts/build_landmark_panel.py` ya dejó en el panel: acá
    no se recalcula nada, se valida que sea coherente.
    """
    mask = np.asarray(train_mask, dtype=bool)
    train = panel.loc[mask]
    columns = (entry_column, exit_column, event_column, landmark_column, gap_column, horizon_column,
               group_column, "event_observed")
    missing = [c for c in columns if c not in train]
    if missing:
        raise KeyError(
            f"El panel no tiene {missing}: el target `cure_window` es del panel de hitos "
            "(`python scripts/build_landmark_panel.py --config configs/data/panel_landmark_ps.yaml`)."
        )
    numeric = train[[entry_column, exit_column, landmark_column, gap_column, horizon_column]].astype(float)
    if numeric.isna().any().any():
        raise ValueError(f"Hay filas de train con NaN en {numeric.columns[numeric.isna().any()].tolist()}")

    entry = numeric[entry_column].to_numpy()
    exit_ = numeric[exit_column].to_numpy()
    event = train[event_column].to_numpy(dtype=int)
    healthy = train["event_observed"].to_numpy(dtype=int) == 0
    at_risk = exit_ > entry
    if (event.astype(bool) & ~at_risk).any():
        raise ValueError("Hay filas con evento en la ventana y salida ≤ entrada: el panel está mal armado")
    if (event.astype(bool) & healthy).any():
        raise ValueError("Hay filas con evento en la ventana y `event_observed == 0`")
    landmark = numeric[landmark_column].to_numpy()
    score_start = landmark + numeric[gap_column].to_numpy()
    score_end = score_start + numeric[horizon_column].to_numpy()

    groups = train[group_column].astype(str).to_numpy()
    width = max(1, max((len(g) for g in groups), default=1))
    y = np.empty(len(train), dtype=[
        ("entry", "f8"), ("exit", "f8"), ("event", "i1"), ("at_risk", "?"), ("healthy", "?"),
        ("landmark", "f8"), ("score_start", "f8"), ("score_end", "f8"), ("group", f"U{width}"),
    ])
    y["entry"], y["exit"], y["event"], y["at_risk"], y["healthy"] = entry, exit_, event, at_risk, healthy
    y["landmark"], y["score_start"], y["score_end"], y["group"] = landmark, score_start, score_end, groups

    by_landmark = {
        f"{value:g}": {
            "n_rows": int((landmark == value).sum()),
            "n_at_risk": int((at_risk & (landmark == value)).sum()),
            "n_events": int(event[landmark == value].sum()),
        }
        for value in np.unique(landmark)
    }
    info = {
        "n_rows": int(len(train)),
        "n_at_risk": int(at_risk.sum()),
        "n_events": int(event.sum()),
        "n_healthy": int(healthy.sum()),
        "by_landmark": by_landmark,
    }
    logger.debug("target cure_window | %s", info)
    return TargetSpec(
        y=y,
        name="cure_window",
        params={"entry_column": entry_column, "exit_column": exit_column, "event_column": event_column,
                "landmark_column": landmark_column, "gap_column": gap_column,
                "horizon_column": horizon_column, "group_column": group_column},
        info=info,
    )


def _final_step(estimator: Any, features: pd.DataFrame) -> tuple[Any, Any]:
    """El último paso de un `Pipeline` y las features ya pasadas por los anteriores."""
    from sklearn.pipeline import Pipeline

    if isinstance(estimator, Pipeline):
        if len(estimator.steps) > 1:
            features = estimator[:-1].transform(features)
        return estimator[-1], features
    return estimator, features


@register_decoder("cure_window")
def decode_cure_predictions(estimator: Any, features: pd.DataFrame, **_: Any) -> TargetPredictions:
    """`score` = π_L(x)·[1 − S_u(L+G+H)/S_u(L+G)], más `pi_incidence`, `p_horizon` y (k, λ).

    Con incidencia de pesos unitarios (P0) viaja además `unit_weight_score`, el puntaje
    s con el que el preregistro mide D1 y D2 de P0.
    """
    model, transformed = _final_step(estimator, features)
    if not hasattr(model, "predict_components"):
        raise TypeError("El target `cure_window` va con `model: cure_mixture` (src/models/cure.py)")
    parts = dict(model.predict_components(transformed))
    score = parts.pop("score")
    return TargetPredictions(score=score, extras=parts)
