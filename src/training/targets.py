"""Objetivos de entrenamiento y decodificación de scores, registrados por nombre.

`src/training/cv.py` evalúa siempre contra la etiqueta binaria dura `label`. Este
módulo decide con qué objetivo se ajusta el estimador y cómo sus salidas vuelven a
un score comparable. Agregar otro modo requiere registrar funciones acá; el loop de
CV no contiene lógica específica de ordinal, supervivencia ni ninguna otra familia.

Todo target se construye exclusivamente con `panel.loc[train_mask]`. Los bins,
horizontes y costos vienen del YAML del experimento, nunca de constantes del código.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TargetSpec:
    """Target alineado 1:1 con `panel.loc[train_mask]` y metadata auditable."""

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
    info = {
        "n_rows": int(len(train)),
        "n_classes": int(len(counts)),
        "class_counts": {str(i): int(n) for i, n in enumerate(counts)},
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
    """Decisión de costo esperado y costo OOF realizado sobre las clases ordinales."""
    from src.eval.metrics import cost_matrix_score

    bins = _validated_bins(bins_km)
    _score_class_min(bins, score_max_tte_km)  # valida trazabilidad del score binario
    if cost_matrix is None:
        raise ValueError("`eval.cost_matrix` es obligatoria para el target ordinal")
    matrix = np.asarray(cost_matrix, dtype=float)
    n_classes = len(bins) + 1
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
    expected_cost = probabilities @ matrix
    predicted = expected_cost.argmin(axis=1).astype(int)
    target = build_ordinal_target(
        predictions, np.ones(len(predictions), dtype=bool), bins_km=bins
    ).y.astype(int)

    predictions["ordinal_target"] = target
    predictions["ordinal_prediction"] = predicted
    predictions["ordinal_expected_cost"] = expected_cost.min(axis=1)

    total = cost_matrix_score(target, predicted, matrix)
    argmax_prediction = probabilities.argmax(axis=1)
    argmax_total = cost_matrix_score(target, argmax_prediction, matrix)
    always_class_0 = cost_matrix_score(target, np.zeros(len(target), dtype=int), matrix)
    always_nearest = cost_matrix_score(
        target, np.full(len(target), n_classes - 1, dtype=int), matrix
    )
    return {
        "name": "ordinal_horizon",
        "n_classes": n_classes,
        "score_max_tte_km": float(score_max_tte_km),
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
        "true_class_counts": {
            str(i): int(n)
            for i, n in enumerate(np.bincount(target, minlength=n_classes))
        },
        "predicted_class_counts": {
            str(i): int(n)
            for i, n in enumerate(np.bincount(predicted, minlength=n_classes))
        },
    }
