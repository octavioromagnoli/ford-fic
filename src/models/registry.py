"""Registry de modelos: `get_model(name, params)` y nada más.

`scripts/train.py` no importa ningún estimador directamente; pide el nombre que
está en el YAML. Agregar un modelo (F3/F5) es registrar un builder acá y escribir
un config, sin tocar el código de entrenamiento.

`baserate` es el piso absoluto (un modelo que no le gana está roto) y `lgbm` el GBM
chico de referencia de F3, contra el que se miden las familias de features nuevas.

Un builder puede pedir un objetivo de entrenamiento distinto de `label` (los
`*_survival` piden `discrete_survival`): eso se declara en el `target:` del YAML y lo
resuelve `src/training/targets.py`, no el registry.
"""

from __future__ import annotations

from typing import Any, Callable

from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier

ModelBuilder = Callable[[dict[str, Any]], BaseEstimator]

_REGISTRY: dict[str, ModelBuilder] = {}


def register(name: str) -> Callable[[ModelBuilder], ModelBuilder]:
    def decorator(builder: ModelBuilder) -> ModelBuilder:
        if name in _REGISTRY:
            raise ValueError(f"El modelo `{name}` ya está registrado")
        _REGISTRY[name] = builder
        return builder

    return decorator


def available_models() -> list[str]:
    return sorted(_REGISTRY)


def get_model(name: str, params: dict[str, Any] | None = None) -> BaseEstimator:
    """Instancia un modelo del registry con los params del config."""
    if name not in _REGISTRY:
        raise KeyError(f"Modelo `{name}` no registrado. Disponibles: {available_models()}")
    return _REGISTRY[name](dict(params or {}))


@register("baserate")
def _build_baserate(params: dict[str, Any]) -> BaseEstimator:
    """Tasa base: predice la prevalencia del train. Piso de referencia de F3."""
    params.setdefault("strategy", "prior")
    return DummyClassifier(**params)


@register("random")
def _build_random(params: dict[str, Any]) -> BaseEstimator:
    """Scores aleatorios uniformes: sirve para verificar que las métricas dan basura."""
    params.setdefault("strategy", "uniform")
    params.setdefault("random_state", 42)
    return DummyClassifier(**params)


@register("lgbm")
def _build_lgbm(params: dict[str, Any]) -> BaseEstimator:
    """GBM chico de referencia de F3 (docs/f3-modelos-candidatos.md §1.3, sin restricciones).

    Los defaults son los del doc para el panel v1: ~2.000 filas y ~250 positivas de
    ~50 vehículos. Con más hojas o menos regularización, un GBM memoriza vehículos.
    """
    from lightgbm import LGBMClassifier  # import adentro: no encarece el import del registry

    params.setdefault("n_estimators", 300)
    params.setdefault("learning_rate", 0.03)
    params.setdefault("num_leaves", 7)
    params.setdefault("min_child_samples", 40)
    params.setdefault("reg_lambda", 5.0)
    params.setdefault("subsample", 0.8)
    params.setdefault("subsample_freq", 1)
    params.setdefault("colsample_bytree", 0.7)
    params.setdefault("class_weight", "balanced")
    params.setdefault("random_state", 42)
    params.setdefault("verbose", -1)
    return LGBMClassifier(**params)


@register("cnn_lstm")
def _build_cnn_lstm(params: dict[str, Any]) -> BaseEstimator:
    """Baseline de la tutora: Conv1D + LSTM sobre la secuencia de `signals`, y las estáticas al final.

    Solo corre sobre el panel secuencial (`scripts/build_seq_panel.py`): la forma del
    tensor sale de `sequence_meta`. Arquitectura y decisiones en `src/models/cnn_lstm.py`.
    """
    from src.models.cnn_lstm import CNNLSTMClassifier  # import adentro: torch es opcional

    params.setdefault("class_weight", "balanced")
    params.setdefault("random_state", 42)
    return CNNLSTMClassifier(**params)


@register("survival_stacking")
def _build_survival_stacking(params: dict[str, Any]) -> BaseEstimator:
    """Supervivencia en tiempo discreto sobre las filas apiladas, sin efecto aleatorio.

    Necesita `target: {name: discrete_survival}` en el YAML: entrena el hazard por bin
    de km y predice el riesgo acumulado a H. Es el control de `gpboost_survival`: la
    diferencia entre los dos es lo que aporta el efecto aleatorio por vehículo, y nada
    más. Detalle en `src/models/survival_stacking.py`.
    """
    from src.models.survival_stacking import DiscreteSurvivalStacker

    params.setdefault("backend", "lightgbm")
    params.setdefault("random_state", 42)
    return DiscreteSurvivalStacker(**params)


@register("gpboost_survival")
def _build_gpboost_survival(params: dict[str, Any]) -> BaseEstimator:
    """Lo mismo con un intercept aleatorio por `vehicle_id` (GPBoost, Sigrist).

    Los ~5 cortes de un vehículo comparten auto, conductor y ruta: el efecto aleatorio
    absorbe ese nivel para que los árboles aprendan lo que cambia **dentro** del
    vehículo. GPBoost es dependencia opcional (`requirements-gpboost.txt`) y por eso el
    import vive adentro del builder.
    """
    from src.models.survival_stacking import DiscreteSurvivalStacker

    params.setdefault("backend", "gpboost")
    params.setdefault("random_state", 42)
    return DiscreteSurvivalStacker(**params)


@register("lgbm_ordinal")
def _build_lgbm_ordinal(params: dict[str, Any]) -> BaseEstimator:
    """LightGBM multiclase chico; `targets.py` acumula clases para recuperar P(evento en H)."""
    from lightgbm import LGBMClassifier  # import pesado adentro del builder

    params.setdefault("objective", "multiclass")
    params.setdefault("n_estimators", 300)
    params.setdefault("learning_rate", 0.03)
    params.setdefault("num_leaves", 7)
    params.setdefault("min_child_samples", 40)
    params.setdefault("reg_lambda", 5.0)
    params.setdefault("subsample", 0.8)
    params.setdefault("subsample_freq", 1)
    params.setdefault("colsample_bytree", 0.7)
    params.setdefault("class_weight", "balanced")
    params.setdefault("random_state", 42)
    params.setdefault("verbose", -1)
    return LGBMClassifier(**params)
