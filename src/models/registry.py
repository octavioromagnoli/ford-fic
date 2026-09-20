"""Registry de modelos: `get_model(name, params)` y nada más.

`scripts/train.py` no importa ningún estimador directamente; pide el nombre que
está en el YAML. Agregar un modelo (F3/F5) es registrar un builder acá y escribir
un config, sin tocar el código de entrenamiento.

En F0 solo vive el predictor por tasa base, que es el piso absoluto: un modelo
que no le gana está roto.
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


# --------------------------------------------------------------------------------------
# F3 · la escalera de modelos. Tres peldaños, de menos a más capacidad, y en ese orden
# a propósito: con 254 filas positivas repartidas en 53 vehículos, la pregunta no es cuál
# gana sino **cuánta capacidad tolera el panel antes de sobreajustar**. Si el GBM no le
# gana a la logística, la respuesta no es buscar otro GBM.
# --------------------------------------------------------------------------------------
@register("logistic")
def _build_logistic(params: dict[str, Any]) -> BaseEstimator:
    """Logística con L2: el primer modelo honesto. Lineal sobre features ya escaladas.

    `class_weight="balanced"` en vez de resamplear: se calcula dentro del fold, así que
    no hay leakage, y con 12,5% de positivos el modelo sin pesos predice la clase mayoritaria.
    `C` bajo por default porque hay más features que vehículos con evento.
    """
    from sklearn.linear_model import LogisticRegression

    params.setdefault("C", 0.1)
    # sklearn 1.9 deprecó `penalty`: la mezcla se declara con `l1_ratio` (0 = L2, 1 = L1).
    params.setdefault("l1_ratio", 0.0)
    params.setdefault("class_weight", "balanced")
    params.setdefault("max_iter", 2000)
    params.setdefault("random_state", 42)
    return LogisticRegression(**params)


@register("logistic_l1")
def _build_logistic_l1(params: dict[str, Any]) -> BaseEstimator:
    """Logística con L1: la misma, pero que además elija features.

    Con ~100 columnas y 53 vehículos con evento, el conjunto de features que sobrevive
    a la L1 dice más que el coeficiente de cualquiera de ellas. `saga` porque es el
    solver que soporta L1 con este tamaño.
    """
    from sklearn.linear_model import LogisticRegression

    params.setdefault("C", 0.05)
    params.setdefault("l1_ratio", 1.0)
    params.setdefault("solver", "saga")
    params.setdefault("class_weight", "balanced")
    params.setdefault("max_iter", 5000)
    params.setdefault("random_state", 42)
    return LogisticRegression(**params)


@register("gbm")
def _build_gbm(params: dict[str, Any]) -> BaseEstimator:
    """Boosting de sklearn (histograma): interacciones sin depender de otra librería.

    Regularización fuerte por default —árboles chicos, pocas hojas, learning rate
    bajo— porque el panel es chico y el modelo tiene con qué memorizar 53 vehículos.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier

    params.setdefault("max_depth", 3)
    params.setdefault("max_leaf_nodes", 7)
    params.setdefault("learning_rate", 0.05)
    params.setdefault("max_iter", 200)
    params.setdefault("min_samples_leaf", 30)
    params.setdefault("l2_regularization", 1.0)
    params.setdefault("early_stopping", False)   # el fold ya es chico: partirlo otra vez es ruido
    params.setdefault("class_weight", "balanced")
    params.setdefault("random_state", 42)
    return HistGradientBoostingClassifier(**params)


@register("lgbm")
def _build_lgbm(params: dict[str, Any]) -> BaseEstimator:
    """LightGBM: el mismo peldaño que `gbm`, para que el resultado no dependa de una implementación."""
    from lightgbm import LGBMClassifier

    params.setdefault("n_estimators", 300)
    params.setdefault("learning_rate", 0.05)
    params.setdefault("num_leaves", 7)
    params.setdefault("min_child_samples", 30)
    params.setdefault("reg_lambda", 1.0)
    params.setdefault("subsample", 0.8)
    params.setdefault("subsample_freq", 1)
    params.setdefault("colsample_bytree", 0.6)
    params.setdefault("class_weight", "balanced")
    params.setdefault("verbosity", -1)
    params.setdefault("random_state", 42)
    return LGBMClassifier(**params)
