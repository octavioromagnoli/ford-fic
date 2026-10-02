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


@register("logistic")
def _build_logistic(params: dict[str, Any]) -> BaseEstimator:
    """Logística con L2 sobre features ya escaladas (el `Pipeline` de `cv.py` escala e imputa por fold).

    Traída de `feat/f3-features-regeneracion` para la re-medición completa sobre v2.
    `class_weight="balanced"` se calcula dentro del fold; `C` bajo porque hay más features
    que vehículos con evento por fold.
    """
    from sklearn.linear_model import LogisticRegression

    params.setdefault("C", 0.1)
    # sklearn 1.8+ declara la mezcla con `l1_ratio` (0 = L2, 1 = L1).
    params.setdefault("l1_ratio", 0.0)
    params.setdefault("class_weight", "balanced")
    params.setdefault("max_iter", 2000)
    params.setdefault("random_state", 42)
    return LogisticRegression(**params)


@register("logistic_l1")
def _build_logistic_l1(params: dict[str, Any]) -> BaseEstimator:
    """La misma logística con L1: además elige features. `saga` es el solver que soporta L1."""
    from sklearn.linear_model import LogisticRegression

    params.setdefault("C", 0.05)
    params.setdefault("l1_ratio", 1.0)
    params.setdefault("solver", "saga")
    params.setdefault("class_weight", "balanced")
    params.setdefault("max_iter", 5000)
    params.setdefault("random_state", 42)
    return LogisticRegression(**params)


@register("lgbm")
def _build_lgbm(params: dict[str, Any]) -> BaseEstimator:
    """GBM chico de referencia de F3 (CLAUDE.md, sin restricciones).

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


@register("gru_seq")
def _build_gru_seq(params: dict[str, Any]) -> BaseEstimator:
    """GRU unidireccional sobre la secuencia de la ventana, con pooling por atención y las estáticas al final.

    La pregunta que responde: ¿el ORDEN en km de la ventana aporta algo que los 53
    agregados del panel v1 tiran? Solo corre sobre un panel secuencial
    (`scripts/build_seq_panel.py`): la forma del tensor sale de `sequence_meta`.
    Arquitectura y defaults en `src/models/gru_seq.py`.
    """
    from src.models.gru_seq import GRUSeqClassifier  # import adentro: torch es opcional

    params.setdefault("class_weight", "balanced")
    params.setdefault("random_state", 42)
    return GRUSeqClassifier(**params)


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


@register("window_survival_stacking")
def _build_window_survival_stacking(params: dict[str, Any]) -> BaseEstimator:
    """El mismo apilado en días desde `c + G`, con el conjunto en riesgo de la ventana (F5 §3.3).

    Necesita `target: {name: window_survival}`. Hazard por tramos con exposición exacta
    (Poisson con offset) y score `1 − S(horizon_days | x)`. Detalle en
    `src/models/window_stacking.py`.
    """
    from src.models.window_stacking import WindowPEMStacker

    params.setdefault("random_state", 42)
    return WindowPEMStacker(**params)


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


@register("cure_mixture")
def _build_cure_mixture(params: dict[str, Any]) -> BaseEstimator:
    """Mixture cure model por hito post-venta: P0 (`incidence: unit_weight`) y P1 (`firth`).

    Va de a pares con `target: {name: cure_window}` y `preprocessing: none`: trae adentro
    la normalización contra la flota, la imputación y la escala por hito, y las ajusta
    con el train del fold. Detalle y verosimilitud en `src/models/cure.py`; qué corre y
    qué decide, en `docs/reproducibilidad.md`.
    """
    from src.models.cure import CureMixtureModel

    params.setdefault("random_state", 42)
    return CureMixtureModel(**params)


@register("external_incidence")
def _build_external_incidence(params: dict[str, Any]) -> BaseEstimator:
    """Incidencia aprendida con el conjunto externo y aplicada congelada (F5 §3.2, A-solo).

    `fit` no aprende nada: carga el ajuste de la fuente (`artifact`, el `.joblib` de
    `scripts/fit_external_incidence.py`) y puntúa. Va con `preprocessing: none`, porque la
    normalización contra la flota, la imputación y la escala son las de la fuente. Detalle en
    `src/models/incidence.py`; qué decide, en `docs/reproducibilidad.md`.
    """
    from src.models.incidence import FrozenIncidenceScorer

    return FrozenIncidenceScorer(**params)


@register("vehicle_bagging")
def _build_vehicle_bagging(params: dict[str, Any]) -> BaseEstimator:
    """Cualquier modelo del registry, promediado sobre `n_bags` bootstraps de vehículos.

    `params`: `base_model` (nombre en este registry), `base_params` (los hiperparámetros
    de la corrida sin bagging, sin cambios), `n_bags` y `random_state` (el sorteo de
    vehículos). Necesita el vehículo en el `y`: `discrete_survival` lo trae y los binarios
    van con `target: {name: grouped_label}`. Es la pieza de E2
    (`docs/reproducibilidad.md`); detalle en `src/models/bagging.py`.
    """
    from src.models.bagging import VehicleBaggingClassifier

    if "base_model" not in params:
        raise KeyError("`vehicle_bagging` necesita `base_model` en model.params")
    if params["base_model"] == "vehicle_bagging":
        raise ValueError("`vehicle_bagging` no se anida en sí mismo")
    params.setdefault("n_bags", 10)
    params.setdefault("random_state", 42)
    return VehicleBaggingClassifier(**params)


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
