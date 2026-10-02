"""Survival stacking en días, con exposición exacta: exponencial por tramos (PEM).

Es el finalista (`src/models/survival_stacking.py`) con la cuenta del riesgo arreglada
(F5 §3.3, `docs/reproducibilidad.md`). Tres cosas cambian:

1. **El reloj.** Los bins son días desde `t0`, la fecha en que el odómetro llega a `c + G`,
   no km. La Fase 1 del cure model mostró que el evento se ordena por días
   (`docs/reproducibilidad.md`). La posición de la fila en el reloj del
   evento la lleva `feat_cut_dss`, que entra como cualquier feature.
2. **El conjunto en riesgo.** Cada fila está en riesgo en `(entry, exit]`, el tramo en que el
   registro de eventos podía anotarla (target `window_survival` de
   `src/training/targets.py`). Un corte anterior a la ventana entra tarde, y un sano sale en
   el fin de la ventana.
3. **La estimación.** La ventana y la entrada tardía cortan bins por la mitad. El apilado
   binario tendría que tirar esos bins o contarlos enteros; acá cada fila apilada lleva su
   exposición exacta en días, y el hazard se ajusta como una regresión de Poisson con
   `log(exposición)` de offset (Bender, Rügamer, Scheipl & Bischl 2020). En LightGBM es
   `objective: poisson` con `init_score`. Con `init_score`, LightGBM no arranca desde el
   promedio: la salida cruda del booster es directamente el log de la tasa por día.

Predicción: el hazard de cada bin de `[0, horizon_days]` es una tasa por día constante
dentro del bin, así que `S(H | x) = exp(−Σ λ_k · ancho_k)` y el score es `1 − S(H | x)`, el
riesgo de que el evento caiga en los `horizon_days` después de `c + G`. El horizonte es fijo:
no se traduce H de km a días con los km del vehículo, porque los futuros no se conocen en el
corte.

Todo lo demás es el finalista: el mismo `Pipeline` del fold (imputar, escalar, one-hot), los
mismos hiperparámetros del booster (`_booster_defaults`) y la misma covariable del hazard
base (el borde izquierdo del bin), ahora en días.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from src.models.survival_stacking import _booster_defaults, _ranges

logger = logging.getLogger(__name__)

#: Nombre de la covariable del hazard base (borde izquierdo del bin, en días desde `c + G`).
BIN_FEATURE = "surv_bin_start_days"
#: Tolerancia de los bordes de bin, en días: un tramo que roza un borde no crea un bin vacío.
EDGE_TOL = 1e-9


class WindowPEMStacker(BaseEstimator, ClassifierMixin):
    """Hazard por tramos en días, con exposición exacta; predice `1 − S(horizon_days | x)`.

    Recibe como `y` el array estructurado de
    `src/training/targets.py::build_window_survival_target` (`entry`, `exit`, `event`,
    `at_risk`, `group`) y como `X` la matriz que ya preprocesó el `Pipeline` del fold.

    Parámetros
    ----------
    horizon_days
        Hasta dónde se acumula el riesgo para el score, en días desde `c + G`.
    bin_days
        Ancho de los bins del hazard base.
    max_horizon_days
        Hasta dónde se entrena el hazard (`None` = `horizon_days`). Más allá, la fila se
        censura. Es el espejo de `max_horizon_km` del finalista.
    model_params
        Hiperparámetros del booster; los defaults son los del finalista. `objective` no se
        puede cambiar: el offset de exposición solo tiene sentido con Poisson.
    """

    def __init__(
        self,
        *,
        horizon_days: float = 30.0,
        bin_days: float = 5.0,
        max_horizon_days: float | None = None,
        model_params: dict[str, Any] | None = None,
        random_state: int = 42,
    ) -> None:
        self.horizon_days = horizon_days
        self.bin_days = bin_days
        self.max_horizon_days = max_horizon_days
        self.model_params = model_params
        self.random_state = random_state

    # ------------------------------------------------------------------ #
    # Apilado
    # ------------------------------------------------------------------ #
    def _n_bins(self, upper_days: float) -> int:
        n_bins = int(np.ceil(float(upper_days) / float(self.bin_days) - EDGE_TOL))
        if n_bins < 1:
            raise ValueError(f"bin_days={self.bin_days} no entra ni una vez en {upper_days} d: revisá el YAML")
        return n_bins

    def stack(
        self, X: np.ndarray, entry: np.ndarray, exit_: np.ndarray, event: np.ndarray, at_risk: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Filas apiladas: una por `(fila, bin con exposición > 0)` dentro de `[0, max_horizon_days]`.

        Devuelve `X` apilada (con el borde del bin al final), el evento, la exposición en
        días, el bin y la fila original de cada fila apilada.
        """
        width = float(self.bin_days)
        upper = self.n_bins_train_ * width
        exit_c = np.minimum(exit_, upper)
        # Censura administrativa: el evento que cae después del horizonte de entrenamiento no
        # se ve, igual que en el finalista.
        beyond = exit_ > upper + EDGE_TOL
        ok = at_risk & (exit_c > entry + EDGE_TOL)
        first = np.floor(entry / width + EDGE_TOL).astype(int)
        last = np.ceil(exit_c / width - EDGE_TOL).astype(int) - 1
        counts = np.where(ok, last - first + 1, 0)
        rows = np.repeat(np.arange(len(entry)), counts)
        if rows.size == 0:
            raise ValueError("Ninguna fila de train quedó en riesgo: ¿la ventana no cubre ningún corte?")
        bins = first[rows] + _ranges(counts)
        lo = bins * width
        exposure = np.minimum(lo + width, exit_c[rows]) - np.maximum(lo, entry[rows])
        # El evento va en el bin que contiene la salida (borde derecho cerrado).
        hazard = ((bins == last[rows]) & (event[rows] == 1) & ~beyond[rows]).astype(float)
        keep = exposure > EDGE_TOL
        if (hazard.astype(bool) & ~keep).any():
            raise RuntimeError("Un evento cayó en un bin sin exposición: revisá los bordes del apilado")
        return {
            "X": np.column_stack([X[rows[keep]], lo[keep]]),
            "event": hazard[keep],
            "exposure": exposure[keep],
            "bin": bins[keep],
            "row": rows[keep],
        }

    # ------------------------------------------------------------------ #
    # sklearn
    # ------------------------------------------------------------------ #
    def fit(self, X, y, **fit_params):  # noqa: D102
        from lightgbm import LGBMRegressor  # import adentro: dependencia del modelo

        X = np.asarray(X, dtype=float)
        entry, exit_, event, at_risk = _unpack_window_target(y)
        if len(entry) != len(X):
            raise ValueError(f"X tiene {len(X)} filas y el target {len(entry)}")
        upper = float(self.max_horizon_days or self.horizon_days)
        if upper < float(self.horizon_days):
            raise ValueError(
                f"max_horizon_days={upper} < horizon_days={self.horizon_days}: no se podría "
                "componer la supervivencia hasta el horizonte del score."
            )
        self.n_bins_train_ = self._n_bins(upper)
        self.n_bins_score_ = self._n_bins(float(self.horizon_days))

        stacked = self.stack(X, entry, exit_, event, at_risk)
        self.n_features_in_ = X.shape[1]
        self.classes_ = np.array([0, 1])
        self.stacking_ = {
            "rows_in": int(len(X)),
            "rows_at_risk": int(np.unique(stacked["row"]).size),
            "rows_late_entry": int((at_risk & (entry > 0)).sum()),
            "rows_stacked": int(len(stacked["event"])),
            "n_bins_train": self.n_bins_train_,
            "n_bins_score": self.n_bins_score_,
            "exposure_days": float(stacked["exposure"].sum()),
            "n_hazard_events": int(stacked["event"].sum()),
            "rate_per_day": float(stacked["event"].sum() / stacked["exposure"].sum()),
        }
        logger.info(
            "window PEM | %d filas -> %d apiladas (%d en riesgo, %d con entrada tardía) en %d bins de "
            "%.0f d | %d eventos en %.0f días-auto",
            self.stacking_["rows_in"], self.stacking_["rows_stacked"], self.stacking_["rows_at_risk"],
            self.stacking_["rows_late_entry"], self.n_bins_train_, float(self.bin_days),
            self.stacking_["n_hazard_events"], self.stacking_["exposure_days"],
        )

        params = _booster_defaults(self.model_params, self.random_state)
        params.pop("class_weight", None)
        objective = params.pop("objective", "poisson")
        if objective != "poisson":
            raise ValueError(f"objective `{objective}` no soportado: el offset de exposición es de Poisson")
        if self.stacking_["n_hazard_events"] == 0:
            raise ValueError("Ningún evento dentro del horizonte de entrenamiento: no hay tasa que estimar")
        # Con `init_score`, LightGBM no arranca desde el promedio (`boost_from_average` se
        # apaga) y, con la tasa de aprendizaje chica del finalista, 300 árboles no alcanzan a
        # bajar el intercepto desde 0 hasta log(0,01). El offset lleva entonces la tasa global
        # del train, que es el MLE del modelo sin covariables, y el booster aprende los
        # desvíos: lo mismo que haría `boost_from_average` sin offset.
        self.base_log_rate_ = float(np.log(self.stacking_["rate_per_day"]))
        self.booster_ = LGBMRegressor(objective="poisson", **params)
        self.booster_.fit(
            stacked["X"], stacked["event"], init_score=np.log(stacked["exposure"]) + self.base_log_rate_
        )
        return self

    def log_rate(self, X, bin_start_days: np.ndarray) -> np.ndarray:
        """Log de la tasa por día de cada fila de `X` en el bin que empieza en `bin_start_days`."""
        X = np.asarray(X, dtype=float)
        raw = self.booster_.predict(np.column_stack([X, bin_start_days]), raw_score=True)
        return self.base_log_rate_ + np.asarray(raw, dtype=float)

    def cumulative_hazard(self, X, horizon_days: float | None = None) -> np.ndarray:
        """`Λ(H | x) = Σ λ_k · ancho_k` sobre los bins de `[0, H]`; el último puede ir cortado."""
        X = np.asarray(X, dtype=float)
        horizon = float(self.horizon_days if horizon_days is None else horizon_days)
        width = float(self.bin_days)
        n_bins = self._n_bins(horizon)
        if n_bins > self.n_bins_train_:
            raise ValueError(f"H = {horizon} d pasa el horizonte de entrenamiento ({self.n_bins_train_ * width:g} d)")
        edges = np.arange(n_bins, dtype=float) * width
        widths = np.minimum(width, horizon - edges)
        n_rows = len(X)
        rows = np.repeat(np.arange(n_rows), n_bins)
        rates = np.exp(self.log_rate(X[rows], np.tile(edges, n_rows))).reshape(n_rows, n_bins)
        return (rates * widths).sum(axis=1)

    def predict_proba(self, X) -> np.ndarray:  # noqa: D102
        risk = -np.expm1(-self.cumulative_hazard(X))
        return np.column_stack([1.0 - risk, risk])

    def predict(self, X) -> np.ndarray:  # noqa: D102
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    @property
    def extra_feature_names_(self) -> list[str]:
        """Columnas que el modelo agrega a las del panel, en orden. Para las auditorías."""
        return [BIN_FEATURE]

    @property
    def feature_importances_(self) -> np.ndarray:
        """Importancias del booster, en el orden `[features de X, bin]`."""
        return np.asarray(self.booster_.feature_importances_, dtype=float)


def _unpack_window_target(y) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """`(entry, exit, evento, en riesgo)` desde el array estructurado del target."""
    y = np.asarray(y)
    if y.dtype.names is None or "entry" not in y.dtype.names:
        raise TypeError(
            "WindowPEMStacker espera el target de la ventana, no `label`. Poné "
            "`target: {name: window_survival}` en el YAML del experimento."
        )
    return (y["entry"].astype(float), y["exit"].astype(float), y["event"].astype(int),
            y["at_risk"].astype(bool))
