"""Supervivencia en tiempo discreto sobre el eje de km, como problema de clasificación.

Craig, Zhong & Tibshirani (2021, arXiv:2107.13480) muestran que apilar un dataset
persona-período con etiqueta binaria y correr un clasificador estándar es
aproximadamente equivalente a un Cox. El panel de este repo **ya es** ese dataset: una
fila por `(vehículo, corte)`. Lo que falta es usarlo como tal.

## Qué cambia respecto de la etiqueta binaria

`label` contesta "¿el evento cae en `[c+G, c+G+H]`?" y descarta tres cosas:

1. **Dónde** dentro del horizonte cae el evento. Un corte a 600 km del evento y otro a
   2.900 km valen lo mismo, y no es lo mismo.
2. **Qué pasa después de H.** Un vehículo que falla a 5.000 km del corte entra como 0,
   igual que uno que no falla nunca. Acá, si `max_horizon_km > H`, ese 0 se convierte en
   "sobrevivió los primeros H y falló después", que es información.
3. **Que los sanos son censurados.** `aux_km_observed_after_cut` dice hasta dónde se los
   observó; más allá de eso la fila sale del conjunto en riesgo en vez de contar como un
   0 que nadie verificó.

## Cómo

La ventana `[0, max_horizon_km)` se parte en bins de `bin_km`. Cada fila del panel se
expande a una fila apilada por bin **en el que estuvo en riesgo**, con el borde
izquierdo del bin como covariable extra (es el hazard base: el modelo aprende la forma
del riesgo con los km, sin que se la impongamos). El objetivo de la fila apilada es el
*hazard* discreto: 1 solo en el bin donde cae el evento.

El origen de la duración es `c + G`, no `c`: el gap de blanking (CLAUDE.md, regla 1) lo
aplica `src/training/targets.py` antes de que el estimador vea nada. Una fila cuya
duración es ≤ 0 no está en riesgo en ningún km visible y no genera filas apiladas.

Predicción: los hazards de los bins que cubren `[0, H]` se componen en la supervivencia
`S(H|x) = Π (1 − h_k)` y el score es el **riesgo acumulado a H**, `1 − S(H|x)`. Esa es
la probabilidad de exactamente el evento que mide `label`, así que el PR-AUC sigue
siendo comparable con cualquier otra corrida de la tabla, con los mismos folds.

## El efecto aleatorio por vehículo

Los ~5 cortes de un vehículo no son independientes: comparten el auto, el conductor y
la ruta. Tratarlos como si lo fueran es la trampa que `docs/f3-modelos-candidatos.md`
§2.1 anota para el Cox por fila. Con `backend: gpboost` (Sigrist, *tree-boosting with
grouped random effects*) el intercept por `vehicle_id` absorbe ese nivel y los árboles
se quedan con lo que varía **dentro** del vehículo, que es donde el EDA dice que está la
señal (ICC 0,3–0,6).

Al predecir, el vehículo de validación **nunca** fue visto en train (el split es
agrupado, regla 2): su efecto aleatorio es desconocido y GPBoost le asigna la media a
priori. O sea que el efecto aleatorio no ayuda en validación por la puerta de atrás —
actúa solo como regularizador al entrenar—, que es exactamente lo que queremos y lo que
hace que el número siga siendo honesto.

`backend: lightgbm` es el mismo apilado sin efecto aleatorio: el control que aísla
cuánto aporta el stacking y cuánto el efecto por vehículo.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

logger = logging.getLogger(__name__)

BACKENDS = ("lightgbm", "gpboost")
#: Nombre de la covariable que lleva el hazard base (borde izquierdo del bin, en km).
BIN_FEATURE = "surv_bin_start_km"


class DiscreteSurvivalStacker(BaseEstimator, ClassifierMixin):
    """Clasificador sklearn-like que entrena un hazard discreto y predice `1 − S(H|x)`.

    Recibe como `y` el array estructurado de
    `src/training/targets.py::build_discrete_survival_target` (`duration_km`, `event`,
    `group`) y como `X` la matriz que ya preprocesó el `Pipeline` del fold. Expone
    `predict_proba` para que `src/training/cv.py` lo trate como cualquier otro modelo.

    Parámetros
    ----------
    horizon_km
        H del panel: hasta dónde se acumula el riesgo para el score. Es lo que define
        `label`, así que tiene que coincidir con el panel o el PR-AUC deja de medir lo
        mismo que las otras corridas.
    bin_km
        Ancho de los bins del hazard. Con Δ = 500 y H = 3.000 salen 6 bins; más finos
        dan más filas apiladas y un hazard base más flexible, pero menos eventos por bin.
    max_horizon_km
        Hasta dónde se entrena el hazard. `None` = `horizon_km`. Más allá de H es donde
        el apilado gana algo que la etiqueta binaria no tiene: los eventos que caen
        después del horizonte dejan de ser ceros indistinguibles de un sano.
    backend
        `lightgbm` (sin efecto aleatorio) o `gpboost` (intercept aleatorio por vehículo).
    model_params
        Hiperparámetros del booster. Los defaults son los del LightGBM chico de F3
        (`docs/f3-modelos-candidatos.md` §1.3) adaptados al tamaño del dataset apilado.
    gp_params
        Extras del `GPModel` de GPBoost (`likelihood`, etc.). Solo con `backend: gpboost`.
    """

    def __init__(
        self,
        *,
        horizon_km: float = 3000.0,
        bin_km: float = 500.0,
        max_horizon_km: float | None = None,
        backend: str = "lightgbm",
        model_params: dict[str, Any] | None = None,
        gp_params: dict[str, Any] | None = None,
        random_state: int = 42,
    ) -> None:
        self.horizon_km = horizon_km
        self.bin_km = bin_km
        self.max_horizon_km = max_horizon_km
        self.backend = backend
        self.model_params = model_params
        self.gp_params = gp_params
        self.random_state = random_state

    # ------------------------------------------------------------------ #
    # Apilado
    # ------------------------------------------------------------------ #
    def _bin_edges(self, upper_km: float) -> np.ndarray:
        n_bins = int(np.ceil(upper_km / float(self.bin_km) - 1e-9))
        if n_bins < 1:
            raise ValueError(
                f"bin_km={self.bin_km} no entra ni una vez en {upper_km} km: revisá el YAML"
            )
        return np.arange(n_bins, dtype=float) * float(self.bin_km)

    def _stack(
        self,
        X: np.ndarray,
        duration_km: np.ndarray,
        event: np.ndarray,
        at_risk: np.ndarray,
        groups: np.ndarray | None,
        entry_km: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray]:
        """Filas apiladas: una por `(fila del panel, bin en riesgo)`.

        Con `entry_km` (entrada tardía, target `window_km_survival`) la fila se apila desde
        el bin que contiene la entrada: los km anteriores no son supervivencia verificada.
        Devuelve `(X_apilada, hazard, grupos_apilados, índice de la fila original)`.
        """
        edges = self._train_edges_
        n_bins = len(edges)
        upper = n_bins * float(self.bin_km)
        # Último bin en el que la fila estuvo en riesgo. Las que el target marcó fuera de
        # riesgo (el evento cayó dentro del gap) no generan ninguna: usarlas sería mirar
        # los km que la regla 1 prohíbe. Las que pasan el final se censuran
        # administrativamente. El borde derecho va cerrado, igual que `label`.
        beyond = duration_km > upper
        last_bin = np.minimum(np.floor(duration_km / float(self.bin_km)).astype(int), n_bins - 1)
        entry = np.zeros(len(duration_km)) if entry_km is None else np.asarray(entry_km, dtype=float)
        first_bin = np.floor(entry / float(self.bin_km)).astype(int)
        # Una fila que entra después del final del entrenamiento no está en riesgo en ningún bin.
        n_at_risk = np.where(at_risk & (entry < upper), np.maximum(last_bin - first_bin + 1, 0), 0)

        rows = np.repeat(np.arange(len(duration_km)), n_at_risk)
        if rows.size == 0:
            raise ValueError(
                "Ninguna fila de train quedó en riesgo: ¿el gap se come todo el "
                "seguimiento? Revisá `gap_km` y `aux_km_observed_after_cut`."
            )
        bins = _ranges(n_at_risk) + first_bin[rows]
        # Hazard = 1 solo en el bin donde cae el evento, y solo si el evento se observó
        # antes de que se acabe la ventana de entrenamiento.
        hazard = ((bins == last_bin[rows]) & (event[rows] == 1) & ~beyond[rows]).astype(np.int8)
        stacked = np.column_stack([X[rows], edges[bins]])
        stacked_groups = groups[rows] if groups is not None else None
        return stacked, hazard, stacked_groups, rows

    # ------------------------------------------------------------------ #
    # sklearn
    # ------------------------------------------------------------------ #
    def fit(self, X, y, **fit_params):  # noqa: D102
        if self.backend not in BACKENDS:
            raise ValueError(f"backend `{self.backend}` desconocido. Opciones: {BACKENDS}")
        X = np.asarray(X, dtype=float)
        duration_km, event, at_risk, groups, entry_km = _unpack_target(y)
        if len(duration_km) != len(X):
            raise ValueError(f"X tiene {len(X)} filas y el target {len(duration_km)}")

        upper = float(self.max_horizon_km or self.horizon_km)
        if upper < float(self.horizon_km):
            raise ValueError(
                f"max_horizon_km={upper} < horizon_km={self.horizon_km}: no se podría "
                "componer la supervivencia hasta H, que es lo que mide `label`."
            )
        self._train_edges_ = self._bin_edges(upper)
        self._score_bins_ = int(np.ceil(float(self.horizon_km) / float(self.bin_km) - 1e-9))

        stacked, hazard, stacked_groups, _ = self._stack(X, duration_km, event, at_risk, groups, entry_km)
        self.n_features_in_ = X.shape[1]
        self.classes_ = np.array([0, 1])
        self.stacking_ = {
            "rows_in": int(len(X)),
            "rows_stacked": int(len(stacked)),
            "n_bins_train": int(len(self._train_edges_)),
            "n_bins_score": self._score_bins_,
            "hazard_rate": float(hazard.mean()),
            "n_hazard_events": int(hazard.sum()),
            "n_rows_not_at_risk": int((~at_risk).sum()),
        }
        logger.info(
            "survival stacking (%s) | %d filas -> %d apiladas en %d bins de %.0f km | "
            "hazard %.4f (%d eventos)",
            self.backend, self.stacking_["rows_in"], self.stacking_["rows_stacked"],
            self.stacking_["n_bins_train"], float(self.bin_km),
            self.stacking_["hazard_rate"], self.stacking_["n_hazard_events"],
        )

        if self.backend == "lightgbm":
            self._fit_lightgbm(stacked, hazard)
        else:
            self._fit_gpboost(stacked, hazard, stacked_groups)
        return self

    def _fit_lightgbm(self, stacked: np.ndarray, hazard: np.ndarray) -> None:
        from lightgbm import LGBMClassifier  # import adentro: dependencia del modelo

        params = _booster_defaults(self.model_params, self.random_state)
        self.booster_ = LGBMClassifier(**params)
        self.booster_.fit(stacked, hazard)

    def _fit_gpboost(self, stacked: np.ndarray, hazard: np.ndarray, groups: np.ndarray | None) -> None:
        try:
            import gpboost as gpb  # import adentro: dependencia opcional
        except ImportError as exc:  # pragma: no cover - depende del entorno
            raise ImportError(
                "GPBoost no está instalado y el YAML pide `backend: gpboost`. "
                "`uv pip install -r requirements-gpboost.txt`, o usá el builder "
                "`survival_stacking` (mismo apilado, sin efecto aleatorio)."
            ) from exc
        if groups is None:
            raise ValueError(
                "`backend: gpboost` necesita el vehículo de cada fila: el target tiene que "
                "traer el campo `group` (src/training/targets.py)."
            )

        params = _booster_defaults(self.model_params, self.random_state)
        n_estimators = int(params.pop("n_estimators"))
        params.pop("class_weight", None)
        gp_params = dict(self.gp_params or {})
        # `optim_params` no es del GPModel: son las opciones de estimación de la
        # varianza (`init_cov_pars`, `estimate_cov_par_index`). Es la vía para fijar
        # σ² en vez de estimarla, que con un hazard tan raro importa (ver docstring).
        optim_params = gp_params.pop("optim_params", None)
        gp_params.setdefault("likelihood", "bernoulli_logit")
        if gp_params["likelihood"] != "bernoulli_logit":
            # `_predict_hazard` aplica el link a mano para no integrar sobre la varianza
            # del intercept (ver ahí). Con otro link el número saldría mal en silencio.
            raise ValueError(
                f"likelihood `{gp_params['likelihood']}` no soportada: el hazard discreto "
                "se predice con el link logit invertido a mano."
            )
        self.gp_model_ = gpb.GPModel(group_data=groups, **gp_params)
        if optim_params:
            self.gp_model_.set_optim_params(params=dict(optim_params))
        data = gpb.Dataset(stacked, label=hazard.astype(float))
        self.booster_ = gpb.train(
            params={**params, "objective": "binary", "verbose": -1},
            train_set=data,
            gp_model=self.gp_model_,
            num_boost_round=n_estimators,
        )
        self.random_effect_variance_ = float(np.ravel(self.gp_model_.get_cov_pars())[0])
        logger.info(
            "efecto aleatorio por vehículo | %d grupos | varianza %.3f (sd %.2f en logit)",
            len(np.unique(groups)), self.random_effect_variance_,
            np.sqrt(self.random_effect_variance_),
        )

    def predict_proba(self, X) -> np.ndarray:  # noqa: D102
        X = np.asarray(X, dtype=float)
        edges = self._train_edges_[: self._score_bins_]
        n_rows, n_bins = len(X), len(edges)
        rows = np.repeat(np.arange(n_rows), n_bins)
        stacked = np.column_stack([X[rows], np.tile(edges, n_rows)])

        hazards = self._predict_hazard(stacked).reshape(n_rows, n_bins)
        # Riesgo acumulado a H: 1 − Π (1 − h_k). El producto en logaritmos evita que se
        # pierda precisión cuando los hazards son chicos, que es el caso normal acá.
        survival = np.exp(np.log1p(-np.clip(hazards, 0.0, 1 - 1e-12)).sum(axis=1))
        risk = 1.0 - survival
        return np.column_stack([1.0 - risk, risk])

    def _predict_hazard(self, stacked: np.ndarray) -> np.ndarray:
        if self.backend == "lightgbm":
            return self.booster_.predict_proba(stacked)[:, 1].astype(float)
        # GPBoost. El vehículo de validación nunca estuvo en train (split agrupado), así
        # que su efecto aleatorio no se puede estimar: se pide con un grupo que no
        # existe, para el que GPBoost devuelve la media a priori (0) en vez de adivinar.
        #
        # Se predice en la escala latente y se aplica el link a mano, en vez de pedir
        # `response_mean`. Los dos ordenan **exactamente igual** (el link es monótono),
        # así que el PR-AUC y la curva de anticipación no cambian; lo que cambia es qué
        # número es. `response_mean` integra sobre la varianza del intercept —que acá se
        # estima enorme, ver la memoria— y devuelve el riesgo *promedio de la flota*,
        # que con esa dispersión da 0,51 y arruina el Brier. Con el efecto aleatorio en
        # su media, el número es el riesgo del **vehículo mediano** con estas features,
        # que es lo que el resto del pipeline trata como probabilidad.
        unseen = np.full(len(stacked), "__unseen__", dtype=object)
        pred = self.booster_.predict(
            data=stacked, group_data_pred=unseen, predict_var=False, pred_latent=True
        )
        latent = np.asarray(pred["fixed_effect"], dtype=float) + np.asarray(
            pred["random_effect_mean"], dtype=float
        )
        return 1.0 / (1.0 + np.exp(-latent))

    def predict(self, X) -> np.ndarray:  # noqa: D102
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    @property
    def extra_feature_names_(self) -> list[str]:
        """Columnas que el modelo agrega a las del panel, en orden. Para las auditorías."""
        return [BIN_FEATURE]

    @property
    def feature_importances_(self) -> np.ndarray:
        """Importancias del booster interno, en el orden `[features de X, bin]`.

        La última entrada es `surv_bin_start_km`, el hazard base. Que pese mucho es
        esperable y sano: el riesgo cambia con los km desde el corte.
        """
        if self.backend == "lightgbm":
            return np.asarray(self.booster_.feature_importances_, dtype=float)
        return np.asarray(self.booster_.feature_importance(), dtype=float)


def _booster_defaults(model_params: dict[str, Any] | None, random_state: int) -> dict[str, Any]:
    """Defaults del LightGBM chico de F3, ajustados al dataset apilado.

    El apilado multiplica las filas por el número de bins en riesgo (≈ 5×) y baja la
    tasa de positivos en la misma proporción, así que `min_child_samples` sube con
    ellas: un hazard estimado con 40 filas apiladas es el mismo aire que 8 filas del
    panel. Sin `class_weight`: el hazard tiene que salir calibrado para que
    `1 − S(H|x)` sea una probabilidad y el Brier signifique algo.
    """
    params = dict(model_params or {})
    params.setdefault("n_estimators", 300)
    params.setdefault("learning_rate", 0.03)
    params.setdefault("num_leaves", 7)
    params.setdefault("min_child_samples", 60)
    params.setdefault("reg_lambda", 5.0)
    params.setdefault("colsample_bytree", 0.7)
    params.setdefault("random_state", random_state)
    params.setdefault("verbose", -1)
    return params


def _unpack_target(y) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """`(duración, evento, en riesgo, grupo, entrada)` desde el array estructurado del target.

    La entrada (`entry_km`) solo la trae `window_km_survival`; sin ella, toda fila entra en 0.
    """
    y = np.asarray(y)
    if y.dtype.names is None or "duration_km" not in y.dtype.names:
        raise TypeError(
            "DiscreteSurvivalStacker espera el target de supervivencia, no `label`. "
            "Poné `target: {name: discrete_survival}` en el YAML del experimento."
        )
    duration = y["duration_km"].astype(float)
    at_risk = (
        y["at_risk"].astype(bool) if "at_risk" in y.dtype.names else duration >= 0
    )
    groups = y["group"].astype(str) if "group" in y.dtype.names else None
    entry = y["entry_km"].astype(float) if "entry_km" in y.dtype.names else None
    return duration, y["event"].astype(int), at_risk, groups, entry


def _ranges(counts: np.ndarray) -> np.ndarray:
    """`[0..c0-1, 0..c1-1, ...]` concatenado, sin bucle en Python."""
    counts = np.asarray(counts, dtype=int)
    total = int(counts.sum())
    if total == 0:
        return np.zeros(0, dtype=int)
    offsets = np.repeat(np.cumsum(counts) - counts, counts)
    return np.arange(total) - offsets
