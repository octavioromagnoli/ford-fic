"""Incidencia a nivel vehículo aprendida con el conjunto externo y aplicada congelada (F5 §3.2, Track B).

La pregunta es *qué auto*, sin *cuándo*: ¿falla en algún momento? Se aprende con los
fallados sin fecha utilizable y sus sanos (`src/data/external.py`,
`scripts/build_external_panel.py`), y se aplica a dev **sin ajustar nada ahí**. Lo que decide
y por qué está en `docs/memoria/f5-preregistro-incidencia-externa.md`.

## El modelo

Una logística con penalización de Firth, sobre las filas elegibles de la fuente:

    logit P(falla) = α_mercado + β · d(x)

- `x` son las cuatro features del índice físico desviadas contra la flota
  (`FleetReferenceNormalizer`), más el uso (`feat_log1p_km_per_day`), imputadas por la
  mediana y estandarizadas con la fuente.
- `d(x)` depende de `design`:
  - `index` (el primario): `[s, z_uso]`, con `s = Σ signo_j · z_j` el índice de pesos
    unitarios de P0. Son dos pendientes, que es lo que admite Riley con ~114 eventos;
  - `separate` (la informativa I3): las cinco z sueltas.
- `α_mercado` es el estrato: un intercepto por mercado de la fuente. El mercado no es
  feature, porque la proporción muestreada de fallados difiere entre mercados.

## Lo que se aplica afuera

El score de una fila es `β · d(x)`, **sin intercepto**: dev no tiene los mercados de la
fuente, y el orden no depende de él. La referencia de flota, la mediana, la escala y β son
todos de la fuente. Por eso `FrozenIncidenceScorer.fit` no aprende nada: carga el ajuste
congelado y verifica que el panel traiga las columnas.

## Referencia de flota sin mercado

El normalizador se ajusta con `levels` del YAML. El primario usa `[[month], []]`: la
mediana por mes de los sanos de la fuente, con los mercados juntos. Así una fila de la
fuente y una de dev se construyen igual. Con `market × month`, dev caería al nivel del mes
de todos modos, porque su mercado no está en la fuente, y se trataría distinto que la
fuente.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from src.models.cure import fit_logistic
from src.training.transformers import MARKET_COLUMN, OUTPUT_PREFIX, FleetReferenceNormalizer

logger = logging.getLogger(__name__)

DESIGNS = ("index", "separate")
INDEX_NAME = "index_s"


@dataclass
class ExternalIncidence:
    """El ajuste congelado: todo lo que hace falta para puntuar una fila nueva."""

    normalizer: FleetReferenceNormalizer
    imputer: SimpleImputer
    scaler: StandardScaler
    covariates: list[str]                 # fleet_* (con signo en `signs`) + la columna de uso
    signs: dict[str, float]               # signo físico de cada fleet_*
    usage_column: str
    design: str
    slope_names: list[str]
    slopes: np.ndarray
    market_levels: list[str]
    intercepts: dict[str, float]
    info: dict[str, Any] = field(default_factory=dict)

    def standardized(self, X: pd.DataFrame) -> pd.DataFrame:
        """Las covariables normalizadas contra la flota, imputadas y estandarizadas con la fuente."""
        # El normalizador rellena con NaN una columna que no está, y la imputación la taparía
        # con la mediana de la fuente sin avisar: la de uso se exige en la entrada.
        if self.usage_column not in X.columns:
            raise KeyError(f"El panel no trae `{self.usage_column}`")
        normalized = self.normalizer.transform(X)
        missing = [c for c in self.covariates if c not in normalized.columns]
        if missing:
            raise KeyError(f"El panel no trae {missing} después del normalizador")
        values = normalized[self.covariates].to_numpy(dtype=float)
        z = self.scaler.transform(self.imputer.transform(values))
        return pd.DataFrame(z, columns=self.covariates, index=X.index)

    def design_matrix(self, z: pd.DataFrame) -> np.ndarray:
        return design_matrix(z, self.design, self.signs, self.usage_column)

    def linear_predictor(self, X: pd.DataFrame) -> np.ndarray:
        """`β · d(x)` sin intercepto: el score que se aplica afuera."""
        return self.design_matrix(self.standardized(X)) @ self.slopes

    def weights(self) -> dict[str, float]:
        """Los pesos por covariable estandarizada: `score = Σ w_j · z_j` (para aplicarlos con otro pipeline)."""
        if self.design == "index":
            a, b = self.slopes
            out = {c: float(a * self.signs[c]) for c in self.covariates if c != self.usage_column}
            out[self.usage_column] = float(b)
            return out
        return {c: float(w) for c, w in zip(self.slope_names, self.slopes)}


def design_matrix(z: pd.DataFrame, design: str, signs: Mapping[str, float], usage_column: str) -> np.ndarray:
    """`[s, z_uso]` (índice) o las z sueltas, en el orden de `slope_names`."""
    if design == "index":
        fleet = [c for c in z.columns if c != usage_column]
        s = z[fleet].to_numpy(dtype=float) @ np.asarray([float(signs[c]) for c in fleet])
        return np.column_stack([s, z[usage_column].to_numpy(dtype=float)])
    if design == "separate":
        return z.to_numpy(dtype=float)
    raise ValueError(f"`design` tiene que ser uno de {DESIGNS}, no `{design}`")


def _strata(markets: pd.Series, levels: Sequence[str]) -> np.ndarray:
    """Un intercepto por mercado: columnas indicadoras sin referencia (el intercepto común no va)."""
    values = markets.astype(str).to_numpy()
    unknown = sorted(set(values) - set(levels))
    if unknown:
        raise ValueError(f"Mercados fuera de los estratos de la fuente: {unknown}")
    return np.column_stack([(values == level).astype(float) for level in levels])


def fit_external_incidence(
    panel: pd.DataFrame,
    *,
    fleet_signs: Mapping[str, float],
    usage_column: str,
    design: str = "index",
    eligible_column: str = "aux_eligible",
    label_column: str = "label",
    market_column: str = MARKET_COLUMN,
    normalizer_params: Mapping[str, Any] | None = None,
    firth: bool = True,
    max_iter: int = 200,
    tol: float = 1e-10,
    eligible: np.ndarray | None = None,
) -> ExternalIncidence:
    """Ajusta el modelo de incidencia con la fuente.

    - La **referencia de flota** usa todos los sanos de `panel`, elegibles o no: es la
      flota típica, no el conjunto de riesgo.
    - La **imputación, la escala y la logística** usan solo las filas elegibles. `eligible`
      permite otra máscara: las sensibilidades de I1.
    """
    if design not in DESIGNS:
        raise ValueError(f"`design` tiene que ser uno de {DESIGNS}, no `{design}`")
    if panel["vehicle_id"].duplicated().any():
        raise ValueError("La fuente tiene que tener una fila por vehículo: la incidencia no tiene hitos")
    features = panel[[c for c in panel.columns if c.startswith(("feat_", "static_"))]]
    y = panel[label_column].to_numpy(dtype=int)
    groups = panel["vehicle_id"].astype(str).to_numpy()
    normalizer = FleetReferenceNormalizer(**dict(normalizer_params or {})).fit(features, (y == 0, groups))
    normalized = normalizer.transform(features)
    fleet = sorted(fleet_signs)
    expected = sorted(c for c in normalized.columns if c.startswith(OUTPUT_PREFIX))
    if fleet != expected:
        raise ValueError(f"`fleet_signs` tiene que traer un signo por feature de flota: {expected}, no {fleet}")
    if not set(np.asarray(list(fleet_signs.values()), dtype=float)) <= {-1.0, 1.0}:
        raise ValueError("Los signos del índice son +1 o −1: se fijan, no se ajustan")
    covariates = [*fleet, usage_column]

    rows = panel[eligible_column].to_numpy(dtype=int) == 1 if eligible is None else np.asarray(eligible, dtype=bool)
    raw = normalized.loc[rows, covariates].to_numpy(dtype=float)
    empty = [c for c, col in zip(covariates, raw.T) if np.isnan(col).all()]
    if empty:
        raise ValueError(f"{empty} no tienen ningún valor entre las filas elegibles")
    imputer = SimpleImputer(strategy="median").fit(raw)
    scaler = StandardScaler().fit(imputer.transform(raw))
    z = pd.DataFrame(scaler.transform(imputer.transform(normalized[covariates].to_numpy(dtype=float))),
                     columns=covariates, index=panel.index)
    d = design_matrix(z, design, fleet_signs, usage_column)
    levels = sorted(panel.loc[rows, market_column].astype(str).unique())
    strata = _strata(panel[market_column], levels)
    X = np.hstack([strata, d])[rows]
    fit = fit_logistic(X, y[rows], firth=firth, max_iter=max_iter, tol=tol)
    if not fit.converged:
        raise RuntimeError("La logística de la fuente no convergió")
    slope_names = [INDEX_NAME, usage_column] if design == "index" else covariates
    slopes = fit.coef[len(levels):]
    intercepts = dict(zip(levels, map(float, fit.coef[:len(levels)])))
    model = ExternalIncidence(
        normalizer=normalizer, imputer=imputer, scaler=scaler, covariates=covariates,
        signs={c: float(fleet_signs[c]) for c in fleet}, usage_column=usage_column, design=design,
        slope_names=slope_names, slopes=np.asarray(slopes, dtype=float), market_levels=levels,
        intercepts=intercepts,
        info={"n_rows": int(rows.sum()), "n_events": int(y[rows].sum()), "n_iter": int(fit.n_iter),
              "objective": float(fit.objective), "firth": bool(firth),
              "reference_levels_used": dict(normalizer.levels_used_),
              "n_reference_vehicles": int(normalizer.n_reference_vehicles_)},
    )
    logger.info("incidencia externa (%s) | %d filas, %d eventos | %s", design, model.info["n_rows"],
                model.info["n_events"], dict(zip(slope_names, np.round(model.slopes, 4))))
    return model


def save_incidence(model: ExternalIncidence, path: str | Path) -> Path:
    import joblib

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def load_incidence(path: str | Path) -> ExternalIncidence:
    import joblib

    model = joblib.load(Path(path))
    if not isinstance(model, ExternalIncidence):
        raise TypeError(f"{path} no es un ajuste de incidencia externa")
    return model


class FrozenIncidenceScorer(ClassifierMixin, BaseEstimator):
    """Aplica el ajuste congelado de la fuente. `fit` **no aprende nada** de los datos que recibe.

    Existe para que el score externo pase por `scripts/train.py` y `scripts/audit_cure.py`
    con los mismos folds, filas y cuentas que P0 y los pisos. La CV recorre los folds, pero el
    score de una fila no depende del fold: es idéntico en las R repeticiones.

    - `artifact`: el `.joblib` que deja `scripts/fit_external_incidence.py`.
    """

    def __init__(self, *, artifact: str | None = None) -> None:
        self.artifact = artifact

    def fit(self, X: pd.DataFrame, y: Any = None) -> "FrozenIncidenceScorer":
        if self.artifact is None:
            raise ValueError("`external_incidence` necesita `artifact`: el ajuste congelado de la fuente")
        if not isinstance(X, pd.DataFrame):
            raise TypeError("`external_incidence` puntúa el panel crudo (con `feat_fm__*`): `preprocessing: none`")
        from src.config import resolve_path

        self.model_ = load_incidence(resolve_path(self.artifact))
        self.model_.standardized(X.head(1))   # falla acá si faltan columnas, no en la predicción
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = X.shape[1]
        return self

    def decision_function(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.linear_predictor(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        # Monótona en el score; no es una probabilidad calibrada para dev (otra prevalencia, sin
        # intercepto de mercado).
        p = expit(self.decision_function(X))
        return np.column_stack([1.0 - p, p])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
