"""Transformadores que se ajustan con el train del fold (F3, cure model, Track B).

Por ahora uno solo: `FleetReferenceNormalizer`, la normalización contra la flota del
punto 2.2 del cure model. Traduce el valor de un vehículo en un mes a **cuánto se
desvía de la flota sana comparable** (mismo mercado, mismo mes), que es lo que el
diagnóstico del 22-09 encontró como rasgo temprano
(`docs/reproducibilidad.md`).

**Por qué no se precalcula en el panel.** La referencia son los sanos del train de cada
fold: es un parámetro ajustado. Calcularla con todo dev usaría las features y las
etiquetas de validación, que es la regla 3 de `CLAUDE.md`. La evidencia de la Fase 1 usó
todos los sanos de dev como referencia; acá se hace bien, y por eso el número puede bajar.

**Entrada.** Las columnas `feat_fm__<feature>__<YYYY-MM>` (valor) y
`feat_fm__w_<feature>__<YYYY-MM>` (peso) que deja `scripts/build_landmark_panel.py`, más
la columna de mercado.

**Salida.** Una columna `fleet_<feature>` por feature —el promedio ponderado por el peso
de (valor − referencia) sobre los meses con dato— y las columnas que no son `feat_fm__*`
tal cual, para que una auditoría pueda sumar covariables sin tocar esto. **El mercado no
pasa**: sirve para armar la celda y nada más (preregistro §3). El modelo nunca ve el mes.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.data.landmark import FM_PREFIX, parse_fleet_column

logger = logging.getLogger(__name__)

MARKET_COLUMN = "static_SalesCountry_cd"
OUTPUT_PREFIX = "fleet_"
# Jerarquía de la referencia, del nivel más específico al último recurso. `market` es la
# columna de mercado y `month` el sufijo de la columna; `global` es la mediana de la
# feature sobre todo el train sano.
DEFAULT_LEVELS: tuple[tuple[str, ...], ...] = (("market", "month"), ("month",), ())
MIN_REF_VEHICLES = 10


def _healthy_and_groups(y: Any, n_rows: int) -> tuple[np.ndarray, np.ndarray]:
    """`(sano, vehículo)` por fila, desde el `y` estructurado del target `cure_window`.

    Acepta también un DataFrame con esas dos columnas, o un par `(healthy, group)`: el
    transformador se puede probar sin levantar el target entero.
    """
    if y is None:
        raise ValueError("`FleetReferenceNormalizer.fit` necesita `y`: la referencia son los sanos del train")
    if isinstance(y, tuple) and len(y) == 2:
        healthy, groups = y
    elif isinstance(y, pd.DataFrame):
        healthy, groups = y["healthy"], y["group"]
    elif getattr(y, "dtype", None) is not None and getattr(y.dtype, "names", None):
        missing = [f for f in ("healthy", "group") if f not in y.dtype.names]
        if missing:
            raise ValueError(f"El `y` estructurado no trae {missing}: los necesita la referencia de flota")
        healthy, groups = y["healthy"], y["group"]
    else:
        raise TypeError("`y` tiene que ser el array estructurado de `cure_window`, un DataFrame o (healthy, group)")
    healthy = np.asarray(healthy).astype(bool)
    groups = np.asarray(groups).astype(str)
    if len(healthy) != n_rows or len(groups) != n_rows:
        raise ValueError(f"`y` tiene {len(healthy)} filas y `X` {n_rows}: no están alineados")
    return healthy, groups


class FleetReferenceNormalizer(TransformerMixin, BaseEstimator):
    """Desvío contra la mediana de los vehículos sanos del train, por mercado × mes.

    Parámetros (todos desde el YAML del experimento):

    - `market_column`: columna con el mercado; se consume y no sale.
    - `min_ref_vehicles`: vehículos sanos mínimos para que una celda sirva como
      referencia. Por debajo se cae al nivel siguiente de `levels`.
    - `levels`: la jerarquía, del nivel más específico al último recurso. El nivel vacío
      es la mediana global de la feature.
    - `drop_columns`: columnas de entrada que no pasan a la salida.

    La referencia es la mediana **entre vehículos**, no entre filas: un vehículo aparece
    en hasta tres hitos y sin deduplicar pesaría el triple. De cada (vehículo, mes) se
    conserva la fila de más peso.
    """

    def __init__(
        self,
        *,
        market_column: str = MARKET_COLUMN,
        min_ref_vehicles: int = MIN_REF_VEHICLES,
        levels: Sequence[Sequence[str]] = DEFAULT_LEVELS,
        drop_columns: Sequence[str] | None = None,
    ) -> None:
        self.market_column = market_column
        self.min_ref_vehicles = min_ref_vehicles
        self.levels = levels
        self.drop_columns = drop_columns

    # -- interno -------------------------------------------------------------------

    def _dropped(self) -> list[str]:
        return list(self.drop_columns) if self.drop_columns is not None else [self.market_column]

    def _long(self, X: pd.DataFrame) -> pd.DataFrame:
        """Una fila por (fila de X, feature, mes) con valor, peso y mercado."""
        parts = []
        market = X[self.market_column].astype(str) if self.market_column in X else pd.Series("", index=X.index)
        for column in X.columns:
            parsed = parse_fleet_column(column)
            if parsed is None or parsed[2]:
                continue
            feature, month, _ = parsed
            weight_column = f"{FM_PREFIX}w_{feature}__{month}"
            if weight_column not in X.columns:
                raise KeyError(f"Falta el peso `{weight_column}` de la celda `{column}`")
            part = pd.DataFrame({
                # Posición, no etiqueta del índice: el transformador no supone nada sobre el índice de X.
                "row": np.arange(len(X)), "feature": feature, "month": month, "market": market.to_numpy(),
                "value": X[column].astype(float).to_numpy(), "weight": X[weight_column].astype(float).to_numpy(),
            })
            parts.append(part)
        if not parts:
            raise ValueError(f"`X` no trae ninguna columna `{FM_PREFIX}*`: ¿es el panel de hitos?")
        long = pd.concat(parts, ignore_index=True)
        return long.loc[long["value"].notna() & long["weight"].gt(0)]

    def _reference_tables(self, cells: pd.DataFrame) -> list[pd.Series]:
        """Una tabla de referencia por nivel, ya filtrada por `min_ref_vehicles`."""
        tables = []
        for level in self.levels:
            keys = ["feature", *level]
            grouped = cells.groupby(keys, observed=True)["value"]
            reference = grouped.median()
            if level:  # el nivel global no se filtra: es el último recurso
                reference = reference.where(grouped.size() >= int(self.min_ref_vehicles)).dropna()
            tables.append(reference.rename("reference"))
        return tables

    def _lookup(self, long: pd.DataFrame) -> pd.DataFrame:
        """Referencia de cada celda y el nivel de la jerarquía que la resolvió."""
        out = long.copy()
        out["reference"] = np.nan
        out["level"] = -1
        for index, (level, table) in enumerate(zip(self.levels, self.reference_)):
            pending = out["reference"].isna()
            if not pending.any() or table.empty:
                continue
            keys = ["feature", *level]
            found = out.loc[pending, keys].merge(table, left_on=keys, right_index=True, how="left")["reference"]
            out.loc[pending, "reference"] = found.to_numpy()
            out.loc[pending & out["reference"].notna(), "level"] = index
        return out

    # -- API de sklearn -------------------------------------------------------------

    def fit(self, X: pd.DataFrame, y: Any = None) -> "FleetReferenceNormalizer":
        if not isinstance(X, pd.DataFrame):
            raise TypeError("`FleetReferenceNormalizer` trabaja sobre un DataFrame (necesita los nombres de columna)")
        healthy, groups = _healthy_and_groups(y, len(X))
        long = self._long(X)
        long["healthy"] = healthy[long["row"].to_numpy()]
        long["vehicle"] = groups[long["row"].to_numpy()]
        cells = long.loc[long["healthy"]]
        # Un vehículo, un voto por mes: se queda la fila (el hito) de más peso.
        cells = (cells.sort_values("weight", kind="stable")
                      .drop_duplicates(["feature", "vehicle", "month"], keep="last"))
        self.reference_ = self._reference_tables(cells)
        self.features_ = sorted(long["feature"].unique())
        self.passthrough_ = [c for c in X.columns
                             if parse_fleet_column(c) is None and c not in self._dropped()]
        self.n_reference_vehicles_ = int(cells["vehicle"].nunique())

        resolved = self._lookup(cells.drop_duplicates(["feature", "market", "month"]))
        self.levels_used_ = {self._level_name(i): int((resolved["level"] == i).sum())
                             for i in range(len(self.levels))}
        self.levels_used_["sin_referencia"] = int((resolved["level"] < 0).sum())
        logger.info("referencia de flota: %d vehículos sanos · celdas por nivel %s",
                    self.n_reference_vehicles_, self.levels_used_)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("`FleetReferenceNormalizer` trabaja sobre un DataFrame")
        long = self._lookup(self._long(X))
        long["deviation"] = (long["value"] - long["reference"]) * long["weight"]
        valid = long.loc[long["reference"].notna()]
        totals = valid.groupby(["row", "feature"], observed=True)[["deviation", "weight"]].sum()
        deviation = (totals["deviation"] / totals["weight"].where(totals["weight"] > 0)).unstack("feature")
        deviation = deviation.reindex(np.arange(len(X)))
        out = pd.DataFrame(index=X.index)
        for feature in self.features_:
            values = deviation[feature].to_numpy() if feature in deviation.columns else np.nan
            out[f"{OUTPUT_PREFIX}{feature}"] = values
        for column in self.passthrough_:
            out[column] = X[column] if column in X.columns else np.nan
        return out

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.asarray([f"{OUTPUT_PREFIX}{f}" for f in self.features_] + list(self.passthrough_), dtype=object)

    def _level_name(self, index: int) -> str:
        level = tuple(self.levels[index])
        return "×".join(level) if level else "global"
