"""Lectura de `trips` / `signals` para un subconjunto de vehículos, ya canonizados.

Es la misma mecánica que `scripts/build_eda_cache.py::read_dev_table`, generalizada
al universo que se pida (dev para el EDA, los 364 del holdout para el panel). El
orden del filtro no es negociable y por eso vive en un solo lugar:

1. **canonizar** `VehicleCode` con `configs/data/vehicle_dedupe.yaml` (los CSV traen
   los 26 códigos clonados sin colapsar);
2. recién ahí **filtrar** por la lista de `vehicle_id`;
3. **deduplicar filas exactas**: las copias de los clones viven en archivos
   distintos, y `signals` trae además un 1,1% de filas repetidas dentro del mismo
   archivo (`docs/reproducibilidad.md`).

Al revés (filtrar antes de canonizar), las filas del código descartado no matchean
ninguna id de la lista y desaparecen sin ruido.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.dedupe import DEFAULT_DEDUPE_CONFIG, VEHICLE_COL, canonical_map
from src.data.loader import DEFAULT_SOURCES_CONFIG, iter_table

logger = logging.getLogger(__name__)

ID_COL = "vehicle_id"
COHORT_COL = "cohort"


def read_table_for_vehicles(
    name: str,
    columns: list[str] | None,
    vehicles: Iterable[str],
    *,
    sources: str | Path = DEFAULT_SOURCES_CONFIG,
    dedupe: str | Path = DEFAULT_DEDUPE_CONFIG,
    chunksize: int = 500_000,
    downcast: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Tabla cruda entera para `vehicles`, con ids canónicos y sin filas repetidas.

    Devuelve `(frame, contadores)`. La columna de vehículo sale renombrada a
    `vehicle_id`; `cohort` no viaja (la etiqueta se resuelve en la tabla estática).

    `columns=None` lee la fila completa, y es lo que corresponde cuando después se
    deduplica: dos mensajes distintos en el mismo odómetro son la misma fila si no
    se leyó el timestamp. Con un subconjunto de columnas el conteo de "duplicados"
    infla y se pierden filas legítimas.
    """
    wanted = {str(v) for v in vehicles}
    mapping = canonical_map(dedupe)
    clone_codes = set(mapping) | set(mapping.values())

    parts: list[pd.DataFrame] = []
    n_raw_total = 0
    n_subset_raw = 0
    for chunk in iter_table(name, sources, chunksize=chunksize, columns=columns):
        n_raw_total += len(chunk)
        chunk[VEHICLE_COL] = chunk[VEHICLE_COL].replace(mapping)      # 1) canonizar
        chunk = chunk[chunk[VEHICLE_COL].isin(wanted)]                 # 2) filtrar
        if not len(chunk):
            continue
        n_subset_raw += len(chunk)
        chunk = chunk.drop(columns=[COHORT_COL], errors="ignore")
        parts.append(_downcast(chunk) if downcast else chunk)

    if not parts:
        raise ValueError(f"Tabla `{name}`: ninguna fila para los {len(wanted)} vehículos pedidos")
    frame = pd.concat(parts, ignore_index=True)
    del parts

    is_clone = frame[VEHICLE_COL].isin(clone_codes)
    n_clone_rows = int(is_clone.sum())
    n_clone_unique = int(len(frame.loc[is_clone].drop_duplicates())) if n_clone_rows else 0
    n_unique = int(len(frame.drop_duplicates()))
    counters = {
        "n_raw_total": n_raw_total,
        "n_subset_raw": n_subset_raw,
        "dup_clones": n_clone_rows - n_clone_unique,
        "dup_otros": (n_subset_raw - n_unique) - (n_clone_rows - n_clone_unique),
    }
    frame = frame.drop_duplicates(ignore_index=True)                   # 3) deduplicar
    counters["n_subset_dedup"] = int(len(frame))
    frame = frame.rename(columns={VEHICLE_COL: ID_COL})
    missing = wanted - set(frame[ID_COL].unique())
    counters["n_vehicles_missing"] = len(missing)
    logger.info(
        "`%s`: %d filas crudas -> %d deduplicadas (%d clones, %d otros) · %d vehículos, %d sin filas",
        name, n_subset_raw, len(frame), counters["dup_clones"], counters["dup_otros"],
        frame[ID_COL].nunique(), len(missing),
    )
    return frame, counters


def _downcast(frame: pd.DataFrame) -> pd.DataFrame:
    """float64 -> float32 para que 13M de filas entren en RAM. Las fechas no se tocan."""
    out = frame.copy()
    for column in out.columns:
        if out[column].dtype == "float64":
            out[column] = out[column].astype("float32")
    return out
