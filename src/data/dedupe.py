"""Colapso de vehículos duplicados. La lista de clones vive en un YAML, no acá.

Los archivos crudos cuentan 13 vehículos cuatro veces: cada uno bajo dos
`VehicleCode` consecutivos, y cada código en las dos cohortes de muestreo. Si eso
llega al panel, el split agrupado por vehículo se cumple en el papel (ningún
`vehicle_id` repetido entre train y validación) y se viola en los hechos, porque
el gemelo del vehículo de train está en validación con los mismos viajes.

Se aplica una sola vez, al construir el panel, antes de cualquier split.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import load_config

logger = logging.getLogger(__name__)

DEFAULT_DEDUPE_CONFIG = "configs/data/vehicle_dedupe.yaml"
VEHICLE_COL = "VehicleCode"
EVENT_COHORT = "failed"


def load_clone_groups(config_path: str | Path = DEFAULT_DEDUPE_CONFIG) -> list[list[str]]:
    """Grupos de códigos que son el mismo vehículo, tal como los congeló F1."""
    cfg = load_config(config_path)
    groups = cfg.get("clone_groups") or []
    return [list(group) for group in groups if len(group) > 1]


def canonical_map(
    config_path: str | Path = DEFAULT_DEDUPE_CONFIG,
) -> dict[str, str]:
    """`{código descartado: código canónico}`. El canónico es el primero del grupo."""
    mapping: dict[str, str] = {}
    for group in load_clone_groups(config_path):
        canonical = group[0]
        for code in group[1:]:
            mapping[code] = canonical
    return mapping


def apply_canonical_ids(
    df: pd.DataFrame,
    mapping: dict[str, str],
    *,
    id_col: str = VEHICLE_COL,
    cohort_col: str | None = "cohort",
) -> pd.DataFrame:
    """Remapea los códigos clonados y colapsa las filas que quedan repetidas.

    Las filas del clon son idénticas a las del canónico salvo por el código y por
    la cohorte de la que vienen, así que alcanza con deduplicar ignorando la
    columna de cohorte. La cohorte se resuelve aparte: si el vehículo tiene alguna
    fila `failed`, el evento existe (`not_failed` es ausencia de registro, no
    evidencia de que esté sano).
    """
    out = df.copy()
    out[id_col] = out[id_col].replace(mapping)

    if cohort_col is not None and cohort_col in out.columns:
        tiene_evento = (
            out.assign(_ev=out[cohort_col].eq(EVENT_COHORT)).groupby(id_col)["_ev"].transform("max")
        )
        out[cohort_col] = np.where(tiene_evento, EVENT_COHORT, "not_failed")

    subset = [c for c in out.columns if c != cohort_col]
    before = len(out)
    out = out.drop_duplicates(subset=subset, ignore_index=True)
    if before != len(out):
        logger.info("Dedupe: %d filas colapsadas (%d -> %d)", before - len(out), before, len(out))
    return out


def dedupe_vehicles(
    vehicles: pd.DataFrame,
    *,
    config_path: str | Path = DEFAULT_DEDUPE_CONFIG,
    id_col: str = VEHICLE_COL,
    cohort_col: str = "cohort",
    event_day_col: str = "IdentificationDate",
) -> pd.DataFrame:
    """Una fila por vehículo real, con la etiqueta resuelta.

    Además del colapso de clones, la tabla estática trae el mismo código en las dos
    cohortes: la fila `failed` es la que tiene `IdentificationDate`, así que se
    queda esa.
    """
    out = vehicles.copy()
    out[id_col] = out[id_col].replace(canonical_map(config_path))

    if event_day_col in out.columns:
        # Ordena dejando primero la fila con el día del evento: es la informativa.
        out = out.sort_values(event_day_col, na_position="last")
    out = out.drop_duplicates(subset=[id_col], keep="first", ignore_index=True)

    if event_day_col in out.columns and cohort_col in out.columns:
        out[cohort_col] = out[event_day_col].notna().map({True: EVENT_COHORT, False: "not_failed"})
    return out.sort_values(id_col, ignore_index=True)
