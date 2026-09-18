"""Derivadas a nivel señal (`signals`): una columna booleana por nivel de `Message`.

`signals` se agrega por ventana sobre `OdometerValue` (0,009% nulo en el universo
del estudio: las filas sin odómetro se descartan y se cuentan). No se alinea contra
`trips`: comparten el eje de km y con eso alcanza, el desfase entre los dos
odómetros es de −25 km de mediana.

Dos columnas que se derivan pero **no entran al modelo** (`aux_` en el panel):

- `regen_marker`: el marcador literal "Regeneration". Se corta el 25-05-2026 para
  toda la flota (0 marcadores en jun-sep 2026 contra ~2.000 caídas de nivel por
  mes en `trips`). Como los eventos caen entre sep-2025 y mar-2026 y la exposición
  sana se concentra en 2026, una tasa de marcadores por km mide el calendario y el
  calendario mide la etiqueta. Se conserva solo para auditar.
- `Acumulation`: es la misma variable que `trips.AirRegenerationEnd` (99,8% de
  coincidencia), y la familia B se construye desde `trips` porque tiene el
  odómetro completo y el grano de viaje. No se emite dos veces.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

ID_COL = "vehicle_id"

# Vocabulario de `Message` (el mismo de `trips.AirFilter*`, errata incluida).
MESSAGE_SLUGS: dict[str, str] = {
    "Air Filter Normal Operation": "normal",
    "Air Filter Full": "full",
    "Air Filter Overloaded": "overloaded",
    "Air Filter Over Limit": "over_limit",
    "Air Filter At Limit": "at_limit",
    "Cleaning Automatically Air Filter": "cleaning_auto",
    "Stopped Cleaning Automatically Air Filter": "stopped_auto",
    "Cleanning Manually Air Filter": "cleaning_manual",
    "Cleaning Manually Air Filter": "cleaning_manual",
    "Stopped Cleanning Manually Air Filter": "stopped_manual",
    "Stopped Cleaning Manually Air Filter": "stopped_manual",
}
SLUGS = ("normal", "full", "overloaded", "over_limit", "at_limit", "cleaning_auto",
         "stopped_auto", "cleaning_manual", "stopped_manual")
REQUIRED = [ID_COL, "OdometerValue", "Message", "Regenerations"]


def derive_signal_columns(signals: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """`msg_<slug>` booleanas + `msg_abnormal` + `regen_marker`; descarta filas sin odómetro."""
    missing = [c for c in REQUIRED if c not in signals.columns]
    if missing:
        raise KeyError(f"`signals` no tiene las columnas requeridas: {missing}")
    out = signals.copy()
    n_in = len(out)
    drop = out["OdometerValue"].isna()
    out = out.loc[~drop].copy()
    counters = {"n_in": n_in, "dropped_null_odometer": int(drop.sum()), "n_out": int(len(out))}

    slug = out["Message"].map(MESSAGE_SLUGS)
    unknown = out["Message"].notna() & slug.isna()
    if unknown.any():
        logger.warning("signals: %d filas con un nivel de Message fuera del vocabulario: %s",
                       int(unknown.sum()), sorted(out.loc[unknown, "Message"].unique())[:5])
    for name in SLUGS:
        out[f"msg_{name}"] = slug.eq(name)
    out["msg_abnormal"] = slug.notna() & slug.ne("normal")
    out["msg_any"] = out["Message"].notna()
    out["regen_marker"] = out["Regenerations"].notna() if out["Regenerations"].dtype != bool else out["Regenerations"]
    out["OdometerValue"] = out["OdometerValue"].astype("float64")
    out = out.sort_values([ID_COL, "OdometerValue"], ignore_index=True)
    logger.info("signals: %d filas -> %d (sin odómetro %d)", n_in, len(out), counters["dropped_null_odometer"])
    return out, counters
