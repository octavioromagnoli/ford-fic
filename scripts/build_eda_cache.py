#!/usr/bin/env python
"""Cache del EDA exhaustivo **sobre dev**: una pasada por los crudos, muchos consumidores.

    python scripts/build_eda_cache.py --config configs/data/eda_cache.yaml
    python scripts/build_eda_cache.py --force        # regenera aunque exista

`trips` son 2,53M de filas y `signals` 10,6M (1,2 GB). Rescanearlos en cada
sección del notebook y en cada recarga del dashboard no es viable, y —lo que
importa más— sería un **segundo lugar donde se decide qué vehículo entra**. Acá se
decide una sola vez: el notebook (`notebooks/eda-exhaustivo-dev.ipynb`) y el
dashboard (`scripts/dashboard_eda.py`) leen los parquet que deja este script y no
vuelven a tocar `data/raw/`.

## El orden del filtro no es negociable

`test_split.json` guarda `vehicle_id` **ya canónicos** (post-dedupe de los 13
clones), pero los CSV crudos todavía traen los 26 `VehicleCode` sin canonizar. Si
se filtra por `dev_vehicles` antes de canonizar, las filas del código descartado
(p. ej. `VEH_0345`) no matchean ninguna id de la lista —que solo tiene
`VEH_0344`— y desaparecen sin ruido. No es leakage, pero sí un error de conteo que
no se ve en ninguna parte. Por eso, siempre:

    1. canonizar  (`src/data/dedupe.canonical_map`)
    2. filtrar a dev  (`dev_vehicles` del holdout congelado)

`src/data/join.iter_enriched_table` ya canoniza antes de procesar; acá se reusa esa
misma mecánica y se le agrega el filtro a dev encima.

## El universo es dev, y dev ya no es "todo lo que no es test"

`test_split.json` congela tres listas, no dos: `dev_vehicles` (290), `test_vehicles`
(74) y `excluded_vehicles` (717, fuera del estudio — ver `src/data/usable.py`). El
EDA corre sobre la primera y nada más. Por eso la guarda es
`assert_dev_only()`, que verifica **pertenencia a dev**, y no "ausencia de test":
con un universo recortado, un vehículo excluido no es test pero tampoco es dev, y
el complemento lo dejaría entrar sin ruido.

## Lo que este script NO hace

No construye el panel de F2, no define W/G/H/Δ y no mira una sola fila de test:
los 74 vehículos del holdout quedan afuera de **todos** los agregados, y cada
tabla materializada pasa por `assert_dev_only()` antes de escribirse.

## Qué deja en `experiments/eda/dev/` (gitignored)

| Archivo | Grano |
|---|---|
| `vehicles_dev.parquet` | un vehículo (290) · estáticas, etiqueta, cobertura, odómetro del evento |
| `trips_agg_dev.parquet` | un vehículo · agregados de `trips` (uso, térmica, postratamiento) |
| `signals_agg_dev.parquet` | un vehículo · agregados de `signals` (severidad, regeneraciones) |
| `severity_scopes_dev.parquet` | vehículo × alcance (`full`, `p70`, `p80`) · severidad truncando la cola |
| `message_counts_dev.parquet` | vehículo × alcance × nivel de `Message` |
| `airfilter_counts_dev.parquet` | vehículo × posición (start/end) × nivel de `AirFilter` |
| `trip_quantiles_dev.parquet` | métrica × cohorte · cuantiles EXACTOS a nivel viaje |
| `trip_hist_dev.parquet` | métrica × cohorte · histograma EXACTO a nivel viaje |
| `trips_daily_dev.parquet` | día calendario × cohorte · viajes, vehículos activos, km |
| `odo_profile_dev.parquet` | día desde producción × cohorte · odómetro mediano de la flota |
| `trips_sample_dev.parquet` | viaje (muestra) · para scatter y detalle |
| `signals_sample_dev.parquet` | señal (muestra) · para scatter y detalle |
| `probe_trips_dev.parquet` / `probe_signals_dev.parquet` | historial completo de unos pocos vehículos |
| `meta.json` | conteos, duplicados, offset del anclaje, semilla y fecha |
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.config import ensure_dir, load_config, resolve_path, set_seed  # noqa: E402
from src.data.dedupe import canonical_map  # noqa: E402
from src.data.loader import iter_table  # noqa: E402
from src.eval.splits import load_test_split  # noqa: E402

logger = logging.getLogger("build_eda_cache")

DEFAULT_CONFIG = "configs/data/eda_cache.yaml"
VEHICLE_COL = "VehicleCode"
ID_COL = "vehicle_id"
COHORT_COL = "cohort"

# --------------------------------------------------------------------------------------
# Identidad visual compartida entre el notebook y el dashboard.
#
# El azul y el rojo son los de `src/eval/plots.py` (mismo repo, misma identidad). Los
# pasos categóricos salen de la paleta validada del skill `dataviz` re-anclada al azul
# Ford: `#2c52a8` es el escalón del azul Ford dentro de la banda de luminosidad que el
# validador exige (el `#1c3f94` de la marca queda por debajo y no se usa como marca de
# datos, solo como tinta de títulos).
#
#   node scripts/validate_palette.js "#2c52a8,#eb6834,#1baf7a,#eda100,#e87ba4,#008300,#4a3aa7,#c0392b" --mode light --surface "#ffffff"
#   -> todos los chequeos pasan (peor par adyacente CVD ΔE 9,1 · visión normal 19,6)
# --------------------------------------------------------------------------------------
FORD_BLUE = "#1c3f94"          # tinta de marca (títulos, no marcas de datos)
ALERT_RED = "#c0392b"

BLUE = "#2c52a8"               # escalón del azul Ford válido como marca de datos
CATEGORICAL = ("#2c52a8", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#c0392b")
# Dos series y solo dos en casi todo el EDA: la cohorte ES la etiqueta.
COHORT_COLORS = {0: BLUE, 1: ALERT_RED}
COHORT_LABELS = {0: "sin evento", 1: "con evento"}
SEQUENTIAL = ("#cfe0f7", "#a9c4ef", "#7ba0e2", "#5882dd", "#426ac3", "#2c52a8", "#1d4095")
ORDINAL = ("#7ba8ff", "#648feb", "#4770c9", "#3158af", "#1d4095")   # severidad creciente
INK = {"primary": "#0b0b0b", "secondary": "#52514e", "muted": "#898781", "grid": "#e1e0d9"}

# Modo oscuro: los mismos tonos re-escalonados para la superficie oscura de Streamlit
# (#0e1117), no un volteo automático. También validados:
#   node scripts/validate_palette.js "#5882dd,#d95926,#199e70,#c98500,#d55181,#008300,#9085e9,#e06c5d" --mode dark --surface "#0e1117"
THEME = {
    "light": {
        "surface": "#ffffff",
        "cohort": {0: BLUE, 1: ALERT_RED},
        "categorical": CATEGORICAL,
        "ordinal": ORDINAL,
        "ink": INK,
    },
    "dark": {
        "surface": "#0e1117",
        "cohort": {0: "#5882dd", 1: "#e06c5d"},
        "categorical": ("#5882dd", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e06c5d"),
        "ordinal": ("#93c2ff", "#7ba8ff", "#5882dd", "#426ac3", "#2c52a8"),
        "ink": {"primary": "#ffffff", "secondary": "#c3c2b7", "muted": "#898781", "grid": "#2c2c2a"},
    },
}

# Vocabulario de `signals.Message`, que resulta ser el mismo de `trips.AirFilter*`.
MESSAGE_SLUGS = {
    "Air Filter Normal Operation": "normal",
    "Air Filter Full": "full",
    "Air Filter Overloaded": "overloaded",
    "Air Filter Over Limit": "over_limit",
    "Air Filter At Limit": "at_limit",
    "Cleaning Automatically Air Filter": "cleaning_auto",
    "Stopped Cleaning Automatically Air Filter": "stopped_auto",
    "Stopped Cleanning Manually Air Filter": "stopped_manual",
    "Cleaning Manually Air Filter": "cleaning_manual",
    "Stopped Cleaning Manually Air Filter": "stopped_manual_alt",
}
NORMAL_MESSAGE = "Air Filter Normal Operation"

TRIP_COLUMNS = [
    VEHICLE_COL, "TripDatetimeStart", "TripDatetimeEnd", "OdometerTripStart", "OdometerTripEnd",
    "KilometerPerHour", "FuelLvlStartPc", "FuelLvlEndPc", "EngineOilLifePCStart",
    "EngineOilLifePCEnd", "EngineTemperatureMin", "EngineTemperatureMax", "EngineTemperatureAvg",
    "AirFilterStart", "AirFilterEnd", "AirRegenerationStart", "AirRegenerationEnd",
    "CoolantTemperatureStart", "CoolantTemperatureEnd", "AirTemperatureStart", "AirTemperatureEnd",
    "AirTemperatureMin", "AirTemperatureMax", "AirTemperatureAvg", "TripNumber",
]
SIGNAL_COLUMNS = [
    VEHICLE_COL, "eventTimestamp", "OdometerValue", "Acumulation", "Message", "Regenerations",
    "DistanceBetweenRegenerations",
]

# Métricas a nivel viaje que se resumen con cuantiles e histograma exactos.
TRIP_METRICS = {
    "trip_km": "Distancia por viaje [km]",
    "trip_duration_min": "Duración del viaje [min]",
    "KilometerPerHour": "Velocidad media [km/h]",
    "EngineTemperatureAvg": "Temperatura media de motor [°C]",
    "EngineTemperatureMax": "Temperatura máxima de motor [°C]",
    "engine_temp_amplitude": "Amplitud térmica del viaje [°C]",
    "CoolantTemperatureEnd": "Refrigerante al final [°C]",
    "AirTemperatureAvg": "Temperatura ambiente media [°C]",
    "EngineOilLifePCStart": "Vida útil del aceite [%]",
    "FuelLvlStartPc": "Nivel de combustible al inicio [%]",
    "AirRegenerationEnd": "AirRegenerationEnd [0-95]",
    "air_regen_delta": "Δ AirRegeneration en el viaje",
}
HIST_CLIP = {"trip_km": (0, 200), "trip_duration_min": (0, 240), "KilometerPerHour": (0, 140)}


# ======================================================================================
# Carga dev-only
# ======================================================================================
def dev_universe(cfg: dict[str, Any]) -> tuple[set[str], set[str], dict[str, Any]]:
    """`(dev_vehicles, test_vehicles, split)` del holdout congelado."""
    split = load_test_split(cfg["test_split"])
    return set(split["dev_vehicles"]), set(split["test_vehicles"]), split


def assert_dev_only(frame: pd.DataFrame, dev_vehicles: set[str], *, name: str, id_col: str = ID_COL) -> None:
    """Guarda ejecutable: falla si la tabla trae un vehículo que no es de dev.

    Va después de armar **cada** tabla, no una vez al principio: la restricción
    dev-only tiene que ser verificable en el artefacto, no una promesa del código.

    Chequea pertenencia, no ausencia de test. Con el universo recortado hay tres
    poblaciones —dev, test y excluidos— y "no es test" ya no implica "es dev": un
    vehículo excluido pasaría el chequeo viejo y entraría a los agregados igual.
    """
    if id_col not in frame.columns:
        raise KeyError(f"`{name}` no tiene la columna `{id_col}`: no se puede verificar el holdout")
    intrusos = set(frame[id_col].astype(str).unique()) - dev_vehicles
    if intrusos:
        raise AssertionError(
            f"`{name}`: {len(intrusos)} vehículo(s) que no son de dev "
            f"(ej.: {sorted(intrusos)[:3]}). Son de test o están fuera del universo "
            "del estudio; ninguno de los dos entra al EDA."
        )


def read_dev_table(
    name: str,
    columns: list[str],
    *,
    dev_vehicles: set[str],
    cfg: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Lee una tabla cruda entera **solo para los vehículos de dev**, ya deduplicada.

    Canoniza primero y filtra después (ver el docstring del módulo). Devuelve la
    tabla en memoria —cabe: dev son ~1,9M viajes y ~8,2M señales downcasteadas— y
    los contadores de duplicados, separando los que explica el colapso de clones de
    los que son filas exactamente repetidas dentro de un archivo.
    """
    mapping = canonical_map(cfg["dedupe"])
    clone_codes = set(mapping) | set(mapping.values())
    chunksize = int(cfg.get("scan", {}).get("chunksize", 500_000))

    parts: list[pd.DataFrame] = []
    n_raw_total = 0
    n_dev_raw = 0
    for chunk in iter_table(name, cfg["sources"], chunksize=chunksize, columns=columns):
        n_raw_total += len(chunk)
        chunk[VEHICLE_COL] = chunk[VEHICLE_COL].replace(mapping)     # 1) canonizar
        chunk = chunk[chunk[VEHICLE_COL].isin(dev_vehicles)]         # 2) recién ahí, dev
        if not len(chunk):
            continue
        n_dev_raw += len(chunk)
        parts.append(_downcast(chunk.drop(columns=[COHORT_COL], errors="ignore")))

    if not parts:
        raise ValueError(f"Tabla `{name}`: ninguna fila de dev. ¿El holdout corresponde a estos datos?")
    frame = pd.concat(parts, ignore_index=True)
    del parts

    # Duplicados, con la misma metodología que `src/data/join.py::duplicate_report`:
    # los de los códigos clonados por un lado (lo que colapsa el dedupe) y el resto
    # por el otro (filas exactamente repetidas dentro de un mismo archivo).
    is_clone = frame[VEHICLE_COL].isin(clone_codes)
    n_clone_rows = int(is_clone.sum())
    n_clone_unique = int(len(frame.loc[is_clone].drop_duplicates())) if n_clone_rows else 0
    n_total_unique = int(len(frame.drop_duplicates()))
    counters = {
        "n_raw_todas_las_cohortes": n_raw_total,
        "n_dev_crudas": n_dev_raw,
        "dup_clones": n_clone_rows - n_clone_unique,
        "dup_otros": (n_dev_raw - n_total_unique) - (n_clone_rows - n_clone_unique),
    }

    frame = frame.drop_duplicates(ignore_index=True)
    counters["n_dev_deduplicadas"] = len(frame)
    frame = frame.rename(columns={VEHICLE_COL: ID_COL})
    logger.info(
        "`%s` dev: %d filas crudas -> %d deduplicadas (%d clones, %d otros)",
        name, n_dev_raw, len(frame), counters["dup_clones"], counters["dup_otros"],
    )
    return frame, counters


def _downcast(frame: pd.DataFrame) -> pd.DataFrame:
    """Baja float64 a float32 y los niveles categóricos a `category`: 13M de filas en RAM."""
    out = frame.copy()
    for column in out.columns:
        kind = out[column].dtype
        if kind == "float64":
            out[column] = out[column].astype("float32")
        elif column in {"AirFilterStart", "AirFilterEnd", "Message"}:
            out[column] = out[column].astype("object")
        elif column == "Regenerations":
            out[column] = out[column].notna()
    return out


# ======================================================================================
# Derivadas a nivel fila
# ======================================================================================
def with_trip_derived(trips: pd.DataFrame) -> pd.DataFrame:
    """Columnas a nivel viaje que el crudo no trae y todo el EDA usa."""
    out = trips
    out["trip_km"] = out["OdometerTripEnd"] - out["OdometerTripStart"]
    out["trip_duration_min"] = (
        out["TripDatetimeEnd"] - out["TripDatetimeStart"]
    ).dt.total_seconds() / 60.0
    out["engine_temp_amplitude"] = out["EngineTemperatureMax"] - out["EngineTemperatureMin"]
    out["air_regen_delta"] = out["AirRegenerationEnd"] - out["AirRegenerationStart"]
    out["oil_life_drop"] = out["EngineOilLifePCStart"] - out["EngineOilLifePCEnd"]
    return out


# ======================================================================================
# Agregados por vehículo
# ======================================================================================
def _slope_per_1000km(frame: pd.DataFrame, x: str, y: str) -> pd.Series:
    """Pendiente OLS de `y` contra `x` por vehículo, expresada por cada 1.000 km.

    Vectorizado con sumas por grupo (n·Sxy − Sx·Sy) / (n·Sxx − Sx²) en vez de un
    `apply` por vehículo: son cientos de grupos y millones de filas.
    """
    data = frame[[ID_COL, x, y]].dropna()
    data = data.assign(_xy=data[x] * data[y], _xx=data[x] * data[x])
    grouped = data.groupby(ID_COL, observed=True)
    n = grouped.size()
    sx, sy = grouped[x].sum(), grouped[y].sum()
    sxy, sxx = grouped["_xy"].sum(), grouped["_xx"].sum()
    denom = n * sxx - sx * sx
    slope = (n * sxy - sx * sy) / denom.where(denom != 0)
    return (slope * 1000.0).rename(f"{y}_slope_per_1000km")


def trip_aggregates(trips: pd.DataFrame, eda: dict[str, Any]) -> pd.DataFrame:
    """Un vehículo por fila: uso, régimen térmico, severidad y postratamiento de `trips`."""
    short = float(eda["short_trip_km"])
    very_short = float(eda["very_short_trip_km"])
    regime = float(eda["regime_temp_c"])
    urban = float(eda["urban_speed_kmh"])
    saturation = float(eda["saturation_level"])

    work = trips.assign(
        _short=(trips["trip_km"] < short),
        _very_short=(trips["trip_km"] < very_short),
        _below_regime=(trips["EngineTemperatureMax"] < regime),
        _urban=(trips["KilometerPerHour"] < urban),
        _km_positive=trips["trip_km"].clip(lower=0),
        _km_negative=(trips["trip_km"] < 0),
        _speed_null=trips["KilometerPerHour"].isna(),
        _regen_saturated=(trips["AirRegenerationEnd"] >= saturation),
        _regen_positive=(trips["air_regen_delta"] > 0),
        _regen_drop=(trips["air_regen_delta"] < -5),
        _refuel=(trips["FuelLvlEndPc"] > trips["FuelLvlStartPc"]),
        _filter_abnormal=(trips["AirFilterEnd"].notna() & trips["AirFilterEnd"].ne(NORMAL_MESSAGE)),
    )
    grouped = work.groupby(ID_COL, observed=True)
    agg = grouped.agg(
        n_trips=("trip_km", "size"),
        trip_km_sum=("_km_positive", "sum"),
        trip_odo_min=("OdometerTripStart", "min"),
        trip_odo_max=("OdometerTripEnd", "max"),
        trip_date_min=("TripDatetimeStart", "min"),
        trip_date_max=("TripDatetimeStart", "max"),
        n_km_negative=("_km_negative", "sum"),
        trip_km_mean=("trip_km", "mean"),
        trip_km_median=("trip_km", "median"),
        trip_km_p25=("trip_km", lambda s: s.quantile(0.25)),
        trip_km_p75=("trip_km", lambda s: s.quantile(0.75)),
        short_trip_frac=("_short", "mean"),
        very_short_trip_frac=("_very_short", "mean"),
        trip_duration_min_median=("trip_duration_min", "median"),
        speed_mean=("KilometerPerHour", "mean"),
        speed_median=("KilometerPerHour", "median"),
        speed_null_frac=("_speed_null", "mean"),
        urban_trip_frac=("_urban", "mean"),
        engine_temp_avg_median=("EngineTemperatureAvg", "median"),
        engine_temp_max_median=("EngineTemperatureMax", "median"),
        engine_temp_amplitude_mean=("engine_temp_amplitude", "mean"),
        below_regime_frac=("_below_regime", "mean"),
        coolant_end_mean=("CoolantTemperatureEnd", "mean"),
        air_temp_avg_mean=("AirTemperatureAvg", "mean"),
        air_temp_min_mean=("AirTemperatureMin", "mean"),
        oil_life_start_mean=("EngineOilLifePCStart", "mean"),
        oil_life_drop_sum=("oil_life_drop", "sum"),
        fuel_start_mean=("FuelLvlStartPc", "mean"),
        refuel_frac=("_refuel", "mean"),
        air_regen_end_mean=("AirRegenerationEnd", "mean"),
        air_regen_end_median=("AirRegenerationEnd", "median"),
        air_regen_end_max=("AirRegenerationEnd", "max"),
        air_regen_saturated_frac=("_regen_saturated", "mean"),
        air_regen_positive_delta_frac=("_regen_positive", "mean"),
        n_air_regen_drops=("_regen_drop", "sum"),
        air_filter_abnormal_frac=("_filter_abnormal", "mean"),
    )

    # Tiempo entre viajes: necesita el orden temporal dentro del vehículo.
    ordered = work[[ID_COL, "TripDatetimeStart"]].dropna().sort_values([ID_COL, "TripDatetimeStart"])
    gap_h = ordered.groupby(ID_COL, observed=True)["TripDatetimeStart"].diff().dt.total_seconds() / 3600.0
    agg["hours_between_trips_median"] = (
        gap_h.groupby(ordered[ID_COL], observed=True).median().reindex(agg.index)
    )

    agg = agg.join(_slope_per_1000km(work, "OdometerTripEnd", "AirRegenerationEnd"))
    agg = agg.rename(columns={"AirRegenerationEnd_slope_per_1000km": "air_regen_slope_per_1000km"})

    span_days = (agg["trip_date_max"] - agg["trip_date_min"]).dt.total_seconds() / 86400.0
    agg["span_days"] = span_days
    agg["km_per_day"] = agg["trip_km_sum"] / span_days.where(span_days > 0)
    agg["trips_per_day"] = agg["n_trips"] / span_days.where(span_days > 0)
    km_k = (agg["trip_km_sum"] / 1000.0).where(agg["trip_km_sum"] > 0)
    agg["oil_life_drop_per_1000km"] = agg.pop("oil_life_drop_sum") / km_k
    agg["air_regen_drops_per_1000km"] = agg.pop("n_air_regen_drops") / km_k
    return agg.reset_index()


def signal_aggregates(signals: pd.DataFrame, km_by_vehicle: pd.Series, eda: dict[str, Any]) -> pd.DataFrame:
    """Un vehículo por fila: severidad, `Acumulation` y ciclo de regeneración de `signals`.

    Las tasas se normalizan por los km de `trips`, no por el odómetro de `signals`:
    `trips` manda como eje (`docs/memoria/f1-calidad-odometro.md`).
    """
    saturation = float(eda["saturation_level"])
    work = signals.assign(
        _odo_null=signals["OdometerValue"].isna(),
        _acu_saturated=(signals["Acumulation"] >= saturation),
        _acu_high=(signals["Acumulation"] >= 50),
    )
    grouped = work.groupby(ID_COL, observed=True)
    agg = grouped.agg(
        n_signals=("Acumulation", "size"),
        signal_odo_null_frac=("_odo_null", "mean"),
        signal_odo_max=("OdometerValue", "max"),
        signal_date_min=("eventTimestamp", "min"),
        signal_date_max=("eventTimestamp", "max"),
        acumulation_mean=("Acumulation", "mean"),
        acumulation_median=("Acumulation", "median"),
        acumulation_max=("Acumulation", "max"),
        acumulation_saturated_frac=("_acu_saturated", "mean"),
        acumulation_high_frac=("_acu_high", "mean"),
        n_regenerations=("Regenerations", "sum"),
        dist_between_regen_mean=("DistanceBetweenRegenerations", "mean"),
        dist_between_regen_median=("DistanceBetweenRegenerations", "median"),
    )
    agg = agg.join(_slope_per_1000km(work, "OdometerValue", "DistanceBetweenRegenerations"))
    agg = agg.rename(
        columns={"DistanceBetweenRegenerations_slope_per_1000km": "dist_between_regen_slope_per_1000km"}
    )
    agg = agg.join(_slope_per_1000km(work, "OdometerValue", "Acumulation"))
    agg = agg.rename(columns={"Acumulation_slope_per_1000km": "acumulation_slope_per_1000km"})

    km_k = (km_by_vehicle.reindex(agg.index) / 1000.0).where(km_by_vehicle.reindex(agg.index) > 0)
    agg["signals_per_1000km"] = agg["n_signals"] / km_k
    agg["regen_per_1000km"] = agg["n_regenerations"] / km_k
    return agg.reset_index()


def message_counts(signals: pd.DataFrame, *, scope: str) -> pd.DataFrame:
    """Conteo por (vehículo, nivel de `Message`). Largo: el ancho lo arma el consumidor."""
    counted = (
        signals.dropna(subset=["Message"])
        .groupby([ID_COL, "Message"], observed=True)
        .size()
        .rename("n")
        .reset_index()
    )
    counted["scope"] = scope
    counted["slug"] = counted["Message"].map(MESSAGE_SLUGS).fillna("otro")
    return counted


def severity_by_scope(
    trips: pd.DataFrame, signals: pd.DataFrame, km_by_vehicle: pd.Series, *, scope: str, eda: dict[str, Any]
) -> pd.DataFrame:
    """Agregados de severidad de un alcance (historial completo o truncado).

    Es la preview empírica del gap de blanking: los mismos números calculados sobre
    el 70% u 80% inicial del odómetro de cada vehículo. Si la correlación con
    `event_observed` se desploma al sacar la cola, ahí está el motivo de la regla 1.
    """
    saturation = float(eda["saturation_level"])
    km_k = (km_by_vehicle / 1000.0).where(km_by_vehicle > 0)

    msg = message_counts(signals, scope=scope)
    wide = msg.pivot_table(index=ID_COL, columns="slug", values="n", aggfunc="sum").fillna(0.0)
    rates = wide.div(km_k.reindex(wide.index), axis=0)
    rates.columns = [f"msg_{c}_per_1000km" for c in rates.columns]

    sig = signals.assign(
        _acu_high=(signals["Acumulation"] >= 50),
        _acu_saturated=(signals["Acumulation"] >= saturation),
    ).groupby(ID_COL, observed=True).agg(
        acumulation_mean=("Acumulation", "mean"),
        acumulation_high_frac=("_acu_high", "mean"),
        acumulation_saturated_frac=("_acu_saturated", "mean"),
        n_regenerations=("Regenerations", "sum"),
        dist_between_regen_mean=("DistanceBetweenRegenerations", "mean"),
    )
    sig["regen_per_1000km"] = sig["n_regenerations"] / km_k.reindex(sig.index)

    tri = trips.assign(
        _short=(trips["trip_km"] < float(eda["short_trip_km"])),
        _below_regime=(trips["EngineTemperatureMax"] < float(eda["regime_temp_c"])),
        _regen_saturated=(trips["AirRegenerationEnd"] >= saturation),
        _filter_abnormal=(trips["AirFilterEnd"].notna() & trips["AirFilterEnd"].ne(NORMAL_MESSAGE)),
    ).groupby(ID_COL, observed=True).agg(
        n_trips=("trip_km", "size"),
        short_trip_frac=("_short", "mean"),
        below_regime_frac=("_below_regime", "mean"),
        air_regen_end_mean=("AirRegenerationEnd", "mean"),
        air_regen_saturated_frac=("_regen_saturated", "mean"),
        air_filter_abnormal_frac=("_filter_abnormal", "mean"),
    )

    out = sig.join(rates, how="outer").join(tri, how="outer")
    out["km_scope"] = km_by_vehicle.reindex(out.index)
    out["scope"] = scope
    return out.reset_index()


# ======================================================================================
# Resúmenes exactos a nivel fila (cuantiles, histogramas, calendario)
# ======================================================================================
def exact_quantiles(trips: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    """Cuantiles EXACTOS por cohorte de cada métrica a nivel viaje (no una muestra)."""
    qs = [0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0]
    event = trips[ID_COL].map(labels)
    rows = []
    for metric in TRIP_METRICS:
        values = trips[metric]
        for flag in (0, 1):
            side = values[event.eq(flag)].dropna()
            if side.empty:
                continue
            quantiles = side.quantile(qs)
            for q, value in quantiles.items():
                rows.append(
                    {"metric": metric, "event_observed": flag, "q": float(q), "value": float(value),
                     "n": int(len(side)), "mean": float(side.mean())}
                )
    return pd.DataFrame(rows)


def exact_histograms(trips: pd.DataFrame, labels: pd.Series, *, bins: int) -> pd.DataFrame:
    """Histograma EXACTO por cohorte: los 1,9M de viajes de dev, no una muestra."""
    event = trips[ID_COL].map(labels)
    rows = []
    for metric in TRIP_METRICS:
        values = trips[metric]
        lo, hi = HIST_CLIP.get(metric, (None, None))
        clean = values.dropna()
        if clean.empty:
            continue
        lo = float(clean.quantile(0.005)) if lo is None else lo
        hi = float(clean.quantile(0.995)) if hi is None else hi
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            continue
        edges = np.linspace(lo, hi, bins + 1)
        for flag in (0, 1):
            side = values[event.eq(flag)].dropna().to_numpy()
            if side.size == 0:
                continue
            counts, _ = np.histogram(np.clip(side, lo, hi), bins=edges)
            rows.append(
                pd.DataFrame(
                    {
                        "metric": metric,
                        "event_observed": flag,
                        "bin_left": edges[:-1],
                        "bin_right": edges[1:],
                        "n": counts,
                        "frac": counts / counts.sum() if counts.sum() else 0.0,
                    }
                )
            )
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def odometer_profile(
    vehicles: pd.DataFrame, trips: pd.DataFrame, origin_day: float, *, step_days: int = 5, max_days: int = 600
) -> pd.DataFrame:
    """Odómetro de cada vehículo en una grilla de días desde producción.

    Es la figura que explica por qué la traducción del evento al eje de km es
    delicada: si la flota pasa los primeros ~85 días con el odómetro casi en cero
    (en el concesionario), una fecha de evento temprana proyecta a ~0 km aunque el
    anclaje sea perfecto.
    """
    epoch = pd.Timestamp("1970-01-01", tz="UTC")
    grid = np.arange(0, max_days + step_days, step_days)
    base = vehicles[[ID_COL, "static_ProductionDay", "event_observed"]].dropna(subset=["static_ProductionDay"])
    long = base.loc[base.index.repeat(len(grid))].copy()
    long["day_since_production"] = np.tile(grid, len(base))
    long["date"] = epoch + pd.to_timedelta(
        origin_day + long["static_ProductionDay"] + long["day_since_production"], unit="D"
    )
    anchors = (
        trips[[ID_COL, "TripDatetimeStart", "OdometerTripEnd"]].dropna()
        .rename(columns={"TripDatetimeStart": "trip_date", "OdometerTripEnd": "odo"})
        .sort_values("trip_date")
    )
    merged = pd.merge_asof(
        long.sort_values("date"), anchors, left_on="date", right_on="trip_date", by=ID_COL, direction="backward"
    )
    merged = merged[merged["date"].le(merged[ID_COL].map(trips.groupby(ID_COL, observed=True)["TripDatetimeStart"].max()))]
    return (
        merged.groupby(["day_since_production", "event_observed"], observed=True)["odo"]
        .agg(p25=lambda s: s.quantile(0.25), median="median", p75=lambda s: s.quantile(0.75), n="size")
        .reset_index()
    )


def daily_activity(trips: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    """Actividad de la flota por día calendario: viajes, vehículos activos y km."""
    work = trips[[ID_COL, "TripDatetimeStart", "trip_km"]].dropna(subset=["TripDatetimeStart"]).copy()
    work["date"] = work["TripDatetimeStart"].dt.tz_convert("UTC").dt.floor("D")
    work["event_observed"] = work[ID_COL].map(labels)
    out = work.groupby(["date", "event_observed"], observed=True).agg(
        n_trips=("trip_km", "size"),
        n_vehicles=(ID_COL, "nunique"),
        km=("trip_km", lambda s: float(s.clip(lower=0).sum())),
    ).reset_index()
    out["weekday"] = out["date"].dt.dayofweek
    out["month"] = out["date"].dt.tz_localize(None).dt.to_period("M").dt.to_timestamp()
    return out


# ======================================================================================
# Anclaje temporal y odómetro del evento
# ======================================================================================
def estimate_origin_day(vehicles: pd.DataFrame) -> tuple[float, pd.Series]:
    """Origen del calendario en días desde epoch, estimado **sobre dev**.

    `primer_viaje − ProductionDay` es el puente que encontró F1
    (`docs/memoria/f1-anclaje-temporal.md`). Acá se re-estima restringido a dev para
    confirmar que el hallazgo se sostiene; es una estimación **exploratoria**, no el
    valor que F2 va a congelar.
    """
    first_trip = pd.to_datetime(vehicles["trip_date_min"], utc=True, errors="coerce")
    days = (first_trip.dt.floor("D") - pd.Timestamp("1970-01-01", tz="UTC")).dt.days
    offset = days - vehicles["static_ProductionDay"]
    return float(offset.median()), offset


def project_to_odometer(
    events: pd.DataFrame, trips: pd.DataFrame, *, date_col: str, prefix: str
) -> pd.DataFrame:
    """Proyecta una fecha candidata de evento sobre el eje de km de cada vehículo.

    Interpola linealmente entre el viaje inmediatamente anterior y el inmediatamente
    posterior a esa fecha. Devuelve, además del odómetro, **cuánto historial queda
    del lado de acá**: `{prefix}_trips_before` y `{prefix}_frac_trips_before` son la
    pregunta que decide si la fecha sirve para F2 —un evento con el 0% del historial
    por delante no deja ventana W que agregar ni gap G que blanquear—.
    """
    ordered = events.dropna(subset=[date_col]).sort_values(date_col)
    anchors = (
        trips[[ID_COL, "TripDatetimeStart", "OdometerTripEnd"]]
        .dropna()
        .rename(columns={"TripDatetimeStart": "trip_date", "OdometerTripEnd": "odo"})
        .sort_values("trip_date")
    )
    before = pd.merge_asof(
        ordered[[ID_COL, date_col]], anchors, left_on=date_col, right_on="trip_date",
        by=ID_COL, direction="backward",
    ).rename(columns={"trip_date": "date_before", "odo": "odo_before"})
    after = pd.merge_asof(
        ordered[[ID_COL, date_col]], anchors, left_on=date_col, right_on="trip_date",
        by=ID_COL, direction="forward",
    )[[ID_COL, "trip_date", "odo"]].rename(columns={"trip_date": "date_after", "odo": "odo_after"})
    merged = before.merge(after, on=ID_COL, how="left")

    span = (merged["date_after"] - merged["date_before"]).dt.total_seconds()
    weight = ((merged[date_col] - merged["date_before"]).dt.total_seconds() / span.where(span > 0)).clip(0, 1)
    interpolated = merged["odo_before"] + weight * (merged["odo_after"] - merged["odo_before"])

    # Cuánto historial de viajes queda antes de la fecha candidata.
    cut = merged.set_index(ID_COL)[date_col]
    trip_dates = trips[[ID_COL, "TripDatetimeStart"]].dropna()
    is_before = trip_dates["TripDatetimeStart"].le(trip_dates[ID_COL].map(cut))
    counts = is_before.groupby(trip_dates[ID_COL], observed=True).agg(["sum", "size"])

    out = pd.DataFrame({ID_COL: merged[ID_COL]})
    out[f"{prefix}_date"] = merged[date_col].to_numpy()
    out[f"{prefix}_odo_km"] = interpolated.where(interpolated.notna(), merged["odo_before"]).to_numpy()
    out[f"{prefix}_source"] = np.where(
        merged["odo_after"].notna() & merged["odo_before"].notna(), "interpolado",
        np.where(merged["odo_before"].notna(), "posterior_al_ultimo_viaje", "previo_al_primer_viaje"),
    )
    out[f"{prefix}_trips_before"] = out[ID_COL].map(counts["sum"]).astype("float64")
    out[f"{prefix}_frac_trips_before"] = out[f"{prefix}_trips_before"] / out[ID_COL].map(counts["size"])
    return out


def event_odometer(vehicles: pd.DataFrame, trips: pd.DataFrame, origin_day: float) -> pd.DataFrame:
    """Traduce el evento al eje de km con las **dos** lecturas posibles del eje de días.

    El diccionario oficial (anexo 7.3) dice que `IdentificationDate` son "días desde
    producción". Con el anclaje de F1 (`docs/memoria/f1-anclaje-temporal.md`):

        A · fecha_evento = origen + ProductionDay + IdentificationDate

    Se calcula además la lectura alternativa —`IdentificationDate` contado desde la
    **venta** en vez de desde producción—:

        B · fecha_evento = origen + ProductionDay + daysUntilSale + IdentificationDate

    No es capricho: en dev, `IdentificationDate == daysUntilSale` exactamente en el
    79% de los vehículos con evento, y bajo la lectura A esos eventos caen con el
    odómetro casi en cero. Las dos se materializan para que el notebook las compare
    con evidencia en vez de elegir de memoria. **Ninguna de las dos se declara
    correcta acá**: eso es una decisión de F2.
    """
    events = vehicles.loc[
        vehicles["event_observed"].eq(1),
        [ID_COL, "static_ProductionDay", "static_daysUntilSale", "event_day_since_production"],
    ].copy()
    epoch = pd.Timestamp("1970-01-01", tz="UTC")
    events["date_a"] = epoch + pd.to_timedelta(
        origin_day + events["static_ProductionDay"] + events["event_day_since_production"], unit="D"
    )
    events["date_b"] = epoch + pd.to_timedelta(
        origin_day + events["static_ProductionDay"] + events["static_daysUntilSale"]
        + events["event_day_since_production"], unit="D"
    )
    a = project_to_odometer(events, trips, date_col="date_a", prefix="event")
    b = project_to_odometer(events, trips, date_col="date_b", prefix="event_alt")
    out = a.merge(b, on=ID_COL, how="outer")
    out["ident_equals_sale"] = out[ID_COL].map(
        events.set_index(ID_COL)["event_day_since_production"].eq(
            events.set_index(ID_COL)["static_daysUntilSale"]
        )
    )
    return out


# ======================================================================================
# Construcción
# ======================================================================================
def build_cache(cfg: dict[str, Any]) -> dict[str, Any]:
    """Arma el cache completo y lo escribe. Devuelve la metadata."""
    rng = np.random.default_rng(int(cfg["seed"]))
    eda = cfg["eda"]
    out_dir = ensure_dir(cfg["output"]["dir"])
    dev_vehicles, test_vehicles, split = dev_universe(cfg)
    universe = split.get("universe") or {}
    logger.info(
        "Holdout: %d de dev, %d de test (intocables), %d fuera del universo del estudio",
        len(dev_vehicles), len(test_vehicles), int(split.get("n_excluded", 0)),
    )

    # --- nivel vehículo -----------------------------------------------------------
    vehicles = pd.read_parquet(resolve_path(cfg["vehicle_table"]))
    vehicles = vehicles[vehicles[ID_COL].isin(dev_vehicles)].reset_index(drop=True)
    assert_dev_only(vehicles, dev_vehicles, name="vehicles_dev")
    if len(vehicles) != len(dev_vehicles):
        raise ValueError(f"vehicles.parquet cubre {len(vehicles)} de los {len(dev_vehicles)} vehículos de dev")

    origin_day, offset = estimate_origin_day(vehicles)
    vehicles["anchor_offset_days"] = offset

    # --- trips --------------------------------------------------------------------
    trips, trip_counters = read_dev_table("trips", TRIP_COLUMNS, dev_vehicles=dev_vehicles, cfg=cfg)
    assert_dev_only(trips, dev_vehicles, name="trips_dev")
    trips = with_trip_derived(trips)

    trips_agg = trip_aggregates(trips, eda)
    assert_dev_only(trips_agg, dev_vehicles, name="trips_agg_dev")
    km_by_vehicle = trips_agg.set_index(ID_COL)["trip_km_sum"]
    labels = vehicles.set_index(ID_COL)["event_observed"]

    events = event_odometer(vehicles, trips, origin_day)
    vehicles = vehicles.merge(events, on=ID_COL, how="left")
    # Nullable: los 230 vehículos sin evento no tienen valor. Sin el cast explícito el
    # left join la deja como `object` y `~columna` hace un NOT de enteros, no booleano.
    vehicles["ident_equals_sale"] = vehicles["ident_equals_sale"].astype("boolean")
    vehicles["km_observed"] = vehicles[ID_COL].map(trips_agg.set_index(ID_COL)["trip_odo_max"])
    # Duración para supervivencia sobre el eje de km: odómetro del evento si hay
    # evento, último odómetro observado si el vehículo está censurado. El censurado
    # NO necesita el anclaje —`trip_odo_max` es dato directo—, así que la curva de
    # los sanos es sólida aunque la traducción del evento esté en discusión.
    for suffix in ("", "_alt"):
        odo = vehicles[f"event{suffix}_odo_km"]
        vehicles[f"duration_km{suffix}"] = np.where(
            vehicles["event_observed"].eq(1) & odo.notna(), odo, vehicles["km_observed"]
        )
    assert_dev_only(vehicles, dev_vehicles, name="vehicles_dev+evento")

    # --- signals ------------------------------------------------------------------
    signals, signal_counters = read_dev_table("signals", SIGNAL_COLUMNS, dev_vehicles=dev_vehicles, cfg=cfg)
    assert_dev_only(signals, dev_vehicles, name="signals_dev")
    signals_agg = signal_aggregates(signals, km_by_vehicle, eda)
    assert_dev_only(signals_agg, dev_vehicles, name="signals_agg_dev")

    # --- severidad por alcance (preview del gap de blanking) ----------------------
    odo_min = trips_agg.set_index(ID_COL)["trip_odo_min"]
    odo_max = trips_agg.set_index(ID_COL)["trip_odo_max"]
    scopes = [severity_by_scope(trips, signals, km_by_vehicle, scope="full", eda=eda)]
    messages = [message_counts(signals, scope="full")]
    for frac in eda["truncation_fracs"]:
        cut = (odo_min + float(frac) * (odo_max - odo_min)).rename("cut")
        trip_cut = trips[ID_COL].map(cut)
        sig_cut = signals[ID_COL].map(cut)
        trips_t = trips[trips["OdometerTripEnd"].le(trip_cut)]
        signals_t = signals[signals["OdometerValue"].le(sig_cut)]
        km_t = trips_t.groupby(ID_COL, observed=True)["trip_km"].apply(lambda s: float(s.clip(lower=0).sum()))
        name = f"p{int(round(float(frac) * 100))}"
        scopes.append(severity_by_scope(trips_t, signals_t, km_t, scope=name, eda=eda))
        messages.append(message_counts(signals_t, scope=name))
    severity = pd.concat(scopes, ignore_index=True)
    message_long = pd.concat(messages, ignore_index=True)
    assert_dev_only(severity, dev_vehicles, name="severity_scopes_dev")
    assert_dev_only(message_long, dev_vehicles, name="message_counts_dev")

    # --- niveles de AirFilter (las columnas que el diccionario oficial no declara) --
    air_filter = pd.concat(
        [
            trips.dropna(subset=[column])
            .groupby([ID_COL, column], observed=True)
            .size().rename("n").reset_index()
            .rename(columns={column: "level"})
            .assign(position="start" if column.endswith("Start") else "end")
            for column in ("AirFilterStart", "AirFilterEnd")
        ],
        ignore_index=True,
    )
    air_filter["slug"] = air_filter["level"].map(MESSAGE_SLUGS).fillna("otro")
    assert_dev_only(air_filter, dev_vehicles, name="airfilter_counts_dev")

    # --- resúmenes exactos y muestras ---------------------------------------------
    quantiles = exact_quantiles(trips, labels)
    histograms = exact_histograms(trips, labels, bins=int(eda["hist_bins"]))
    daily = daily_activity(trips, labels)
    odo_profile = odometer_profile(vehicles, trips, origin_day)

    trips_sample = _sample(trips, int(eda["sample_rows_trips"]), rng)
    signals_sample = _sample(signals, int(eda["sample_rows_signals"]), rng)
    assert_dev_only(trips_sample, dev_vehicles, name="trips_sample_dev")
    assert_dev_only(signals_sample, dev_vehicles, name="signals_sample_dev")

    # --- sonda: historial completo de unos pocos vehículos ------------------------
    n_probe = int(eda["probe_vehicles"])
    con_evento = vehicles.loc[vehicles["event_observed"].eq(1), ID_COL].to_numpy()
    sin_evento = vehicles.loc[vehicles["event_observed"].eq(0), ID_COL].to_numpy()
    probe_ids = np.concatenate(
        [
            rng.choice(con_evento, size=min(n_probe // 2, len(con_evento)), replace=False),
            rng.choice(sin_evento, size=min(n_probe - n_probe // 2, len(sin_evento)), replace=False),
        ]
    )
    probe_trips = trips[trips[ID_COL].isin(probe_ids)].copy()
    probe_signals = signals[signals[ID_COL].isin(probe_ids)].copy()
    assert_dev_only(probe_trips, dev_vehicles, name="probe_trips_dev")
    assert_dev_only(probe_signals, dev_vehicles, name="probe_signals_dev")

    # --- escritura ----------------------------------------------------------------
    tables = {
        "vehicles_dev": vehicles,
        "trips_agg_dev": trips_agg,
        "signals_agg_dev": signals_agg,
        "severity_scopes_dev": severity,
        "message_counts_dev": message_long,
        "airfilter_counts_dev": air_filter,
        "trip_quantiles_dev": quantiles,
        "trip_hist_dev": histograms,
        "trips_daily_dev": daily,
        "odo_profile_dev": odo_profile,
        "trips_sample_dev": trips_sample,
        "signals_sample_dev": signals_sample,
        "probe_trips_dev": probe_trips,
        "probe_signals_dev": probe_signals,
    }
    for name, frame in tables.items():
        # Los resúmenes ya agregados (cuantiles, histogramas, calendario) no llevan
        # `vehicle_id`: salen de tablas que YA pasaron el assert, así que la guarda
        # se aplica donde puede verificarse fila a fila.
        if ID_COL in frame.columns:
            assert_dev_only(frame, dev_vehicles, name=name)
        frame.to_parquet(out_dir / f"{name}.parquet", index=False)
        logger.info("  %-24s %8d filas x %3d columnas", name, len(frame), frame.shape[1])

    meta = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": cfg.get("_config_path"),
        "seed": int(cfg["seed"]),
        "scope": "dev-only",
        "n_dev_vehicles": int(len(vehicles)),
        "n_test_vehicles_excluidos": int(len(test_vehicles)),
        "n_fuera_del_universo": int(split.get("n_excluded", 0)),
        "universe": {k: v for k, v in universe.items() if k != "markets"},
        "market_usability": universe.get("markets"),
        "n_event_vehicles": int(vehicles["event_observed"].sum()),
        "event_rate": float(vehicles["event_observed"].mean()),
        "test_split_created_at": split.get("created_at"),
        "test_split_seed": split.get("seed"),
        "anchor_origin_day_since_epoch": origin_day,
        "anchor_offset_std_days": float(offset.std()),
        "anchor_offset_iqr_days": float(offset.quantile(0.75) - offset.quantile(0.25)),
        "anchor_offset_range_days": float(offset.max() - offset.min()),
        "ident_equals_sale_frac": float(vehicles.loc[vehicles["event_observed"].eq(1), "ident_equals_sale"].mean()),
        "event_odo_median_km": float(vehicles.loc[vehicles["event_observed"].eq(1), "event_odo_km"].median()),
        "event_odo_alt_median_km": float(vehicles.loc[vehicles["event_observed"].eq(1), "event_alt_odo_km"].median()),
        "trips": trip_counters,
        "signals": signal_counters,
        "probe_vehicles": sorted(probe_ids.tolist()),
        "truncation_fracs": list(eda["truncation_fracs"]),
        "thresholds": {k: eda[k] for k in ("short_trip_km", "very_short_trip_km", "regime_temp_c", "urban_speed_kmh", "saturation_level")},
        "tables": {name: {"rows": int(len(f)), "cols": int(f.shape[1])} for name, f in tables.items()},
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta


def _sample(frame: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Muestra determinística. Solo para dibujar puntos: los cuantiles del cache son exactos."""
    if len(frame) <= n:
        return frame.copy()
    idx = rng.choice(len(frame), size=n, replace=False)
    return frame.iloc[np.sort(idx)].reset_index(drop=True)


# ======================================================================================
# Consumo (notebook + dashboard: un solo lugar)
# ======================================================================================
CACHE_TABLES = (
    "vehicles_dev", "trips_agg_dev", "signals_agg_dev", "severity_scopes_dev", "message_counts_dev",
    "airfilter_counts_dev", "trip_quantiles_dev", "trip_hist_dev", "trips_daily_dev",
    "odo_profile_dev", "trips_sample_dev", "signals_sample_dev", "probe_trips_dev", "probe_signals_dev",
)


def cache_dir(config_path: str | Path = DEFAULT_CONFIG) -> Path:
    return resolve_path(load_config(config_path)["output"]["dir"])


def cache_exists(config_path: str | Path = DEFAULT_CONFIG) -> bool:
    directory = cache_dir(config_path)
    return (directory / "meta.json").exists() and all(
        (directory / f"{name}.parquet").exists() for name in CACHE_TABLES
    )


def load_eda_cache(config_path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Levanta el cache dev-only. Es la única puerta de entrada a los datos del EDA.

    El notebook y el dashboard llaman a esto: si cada uno filtrara por su cuenta,
    tarde o temprano uno se olvida de canonizar antes de filtrar y el conteo cambia
    sin que nadie lo note.
    """
    directory = cache_dir(config_path)
    if not cache_exists(config_path):
        raise FileNotFoundError(
            f"No hay cache del EDA en {directory}. Construilo con:\n"
            f"    python scripts/build_eda_cache.py --config {config_path}"
        )
    data: dict[str, Any] = {"meta": json.loads((directory / "meta.json").read_text(encoding="utf-8"))}
    for name in CACHE_TABLES:
        data[name] = pd.read_parquet(directory / f"{name}.parquet")
    return data


def vehicle_matrix(cache: dict[str, Any]) -> pd.DataFrame:
    """Una fila por vehículo con TODO lo agregado: estáticas, uso, severidad y etiqueta.

    Es la tabla sobre la que corren las correlaciones y los boxplots por cohorte.
    """
    base = cache["vehicles_dev"]
    trips = cache["trips_agg_dev"].drop(columns=["trip_date_min", "trip_date_max"], errors="ignore")
    signals = cache["signals_agg_dev"].drop(columns=["signal_date_min", "signal_date_max"], errors="ignore")
    severity = cache["severity_scopes_dev"]
    rates = severity[severity["scope"].eq("full")].drop(columns=["scope"], errors="ignore")
    rates = rates[[ID_COL] + [c for c in rates.columns if c.startswith("msg_")]]
    out = base.merge(trips, on=ID_COL, how="left", suffixes=("", "_dup"))
    out = out.merge(signals, on=ID_COL, how="left", suffixes=("", "_dup"))
    out = out.merge(rates, on=ID_COL, how="left", suffixes=("", "_dup"))
    return out.drop(columns=[c for c in out.columns if c.endswith("_dup")])


# ======================================================================================
# Diccionario de datos y factibilidad de las features del plan §4
#
# Vive acá —y no en el notebook ni en el dashboard— por el mismo motivo que el filtro
# a dev: los dos lo muestran, así que si estuviera escrito dos veces, un día dirían
# cosas distintas.
# ======================================================================================

# Columnas del diccionario oficial, transcritas del PDF de Ford
# (`data/raw/FIC_III___Data_Driven_Powertrain_Intelligence (2).pdf`, anexos 7.1-7.3).
# El PDF no es legible por código; el orden es el del anexo.
PDF_COLUMNS: dict[str, list[str]] = {
    "trips": [
        "Vin", "TripDatetimeStart", "TripDatetimeEnd", "OdometerTripStart", "OdometerTripEnd",
        "KilometerPerHour", "FuelLvlStartPc", "FuelLvlEndPc", "FuelLvlAutonomyStart",
        "EngineOilLifePCStart", "EngineOilLifePCEnd", "EngineTemperatureMin", "EngineTemperatureMax",
        "EngineTemperatureAvg", "CoolantTemperatureStart", "CoolantTemperatureEnd",
        "DieselParticulateFilterStart", "DieselParticulateFilterEnd",
        "VehicleGPSLatDataStart", "VehicleGPSLatDataEnd", "VehicleGPSLongDataStart",
        "VehicleGPSLongDataEnd", "VehicleElevationRangeStart", "VehicleElevationRangeEnd",
        "TirePressureLFStart", "TirePressureLFEnd", "TirePressureRFStart", "TirePressureRFEnd",
        "TirePressureLRStart", "TirePressureLREnd", "TirePressureRRStart", "TirePressureRREnd",
        "AirTemperatureStart", "AirTemperatureEnd", "AirTemperatureMin", "AirTemperatureMax",
        "AirTemperatureAvg", "ManualRegenerationSootStart", "ManualRegenerationSootEnd", "TripNumber",
    ],
    "signals": [
        "VehicleCode", "EventTimestamp", "OdometerValue", "Acumulation", "Message", "Regenerations",
        "DistanceBetweenRegenerations",
    ],
    "vehicles": [
        "VehCode", "IdentificationDate", "daysUntilSale", "ProductionDay", "Engine", "ModelSeries",
        "SalesCountryCd",
    ],
}

# Renombres cosméticos (mismo concepto, otro nombre): no se cuentan como faltantes.
PDF_RENAMES = {
    "Vin": "VehicleCode",
    "VehCode": "VehicleCode",
    "EventTimestamp": "eventTimestamp",
    "SalesCountryCd": "SalesCountry_cd",
}

# Familias de columnas prometidas por el anexo 7.1 que no están en ningún archivo.
MISSING_FAMILIES = {
    "GPS": ["VehicleGPSLatDataStart", "VehicleGPSLatDataEnd", "VehicleGPSLongDataStart", "VehicleGPSLongDataEnd"],
    "elevación": ["VehicleElevationRangeStart", "VehicleElevationRangeEnd"],
    "presión de neumáticos": [f"TirePressure{p}{s}" for p in ("LF", "RF", "LR", "RR") for s in ("Start", "End")],
    "DPF (por ese nombre)": ["DieselParticulateFilterStart", "DieselParticulateFilterEnd"],
    "hollín de regeneración manual": ["ManualRegenerationSootStart", "ManualRegenerationSootEnd"],
    "autonomía de combustible": ["FuelLvlAutonomyStart"],
}

# (feature reservada en scripts/make_dummy.py::FEATURE_SPECS, familia del plan §4,
#  columna cruda que necesita, tabla).
PLAN_FEATURE_REQUIREMENTS = [
    ("feat_short_trip_frac_5km", "A", "OdometerTripStart/End", "trips"),
    ("feat_trips_below_regime_temp_frac", "A", "EngineTemperatureMax", "trips"),
    ("feat_engine_temp_avg_median", "A", "EngineTemperatureAvg", "trips"),
    ("feat_engine_temp_amplitude_mean", "A", "EngineTemperatureMin/Max", "trips"),
    ("feat_coolant_temp_end_mean", "A", "CoolantTemperatureEnd", "trips"),
    ("feat_trip_distance_median_km", "A", "OdometerTripStart/End", "trips"),
    ("feat_trip_distance_p25_km", "A", "OdometerTripStart/End", "trips"),
    ("feat_dpf_end_slope_per_1000km", "B", "DieselParticulateFilterEnd", "trips"),
    ("feat_dpf_end_mean", "B", "DieselParticulateFilterEnd", "trips"),
    ("feat_dpf_end_max", "B", "DieselParticulateFilterEnd", "trips"),
    ("feat_dpf_positive_delta_frac", "B", "DieselParticulateFilterStart/End", "trips"),
    ("feat_regenerations_per_1000km", "B", "Regenerations", "signals"),
    ("feat_distance_between_regen_mean_km", "B", "DistanceBetweenRegenerations", "signals"),
    ("feat_distance_between_regen_trend", "B", "DistanceBetweenRegenerations", "signals"),
    ("feat_manual_regen_per_1000km", "B", "ManualRegenerationSootStart/End", "trips"),
    ("feat_accumulation_mean", "B", "Acumulation", "signals"),
    ("feat_accumulation_slope_per_1000km", "B", "Acumulation", "signals"),
    ("feat_speed_kmh_mean", "C", "KilometerPerHour", "trips"),
    ("feat_trips_below_30kmh_frac", "C", "KilometerPerHour", "trips"),
    ("feat_km_per_day", "C", "OdometerTripStart/End", "trips"),
    ("feat_trips_per_day", "C", "TripDatetimeStart", "trips"),
    ("feat_hours_between_trips_median", "C", "TripDatetimeStart/End", "trips"),
    ("feat_air_temp_avg", "C", "AirTemperatureAvg", "trips"),
    ("feat_air_temp_min", "C", "AirTemperatureMin", "trips"),
    ("feat_elevation_mean_m", "C", "VehicleElevationRangeStart/End", "trips"),
    ("feat_elevation_range_m", "C", "VehicleElevationRangeStart/End", "trips"),
    ("feat_oil_life_drop_per_1000km", "D", "EngineOilLifePCStart/End", "trips"),
    ("feat_tire_pressure_mean", "D", "TirePressure{LF,RF,LR,RR}{Start,End}", "trips"),
    ("feat_tire_pressure_below_thr_frac", "D", "TirePressure{LF,RF,LR,RR}{Start,End}", "trips"),
    ("feat_n_trips_window", "·", "OdometerTripEnd", "trips"),
    ("feat_window_km_covered", "·", "OdometerTripStart/End", "trips"),
]

# Sustitutos medidos en el notebook (§6.4): el dato existe, con otro nombre.
COLUMN_SUBSTITUTES = {
    "DieselParticulateFilterEnd": "AirRegenerationEnd (= signals.Acumulation en el 99,8% de los pares)",
    "DieselParticulateFilterStart/End": "AirRegenerationStart/End (= signals.Acumulation en el 99,8%)",
    "ManualRegenerationSootStart/End": "parcial · niveles 'Cleanning/Stopped ... Manually' de Message/AirFilter",
}

# Columnas que **existen** pero cuyo contenido no sirve para la feature tal como está
# especificada en el plan §4. Medido en el notebook §5.3.
DEGENERATE_REQUIREMENTS = {
    "EngineOilLifePCStart/End": (
        "`EngineOilLifePCStart == EngineOilLifePCEnd` en el 100% de los viajes: el delta "
        "intra-viaje es exactamente 0. El nivel sí varía entre viajes (0–100), así que la "
        "feature hay que redefinirla como pendiente del NIVEL contra odómetro dentro de la ventana."
    ),
}


def raw_columns(config_path: str | Path = DEFAULT_CONFIG) -> dict[str, list[str]]:
    """Encabezado real de cada CSV. Se lee del archivo, no del contrato."""
    cfg = load_config(config_path)
    tables = load_config(cfg["sources"])["tables"]
    return {
        name: list(
            pd.read_csv(resolve_path(next(iter(spec["parts"].values()))), nrows=0, encoding="utf-8-sig").columns
        )
        for name, spec in tables.items()
    }


def dictionary_comparison(config_path: str | Path = DEFAULT_CONFIG) -> pd.DataFrame:
    """Comparación de tres vías: CSV real vs. `raw_sources.yaml` vs. PDF oficial."""
    cfg = load_config(config_path)
    tables = load_config(cfg["sources"])["tables"]
    reales = raw_columns(config_path)
    rows = []
    for name, spec in tables.items():
        declaradas = set(spec.get("dtypes", {}) or {}) | set(spec.get("date_columns", []) or [])
        oficiales = {PDF_RENAMES.get(c, c): c for c in PDF_COLUMNS[name]}
        for column in sorted(set(reales[name]) | declaradas | set(oficiales)):
            rows.append(
                {
                    "tabla": name,
                    "columna": column,
                    "en_csv": column in reales[name],
                    "en_raw_sources": column in declaradas,
                    "en_pdf": column in oficiales,
                    "nombre_en_pdf": oficiales.get(column, ""),
                }
            )
    out = pd.DataFrame(rows)
    out["estado"] = np.select(
        [out["en_csv"] & out["en_pdf"], out["en_csv"] & ~out["en_pdf"], ~out["en_csv"] & out["en_pdf"]],
        ["documentada y presente", "presente SIN documentar", "documentada y AUSENTE"],
        default="?",
    )
    return out


def _expand_requirement(requirement: str) -> list[str]:
    """`EngineTemperatureMin/Max` -> las dos columnas; el patrón de neumáticos -> las ocho."""
    if requirement == "TirePressure{LF,RF,LR,RR}{Start,End}":
        return [f"TirePressure{p}{s}" for p in ("LF", "RF", "LR", "RR") for s in ("Start", "End")]
    for sufijo in ("Start/End", "Min/Max"):
        if requirement.endswith(sufijo):
            base = requirement[: -len(sufijo)]
            return [base + parte for parte in sufijo.split("/")]
    return [requirement]


def feature_feasibility(config_path: str | Path = DEFAULT_CONFIG) -> pd.DataFrame:
    """Qué feature del plan §4 se puede construir con los CSV reales, y cuál no."""
    reales = raw_columns(config_path)
    rows = []
    for feature, familia, requirement, table in PLAN_FEATURE_REQUIREMENTS:
        presente = all(part in reales[table] for part in _expand_requirement(requirement))
        sustituto = COLUMN_SUBSTITUTES.get(requirement, "") or DEGENERATE_REQUIREMENTS.get(requirement, "")
        if presente and requirement in DEGENERATE_REQUIREMENTS:
            veredicto, estado = "REQUIERE REDEFINIR · la columna está pero el contenido no sirve", "degenerada"
        elif presente:
            veredicto, estado = "se puede construir", "disponible"
        elif sustituto.startswith("parcial"):
            veredicto, estado = "parcial con sustituto", "parcial"
        elif sustituto:
            veredicto, estado = "se puede construir (con sustituto)", "disponible"
        else:
            veredicto, estado = "BLOQUEADA · no hay dato", "bloqueada"
        rows.append(
            {
                "feature": feature, "familia": familia, "columna_requerida": requirement,
                "tabla": table, "existe": presente, "sustituto": sustituto,
                "veredicto": veredicto, "estado": estado,
            }
        )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--force", action="store_true", help="reconstruye aunque el cache ya exista")
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg["seed"]))

    if cache_exists(args.config) and not args.force:
        print(f"El cache ya existe en {cache_dir(args.config)}. Se reconstruye con --force.")
        return 0

    meta = build_cache(cfg)
    print("\n== Cache del EDA (dev-only) ==")
    print(f"vehículos dev            : {meta['n_dev_vehicles']} ({meta['n_event_vehicles']} con evento, "
          f"tasa {meta['event_rate']:.4f})")
    print(f"vehículos de test fuera  : {meta['n_test_vehicles_excluidos']}")
    print(f"trips dev                : {meta['trips']['n_dev_crudas']} -> {meta['trips']['n_dev_deduplicadas']} "
          f"(clones {meta['trips']['dup_clones']}, otros {meta['trips']['dup_otros']})")
    print(f"signals dev              : {meta['signals']['n_dev_crudas']} -> {meta['signals']['n_dev_deduplicadas']} "
          f"(clones {meta['signals']['dup_clones']}, otros {meta['signals']['dup_otros']})")
    print(f"origen del calendario    : día {meta['anchor_origin_day_since_epoch']:.0f} desde epoch "
          f"(std {meta['anchor_offset_std_days']:.2f} d, IQR {meta['anchor_offset_iqr_days']:.2f} d)")
    print(f"\nEscrito en {cache_dir(args.config)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
