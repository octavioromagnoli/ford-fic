#!/usr/bin/env python
"""Complemento del EDA exhaustivo: lo que el notebook no midió y F2 necesita.

    python scripts/eda_gaps.py --config configs/data/eda_cache.yaml

Corre **solo sobre dev** (misma guarda y misma carga que `scripts/build_eda_cache.py`:
canonizar, filtrar a `dev_vehicles`, deduplicar) y deja CSVs en
`experiments/eda/dev/gaps/`. Cada bloque responde una pregunta concreta que el EDA
de `notebooks/eda-exhaustivo-dev.ipynb` dejó abierta o no formuló:

1. **Factibilidad de (W, G, H, Δ)**: cuántos eventos y cuántos sanos sobreviven a
   cada terna, y cuántas filas/positivos tendría el panel. Es el insumo para
   completar `configs/data/panel_v1.yaml`, que hoy tiene `null` en los cuatro.
2. **Perfil alineado al evento**: cada agregado en bins de km *hasta el evento*,
   contra el mismo agregado en bins de odómetro absoluto de los sanos. Dice qué
   features se mueven temprano (anticipan) y cuáles solo cerca del evento
   (detectan). Es la versión fina del truncamiento al 70% de §7.4.
3. **Post-evento**: si el uso cambia después de `IdentificationDate`. Valida que la
   fecha marca una intervención real y justifica descartar el historial posterior.
4. **Regeneraciones interrumpidas**: `AirFilterEnd == "Stopped Cleaning
   Automatically"` es un viaje que terminó en medio de una regeneración. Es la
   traducción literal de "operación que no permite completar los ciclos".
5. **Arranques en frío, consumo por km, tiempo entre viajes cortos**: familias A y C
   con columnas que el EDA no había explotado (`CoolantTemperatureStart`,
   `FuelLvl*Pc`, encadenamiento de viajes).
6. **Completitud del marcador `Regenerations`** por vehículo, y su relación con las
   caídas de `AirRegeneration*`.
7. **Estabilidad intra-vehículo** de los agregados en ventanas de 2.000 km: cuánta
   varianza es del vehículo (rasgo) y cuánta de la ventana (estado). Una feature
   100% rasgo no cambia con el corte; una 100% estado es ruido.
8. Chequeos de dato que faltaban: `TripNumber`, monotonía del odómetro con los tres
   órdenes posibles (fecha / `TripNumber` / odómetro), cobertura `signals` vs `trips`,
   nulos de `KilometerPerHour` según el viaje, hora del día y fin de semana.
9. **Calendario**: qué registra cada canal por mes (el marcador `Regenerations` se
   corta el 25-05-2026 para toda la flota), en qué meses caen los eventos y en cuáles
   la exposición de los sanos. Es la evidencia del confusor calendario que obliga a
   emparejar los cortes de los sanos también por mes.
10. **Cadencia**: cuántos km y cuántos viajes junta un vehículo en 7/15/30 días, y qué
   fracción de esas ventanas no pasaría el QC. Separa la *ventana de agregación* (km,
   que es el eje del panel) de la *cadencia de emisión* (calendario).

Deja además `trips_dev_full.parquet` / `signals_dev_full.parquet` en `gaps/` (dev
completo, ya canonizado y deduplicado) para no releer 1,2 GB en cada iteración.
Los hallazgos están resumidos en `docs/memoria/f2-eda-revision-y-features.md`.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from scripts.build_eda_cache import (  # noqa: E402
    MESSAGE_SLUGS,
    NORMAL_MESSAGE,
    SIGNAL_COLUMNS,
    TRIP_COLUMNS,
    assert_dev_only,
    dev_universe,
    load_eda_cache,
    read_dev_table,
    with_trip_derived,
)
from src.config import ensure_dir, load_config, set_seed  # noqa: E402

logger = logging.getLogger("eda_gaps")

ID = "vehicle_id"
STOPPED_AUTO = "Stopped Cleaning Automatically Air Filter"
CLEANING_AUTO = "Cleaning Automatically Air Filter"

# Cortes de lectura (mismos que el cache donde ya existían; los nuevos, declarados acá).
SHORT_KM = 5.0
REGIME_C = 70.0
URBAN_KMH = 30.0
COLD_START_C = 40.0        # refrigerante al arrancar por debajo => motor frío
CHAINED_MIN = 30.0         # viaje que arranca < 30 min después del anterior: motor aún caliente
REGEN_DROP = 5.0           # caída de AirRegeneration dentro del viaje que cuenta como regeneración
SATURATION = 95.0


def _print(title: str, frame: pd.DataFrame | pd.Series | str, **kw: Any) -> None:
    print(f"\n== {title} ==")
    if isinstance(frame, (pd.DataFrame, pd.Series)):
        print(frame.to_string(**kw))
    else:
        print(frame)


# ======================================================================================
# 0 · Carga (dev-only) y derivadas a nivel viaje
# ======================================================================================
def load_dev(cfg: dict[str, Any], out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Carga dev-only. Las tablas completas de dev se dejan en parquet dentro de `gaps/`
    para no releer 1,2 GB en cada iteración; el filtro sigue siendo el del cache."""
    dev_vehicles, _, _ = dev_universe(cfg)
    cache = load_eda_cache(cfg["_config_path"])
    veh = cache["vehicles_dev"].copy()
    assert_dev_only(veh, dev_vehicles, name="vehicles_dev")

    trips_path, signals_path = out_dir / "trips_dev_full.parquet", out_dir / "signals_dev_full.parquet"
    if trips_path.exists() and signals_path.exists():
        trips, signals = pd.read_parquet(trips_path), pd.read_parquet(signals_path)
    else:
        trips, _ = read_dev_table("trips", TRIP_COLUMNS, dev_vehicles=dev_vehicles, cfg=cfg)
        trips = with_trip_derived(trips)
        trips = trips.sort_values([ID, "OdometerTripEnd", "TripDatetimeStart"], ignore_index=True)
        signals, _ = read_dev_table("signals", SIGNAL_COLUMNS, dev_vehicles=dev_vehicles, cfg=cfg)
        signals = signals.sort_values([ID, "OdometerValue"], ignore_index=True)
        trips.to_parquet(trips_path, index=False)
        signals.to_parquet(signals_path, index=False)
    assert_dev_only(trips, dev_vehicles, name="trips_dev")
    assert_dev_only(signals, dev_vehicles, name="signals_dev")
    return veh, trips, signals


def enrich_trips(trips: pd.DataFrame, veh: pd.DataFrame) -> pd.DataFrame:
    """Columnas a nivel viaje que el EDA no había derivado."""
    out = trips
    info = veh.set_index(ID)
    out["event_observed"] = out[ID].map(info["event_observed"]).astype(int)
    out["event_odo_km"] = out[ID].map(info["event_odo_km"])
    out["km_to_event"] = out["event_odo_km"] - out["OdometerTripEnd"]

    # Velocidad recalculada desde odómetro y tiempo (KilometerPerHour es 35% nulo y llega a 135.600).
    hours = out["trip_duration_min"] / 60.0
    out["speed_calc"] = (out["trip_km"] / hours.where(hours > 0)).where(out["trip_km"] >= 0)
    out["cold_start"] = out["CoolantTemperatureStart"] < COLD_START_C
    out["short"] = out["trip_km"].between(0, SHORT_KM, inclusive="left")
    out["below_regime"] = out["EngineTemperatureMax"] < REGIME_C
    # El 35% de los "viajes" tiene 0 km (motor encendido sin desplazamiento, mediana 0,8 min).
    # Se separan: `idle` es un evento propio, y las fracciones "_moving" se calculan solo
    # entre viajes con desplazamiento (NaN en los idle, así `mean` los ignora).
    out["idle"] = out["trip_km"].eq(0)
    moving = out["trip_km"].gt(0)
    out["short_moving"] = out["short"].astype(float).where(moving)
    out["below_regime_moving"] = out["below_regime"].astype(float).where(moving)
    out["trip_km_moving"] = out["trip_km"].where(moving)
    out["duration_moving"] = out["trip_duration_min"].where(moving)
    out["urban"] = out["speed_calc"] < URBAN_KMH
    out["regen_drop"] = out["air_regen_delta"] < -REGEN_DROP
    out["saturated_end"] = out["AirRegenerationEnd"] >= SATURATION
    out["filter_abnormal_end"] = out["AirFilterEnd"].notna() & out["AirFilterEnd"].ne(NORMAL_MESSAGE)
    out["stopped_auto_end"] = out["AirFilterEnd"].eq(STOPPED_AUTO)
    out["cleaning_auto_end"] = out["AirFilterEnd"].eq(CLEANING_AUTO)
    out["cleaning_auto_start"] = out["AirFilterStart"].eq(CLEANING_AUTO)
    # Consumo: caída del nivel de combustible por 100 km en viajes sin recarga y con distancia.
    fuel_drop = out["FuelLvlStartPc"] - out["FuelLvlEndPc"]
    valid = (fuel_drop >= 0) & (out["trip_km"] >= 5)
    out["fuel_pct_per_100km"] = (fuel_drop / out["trip_km"] * 100.0).where(valid)

    # Encadenamiento: minutos desde el fin del viaje anterior (orden temporal dentro del vehículo).
    by_time = out.sort_values([ID, "TripDatetimeStart"])
    prev_end = by_time.groupby(ID, observed=True)["TripDatetimeEnd"].shift()
    soak_min = (by_time["TripDatetimeStart"] - prev_end).dt.total_seconds() / 60.0
    out["soak_min"] = soak_min.reindex(out.index)
    out["chained"] = out["soak_min"].between(0, CHAINED_MIN)
    hour = out["TripDatetimeStart"].dt.hour
    out["night"] = hour.lt(6) | hour.ge(22)
    out["weekend"] = out["TripDatetimeStart"].dt.dayofweek.ge(5)
    return out


def enrich_signals(signals: pd.DataFrame, veh: pd.DataFrame) -> pd.DataFrame:
    out = signals
    info = veh.set_index(ID)
    out["event_observed"] = out[ID].map(info["event_observed"]).astype(int)
    out["km_to_event"] = out[ID].map(info["event_odo_km"]) - out["OdometerValue"]
    out["slug"] = out["Message"].map(MESSAGE_SLUGS).fillna("otro")
    return out


# ======================================================================================
# 1 · Factibilidad de (W, G, H, Δ)
# ======================================================================================
def feasibility(veh: pd.DataFrame, trips: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    first = trips.groupby(ID, observed=True)["OdometerTripEnd"].min()
    last = trips.groupby(ID, observed=True)["OdometerTripEnd"].max()
    v = veh.set_index(ID)
    v["first_odo"] = first.reindex(v.index)
    v["last_odo"] = last.reindex(v.index)
    v["pre_event_km"] = v["event_odo_km"] - v["first_odo"]
    v["post_event_km"] = v["last_odo"] - v["event_odo_km"]
    ev = v[v["event_observed"].eq(1)]
    he = v[v["event_observed"].eq(0)]

    resumen = pd.DataFrame({
        "eventos · pre_event_km": ev["pre_event_km"].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]),
        "eventos · post_event_km": ev["post_event_km"].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]),
        "sanos · km observados": (he["last_odo"] - he["first_odo"]).describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]),
    }).round(0)

    rows = []
    for W in (500, 1000, 1500, 2000, 3000):
        for G in (250, 500, 1000):
            for H in (2000, 3000, 5000):
                for step in (250, 500):
                    pre = ev["pre_event_km"].to_numpy()
                    # cortes de un vehículo con evento: [first+W, E-G] cada Δ
                    n_cuts_ev = np.floor(np.clip(pre - W - G, -step, None) / step) + 1
                    n_cuts_ev = np.clip(n_cuts_ev, 0, None)
                    n_pos = np.clip(np.minimum(H, np.clip(pre - W - G, 0, None)) / step + 1, 0, None) * (pre >= W + G)
                    n_pos = np.minimum(n_pos, n_cuts_ev)
                    span_he = (he["last_odo"] - he["first_odo"]).to_numpy()
                    # cortes verificables de un sano: [first+W, last-G-H]
                    n_cuts_he = np.clip(np.floor((span_he - W - G - H) / step) + 1, 0, None)
                    rows.append({
                        "W": W, "G": G, "H": H, "step": step,
                        "eventos_con_algun_corte": int((pre >= W + G).sum()),
                        "eventos_con_positivo": int((pre >= W + G).sum()),
                        "eventos_con_negativo_previo": int((pre > W + G + H).sum()),
                        "sanos_con_corte_verificable": int((span_he >= W + G + H).sum()),
                        "filas_eventos": int(n_cuts_ev.sum()),
                        "filas_positivas": int(n_pos.sum()),
                        "filas_sanos_verificables": int(n_cuts_he.sum()),
                    })
    grid = pd.DataFrame(rows)
    grid["tasa_base_estimada"] = grid["filas_positivas"] / (grid["filas_eventos"] + grid["filas_sanos_verificables"])
    return resumen, grid


# ======================================================================================
# 2 · Perfil alineado al evento
# ======================================================================================
TRIP_PROFILE = {
    "short_trip_frac": ("short", "mean"),
    "idle_frac": ("idle", "mean"),
    "short_moving_frac": ("short_moving", "mean"),
    "below_regime_moving_frac": ("below_regime_moving", "mean"),
    "trip_km_moving_median": ("trip_km_moving", "median"),
    "duration_moving_median": ("duration_moving", "median"),
    "below_regime_frac": ("below_regime", "mean"),
    "cold_start_frac": ("cold_start", "mean"),
    "chained_frac": ("chained", "mean"),
    "urban_frac": ("urban", "mean"),
    "trip_km_median": ("trip_km", "median"),
    "speed_calc_median": ("speed_calc", "median"),
    "engine_temp_avg_median": ("EngineTemperatureAvg", "median"),
    "coolant_end_mean": ("CoolantTemperatureEnd", "mean"),
    "air_temp_avg_mean": ("AirTemperatureAvg", "mean"),
    "fuel_pct_per_100km_median": ("fuel_pct_per_100km", "median"),
    "oil_life_mean": ("EngineOilLifePCStart", "mean"),
    "air_regen_end_mean": ("AirRegenerationEnd", "mean"),
    "saturated_end_frac": ("saturated_end", "mean"),
    "filter_abnormal_end_frac": ("filter_abnormal_end", "mean"),
    "stopped_auto_end_frac": ("stopped_auto_end", "mean"),
    "regen_drop_frac": ("regen_drop", "mean"),
    "night_frac": ("night", "mean"),
    "weekend_frac": ("weekend", "mean"),
}


def _per_vehicle_bin(frame: pd.DataFrame, bin_col: str, specs: dict[str, tuple[str, str]], *, min_rows: int = 20) -> pd.DataFrame:
    grouped = frame.groupby([ID, bin_col], observed=True)
    agg = grouped.agg(n=(bin_col, "size"), **{k: v for k, v in specs.items()})
    agg = agg[agg["n"] >= min_rows]
    # km recorridos en el bin, para tasas por 1000 km
    km = grouped["trip_km"].apply(lambda s: float(s.clip(lower=0).sum())) if "trip_km" in frame.columns else None
    if km is not None:
        agg["km"] = km.reindex(agg.index)
        agg["regen_drops_per_1000km"] = grouped["regen_drop"].sum().reindex(agg.index) / (agg["km"] / 1000.0).where(agg["km"] > 0)
        agg["idle_per_1000km"] = grouped["idle"].sum().reindex(agg.index) / (agg["km"] / 1000.0).where(agg["km"] > 0)
        agg["moving_trips_per_1000km"] = (agg["n"] - grouped["idle"].sum().reindex(agg.index)) / (agg["km"] / 1000.0).where(agg["km"] > 0)
        agg["stopped_auto_per_1000km"] = grouped["stopped_auto_end"].sum().reindex(agg.index) / (agg["km"] / 1000.0).where(agg["km"] > 0)
    return agg.reset_index()


def event_aligned_profile(trips: pd.DataFrame, signals: pd.DataFrame, *, bin_km: float = 1000.0) -> pd.DataFrame:
    ev = trips[trips["event_observed"].eq(1) & trips["km_to_event"].notna()].copy()
    ev["bin"] = (np.floor(ev["km_to_event"] / bin_km) * bin_km).astype(int)
    per_ev = _per_vehicle_bin(ev, "bin", TRIP_PROFILE)

    he = trips[trips["event_observed"].eq(0) & trips["OdometerTripEnd"].notna()].copy()
    he["bin"] = (np.floor(he["OdometerTripEnd"] / bin_km) * bin_km).astype(int)
    per_he = _per_vehicle_bin(he, "bin", TRIP_PROFILE)

    # señales: tasas por 1000 km por (vehículo, bin), normalizadas por km de trips en ese bin
    def signal_rates(frame: pd.DataFrame, axis_col: str, km_table: pd.DataFrame) -> pd.DataFrame:
        f = frame.dropna(subset=[axis_col]).copy()
        f["bin"] = (np.floor(f[axis_col] / bin_km) * bin_km).astype(int)
        counts = f.pivot_table(index=[ID, "bin"], columns="slug", values="Message", aggfunc="size", fill_value=0)
        counts["regen_marker"] = f.groupby([ID, "bin"], observed=True)["Regenerations"].sum().reindex(counts.index).fillna(0)
        dbr = f.groupby([ID, "bin"], observed=True)["DistanceBetweenRegenerations"].median()
        km = km_table.set_index([ID, "bin"])["km"].reindex(counts.index)
        rates = counts.div((km / 1000.0).where(km > 0), axis=0)
        rates.columns = [f"msg_{c}_per_1000km" if not c.startswith("regen") else "regen_marker_per_1000km" for c in rates.columns]
        rates["dist_between_regen_median"] = dbr.reindex(rates.index)
        return rates.reset_index()

    sig_ev = signal_rates(signals[signals["event_observed"].eq(1)], "km_to_event", per_ev)
    sig_he = signal_rates(signals[signals["event_observed"].eq(0)], "OdometerValue", per_he)
    per_ev = per_ev.merge(sig_ev, on=[ID, "bin"], how="left")
    per_he = per_he.merge(sig_he, on=[ID, "bin"], how="left")

    metrics = [c for c in per_ev.columns if c not in {ID, "bin", "n", "km"}]
    prof_ev = per_ev.groupby("bin")[metrics].median()
    prof_ev["n_vehicles"] = per_ev.groupby("bin")[ID].nunique()
    prof_ev = prof_ev[prof_ev["n_vehicles"] >= 10]
    prof_ev.index.name = "km_to_event_bin"
    # referencia sana: mediana entre vehículos por bin de odómetro absoluto, y mediana global
    prof_he = per_he.groupby("bin")[metrics].median()
    prof_he["n_vehicles"] = per_he.groupby("bin")[ID].nunique()
    prof_he = prof_he[prof_he["n_vehicles"] >= 10]
    prof_he.index.name = "odo_bin"
    global_he = per_he.groupby(ID)[metrics].median().median().rename("sanos_mediana_global")
    return prof_ev, prof_he, global_he, per_ev, per_he


def anticipation_ranking(per_ev: pd.DataFrame, per_he: pd.DataFrame) -> pd.DataFrame:
    """Para cada agregado: separación (AUC de Mann-Whitney) sanos vs. eventos, por tramo previo al evento.

    Tramos: `lejos` = 4.000–8.000 km antes, `medio` = 2.000–4.000, `cerca` = 0–2.000,
    `post` = después del evento. El comparador es SIEMPRE el mismo: el agregado por
    vehículo sano sobre bins de odómetro absoluto 2.000–10.000 (donde viven los eventos).
    """
    metrics = [c for c in per_ev.columns if c not in {ID, "bin", "n", "km"}]
    he = per_he[per_he["bin"].between(2000, 10000)].groupby(ID)[metrics].median()
    tramos = {"lejos (4-8k)": (4000, 8000), "medio (2-4k)": (2000, 4000), "cerca (0-2k)": (0, 2000), "post (<0)": (-6000, 0)}
    rows = []
    for metric in metrics:
        row = {"metric": metric}
        for name, (lo, hi) in tramos.items():
            side = per_ev[per_ev["bin"].between(lo, hi - 1)].groupby(ID)[metric].median().dropna()
            ref = he[metric].dropna()
            if len(side) < 8 or len(ref) < 8:
                row[name] = np.nan
                continue
            u = mannwhitneyu(side, ref, alternative="two-sided").statistic
            row[name] = u / (len(side) * len(ref))       # P(evento > sano)
            row[f"n {name}"] = len(side)
        rows.append(row)
    out = pd.DataFrame(rows).set_index("metric")
    return out


# ======================================================================================
# 3 · Post-evento: ¿cambia el uso?
# ======================================================================================
def post_event_change(trips: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
    ev = trips[trips["event_observed"].eq(1) & trips["km_to_event"].notna()].copy()
    ev["fase"] = np.where(ev["km_to_event"] > 0, "pre", "post")
    keep = [ID, "fase", "trip_km", "short", "below_regime", "cold_start", "regen_drop", "saturated_end",
            "filter_abnormal_end", "stopped_auto_end", "fuel_pct_per_100km", "AirRegenerationEnd", "speed_calc"]
    g = ev[keep].groupby([ID, "fase"], observed=True)
    agg = g.agg(n=("trip_km", "size"), km=("trip_km", lambda s: float(s.clip(lower=0).sum())),
                short_trip_frac=("short", "mean"), below_regime_frac=("below_regime", "mean"),
                cold_start_frac=("cold_start", "mean"), saturated_end_frac=("saturated_end", "mean"),
                filter_abnormal_end_frac=("filter_abnormal_end", "mean"),
                stopped_auto_end_frac=("stopped_auto_end", "mean"),
                fuel_pct_per_100km_median=("fuel_pct_per_100km", "median"),
                air_regen_end_mean=("AirRegenerationEnd", "mean"), speed_calc_median=("speed_calc", "median"),
                n_regen_drops=("regen_drop", "sum"))
    agg["regen_drops_per_1000km"] = agg["n_regen_drops"] / (agg["km"] / 1000.0).where(agg["km"] > 0)
    agg = agg[agg["n"] >= 30].reset_index()
    sg = signals[signals["event_observed"].eq(1) & signals["km_to_event"].notna()].copy()
    sg["fase"] = np.where(sg["km_to_event"] > 0, "pre", "post")
    msg = sg.pivot_table(index=[ID, "fase"], columns="slug", values="Message", aggfunc="size", fill_value=0)
    msg["regen_marker"] = sg.groupby([ID, "fase"], observed=True)["Regenerations"].sum()
    msg = msg.reset_index().merge(agg[[ID, "fase", "km"]], on=[ID, "fase"])
    for c in ("full", "overloaded", "over_limit", "at_limit", "cleaning_auto", "stopped_auto", "regen_marker"):
        if c in msg.columns:
            agg = agg.merge(
                (msg[c] / (msg["km"] / 1000.0).where(msg["km"] > 0)).rename(f"{c}_per_1000km").to_frame()
                .assign(**{ID: msg[ID], "fase": msg["fase"]}), on=[ID, "fase"], how="left")
    wide = agg.set_index([ID, "fase"]).drop(columns=["n", "km", "n_regen_drops"]).unstack("fase")
    rows = []
    for metric in wide.columns.get_level_values(0).unique():
        pair = wide[metric].dropna()
        if len(pair) < 10 or "post" not in pair.columns:
            continue
        delta = pair["post"] - pair["pre"]
        rows.append({"metric": metric, "n_vehiculos": len(pair), "pre_mediana": pair["pre"].median(),
                     "post_mediana": pair["post"].median(), "delta_mediana": delta.median(),
                     "frac_que_baja": float((delta < 0).mean())})
    return pd.DataFrame(rows).set_index("metric")


# ======================================================================================
# 4-5 · Agregados nuevos por vehículo (historial completo) y separación por cohorte
# ======================================================================================
def new_vehicle_aggregates(trips: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
    g = trips.groupby(ID, observed=True)
    km_k = (g["trip_km"].apply(lambda s: float(s.clip(lower=0).sum())) / 1000.0)
    out = pd.DataFrame({
        "event_observed": g["event_observed"].max(),
        "cold_start_frac": g["cold_start"].mean(),
        "chained_frac": g["chained"].mean(),
        "soak_min_median": g["soak_min"].median(),
        "night_frac": g["night"].mean(),
        "weekend_frac": g["weekend"].mean(),
        "fuel_pct_per_100km_median": g["fuel_pct_per_100km"].median(),
        "fuel_valid_frac": g["fuel_pct_per_100km"].apply(lambda s: s.notna().mean()),
        "speed_calc_median": g["speed_calc"].median(),
        "urban_calc_frac": g["urban"].mean(),
        "stopped_auto_end_per_1000km": g["stopped_auto_end"].sum() / km_k.where(km_k > 0),
        "cleaning_auto_end_per_1000km": g["cleaning_auto_end"].sum() / km_k.where(km_k > 0),
        "regen_drops_per_1000km": g["regen_drop"].sum() / km_k.where(km_k > 0),
        "coolant_start_median": g["CoolantTemperatureStart"].median(),
        "engine_temp_min_median": g["EngineTemperatureMin"].median(),
    })
    # interrumpidas / (interrumpidas + completadas), con las caídas como "completadas"
    out["interrupted_regen_ratio"] = out["stopped_auto_end_per_1000km"] / (
        out["stopped_auto_end_per_1000km"] + out["regen_drops_per_1000km"]).where(
        (out["stopped_auto_end_per_1000km"] + out["regen_drops_per_1000km"]) > 0)
    # tamaño de la caída y nivel residual tras regenerar
    drops = trips[trips["regen_drop"]]
    gd = drops.groupby(ID, observed=True)
    out["regen_drop_size_mean"] = (-gd["air_regen_delta"].mean()).reindex(out.index)
    out["regen_residual_mean"] = gd["AirRegenerationEnd"].mean().reindex(out.index)
    out["regen_start_level_mean"] = gd["AirRegenerationStart"].mean().reindex(out.index)
    # señales: transiciones cleaning_auto -> stopped_auto en la secuencia de mensajes
    sg = signals.sort_values([ID, "eventTimestamp"])
    nxt = sg.groupby(ID, observed=True)["slug"].shift(-1)
    started = sg["slug"].eq("cleaning_auto")
    trans = pd.DataFrame({ID: sg[ID], "started": started, "to_stopped": started & nxt.eq("stopped_auto"),
                          "to_normal": started & nxt.eq("normal")})
    t = trans.groupby(ID, observed=True).sum()
    out["sig_cleaning_started"] = t["started"].reindex(out.index)
    out["sig_interrupted_frac"] = (t["to_stopped"] / t["started"].where(t["started"] > 0)).reindex(out.index)
    out["sig_stopped_auto_per_1000km"] = (sg.groupby(ID, observed=True)["slug"].apply(lambda s: s.eq("stopped_auto").sum())
                                          / km_k.where(km_k > 0)).reindex(out.index)
    return out.reset_index()


def cohort_separation(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = []
    y = frame["event_observed"]
    for c in columns:
        x = frame[c]
        ok = x.notna()
        if ok.sum() < 50 or x[ok].nunique() < 3:
            continue
        a, b = x[ok & y.eq(1)], x[ok & y.eq(0)]
        u = mannwhitneyu(a, b, alternative="two-sided")
        rows.append({"variable": c, "n": int(ok.sum()), "mediana_sanos": b.median(), "mediana_evento": a.median(),
                     "prob_supera": u.statistic / (len(a) * len(b)), "p_mw": u.pvalue,
                     "rho_spearman": spearmanr(x[ok], y[ok]).statistic, "frac_cero": float((x[ok] == 0).mean())})
    return pd.DataFrame(rows).set_index("variable").sort_values("prob_supera", key=lambda s: (s - 0.5).abs(), ascending=False)


# ======================================================================================
# 6 · Completitud del marcador Regenerations
# ======================================================================================
def regen_marker_completeness(signals: pd.DataFrame, trips: pd.DataFrame) -> pd.DataFrame:
    marked = signals[signals["Regenerations"]]
    last_marker = marked.groupby(ID, observed=True)["OdometerValue"].max()
    n_marker = marked.groupby(ID, observed=True).size()
    last_signal = signals.groupby(ID, observed=True)["OdometerValue"].max()
    last_trip = trips.groupby(ID, observed=True)["OdometerTripEnd"].max()
    first_trip = trips.groupby(ID, observed=True)["OdometerTripEnd"].min()
    drops = trips.groupby(ID, observed=True)["regen_drop"].sum()
    last_drop = trips[trips["regen_drop"]].groupby(ID, observed=True)["OdometerTripEnd"].max()
    ev = trips.groupby(ID, observed=True)["event_observed"].max()
    out = pd.DataFrame({
        "event_observed": ev, "n_marker": n_marker, "n_drops": drops,
        "last_marker_frac_of_history": ((last_marker - first_trip) / (last_trip - first_trip)),
        "last_drop_frac_of_history": ((last_drop - first_trip) / (last_trip - first_trip)),
        "last_signal_minus_last_trip_km": last_signal - last_trip,
    })
    out["n_marker"] = out["n_marker"].fillna(0)
    return out.reset_index()


# ======================================================================================
# 7 · Estabilidad intra-vehículo en ventanas de 2.000 km
# ======================================================================================
def within_vehicle_stability(trips: pd.DataFrame, signals: pd.DataFrame, *, window_km: float = 2000.0) -> pd.DataFrame:
    he = trips[trips["event_observed"].eq(0) & trips["OdometerTripEnd"].notna()].copy()
    he["bin"] = (np.floor(he["OdometerTripEnd"] / window_km)).astype(int)
    per = _per_vehicle_bin(he, "bin", TRIP_PROFILE, min_rows=20)
    sg = signals[signals["event_observed"].eq(0)].dropna(subset=["OdometerValue"]).copy()
    sg["bin"] = (np.floor(sg["OdometerValue"] / window_km)).astype(int)
    counts = sg.pivot_table(index=[ID, "bin"], columns="slug", values="Message", aggfunc="size", fill_value=0)
    counts["regen_marker"] = sg.groupby([ID, "bin"], observed=True)["Regenerations"].sum()
    km = per.set_index([ID, "bin"])["km"].reindex(counts.index)
    rates = counts.div((km / 1000.0).where(km > 0), axis=0)
    rates.columns = [f"msg_{c}_per_1000km" if not c.startswith("regen") else "regen_marker_per_1000km" for c in rates.columns]
    per = per.merge(rates.reset_index(), on=[ID, "bin"], how="left")
    metrics = [c for c in per.columns if c not in {ID, "bin", "n", "km"}]
    rows = []
    for m in metrics:
        d = per[[ID, m]].dropna()
        counts_v = d.groupby(ID).size()
        d = d[d[ID].isin(counts_v[counts_v >= 3].index)]
        if d[ID].nunique() < 30:
            continue
        grand = d[m].mean()
        means = d.groupby(ID)[m].mean()
        n_i = d.groupby(ID).size()
        ss_between = float((n_i * (means - grand) ** 2).sum())
        ss_within = float(((d[m] - d[ID].map(means)) ** 2).sum())
        k = d[ID].nunique()
        n = len(d)
        ms_b = ss_between / (k - 1)
        ms_w = ss_within / (n - k)
        n0 = (n - (n_i ** 2).sum() / n) / (k - 1)
        icc = (ms_b - ms_w) / (ms_b + (n0 - 1) * ms_w)
        rows.append({"metric": m, "vehiculos": k, "ventanas": n, "icc_vehiculo": icc,
                     "cv_intra_mediano": float((d.groupby(ID)[m].std() / d.groupby(ID)[m].mean().abs()).median())})
    return pd.DataFrame(rows).set_index("metric").sort_values("icc_vehiculo", ascending=False)


# ======================================================================================
# 8 · Chequeos de dato que faltaban
# ======================================================================================
def data_checks(veh: pd.DataFrame, trips: pd.DataFrame, signals: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    # TripNumber: ¿secuencial por vehículo?
    by_time = trips.sort_values([ID, "TripDatetimeStart"])
    tn = by_time.groupby(ID, observed=True)["TripNumber"]
    out["tripnumber_min_median"] = float(tn.min().median())
    out["tripnumber_is_monotone_by_time_frac"] = float(tn.apply(lambda s: s.dropna().is_monotonic_increasing).mean())
    out["tripnumber_unique_frac"] = float(tn.apply(lambda s: s.dropna().is_unique).mean())
    diffs = tn.diff().dropna()
    out["tripnumber_diff_eq_1_frac"] = float(diffs.eq(1).mean())
    out["tripnumber_gap_gt_1_frac"] = float(diffs.gt(1).mean())
    out["tripnumber_null_frac"] = float(trips["TripNumber"].isna().mean())

    # Monotonía del odómetro: DEPENDE del orden con que se recorran los viajes. El 0,7%
    # de los viajes comparte `TripDatetimeStart` con otro del mismo vehículo, y ordenar
    # por fecha deja esos empates barajados: el "retroceso" que aparece es el desempate,
    # no el odómetro. Se miden los tres órdenes para que la diferencia quede a la vista.
    for label, keys in (("by_time", [ID, "TripDatetimeStart"]),
                        ("by_tripnumber", [ID, "TripNumber"]),
                        ("by_odo", [ID, "OdometerTripStart", "OdometerTripEnd"])):
        d = trips.sort_values(keys)
        back = d["OdometerTripStart"] - d.groupby(ID, observed=True)["OdometerTripEnd"].shift()
        out[f"odo_backwards_gt_1km_trips__{label}"] = int((back < -1).sum())
        out[f"odo_backwards_gt_1km_vehicles__{label}"] = int(d.loc[back < -1, ID].nunique())
        out[f"odo_backwards_gt_500km_trips__{label}"] = int((back < -500).sum())
        out[f"odo_backwards_gt_500km_vehicles__{label}"] = int(d.loc[back < -500, ID].nunique())
        # Costo de forzar monotonía con un máximo acumulado sobre ese mismo orden.
        delta = d.groupby(ID, observed=True)["OdometerTripEnd"].cummax() - d["OdometerTripEnd"]
        out[f"cummax_rows_fixed__{label}"] = int((delta > 0.5).sum())
        out[f"cummax_vehicles_fixed__{label}"] = int(d.loc[delta > 0.5, ID].nunique())
        out[f"cummax_km_fixed_median__{label}"] = (
            float(delta[delta > 0.5].median()) if (delta > 0.5).any() else 0.0)

    # Empates de timestamp: la causa del desorden aparente.
    dup_ts = trips.duplicated([ID, "TripDatetimeStart"], keep=False)
    out["trips_sharing_timestamp_frac"] = float(dup_ts.mean())
    out["vehicles_with_timestamp_tie"] = int(trips.loc[dup_ts, ID].nunique())

    # Con orden por TripNumber, ¿el reloj retrocede alguna vez? ¿se solapan los viajes?
    by_tn = trips.sort_values([ID, "TripNumber"])
    dt_h = (by_tn["TripDatetimeStart"] - by_tn.groupby(ID, observed=True)["TripDatetimeStart"].shift()
            ).dt.total_seconds() / 3600.0
    out["time_backwards_trips__by_tripnumber"] = int((dt_h < 0).sum())
    gap_h = (by_tn["TripDatetimeStart"] - by_tn.groupby(ID, observed=True)["TripDatetimeEnd"].shift()
             ).dt.total_seconds() / 3600.0
    out["overlapping_trips__by_tripnumber"] = int((gap_h < 0).sum())

    # Coherencia odómetro <-> reloj: velocidad implícita entre viajes consecutivos.
    step_km = by_tn["OdometerTripStart"] - by_tn.groupby(ID, observed=True)["OdometerTripEnd"].shift()
    ok_pair = step_km.gt(1) & gap_h.gt(0.05)
    implied = step_km[ok_pair] / gap_h[ok_pair]
    out["implied_speed_between_trips_max"] = float(implied.max())
    out["implied_speed_between_trips_p999"] = float(implied.quantile(0.999))
    out["implied_speed_between_trips_gt200_trips"] = int((implied > 200).sum())

    # Vehículos con el odómetro realmente corrupto: los que retroceden con el orden bueno.
    bad = sorted(by_tn.loc[step_km < -1, ID].unique().tolist())
    out["odo_corrupt_vehicles"] = bad
    out["odo_corrupt_vehicles_frac"] = float(len(bad) / trips[ID].nunique())
    out["odo_backwards_gt_1km_trips"] = int((step_km < -1).sum())
    out["odo_backwards_gt_1km_vehicles"] = len(bad)
    out["odo_backwards_gt_100km_trips"] = int((step_km < -100).sum())
    out["trip_km_negative_trips"] = int((trips["trip_km"] < 0).sum())
    out["trip_km_negative_vehicles"] = int(trips.loc[trips["trip_km"] < 0, ID].nunique())
    out["trip_km_zero_frac"] = float(trips["trip_km"].eq(0).mean())
    out["trips_datetime_null_frac"] = float(trips["TripDatetimeStart"].isna().mean())
    out["trips_duration_le0_frac"] = float(trips["trip_duration_min"].le(0).mean())

    # KilometerPerHour nulo: ¿en qué viajes?
    null_speed = trips["KilometerPerHour"].isna()
    out["speed_null_frac"] = float(null_speed.mean())
    out["speed_null_trip_km_median"] = float(trips.loc[null_speed, "trip_km"].median())
    out["speed_notnull_trip_km_median"] = float(trips.loc[~null_speed, "trip_km"].median())
    out["speed_null_when_km_zero_frac"] = float(null_speed[trips["trip_km"].eq(0)].mean())
    out["speed_null_when_km_pos_frac"] = float(null_speed[trips["trip_km"].gt(0)].mean())
    out["speed_null_when_duration_zero_frac"] = float(null_speed[trips["trip_duration_min"].le(0)].mean())
    both = trips[["KilometerPerHour", "speed_calc"]].dropna()
    both = both[both["KilometerPerHour"].lt(200)]
    out["speed_reported_vs_calc_pearson"] = float(both.corr().iloc[0, 1])
    out["speed_reported_vs_calc_median_ratio"] = float((both["KilometerPerHour"] / both["speed_calc"].where(both["speed_calc"] > 0)).median())

    # Cobertura signals vs trips (dev)
    st = signals.groupby(ID, observed=True).agg(sig_odo_max=("OdometerValue", "max"), sig_date_max=("eventTimestamp", "max"),
                                                 sig_date_min=("eventTimestamp", "min"), n=("OdometerValue", "size"))
    tt = trips.groupby(ID, observed=True).agg(trip_odo_max=("OdometerTripEnd", "max"), trip_date_max=("TripDatetimeStart", "max"),
                                               trip_date_min=("TripDatetimeStart", "min"))
    cov = st.join(tt)
    d_km = cov["sig_odo_max"] - cov["trip_odo_max"]
    d_days = (cov["sig_date_max"] - cov["trip_date_max"]).dt.total_seconds() / 86400
    out["signals_minus_trips_odo_max_km"] = d_km.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).round(0).to_dict()
    out["signals_minus_trips_date_max_days"] = d_days.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).round(1).to_dict()
    out["vehicles_signals_end_gt_30d_before_trips"] = int((d_days < -30).sum())
    out["signals_per_trip_median"] = float((cov["n"] / trips.groupby(ID, observed=True).size()).median())

    # Estáticas nulas en dev
    out["static_daysUntilSale_null"] = int(veh["static_daysUntilSale"].isna().sum())
    out["static_ProductionDay_null"] = int(veh["static_ProductionDay"].isna().sum())

    # Nivel de AirRegeneration: ¿positivo dentro del viaje = carga?
    delta = trips["air_regen_delta"].dropna()
    out["air_regen_delta_frac_pos"] = float((delta > 0).mean())
    out["air_regen_delta_frac_zero"] = float((delta == 0).mean())
    out["air_regen_delta_frac_neg"] = float((delta < 0).mean())
    out["air_regen_delta_frac_drop_gt5"] = float((delta < -5).mean())
    out["air_regen_levels"] = sorted(pd.unique(trips["AirRegenerationEnd"].dropna()).astype(int).tolist())

    # ¿Cuántos viajes hay en 1.000 km?
    km_per_vehicle = trips.groupby(ID, observed=True)["trip_km"].apply(lambda s: float(s.clip(lower=0).sum()))
    n_trips = trips.groupby(ID, observed=True).size()
    out["trips_per_1000km_median"] = float((n_trips / (km_per_vehicle / 1000)).median())
    out["trips_per_1000km_p10"] = float((n_trips / (km_per_vehicle / 1000)).quantile(0.1))
    # Frecuencia de mensajes por 1000 km (para dimensionar la ventana)
    n_sig = signals.groupby(ID, observed=True).size()
    out["signals_per_1000km_median"] = float((n_sig / (km_per_vehicle / 1000)).median())
    return out


# ======================================================================================
# 9 · Calendario: cortes de registro y estacionalidad
# ======================================================================================
def calendar_checks(veh: pd.DataFrame, trips: pd.DataFrame, signals: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Qué se registra en cada mes calendario. Un canal que se corta en una fecha es una
    feature que mide el calendario, y el calendario correlaciona con la etiqueta vía
    `ProductionDay` (los vehículos con evento se produjeron antes)."""
    out: dict[str, pd.DataFrame] = {}
    sg = signals.dropna(subset=["eventTimestamp"])
    month = sg["eventTimestamp"].dt.tz_localize(None).dt.to_period("M")
    by_slug = sg.assign(_m=month).pivot_table(index="_m", columns="slug", values="Message", aggfunc="size", fill_value=0)
    by_slug["regen_marker"] = sg.assign(_m=month).groupby("_m")["Regenerations"].sum()
    tr = trips.dropna(subset=["TripDatetimeStart"])
    tmonth = tr["TripDatetimeStart"].dt.tz_localize(None).dt.to_period("M")
    by_slug["regen_drops_trips"] = tr.assign(_m=tmonth).groupby("_m")["regen_drop"].sum()
    by_slug["vehiculos_activos"] = tr.assign(_m=tmonth).groupby("_m")[ID].nunique()
    by_slug["km_flota"] = tr.assign(_m=tmonth).groupby("_m")["trip_km"].apply(lambda s: float(s.clip(lower=0).sum())).round(0)
    for c in ("full", "overloaded", "over_limit", "cleaning_auto", "regen_marker", "regen_drops_trips"):
        if c in by_slug.columns:
            by_slug[f"{c}_per_1000km"] = (by_slug[c] / (by_slug["km_flota"] / 1000.0).where(by_slug["km_flota"] > 0)).round(2)
    out["por_mes"] = by_slug

    mk = sg[sg["Regenerations"]]
    last_mk = mk.groupby(ID, observed=True)["eventTimestamp"].max()
    last_tr = tr.groupby(ID, observed=True)["TripDatetimeStart"].max()
    d = pd.DataFrame({"event_observed": veh.set_index(ID)["event_observed"], "last_marker": last_mk, "last_trip": last_tr})
    d["dias_sin_marcador_al_final"] = (d["last_trip"] - d["last_marker"]).dt.total_seconds() / 86400
    out["ultimo_marcador"] = d.groupby("event_observed").agg(
        n=("last_marker", "size"), sin_marcador=("last_marker", lambda s: int(s.isna().sum())),
        ultimo_marcador_p50=("last_marker", lambda s: s.quantile(0.5)), ultimo_marcador_p90=("last_marker", lambda s: s.quantile(0.9)),
        fin_trips_p50=("last_trip", "median"), dias_sin_marcador_p50=("dias_sin_marcador_al_final", "median"))

    ev = veh[veh["event_observed"].eq(1)].copy()
    ev_month = pd.to_datetime(ev["event_date"]).dt.tz_localize(None).dt.to_period("M")
    he_km = tr[tr["event_observed"].eq(0)].assign(_m=tmonth).groupby("_m")["trip_km"].apply(lambda s: float(s.clip(lower=0).sum()))
    ev_km = tr[tr["event_observed"].eq(1) & tr["km_to_event"].gt(0)].assign(_m=tmonth).groupby("_m")["trip_km"].apply(lambda s: float(s.clip(lower=0).sum()))
    cal = pd.DataFrame({"eventos": ev_month.value_counts().sort_index(), "km_sanos": he_km.round(0), "km_evento_pre": ev_km.round(0)}).fillna(0)
    cal["frac_km_sanos"] = (cal["km_sanos"] / cal["km_sanos"].sum()).round(3)
    cal["frac_km_evento_pre"] = (cal["km_evento_pre"] / cal["km_evento_pre"].sum()).round(3)
    cal["air_temp_mediana"] = tr.assign(_m=tmonth).groupby("_m")["AirTemperatureAvg"].median().round(1)
    out["eventos_y_exposicion_por_mes"] = cal
    return out


# ======================================================================================
# 10 · Cadencia: qué trae una ventana de calendario frente a la ventana de km
# ======================================================================================
def cadence_feasibility(trips: pd.DataFrame, *, day_windows: tuple[int, ...] = (7, 15, 30),
                        min_trips: int = 5, min_km: float = 500.0) -> dict[str, pd.DataFrame]:
    """Cuánto historial junta un vehículo en N días, y cuántas de esas ventanas pasan el QC.

    El panel corta sobre odómetro (regla 4) y el QC pide `min_trips_in_window` viajes y
    una fracción de W en km recorridos. Emitir predicciones cada N días es una decisión
    de **cadencia**, distinta de la **ventana de agregación**: esta función mide si una
    ventana de calendario podría además reemplazar a la de km, y la respuesta depende de
    cuán disparejo sea el uso entre vehículos.
    """
    tr = trips.dropna(subset=["TripDatetimeStart"]).copy()
    tr["_moving"] = tr["trip_km"].gt(0)

    per_v = tr.groupby(ID, observed=True).agg(
        odo_min=("OdometerTripStart", "min"), odo_max=("OdometerTripEnd", "max"),
        d0=("TripDatetimeStart", "min"), d1=("TripDatetimeEnd", "max"), n_trips=("trip_km", "size"))
    per_v["span_km"] = per_v["odo_max"] - per_v["odo_min"]
    per_v["days"] = (per_v["d1"] - per_v["d0"]).dt.total_seconds() / 86400.0
    per_v["km_per_day"] = per_v["span_km"] / per_v["days"].clip(lower=1)

    rows = []
    for n_days in day_windows:
        bucket = tr.groupby(ID, observed=True)["TripDatetimeStart"].transform(
            lambda s: (s - s.min()).dt.days // n_days)
        w = tr.assign(_b=bucket).groupby([ID, "_b"]).agg(
            n_trips=("trip_km", "size"), n_moving=("_moving", "sum"),
            km_hi=("OdometerTripEnd", "max"), km_lo=("OdometerTripStart", "min"))
        w["km"] = w["km_hi"] - w["km_lo"]
        # Cobertura: ventanas con datos sobre ventanas que abarca la vida del vehículo.
        span_windows = (per_v["days"] / n_days).apply(lambda v: max(1.0, np.ceil(v)))
        cov = w.groupby(ID).size() / span_windows
        km = per_v["km_per_day"] * n_days
        rows.append({
            "dias": n_days,
            "km_p10": km.quantile(0.10), "km_p50": km.quantile(0.50), "km_p90": km.quantile(0.90),
            "ratio_km_p90_p10": km.quantile(0.90) / max(km.quantile(0.10), 1e-9),
            "ventanas": float(len(w)),
            "ventanas_por_vehiculo_p50": float(w.groupby(ID).size().median()),
            "frac_sin_min_viajes_moviles": float(w["n_moving"].lt(min_trips).mean()),
            "frac_bajo_min_km": float(w["km"].lt(min_km).mean()),
            "cobertura_p10": float(cov.quantile(0.10)), "cobertura_p50": float(cov.quantile(0.50)),
        })
    resumen = pd.DataFrame(rows).set_index("dias")
    # Referencia: cuántos cortes deja la grilla de km del panel v1.
    resumen.loc["grilla_500km", "ventanas_por_vehiculo_p50"] = float((per_v["span_km"] / 500).median())

    # Huecos de registro, en días (un hueco no es un vehículo detenido: es dato que falta).
    by_tn = tr.sort_values([ID, "TripNumber"])
    gap_d = by_tn.groupby(ID, observed=True)["TripDatetimeStart"].diff().dt.days
    per_gap = by_tn.assign(_g=gap_d).groupby(ID, observed=True)["_g"].max()
    huecos = per_gap.describe(percentiles=[0.5, 0.75, 0.9, 0.99]).to_frame("hueco_max_dias")
    huecos.loc["veh_con_hueco_gt_15d", "hueco_max_dias"] = float((per_gap > 15).sum())
    huecos.loc["veh_con_hueco_gt_30d", "hueco_max_dias"] = float((per_gap > 30).sum())

    return {"resumen": resumen, "huecos": huecos, "por_vehiculo": per_v}


# ======================================================================================
def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/data/eda_cache.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg["seed"]))
    out_dir = ensure_dir(str(cfg["output"]["dir"]).rstrip("/") + "/gaps")

    veh, trips, signals = load_dev(cfg, out_dir)
    trips = enrich_trips(trips, veh)
    signals = enrich_signals(signals, veh)
    print(f"dev: {len(veh)} vehículos · {len(trips):,} viajes · {len(signals):,} señales")

    # 1
    resumen, grid = feasibility(veh, trips)
    _print("1 · km antes y después del evento, y km observados de los sanos", resumen)
    best = grid[grid["step"].eq(500)].sort_values(["W", "G", "H"])
    _print("1 · factibilidad de (W, G, H) con Δ = 500 km (dev)", best, index=False, float_format=lambda v: f"{v:.3f}")
    grid.to_csv(out_dir / "feasibility_grid.csv", index=False)

    # 2
    prof_ev, prof_he, global_he, per_ev, per_he = event_aligned_profile(trips, signals)
    cols = ["n_vehicles", "short_trip_frac", "idle_frac", "idle_per_1000km", "short_moving_frac", "below_regime_frac",
            "below_regime_moving_frac", "trip_km_moving_median", "duration_moving_median", "cold_start_frac", "chained_frac", "trip_km_median",
            "fuel_pct_per_100km_median", "air_regen_end_mean", "saturated_end_frac", "filter_abnormal_end_frac",
            "stopped_auto_end_frac", "regen_drops_per_1000km", "stopped_auto_per_1000km",
            "msg_full_per_1000km", "msg_overloaded_per_1000km", "msg_over_limit_per_1000km", "msg_cleaning_auto_per_1000km",
            "msg_stopped_auto_per_1000km", "regen_marker_per_1000km", "dist_between_regen_median"]
    cols = [c for c in cols if c in prof_ev.columns]
    _print("2 · perfil alineado al evento (mediana entre vehículos con evento; bin = km hasta el evento, negativo = después)",
           prof_ev[cols].sort_index(ascending=False), float_format=lambda v: f"{v:.3f}")
    _print("2 · referencia sanos por bin de odómetro absoluto", prof_he[cols].head(12), float_format=lambda v: f"{v:.3f}")
    _print("2 · referencia sanos · mediana global por vehículo", global_he[[c for c in cols if c in global_he.index]].round(3))
    ranking = anticipation_ranking(per_ev, per_he)
    _print("2 · P(vehículo con evento > sano) por tramo previo al evento (0,5 = sin diferencia)",
           ranking, float_format=lambda v: f"{v:.3f}")
    prof_ev.to_csv(out_dir / "event_aligned_profile.csv")
    prof_he.to_csv(out_dir / "healthy_odo_profile.csv")
    ranking.to_csv(out_dir / "anticipation_ranking.csv")

    # 3
    post = post_event_change(trips, signals)
    _print("3 · antes vs. después del evento (mismo vehículo)", post, float_format=lambda v: f"{v:.3f}")
    post.to_csv(out_dir / "post_event_change.csv")

    # 4-5
    agg = new_vehicle_aggregates(trips, signals)
    sep = cohort_separation(agg, [c for c in agg.columns if c not in {ID, "event_observed"}])
    _print("4-5 · agregados nuevos por vehículo (historial completo): separación por cohorte", sep,
           float_format=lambda v: f"{v:.3f}")
    agg.to_csv(out_dir / "new_vehicle_aggregates.csv", index=False)
    sep.to_csv(out_dir / "new_aggregates_separation.csv")

    # 6
    comp = regen_marker_completeness(signals, trips)
    _print("6 · completitud del marcador Regenerations (por cohorte)",
           comp.groupby("event_observed")[["n_marker", "n_drops", "last_marker_frac_of_history",
                                           "last_drop_frac_of_history", "last_signal_minus_last_trip_km"]]
           .describe(percentiles=[0.1, 0.5, 0.9]).T, float_format=lambda v: f"{v:.2f}")
    _print("6 · vehículos cuyo último marcador cae antes del 80% del historial",
           comp.assign(early=comp["last_marker_frac_of_history"].lt(0.8) | comp["n_marker"].eq(0))
           .groupby("event_observed")["early"].agg(["sum", "size", "mean"]))
    comp.to_csv(out_dir / "regen_marker_completeness.csv", index=False)

    # 7
    stab = within_vehicle_stability(trips, signals)
    _print("7 · estabilidad intra-vehículo en ventanas de 2.000 km (sanos): ICC del vehículo", stab,
           float_format=lambda v: f"{v:.3f}")
    stab.to_csv(out_dir / "within_vehicle_stability.csv")

    # 9
    cal = calendar_checks(veh, trips, signals)
    _print("9 · registro por mes calendario (flota dev)",
           cal["por_mes"][[c for c in cal["por_mes"].columns if c.endswith("per_1000km") or c in ("vehiculos_activos", "km_flota")]])
    _print("9 · último marcador `Regenerations` por cohorte", cal["ultimo_marcador"])
    _print("9 · eventos por mes vs. exposición (km) de sanos y de eventos pre-evento", cal["eventos_y_exposicion_por_mes"])
    for name, frame in cal.items():
        frame.to_csv(out_dir / f"calendar_{name}.csv")

    # 10
    cad = cadence_feasibility(trips)
    _print("10 · cadencia: qué junta una ventana de calendario (dev) vs. la grilla de 500 km",
           cad["resumen"], float_format=lambda v: f"{v:.3f}")
    _print("10 · huecos de registro por vehículo (días sin ningún viaje)", cad["huecos"],
           float_format=lambda v: f"{v:.1f}")
    for name in ("resumen", "huecos"):
        cad[name].to_csv(out_dir / f"cadence_{name}.csv")

    # 8
    checks = data_checks(veh, trips, signals)
    _print("8 · chequeos de dato", json.dumps(checks, indent=2, ensure_ascii=False, default=str))
    (out_dir / "data_checks.json").write_text(json.dumps(checks, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\nEscrito en {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
