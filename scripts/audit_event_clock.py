#!/usr/bin/env python
"""En qué reloj ocurre el evento y en qué ventana de calendario se registra (solo dev).

    python scripts/audit_event_clock.py --config configs/data/event_clock.yaml [--refresh]

Es la Fase 1 del cure model: convierte en código del repo la evidencia que motivó el
panel de hitos post-venta. No entrena nada. Bloques (lectura en
`docs/reproducibilidad.md`):

- **H1 · escala.** Dispersión del momento del evento en cada escala de uso, entre los
  fallados con telemetría desde casi 0 km. La escala buena es la que la minimiza
  (Kordonsky & Gertsbakh 1993; Duchesne & Lawless 2000).
- **H2 · dosis-respuesta.** El uso de los primeros km contra km y días al evento.
- **H3 · ventana del registro.** Autos en riesgo y eventos por mes calendario; verifica
  que ningún evento de dev caiga fuera de `event_window`.
- **H4 · riesgo** por 100 autos-mes dentro de la ventana, por edad y por días post-venta.
- **H4b · origen del reloj.** Poisson sobre los autos-día post-venta dentro de la ventana:
  ¿explica el riesgo la edad, los días post-venta o el calendario? La regresión ingenua
  entre fallados se reporta al lado para mostrar por qué no sirve (la ventana trunca).
- **H5 · trayectoria.** Exceso de idle/frío de los fallados contra los sanos del mismo
  mercado × mes × etapa, por distancia al evento.
- **H6 · rasgo temprano.** AUC por vehículo del desvío contra la flota con los primeros
  días post-venta, el confusor de producción, el nulo estratificado, la detección
  alertando al 5% de los sanos y cuánto de esa cola son autos quietos (casi sin km).
- **H9 · panel v1.** Qué parte de los horizontes sanos cae fuera de la ventana.
- **Factibilidad** del panel de hitos: conteos por hito con la exposición en ventana.

Fechas (`src/data/anchor.py`, origen congelado en `panel_meta.json`):
producción = origen + ProductionDay; venta = producción + daysUntilSale;
evento = producción + IdentificationDate. Así la edad al evento es exactamente
`IdentificationDate` y los días post-venta al evento, `IdentificationDate − daysUntilSale`.

Solo mira `dev_vehicles` del holdout, y falla si le llega un viaje de otro vehículo.
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
from scipy import optimize, stats
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.config import ensure_dir, load_config, resolve_path  # noqa: E402
from src.data.anchor import EPOCH, estimate_origin_day, project_dates_to_odometer  # noqa: E402
from src.data.join import load_vehicle_static  # noqa: E402
from src.data.subset import read_table_for_vehicles  # noqa: E402
from src.eval.splits import load_test_split, test_split_masks  # noqa: E402
from src.features.trips import derive_trip_columns  # noqa: E402

logger = logging.getLogger("audit_event_clock")

ID = "vehicle_id"
DAY = pd.Timedelta(days=1)
TRIP_COLUMNS = [
    "TripNumber", "TripDatetimeStart", "TripDatetimeEnd", "OdometerTripStart", "OdometerTripEnd",
    "FuelLvlStartPc", "FuelLvlEndPc", "EngineOilLifePCStart", "EngineTemperatureMin", "EngineTemperatureMax",
    "EngineTemperatureAvg", "AirFilterStart", "AirFilterEnd", "AirRegenerationStart", "AirRegenerationEnd",
    "CoolantTemperatureStart", "CoolantTemperatureEnd", "AirTemperatureMin", "AirTemperatureAvg",
]
KEEP = [ID, "TripNumber", "TripDatetimeStart", "OdometerTripStart", "OdometerTripEnd", "trip_km",
        "trip_duration_min", "idle", "moving", "below_regime", "below_regime_moving"]


# --- entrada ------------------------------------------------------------------------

def load_trips(cfg: dict[str, Any], dev: set[str], *, refresh: bool) -> pd.DataFrame:
    """Viajes de dev con las derivadas de `derive_trip_columns`, en orden de `TripNumber`."""
    cache = ensure_dir(cfg["output_dir"]) / cfg["trips_cache"]
    if cache.exists() and not refresh:
        trips = pd.read_parquet(cache)
        logger.info("viajes de dev desde el cache %s (%d filas)", cache, len(trips))
    else:
        raw, _ = read_table_for_vehicles("trips", None, dev, sources=cfg["sources"], dedupe=cfg["dedupe"])
        raw = raw[[c for c in raw.columns if c in {ID, *TRIP_COLUMNS}]]
        spec = load_config(cfg["features_spec"])
        trips, _ = derive_trip_columns(raw, thresholds=spec.get("thresholds"), clip=spec.get("clip"))
        trips = trips[KEEP]
        trips.to_parquet(cache, index=False)
    foreign = set(trips[ID].astype(str).unique()) - dev
    if foreign:
        raise ValueError(f"{len(foreign)} vehículo(s) fuera de dev en los viajes: el script solo mira dev")
    # Orden canónico de los viajes (docs de la rama feat/zself-cusum: f3-cadencia-y-ventana).
    return trips.sort_values([ID, "TripNumber"], kind="stable", ignore_index=True)


def build_vehicles(cfg: dict[str, Any], dev: set[str], trips: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Una fila por vehículo de dev con las fechas del reloj y el odómetro del evento."""
    static = load_vehicle_static(config_path=cfg["sources"], dedupe_config=cfg["dedupe"],
                                 panel_config=cfg["panel_config"])
    static = static[static[ID].astype(str).isin(dev)].set_index(ID)
    missing = dev - set(static.index.astype(str))
    if missing:
        raise ValueError(f"{len(missing)} vehículo(s) de dev sin fila estática")

    meta = json.loads(resolve_path(cfg["anchor_meta"]).read_text(encoding="utf-8"))
    origin = float(meta["anchor"]["origin_day_since_epoch"])
    first = trips.groupby(ID, observed=True)["TripDatetimeStart"].min()
    check, anchor_stats = estimate_origin_day(first.reindex(static.index), static["static_ProductionDay"])
    if abs(check - origin) > 0.5:
        raise ValueError(f"El origen re-estimado sobre dev ({check}) no coincide con el congelado ({origin})")

    v = pd.DataFrame(index=static.index)
    v["event_observed"] = static["event_observed"].astype(int)
    v["event_age"] = pd.to_numeric(static["event_day_since_production"], errors="coerce").where(v["event_observed"].eq(1))
    v["production_day"] = pd.to_numeric(static["static_ProductionDay"], errors="coerce")
    v["days_until_sale"] = pd.to_numeric(static["static_daysUntilSale"], errors="coerce")
    v["market"] = static["static_SalesCountry_cd"].astype(str)
    v["prod_date"] = EPOCH + pd.to_timedelta(origin + v["production_day"], unit="D")
    v["sale_date"] = v["prod_date"] + pd.to_timedelta(v["days_until_sale"], unit="D")
    v["event_date"] = v["prod_date"] + pd.to_timedelta(v["event_age"], unit="D")
    v["event_dss"] = v["event_age"] - v["days_until_sale"]
    g = trips.groupby(ID, observed=True)
    v["first_trip"] = g["TripDatetimeStart"].min()
    v["last_trip"] = g["TripDatetimeStart"].max()
    v["odo_min"] = g["OdometerTripStart"].min().astype(float)
    v["span_days"] = (v["last_trip"] - v["first_trip"]) / DAY
    events = project_dates_to_odometer(v["event_date"].where(v["event_observed"].eq(1)), trips)
    v["event_odo_km"] = events["event_odo_km"]
    v["event_after_last_trip"] = events["event_odo_source"].eq("posterior_al_ultimo_viaje").reindex(v.index, fill_value=False)
    for column in ("prod_date", "sale_date", "event_date", "first_trip", "last_trip"):
        v[column] = v[column].dt.tz_convert(None)
    info = {"origin_day_since_epoch": origin, "origin_reestimated": check, **anchor_stats,
            "first_trip_minus_production_days_iqr": anchor_stats["iqr_days"]}
    return v, info


def annotate_trips(trips: pd.DataFrame, v: pd.DataFrame) -> pd.DataFrame:
    """Cada viaje con su edad, días post-venta, mes y si es anterior al evento."""
    t = trips.copy()
    t["date"] = t["TripDatetimeStart"].dt.tz_convert(None)
    t = t.join(v[["event_observed", "prod_date", "sale_date", "event_date", "market"]], on=ID)
    t["age"] = (t["date"] - t["prod_date"]) / DAY
    t["dss"] = (t["date"] - t["sale_date"]) / DAY
    t["post_sale"] = t["dss"].gt(0)
    t["month"] = t["date"].dt.to_period("M")
    t["before_event"] = t["event_date"].isna() | (t["date"] < t["event_date"])
    t["idle_f"] = t["idle"].astype(float)
    t["below_mov"] = t["below_regime_moving"].astype(float)
    t["idle_min"] = t["trip_duration_min"].where(t["idle"], 0.0)
    t["OdometerTripEnd"] = t["OdometerTripEnd"].astype("float64")
    return t


# --- utilidades -----------------------------------------------------------------------

def dispersion(x: pd.Series) -> dict[str, float]:
    x = np.asarray(x, float)
    x = x[np.isfinite(x) & (x > 0)]
    q1, q2, q3 = np.quantile(x, [0.25, 0.5, 0.75])
    return {"n": int(len(x)), "sd_log": float(np.std(np.log(x), ddof=1)),
            "cv": float(np.std(x, ddof=1) / np.mean(x)), "qcd": float((q3 - q1) / q2), "median": float(q2)}


def auc_ci(y: np.ndarray, s: np.ndarray, rng: np.random.Generator, n_boot: int) -> tuple[float, float, float]:
    ok = np.isfinite(s)
    y, s = y[ok], s[ok]
    auc = roc_auc_score(y, s)
    boot = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            boot.append(roc_auc_score(y[i], s[i]))
    return float(auc), float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def zsum(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    """Pesos unitarios: suma de las columnas estandarizadas sobre la muestra."""
    return sum((frame[c] - frame[c].mean()) / frame[c].std() for c in columns)


# --- H1 · escala --------------------------------------------------------------------------

SCALES = {
    "odometer": "odómetro", "km": "km recorridos", "engine_h": "horas de motor", "n_trips": "viajes",
    "idle_n": "cantidad de idle", "idle_min": "minutos de idle", "cold_n": "viajes bajo régimen",
    "cold_min": "minutos bajo régimen", "cold_km": "km bajo régimen",
    "age_days": "edad (días desde producción)", "dss_days": "días desde la venta",
}


def h1_scales(v: pd.DataFrame, t: pd.DataFrame, cfg: dict[str, Any], rng: np.random.Generator):
    c = cfg["scales"]
    b = t.loc[t["before_event"]].copy()
    b["cold_n"] = b["below_regime"].astype(float)
    b["cold_km"] = b["trip_km"].where(b["below_regime"] & b["moving"], 0.0)
    b["cold_min"] = b["trip_duration_min"].where(b["below_regime"], 0.0)
    agg = b.groupby(ID, observed=True).agg(
        km=("trip_km", "sum"), engine_h=("trip_duration_min", lambda s: s.sum() / 60), n_trips=("trip_km", "size"),
        idle_n=("idle_f", "sum"), idle_min=("idle_min", "sum"), cold_n=("cold_n", "sum"),
        cold_min=("cold_min", "sum"), cold_km=("cold_km", "sum"))
    failed = v.loc[v["event_observed"].eq(1)].join(agg, how="inner")
    failed = failed.loc[(failed["odo_min"] < c["max_first_trip_odo_km"]) & (failed["km"] > c["min_km_before_event"])].copy()
    failed["odometer"] = failed["event_odo_km"]
    failed["age_days"] = failed["event_age"]
    failed["dss_days"] = failed["event_dss"]

    table = pd.DataFrame({k: dispersion(failed[k]) for k in SCALES}).T
    table.insert(0, "escala", [SCALES[k] for k in table.index])
    table = table.sort_values("sd_log")

    boot_rows = []
    for scale in [k for k in SCALES if k not in ("odometer", "km")]:
        pair = failed[["odometer", scale]].astype(float)
        pair = pair.loc[(pair > 0).all(axis=1)]
        a, s = np.log(pair["odometer"].to_numpy()), np.log(pair[scale].to_numpy())
        diffs = []
        for _ in range(int(c["n_bootstrap"])):
            i = rng.integers(0, len(a), len(a))
            diffs.append(np.std(a[i], ddof=1) - np.std(s[i], ddof=1))
        diffs = np.asarray(diffs)
        boot_rows.append({"escala": SCALES[scale], "n": len(pair), "diff_mean": diffs.mean(),
                          "ci_low": np.quantile(diffs, 0.025), "ci_high": np.quantile(diffs, 0.975)})
    boot = pd.DataFrame(boot_rows)

    all_events = v.loc[v["event_observed"].eq(1), "event_dss"].dropna()
    dss = {"n": int(len(all_events)), "median": float(all_events.median()),
           "p25": float(all_events.quantile(0.25)), "p75": float(all_events.quantile(0.75)),
           "max": float(all_events.max()), "sd_days": float(all_events.std()),
           # sd(log) no es invariante a correr el origen: en días absolutos las dos escalas se parecen.
           "age_sd_days": float(v.loc[all_events.index, "event_age"].std())}
    return failed, table, boot, dss


# --- H2 · dosis-respuesta -----------------------------------------------------------------

def h2_dose(v: pd.DataFrame, t: pd.DataFrame, failed: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, int]:
    c = cfg["dose_response"]
    e = t.loc[t["before_event"] & t[ID].isin(failed.index)].join(v[["odo_min"]], on=ID)
    e = e.loc[e["OdometerTripEnd"] <= e["odo_min"] + c["early_km"]]
    g = e.groupby(ID, observed=True).agg(
        km=("trip_km", "sum"), idle_min=("idle_min", "sum"), idle_n=("idle_f", "sum"),
        cold_n=("below_regime", "sum"), n=("trip_km", "size"), h=("trip_duration_min", "sum"),
        first=("date", "min"), last=("date", "max"))
    km = g["km"].where(g["km"] > 0)
    g["km_per_day"] = g["km"] / ((g["last"] - g["first"]) / DAY).clip(lower=1)
    g["idle_min_per_1000km"] = g["idle_min"] / km * 1000
    g["idle_n_per_1000km"] = g["idle_n"] / km * 1000
    g["cold_frac_trips"] = g["cold_n"] / g["n"]
    g["engine_h_per_1000km"] = g["h"] / 60 / km * 1000
    columns = ["km_per_day", "idle_min_per_1000km", "idle_n_per_1000km", "cold_frac_trips", "engine_h_per_1000km"]
    fe = failed.join(g[columns], how="inner")
    fe = fe.loc[fe["odometer"] > c["early_km"] + c["gap_km"]]
    rows = []
    for column in columns:
        r_km = stats.spearmanr(fe[column], fe["odometer"], nan_policy="omit")
        r_days = stats.spearmanr(fe[column], fe["age_days"], nan_policy="omit")
        rows.append({"uso_temprano": column, "rho_km_evento": r_km.statistic, "p_km": r_km.pvalue,
                     "rho_dias_evento": r_days.statistic, "p_dias": r_days.pvalue})
    return pd.DataFrame(rows), int(len(fe))


# --- H3 · ventana del registro ------------------------------------------------------------------

def h3_calendar(v: pd.DataFrame, cfg: dict[str, Any], window: tuple[pd.Timestamp, pd.Timestamp]):
    c = cfg["calendar"]
    ages = v["event_age"].dropna()
    lo, hi = float(ages.min()), float(ages.max())
    ev = v["event_date"]
    rows = []
    for month in pd.period_range(c["first_month"], c["last_month"], freq="M"):
        start, end = month.start_time, month.end_time
        age_start = (start - v["prod_date"]) / DAY
        age_end = (end - v["prod_date"]) / DAY
        risk = (age_end >= lo) & (age_start <= hi) & (v["last_trip"] >= start) & (ev.isna() | (ev >= start))
        n_events = int(((ev >= start) & (ev <= end)).sum())
        rows.append({"mes": str(month), "en_riesgo": int(risk.sum()),
                     "en_riesgo_sanos": int((risk & v["event_observed"].eq(0)).sum()), "eventos": n_events})
    table = pd.DataFrame(rows)
    table["por_100"] = (100 * table["eventos"] / table["en_riesgo"].replace(0, np.nan)).round(1)
    w0, w1 = window
    evs = ev.dropna()
    check = {
        "first_event": str(evs.min().date()), "last_event": str(evs.max().date()),
        "n_events": int(len(evs)), "outside_window": int(((evs < w0) | (evs > w1)).sum()),
        "latency_age_range": [lo, hi],
        "events_before_sale": int((v["event_dss"] <= 0).sum()),
        "events_after_last_trip": int(v["event_after_last_trip"].sum()),
    }
    return table, check


# --- H4 · riesgo dentro de la ventana -------------------------------------------------------------

def h4_hazard(v: pd.DataFrame, cfg: dict[str, Any], window: tuple[pd.Timestamp, pd.Timestamp]):
    """Autos-día dentro de la ventana, desde la producción hasta el último viaje o el evento."""
    c = cfg["hazard"]
    w0, w1 = window
    days = pd.date_range(w0, w1, freq="D")
    parts = []
    for _, r in v.iterrows():
        start = max(w0, r["prod_date"].normalize())
        end = min(r["last_trip"], w1) if pd.isna(r["event_date"]) else min(r["event_date"], r["last_trip"], w1)
        d = days[(days >= start) & (days <= end)]
        if not len(d):
            continue
        is_event = (d == r["event_date"].normalize()) if pd.notna(r["event_date"]) else np.zeros(len(d), bool)
        parts.append(pd.DataFrame({
            "age": np.asarray((d - r["prod_date"]) / DAY),
            "dss": np.asarray((d - r["sale_date"]) / DAY) if pd.notna(r["sale_date"]) else np.full(len(d), np.nan),
            "event": np.asarray(is_event)}))
    person_days = pd.concat(parts, ignore_index=True)
    tables = {}
    for column, bins in (("age", c["age_bins"]), ("dss", c["dss_bins"])):
        bucket = pd.cut(person_days[column], bins)
        g = person_days.groupby(bucket, observed=True).agg(autos_mes=("event", lambda s: len(s) / 30.4),
                                                           eventos=("event", "sum"))
        g["por_100_autos_mes"] = (100 * g["eventos"] / g["autos_mes"]).round(2)
        tables[column] = g.reset_index().rename(columns={column: "tramo"}).assign(tramo=lambda d: d["tramo"].astype(str))
    totals = {"events_in_window": int(person_days["event"].sum()), "car_months": float(len(person_days) / 30.4)}
    return tables["age"], tables["dss"], totals


# --- H4b · ¿el reloj arranca en la producción o en la venta? ------------------------------------

def _poisson_loglik(y: np.ndarray, X: np.ndarray) -> float:
    b0 = np.zeros(X.shape[1])
    b0[0] = np.log(y.mean())
    res = optimize.minimize(lambda b: np.sum(np.exp(X @ b)) - y @ (X @ b), b0,
                            jac=lambda b: X.T @ (np.exp(X @ b) - y), method="BFGS",
                            options={"gtol": 1e-8, "maxiter": 5000})
    return float(-res.fun)  # sin el término log(y!), que es 0 con y ∈ {0, 1}


def h4b_clock(v: pd.DataFrame, cfg: dict[str, Any], window: tuple[pd.Timestamp, pd.Timestamp]):
    """Riesgo por autos-día dentro de la ventana, desde el día siguiente a la venta.

    La regresión ingenua entre fallados (edad al evento contra daysUntilSale) está
    sesgada por la ventana: el registro trunca la fecha del evento. Los autos-día dentro
    de la ventana manejan ese truncamiento. Edad y días post-venta difieren en
    daysUntilSale, así que se pueden separar. Se compara un Poisson con edad, con días
    post-venta, con calendario y con combinaciones (términos lineal y cuadrático).
    Arranca en la venta porque ningún evento cae antes (convención del registro, H3):
    contar los días previos favorecería a la venta por construcción.
    """
    w0, w1 = window
    days = pd.date_range(w0, w1, freq="D")
    parts = []
    for _, r in v.loc[v["sale_date"].notna()].iterrows():
        start = max(w0, r["sale_date"].normalize() + DAY)
        end = min(r["last_trip"], w1) if pd.isna(r["event_date"]) else min(r["event_date"], r["last_trip"], w1)
        d = days[(days >= start) & (days <= end)]
        if not len(d):
            continue
        is_event = (d == r["event_date"].normalize()) if pd.notna(r["event_date"]) else np.zeros(len(d), bool)
        parts.append(pd.DataFrame({"age": np.asarray((d - r["prod_date"]) / DAY),
                                   "dss": np.asarray((d - r["sale_date"]) / DAY),
                                   "cal": np.asarray((d - w0) / DAY), "event": np.asarray(is_event, float)}))
    p = pd.concat(parts, ignore_index=True)
    y = p["event"].to_numpy()
    z = {c: ((p[c] - p[c].mean()) / p[c].std()).to_numpy() for c in ("age", "dss", "cal")}
    models = {"nulo": [], "edad": ["age"], "post_venta": ["dss"], "calendario": ["cal"],
              "edad+post_venta": ["age", "dss"], "edad+calendario": ["age", "cal"],
              "post_venta+calendario": ["dss", "cal"], "los_tres": ["age", "dss", "cal"]}
    rows, ll = [], {}
    for terms in ("lineal", "cuadratico"):
        for name, cols in models.items():
            X = [np.ones(len(p))]
            for c in cols:
                X += [z[c], z[c] ** 2] if terms == "cuadratico" else [z[c]]
            X = np.column_stack(X)
            ll[(terms, name)] = _poisson_loglik(y, X)
            rows.append({"terminos": terms, "modelo": name, "loglik": ll[(terms, name)],
                         "aic": 2 * X.shape[1] - 2 * ll[(terms, name)]})
    fits = pd.DataFrame(rows)
    tests = []
    for terms, k in (("lineal", 1), ("cuadratico", 2)):
        for base, full in (("edad", "edad+post_venta"), ("post_venta", "edad+post_venta"),
                           ("edad+calendario", "los_tres"), ("post_venta+calendario", "los_tres")):
            lr = 2 * (ll[(terms, full)] - ll[(terms, base)])
            added = sorted(set(models[full]) - set(models[base]))
            tests.append({"terminos": terms, "base": base, "agrega": "+".join(added), "lr": lr,
                          "p": float(stats.chi2.sf(lr, k * len(added)))})
    e = v.loc[v["event_observed"].eq(1)].dropna(subset=["days_until_sale"])
    naive = stats.linregress(e["days_until_sale"], e["event_age"])
    info = {"car_days": int(len(p)), "events": int(y.sum()),
            "rho_age_dss_car_days": float(np.corrcoef(p["age"], p["dss"])[0, 1]),
            "naive_slope_age_on_dus": float(naive.slope), "naive_slope_se": float(naive.stderr),
            "naive_rho_dus_dss": float(stats.spearmanr(e["days_until_sale"], e["event_dss"]).statistic)}
    return fits, pd.DataFrame(tests), info


# --- H5 · trayectoria --------------------------------------------------------------------------

def h5_trajectory(t: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    bins = cfg["trajectory"]["days_before_bins"]
    key = ["market", "month", "post_sale"]
    ref = (t.loc[t["event_observed"].eq(0)].groupby(key, observed=True)
           .agg(r_idle=("idle_f", "mean"), r_below=("below_mov", "mean")))
    f = t.loc[t["event_observed"].eq(1) & t["post_sale"]].join(ref, on=key)
    f["days_before"] = (f["event_date"] - f["date"]) / DAY
    f = f.loc[f["days_before"] > 0]
    labels = [f"{a}-{b}" if b < 100000 else f">{a}" for a, b in zip(bins[:-1], bins[1:])]
    f["tramo"] = pd.cut(f["days_before"], bins, labels=labels)
    f["ex_idle"] = f["idle_f"] - f["r_idle"]
    f["ex_below"] = f["below_mov"] - f["r_below"]
    per_vehicle = f.groupby([ID, "tramo"], observed=True)[["ex_idle", "ex_below"]].mean().reset_index()
    return (per_vehicle.groupby("tramo", observed=True)
            .agg(vehiculos=(ID, "size"), exceso_idle=("ex_idle", "mean"), exceso_bajo_regimen=("ex_below", "mean"))
            .reset_index().assign(tramo=lambda d: d["tramo"].astype(str)))


# --- H6 · rasgo temprano -------------------------------------------------------------------------

def post_sale_deviation(t: pd.DataFrame) -> pd.DataFrame:
    """Viajes post-venta con su desvío contra los sanos post-venta del mismo mercado × mes."""
    ps = t.loc[t["post_sale"]].copy()
    ref = (ps.loc[ps["event_observed"].eq(0)].groupby(["market", "month"], observed=True)
           .agg(r_idle=("idle_f", "mean"), r_below=("below_mov", "mean")))
    ps = ps.join(ref, on=["market", "month"])
    ps["d_idle"] = ps["idle_f"] - ps["r_idle"]
    ps["d_below"] = ps["below_mov"] - ps["r_below"]
    return ps


def trait_at(ps: pd.DataFrame, v: pd.DataFrame, landmark: float, min_trips: int) -> pd.DataFrame:
    """Rasgo por vehículo con los viajes post-venta hasta el hito, vivos en el hito."""
    s = ps.loc[(ps["dss"] <= landmark) & ps["before_event"]]
    f = s.groupby(ID, observed=True).agg(n=("idle_f", "size"), di=("d_idle", "mean"), db=("d_below", "mean"),
                                         km=("trip_km", "sum"), odo=("OdometerTripEnd", "max"))
    f = f.loc[f["n"] >= min_trips].join(v[["event_observed", "event_dss", "event_odo_km", "production_day",
                                           "days_until_sale", "market", "span_days"]])
    alive = f["event_observed"].eq(0) | (f["event_dss"] > landmark)
    return f.loc[alive].dropna(subset=["di", "db"])


def h6_trait(v: pd.DataFrame, t: pd.DataFrame, cfg: dict[str, Any], rng: np.random.Generator):
    c = cfg["early_trait"]
    ps = post_sale_deviation(t)
    failed = v["event_observed"].eq(1)
    pd_max = float(v.loc[failed, "production_day"].max())
    follow = float(v.loc[failed, "event_age"].max())
    comparable = v.index[failed | (v["event_observed"].eq(0) & (v["production_day"] <= pd_max) & (v["span_days"] >= follow))]
    population = {"production_day_max": pd_max, "min_follow_days": follow,
                  "n_failed": int(failed.sum()), "n_healthy_comparable": int(len(comparable) - failed.sum())}

    auc_rows = []
    for landmark in c["landmarks_days"]:
        f = trait_at(ps, v, landmark, int(c["min_trips"]))
        f = f.loc[f.index.isin(comparable)]
        y = f["event_observed"].to_numpy()
        row = {"hito_dias": landmark, "fallados": int(y.sum()), "sanos": int((1 - y).sum())}
        for name, column in (("idle", "di"), ("bajo_regimen", "db")):
            auc, lo, hi = auc_ci(y, f[column].to_numpy(), rng, int(c["n_bootstrap"]))
            row.update({f"auc_{name}": auc, f"auc_{name}_lo": lo, f"auc_{name}_hi": hi})
        # Piso de uso (lo que después mide A4): menos km hasta el hito ⇒ más riesgo.
        row["auc_menos_km"] = float(roc_auc_score(y, -f["km"]))
        row["rho_bajo_regimen_km"] = float(stats.spearmanr(f["db"], f["km"]).statistic)
        row["rho_idle_km"] = float(stats.spearmanr(f["di"], f["km"]).statistic)
        auc_rows.append(row)

    # Confusor de producción y nulo estratificado, en el hito declarado.
    landmark = c["confound_landmark_days"]
    f = trait_at(ps, v, landmark, int(c["min_trips"]))
    f = f.loc[f.index.isin(comparable)].copy()
    f["combo"] = zsum(f, ["di", "db"])
    y = f["event_observed"].to_numpy()
    confound: dict[str, Any] = {"hito_dias": landmark, "fallados": int(y.sum()), "sanos": int((1 - y).sum())}
    for column in ("di", "db", "combo"):
        confound[f"auc_{column}"] = float(roc_auc_score(y, f[column]))
        confound[f"rho_{column}_production"] = float(stats.spearmanr(f[column], f["production_day"]).statistic)
        confound[f"rho_{column}_days_until_sale"] = float(
            stats.spearmanr(f[column], f["days_until_sale"], nan_policy="omit").statistic)
    f["tercil"] = pd.qcut(f["production_day"], int(c["production_terciles"]), labels=False)
    strata_rows = []
    strata = [("tercil", k, g) for k, g in f.groupby("tercil")] + [("mercado", k, g) for k, g in f.groupby("market")]
    for kind, key, d in strata:
        if d["event_observed"].nunique() == 2:
            strata_rows.append({"estrato": kind, "valor": str(key), "fallados": int(d["event_observed"].sum()),
                                "sanos": int((1 - d["event_observed"]).sum()),
                                "auc_bajo_regimen": roc_auc_score(d["event_observed"], d["db"]),
                                "auc_idle": roc_auc_score(d["event_observed"], d["di"]),
                                "auc_combo": roc_auc_score(d["event_observed"], d["combo"])})
    real = float(roc_auc_score(y, f["combo"]))
    groups = f.reset_index().groupby(["tercil", "market"]).indices
    combo = f["combo"].to_numpy()
    null = np.empty(int(c["n_permutations"]))
    for k in range(len(null)):
        yy = y.copy()
        for idx in groups.values():
            yy[idx] = rng.permutation(yy[idx])
        null[k] = roc_auc_score(yy, combo)
    confound.update({"null_auc_real": real, "null_mean": float(null.mean()), "null_sd": float(null.std()),
                     "null_p": float((1 + (null >= real).sum()) / (1 + len(null)))})

    # Punto de operación contra TODOS los sanos de dev, y cuánto de eso son autos quietos
    # (vendidos pero casi sin km hasta el hito: sus viajes son idle y la fracción explota).
    det_rows, still_rows = [], []
    for landmark in c["detection_landmarks_days"]:
        f = trait_at(ps, v, landmark, int(c["min_trips"])).copy()
        f["combo"] = zsum(f, ["di", "db"])
        y = f["event_observed"].to_numpy()
        s = f["combo"].to_numpy()
        still = f["km"] < float(c["immobile_km"])
        moved = f.loc[~still]
        still_rows.append({"hito_dias": landmark, "quietos": int(still.sum()),
                           "quietos_fallados": int(f.loc[still, "event_observed"].sum()),
                           "tasa_quietos": float(f.loc[still, "event_observed"].mean()),
                           "tasa_resto": float(moved["event_observed"].mean()),
                           "auc_combo_todos": float(roc_auc_score(y, s)),
                           "auc_combo_sin_quietos": float(roc_auc_score(moved["event_observed"], moved["combo"])),
                           "auc_bajo_regimen_sin_quietos": float(roc_auc_score(moved["event_observed"], moved["db"])),
                           "auc_idle_sin_quietos": float(roc_auc_score(moved["event_observed"], moved["di"]))})
        thr = np.quantile(s[y == 0], 1 - float(c["healthy_alert_frac"]))
        hit = (y == 1) & (s > thr)
        lead = f.loc[hit, "event_dss"] - landmark
        lead_km = (f["event_odo_km"] - f["odo"]).loc[hit]
        det_rows.append({"hito_dias": landmark, "fallados": int(y.sum()), "sanos": int((1 - y).sum()),
                         "auc_todos": float(roc_auc_score(y, s)), "detectados": int(hit.sum()),
                         "deteccion": float(hit.sum() / y.sum()),
                         "anticipacion_mediana_dias": float(lead.median()) if len(lead) else np.nan,
                         "anticipacion_mediana_km": float(lead_km.median()) if len(lead) else np.nan,
                         # Los que alertan, ¿se movieron? H2: km/día maneja los km al evento (A4 lo mide).
                         "km_hasta_hito_detectados": float(f.loc[hit, "km"].median()) if len(lead) else np.nan,
                         "km_hasta_hito_fallados": float(f.loc[y == 1, "km"].median()),
                         "km_hasta_hito_sanos": float(f.loc[y == 0, "km"].median())})
    return (population, pd.DataFrame(auc_rows), confound, pd.DataFrame(strata_rows), pd.DataFrame(det_rows),
            pd.DataFrame(still_rows))


# --- H9 · panel v1 -----------------------------------------------------------------------------

def h9_panel_v1(cfg: dict[str, Any], split: dict[str, Any], t: pd.DataFrame,
                window: tuple[pd.Timestamp, pd.Timestamp]) -> dict[str, Any]:
    columns = [ID, "cut_odo", "label", "event_observed", "gap_km", "horizon_km"]
    panel = pd.read_parquet(resolve_path(cfg["panel_v1"]), columns=columns)
    dev_mask, _ = test_split_masks(panel, split, strict=True)
    pan = panel.loc[dev_mask].reset_index(drop=True)
    anchors = (t[[ID, "OdometerTripEnd", "date"]].dropna().sort_values("OdometerTripEnd"))

    def date_at(odo: pd.Series) -> pd.Series:
        q = pd.DataFrame({ID: pan[ID], "OdometerTripEnd": odo.astype("float64"), "_i": pan.index})
        m = pd.merge_asof(q.sort_values("OdometerTripEnd"), anchors, on="OdometerTripEnd", by=ID, direction="forward")
        return m.set_index("_i")["date"].reindex(pan.index)

    h_start = date_at(pan["cut_odo"] + pan["gap_km"])
    h_end = date_at(pan["cut_odo"] + pan["gap_km"] + pan["horizon_km"])
    w0, w1 = window
    neg = pan["event_observed"].eq(0)
    pos = pan["label"].eq(1)
    return {
        "healthy_rows": int(neg.sum()), "healthy_vehicles": int(pan.loc[neg, ID].nunique()),
        "inside": float(((h_start >= w0) & (h_end <= w1))[neg].mean()),
        "exits_after": float(((h_end > w1) & (h_start <= w1))[neg].mean()),
        "fully_after": float((h_start > w1)[neg].mean()),
        "starts_before": float((h_start < w0)[neg].mean()),
        "horizon_end_unknown": float(h_end.isna()[neg].mean()),
        "positive_rows": int(pos.sum()), "positive_starts_before": float((h_start < w0)[pos].mean()),
    }


# --- factibilidad del panel de hitos -------------------------------------------------------------

def feasibility(v: pd.DataFrame, t: pd.DataFrame, cfg: dict[str, Any],
                window: tuple[pd.Timestamp, pd.Timestamp]) -> tuple[pd.DataFrame, dict[str, Any]]:
    c = cfg["feasibility"]
    w0, w1 = window
    gap, horizon = pd.Timedelta(days=c["gap_days"]), pd.Timedelta(days=c["horizon_days"])
    ps = t.loc[t["post_sale"]]
    rows = []
    for landmark in c["landmarks_days"]:
        n_trips = ps.loc[ps["dss"] <= landmark].groupby(ID, observed=True).size()
        d = v.loc[v["sale_date"].notna()].copy()
        d["land"] = d["sale_date"] + pd.Timedelta(days=landmark)
        reached = d["land"] <= d["last_trip"]
        before_gap = d["event_date"].notna() & (d["event_date"] <= d["land"] + gap)
        enough = n_trips.reindex(d.index, fill_value=0) >= int(c["min_trips"])
        keep = reached & ~before_gap & enough
        d = d.loc[keep]
        entry = np.maximum(d["land"] + gap, w0)
        exit_ = d["event_date"].where(d["event_date"].notna(), np.minimum(d["last_trip"], w1))
        exposure = (exit_ - entry) / DAY
        healthy = d["event_observed"].eq(0)
        row = {"hito_dias": landmark, "sin_venta": int(v["sale_date"].isna().sum()),
               "no_llega_al_hito": int((~reached).sum()),
               "evento_antes_o_en_gap": int((reached & before_gap).sum()),
               "pocos_viajes": int((reached & ~before_gap & ~enough).sum()),
               "en_riesgo": int(len(d)), "eventos": int((~healthy).sum()),
               "positivos_en_horizonte": int((d["event_date"] <= d["land"] + gap + horizon).sum()),
               "sanos": int(healthy.sum()), "sanos_sin_exposicion": int((exposure[healthy] <= 0).sum())}
        for threshold in c["exposure_thresholds_days"]:
            row[f"sanos_exp_ge_{threshold}"] = int((exposure[healthy] >= threshold).sum())
        top = max(c["exposure_thresholds_days"])
        row[f"sanos_exp_1_a_{top - 1}"] = int(((exposure[healthy] > 0) & (exposure[healthy] < top)).sum())
        rows.append(row)
    km_day = (ps.groupby(ID, observed=True)
              .apply(lambda g: g["trip_km"].sum() / max((g["date"].max() - g["date"].min()) / DAY, 1.0),
                     include_groups=False))
    usage = {"km_per_day_p25": float(km_day.quantile(0.25)), "km_per_day_p50": float(km_day.median()),
             "km_per_day_p75": float(km_day.quantile(0.75)), "days_per_500km": float(500 / km_day.median()),
             "no_days_until_sale": int(v["days_until_sale"].isna().sum())}
    return pd.DataFrame(rows), usage


# --- salida ----------------------------------------------------------------------------------

def show(title: str, frame: pd.DataFrame) -> None:
    print(f"\n== {title} ==")
    print(frame.round(3).to_string(index=False))


def versus(label: str, measured: Any, expected: Any) -> None:
    print(f"  {label:<52s} medido {measured!s:<28s} §2 {expected!s}")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/data/event_clock.yaml")
    parser.add_argument("--refresh", action="store_true", help="vuelve a leer los viajes crudos")
    args = parser.parse_args()
    cfg = load_config(args.config)
    rng = np.random.default_rng(int(cfg["seed"]))
    out = ensure_dir(cfg["output_dir"])

    split = load_test_split(cfg["test_split"])
    dev = {str(x) for x in split["dev_vehicles"]}
    trips = load_trips(cfg, dev, refresh=args.refresh)
    v, anchor = build_vehicles(cfg, dev, trips)
    t = annotate_trips(trips, v)
    window = (pd.Timestamp(cfg["event_window"]["start"]), pd.Timestamp(cfg["event_window"]["end"]))
    print(f"dev: {len(v)} vehículos ({int(v['event_observed'].sum())} con evento) · {len(t)} viajes · "
          f"origen día {anchor['origin_day_since_epoch']:.0f} (re-estimado {anchor['origin_reestimated']:.0f}, "
          f"IQR {anchor['iqr_days']:.1f} d)")

    failed, scales, boot, dss = h1_scales(v, t, cfg, rng)
    show(f"H1 · dispersión del momento del evento por escala ({len(failed)} fallados con telemetría desde "
         f"< {cfg['scales']['max_first_trip_odo_km']} km)", scales)
    show("H1 · bootstrap de sd_log(odómetro) − sd_log(escala) (> 0: la escala concentra más)", boot)
    print(f"días post-venta al evento (los {dss['n']} eventos de dev): mediana {dss['median']:.0f}, "
          f"IQR {dss['p25']:.0f}–{dss['p75']:.0f}, máximo {dss['max']:.0f} · desvío en días: post-venta "
          f"{dss['sd_days']:.0f}, edad {dss['age_sd_days']:.0f}")

    dose, n_dose = h2_dose(v, t, failed, cfg)
    show(f"H2 · uso en los primeros {cfg['dose_response']['early_km']} km contra el evento (n = {n_dose})", dose)

    calendar, check = h3_calendar(v, cfg, window)
    show(f"H3 · autos en riesgo y eventos por mes (edad al evento en dev: {check['latency_age_range'][0]:.0f}–"
         f"{check['latency_age_range'][1]:.0f} días)", calendar)
    print(f"eventos de dev: {check['first_event']} → {check['last_event']} · fuera de la ventana congelada "
          f"{window[0].date()} → {window[1].date()}: {check['outside_window']} · antes de la venta: "
          f"{check['events_before_sale']} · después del último viaje: {check['events_after_last_trip']}")

    haz_age, haz_dss, haz_tot = h4_hazard(v, cfg, window)
    show("H4 · tasa por 100 autos-mes dentro de la ventana, por edad", haz_age)
    show("H4 · tasa por 100 autos-mes dentro de la ventana, por días desde la venta", haz_dss)

    clock_fits, clock_tests, clock = h4b_clock(v, cfg, window)
    print(f"\n== H4b · ¿producción o venta? Poisson sobre {clock['car_days']} autos-día post-venta en la ventana "
          f"({clock['events']} eventos; ρ(edad, post-venta) = {clock['rho_age_dss_car_days']:.2f}) ==")
    print(clock_fits.round(2).to_string(index=False))
    print(clock_tests.round(4).to_string(index=False))
    print(f"  regresión ingenua entre fallados (sesgada por la ventana): edad al evento ~ daysUntilSale, "
          f"pendiente {clock['naive_slope_age_on_dus']:.2f} ± {clock['naive_slope_se']:.2f}; "
          f"ρ(daysUntilSale, días post-venta al evento) = {clock['naive_rho_dus_dss']:+.2f}")

    trajectory = h5_trajectory(t, cfg)
    show("H5 · fallados post-venta: exceso contra sanos del mismo mercado × mes × etapa, por días antes del evento",
         trajectory)

    population, trait_auc, confound, strata, detection, still = h6_trait(v, t, cfg, rng)
    show(f"H6 · AUC por vehículo, población comparable ({population['n_failed']} fallados; sanos producidos hasta "
         f"el día {population['production_day_max']:.0f} y seguidos ≥ {population['min_follow_days']:.0f} d: "
         f"{population['n_healthy_comparable']})", trait_auc)
    print(f"\n== H6 · confusor de producción (hito {confound['hito_dias']} d: {confound['fallados']} vs "
          f"{confound['sanos']}) ==")
    for column in ("di", "db", "combo"):
        print(f"  {column:6s} AUC {confound[f'auc_{column}']:.3f} · ρ con ProductionDay "
              f"{confound[f'rho_{column}_production']:+.2f} · ρ con daysUntilSale "
              f"{confound[f'rho_{column}_days_until_sale']:+.2f}")
    print(strata.round(3).to_string(index=False))
    print(f"  combo {confound['null_auc_real']:.3f} · nulo estratificado (producción × mercado) "
          f"{confound['null_mean']:.3f} ± {confound['null_sd']:.3f} · p = {confound['null_p']:.4f}")
    show(f"H6 · detección alertando al {cfg['early_trait']['healthy_alert_frac']:.0%} de TODOS los sanos", detection)
    show(f"H6 · autos quietos (< {cfg['early_trait']['immobile_km']} km post-venta hasta el hito), todos los sanos",
         still)

    h9 = h9_panel_v1(cfg, split, t, window)
    print(f"\n== H9 · panel v1: horizontes de las {h9['healthy_rows']} filas sanas de dev "
          f"({h9['healthy_vehicles']} vehículos) ==")
    print(f"  enteros dentro de la ventana {h9['inside']:.1%} · se salen después {h9['exits_after']:.1%} · "
          f"enteros después {h9['fully_after']:.1%} · empiezan antes {h9['starts_before']:.1%} · "
          f"fin desconocido {h9['horizon_end_unknown']:.1%}")
    print(f"  filas positivas {h9['positive_rows']}: horizonte que empieza antes de la ventana "
          f"{h9['positive_starts_before']:.1%}")

    feas, usage = feasibility(v, t, cfg, window)
    show(f"Factibilidad del panel de hitos (G = {cfg['feasibility']['gap_days']} d, H = "
         f"{cfg['feasibility']['horizon_days']} d, ≥ {cfg['feasibility']['min_trips']} viajes post-venta)", feas)
    print(f"km/día post-venta: p25 {usage['km_per_day_p25']:.0f} · p50 {usage['km_per_day_p50']:.0f} · "
          f"p75 {usage['km_per_day_p75']:.0f} → 500 km ≈ {usage['days_per_500km']:.0f} días · sin daysUntilSale: "
          f"{usage['no_days_until_sale']}")

    # Lado a lado con lo que midió el scratch (§2 del prompt).
    e = cfg["expected"]
    sd = scales.set_index("escala")["sd_log"]
    days_row = boot.set_index("escala").loc[SCALES["age_days"]]
    rate_dss = haz_dss["por_100_autos_mes"].round(1).tolist()
    rate_age = haz_age["por_100_autos_mes"].round(1).tolist()
    presale = haz_dss.iloc[0]
    traj = trajectory.set_index("tramo")
    far, near = traj.index[-1], traj.index[0]
    print("\n== Contra §2 ==")
    versus("H1 fallados usables", len(failed), e["h1_n_failed"])
    versus("H1 sd(log) edad", round(sd[SCALES["age_days"]], 2), e["h1_sdlog_age"])
    versus("H1 sd(log) odómetro", round(sd[SCALES["odometer"]], 2), e["h1_sdlog_odometer"])
    versus("H1 odómetro − edad [IC95]", f"{days_row['diff_mean']:+.2f} [{days_row['ci_low']:+.2f}, "
           f"{days_row['ci_high']:+.2f}]", e["h1_diff_odometer_minus_days"])
    versus("H1 días post-venta al evento (mediana, IQR, máx)",
           f"{dss['median']:.0f}, {dss['p25']:.0f}–{dss['p75']:.0f}, {dss['max']:.0f}",
           f"{e['h1_dss_median']}, {e['h1_dss_iqr'][0]}–{e['h1_dss_iqr'][1]}, {e['h1_dss_max']}")
    kmday = dose.set_index("uso_temprano").loc["km_per_day"]
    versus("H2 n", n_dose, e["h2_n"])
    versus("H2 ρ(km/día, km) · ρ(km/día, días)", f"{kmday['rho_km_evento']:+.2f} · {kmday['rho_dias_evento']:+.2f}",
           f"{e['h2_rho_kmday_km']:+.2f} · {e['h2_rho_kmday_days']:+.2f}")
    versus("H3 primer y último evento", f"{check['first_event']} → {check['last_event']}",
           f"{e['h3_first_event']} → {e['h3_last_event']}")
    versus("H3 eventos fuera de la ventana", check["outside_window"], 0)
    versus("H4 autos-mes antes de la venta (eventos)", f"{presale['autos_mes']:.0f} ({int(presale['eventos'])})",
           f"{e['h4_presale_car_months']} (0)")
    versus("H4 tasa por días post-venta", rate_dss, e["h4_dss_rates"])
    versus("H4 tasa por edad", rate_age, e["h4_age_rates"])
    versus("H5 exceso idle lejos → cerca", f"{traj.loc[far, 'exceso_idle']:+.2f} → {traj.loc[near, 'exceso_idle']:+.2f}",
           e["h5_idle_far_near"])
    versus("H5 exceso bajo régimen lejos → cerca", f"{traj.loc[far, 'exceso_bajo_regimen']:+.2f} → "
           f"{traj.loc[near, 'exceso_bajo_regimen']:+.2f}", e["h5_below_far_near"])
    first = trait_auc.iloc[0]
    versus("H6 AUC bajo régimen ≤ 30 d [IC95]", f"{first['auc_bajo_regimen']:.2f} [{first['auc_bajo_regimen_lo']:.2f}, "
           f"{first['auc_bajo_regimen_hi']:.2f}]", e["h6_auc_below_30"])
    versus("H6 combo contra nulo estratificado (p)", f"{confound['null_auc_real']:.3f} (p = {confound['null_p']:.4f})",
           f"{e['h6_null_auc']} (p = {e['h6_null_p']})")
    versus("H6 detección al 5% de los sanos", [f"{r.detectados}/{r.fallados}" for r in detection.itertuples()],
           e["h6_detection"])
    versus("H9 dentro · sale después · entero después · antes",
           f"{h9['inside']:.3f} · {h9['exits_after']:.3f} · {h9['fully_after']:.3f} · {h9['starts_before']:.3f}",
           f"{e['h9_inside']} · {e['h9_exits_after']} · {e['h9_fully_after']} · {e['h9_starts_before']}")
    versus("Factibilidad: eventos por hito", feas["eventos"].tolist(), e["feasibility_events"])
    versus("Factibilidad: sanos por hito", feas["sanos"].tolist(), e["feasibility_healthy"])
    versus("Factibilidad: sanos con ≥ 120 d en ventana", feas["sanos_exp_ge_120"].tolist(), e["feasibility_exposed_120"])
    versus("Factibilidad: sanos sin exposición", feas["sanos_sin_exposicion"].tolist(), e["feasibility_no_exposure"])

    for name, frame in {"h1_scales": scales, "h1_bootstrap": boot, "h2_dose_response": dose,
                        "h3_calendar": calendar, "h4_hazard_age": haz_age, "h4_hazard_dss": haz_dss,
                        "h4b_clock_fits": clock_fits, "h4b_clock_tests": clock_tests,
                        "h5_trajectory": trajectory, "h6_early_auc": trait_auc, "h6_strata": strata,
                        "h6_detection": detection, "h6_immobile": still, "feasibility": feas}.items():
        frame.to_csv(out / f"{name}.csv", index=False)
    summary = {"config": cfg.get("_config_path"), "anchor": anchor, "event_window": cfg["event_window"],
               "h1_n_failed": int(len(failed)), "h1_dss_all_events": dss, "h2_n": n_dose, "h3_check": check,
               "h4_totals": haz_tot, "h4b_clock": clock, "h6_population": population, "h6_confound": confound, "h9_panel_v1": h9,
               "usage": usage}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\nescrito: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
