#!/usr/bin/env python
"""EDA de la entrega v2 sobre el dev nuevo, con las comparaciones estratificadas por mercado.

    python scripts/eda_v2.py --config configs/eda_v2.yaml

Corre DESPUÉS de `build_eda_cache.py` y `eda_gaps.py` con `configs/data/eda_cache_v2.yaml`
(lee su cache: el filtro a dev se decide en un solo lugar). Solo dev: falla si aparece
un vehículo que no está en `dev_vehicles`. Cada sección contesta una pregunta que la
EDA del 18-09 contestó con la entrega 1 y que v2 puede cambiar:

A. **Composición**: mercado × motor × evento; modelo × motor; eventos por auto.
B. **Riesgo por días desde la venta** (Poisson con exposición): cómo cambia el riesgo
   con la edad post-venta, por mercado, por motor y con la altura de la ciudad.
C. **¿Sigue habiendo ventana del registro?** El mismo Poisson con el mes calendario.
D. **Reloj del evento**: ¿el evento se ordena por días o por km? (Kordonsky-Gertsbakh)
E. **Perfil alineado al evento, dentro del mercado**: P(fallado > sano) por tramo de km
   antes de la fecha registrada y de la fecha − 21 d, contra sanos del mismo mercado y
   el mismo tramo de odómetro (y, aparte, del mismo mes calendario). Al lado, la versión
   agrupada, para ver cuánto era mercado.
F. **Rasgo temprano**: los primeros 60/90 días post-venta, dentro del mercado.
G. **Estáticas de exposición** dentro de la ventana de producción del universo.
H. **Alrededor de la fecha registrada**: idle largo (auto parado con el motor en marcha)
   día a día, contra la fecha registrada v2 y la de la entrega 1.
I. **Eventos repetidos**.
P. **Período de producción** (si el config declara `production_window_days`): cuántos
   fallados y sanos hay antes, dentro y después de la ventana en la que se muestrearon las
   dos cohortes, su tasa con exposición y cuánto ordena la fecha de producción sola. Es la
   sección que dice cuánto pesa el universo sin la ventana (decisión del equipo, 26-09).

Deja CSVs y `summary.json` en `output_dir`. La lectura está en `docs/memoria/f9-eda-v2.md`.
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
import yaml
from scipy.stats import chi2, mannwhitneyu, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.config import ensure_dir, load_config, resolve_path, set_seed  # noqa: E402
from src.eval.splits import load_test_split  # noqa: E402
from src.features.trips import derive_trip_columns  # noqa: E402

logger = logging.getLogger("eda_v2")
ID = "vehicle_id"
MKT = "static_SalesCountry_cd"
ENG = "static_Engine"
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 60)
pd.set_option("display.max_rows", 200)


def show(title: str, obj: Any, **kw: Any) -> None:
    print(f"\n== {title} ==")
    if isinstance(obj, (pd.DataFrame, pd.Series)):
        print(obj.to_string(**kw))
    else:
        print(obj)


# ======================================================================================
# Carga
# ======================================================================================
def load(cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    split = load_test_split(cfg["test_split"])
    dev = set(split["dev_vehicles"])
    cache = resolve_path(cfg["eda_cache_dir"])
    meta = json.loads((cache / "meta.json").read_text(encoding="utf-8"))
    veh = pd.read_parquet(cache / "vehicles_dev.parquet")
    if set(veh[ID]) != dev:
        raise AssertionError("vehicles_dev no coincide con dev_vehicles del holdout: el cache es de otro split")

    origin = pd.Timestamp("1970-01-01") + pd.Timedelta(days=float(meta["anchor_origin_day_since_epoch"]))
    end = pd.Timestamp(cfg["extraction_end"])
    veh["prod_date"] = origin + pd.to_timedelta(veh["static_ProductionDay"], unit="D")
    veh["sale_date"] = veh["prod_date"] + pd.to_timedelta(veh["static_daysUntilSale"], unit="D")
    veh["event_date_rec"] = veh["prod_date"] + pd.to_timedelta(veh["event_day_since_production"], unit="D")
    veh["y"] = (veh["event_observed"].eq(1) & veh["event_date_rec"].le(end)).astype(int)
    veh["exit_date"] = veh["event_date_rec"].where(veh["y"].eq(1), end)
    window = cfg.get("production_window_days")
    if window is not None:
        day = veh["static_ProductionDay"]
        veh["prod_band"] = np.where(day < window[0], "antes", np.where(day > window[1], "después", "dentro"))
    else:
        veh["prod_band"] = "todo"

    elev = yaml.safe_load(resolve_path(cfg["city_elevation"]).read_text(encoding="utf-8"))["cities"]
    table = {(e["country"], e["city"]): e.get("elevation_m") for e in elev}
    veh["elev_m"] = [table.get((m, c)) for m, c in zip(veh[MKT], veh["static_SalesCity"])]
    veh["elev_m"] = pd.to_numeric(veh["elev_m"], errors="coerce")

    gaps = resolve_path(cfg["gaps_dir"])
    trips = pd.read_parquet(gaps / "trips_dev_full.parquet")
    signals = pd.read_parquet(gaps / "signals_dev_full.parquet", columns=[ID, "eventTimestamp", "Message"])
    for name, frame in (("trips", trips), ("signals", signals)):
        extra = set(frame[ID]) - dev
        if extra:
            raise AssertionError(f"{name}: {len(extra)} vehículo(s) que no son de dev")

    spec = load_config(cfg["features_spec"])
    raw_cols = [c for c in trips.columns if c not in {"trip_km", "trip_duration_min", "engine_temp_amplitude",
                                                      "air_regen_delta", "oil_life_drop"}]
    trips, counters = derive_trip_columns(trips[raw_cols], thresholds=spec.get("thresholds"), clip=spec.get("clip"))
    logger.info("trips derivados: %s", counters)
    trips["date"] = trips["TripDatetimeStart"].dt.tz_convert(None)
    trips["day"] = trips["date"].dt.normalize()
    trips["long_idle"] = trips["idle"] & trips["trip_duration_min"].ge(float(cfg["aligned"]["long_idle_min"]))
    info = veh.set_index(ID)
    for col in (MKT, ENG, "y"):
        trips[col] = trips[ID].map(info[col])
    return veh, trips, signals, origin


# ======================================================================================
# Poisson con exposición (IRLS; statsmodels no está en el entorno)
# ======================================================================================
def poisson_fit(X: np.ndarray, y: np.ndarray, offset: np.ndarray, max_iter: int = 100) -> dict[str, Any]:
    beta = np.zeros(X.shape[1])
    beta[0] = np.log(max(y.sum(), 0.5) / np.exp(offset).sum())
    for _ in range(max_iter):
        eta = X @ beta + offset
        mu = np.exp(eta)
        z = eta - offset + (y - mu) / mu
        XtW = X.T * mu
        step = np.linalg.solve(XtW @ X + 1e-9 * np.eye(X.shape[1]), XtW @ z)
        if np.max(np.abs(step - beta)) < 1e-10:
            beta = step
            break
        beta = step
    mu = np.exp(X @ beta + offset)
    cov = np.linalg.inv((X.T * mu) @ X + 1e-9 * np.eye(X.shape[1]))
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(y > 0, y * np.log(y / mu), 0.0)
    deviance = float(2.0 * np.sum(term - (y - mu)))
    return {"beta": beta, "se": np.sqrt(np.diag(cov)), "deviance": deviance}


def design(frame: pd.DataFrame, cats: dict[str, str], nums: list[str] = ()) -> tuple[np.ndarray, list[str]]:
    """Intercepto + dummies (con el nivel de referencia dado) + numéricas."""
    parts, names = [np.ones((len(frame), 1))], ["(intercepto)"]
    for col, ref in cats.items():
        levels = [lv for lv in sorted(frame[col].astype(str).unique()) if lv != str(ref)]
        for lv in levels:
            parts.append(frame[col].astype(str).eq(lv).to_numpy(float)[:, None])
            names.append(f"{col}={lv}")
    for col in nums:
        parts.append(frame[col].to_numpy(float)[:, None])
        names.append(col)
    return np.hstack(parts), names


def poisson_table(frame: pd.DataFrame, cats: dict[str, str], nums: list[str] = ()) -> tuple[pd.DataFrame, float, int]:
    X, names = design(frame, cats, nums)
    fit = poisson_fit(X, frame["events"].to_numpy(float), np.log(frame["days"].to_numpy(float)))
    out = pd.DataFrame({"term": names, "coef": fit["beta"], "se": fit["se"]})
    out["rr"] = np.exp(out["coef"])
    out["rr_lo95"] = np.exp(out["coef"] - 1.96 * out["se"])
    out["rr_hi95"] = np.exp(out["coef"] + 1.96 * out["se"])
    return out, fit["deviance"], X.shape[1]


# ======================================================================================
# A · Composición
# ======================================================================================
def composition(veh: pd.DataFrame) -> dict[str, pd.DataFrame]:
    by = veh.groupby([MKT, ENG]).agg(vehiculos=(ID, "size"), eventos=("y", "sum"))
    by["frac_eventos"] = by["eventos"] / by["vehiculos"]
    mkt = veh.groupby(MKT).agg(vehiculos=(ID, "size"), eventos=("y", "sum"))
    mkt["frac_eventos"] = mkt["eventos"] / mkt["vehiculos"]
    model = pd.crosstab(veh["static_ModelSeries"], [veh[ENG], veh["y"]])
    return {"mercado_motor": by, "mercado": mkt, "modelo_motor": model,
            "eventos_por_auto": veh.loc[veh["event_observed"].eq(1), "n_events_recorded"].value_counts().sort_index()}


# ======================================================================================
# B, C · Riesgo con exposición (un registro por auto-día desde la venta hasta la salida)
# ======================================================================================
def daily_risk(veh: pd.DataFrame, dss_bins: list[int]) -> pd.DataFrame:
    v = veh.dropna(subset=["sale_date"]).copy()
    v = v[v["exit_date"] >= v["sale_date"]]
    n_days = ((v["exit_date"] - v["sale_date"]).dt.days + 1).to_numpy()
    rep = np.repeat(np.arange(len(v)), n_days)
    offs = np.concatenate([np.arange(n) for n in n_days])
    base = v.iloc[rep].reset_index(drop=True)
    day = base["sale_date"] + pd.to_timedelta(offs, unit="D")
    daily = pd.DataFrame({
        ID: base[ID], MKT: base[MKT], ENG: base[ENG], "elev_m": base["elev_m"], "prod_band": base["prod_band"],
        "dss": offs, "month": day.dt.to_period("M").astype(str),
        "sale_q": base["sale_date"].dt.to_period("Q").astype(str),
        "events": ((day == base["exit_date"]) & base["y"].eq(1)).astype(int).to_numpy(),
    })
    daily["dss_bin"] = pd.cut(daily["dss"], dss_bins, right=False).astype(str)
    return daily


def cells(daily: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    out = daily.groupby(keys, observed=True).agg(days=("events", "size"), events=("events", "sum")).reset_index()
    return out[out["days"] > 0]


def rate_table(daily: pd.DataFrame, key: str) -> pd.DataFrame:
    t = daily.groupby(key, observed=True).agg(autos_mes=("events", lambda s: len(s) / 30.44), eventos=("events", "sum"))
    t["tasa_100_autos_mes"] = 100 * t["eventos"] / t["autos_mes"]
    return t.round(2)


def hazard_models(daily: pd.DataFrame, first_month: str) -> dict[str, Any]:
    """Poisson con exposición. PER queda afuera de los modelos: 0 eventos, su coeficiente
    no existe (se reporta su exposición en las tablas de tasas)."""
    out: dict[str, Any] = {}
    daily = daily[daily[MKT].ne("PER")].copy()
    # Los meses anteriores a `first_month` casi no tienen autos vendidos ni eventos: se
    # funden en un nivel para que el IRLS no persiga un coeficiente de −∞.
    daily["month"] = daily["month"].where(daily["month"] >= first_month, f"<{first_month}")
    base = cells(daily, ["dss_bin", MKT, ENG])
    t1, dev1, k1 = poisson_table(base, {"dss_bin": "[0, 60)", MKT: "COL", ENG: "ENG_2"})
    out["glm_dss_mercado_motor"] = t1

    with_elev = daily.dropna(subset=["elev_m"]).copy()
    with_elev["elev_km"] = with_elev["elev_m"] / 1000.0
    c2 = with_elev.groupby(["dss_bin", MKT, ENG, "elev_km"], observed=True).agg(
        days=("events", "size"), events=("events", "sum")).reset_index()
    t2, _, _ = poisson_table(c2, {"dss_bin": "[0, 60)", MKT: "COL", ENG: "ENG_2"}, ["elev_km"])
    out["glm_con_altura"] = t2
    # En COL solo falla ENG_2 (67 de 67): el modelo de altura se ajusta dentro de ENG_2.
    col = with_elev[with_elev[MKT].eq("COL") & with_elev[ENG].eq("ENG_2")]
    c3 = col.groupby(["dss_bin", "elev_km"], observed=True).agg(days=("events", "size"), events=("events", "sum")).reset_index()
    t3, _, _ = poisson_table(c3, {"dss_bin": "[0, 60)"}, ["elev_km"])
    out["glm_altura_solo_col"] = t3

    # C · ¿el mes calendario agrega algo sobre los días desde la venta y el mercado?
    cm = cells(daily, ["dss_bin", MKT, ENG, "month"])
    t_no, dev_no, k_no = poisson_table(cm, {"dss_bin": "[0, 60)", MKT: "COL", ENG: "ENG_2"})
    t_m, dev_m, k_m = poisson_table(cm, {"dss_bin": "[0, 60)", MKT: "COL", ENG: "ENG_2", "month": "2025-12"})
    lr = dev_no - dev_m
    out["lr_mes"] = {"lr": lr, "df": k_m - k_no, "p": float(chi2.sf(lr, k_m - k_no))}
    out["glm_mes"] = t_m[t_m["term"].str.startswith("month=")]
    # Mismo test con la cohorte de venta (trimestre): ¿los vendidos antes fallan más a igual edad?
    cq = cells(daily, ["dss_bin", MKT, ENG, "sale_q"])
    _, dq0, kq0 = poisson_table(cq, {"dss_bin": "[0, 60)", MKT: "COL", ENG: "ENG_2"})
    tq, dq1, kq1 = poisson_table(cq, {"dss_bin": "[0, 60)", MKT: "COL", ENG: "ENG_2", "sale_q": "2025Q2"})
    out["lr_trimestre_venta"] = {"lr": dq0 - dq1, "df": kq1 - kq0, "p": float(chi2.sf(dq0 - dq1, kq1 - kq0))}
    out["glm_trimestre_venta"] = tq[tq["term"].str.startswith("sale_q=")]
    return out


# ======================================================================================
# D · Reloj del evento
# ======================================================================================
def event_clock(veh: pd.DataFrame, trips: pd.DataFrame, cfg: dict[str, Any], rng: np.random.Generator) -> dict[str, Any]:
    first = trips.groupby(ID).agg(first_odo=("OdometerTripStart", "min"), first_date=("date", "min"),
                                  last_date=("date", "max"))
    f = veh[veh["y"].eq(1)].join(first, on=ID)
    f = f[f["first_odo"].le(cfg["max_first_trip_odo_km"]) & f["event_date_rec"].between(f["first_date"], f["last_date"])]
    f = f[f["event_odo_km"].gt(0)]
    age = (f["event_date_rec"] - f["prod_date"]).dt.days.astype(float)
    dss = (f["event_date_rec"] - f["sale_date"]).dt.days.astype(float)
    odo = f["event_odo_km"].astype(float)
    km_day = (odo - f["first_odo"]) / (f["event_date_rec"] - f["first_date"]).dt.days.clip(lower=1)

    def sdlog(x: pd.Series) -> float:
        return float(np.std(np.log(x[x > 0]), ddof=1))

    boot = []
    for _ in range(int(cfg["n_bootstrap"])):
        idx = rng.integers(0, len(f), len(f))
        boot.append(sdlog(odo.iloc[idx]) - sdlog(age.iloc[idx]))
    return {
        "n": int(len(f)),
        "sdlog_edad": sdlog(age), "sdlog_dias_desde_venta": sdlog(dss), "sdlog_odometro": sdlog(odo),
        "dif_odo_menos_edad": sdlog(odo) - sdlog(age),
        "dif_ic95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        "edad_mediana": float(age.median()), "dss_mediana": float(dss.median()),
        "dss_iqr": [float(dss.quantile(0.25)), float(dss.quantile(0.75))], "odo_mediana": float(odo.median()),
        "rho_kmdia_km": float(spearmanr(km_day, odo).statistic),
        "rho_kmdia_dias": float(spearmanr(km_day, dss).statistic),
    }


# ======================================================================================
# E · Perfil alineado, dentro del mercado
# ======================================================================================
METRIC_SPECS = {
    "idle_frac": ("idle", "mean"),
    "below_regime_frac": ("below_regime", "mean"),
    "below_regime_moving_frac": ("below_regime_moving", "mean"),
    "short_moving_frac": ("short_trip_moving", "mean"),
    "urban_moving_frac": ("urban_moving", "mean"),
    "speed_moving_median": ("speed_kmh_moving", "median"),
    "trip_km_moving_median": ("trip_km_moving", "median"),
    "duration_moving_median": ("trip_duration_min_moving", "median"),
    "coolant_end_mean": ("CoolantTemperatureEnd", "mean"),
    "engine_temp_max_moving_mean": ("EngineTemperatureMax_moving", "mean"),
    "cold_start_frac": ("cold_start", "mean"),
    "chained_frac": ("chained", "mean"),
    "dpf_end_mean": ("AirRegenerationEnd", "mean"),
    "dpf_saturated_frac": ("dpf_saturated_end", "mean"),
    "filter_abnormal_frac": ("filter_abnormal_end", "mean"),
    "cleaning_auto_frac": ("filter_cleaning_auto_end", "mean"),
    "fuel_per_100km_median": ("fuel_pct_per_100km", "median"),
    "air_temp_mean": ("AirTemperatureAvg", "mean"),
}
COUNT_SPECS = {"idle": "idle_per_1000km", "regen_drop": "regen_per_1000km", "long_idle": "long_idle_per_1000km"}


def window_metrics(frame: pd.DataFrame, keys: list[str], min_trips: int) -> pd.DataFrame:
    g = frame.groupby(keys, observed=True)
    agg = g.agg(n_trips=("trip_km", "size"), km=("trip_km", "sum"), **{k: v for k, v in METRIC_SPECS.items()},
                **{f"_{c}": (c, "sum") for c in COUNT_SPECS})
    agg = agg[agg["n_trips"] >= min_trips]
    for col, name in COUNT_SPECS.items():
        agg[name] = agg[f"_{col}"] / (agg["km"] / 1000.0).where(agg["km"] > 0)
    return agg.drop(columns=[f"_{c}" for c in COUNT_SPECS]).reset_index()


def odometer_at(trips_v: pd.DataFrame, when: pd.Timestamp) -> float:
    # Las dos puntas en segundos: pandas 3 guarda las fechas en µs y `Timestamp.value`
    # está en ns; mezclarlas deja la referencia 1000× afuera y np.interp la pega al final.
    t = trips_v.sort_values("date")
    x = t["date"].to_numpy().astype("datetime64[s]").astype("int64")
    w = np.datetime64(when.to_datetime64(), "s").astype("int64")
    return float(np.interp(w, x, t["OdometerTripEnd"].to_numpy(float)))


def aligned_within_market(veh: pd.DataFrame, trips: pd.DataFrame, cfg: dict[str, Any],
                          rng: np.random.Generator) -> pd.DataFrame:
    metrics = list(METRIC_SPECS) + list(COUNT_SPECS.values())
    bin_km = float(cfg["odo_bin_km"])
    healthy = trips[trips["y"].eq(0)].copy()
    healthy["odo_bin"] = (np.floor(healthy["OdometerTripEnd"] / bin_km) * bin_km).astype(int)
    healthy["month"] = healthy["date"].dt.to_period("M").astype(str)
    h_odo = window_metrics(healthy, [ID, "odo_bin"], cfg["min_trips"])
    h_mon = window_metrics(healthy, [ID, "month"], cfg["min_trips"])
    mk = veh.set_index(ID)[MKT]
    h_odo[MKT] = h_odo[ID].map(mk)
    h_mon[MKT] = h_mon[ID].map(mk)
    groups_odo = {k: g for k, g in h_odo.groupby([MKT, "odo_bin"])}
    groups_odo_pooled = {k: g for k, g in h_odo.groupby("odo_bin")}
    groups_mon = {k: g for k, g in h_mon.groupby([MKT, "month"])}

    failed = veh[veh["y"].eq(1)]
    by_vehicle = {v: g for v, g in trips[trips["y"].eq(1)].groupby(ID)}
    rows = []
    for offset in cfg["offsets_days"]:
        for r in failed.itertuples():
            tv = by_vehicle.get(r.vehicle_id)
            if tv is None:
                continue
            ref = r.event_date_rec - pd.Timedelta(days=int(offset))
            if not (tv["date"].min() <= ref <= tv["date"].max()):
                continue  # la referencia cae fuera de la telemetría: el eje de km no está anclado
            odo_ref = odometer_at(tv, ref)
            before = odo_ref - tv["OdometerTripEnd"].to_numpy(float)
            for tramo, (lo, hi) in cfg["tramos_km"].items():
                sel = tv[(before >= lo) & (before < hi)]
                if len(sel) < cfg["min_trips"]:
                    continue
                m = window_metrics(sel.assign(_k=0), [ID, "_k"], cfg["min_trips"])
                if m.empty:
                    continue
                vals = m.iloc[0]
                odo_mid = odo_ref - (lo + hi) / 2.0
                obin = int(np.floor(odo_mid / bin_km) * bin_km)
                month = str(sel["date"].median().to_period("M"))
                comps = {
                    "mercado_odometro": groups_odo.get((r.static_SalesCountry_cd, obin)),
                    "agrupado_odometro": groups_odo_pooled.get(obin),
                    "mercado_mes": groups_mon.get((r.static_SalesCountry_cd, month)),
                }
                for comp_name, comp in comps.items():
                    if comp is None or len(comp) < cfg["min_comparators"]:
                        continue
                    for metric in metrics:
                        x = vals[metric]
                        ref_vals = comp[metric].dropna().to_numpy(float)
                        if pd.isna(x) or len(ref_vals) < cfg["min_comparators"]:
                            continue
                        score = float(np.mean(ref_vals < x) + 0.5 * np.mean(ref_vals == x))
                        rows.append({"offset_d": offset, "tramo": tramo, "comparador": comp_name, "metric": metric,
                                     ID: r.vehicle_id, "mercado": r.static_SalesCountry_cd, "score": score})
    scores = pd.DataFrame(rows)

    def summarize(g: pd.DataFrame) -> pd.Series:
        s = g["score"].to_numpy()
        boot = [np.mean(s[rng.integers(0, len(s), len(s))]) for _ in range(int(cfg["n_bootstrap"]))]
        return pd.Series({"p": float(np.mean(s)), "lo90": float(np.percentile(boot, 5)),
                          "hi90": float(np.percentile(boot, 95)), "n": len(s)})

    summary = scores.groupby(["offset_d", "comparador", "metric", "tramo"]).apply(summarize).reset_index()
    return scores, summary


# ======================================================================================
# F · Rasgo temprano post-venta
# ======================================================================================
def stratified_auc(frame: pd.DataFrame, value: str, strata: str) -> dict[str, float]:
    num, den, pooled_n = 0.0, 0.0, 0
    for _, g in frame.dropna(subset=[value]).groupby(strata):
        a, b = g.loc[g["y"].eq(1), value], g.loc[g["y"].eq(0), value]
        if len(a) < 3 or len(b) < 3:
            continue
        u = mannwhitneyu(a, b, alternative="two-sided").statistic
        num += u
        den += len(a) * len(b)
        pooled_n += len(a) + len(b)
    d = frame.dropna(subset=[value])
    a, b = d.loc[d["y"].eq(1), value], d.loc[d["y"].eq(0), value]
    pooled = mannwhitneyu(a, b, alternative="two-sided").statistic / (len(a) * len(b)) if len(a) and len(b) else np.nan
    return {"auc_estratificado": num / den if den else np.nan, "auc_agrupado": float(pooled), "n": pooled_n}


def early_trait(veh: pd.DataFrame, trips: pd.DataFrame, cfg: dict[str, Any], end: pd.Timestamp) -> pd.DataFrame:
    rows = []
    sale = veh.set_index(ID)["sale_date"]
    for L in cfg["landmarks_days"]:
        t = trips.assign(_sale=trips[ID].map(sale))
        t = t[(t["date"] >= t["_sale"]) & (t["date"] < t["_sale"] + pd.Timedelta(days=L))]
        m = window_metrics(t.assign(_k=0), [ID, "_k"], cfg["min_trips"]).drop(columns="_k")
        v = veh.set_index(ID)
        horizon = v["sale_date"] + pd.Timedelta(days=L + cfg["gap_days"])
        eligible = (v["y"].eq(1) & v["event_date_rec"].gt(horizon)) | (v["y"].eq(0) & horizon.le(end))
        m = m[m[ID].map(eligible).fillna(False).astype(bool)]
        m["y"] = m[ID].map(v["y"])
        m[MKT] = m[ID].map(v[MKT])
        m["mkt_saleq"] = m[MKT] + "|" + m[ID].map(v["sale_date"].dt.to_period("Q").astype(str))
        for metric in list(METRIC_SPECS) + list(COUNT_SPECS.values()):
            a = stratified_auc(m, metric, MKT)
            b = stratified_auc(m, metric, "mkt_saleq")
            rows.append({"landmark_d": L, "metric": metric, "n_fallados": int(m["y"].sum()),
                         "n_sanos": int((m["y"] == 0).sum()), "auc_agrupado": a["auc_agrupado"],
                         "auc_mercado": a["auc_estratificado"], "auc_mercado_trim_venta": b["auc_estratificado"]})
    return pd.DataFrame(rows)


# ======================================================================================
# G · Estáticas de exposición dentro de la ventana de producción
# ======================================================================================
def exposure_statics(veh: pd.DataFrame, daily: pd.DataFrame) -> dict[str, Any]:
    v = veh.dropna(subset=["sale_date"]).copy()
    v["sale_ord"] = (v["sale_date"] - v["sale_date"].min()).dt.days
    rho = {c: float(spearmanr(v[c], v["y"], nan_policy="omit").statistic)
           for c in ("static_ProductionDay", "static_daysUntilSale", "sale_ord")}
    rho_mkt = {}
    for c in ("static_ProductionDay", "sale_ord"):
        num = den = 0.0
        for _, g in v.groupby(MKT):
            if g["y"].nunique() < 2:
                continue
            u = mannwhitneyu(g.loc[g["y"].eq(1), c], g.loc[g["y"].eq(0), c]).statistic
            num += u
            den += g["y"].sum() * (g["y"] == 0).sum()
        rho_mkt[c] = num / den if den else np.nan
    q = pd.qcut(v["static_ProductionDay"], 5, labels=False)
    by_q = v.assign(q=q).groupby("q").agg(n=(ID, "size"), eventos=("y", "sum"))
    by_q["tasa_vehiculo"] = by_q["eventos"] / by_q["n"]
    return {"rho_con_evento": rho, "auc_dentro_mercado": rho_mkt, "por_quintil_produccion": by_q}


# ======================================================================================
# H · Alrededor de la fecha registrada
# ======================================================================================
def around_event(veh: pd.DataFrame, trips: pd.DataFrame, cfg: dict[str, Any], origin: pd.Timestamp,
                 v1_failed: str) -> pd.DataFrame:
    old = pd.read_csv(resolve_path(v1_failed), encoding="utf-8-sig")
    old["v1_date"] = origin + pd.to_timedelta(old["ProductionDay"] + old["IdentificationDate"], unit="D")
    v1 = old.set_index("VehicleCode")["v1_date"]
    daily = trips.groupby([ID, "day"]).agg(viajes=(ID, "size"), idle=("idle", "sum"), long_idle=("long_idle", "sum"),
                                           km=("trip_km", "sum")).reset_index()
    lo, hi = -int(cfg["days_before"]), int(cfg["days_after"])
    b_lo, b_hi = cfg["baseline_days"]
    out = []
    failed = veh[veh["y"].eq(1)]
    for ref_name in ("v2_registrada", "v1_entrega1"):
        for r in failed.itertuples():
            ref = r.event_date_rec if ref_name == "v2_registrada" else v1.get(r.vehicle_id)
            if ref is None or pd.isna(ref):
                continue
            d = daily[daily[ID].eq(r.vehicle_id)].set_index("day")
            span = pd.date_range(ref + pd.Timedelta(days=b_lo), ref + pd.Timedelta(days=hi))
            x = d.reindex(span)
            rel = (span - ref).days
            out.append(pd.DataFrame({"ref": ref_name, ID: r.vehicle_id, "rel": rel, "activo": x["viajes"].notna().to_numpy(),
                                     "long_idle": x["long_idle"].fillna(0).to_numpy(), "idle": x["idle"].fillna(0).to_numpy()}))
    days = pd.concat(out, ignore_index=True)
    base = days[days["rel"].between(b_lo, b_hi)].groupby(["ref"]).agg(base_long_idle=("long_idle", "mean"),
                                                                    base_activo=("activo", "mean"))
    days = days[days["rel"].between(lo, hi)]
    days["bin"] = pd.cut(days["rel"], [lo - 1, -45, -30, -21, -14, -7, -3, 0, 3, 7, 14, hi], right=True)
    prof = days.groupby(["ref", "bin"], observed=True).agg(n_autos=(ID, "nunique"), activo=("activo", "mean"),
                                                            long_idle_por_dia=("long_idle", "mean"),
                                                            idle_por_dia=("idle", "mean")).reset_index()
    prof = prof.join(base, on="ref")
    prof["long_idle_vs_base"] = prof["long_idle_por_dia"] / prof["base_long_idle"]
    return prof


# ======================================================================================
def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/eda_v2.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg["seed"]))
    rng = np.random.default_rng(int(cfg["seed"]))
    out_dir = ensure_dir(cfg["output_dir"])
    end = pd.Timestamp(cfg["extraction_end"])

    veh, trips, signals, origin = load(cfg)
    summary: dict[str, Any] = {"n_dev": int(len(veh)), "n_eventos": int(veh["y"].sum()),
                               "n_eventos_despues_del_fin": int((veh["event_observed"].eq(1) & veh["y"].eq(0)).sum()),
                               "origen": str(origin.date())}
    print(f"dev: {len(veh)} vehículos · {int(veh['y'].sum())} eventos hasta {end.date()} · "
          f"{summary['n_eventos_despues_del_fin']} con el evento registrado después (censurados ahí)")

    # A
    comp = composition(veh)
    for name, frame in comp.items():
        show(f"A · {name}", frame)
        frame.to_csv(out_dir / f"A_{name}.csv")
    summary["idle_frac_filas"] = float(trips["idle"].mean())

    # B, C
    daily = daily_risk(veh, cfg["hazard"]["dss_bins"])
    for key in ("dss_bin", MKT, ENG):
        t = rate_table(daily, key)
        show(f"B · tasa por {key}", t)
        t.to_csv(out_dir / f"B_tasa_{key}.csv")
    bra = daily[daily[MKT].eq("BRA")]
    show("B · Brasil por motor", rate_table(bra, ENG))
    col = daily[daily[MKT].eq("COL")].dropna(subset=["elev_m"]).copy()
    col["banda"] = pd.cut(col["elev_m"], [-100, 1000, 2000, 5000], labels=["<1000", "1000-2000", ">=2000"])
    show("B · Colombia por altura de la ciudad", rate_table(col, "banda"))
    models = hazard_models(daily, cfg["hazard"]["first_month"])
    for name, value in models.items():
        show(f"B/C · {name}", value if not isinstance(value, dict) else json.dumps(value, indent=1),
             **({"float_format": lambda x: f"{x:.3f}"} if isinstance(value, pd.DataFrame) else {}))
        if isinstance(value, pd.DataFrame):
            value.to_csv(out_dir / f"BC_{name}.csv", index=False)
    summary["lr_mes"] = models["lr_mes"]
    summary["lr_trimestre_venta"] = models["lr_trimestre_venta"]
    summary["rr"] = {r.term: [round(r.rr, 3), round(r.rr_lo95, 3), round(r.rr_hi95, 3)]
                     for r in models["glm_dss_mercado_motor"].itertuples() if r.term != "(intercepto)"}
    elev = models["glm_con_altura"].set_index("term").loc["elev_km"]
    elev_col = models["glm_altura_solo_col"].set_index("term").loc["elev_km"]
    summary["rr_altura_por_1000m"] = [round(elev.rr, 3), round(elev.rr_lo95, 3), round(elev.rr_hi95, 3)]
    summary["rr_altura_por_1000m_solo_col"] = [round(elev_col.rr, 3), round(elev_col.rr_lo95, 3), round(elev_col.rr_hi95, 3)]

    # D
    clock = event_clock(veh, trips, cfg["clock"], rng)
    show("D · reloj del evento", json.dumps(clock, indent=1))
    summary["reloj"] = clock

    # E
    scores, aligned = aligned_within_market(veh, trips, cfg["aligned"], rng)
    scores.to_csv(out_dir / "E_scores.csv", index=False)
    aligned.to_csv(out_dir / "E_perfil_alineado.csv", index=False)
    for offset in cfg["aligned"]["offsets_days"]:
        for comp_name in ("mercado_odometro", "agrupado_odometro", "mercado_mes"):
            sub = aligned[(aligned["offset_d"] == offset) & (aligned["comparador"] == comp_name)]
            piv = sub.pivot(index="metric", columns="tramo", values="p")[list(cfg["aligned"]["tramos_km"])]
            show(f"E · P(fallado > sano) · referencia = registrada − {offset} d · comparador {comp_name}",
                 piv, float_format=lambda x: f"{x:.3f}")

    # F
    early = early_trait(veh, trips, cfg["early"], end)
    early.to_csv(out_dir / "F_rasgo_temprano.csv", index=False)
    show("F · rasgo temprano (AUC por vehículo)", early.round(3))

    # G
    stat = exposure_statics(veh, daily)
    show("G · estáticas de exposición", json.dumps({k: v for k, v in stat.items() if k != "por_quintil_produccion"}, indent=1))
    show("G · tasa por quintil de ProductionDay (dentro de la ventana)", stat["por_quintil_produccion"])
    summary["estaticas"] = {k: v for k, v in stat.items() if k != "por_quintil_produccion"}

    # H
    around = around_event(veh, trips, cfg["around_event"], origin, cfg["v1_static"]["failed"])
    around.to_csv(out_dir / "H_alrededor_del_evento.csv", index=False)
    show("H · idle largo (>= 15 min a 0 km) alrededor de la fecha", around.round(3))

    # P
    if cfg.get("production_window_days") is not None:
        comp = pd.crosstab([veh["prod_band"]], [veh[MKT], veh["y"]], margins=True)
        show("P · vehículos por período de producción, mercado y evento", comp)
        comp.to_csv(out_dir / "P_produccion_mercado.csv")
        rates = rate_table(daily, "prod_band")
        show("P · tasa con exposición por período de producción", rates)
        rates.to_csv(out_dir / "P_tasa_produccion.csv")
        v = veh.dropna(subset=["static_ProductionDay"])
        auc_mkt = stratified_auc(v.assign(neg_pd=-v["static_ProductionDay"]), "neg_pd", MKT)
        auc_in = stratified_auc(v[v["prod_band"] == "dentro"].assign(neg_pd=-v["static_ProductionDay"]), "neg_pd", MKT)
        summary["produccion"] = {
            "por_periodo": {b: {"fallados": int(g["y"].sum()), "sanos": int((g["y"] == 0).sum())}
                            for b, g in veh.groupby("prod_band")},
            "auc_menos_production_day": {"agrupado": auc_mkt["auc_agrupado"], "dentro_mercado": auc_mkt["auc_estratificado"],
                                         "solo_dentro_de_la_ventana": auc_in["auc_estratificado"]},
        }
        show("P · AUC por vehículo de −ProductionDay sola", json.dumps(summary["produccion"], indent=1))

    # I
    rep = veh[veh["event_observed"].eq(1)]["n_events_recorded"]
    summary["eventos_repetidos"] = {"autos_con_mas_de_uno": int((rep > 1).sum()), "de": int(len(rep))}

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=float),
                                          encoding="utf-8")
    print(f"\nEscrito en {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
