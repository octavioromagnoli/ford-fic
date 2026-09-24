#!/usr/bin/env python
"""¿La telemetría deja una marca de intervención en la fecha del evento? (Fase 0, solo dev)

    python scripts/audit_event_dating.py --config configs/data/event_dating.yaml [--refresh]

Es la Fase 0 de "datar eventos": antes de intentar fechar a los fallados sin fecha
utilizable, mide sobre los fallados de dev que sí la tienen si alguna marca discreta de
intervención (días sin uso, reseteo de aceite, caída del DPF con el motor apagado, km sin
telemetría, mensajes de limpieza manual o de filtro al límite) cae cerca del evento.

Por marca y tolerancia k (días):
- **acierto**: fracción de fallados elegibles con una marca a ≤ k días de su fecha;
- **nulo**: la misma probabilidad en los días del mismo auto lejos del evento (conserva la
  frecuencia de marcas de cada auto), con p de Monte Carlo;
- **sanos**: la misma probabilidad en los días de los sanos dentro de la ventana;
- **regla de fechado**: la primera marca dentro de [max(venta, inicio de ventana),
  min(fin de ventana, último viaje)], su error sobre los fechados y qué fracción de
  fallados y de sanos tiene alguna marca en ese intervalo.

El veredicto por marca sale del bloque `decision` del YAML, fijado antes de medir. Las
marcas no son features del panel: fechar con una feature pone la regla en la etiqueta.
No toca a los excluidos ni al test, no cambia etiquetas y no entrena nada.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.config import ensure_dir, load_config, resolve_path  # noqa: E402
from src.data.anchor import EPOCH  # noqa: E402
from src.data.join import load_vehicle_static  # noqa: E402
from src.data.subset import read_table_for_vehicles  # noqa: E402
from src.eval.splits import load_test_split  # noqa: E402
from src.features.signals import MESSAGE_SLUGS  # noqa: E402
from src.features.trips import MANUAL_LEVELS  # noqa: E402

logger = logging.getLogger("audit_event_dating")

ID = "vehicle_id"
DAY = pd.Timedelta(days=1)
TRIP_COLUMNS = [ID, "TripDatetimeStart", "TripDatetimeEnd", "OdometerTripStart", "OdometerTripEnd",
                "EngineOilLifePCStart", "EngineOilLifePCEnd", "AirRegenerationStart", "AirRegenerationEnd",
                "AirFilterStart", "AirFilterEnd"]


def naive_utc(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, utc=True, errors="coerce").dt.tz_convert(None)


# --- entrada ------------------------------------------------------------------------

def load_trips(cfg: dict[str, Any], dev: set[str], refresh: bool) -> pd.DataFrame:
    cache = ensure_dir(cfg["output_dir"]) / cfg["trips_cache"]
    if cache.exists() and not refresh:
        trips = pd.read_parquet(cache)
    else:
        raw, _ = read_table_for_vehicles("trips", None, dev, sources=cfg["sources"], dedupe=cfg["dedupe"])
        trips = raw[TRIP_COLUMNS].copy()
        for c in ("TripDatetimeStart", "TripDatetimeEnd"):
            trips[c] = naive_utc(trips[c])
        trips.to_parquet(cache, index=False)
    foreign = set(trips[ID].astype(str).unique()) - dev
    if foreign:
        raise ValueError(f"{len(foreign)} vehículo(s) fuera de dev en los viajes")
    trips = trips.dropna(subset=["TripDatetimeStart"])
    return trips.sort_values([ID, "TripDatetimeStart"], kind="stable", ignore_index=True)


def load_signal_messages(cfg: dict[str, Any], dev: set[str], refresh: bool) -> pd.DataFrame:
    """Solo las filas de `signals` con un mensaje de alguna marca (el resto no hace falta)."""
    cache = ensure_dir(cfg["output_dir"]) / cfg["signals_cache"]
    wanted = set(cfg["markers"]["manual_clean"]["slugs"]) | set(cfg["markers"]["severe_msg"]["slugs"])
    if cache.exists() and not refresh:
        sig = pd.read_parquet(cache)
    else:
        raw, _ = read_table_for_vehicles("signals", None, dev, sources=cfg["sources"], dedupe=cfg["dedupe"])
        raw["slug"] = raw["Message"].map(MESSAGE_SLUGS)
        sig = raw.loc[raw["slug"].isin(wanted), [ID, "eventTimestamp", "slug"]].copy()
        sig["eventTimestamp"] = naive_utc(sig["eventTimestamp"])
        sig.to_parquet(cache, index=False)
    foreign = set(sig[ID].astype(str).unique()) - dev
    if foreign:
        raise ValueError(f"{len(foreign)} vehículo(s) fuera de dev en las señales")
    return sig.dropna(subset=["eventTimestamp"])


def build_vehicles(cfg: dict[str, Any], dev: set[str], trips: pd.DataFrame) -> pd.DataFrame:
    static = load_vehicle_static(config_path=cfg["sources"], dedupe_config=cfg["dedupe"],
                                 panel_config=cfg["panel_config"])
    static = static[static[ID].astype(str).isin(dev)].set_index(ID)
    if len(static) != len(dev):
        raise ValueError(f"{len(dev) - len(static)} vehículo(s) de dev sin fila estática")
    meta = json.loads(resolve_path(cfg["anchor_meta"]).read_text(encoding="utf-8"))
    origin = float(meta["anchor"]["origin_day_since_epoch"])
    v = pd.DataFrame(index=static.index)
    v["event_observed"] = static["event_observed"].astype(int)
    v["market"] = static["static_SalesCountry_cd"].astype(str)
    prod = (EPOCH + pd.to_timedelta(origin + pd.to_numeric(static["static_ProductionDay"]), unit="D")).dt.tz_convert(None)
    v["sale_date"] = prod + pd.to_timedelta(pd.to_numeric(static["static_daysUntilSale"]), unit="D")
    age = pd.to_numeric(static["event_day_since_production"], errors="coerce").where(v["event_observed"].eq(1))
    v["event_date"] = prod + pd.to_timedelta(age, unit="D")
    g = trips.groupby(ID, observed=True)
    v["first_trip"] = g["TripDatetimeStart"].min()
    v["last_trip"] = g["TripDatetimeEnd"].max()
    return v


# --- marcas -------------------------------------------------------------------------

def build_markers(trips: pd.DataFrame, sig: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Una fila por marca: (vehicle_id, marker, a, b), intervalo de calendario."""
    m = cfg["markers"]
    t = trips
    g = t.groupby(ID, observed=True)
    prev_end = g["TripDatetimeEnd"].shift()
    prev_oil = g["EngineOilLifePCEnd"].shift().fillna(g["EngineOilLifePCStart"].shift())
    prev_dpf = g["AirRegenerationEnd"].shift()
    prev_odo = g["OdometerTripEnd"].shift()
    rest_days = (t["TripDatetimeStart"] - prev_end) / DAY
    between = pd.DataFrame({ID: t[ID], "a": prev_end, "b": t["TripDatetimeStart"]})
    rules = {
        "gap": rest_days.ge(m["gap"]["min_days"]),
        "gap_long": rest_days.ge(m["gap_long"]["min_days"]),
        "oil_reset": (t["EngineOilLifePCStart"] - prev_oil).ge(m["oil_reset"]["min_jump_points"]),
        "dpf_offtrip": (t["AirRegenerationStart"] - prev_dpf).le(-m["dpf_offtrip"]["min_drop_points"]),
        "odo_jump": (t["OdometerTripStart"] - prev_odo).ge(m["odo_jump"]["min_km"]),
    }
    parts = [between.loc[mask & prev_end.notna()].assign(marker=name) for name, mask in rules.items()]
    manual_trip = t["AirFilterStart"].isin(MANUAL_LEVELS) | t["AirFilterEnd"].isin(MANUAL_LEVELS)
    parts.append(pd.DataFrame({ID: t.loc[manual_trip, ID], "a": t.loc[manual_trip, "TripDatetimeStart"],
                               "b": t.loc[manual_trip, "TripDatetimeEnd"], "marker": "manual_clean"}))
    for name in ("manual_clean", "severe_msg"):
        s = sig[sig["slug"].isin(m[name]["slugs"])]
        parts.append(pd.DataFrame({ID: s[ID], "a": s["eventTimestamp"], "b": s["eventTimestamp"], "marker": name}))
    out = pd.concat(parts, ignore_index=True)
    out["b"] = out["b"].where(out["b"] >= out["a"], out["a"])
    # Unión de las marcas de intervención (sin `gap` corto, que es uso normal).
    union = out[out["marker"].isin(["gap_long", "oil_reset", "dpf_offtrip", "odo_jump", "manual_clean"])]
    out = pd.concat([out, union.assign(marker="any_intervention")], ignore_index=True)
    return out.sort_values([ID, "marker", "a"], ignore_index=True)


def min_distance_days(a: np.ndarray, b: np.ndarray, dates: np.ndarray) -> np.ndarray:
    """Distancia en días de cada fecha al intervalo [a, b] más cercano (inf sin marcas)."""
    if len(a) == 0:
        return np.full(len(dates), np.inf)
    d = np.maximum(np.timedelta64(0, "ns"), np.maximum(a[None, :] - dates[:, None], dates[:, None] - b[None, :]))
    return d.min(axis=1) / np.timedelta64(1, "D")


# --- medición -----------------------------------------------------------------------

def measure(v: pd.DataFrame, markers: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    rng = np.random.default_rng(cfg["seed"])
    ks = cfg["tolerances_days"]
    win0, win1 = pd.Timestamp(cfg["event_window"]["start"]), pd.Timestamp(cfg["event_window"]["end"])
    cov = cfg["coverage_days"] * DAY
    far = cfg["null_min_distance_days"]

    failed = v[v["event_observed"].eq(1)]
    eligible = failed[(failed["first_trip"] <= failed["event_date"] - cov) & (failed["last_trip"] >= failed["event_date"] + cov)]
    healthy = v[v["event_observed"].eq(0)]
    counts = {"dev_failed": int(len(failed)), "eligible_failed": int(len(eligible)), "dev_healthy": int(len(healthy))}

    by = {key: grp for key, grp in markers.groupby([ID, "marker"], observed=True)}
    names = sorted(markers["marker"].unique())

    def arrays(vid: str, name: str) -> tuple[np.ndarray, np.ndarray]:
        grp = by.get((vid, name))
        if grp is None:
            return np.array([], dtype="datetime64[ns]"), np.array([], dtype="datetime64[ns]")
        return grp["a"].to_numpy("datetime64[ns]"), grp["b"].to_numpy("datetime64[ns]")

    rows, per_vehicle = [], []
    for name in names:
        dist_true, null_p = [], {k: [] for k in ks}
        rule_err, has_c_failed = [], []
        for vid, r in eligible.iterrows():
            a, b = arrays(vid, name)
            ev = np.datetime64(r["event_date"], "ns")
            d0 = float(min_distance_days(a, b, np.array([ev]))[0])
            dist_true.append(d0)
            # nulo: días del mismo auto lejos del evento
            span0, span1 = r["first_trip"].normalize(), r["last_trip"].normalize()
            grid = pd.date_range(span0, span1, freq="D") + (r["event_date"] - r["event_date"].normalize())
            gd = grid.to_numpy("datetime64[ns]")
            dist_grid = min_distance_days(a, b, gd)
            off = np.abs((gd - ev) / np.timedelta64(1, "D"))
            for k in ks:
                ok = (off >= far) & (gd >= np.datetime64(r["first_trip"] + k * DAY)) & (gd <= np.datetime64(r["last_trip"] - k * DAY))
                null_p[k].append(float(np.mean(dist_grid[ok] <= k)) if ok.any() else np.nan)
            # regla de fechado
            c0, c1 = max(r["sale_date"], win0), min(win1, r["last_trip"])
            inside = (b >= np.datetime64(c0)) & (a <= np.datetime64(c1))
            has_c_failed.append(bool(inside.any()))
            if inside.any():
                first = max(pd.Timestamp(a[inside].min()), c0)
                rule_err.append(abs((first - r["event_date"]) / DAY))
            else:
                rule_err.append(np.inf)
            per_vehicle.append({ID: vid, "marker": name, "dist_to_event_days": d0,
                                "rule_error_days": rule_err[-1], "market": r["market"]})
        dist_true = np.array(dist_true)
        rule_err = np.array(rule_err)

        # sanos: probabilidad por día dentro de la ventana, y si tienen marca en su intervalo
        healthy_p = {k: [] for k in ks}
        has_c_healthy = []
        for vid, r in healthy.iterrows():
            a, b = arrays(vid, name)
            c0, c1 = max(r["sale_date"], win0, r["first_trip"]), min(win1, r["last_trip"])
            if pd.isna(c0) or pd.isna(c1) or c1 <= c0:
                continue
            inside = (b >= np.datetime64(c0)) & (a <= np.datetime64(c1))
            has_c_healthy.append(bool(inside.any()))
            gd = pd.date_range(c0.normalize(), c1.normalize(), freq="D").to_numpy("datetime64[ns]")
            dist_grid = min_distance_days(a, b, gd)
            for k in ks:
                healthy_p[k].append(float(np.mean(dist_grid <= k)))

        for k in ks:
            hits = dist_true <= k
            p = np.array(null_p[k])
            valid = ~np.isnan(p)
            sims = (rng.random((cfg["n_monte_carlo"], int(valid.sum()))) < p[valid]).sum(axis=1)
            obs = int(hits[valid].sum())
            rows.append({
                "marker": name, "k_days": k, "n": int(len(dist_true)), "n_null": int(valid.sum()),
                "hit_rate": float(hits.mean()) if len(hits) else np.nan,
                "null_rate": float(p[valid].mean()) if valid.any() else np.nan,
                "p_value": float((1 + (sims >= obs).sum()) / (1 + len(sims))),
                "healthy_rate": float(np.mean(healthy_p[k])) if healthy_p[k] else np.nan,
                "rule_within": float(np.mean(rule_err <= k)) if len(rule_err) else np.nan,
                "rule_error_median": float(np.median(rule_err)) if len(rule_err) else np.nan,
                "failed_with_marker_in_window": float(np.mean(has_c_failed)) if has_c_failed else np.nan,
                "healthy_with_marker_in_window": float(np.mean(has_c_healthy)) if has_c_healthy else np.nan,
            })
    table = pd.DataFrame(rows)
    table["excess"] = table["hit_rate"] - table["null_rate"]
    return table, pd.DataFrame(per_vehicle), counts


def verdict(table: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    d = cfg["decision"]
    t = table[table["k_days"].eq(d["tolerance_days"])].copy()
    t["c1_hit"] = t["hit_rate"] >= d["min_hit_rate"]
    t["c2_excess"] = (t["excess"] >= d["min_excess_over_null"]) & (t["p_value"] < d["max_p_value"])
    t["c3_rule"] = t["rule_within"] >= d["min_rule_within"]
    t["passes"] = t["c1_hit"] & t["c2_excess"] & t["c3_rule"]
    return t[["marker", "hit_rate", "null_rate", "excess", "p_value", "rule_within", "c1_hit", "c2_excess", "c3_rule", "passes"]]


def diagnose_dpf(v: pd.DataFrame, trips: pd.DataFrame, cfg: dict[str, Any]) -> None:
    """Diagnóstico POSTERIOR al veredicto, no preregistrado: la caída más grande del DPF con
    el motor apagado dentro de la ventana, como fecha y como firma del vehículo.

    Ojo: mira toda la ventana, incluido lo posterior al evento. Es una firma de la etiqueta,
    no un predictor, y nunca puede entrar como feature.
    """
    from sklearn.metrics import roc_auc_score

    t = trips.copy()
    g = t.groupby(ID, observed=True)
    t["drop"] = t["AirRegenerationStart"] - g["AirRegenerationEnd"].shift()
    thr = -cfg["markers"]["dpf_offtrip"]["min_drop_points"]
    win0, win1 = pd.Timestamp(cfg["event_window"]["start"]), pd.Timestamp(cfg["event_window"]["end"])
    rows = []
    for vid, x in t.groupby(ID, observed=True):
        r = v.loc[vid]
        c0, c1 = max(r["sale_date"], win0, r["first_trip"]), min(win1, r["last_trip"])
        x = x[(x["TripDatetimeStart"] >= c0) & (x["TripDatetimeStart"] <= c1) & x["drop"].notna()]
        if c1 <= c0 or x.empty:
            continue
        i = x["drop"].idxmin()
        signed = (x.loc[i, "TripDatetimeStart"] - r["event_date"]) / DAY
        rows.append({"failed": int(r["event_observed"]), "max_drop": -x.loc[i, "drop"],
                     "per_100d": (x["drop"] <= thr).sum() / ((c1 - c0) / DAY) * 100,
                     "days_in_window": (c1 - c0) / DAY, "signed_days": signed})
    d = pd.DataFrame(rows)
    f = d[d["failed"].eq(1)]
    print("\n== diagnóstico posterior (no preregistrado): caída máxima del DPF con motor apagado ==")
    print(f"vehículos: {len(f)} fallados, {int(d['failed'].eq(0).sum())} sanos")
    print("como fecha: a ±7/14/30 d =",
          [round(float((f["signed_days"].abs() <= k).mean()), 3) for k in (7, 14, 30)],
          "| error con signo p10/p25/p50/p75/p90 =",
          [round(float(q), 1) for q in f["signed_days"].quantile([.1, .25, .5, .75, .9])])
    print("como firma del vehículo: AUC caída máxima =", round(roc_auc_score(d["failed"], d["max_drop"]), 3),
          "| AUC caídas por 100 d =", round(roc_auc_score(d["failed"], d["per_100d"]), 3),
          "| AUC días en ventana (largo, regla 6) =", round(roc_auc_score(d["failed"], d["days_in_window"]), 3))
    # Nulo que conserva el largo (regla 6): el máximo sobre una ventana más larga es más grande por
    # construcción. Se permuta la etiqueta dentro de terciles de días en ventana.
    rng = np.random.default_rng(cfg["seed"])
    strata = pd.qcut(d["days_in_window"], 3, labels=False, duplicates="drop")
    obs = roc_auc_score(d["failed"], d["max_drop"])
    null = []
    for _ in range(2000):
        y = d["failed"].copy()
        for s in strata.unique():
            idx = strata.eq(s).to_numpy()
            y.iloc[idx] = rng.permutation(y.iloc[idx].to_numpy())
        null.append(roc_auc_score(y, d["max_drop"]))
    null = np.array(null)
    print(f"nulo por terciles de largo: AUC media {null.mean():.3f}, p95 {np.quantile(null, .95):.3f}, "
          f"p = {(1 + (null >= obs).sum()) / (1 + len(null)):.4f}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--refresh", action="store_true", help="relee los crudos en vez del cache")
    parser.add_argument("--diagnose", action="store_true",
                        help="diagnóstico posterior, no preregistrado: caída máxima del DPF con motor apagado")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config)
    out = ensure_dir(cfg["output_dir"])

    split = load_test_split(cfg["test_split"])
    dev = {str(x) for x in split["dev_vehicles"]}
    trips = load_trips(cfg, dev, args.refresh)
    sig = load_signal_messages(cfg, dev, args.refresh)
    v = build_vehicles(cfg, dev, trips)
    markers = build_markers(trips, sig, cfg)

    freq = (markers.groupby("marker").agg(n=("a", "size"), vehicles=(ID, "nunique")))
    table, per_vehicle, counts = measure(v, markers, cfg)
    ver = verdict(table, cfg)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    print("\n== conteos ==\n", json.dumps(counts))
    print("\n== marcas en dev (todas) ==\n", freq.to_string())
    print("\n== acierto, nulo, sanos y regla de fechado ==")
    print(table.round(3).to_string(index=False))
    print(f"\n== veredicto (tolerancia {cfg['decision']['tolerance_days']} d) ==")
    print(ver.round(3).to_string(index=False))
    passed = ver.loc[ver["passes"], "marker"].tolist()
    print("\nPASAN:", passed if passed else "ninguna -> la Fase 1 no corre")

    table.to_csv(out / "summary_by_marker.csv", index=False)
    per_vehicle.to_csv(out / "per_vehicle.csv", index=False)
    freq.to_csv(out / "marker_frequency.csv")
    (out / "summary.json").write_text(json.dumps(
        {"counts": counts, "passed": passed, "verdict": ver.to_dict(orient="records")}, indent=2, default=float),
        encoding="utf-8")
    if args.diagnose:
        diagnose_dpf(v, trips, cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
