#!/usr/bin/env python
"""Panel del conjunto externo para la incidencia (F5 §3.2, Track A).

    python scripts/build_external_panel.py --config configs/data/panel_external_incidence.yaml --counts-only
    python scripts/build_external_panel.py --config configs/data/panel_external_incidence.yaml

Hace, en este orden:

1. **Quién entra** (`src/data/external.py`): reproduce el sorteo de los 1081 y lo verifica
   contra la huella de `test_split.json`; se queda con los excluidos del lado dev del padre,
   en los mercados de `external.markets`. Falla si alguno está en dev o en test.
2. **Crudos.** `trips` de esos vehículos y de nadie más, canonizados y sin filas repetidas,
   con las derivadas de `derive_trip_columns` (cache en `trips_cache`).
3. **Puente de calendario.** El origen congelado en F2 se estimó sobre dev; acá se verifica
   que `primer viaje − ProductionDay` dé el mismo origen en el conjunto externo.
4. **Clones que se escaparan del dedupe:** ningún vehículo externo puede compartir más de
   `clone_check.max_shared_trip_frac` de sus viajes con uno del universo (dev o test).
5. **Filas y features** con `build_landmark_panel` (un hito: 30 días), con la ventana de
   features fija. El evento de fecha por defecto no tiene fecha: `event_dss` queda en NaN,
   así que ningún fallado se descarta por "evento antes del gap" (trampa 2 del preregistro).
   En este panel `label` = `event_observed`: es incidencia, sin horizonte.
6. **Conteos** por mercado × cohorte (el punto de control de la Fase 1) y, sin
   `--counts-only`, el panel.

Con `--counts-only` no se escribe el panel: los conteos de la Fase 1 se miran antes del
preregistro, y ninguna feature contra la etiqueta.
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

from scripts.build_landmark_panel import _json_default, load_trips  # noqa: E402
from src.config import ensure_dir, load_config, repo_root, resolve_path, set_seed  # noqa: E402
from src.data.anchor import EPOCH, estimate_origin_day, project_dates_to_odometer  # noqa: E402
from src.data.external import MARKET_COL, reproduce_parent_split, select_external  # noqa: E402
from src.data.join import load_vehicle_static  # noqa: E402
from src.data.landmark import (  # noqa: E402
    FM_PREFIX,
    TRIP_REQUIRED,
    build_landmark_panel,
    landmark_config,
    load_fleet_features,
)
from src.eval.splits import load_test_split  # noqa: E402

logger = logging.getLogger("build_external_panel")

ID = "vehicle_id"
DAY = pd.Timedelta(days=1)
# Columnas del riesgo en la ventana del registro: con la fecha por defecto no hay entrada ni
# salida que calcular. Se sacan para que nadie las lea como si significaran algo.
RISK_COLUMNS = ["aux_dss_entry", "aux_dss_exit", "aux_event_in_window", "aux_inwindow_exposure_days",
                "aux_days_to_event_after_cut", "aux_resolved_exposure_days", "aux_resolved_negative"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/data/panel_external_incidence.yaml")
    parser.add_argument("--refresh", action="store_true", help="vuelve a leer los crudos aunque haya cache")
    parser.add_argument("--counts-only", action="store_true", help="solo los conteos de la Fase 1, sin escribir el panel")
    return parser.parse_args()


def build_vehicles(cfg: dict[str, Any], external: pd.DataFrame, trips: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Una fila por vehículo externo: reloj, evento (sin fecha si es la de por defecto) y columnas que pasan."""
    static = external.set_index(ID)
    meta = json.loads(resolve_path(cfg["anchor_meta"]).read_text(encoding="utf-8"))
    origin = float(meta["anchor"]["origin_day_since_epoch"])
    first = trips.groupby(ID, observed=True)["TripDatetimeStart"].min()
    check, stats = estimate_origin_day(first.reindex(static.index), static["static_ProductionDay"])
    if abs(check - origin) > float(cfg.get("anchor_tolerance_days", 0.5)):
        raise ValueError(f"El puente de calendario no vale en el conjunto externo: origen {check} contra {origin} "
                         "congelado sobre dev")

    production = pd.to_numeric(static["static_ProductionDay"], errors="coerce")
    until_sale = pd.to_numeric(static["static_daysUntilSale"], errors="coerce")
    event_age = pd.to_numeric(static["event_day_since_production"], errors="coerce")
    v = pd.DataFrame(index=static.index)
    v["event_observed"] = static["event_observed"].astype(int)
    default = static["default_event_date"].astype(bool)
    dated = v["event_observed"].eq(1) & ~default
    prod_date = EPOCH + pd.to_timedelta(origin + production, unit="D")
    v["sale_date"] = prod_date + pd.to_timedelta(until_sale, unit="D")
    # Fecha por defecto = sin fecha: el evento cayó en algún momento después de la venta.
    v["event_dss"] = (event_age - until_sale).where(dated)
    v["last_trip"] = trips.groupby(ID, observed=True)["TripDatetimeStart"].max()
    event_date = (prod_date + pd.to_timedelta(event_age, unit="D")).where(dated)
    v["event_odo_km"] = project_dates_to_odometer(event_date, trips)["event_odo_km"]
    for column in cfg["static_columns"]:
        v[f"static_{column}"] = static[f"static_{column}"]
    for column in cfg["aux_static"]:
        v[f"aux_static_{column}"] = static[f"static_{column}"]
    v["aux_sale_day"] = production + until_sale
    v["aux_default_event_date"] = default.astype(int)
    info = {"origin_day_since_epoch": origin, "origin_reestimated_on_external": check, **stats}
    return v, info


def clone_check(cfg: dict[str, Any], trips: pd.DataFrame) -> dict[str, Any]:
    """Máxima fracción de viajes que un vehículo externo comparte con uno del universo."""
    spec = cfg["clone_check"]
    universe = pd.read_parquet(resolve_path(spec["universe_trips"]), columns=[ID, "TripDatetimeStart", "OdometerTripEnd"])

    def keys(frame: pd.DataFrame) -> pd.DataFrame:
        start = frame["TripDatetimeStart"].dt.tz_convert(None).astype("datetime64[ns]").astype("int64") // 10**9
        return pd.DataFrame({ID: frame[ID].astype(str).to_numpy(), "start": start.to_numpy(),
                             "odo": np.round(frame["OdometerTripEnd"].to_numpy(dtype=float), 1)}).drop_duplicates()

    ext, uni = keys(trips), keys(universe)
    shared = ext.merge(uni, on=["start", "odo"], suffixes=("", "_universe"))
    per_pair = shared.groupby([ID, f"{ID}_universe"]).size()
    n_trips = ext.groupby(ID).size()
    frac = (per_pair / n_trips.reindex(per_pair.index.get_level_values(0)).to_numpy()) if len(per_pair) else per_pair
    worst = float(frac.max()) if len(frac) else 0.0
    out = {"max_shared_trip_frac": worst, "pairs_with_any_shared_trip": int(len(per_pair)),
           "threshold": float(spec["max_shared_trip_frac"])}
    if worst > float(spec["max_shared_trip_frac"]):
        top = frac.sort_values(ascending=False).head(5)
        raise ValueError(f"Un vehículo externo comparte {worst:.0%} de sus viajes con uno del universo: {top.to_dict()}")
    return out


def counts_by_market(v: pd.DataFrame, trips: pd.DataFrame, panel: pd.DataFrame, lm, immobile_km: float,
                     e_min: float, sensitivity: list[float]) -> pd.DataFrame:
    """El embudo por mercado × cohorte (Fase 1): quién llega a una fila y quién es elegible."""
    landmark = lm.landmarks_days[0]
    has_sale = v["sale_date"].notna()
    land = v["sale_date"] + pd.Timedelta(days=landmark)
    reached = has_sale & (land <= v["last_trip"])
    t = trips.join(v[["sale_date"]], on=ID)
    in_window = (t["TripDatetimeStart"] > t["sale_date"]) & (t["TripDatetimeEnd"] <= t[ID].map(land))
    n_trips = t.loc[in_window].groupby(ID, observed=True).size().reindex(v.index, fill_value=0)
    before_gap = v["event_dss"] <= landmark + lm.gap_days
    rows = panel.set_index(ID)
    frame = pd.DataFrame({
        "market": v[MARKET_COL].astype(str),
        "cohort": np.where(v["event_observed"].eq(1), "fallado", "sano"),
        "vehiculos": 1,
        "con_venta": has_sale.astype(int),
        "llega_al_hito": reached.astype(int),
        "evento_fechado_antes_del_gap": (reached & before_gap).astype(int),
        "pocos_viajes": (reached & ~before_gap & (n_trips < lm.min_trips)).astype(int),
        "fila": v.index.isin(rows.index).astype(int),
    }, index=v.index)
    frame["quietos"] = rows["window_km"].lt(immobile_km).reindex(v.index, fill_value=False).astype(int)
    for e in [e_min, *sensitivity]:
        eligible = rows["aux_potential_exposure_days"].ge(e).reindex(v.index, fill_value=False)
        frame[f"elegibles_e{e:g}"] = eligible.astype(int)
    out = frame.groupby(["market", "cohort"]).sum(numeric_only=True)
    total = frame.groupby("cohort").sum(numeric_only=True)
    total.index = pd.MultiIndex.from_product([["total"], total.index])
    return pd.concat([out, total]).reset_index()


def sale_dates(v: pd.DataFrame, panel: pd.DataFrame, lm) -> pd.DataFrame:
    """Fecha de venta por mercado × cohorte, entre los que tienen fila, contra la ventana del registro."""
    rows = v.loc[v.index.isin(panel[ID])]
    out = []
    for (market, event), g in rows.groupby([MARKET_COL, "event_observed"]):
        sale = g["sale_date"].dt.tz_convert(None)
        out.append({
            "market": market, "cohort": "fallado" if event else "sano", "n": int(len(g)),
            "p10": sale.quantile(0.1).date(), "mediana": sale.median().date(), "p90": sale.quantile(0.9).date(),
            "max": sale.max().date(),
            "vendidos_despues_del_fin": int((g["sale_date"] > lm.window_end).sum()),
            "vendidos_antes_del_inicio": int((g["sale_date"] < lm.window_start).sum()),
        })
    return pd.DataFrame(out)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    args = parse_args()
    cfg = load_config(args.config)
    seed = set_seed(int(cfg["seed"]))
    clock = load_config(cfg["event_clock"])
    lm = landmark_config(cfg, clock["event_window"])
    if len(lm.landmarks_days) != 1:
        raise ValueError("El panel externo es de un solo hito: la incidencia es una fila por vehículo")
    features = load_fleet_features(cfg["fleet_features"], cfg["features_spec"])

    split = load_test_split(cfg["test_split"])
    static = load_vehicle_static(config_path=cfg["sources"], dedupe_config=cfg["dedupe"],
                                 panel_config=cfg["panel_config"])
    parent = reproduce_parent_split(static, split)
    ext_cfg = cfg["external"]
    external, selection = select_external(static, split, parent, side=ext_cfg["side"], markets=ext_cfg["markets"])
    ids = set(external[ID].astype(str))

    columns = {*TRIP_REQUIRED, "TripNumber", *(f.column for f in features)}
    trips = load_trips(cfg, ids, columns, refresh=args.refresh)
    missing_trips = sorted(ids - set(trips[ID].astype(str).unique()))
    clones = clone_check(cfg, trips)
    vehicles, anchor = build_vehicles(cfg, external, trips)
    panel, report = build_landmark_panel(vehicles, trips, lm, features)

    e_min = lm.e_min_days
    panel["label"] = panel["event_observed"].astype(int)
    panel["aux_default_event_date"] = panel[ID].map(vehicles["aux_default_event_date"]).astype(int)
    for e in [e_min, *lm.e_min_sensitivity_days]:
        suffix = "" if e == e_min else f"_e{e:g}"
        panel[f"aux_eligible{suffix}"] = panel["aux_potential_exposure_days"].ge(e).astype(int)
    panel = panel.drop(columns=[c for c in panel.columns if c in RISK_COLUMNS or c.startswith("aux_resolved_negative_")])

    immobile_km = float(cfg["immobile_km"])
    counts = counts_by_market(vehicles, trips, panel, lm, immobile_km, e_min, list(lm.e_min_sensitivity_days))
    sales = sale_dates(vehicles, panel, lm)
    eligible = panel.loc[panel["aux_eligible"].eq(1)]
    summary = {
        "positives_eligible": int(eligible["label"].sum()),
        "negatives_eligible": int(eligible["label"].eq(0).sum()),
        "positives_eligible_default_date": int(eligible.loc[eligible["label"].eq(1), "aux_default_event_date"].sum()),
        "rows": int(len(panel)),
        "vehicles_without_trips": missing_trips,
    }
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": cfg.get("_config_path"),
        "parent_split_sha256_16": parent["sha256_16"],
        "selection": selection,
        "anchor": anchor,
        "clone_check": clones,
        "counts_by_market": counts.to_dict(orient="records"),
        "sale_dates": sales.to_dict(orient="records"),
        "summary": summary,
    }
    counts_path = resolve_path(cfg["output"]["counts"])
    ensure_dir(counts_path.parent)
    counts_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")

    pd.set_option("display.width", 220)
    print("\n== Conjunto externo · Fase 1 (solo conteos) ==")
    print(f"sorteo padre reproducido, huella {parent['sha256_16']} · lado {ext_cfg['side']} · "
          f"mercados {ext_cfg['markets']}")
    print(f"excluidos del lado {ext_cfg['side']} por mercado:")
    print(pd.DataFrame(selection["excluded_on_side"]).to_string(index=False))
    print(f"fuera por mercado (fallados sin sanos en el conjunto): "
          f"{sum(r['vehicles'] for r in selection['dropped_by_market'])}")
    print(f"excluidos del otro lado del padre, sin tocar: {selection['excluded_on_other_side_untouched']}")
    print(f"puente de calendario: origen {anchor['origin_reestimated_on_external']:.0f} contra "
          f"{anchor['origin_day_since_epoch']:.0f} congelado (IQR {anchor['iqr_days']:.2f} d, n={anchor['n']})")
    print(f"clones: máxima fracción de viajes compartidos con el universo {clones['max_shared_trip_frac']:.3f} "
          f"({clones['pairs_with_any_shared_trip']} pares con algún viaje en común)")
    print(f"vehículos sin viajes: {len(missing_trips)}")
    print("\n-- embudo por mercado × cohorte (hito 30 d, ≥ "
          f"{lm.min_trips} viajes; elegible = exposición potencial en la ventana ≥ E días) --")
    print(counts.to_string(index=False))
    print("\n-- fecha de venta de los que tienen fila, contra la ventana "
          f"{lm.window_start.date()} → {lm.window_end.date()} --")
    print(sales.to_string(index=False))
    print(f"\nelegibles (E = {e_min:g} d): {summary['positives_eligible']} fallados "
          f"({summary['positives_eligible_default_date']} con fecha por defecto) · {summary['negatives_eligible']} sanos")

    if args.counts_only:
        rel = counts_path.relative_to(repo_root()) if counts_path.is_relative_to(repo_root()) else counts_path
        print(f"\nsolo conteos: {rel} (el panel no se escribe)")
        return 0

    out_path = resolve_path(cfg["output"]["panel"])
    panel.to_parquet(out_path, index=False)
    fm = [c for c in panel.columns if c.startswith(FM_PREFIX)]
    meta = {
        **payload,
        "seed": seed,
        "landmark": {k: (list(v) if isinstance(v, tuple) else v) for k, v in lm.__dict__.items()},
        "fleet_features": [f.__dict__ for f in features],
        "months": report["months"],
        "label": "event_observed (incidencia, sin horizonte)",
        "columns": {"feat": [c for c in panel.columns if c.startswith("feat_") and not c.startswith(FM_PREFIX)],
                    "feat_fm": len(fm),
                    "static": [c for c in panel.columns if c.startswith("static_")],
                    "aux": [c for c in panel.columns if c.startswith("aux_")]},
    }
    meta_path = resolve_path(cfg["output"]["meta"])
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    rel = out_path.relative_to(repo_root()) if out_path.is_relative_to(repo_root()) else out_path
    print(f"\nescrito: {rel} ({len(panel)} filas, {len(fm)} feat_fm__, {len(report['months'])} meses) · {meta_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
