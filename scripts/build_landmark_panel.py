#!/usr/bin/env python
"""Panel de hitos post-venta para el cure model (Fase 3, Track A).

    python scripts/build_landmark_panel.py --config configs/data/panel_landmark_ps.yaml [--refresh]

Hace, en este orden:

1. **Universo.** Los 364 vehículos de `dev_vehicles` + `test_vehicles` del holdout
   congelado, y nadie más.
2. **Crudos.** `trips` de esos vehículos, canonizados y sin filas repetidas
   (`src/data/subset.py`), con las derivadas de `derive_trip_columns`. Se cachean en
   `trips_cache`; `--refresh` los vuelve a leer.
3. **Reloj.** Origen del calendario congelado en `panel_meta.json` (se re-estima sobre dev
   y falla si no coincide); venta, evento y días post-venta al evento por vehículo; odómetro
   del evento (`src/data/anchor.py`). Terciles de `ProductionDay` estimados sobre dev.
4. **Filas, riesgo y features** con `src/data/landmark.py::build_landmark_panel`: una fila
   por (vehículo, hito), la entrada y la salida del riesgo dentro de `event_window`
   (`configs/data/event_clock.yaml`), y las cuatro features por mes calendario con su peso.
5. **Salida.** `panel_landmark_ps.parquet` y `panel_landmark_ps_meta.json`, con los conteos
   por hito de dev (el punto de control 3a) al lado de la factibilidad de la Fase 1.

No arma folds: los splits se congelan aparte (Fase 7, `extend_splits`). Del test solo se
registra cuántas filas y vehículos quedaron.
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

from src.config import ensure_dir, load_config, repo_root, resolve_path, set_seed  # noqa: E402
from src.data.anchor import EPOCH, estimate_origin_day, project_dates_to_odometer  # noqa: E402
from src.data.join import load_vehicle_static  # noqa: E402
from src.data.landmark import (  # noqa: E402
    FM_PREFIX,
    TRIP_REQUIRED,
    assign_tercile,
    build_landmark_panel,
    landmark_config,
    load_fleet_features,
    parse_fleet_column,
    production_tercile_edges,
)
from src.data.subset import read_table_for_vehicles  # noqa: E402
from src.eval.splits import load_test_split, test_split_masks  # noqa: E402
from src.features.trips import derive_trip_columns  # noqa: E402

logger = logging.getLogger("build_landmark_panel")

ID = "vehicle_id"
DAY = pd.Timedelta(days=1)
RAW_TRIP_COLUMNS = [
    "TripNumber", "TripDatetimeStart", "TripDatetimeEnd", "OdometerTripStart", "OdometerTripEnd",
    "FuelLvlStartPc", "FuelLvlEndPc", "EngineOilLifePCStart", "EngineTemperatureMin", "EngineTemperatureMax",
    "EngineTemperatureAvg", "AirFilterStart", "AirFilterEnd", "AirRegenerationStart", "AirRegenerationEnd",
    "CoolantTemperatureStart", "CoolantTemperatureEnd", "AirTemperatureMin", "AirTemperatureAvg",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/data/panel_landmark_ps.yaml")
    parser.add_argument("--refresh", action="store_true", help="vuelve a leer los crudos aunque haya cache")
    return parser.parse_args()


def load_trips(cfg: dict[str, Any], universe: set[str], columns: set[str], *, refresh: bool) -> pd.DataFrame:
    """Viajes derivados del universo, con las columnas que usa el panel, en orden de `TripNumber`."""
    cache = resolve_path(cfg["trips_cache"])
    if cache.exists() and not refresh:
        trips = pd.read_parquet(cache)
        logger.info("viajes del universo desde el cache %s (%d filas)", cache, len(trips))
        if not columns <= set(trips.columns):
            raise ValueError(f"El cache no trae {sorted(columns - set(trips.columns))}: correr con --refresh")
    else:
        chunksize = int((cfg.get("scan") or {}).get("chunksize", 500_000))
        # Fila completa (columns=None): el dedupe exacto la necesita.
        raw, _ = read_table_for_vehicles("trips", None, universe, sources=cfg["sources"],
                                         dedupe=cfg["dedupe"], chunksize=chunksize)
        raw = raw[[c for c in raw.columns if c in {ID, *RAW_TRIP_COLUMNS}]]
        spec = load_config(cfg["features_spec"])
        trips, _ = derive_trip_columns(raw, thresholds=spec.get("thresholds"), clip=spec.get("clip"))
        trips = trips[[c for c in trips.columns if c in columns]]
        ensure_dir(cache.parent)
        trips.to_parquet(cache, index=False)
    foreign = set(trips[ID].astype(str).unique()) - universe
    if foreign:
        raise ValueError(f"{len(foreign)} vehículo(s) fuera del universo en los viajes")
    return trips.sort_values([ID, "TripNumber"], kind="stable", ignore_index=True)


def build_vehicles(cfg: dict[str, Any], universe: set[str], dev: set[str],
                   trips: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Una fila por vehículo del universo: reloj, odómetro del evento y columnas que pasan al panel."""
    static = load_vehicle_static(config_path=cfg["sources"], dedupe_config=cfg["dedupe"],
                                 panel_config=cfg["panel_config"])
    static = static[static[ID].astype(str).isin(universe)].set_index(ID)
    missing = sorted(universe - set(static.index.astype(str)))
    if missing:
        raise ValueError(f"{len(missing)} vehículo(s) del holdout sin fila estática (ej.: {missing[:3]})")

    meta = json.loads(resolve_path(cfg["anchor_meta"]).read_text(encoding="utf-8"))
    origin = float(meta["anchor"]["origin_day_since_epoch"])
    first = trips.groupby(ID, observed=True)["TripDatetimeStart"].min()
    dev_index = static.index[static.index.astype(str).isin(dev)]
    check, anchor_stats = estimate_origin_day(first.reindex(dev_index), static.loc[dev_index, "static_ProductionDay"])
    if abs(check - origin) > 0.5:
        raise ValueError(f"El origen re-estimado sobre dev ({check}) no coincide con el congelado ({origin})")

    production = pd.to_numeric(static["static_ProductionDay"], errors="coerce")
    until_sale = pd.to_numeric(static["static_daysUntilSale"], errors="coerce")
    event_age = pd.to_numeric(static["event_day_since_production"], errors="coerce")
    v = pd.DataFrame(index=static.index)
    v["event_observed"] = static["event_observed"].astype(int)
    event_age = event_age.where(v["event_observed"].eq(1))
    prod_date = EPOCH + pd.to_timedelta(origin + production, unit="D")
    v["sale_date"] = prod_date + pd.to_timedelta(until_sale, unit="D")
    event_date = prod_date + pd.to_timedelta(event_age, unit="D")
    v["event_dss"] = event_age - until_sale
    v["last_trip"] = trips.groupby(ID, observed=True)["TripDatetimeStart"].max()
    v["event_odo_km"] = project_dates_to_odometer(event_date.where(v["event_observed"].eq(1)), trips)["event_odo_km"]

    for column in cfg["static_columns"]:
        v[f"static_{column}"] = static[f"static_{column}"]
    for column in cfg["aux_static"]:
        v[f"aux_static_{column}"] = static[f"static_{column}"]
    # Fecha de venta en el eje de `ProductionDay` (piso C2: score = −fecha de venta).
    v["aux_sale_day"] = production + until_sale
    edges = production_tercile_edges(production.loc[dev_index], int(cfg["production_terciles"]))
    v["aux_production_tercile"] = assign_tercile(production, edges)

    info = {"origin_day_since_epoch": origin, "origin_reestimated_on_dev": check, **anchor_stats,
            "production_tercile_edges": edges, "production_terciles_estimated_on": "dev_vehicles"}
    return v, info


def dev_counts(panel: pd.DataFrame, landmarks: list[float], e_min: float, sensitivity: list[float],
               immobile_km: float) -> pd.DataFrame:
    """Conteos del punto de control 3a, por hito (solo dev)."""
    rows = []
    for landmark in landmarks:
        p = panel.loc[panel["aux_landmark_day"].eq(landmark)]
        healthy = p["event_observed"].eq(0)
        exposure = p["aux_inwindow_exposure_days"]
        at_risk = p["aux_dss_exit"] > p["aux_dss_entry"]
        row = {
            "hito_dias": int(landmark), "filas": len(p), "vehiculos": p[ID].nunique(),
            "eventos_delta1": int(p["aux_event_in_window"].sum()), "label_1": int(p["label"].sum()),
            "sanos": int(healthy.sum()), "sanos_sin_exposicion": int((healthy & ~at_risk).sum()),
            "sanos_exp_1_119": int((healthy & at_risk & (exposure < 120)).sum()),
            "sanos_exp_ge_120": int((healthy & (exposure >= 120)).sum()),
            f"resueltos_e{e_min:g}": int(p["aux_resolved_negative"].sum()),
        }
        for e in sensitivity:
            row[f"resueltos_e{e:g}"] = int(p[f"aux_resolved_negative_e{e:g}"].sum())
        row["filas_en_riesgo"] = int(at_risk.sum())
        row["quietos"] = int((p["window_km"] < immobile_km).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    args = parse_args()
    cfg = load_config(args.config)
    seed = set_seed(int(cfg["seed"]))
    clock = load_config(cfg["event_clock"])
    lm_cfg = landmark_config(cfg, clock["event_window"])
    features = load_fleet_features(cfg["fleet_features"], cfg["features_spec"])

    split = load_test_split(cfg["test_split"])
    dev = {str(v) for v in split["dev_vehicles"]}
    test = {str(v) for v in split["test_vehicles"]}
    universe = dev | test
    logger.info("Universo: %d vehículos (%d dev, %d test)", len(universe), len(dev), len(test))

    columns = {*TRIP_REQUIRED, "TripNumber", *(f.column for f in features)}
    trips = load_trips(cfg, universe, columns, refresh=args.refresh)
    vehicles, anchor = build_vehicles(cfg, universe, dev, trips)
    panel, report = build_landmark_panel(vehicles, trips, lm_cfg, features)

    dev_mask, test_mask = test_split_masks(panel, split, strict=True)
    dev_panel = panel.loc[dev_mask]
    events_outside = dev_panel.loc[dev_panel["event_observed"].eq(1) & dev_panel["aux_event_in_window"].eq(0)]
    if len(events_outside):
        raise ValueError(f"{events_outside[ID].nunique()} fallado(s) de dev con fila y δ = 0: la ventana o el "
                         "reloj no reproducen la Fase 1")

    out_path = resolve_path(cfg["output"]["panel"])
    ensure_dir(out_path.parent)
    panel.to_parquet(out_path, index=False)

    immobile_km = float(clock["early_trait"]["immobile_km"])
    counts = dev_counts(dev_panel, list(lm_cfg.landmarks_days), lm_cfg.e_min_days,
                        list(lm_cfg.e_min_sensitivity_days), immobile_km)
    # Descartes de dev por hito, recalculados solo sobre dev: los del reporte cuentan el universo.
    dev_vehicles_frame = vehicles.loc[vehicles.index.astype(str).isin(dev)]
    _, dev_report = build_landmark_panel(dev_vehicles_frame, trips.loc[trips[ID].isin(dev)], lm_cfg, features)
    discards = pd.DataFrame([{"hito_dias": int(float(k)), **v} for k, v in dev_report["landmarks"].items()])

    fm = [c for c in panel.columns if c.startswith(FM_PREFIX)]
    cell_nan = {}
    for f in features:
        values = [c for c in fm if parse_fleet_column(c)[0] == f.name and not parse_fleet_column(c)[2]]
        weights = [c for c in fm if parse_fleet_column(c)[0] == f.name and parse_fleet_column(c)[2]]
        has_trips = dev_panel[weights].notna().to_numpy()
        below = has_trips & dev_panel[values].isna().to_numpy()
        cell_nan[f.name] = {"cells_with_trips": int(has_trips.sum()), "below_min_weight": int(below.sum()),
                            "rows_without_any_month": int((~(has_trips & ~below)).all(axis=1).sum())}

    expected = clock.get("expected") or {}
    versus = pd.DataFrame({
        "hito_dias": counts["hito_dias"],
        "eventos": counts["eventos_delta1"], "eventos_fase1": expected.get("feasibility_events"),
        "sanos": counts["sanos"], "sanos_fase1": expected.get("feasibility_healthy"),
        "exp_ge_120": counts["sanos_exp_ge_120"], "exp_ge_120_fase1": expected.get("feasibility_exposed_120"),
        "sin_exposicion": counts["sanos_sin_exposicion"], "sin_exposicion_fase1": expected.get("feasibility_no_exposure"),
    })

    feat_cols = [c for c in panel.columns if c.startswith("feat_")]
    meta = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": cfg.get("_config_path"),
        "event_clock": str(cfg["event_clock"]),
        "features_spec": str(cfg["features_spec"]),
        "seed": seed,
        "landmark": {k: (list(v) if isinstance(v, tuple) else v) for k, v in lm_cfg.__dict__.items()},
        "fleet_features": [f.__dict__ for f in features],
        "anchor": anchor,
        "months": report["months"],
        "rows": report["rows"],
        "duplicated_vehicle_cut_odo": report["duplicated_vehicle_cut_odo"],
        "dev": {"rows": int(dev_mask.sum()), "vehicles": int(dev_panel[ID].nunique()),
                # Los fallados de dev con y sin fila: D2 sobre todos los eventos cuenta como
                # perdidos a los que no llegan a ningún hito (preregistro §5).
                "event_vehicles": int(dev_vehicles_frame["event_observed"].eq(1).sum()),
                "event_vehicles_with_row": int(dev_panel.loc[dev_panel["event_observed"].eq(1), ID].nunique()),
                "by_landmark": counts.to_dict(orient="records"), "discards": discards.to_dict(orient="records"),
                "cells": cell_nan},
        # Del test solo cuántas filas y vehículos: es el holdout.
        "test": {"rows": int(test_mask.sum()), "vehicles": int(panel.loc[test_mask, ID].nunique())},
        "columns": {"feat": [c for c in feat_cols if not c.startswith(FM_PREFIX)],
                    "feat_fm": len(fm),
                    "static": [c for c in panel.columns if c.startswith("static_")],
                    "aux": [c for c in panel.columns if c.startswith("aux_")]},
    }
    meta_path = resolve_path(cfg["output"]["meta"])
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")

    pd.set_option("display.width", 200)
    print("\n== Panel de hitos post-venta ==")
    print(f"hitos {list(map(int, lm_cfg.landmarks_days))} d · G {lm_cfg.gap_days:g} d · H {lm_cfg.horizon_days:g} d · "
          f"≥ {lm_cfg.min_trips} viajes · ventana {lm_cfg.window_start.date()} → {lm_cfg.window_end.date()}")
    print(f"filas {len(panel)} (dev {int(dev_mask.sum())}, test {int(test_mask.sum())}) · "
          f"{len(meta['columns']['feat'])} feat_ de diseño + {len(fm)} feat_fm__ ({len(report['months'])} meses) · "
          f"{len(meta['columns']['aux'])} aux_ · (vehículo, cut_odo) repetidos: {report['duplicated_vehicle_cut_odo']}")
    print(f"terciles de producción (dev): {anchor['production_tercile_edges']}")
    print("\n-- dev · descartes por hito --")
    print(discards.to_string(index=False))
    print("\n-- dev · conteos por hito (punto de control 3a) --")
    print(counts.to_string(index=False))
    print("\n-- contra la factibilidad de la Fase 1 (la salida de los sanos ya no se acota por el último viaje) --")
    print(versus.to_string(index=False))
    print("\n-- dev · celdas vehículo × mes bajo el peso mínimo --")
    print(pd.DataFrame(cell_nan).T.to_string())
    rel = out_path.relative_to(repo_root()) if out_path.is_relative_to(repo_root()) else out_path
    print(f"\nescrito: {rel} · {meta_path.name}")
    return 0


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
