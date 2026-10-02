#!/usr/bin/env python
"""Materializa el panel real (F2, Track A).

    python scripts/build_dataset.py --config configs/data/panel_v1.yaml

Hace, en este orden y sin excepciones:

1. **Universo.** Lee el holdout congelado (`test_split.json`): el panel se construye
   con los 364 vehículos de `dev_vehicles` + `test_vehicles`, y con nadie más. Los
   717 excluidos no entran (`src/data/usable.py`).
2. **Crudos.** `trips` y `signals` para esos vehículos, canonizados (13 clones),
   filtrados y sin filas exactamente repetidas (`src/data/subset.py`). Derivadas a
   nivel viaje/señal con los umbrales del YAML de features (`src/features/trips.py`,
   `src/features/signals.py`).
3. **Evento sobre el eje de km.** Origen del calendario estimado **sobre dev** y
   congelado en la metadata; fecha del evento = origen + ProductionDay +
   IdentificationDate, interpolada sobre los viajes (`src/data/anchor.py`).
4. **Cortes, etiqueta y features.** Grilla de Δ km, ventana W hacia atrás, gap G y
   horizonte H (`src/data/panel.py` + `src/features/windows.py`). Lo posterior al
   evento no entra en ninguna ventana.
5. **Sanos emparejados** con la distribución de (odómetro, mes) de los positivos de
   dev, para que ni el kilometraje ni el calendario sean un atajo.
6. **Salida.** `panel.parquet` con el esquema del contrato (CLAUDE.md), `splits.json`
   con los 5 folds de CV **sobre dev**, y `panel_meta.json` con todo lo necesario
   para reproducir y auditar (terna, origen, conteos, descartes, features).

Lo que NO hace: mirar el test. Las filas de test se escriben porque el contrato lo
pide; `scripts/train.py` las recorta con `select_dev()` antes de armar los folds.
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
from src.data.anchor import estimate_origin_day, event_dates, project_dates_to_odometer  # noqa: E402
from src.data.join import load_vehicle_static  # noqa: E402
from src.data.panel import LabelConfig, build_panel, match_healthy_cuts  # noqa: E402
from src.data.subset import read_table_for_vehicles  # noqa: E402
from src.eval.splits import (  # noqa: E402
    load_test_split,
    make_splits,
    save_splits,
    split_options,
    test_split_masks,
)
from src.features.signals import derive_signal_columns  # noqa: E402
from src.features.trips import derive_trip_columns  # noqa: E402
from src.features.windows import load_feature_specs  # noqa: E402

logger = logging.getLogger("build_dataset")

ID = "vehicle_id"
TRIP_COLUMNS = [
    "VehicleCode", "TripDatetimeStart", "TripDatetimeEnd", "OdometerTripStart", "OdometerTripEnd",
    "FuelLvlStartPc", "FuelLvlEndPc", "EngineOilLifePCStart", "EngineTemperatureMin", "EngineTemperatureMax",
    "EngineTemperatureAvg", "AirFilterStart", "AirFilterEnd", "AirRegenerationStart", "AirRegenerationEnd",
    "CoolantTemperatureStart", "CoolantTemperatureEnd", "AirTemperatureMin", "AirTemperatureAvg",
]
SIGNAL_COLUMNS = ["VehicleCode", "OdometerValue", "Message", "Regenerations"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/data/panel_v1.yaml")
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    args = parse_args()
    cfg = load_config(args.config)
    seed = set_seed(int(cfg["seed"]))
    # Entrega v2: la fecha registrada cae ~2 semanas DESPUÉS de la intervención (el cambio de
    # aceite y el fin de la sobrecarga del filtro quedan antes; docs/reproducibilidad.md).
    # La referencia del evento se corre hacia atrás para que el gap G blanquee el taller y
    # los síntomas, no los días siguientes. 0 = la fecha tal cual (entrega 1).
    reference_offset_days = float(cfg["label"].get("reference_offset_days", 0) or 0)
    label_cfg = LabelConfig(**{k: v for k, v in cfg["label"].items() if k != "reference_offset_days"})
    chunksize = int((cfg.get("scan") or {}).get("chunksize", 500_000))

    # 1 · universo -----------------------------------------------------------------
    split = load_test_split(cfg["test_split"])
    dev = {str(v) for v in split["dev_vehicles"]}
    test = {str(v) for v in split["test_vehicles"]}
    universe = dev | test
    if "excluded_vehicles" not in split:
        logger.warning("El holdout no trae `excluded_vehicles`: ¿es un holdout sin recorte de universo?")
    logger.info("Universo: %d vehículos (%d dev, %d test); %d excluidos del estudio",
                len(universe), len(dev), len(test), len(split.get("excluded_vehicles", [])))

    static = load_vehicle_static(config_path=cfg["sources"], dedupe_config=cfg["dedupe"], panel_config=args.config)
    static = static[static[ID].astype(str).isin(universe)].set_index(ID)
    missing = sorted(universe - set(static.index.astype(str)))
    if missing:
        raise ValueError(f"{len(missing)} vehículo(s) del holdout no están en la tabla estática (ej.: {missing[:3]})")

    # 2 · crudos + derivadas --------------------------------------------------------
    spec_path = cfg["features"]["spec"]
    spec_cfg = load_config(spec_path)
    specs = load_feature_specs(spec_path, enabled_families=cfg["features"].get("enabled_families"))
    # Se lee la fila COMPLETA (columns=None) porque el dedupe exacto lo necesita; las
    # columnas que el panel no usa se sueltan recién después.
    trips, trip_counters = read_table_for_vehicles("trips", None, universe, sources=cfg["sources"],
                                                   dedupe=cfg["dedupe"], chunksize=chunksize)
    trips = trips[[c for c in trips.columns if c in {ID, *TRIP_COLUMNS}]]
    trips, trip_derive = derive_trip_columns(trips, thresholds=spec_cfg.get("thresholds"), clip=spec_cfg.get("clip"))
    signals, signal_counters = read_table_for_vehicles("signals", None, universe, sources=cfg["sources"],
                                                       dedupe=cfg["dedupe"], chunksize=chunksize)
    signals = signals[[c for c in signals.columns if c in {ID, *SIGNAL_COLUMNS}]]
    signals, signal_derive = derive_signal_columns(signals)

    # 3 · evento sobre el eje de km -------------------------------------------------
    first_trip = trips.groupby(ID, observed=True)["TripDatetimeStart"].min()
    dev_index = static.index[static.index.astype(str).isin(dev)]
    origin_day, anchor_stats = estimate_origin_day(first_trip.reindex(dev_index), static.loc[dev_index, "static_ProductionDay"])
    is_event = static["event_observed"].eq(1)
    dates = event_dates(static.loc[is_event], origin_day) - pd.Timedelta(days=reference_offset_days)
    events = project_dates_to_odometer(dates, trips)
    vehicles = static.join(events, how="left")
    vehicles["event_odo_km"] = vehicles["event_odo_km"].where(is_event)
    source_counts = events["event_odo_source"].value_counts().to_dict()
    logger.info("Eventos ubicados: %d · odómetro mediano %.0f km · fuentes %s · incertidumbre mediana %.0f km",
                len(events), events["event_odo_km"].median(), source_counts, events["event_odo_uncertainty_km"].median())

    # 4 · cortes, etiqueta y features ---------------------------------------------
    static_columns = list(cfg["features"].get("static_columns") or [])
    static_excluded = list(cfg["features"].get("static_excluded") or [])
    panel, report = build_panel(trips, signals, vehicles, specs, label_cfg,
                                static_columns=static_columns, static_excluded=static_excluded)

    # 5 · sanos emparejados con los positivos de dev --------------------------------
    sampling = cfg.get("sampling") or {}
    sampling_summary: dict[str, Any] | None = None
    if sampling.get("match_healthy_cut_distribution", False):
        keep, sampling_summary = match_healthy_cuts(
            panel, dev, match_on=tuple(sampling.get("match_on", ["cut_odo", "cut_date"])),
            odo_bin_km=float(sampling.get("odo_bin_km", 2000)), date_freq=str(sampling.get("date_freq", "M")),
            max_shortfall=float(sampling.get("max_shortfall", 0.1)), seed=seed,
        )
        panel = panel.loc[keep].reset_index(drop=True)

    # 6 · holdout, splits de CV sobre dev, salida ----------------------------------
    dev_mask, test_mask = test_split_masks(panel, split, strict=True)
    dev_panel = panel.loc[dev_mask].reset_index(drop=True)
    splits = make_splits(dev_panel, **split_options(cfg))

    panel_path = resolve_path(cfg["output"]["panel"])
    ensure_dir(panel_path.parent)
    panel.to_parquet(panel_path, index=False)
    splits_path = save_splits(splits, cfg["output"]["splits"])

    feat_cols = [c for c in panel.columns if c.startswith("feat_")]
    aux_cols = [c for c in panel.columns if c.startswith("aux_")]
    static_cols = [c for c in panel.columns if c.startswith("static_")]
    meta = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": cfg.get("_config_path"),
        "features_spec": str(spec_path),
        "seed": seed,
        "label": {**label_cfg.__dict__, "reference_offset_days": reference_offset_days},
        "anchor": {"origin_day_since_epoch": origin_day, "estimated_on": "dev_vehicles", **anchor_stats},
        "events": {
            "n": int(len(events)),
            "odo_km_median": float(events["event_odo_km"].median()),
            "odo_km_p25": float(events["event_odo_km"].quantile(0.25)),
            "odo_km_p75": float(events["event_odo_km"].quantile(0.75)),
            "uncertainty_km_median": float(events["event_odo_uncertainty_km"].median()),
            "sources": {str(k): int(v) for k, v in source_counts.items()},
        },
        "raw": {"trips": {**trip_counters, **trip_derive}, "signals": {**signal_counters, **signal_derive}},
        "panel": report,
        "sampling": sampling_summary,
        "dev": _side_summary(panel.loc[dev_mask]),
        # Del test solo se registra cuántos vehículos quedaron con filas: ni positivos ni
        # tasa. Es el holdout; hasta la corrida final nadie lo mira, ni en agregado.
        "test": {"rows": int(test_mask.sum()), "vehicles": int(panel.loc[test_mask, ID].nunique())},
        "columns": {"feat": feat_cols, "static": static_cols, "aux": aux_cols},
        "thresholds": spec_cfg.get("thresholds"),
        "clip": spec_cfg.get("clip"),
        "splits": {
            "n_splits": splits["n_splits"],
            "seed": splits["seed"],
            "stratify": splits["stratify"],
            "min_valid_positives": splits["min_valid_positives"],
            "n_repeats": splits["n_repeats"],
            "panel": splits["panel"],
        },
    }
    meta_path = resolve_path(cfg["output"]["meta"])
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")

    print("\n== Panel real ==")
    print(f"terna          : W={label_cfg.window_km:.0f} G={label_cfg.gap_km:.0f} H={label_cfg.horizon_km:.0f} "
          f"Δ={label_cfg.cut_step_km:.0f} km · censurados: {label_cfg.censored_policy}")
    print(f"origen         : día {origin_day:.0f} desde epoch (IQR {anchor_stats['iqr_days']:.2f} d) · "
          f"evento mediano {meta['events']['odo_km_median']:.0f} km · fuentes {source_counts}")
    print(f"filas          : {len(panel)} (dev + test) · features {len(feat_cols)} feat_ + "
          f"{len(static_cols)} static_ + {len(aux_cols)} aux_")
    s = _side_summary(panel.loc[dev_mask])
    print(f"{'dev':<15}: {s['rows']} filas · {s['vehicles']} vehículos ({s['event_vehicles']} con evento, "
          f"{s['event_vehicles_with_positive']} con positivo) · {s['rows_positive']} positivas · tasa {s['positive_rate']:.4f}")
    print(f"{'test':<15}: {int(test_mask.sum())} filas · {panel.loc[test_mask, ID].nunique()} vehículos (holdout: no se mira nada más)")
    if sampling_summary:
        print(f"sanos          : {sampling_summary['healthy_rows_before']} -> {sampling_summary['healthy_rows_after']} filas "
              f"({sampling_summary['healthy_vehicles_before']} -> {sampling_summary['healthy_vehicles_after']} vehículos) "
              f"emparejando sobre {sampling_summary['match_on']}")
    print(f"descartados    : { {k: len(v) for k, v in report['dropped_vehicles'].items()} } · filas por QC {report['rows_dropped_qc']}")
    print(f"escrito        : {panel_path.relative_to(repo_root()) if panel_path.is_relative_to(repo_root()) else panel_path}"
          f" · {splits_path.name} ({splits['n_splits']} folds sobre dev) · {meta_path.name}")
    return 0


def _side_summary(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {"rows": 0, "vehicles": 0, "event_vehicles": 0, "event_vehicles_with_positive": 0,
                "rows_positive": 0, "positive_rate": float("nan")}
    return {
        "rows": int(len(frame)),
        "vehicles": int(frame[ID].nunique()),
        "event_vehicles": int(frame.groupby(ID)["event_observed"].max().sum()),
        "event_vehicles_with_positive": int(frame.loc[frame["label"].eq(1), ID].nunique()),
        "rows_positive": int(frame["label"].sum()),
        "positive_rate": float(frame["label"].mean()),
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
