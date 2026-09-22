#!/usr/bin/env python
"""Variante del panel de supervivencia con el desvío respecto de la historia del vehículo.

    python scripts/build_history_panel.py --config configs/data/panel_history_v1.yaml

Toma `panel_survival.parquet` (panel v1 + `feat_cut_odo`) y le agrega, para cada corte
**ya existente**, `feat_<x>_hist_delta = agg(historia previa) − agg(ventana)`
(`src/features/windows.py::compute_history_deviation`, spec en
`configs/data/features_history.yaml`). No arma cortes, no etiqueta, no empareja: las
filas son las del panel base, en el mismo orden, así que la huella es la misma y el
`splits_r3.json` congelado sirve tal cual.

Los crudos se leen igual que en `scripts/build_dataset.py` (mismo lector, mismo dedupe,
mismas derivadas con los umbrales del panel v1). Dos verificaciones antes de escribir:

1. El lado de la ventana de cada desvío reproduce la columna del panel v1 con el mismo
   agregador (`window_check` del spec): si no, las derivadas no son las del panel.
2. `vehicle_id`, `cut_odo` y `label` coinciden fila a fila con `panel.parquet`.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.config import ensure_dir, load_config, resolve_path  # noqa: E402
from src.data.panel import attach_columns  # noqa: E402
from src.data.subset import read_table_for_vehicles  # noqa: E402
from src.features.trips import derive_trip_columns  # noqa: E402
from src.features.windows import (  # noqa: E402
    VehicleArrays,
    compute_history_deviation,
    load_history_deviation_config,
)

from scripts.build_dataset import ID, TRIP_COLUMNS  # noqa: E402

logger = logging.getLogger("build_history_panel")

ROW_CHECK = ["vehicle_id", "cut_odo", "label"]


def check_same_rows(panel: pd.DataFrame, reference: pd.DataFrame) -> None:
    """`vehicle_id`, `cut_odo` y `label` iguales fila a fila: si no, los folds no sirven."""
    if len(panel) != len(reference):
        raise ValueError(f"{len(panel)} filas contra {len(reference)} del panel v1")
    for col in ROW_CHECK:
        a, b = panel[col].reset_index(drop=True), reference[col].reset_index(drop=True)
        same = (a == b) | (a.isna() & b.isna())
        if not same.all():
            raise ValueError(f"`{col}` difiere del panel v1 en {int((~same).sum())} filas")


def main() -> None:
    parser = argparse.ArgumentParser(description="Panel de supervivencia + feat_*_hist_delta")
    parser.add_argument("--config", default="configs/data/panel_history_v1.yaml")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    cfg = load_config(args.config)
    panel_cfg = load_config(cfg["panel_config"])
    spec_cfg = load_config(cfg["features"])
    hist_cfg = load_history_deviation_config(cfg["features"])
    base_spec = load_config(spec_cfg["base_spec"])
    window_km = float(panel_cfg["label"]["window_km"])

    panel = pd.read_parquet(resolve_path(cfg["base_panel"]))
    reference = pd.read_parquet(resolve_path(panel_cfg["output"]["panel"]))
    check_same_rows(panel, reference)

    vehicles = set(panel[ID].astype(str))
    trips, _ = read_table_for_vehicles("trips", None, vehicles, sources=panel_cfg["sources"],
                                       dedupe=panel_cfg["dedupe"],
                                       chunksize=int(cfg["scan"]["chunksize"]))
    trips = trips[[c for c in trips.columns if c in {ID, *TRIP_COLUMNS}]]
    trips, _ = derive_trip_columns(trips, thresholds=base_spec.get("thresholds"), clip=base_spec.get("clip"))

    columns = sorted({s.column for s in hist_cfg.specs})
    groups = {str(k): g for k, g in trips.groupby(ID, observed=True, sort=False)}
    frames = []
    for vid, cuts in panel.groupby(ID, observed=True, sort=False)["cut_odo"]:
        arrays = VehicleArrays.from_frame(groups[str(vid)], position_col="OdometerTripEnd", columns=columns,
                                          trip_km_col="trip_km", time_cols=("TripDatetimeStart", "TripDatetimeEnd"))
        feats = compute_history_deviation(arrays, cuts.to_numpy(), window_km=window_km, config=hist_cfg)
        feats.insert(0, "cut_odo", cuts.to_numpy())
        feats.insert(0, ID, vid)
        frames.append(feats)
    out_panel = attach_columns(panel, pd.concat(frames, ignore_index=True))
    check_same_rows(out_panel, reference)

    # El lado de la ventana es la columna del panel v1: mismas derivadas, mismo agregador.
    for name, v1_col in (spec_cfg.get("window_check") or {}).items():
        mine, theirs = out_panel[f"aux_{name}_window"], out_panel[v1_col]
        ok = np.isclose(mine, theirs, rtol=1e-9, atol=1e-9, equal_nan=True)
        if not ok.all():
            raise ValueError(f"`aux_{name}_window` no reproduce `{v1_col}` en {int((~ok).sum())} filas: "
                             "las derivadas no son las del panel v1")
        logger.info("ventana de %s == %s en las %d filas", name, v1_col, len(out_panel))

    out = resolve_path(cfg["output"]["panel"])
    ensure_dir(out.parent)
    out_panel.to_parquet(out, index=False)
    # Sin `_meta.json` propio a propósito: `train.py` toma el build del panel de ahí y
    # un derivado tiene que heredar el del panel base (`panel_meta.json`).
    new = [s.output for s in hist_cfg.specs]
    coverage = {c: round(float(out_panel[c].notna().mean()), 3) for c in new}
    logger.info("%s | %d filas | %d vehículos | +%s | cobertura %s", out.name, len(out_panel),
                out_panel[ID].nunique(), new, coverage)


if __name__ == "__main__":
    main()
