#!/usr/bin/env python
"""Variante del panel v1 con el odómetro del corte como covariable (`feat_cut_odo`).

    python scripts/build_survival_panel.py --config configs/data/panel_survival.yaml

`cut_odo` vive en el panel desde F2, pero como identificador de la fila: `cv.py`
selecciona features por prefijo, así que ningún modelo lo ve. Un modelo de
supervivencia lo necesita —el hazard base crece con los km— y el emparejado de sanos
por (bin de odómetro × mes) es lo que hace que no sea un atajo: positivos y sanos
tienen la misma distribución de odómetro por construcción.

Copia, no calcula: mismas filas, mismo orden, misma huella. El `splits.json` congelado
sirve tal cual y la comparación contra `configs/exp_lgbm_panel_v1.yaml` es la de
siempre (mismas filas, mismos folds, un modelo distinto).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ensure_dir, load_config, resolve_path  # noqa: E402
from src.data.panel import PANEL_KEY, attach_columns  # noqa: E402

logger = logging.getLogger("build_survival_panel")

FOLLOWUP_COLUMN = "aux_km_observed_after_cut"


def main() -> None:
    parser = argparse.ArgumentParser(description="Panel v1 + feat_cut_odo")
    parser.add_argument("--config", default="configs/data/panel_survival.yaml")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    cfg = load_config(args.config)
    panel_cfg = load_config(cfg["panel_config"])
    panel = pd.read_parquet(resolve_path(panel_cfg["output"]["panel"]))

    if FOLLOWUP_COLUMN not in panel.columns:
        raise KeyError(
            f"El panel no trae `{FOLLOWUP_COLUMN}`: es anterior a la columna de seguimiento y "
            "el objetivo de supervivencia no puede saber hasta dónde se observó cada censurado. "
            f"Reconstruilo: `python scripts/build_dataset.py --config {cfg['panel_config']}`."
        )

    extra = panel[PANEL_KEY].copy()
    for new, source in cfg["columns"].items():
        if source not in panel.columns:
            raise KeyError(f"El panel no tiene la columna `{source}` que pide el config")
        extra[new] = panel[source]
    out_panel = attach_columns(panel, extra)

    out = resolve_path(cfg["output"]["panel"])
    ensure_dir(out.parent)
    out_panel.to_parquet(out, index=False)
    logger.info(
        "%s | %d filas | %d vehículos | +%s",
        out.name, len(out_panel), out_panel["vehicle_id"].nunique(), list(cfg["columns"]),
    )


if __name__ == "__main__":
    main()
