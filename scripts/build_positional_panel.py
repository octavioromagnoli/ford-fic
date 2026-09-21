#!/usr/bin/env python
"""Panel del piso posicional: mismas filas del panel v1, una sola feature (`feat_cut_odo`).

    python scripts/build_positional_panel.py --config configs/data/panel_positional.yaml

**Por qué existe.** El eje "cuándo" —`(a')`, ROC intra-vehículo, PR-AUC entre fallados—
se venía leyendo contra cero, y cero no es el piso. Dentro de un vehículo fallado la
etiqueta es una función determinista de la posición del corte: los positivos son
exactamente los últimos ~6 (la fracción positiva va 1,00 / 1,00 / 0,98 / 0,93 / 0,92 /
0,87 y después 0,00 de golpe), porque el historial del fallado termina en el evento. Un
score que sea literalmente el odómetro ordena eso perfecto sin saber nada de degradación.

Este panel arma el modelo que **solo** sabe eso. Es al eje "cuándo" lo que
`cohort_ceiling()` es al eje "qué auto": la referencia contra la cual una afirmación de
anticipación significa algo. No es un candidato y no gasta presupuesto de comparaciones.

Copia, no calcula: mismas filas, mismo orden, misma huella, así que el `splits.json`
congelado sirve tal cual y la comparación contra `configs/exp_lgbm_panel_v1.yaml` es la
de siempre (mismas filas, mismos folds, un modelo distinto).

A diferencia de `build_survival_panel.py`, que **agrega** `feat_cut_odo` a las 53 que ya
están, este **descarta** todo lo demás: si quedara una sola feature real, el piso dejaría
de ser un piso.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ensure_dir, load_config, resolve_path  # noqa: E402
from src.training.cv import FEATURE_PREFIXES  # noqa: E402

logger = logging.getLogger("build_positional_panel")


def main() -> None:
    parser = argparse.ArgumentParser(description="Panel v1 con feat_cut_odo como única feature")
    parser.add_argument("--config", default="configs/data/panel_positional.yaml")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    cfg = load_config(args.config)
    panel_cfg = load_config(cfg["panel_config"])
    panel = pd.read_parquet(resolve_path(panel_cfg["output"]["panel"]))

    dropped = [c for c in panel.columns if c.startswith(FEATURE_PREFIXES)]
    out_panel = panel.drop(columns=dropped).copy()

    for new, source in cfg["columns"].items():
        if source not in panel.columns:
            raise KeyError(f"El panel no tiene la columna `{source}` que pide el config")
        out_panel[new] = panel[source]

    kept = [c for c in out_panel.columns if c.startswith(FEATURE_PREFIXES)]
    if sorted(kept) != sorted(cfg["columns"]):
        raise RuntimeError(
            f"El panel posicional quedó con features inesperadas: {sorted(kept)}. "
            f"Se esperaban exactamente {sorted(cfg['columns'])}."
        )
    if len(out_panel) != len(panel):
        raise RuntimeError("El panel posicional cambió de filas: los folds congelados dejan de servir")

    out = resolve_path(cfg["output"]["panel"])
    ensure_dir(out.parent)
    out_panel.to_parquet(out, index=False)
    logger.info(
        "%s | %d filas | %d vehículos | features: %s (descartadas %d)",
        out.name, len(out_panel), out_panel["vehicle_id"].nunique(), kept, len(dropped),
    )


if __name__ == "__main__":
    main()
