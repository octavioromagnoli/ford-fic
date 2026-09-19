#!/usr/bin/env python
"""Variante del panel v1 con `feat_tfm_*`: ¿el pronóstico de TimesFM le suma algo a un GBM?

    python scripts/build_timesfm_panel.py --config configs/data/panel_timesfm_v1.yaml

Toma el panel v1 y los pronósticos que dejó `scripts/eval_timesfm.py` (una corrida
por canal, `forecasts.parquet`, calculados para todas las filas del panel) y escribe
el panel con **las mismas filas** más tres columnas por canal:

* `feat_tfm_<canal>_delta`: lo que el pronóstico dice que cambia respecto de la
  ventana reciente (`fc_q50 − ctx_mean_w`);
* `feat_tfm_<canal>_spread`: cuánta duda tiene (`fc_q90 − fc_q10`);
* `feat_tfm_<canal>_q90_max`: el peor bin del cuantil alto en `[G, G+H]`.

El nivel mirado hacia atrás no se agrega: ya está en el panel v1 (`dpf_end_mean`,
`regenerations_per_1000km`, `msg_*_per_1000km`). La comparación honesta es el mismo
modelo sobre el panel v1 (`configs/exp_lgbm_panel_v1.yaml`) contra este panel
(`configs/exp_lgbm_timesfm.yaml`): mismas filas, mismos folds.

Todo pronóstico se calculó con contexto estrictamente anterior al corte y sin
ajustar nada, así que estas columnas no necesitan fitearse por fold (regla 3).
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

logger = logging.getLogger("build_timesfm_panel")

REQUIRED = ["ctx_mean_w", "fc_q10", "fc_q50", "fc_q90", "fc_q90_max"]


def channel_features(forecasts: pd.DataFrame, short: str) -> pd.DataFrame:
    out = forecasts[PANEL_KEY].copy()
    out[f"feat_tfm_{short}_delta"] = forecasts["fc_q50"] - forecasts["ctx_mean_w"]
    out[f"feat_tfm_{short}_spread"] = forecasts["fc_q90"] - forecasts["fc_q10"]
    out[f"feat_tfm_{short}_q90_max"] = forecasts["fc_q90_max"]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Panel v1 + feat_tfm_*")
    parser.add_argument("--config", default="configs/data/panel_timesfm_v1.yaml")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    cfg = load_config(args.config)
    panel_cfg = load_config(cfg["panel_config"])
    panel = pd.read_parquet(resolve_path(panel_cfg["output"]["panel"]))

    for short, path in cfg["forecasts"].items():
        forecasts = pd.read_parquet(resolve_path(path))
        missing = [c for c in REQUIRED if c not in forecasts]
        if missing:
            raise KeyError(f"{path} no tiene {missing}: correr de nuevo scripts/eval_timesfm.py")
        # Los pronósticos tienen que ser de ESTE panel: misma clave, fila a fila.
        keys = forecasts[PANEL_KEY].merge(panel[PANEL_KEY], on=PANEL_KEY, how="outer", indicator=True)
        if (keys["_merge"] != "both").any():
            raise ValueError(f"{path}: los cortes no coinciden con el panel ({keys['_merge'].value_counts().to_dict()})")
        panel = attach_columns(panel, channel_features(forecasts, short))

    out = resolve_path(cfg["output"]["panel"])
    ensure_dir(out.parent)
    panel.to_parquet(out, index=False)
    n_tfm = sum(c.startswith("feat_tfm_") for c in panel)
    logger.info("%s | %d filas | %d vehículos | +%d columnas feat_tfm_*", out.name, len(panel),
                panel["vehicle_id"].nunique(), n_tfm)


if __name__ == "__main__":
    main()
