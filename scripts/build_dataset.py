#!/usr/bin/env python
"""Materializa el panel real. **Lo implementa F2 (Track A), después del lock de F1.**

    python scripts/build_dataset.py --config configs/data/panel_v1.yaml

Lo que tiene que hacer, para que quien lo escriba no tenga que releer el plan:

1. Cargar las tres tablas con `src.data.loader.load_raw_tables` y unificarlas en
   un `vehicle_id` único.
2. Recorrer la vida de cada vehículo cada Δ km. Para cada corte, agregar las
   señales de la ventana de W km *hacia atrás* (nunca hacia adelante).
3. Etiquetar: `label = 1` si el evento cae en `[corte+G, corte+G+H]`. El gap G es
   lo que convierte el problema en anticipación en vez de detección.
4. Muestrear los cortes de los vehículos sanos con la misma distribución de
   odómetro que los de los vehículos con evento.
5. Emitir exactamente el esquema del contrato (ver CLAUDE.md) y registrar el
   panel y los splits como wandb Artifacts.
6. Incluir a **los 1081 vehículos**, test incluido. El panel no se recorta: el
   holdout congelado (`data/processed/test_split.json`) se aplica al usarlo, con
   `src.eval.splits.test_split_masks(panel, split)`. Los folds de CV se arman
   sobre `dev_mask`, y las filas de test quedan escritas sin que nadie las mire
   hasta la corrida final.

Hasta entonces, B y C trabajan contra `scripts/make_dummy.py`, que ya emite ese
mismo esquema.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Construye el panel real (F2)")
    parser.add_argument("--config", default="configs/data/panel_v1.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)

    raise NotImplementedError(
        "F2 todavía no está implementada. El contrato de datos se congela al final de F1 "
        f"y recién ahí se completan W/G/H/Δ en {cfg['_config_path']}. "
        "Mientras tanto: python scripts/make_dummy.py --config configs/data/dummy_v1.yaml"
    )


if __name__ == "__main__":
    main()
