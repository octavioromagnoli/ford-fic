#!/usr/bin/env python
"""Rearma los folds de CV sobre un panel que ya existe, sin reconstruirlo.

    python scripts/make_splits.py --config configs/data/panel_v1.yaml

`build_dataset.py` deja `splits.json` como último paso del panel, pero cambiar
*cómo se reparten los folds* (estratificación, mínimo de positivos, repeticiones)
no cambia ni una fila del panel: volver a leer 13M de filas de crudos para
reescribir un JSON de listas de `vehicle_id` es tiempo perdido, y tentaba a
editar el archivo a mano.

Lee las mismas claves del mismo YAML (`splits:` vía `split_options`) y recorta a
dev con el mismo holdout congelado que usa el entrenamiento, así el resultado es
idéntico al que dejaría `build_dataset.py`. Con `--dry-run` no escribe nada:
imprime la tabla de folds y, si ya hay un `splits.json`, dice si el reparto cambia.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config, resolve_path  # noqa: E402
from src.eval.splits import (  # noqa: E402
    load_splits,
    load_test_split,
    make_splits,
    save_splits,
    split_options,
    test_split_masks,
)

logger = logging.getLogger("make_splits")


def fold_table(splits: dict) -> pd.DataFrame:
    """Tabla de folds por repetición: vehículos, eventos, positivos y tasa."""
    rows = []
    for repeat in splits["repeats"]:
        for fold in repeat["folds"]:
            rows.append(
                {
                    "repeat": repeat["repeat"],
                    "fold": fold["fold"],
                    "vehiculos": len(fold["valid_vehicles"]),
                    "veh_evento": fold["n_valid_event_vehicles"],
                    "veh_positivos": fold["n_valid_positive_vehicles"],
                    "filas": fold["n_valid_rows"],
                    "label_1": fold["n_valid_positives"],
                    "tasa": round(fold["valid_positive_rate"], 4),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rearma splits.json sobre un panel existente")
    parser.add_argument("--config", required=True, help="YAML del panel (usa `splits:` y `output:`)")
    parser.add_argument("--panel", default=None, help="Override del panel a leer")
    parser.add_argument("--out", default=None, help="Override de dónde escribir splits.json")
    parser.add_argument("--dry-run", action="store_true", help="No escribe: solo muestra los folds")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    cfg = load_config(args.config)

    panel_path = resolve_path(args.panel or cfg["output"]["panel"])
    panel = pd.read_parquet(panel_path)

    # Mismo recorte a dev que hace el entrenamiento: los folds se sortean solo sobre
    # dev, y el test no se mira ni para esto.
    if cfg.get("test_split"):
        split = load_test_split(cfg["test_split"])
        dev_mask, _ = test_split_masks(panel, split, strict=True)
        panel = panel.loc[dev_mask].reset_index(drop=True)

    options = split_options(cfg)
    logger.info(
        "Panel %s | %d filas de dev | %d vehículos | estratifica por %s (%s) | R=%d",
        panel_path.name, len(panel), panel["vehicle_id"].nunique(),
        options["stratify_column"], options["stratify_level"], options["n_repeats"],
    )
    splits = make_splits(panel, **options)

    print()
    print(fold_table(splits).to_string(index=False))
    print()

    out_path = resolve_path(args.out or cfg["output"]["splits"])
    if out_path.exists():
        previous = load_splits(out_path)
        same = [f["valid_vehicles"] for f in previous.get("folds", [])] == [
            f["valid_vehicles"] for f in splits["repeats"][0]["folds"]
        ]
        logger.info(
            "Reparto de la repetición 0 vs. %s: %s",
            out_path.name,
            "IDÉNTICO (las métricas viejas siguen siendo comparables)"
            if same
            else "DISTINTO (hay que volver a correr los experimentos para comparar)",
        )

    if args.dry_run:
        logger.info("--dry-run: no se escribió nada")
        return
    save_splits(splits, out_path)
    logger.info("Splits: %d folds × %d repetición(es) | %s",
                splits["n_splits"], splits["n_repeats"], out_path)


if __name__ == "__main__":
    main()
