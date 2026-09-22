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

Con `splits.extend_from` en el YAML no sortea de cero: **extiende** ese archivo
(`src/eval/splits.py::extend_splits`). Los vehículos que ya tenían fold lo conservan en
cada repetición y solo se reparten los nuevos; `splits.guard_by` agrega la guarda de
positivos por grupo (por hito, en el panel del cure model):

    python scripts/make_splits.py --config configs/data/panel_landmark_ps.yaml

Un archivo que ya existe **no se pisa** si el reparto cambia (hace falta `--force`) y no
se reescribe si es idéntico: los folds congelados son la marca de tiempo de lo que se
comparó.
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
    extend_splits,
    load_splits,
    load_test_split,
    make_splits,
    save_splits,
    split_options,
    test_split_masks,
)

logger = logging.getLogger("make_splits")


def fold_table(splits: dict) -> pd.DataFrame:
    """Tabla de folds por repetición: vehículos, eventos, positivos y tasa.

    Si los folds se extendieron, suma cuántos vehículos nuevos tiene cada uno y los
    positivos por grupo de la guarda (`pos_<valor>`).
    """
    rows = []
    for repeat in splits["repeats"]:
        for fold in repeat["folds"]:
            row = {
                "repeat": repeat["repeat"],
                "fold": fold["fold"],
                "vehiculos": len(fold["valid_vehicles"]),
                "veh_evento": fold["n_valid_event_vehicles"],
                "veh_positivos": fold["n_valid_positive_vehicles"],
                "filas": fold["n_valid_rows"],
                "label_1": fold["n_valid_positives"],
                "tasa": round(fold["valid_positive_rate"], 4),
            }
            if "n_valid_new_vehicles" in fold:
                row["nuevos"] = fold["n_valid_new_vehicles"]
            for group, count in (fold.get("n_valid_positive_vehicles_by") or {}).items():
                row[f"pos_{group}"] = count
            rows.append(row)
    return pd.DataFrame(rows)


def same_folds(a: dict, b: dict) -> bool:
    """¿Los dos archivos reparten igual los vehículos, en todas las repeticiones?"""
    def folds(splits: dict) -> list[list[list[str]]]:
        repeats = splits.get("repeats") or [{"folds": splits.get("folds", [])}]
        return [[sorted(f["valid_vehicles"]) for f in r["folds"]] for r in repeats]
    return folds(a) == folds(b)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rearma splits.json sobre un panel existente")
    parser.add_argument("--config", required=True, help="YAML del panel (usa `splits:` y `output:`)")
    parser.add_argument("--panel", default=None, help="Override del panel a leer")
    parser.add_argument("--out", default=None, help="Override de dónde escribir splits.json")
    parser.add_argument("--dry-run", action="store_true", help="No escribe: solo muestra los folds")
    parser.add_argument("--force", action="store_true", help="Pisa un splits.json existente con otro reparto")
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
    block = cfg.get("splits") or {}
    logger.info(
        "Panel %s | %d filas de dev | %d vehículos | estratifica por %s (%s) | R=%d",
        panel_path.name, len(panel), panel["vehicle_id"].nunique(),
        options["stratify_column"], options["stratify_level"], options["n_repeats"],
    )
    if block.get("extend_from"):
        base_path = block["extend_from"]
        splits = extend_splits(load_splits(base_path), panel, base_path=base_path,
                               guard_by=block.get("guard_by"), **options)
        origin = splits["extended_from"]
        logger.info(
            "Extiende %s: %d vehículos conservan su fold, %d nuevos (%d positivos) se sortean, "
            "%d del split base no están en este panel",
            Path(str(base_path)).name, origin["n_kept"], origin["n_new"], origin["n_new_positive"],
            origin["n_dropped"],
        )
    else:
        splits = make_splits(panel, **options)

    print()
    print(fold_table(splits).to_string(index=False))
    print()

    out_path = resolve_path(args.out or cfg["output"]["splits"])
    same = None
    if out_path.exists():
        same = same_folds(load_splits(out_path), splits)
        logger.info(
            "Reparto vs. %s: %s",
            out_path.name,
            "IDÉNTICO (las métricas viejas siguen siendo comparables)"
            if same
            else "DISTINTO (hay que volver a correr los experimentos para comparar)",
        )

    if args.dry_run:
        logger.info("--dry-run: no se escribió nada")
        return
    if same:
        logger.info("%s ya tiene este reparto: no se reescribe (queda la fecha del congelado)", out_path.name)
        return
    if same is False and not args.force:
        raise SystemExit(
            f"{out_path} ya existe con otro reparto. Los folds congelados no se pisan en silencio: "
            "--force si es a propósito (y las corridas que lo usaron dejan de ser comparables)."
        )
    save_splits(splits, out_path)
    logger.info("Splits: %d folds × %d repetición(es) | %s",
                splits["n_splits"], splits["n_repeats"], out_path)


if __name__ == "__main__":
    main()
