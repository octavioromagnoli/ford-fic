#!/usr/bin/env python
"""Congela el holdout dev/test y deja la auditoría del join a nivel vehículo.

    python scripts/make_test_split.py --config configs/data/test_split.yaml

Corre ANTES de F2 a propósito: el test se elige sin haber mirado una sola feature,
así ninguna decisión de diseño (qué agregar en la ventana, qué W/G/H, qué umbral)
se toma con el test a la vista. La auditoría de esquema y calidad sí se hace sobre
el dataset completo, porque no depende de qué vehículo caiga de cada lado.

Valida el join en los dos niveles que el dato admite: por id (los mismos 1081
vehículos en las tres tablas) y fila a fila (`trips` y `signals` enriquecidas cada
una por su lado con las estáticas, sin huérfanas y sin cambiar de tamaño).
`trips` y `signals` no se mergean entre sí: no comparten clave fila a fila.

Escribe (todo bajo `data/`, que no se versiona):

- `test_split.json`  · listas de `vehicle_id` de dev y test + metadata (semilla,
  test_size, fecha, huella del universo). Es el archivo que se congela.
- `vehicles.parquet` · la unión a nivel vehículo, para que F2 no vuelva a leer 1,2 GB.
- `raw_quality.csv`  · nulos y tipo por columna de las tres tablas crudas.

El detalle y los números están en `docs/memoria/f2-union-y-holdout-dev-test.md`.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.config import ensure_dir, load_config, resolve_path, set_seed  # noqa: E402
from src.data.join import (  # noqa: E402
    build_vehicle_table,
    duplicate_report,
    enrichment_report,
    join_coverage,
    load_vehicle_static,
    quality_report,
    scan_raw_tables,
)
from src.eval.splits import make_test_split, save_test_split, split_balance  # noqa: E402

logger = logging.getLogger("make_test_split")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/data/test_split.yaml")
    parser.add_argument(
        "--force",
        action="store_true",
        help="sobrescribe un holdout ya congelado (no se hace sin motivo: ver el YAML)",
    )
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg["seed"]))

    target = resolve_path(cfg["output"]["test_split"])
    if target.exists() and not args.force:
        print(f"Ya existe un holdout congelado en {target}. Se regenera con --force.")
        return 0

    scans = scan_raw_tables(
        config_path=cfg["sources"],
        dedupe_config=cfg["dedupe"],
        chunksize=int(cfg.get("scan", {}).get("chunksize", 500_000)),
    )
    static = load_vehicle_static(
        config_path=cfg["sources"], dedupe_config=cfg["dedupe"], panel_config=cfg["panel"]
    )
    vehicles = build_vehicle_table(static, scans)

    print("\n== Cobertura del join ==")
    print(join_coverage(static, scans).to_string(index=False))
    print("\n== Enriquecimiento de las tablas grandes (cada una por su lado) ==")
    print(
        enrichment_report(
            static,
            scans,
            config_path=cfg["sources"],
            dedupe_config=cfg["dedupe"],
            chunksize=int(cfg.get("scan", {}).get("chunksize", 500_000)),
        ).to_string(index=False)
    )
    print("\n== Duplicados ==")
    print(duplicate_report(scans).to_string(index=False))
    print("\n== Columnas con nulos ==")
    quality = quality_report(scans)
    print(quality[quality["n_null"] > 0].to_string(index=False))

    split = make_test_split(
        vehicles,
        test_size=float(cfg["test_size"]),
        seed=int(cfg["seed"]),
        group_column=cfg.get("group_column", "vehicle_id"),
        event_column=cfg.get("event_column", "event_observed"),
    )
    print("\n== Holdout dev/test ==")
    print(
        pd.DataFrame([split["dev"], split["test"]], index=["dev", "test"]).to_string(
            float_format=lambda v: f"{v:.4f}"
        )
    )
    print("\n== Balance de las estáticas (reporte, no estratificación) ==")
    balance = split_balance(vehicles, split, list(cfg.get("balance_columns", [])))
    print(balance.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    ensure_dir(target.parent)
    save_test_split(split, target)
    vehicles.to_parquet(resolve_path(cfg["output"]["vehicle_table"]), index=False)
    quality.to_csv(resolve_path(cfg["output"]["quality"]), index=False)
    print(f"\nHoldout congelado en {target} ({split['test']['n_vehicles']} vehículos de test).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
