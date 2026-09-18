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

## Dos etapas, en este orden

1. **Sortear** los 1081 vehículos en dev/test (`make_test_split`, semilla 42). Es el
   sorteo de siempre, bit a bit: la huella de su reparto queda en `parent`.
2. **Recortar** ese resultado al universo del estudio (`select_universe` +
   `restrict_test_split`), sacando los positivos cuyo evento no se puede ubicar en
   el tiempo y los mercados donde ningún evento es observable. Cada vehículo que
   sobrevive conserva el lado que le tocó en (1).

El orden no es intercambiable: sortear el universo ya recortado sería un dado
nuevo, tirado después de haber mirado los datos que motivaron el recorte. Los
criterios salen de `universe` en el YAML; el porqué, de `src/data/usable.py`.

Escribe (todo bajo `data/`, que no se versiona):

- `test_split.json`  · listas de `vehicle_id` de dev y test, los excluidos, el
  informe del universo y la metadata del sorteo padre. Es el archivo que se congela.
- `vehicles.parquet` · la unión a nivel vehículo. Van **los 1081**, sin recortar: es
  el artefacto de auditoría, y el universo se aplica leyendo el holdout.
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
from src.data.usable import market_usability, select_universe  # noqa: E402
from src.eval.splits import (  # noqa: E402
    make_test_split,
    restrict_test_split,
    save_test_split,
    split_balance,
)

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
    print("\n== Etapa 1 · sorteo sobre el universo completo ==")
    print(
        pd.DataFrame([split["dev"], split["test"]], index=["dev", "test"]).to_string(
            float_format=lambda v: f"{v:.4f}"
        )
    )

    print("\n== Fecha del evento utilizable, por mercado ==")
    print(market_usability(vehicles).to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    universe_cfg = cfg.get("universe") or {}
    keep_mask, universe = select_universe(
        vehicles,
        require_usable_event_date=bool(universe_cfg.get("require_usable_event_date", False)),
        keep_markets=universe_cfg.get("keep_markets"),
    )
    split = restrict_test_split(
        split,
        vehicles.loc[keep_mask, "vehicle_id"].astype(str),
        events=vehicles.set_index("vehicle_id")["event_observed"],
        universe=universe,
    )

    print(
        f"\n== Etapa 2 · universo del estudio: {universe['n_input']} -> "
        f"{universe['n_kept']} vehículos · {universe['events_input']} -> "
        f"{universe['events_kept']} eventos =="
    )
    print(f"  -{universe['n_dropped_no_usable_date']:>4d} sin fecha de evento utilizable")
    print(f"  -{universe['n_dropped_market']:>4d} de mercados sin eventos observables")
    print(
        pd.DataFrame([split["dev"], split["test"]], index=["dev", "test"]).to_string(
            float_format=lambda v: f"{v:.4f}"
        )
    )

    print("\n== Balance de las estáticas (reporte, no estratificación) ==")
    balance = split_balance(
        vehicles.loc[keep_mask], split, list(cfg.get("balance_columns", []))
    )
    print(balance.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    ensure_dir(target.parent)
    save_test_split(split, target)
    vehicles.to_parquet(resolve_path(cfg["output"]["vehicle_table"]), index=False)
    quality.to_csv(resolve_path(cfg["output"]["quality"]), index=False)
    print(
        f"\nHoldout congelado en {target}: {split['dev']['n_vehicles']} de dev, "
        f"{split['test']['n_vehicles']} de test, {split['n_excluded']} fuera del universo."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
