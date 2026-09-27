#!/usr/bin/env python
"""AUC por vehículo agrupado y dentro de estratos, sobre todos los autos y sobre subconjuntos.

    python scripts/audit_vehicle_strata.py --config configs/audit_vehicle_strata_v2.yaml

Para cada corrida (su `predictions.parquet`, fuera de fold) el score de un auto es el promedio de
sus cortes (`mean`, el del repo), por repetición. De ahí sale el AUC por vehículo:

- **agrupado**: todos contra todos;
- **dentro de un estrato** (p. ej. mercado, o mercado × motor): el AUC de Mann-Whitney sumado
  sobre los estratos que tienen las dos clases (van Elteren). Es lo que un modelo ordena **entre
  autos comparables**, sin el crédito de saber qué celda falla más.

Se mide sobre todos los autos de la corrida y, además, sobre cada subconjunto declarado (p. ej.
los autos del universo con la ventana de producción). Esa segunda cuenta contesta si agregar autos
de otra población ayuda **a los autos comparables**, o solo agrega separación entre poblaciones.
Al lado va un piso sin modelo: `−ProductionDay` sola como score.

Solo lee dev (las predicciones fuera de fold de `train.py` ya son solo dev). Deja una tabla en
`output`. Lectura: `docs/memoria/f9-remedicion-v2.md`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.config import ensure_dir, load_config, resolve_path  # noqa: E402

ID = "vehicle_id"
pd.set_option("display.width", 250)


def stratified_auc(frame: pd.DataFrame, score: str, strata: list[str]) -> float:
    """AUC de Mann-Whitney sumado sobre los estratos con las dos clases (agrupado si no hay estratos)."""
    groups = [frame] if not strata else [g for _, g in frame.groupby(strata)]
    num = den = 0.0
    for g in groups:
        pos, neg = g.loc[g["y"] == 1, score], g.loc[g["y"] == 0, score]
        if len(pos) and len(neg):
            num += mannwhitneyu(pos, neg).statistic
            den += len(pos) * len(neg)
    return num / den if den else float("nan")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/audit_vehicle_strata_v2.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)

    static = pd.read_parquet(resolve_path(cfg["vehicle_table"])).set_index(ID)
    subsets: dict[str, set[str] | None] = {}
    for name, path in (cfg.get("subsets") or {}).items():
        subsets[name] = None if path is None else set(json.loads(resolve_path(path).read_text(encoding="utf-8"))["dev_vehicles"])
    strata = {name: list(cols or []) for name, cols in cfg["strata"].items()}

    rows = []
    seen: dict[frozenset, str] = {}
    for run in cfg["runs"]:
        pred = pd.read_parquet(resolve_path(cfg["experiments_dir"]) / run / "predictions.parquet")
        reps = [c for c in pred.columns if c.startswith("score_r")] or ["score"]
        base = pred.groupby(ID)[reps].mean().join(static, how="left")
        base["y"] = base["event_observed"].astype(int)
        for sub_name, members in subsets.items():
            frame = base if members is None else base[base.index.isin(members)]
            seen.setdefault(frozenset(frame.index), f"{sub_name} de {run}")
            row = {"corrida": run, "autos": sub_name, "n": len(frame), "fallados": int(frame["y"].sum())}
            for st_name, cols in strata.items():
                values = [stratified_auc(frame, r, cols) for r in reps]
                row[st_name] = f"{np.mean(values):.3f} ± {np.std(values):.3f}" if len(values) > 1 else f"{values[0]:.3f}"
            rows.append(row)
    # Piso sin modelo, sobre cada conjunto de autos distinto: la fecha de producción sola
    # (producido antes = más riesgo).
    for ids, label in seen.items():
        frame = static[static.index.isin(ids)].copy()
        frame["y"] = frame["event_observed"].astype(int)
        frame["neg_pd"] = -frame["static_ProductionDay"]
        row = {"corrida": "−ProductionDay (sin modelo)", "autos": label, "n": len(frame), "fallados": int(frame["y"].sum())}
        for st_name, cols in strata.items():
            row[st_name] = f"{stratified_auc(frame, 'neg_pd', cols):.3f}"
        rows.append(row)

    table = pd.DataFrame(rows)
    print(table.to_string(index=False))
    out = resolve_path(cfg["output"])
    ensure_dir(out.parent)
    table.to_csv(out, index=False)
    print(f"\nEscrito en {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
