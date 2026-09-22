#!/usr/bin/env python
"""Chequeo univariado de las `feat_*_hist_delta` sobre dev, antes de entrenar nada.

    python scripts/audit_history_univariate.py --config configs/exp_survival_stacking_history_r3.yaml

Por cada feature de `configs/data/features_history.yaml`: ROC contra `label` en dev (sin
orientar: > 0,5 es "historia por encima de la ventana ⇒ más riesgo") y sus ρ de Spearman
con el mes del corte y con `cut_odo`. La referencia es la versión sin modelo del doc de
TimesFM: ROC 0,602 para regeneraciones y 0,560 para el nivel
(docs/memoria/f3-timesfm-zeroshot.md, medida sobre el build `2026-09-19`).

Van también los controles que la regla 6 pide para toda métrica que dependa del largo
del historial: el ROC de `aux_hist_km_covered` solo y la ρ de cada feature con él. Si el
largo de la historia ya separa, el desvío puede estar midiéndolo a él.

El recorte a dev sale de `scripts/train.py::select_dev`, igual que en el entrenamiento.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.config import load_config, resolve_path  # noqa: E402

from scripts.train import select_dev  # noqa: E402


def univariate(dev: pd.DataFrame, column: str) -> dict[str, float]:
    ok = dev[column].notna()
    frame = dev.loc[ok]
    month = frame["cut_date"].dt.year * 12 + frame["cut_date"].dt.month
    return {
        "column": column,
        "coverage": float(ok.mean()),
        "n_pos": int(frame["label"].sum()),
        "roc_auc": float(roc_auc_score(frame["label"], frame[column])),
        "rho_month": float(frame[column].corr(month, method="spearman")),
        "rho_cut_odo": float(frame[column].corr(frame["cut_odo"], method="spearman")),
        "rho_hist_km": float(frame[column].corr(frame["aux_hist_km_covered"], method="spearman")),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, help="YAML del experimento (panel + holdout)")
    parser.add_argument("--features", default="configs/data/features_history.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    panel = pd.read_parquet(resolve_path(cfg["data"]["panel"]))
    dev = select_dev(panel, cfg)
    spec = load_config(args.features)
    names = [e["name"] for e in spec["features"]]
    columns = [f"feat_{n}" for n in names]
    sides = [f"aux_{n}_{side}" for n in names for side in ("hist", "window")]
    controls = ["aux_hist_km_covered", "feat_regenerations_trend_per_1000km", "feat_idle_frac"]

    table = pd.DataFrame([univariate(dev, c) for c in columns + sides + controls])
    print(f"dev: {len(dev)} filas · {dev['vehicle_id'].nunique()} vehículos · {int(dev['label'].sum())} positivas")
    print(table.round(3).to_string(index=False))
    for name, column in zip(names, columns):
        base = spec["window_check"][name]
        rho = dev[column].corr(dev[base], method="spearman")
        rho_trend = dev[column].corr(dev["feat_regenerations_trend_per_1000km"], method="spearman")
        print(f"ρ({column}, {base}) = {rho:+.3f} · ρ con la tendencia dentro de W = {rho_trend:+.3f}")


if __name__ == "__main__":
    main()
