#!/usr/bin/env python
"""Punto de operación de K2 bajo escenarios de costo (reporte, nunca selección de modelo).

    python scripts/cost_scenarios.py --config configs/cost_scenarios_k2.yaml

Para cada escenario de costos × efectividad de la prevención × prevalencia real, elige el umbral
que minimiza el costo esperado por vehículo sobre la curva de detección contra falsas alarmas de
cada repetición (la misma de `train.py`, con las dos políticas triviales como extremos) y reporta:
la tasa de falsas alarmas y la detección óptimas, el ahorro contra la mejor política trivial y el
ahorro en USD cada 1.000 vehículos. La cuenta está en el encabezado del YAML.

El umbral se elige sobre las mismas predicciones que se miden (como `cost_ratio_sweep`), así que el
ahorro es optimista. Se marca cuándo el óptimo cae por encima de lo verificado fuera de muestra.
Deja `experiments/decision-k2/cost_scenarios.csv`.
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.config import ensure_dir, load_config  # noqa: E402
from src.eval.dashboard_data import load_k2, repeat_curve, rows_for  # noqa: E402


def curve_points(curve: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(FPR, TPR) de la curva, con las dos políticas triviales agregadas como extremos."""
    fpr = np.append(curve["false_alarms_per_1000"].to_numpy(dtype=float) / 1000.0, [0.0, 1.0])
    tpr = np.append(curve["detection_rate"].to_numpy(dtype=float), [0.0, 1.0])
    return fpr, tpr


def expected_cost(fpr: np.ndarray, tpr: np.ndarray, *, pi: float, insp: float, prev: float, fail: float,
                  e: float) -> np.ndarray:
    """Costo esperado por vehículo de la flota (USD) en cada punto de la curva."""
    benefit = e * fail - prev - insp
    return pi * fail - pi * tpr * benefit + (1.0 - pi) * fpr * insp


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    d = load_k2(cfg["dashboard_config"])
    rows = rows_for(d, cfg["label"])
    curves = [curve_points(repeat_curve(rows, r, d.eval_cfg)) for r in range(d.n_repeats)]

    # Regla 6: el nulo que conserva el largo de cada historial también detecta algo. En los presupuestos
    # de la capa de decisión, K2 y el nulo se comparan en los mismos puntos (más los dos triviales).
    pts = pd.DataFrame(d.decision["labels"][cfg["label"]]["curve_points"]).groupby("budget_per_1000").mean(numeric_only=True)
    budgets = pts.index.to_numpy(dtype=float) / 1000.0
    k2_pts = (np.r_[0.0, pts["fa_realized_per_1000"].to_numpy() / 1000.0, 1.0], np.r_[0.0, pts["detection"].to_numpy(), 1.0])
    null_pts = (np.r_[0.0, budgets, 1.0], np.r_[0.0, pts["null_mean"].to_numpy(), 1.0])

    records = []
    for (name, c), e, pi in itertools.product(cfg["scenarios"].items(), cfg["prevention_effectiveness"],
                                              cfg["prevalence"]):
        per_repeat = []
        for fpr, tpr in curves:
            cost = expected_cost(fpr, tpr, pi=pi, insp=c["insp"], prev=c["prev"], fail=c["fail"], e=e)
            i = int(np.argmin(cost))
            # Las triviales, explícitas (la curva viene ordenada por umbral, no por FPR).
            none, everyone = (expected_cost(np.array([f]), np.array([t]), pi=pi, insp=c["insp"], prev=c["prev"],
                                            fail=c["fail"], e=e)[0] for f, t in ((0.0, 0.0), (1.0, 1.0)))
            per_repeat.append((fpr[i], tpr[i], cost[i], none, everyone))
        a = np.array(per_repeat)
        best_trivial = np.minimum(a[:, 3], a[:, 4]).mean()
        kw = dict(pi=pi, insp=c["insp"], prev=c["prev"], fail=c["fail"], e=e)
        k2_on_budgets = expected_cost(*k2_pts, **kw).min()
        null_on_budgets = expected_cost(*null_pts, **kw).min()
        records.append({
            "escenario": name, "e": e, "pi": pi,
            "ratio_B_insp": (e * c["fail"] - c["prev"] - c["insp"]) / c["insp"],
            "fa_opt": a[:, 0].mean(), "fa_opt_max": a[:, 0].max(), "det_opt": a[:, 1].mean(),
            "costo_modelo": a[:, 2].mean(), "costo_no_alertar": a[:, 3].mean(), "costo_todos": a[:, 4].mean(),
            "trivial": "no alertar" if a[:, 3].mean() <= a[:, 4].mean() else "alertar a todos",
            "ahorro_vs_trivial": 1.0 - a[:, 2].mean() / best_trivial if best_trivial > 0 else 0.0,
            "ahorro_usd_por_1000": 1000.0 * (best_trivial - a[:, 2].mean()),
            # Lo que ahorra K2 por encima de un score al azar del mismo largo, en los presupuestos de la capa
            # de decisión (2/5/10/20%): esa es la parte del ahorro que sale del modelo.
            "ahorro_sobre_nulo_usd_por_1000": 1000.0 * (null_on_budgets - k2_on_budgets),
            "fuera_de_lo_validado": bool(a[:, 0].max() > cfg["validated_max_fa"]),
        })
    out = pd.DataFrame(records)
    path = ensure_dir(Path("experiments") / "decision-k2") / "cost_scenarios.csv"
    out.to_csv(path, index=False)
    pd.set_option("display.width", 220)
    show = out.assign(fa_opt=out["fa_opt"].map("{:.1%}".format), det_opt=out["det_opt"].map("{:.1%}".format),
                      ahorro_vs_trivial=out["ahorro_vs_trivial"].map("{:.0%}".format),
                      ahorro_usd_por_1000=out["ahorro_usd_por_1000"].round(0))
    print(show[["escenario", "e", "pi", "ratio_B_insp", "fa_opt", "det_opt", "trivial", "ahorro_vs_trivial",
                "ahorro_usd_por_1000", "ahorro_sobre_nulo_usd_por_1000", "fuera_de_lo_validado"]].round(2).to_string(index=False))
    print(f"\nescrito: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
