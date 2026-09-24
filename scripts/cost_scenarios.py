#!/usr/bin/env python
"""Punto de operación de K2 bajo escenarios de costo (reporte, nunca selección de modelo).

    python scripts/cost_scenarios.py --config configs/cost_scenarios_k2.yaml

Para cada escenario de costos × efectividad de la prevención × prevalencia real, elige el umbral
que minimiza el costo esperado por vehículo sobre la curva de detección contra falsas alarmas de
cada repetición (la misma de `train.py`, con las dos políticas triviales como extremos) y reporta:
la tasa de falsas alarmas y la detección óptimas, el ahorro contra la mejor política trivial, el
ahorro en USD cada 1.000 vehículos y la parte de ese ahorro que no logra un score al azar del mismo
largo (regla 6). La cuenta está en el encabezado del YAML y en
`src/eval/dashboard_data.py::expected_cost`, la misma que usa la página de costos del dashboard.

El umbral se elige sobre las mismas predicciones que se miden (como `cost_ratio_sweep`), así que el
ahorro es optimista. Se marca cuándo el óptimo cae por encima de lo verificado fuera de muestra.
Deja `experiments/decision-k2/cost_scenarios.csv`.
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.config import ensure_dir, load_config  # noqa: E402
from src.eval.dashboard_data import cost_optimum, curve_points, load_k2, repeat_curve, rows_for  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    d = load_k2(cfg["dashboard_config"])
    if d.decision is None:
        raise FileNotFoundError("Falta la capa de decisión: python scripts/decision_layer.py --config "
                                "configs/exp_decision_k2.yaml")
    rows = rows_for(d, cfg["label"])
    curves = [curve_points(repeat_curve(rows, r, d.eval_cfg)) for r in range(d.n_repeats)]

    records = []
    for (name, c), e, pi in itertools.product(cfg["scenarios"].items(), cfg["prevention_effectiveness"],
                                              cfg["prevalence"]):
        opt = cost_optimum(curves, d.decision, cfg["label"], pi=pi, insp=c["insp"], prev=c["prev"],
                           fail=c["fail"], e=e)
        records.append({"escenario": name, "e": e, "pi": pi, **opt,
                        "fuera_de_lo_validado": opt["fa_opt_max"] > cfg["validated_max_fa"]})
    out = pd.DataFrame(records)
    path = ensure_dir(Path("experiments") / "decision-k2") / "cost_scenarios.csv"
    out.to_csv(path, index=False)
    pd.set_option("display.width", 220)
    show = out.assign(fa_opt=out["fa_opt"].map("{:.1%}".format), det_opt=out["det_opt"].map("{:.1%}".format),
                      savings_vs_trivial=out["savings_vs_trivial"].map("{:.0%}".format))
    print(show[["escenario", "e", "pi", "ratio_benefit_insp", "fa_opt", "det_opt", "trivial", "savings_vs_trivial",
                "savings_per_1000", "savings_over_null_per_1000", "fuera_de_lo_validado"]].round(2).to_string(index=False))
    print(f"\nescrito: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
