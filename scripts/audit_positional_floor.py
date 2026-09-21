#!/usr/bin/env python
"""Compara una corrida contra el piso posicional: ¿le gana al odómetro pelado?

    python scripts/audit_positional_floor.py f3-survival-stacking
    python scripts/audit_positional_floor.py f3-lgbm-panel-v1-regen15 --budget 50

Lee `experiments/<corrida>/predictions.parquet`, calcula las métricas de la corrida y
las del piso —el mismo frame con `score = cut_odo`— y dice, métrica por métrica, si la
corrida le gana.

**Por qué existe.** Dentro de un vehículo que falla, la etiqueta es una función
determinista de la posición del corte: el historial de un fallado termina en el evento,
así que "estar cerca del final" y "estar etiquetado" son la misma cosa. Medido sobre dev,
la fracción positiva por posición desde el final va 1,00 / 1,00 / 0,98 / 0,93 / 0,92 /
0,87 y después cae a 0,00 de golpe. Un score que sea literalmente el odómetro ordena eso
casi perfecto **sin saber nada de degradación**.

Consecuencia: las métricas del eje "cuándo" —(a'), PR-AUC entre fallados— y el PR-AUC por
fila **no se leen contra cero ni contra la tasa base**. Se leen contra esto. Es la
contraparte de `cohort_ceiling()` en el eje "qué auto".

Las dos métricas que el piso **no** gana son las que dependen de las falsas alarmas (el
punto de operación) y el eje vehículo con agregación `mean`: un score posicional también
alerta los últimos cortes de los sanos, y ahí paga. Son las honestas.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import resolve_path  # noqa: E402
from src.eval.metrics import (  # noqa: E402
    lead_time_curve,
    operating_point,
    pr_auc,
    pr_auc_within_failed,
    roc_auc,
    vehicle_metrics,
    when_contribution,
)

logger = logging.getLogger("audit_positional_floor")

# Escala fija, no ajustada: solo lleva el odómetro a [0, 1] sin tocar el orden.
SCALE_KM = 100_000.0


def measure(frame: pd.DataFrame, *, budget: float, k: int) -> dict[str, float]:
    point = operating_point(lead_time_curve(frame, k_consecutive=k), max_false_alarms_per_1000=budget)
    return {
        "pr_auc": pr_auc(frame["label"], frame["score"]),
        "roc_auc": roc_auc(frame["label"], frame["score"]),
        "within_failed": pr_auc_within_failed(
            frame["label"], frame["score"], frame["event_observed"]
        )["pr_auc"],
        "when_delta": when_contribution(frame)["delta"],
        "vehicle_lift_mean": vehicle_metrics(frame, how="mean")["pr_auc_lift"],
        "detection_rate": point["detection_rate"] if point else float("nan"),
        "median_lead_km": point["median_lead_km"] if point else float("nan"),
    }


# métrica -> (etiqueta, ¿el piso la gana por construcción?)
METRICS = {
    "pr_auc": ("PR-AUC por fila", True),
    "roc_auc": ("ROC-AUC por fila", True),
    "within_failed": ("PR-AUC entre fallados", True),
    "when_delta": ("(a') aporte del cuándo", True),
    "vehicle_lift_mean": ("lift por vehículo (mean)", False),
    "detection_rate": ("detección @ presupuesto", False),
    "median_lead_km": ("anticipación mediana [km]", False),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Corrida contra el piso posicional")
    parser.add_argument("run", help="nombre de la corrida en experiments/")
    parser.add_argument("--experiments", default="experiments")
    parser.add_argument("--budget", type=float, default=50.0, help="falsas alarmas por 1000 sanos")
    parser.add_argument("--k-consecutive", type=int, default=2)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    path = resolve_path(args.experiments) / args.run / "predictions.parquet"
    if not path.exists():
        raise FileNotFoundError(f"No existe {path}. ¿Corriste `scripts/train.py` para `{args.run}`?")
    run_frame = pd.read_parquet(path)

    floor_frame = run_frame.copy()
    floor_frame["score"] = floor_frame["cut_odo"] / SCALE_KM

    model = measure(run_frame, budget=args.budget, k=args.k_consecutive)
    floor = measure(floor_frame, budget=args.budget, k=args.k_consecutive)

    logger.info("\n%s  vs.  piso posicional (score = cut_odo)", args.run)
    logger.info("presupuesto: <= %.0f falsas alarmas / 1000 sanos | k_consecutive=%d\n",
                args.budget, args.k_consecutive)
    logger.info("%-28s %10s %10s   %s", "métrica", "corrida", "piso", "veredicto")
    logger.info("%s", "-" * 72)

    degenerate = []
    for key, (label, floor_expected_to_win) in METRICS.items():
        a, b = model[key], floor[key]
        wins = a > b
        mark = "gana" if wins else "PIERDE contra el piso"
        logger.info("%-28s %10.4f %10.4f   %s", label, a, b, mark)
        if not wins and floor_expected_to_win:
            degenerate.append(label)

    logger.info("%s", "-" * 72)
    if degenerate:
        logger.info(
            "\nMétricas donde el piso gana: %s.\n"
            "No son evidencia de anticipación: un score sin información de degradación\n"
            "las alcanza. Las que valen son el punto de operación y el eje vehículo.",
            "; ".join(degenerate),
        )
    else:
        logger.info("\nLa corrida le gana al piso en todas las métricas del eje `cuándo`.")


if __name__ == "__main__":
    main()
