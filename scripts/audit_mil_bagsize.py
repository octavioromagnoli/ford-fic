#!/usr/bin/env python
"""Auditoría del eje de vehículo: ¿el lift es señal o es el tamaño de la bolsa?

    python scripts/audit_mil_bagsize.py f3-lgbm-panel-v1-mil

Agregar los cortes de un vehículo a un score único (MIL) sube el lift respecto de
la métrica por fila, pero parte de esa suba puede no ser del modelo: en el panel v1
las bolsas de los vehículos con evento son más grandes que las de los sanos (el
emparejado por odómetro × mes le da al sano solo los cortes que hacen falta para
llenar la celda), y `max`, `topk` y sobre todo `noisy_or` crecen con el tamaño de
la bolsa aunque los scores no digan nada.

El control es una permutación: se barajan los scores entre todas las filas —se
destruye cualquier relación score↔etiqueta y se conserva exactamente el tamaño de
cada bolsa— y se vuelve a medir. Lo que queda bajo el nulo es lo que la agregación
saca del tamaño; el lift de una agregación solo cuenta si supera a *su* nulo, no al
1,0 de una tasa base. CLAUDE.md, regla 6.

Imprime además la detección a nivel vehículo bajo el presupuesto de falsas alarmas
de la corrida: cuántos de los vehículos con evento quedan marcados si se acepta
marcar como mucho ese porcentaje de sanos.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import resolve_path  # noqa: E402
from src.eval.metrics import (  # noqa: E402
    DEFAULT_TOPK,
    VEHICLE_AGGREGATIONS,
    pr_auc,
    roc_auc,
    vehicle_scores,
)


def detection_at_budget(bags: pd.DataFrame, false_alarms_per_1000: float) -> dict[str, float]:
    """Detección de vehículos con evento aceptando como mucho ese % de sanos marcados.

    El umbral se elige sobre los scores de las bolsas: el más bajo que respeta el
    presupuesto. Es el análogo por vehículo de `operating_point()`, que trabaja sobre
    la regla de alerta sostenida por cortes.
    """
    healthy = bags.loc[bags["label"] == 0, "score"].to_numpy()
    events = bags.loc[bags["label"] == 1, "score"].to_numpy()
    budget = false_alarms_per_1000 / 1000.0
    best = {"threshold": float("nan"), "detection_rate": 0.0, "false_alarm_rate": 0.0}
    for threshold in np.unique(bags["score"].to_numpy()):
        far = float((healthy >= threshold).mean())
        if far > budget:
            continue
        detection = float((events >= threshold).mean())
        if detection > best["detection_rate"]:
            best = {
                "threshold": float(threshold),
                "detection_rate": detection,
                "false_alarm_rate": far,
            }
    return best


def audit(
    predictions: pd.DataFrame,
    *,
    hows: tuple[str, ...],
    k: int,
    n_permutations: int,
    seed: int,
    budget: float,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    scores = predictions["score"].to_numpy(dtype=float)
    base_rate = float(vehicle_scores(predictions, "max")["label"].mean())

    real = {}
    for how in hows:
        bags = vehicle_scores(predictions, how, k=k)
        point = detection_at_budget(bags, budget)
        real[how] = {
            "pr_auc": pr_auc(bags["label"], bags["score"]),
            "roc_auc": roc_auc(bags["label"], bags["score"]),
            "detection_rate": point["detection_rate"],
            "false_alarm_rate": point["false_alarm_rate"],
        }

    null: dict[str, dict[str, list[float]]] = {
        how: {"pr_auc": [], "roc_auc": [], "detection_rate": []} for how in hows
    }
    shuffled = predictions.copy()
    for _ in range(n_permutations):
        shuffled["score"] = rng.permutation(scores)
        for how in hows:
            bags = vehicle_scores(shuffled, how, k=k)
            null[how]["pr_auc"].append(pr_auc(bags["label"], bags["score"]))
            null[how]["roc_auc"].append(roc_auc(bags["label"], bags["score"]))
            null[how]["detection_rate"].append(detection_at_budget(bags, budget)["detection_rate"])

    rows = []
    for how in hows:
        draws = np.asarray(null[how]["pr_auc"], dtype=float)
        roc_draws = np.asarray(null[how]["roc_auc"], dtype=float)
        det_draws = np.asarray(null[how]["detection_rate"], dtype=float)
        rows.append(
            {
                "how": how,
                "pr_auc": real[how]["pr_auc"],
                "lift": real[how]["pr_auc"] / base_rate,
                "roc_auc": real[how]["roc_auc"],
                "lift_null": float(draws.mean() / base_rate),
                "lift_null_std": float(draws.std(ddof=0) / base_rate),
                "lift_null_p95": float(np.quantile(draws, 0.95) / base_rate),
                "p_value": float((draws >= real[how]["pr_auc"]).mean()),
                "roc_auc_null": float(roc_draws.mean()),
                "detection_rate": real[how]["detection_rate"],
                "detection_rate_null": float(det_draws.mean()),
                "false_alarm_rate": real[how]["false_alarm_rate"],
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", help="Nombre de la corrida en experiments/")
    parser.add_argument("--dir", default="experiments")
    parser.add_argument("--n-permutations", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--topk", type=int, default=DEFAULT_TOPK)
    args = parser.parse_args()

    run_dir = resolve_path(args.dir) / args.run
    predictions = pd.read_parquet(run_dir / "predictions.parquet")
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    budget = float(metrics.get("operating_point_budget_per_1000", 50))

    bags = vehicle_scores(predictions, "max")
    sizes = bags.groupby("label")["n_cuts"]
    print(f"Corrida: {args.run} | {len(bags)} vehículos, {int(bags['label'].sum())} con evento "
          f"| tasa base de bolsas {bags['label'].mean():.4f} (la de filas: "
          f"{metrics['oof']['base_rate']:.4f})")
    print(f"Tamaño de bolsa | con evento: media {sizes.mean()[1]:.1f}, mediana {sizes.median()[1]:.0f}"
          f" | sanos: media {sizes.mean()[0]:.1f}, mediana {sizes.median()[0]:.0f}")
    size_ap = pr_auc(bags["label"], bags["n_cuts"])
    print(f"El tamaño de bolsa SOLO, como score: PR-AUC={size_ap:.4f} "
          f"(lift {size_ap / bags['label'].mean():.2f}×), ROC-AUC={roc_auc(bags['label'], bags['n_cuts']):.4f}")
    print()

    table = audit(
        predictions,
        hows=VEHICLE_AGGREGATIONS,
        k=args.topk,
        n_permutations=args.n_permutations,
        seed=args.seed,
        budget=budget,
    )
    print(f"Nulo: {args.n_permutations} permutaciones del score entre filas (semilla {args.seed}); "
          f"detección con <= {budget:.0f} falsas alarmas / 1000 sanos")
    print()
    print(f"| {'agregación':<10} | PR-AUC | lift | lift nulo | p | ROC-AUC | ROC nulo | detección | det. nula |")
    print(f"|{'-' * 12}|--------|------|-----------|---|---------|----------|-----------|-----------|")
    for row in table.itertuples():
        print(
            f"| {row.how:<10} | {row.pr_auc:.3f}  | {row.lift:.2f}× | "
            f"{row.lift_null:.2f} ± {row.lift_null_std:.2f} | {row.p_value:.3f} | "
            f"{row.roc_auc:.3f}   | {row.roc_auc_null:.3f}    | "
            f"{100 * row.detection_rate:.0f}%       | {100 * row.detection_rate_null:.0f}% |"
        )
    print()
    print("Una agregación solo aporta si su lift supera al de su propio nulo: el 1,0 de la "
          "tasa base no es el piso cuando la bolsa de los que fallan es más grande.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
