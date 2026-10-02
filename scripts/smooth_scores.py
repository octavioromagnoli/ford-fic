#!/usr/bin/env python
"""Media acumulada causal del score por vehículo sobre una corrida existente (F6, K1 y K3).

    python scripts/smooth_scores.py --config configs/exp_ss_cummean_r3.yaml            # arma y mide
    python scripts/smooth_scores.py --config configs/exp_ss_cummean_r3.yaml --audit    # + (a0) y (b)

Es K1 y K3 de `docs/reproducibilidad.md`, escrito para aplicarlo tal
cual:

* **Score.** Para cada repetición `r` y cada vehículo, las filas de dev ordenadas por
  `cut_odo`: el score de la fila `i` es el promedio de `score_r` en las filas `1..i` del mismo
  vehículo, la propia incluida. Solo mira hacia atrás, y todas las filas de un vehículo son
  out-of-fold del mismo fold (el split es por vehículo; se verifica). `score` es el promedio de
  las repeticiones, como en toda corrida.
* **No se reentrena nada:** el miembro es una corrida de `experiments/` con sus predicciones.
* **Métricas:** `scripts/train.py::evaluate_predictions`, la misma cuenta que toda corrida. El
  veredicto del preregistro (etiqueta corregida, pisos, nulo) lo aplican
  `scripts/eval_window_label.py` y `scripts/audit_detection_null.py` sobre la salida.

`--audit` reentrena al miembro con las features permutadas entre todas las filas (a0) y con las
`aux_` de calendario como `feat_` (b), con las funciones de `scripts/audit_model.py`, y les aplica
la misma media. Deja `audit.json` con el formato de `audit_model.py`, que es el que lee
`eval_window_label.py`.

Deja en `experiments/<name>/` lo mismo que `train.py` (`predictions.parquet`, `metrics.json`,
`config.yaml`, `lead_time_curve.csv`).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from scripts.audit_model import CALENDAR_AUX, evaluate, permute_all, promote_aux, score_metrics  # noqa: E402
from scripts.ensemble_rank import compare, decision_metrics, floor_metrics, load_metrics, n_repeats_of  # noqa: E402
from scripts.train import evaluate_predictions, load_or_make_splits, select_dev  # noqa: E402
from src.config import ensure_dir, load_config, repo_root, resolve_path, set_seed  # noqa: E402
from src.eval.metrics import dispersion  # noqa: E402

logger = logging.getLogger("smooth_scores")

KEY = ["vehicle_id", "cut_odo"]
METHODS = ("causal_cummean",)


def causal_cummean(predictions: pd.DataFrame) -> pd.DataFrame:
    """El promedio de cada `score_r{r}` en las filas del vehículo hasta la propia, por `cut_odo`.

    Falla si un vehículo tiene filas en dos folds de la misma repetición: el promedio mezclaría
    scores de modelos distintos, y alguno podría haber visto al vehículo.
    """
    if predictions.duplicated(KEY).any():
        raise ValueError(f"Hay pares {KEY} repetidos: la media acumulada necesita una fila por corte")
    n_repeats = n_repeats_of(predictions)
    out = predictions.sort_values(KEY, kind="stable").reset_index(drop=True)
    groups = out.groupby("vehicle_id", observed=True, sort=False)
    count = groups.cumcount().to_numpy(dtype=float) + 1.0
    stacked = []
    for r in range(n_repeats):
        folds = groups[f"fold_r{r}"].nunique()
        if (folds > 1).any():
            raise ValueError(f"{int((folds > 1).sum())} vehículo(s) con filas en dos folds de la repetición {r}")
        column = f"score_r{r}"
        smoothed = groups[column].cumsum().to_numpy(dtype=float) / count
        out[column] = smoothed
        stacked.append(smoothed)
    stacked = np.vstack(stacked)
    out["score"] = stacked.mean(axis=0)
    out["score_std"] = stacked.std(axis=0, ddof=0)
    return out


def smooth(predictions: pd.DataFrame, method: str) -> pd.DataFrame:
    if method not in METHODS:
        raise ValueError(f"method `{method}` desconocido; el preregistro es {METHODS}")
    return causal_cummean(predictions)


def audit_member(member: dict[str, Any], method: str, *, seed: int) -> dict[str, Any]:
    """(a0) y (b) con el miembro reentrenado igual que en `audit_model.py`, más la media."""
    cfg = load_config(member["config"])
    set_seed(int(cfg.get("seed", 42)))
    panel = select_dev(pd.read_parquet(resolve_path(cfg["data"]["panel"])), cfg)
    splits = load_or_make_splits(panel, cfg)
    reference = smooth(pd.read_parquet(resolve_path("experiments") / member["run"] / "predictions.parquet"), method)
    _, pred_a0 = evaluate(permute_all(panel, seed=seed), splits, cfg, label=f"(a0) {member['run']}")
    promoted, present = promote_aux(panel, CALENDAR_AUX)
    _, pred_b = evaluate(promoted, splits, cfg, label=f"(b) {member['run']} +{len(present)} aux_")
    ref_metrics = score_metrics(reference, label="referencia (media acumulada del miembro)")
    null = score_metrics(smooth(pred_a0, method), label=f"(a0) features permutadas entre todos (seed {seed}) + media")
    calendar = score_metrics(smooth(pred_b, method), label=f"(b) +{len(present)} columnas aux_ de calendario + media")
    delta_null = null["pr_auc"] - null["base_rate"]
    delta_roc = calendar["roc_auc"] - ref_metrics["roc_auc"]
    return {
        "config": member["config"], "seed": seed, "member": member["run"], "method": method,
        "audits": [ref_metrics, null, calendar],
        "passed": {"a0": bool(abs(delta_null) < 0.02)},
        "a0": {"pr_auc": null["pr_auc"], "base_rate": null["base_rate"], "delta": delta_null},
        "b": {"roc_reference": ref_metrics["roc_auc"], "roc_with_calendar": calendar["roc_auc"],
              "delta_roc": delta_roc, "marks": bool(delta_roc > 0.02), "columns": CALENDAR_AUX},
    }


def pct(values: list[float]) -> str:
    d = dispersion(values)
    return f"{100 * d['mean']:.1f}% ± {100 * d['std']:.1f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--audit", action="store_true", help="reentrena al miembro para (a0) y (b)")
    parser.add_argument("--audit-seed", type=int, default=0, help="semilla de la permutación de (a0)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    cfg = load_config(args.config)
    seed = set_seed(int(cfg.get("seed", 42)))
    scfg = cfg["smoothing"]
    method, member = scfg["method"], scfg["member"]
    experiments = resolve_path(cfg.get("output_dir", "experiments"))

    raw = pd.read_parquet(experiments / member["run"] / "predictions.parquet")
    predictions = smooth(raw, method)
    n_repeats = n_repeats_of(predictions)
    logger.info("%s de %s | %d filas | %d vehículos | R=%d", method, member["run"], len(predictions),
                predictions["vehicle_id"].nunique(), n_repeats)

    evaluated, curve = evaluate_predictions(predictions, cfg, n_repeats=n_repeats, seed=seed)
    candidate = decision_metrics(evaluated)
    reference_run = cfg["compare"]["reference"]
    reference = decision_metrics(load_metrics(reference_run, experiments))
    member_decision = decision_metrics(load_metrics(member["run"], experiments))
    floor = floor_metrics(predictions, cfg.get("eval", {}))
    audit = audit_member(member, method, seed=args.audit_seed) if args.audit else None

    metrics = {
        **evaluated,
        "panel_build": load_metrics(member["run"], experiments).get("panel_build"),
        "smoothing": {"method": method, "member": member,
                      "checks": {"rows": len(predictions), "one_fold_per_vehicle": True}},
        "decision_hard_label": {
            "candidate": candidate,
            "vs_reference": compare(candidate, reference),
            "vs_member": compare(candidate, member_decision),
            "vs_positional_floor": compare(candidate, floor),
            "note": "Etiqueta dura (D), informativa. Decide la corregida (V): eval_window_label.py",
        },
        "smoothing_audit": audit,
    }
    run_name = cfg["name"]
    out_dir = ensure_dir(experiments / run_name)
    predictions.to_parquet(out_dir / "predictions.parquet", index=False)
    if not curve.empty:
        curve.to_csv(out_dir / "lead_time_curve.csv", index=False)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float), encoding="utf-8")
    saved_cfg = {**cfg, "model": cfg.get("model") or {"name": f"{method}({member['run']})"}}
    (out_dir / "config.yaml").write_text(yaml.safe_dump(saved_cfg, sort_keys=False, allow_unicode=True),
                                         encoding="utf-8")
    if audit is not None:
        (out_dir / "audit.json").write_text(json.dumps(audit, indent=2, default=float), encoding="utf-8")

    print(f"\n== {run_name}: {method} de {member['run']} (etiqueta dura D, informativa) ==")
    for label, d in (("candidato", candidate), (f"miembro {member['run']}", member_decision),
                     (f"referencia {reference_run}", reference), ("piso cut_odo", floor)):
        lead = dispersion(d["lead_km"])
        lift = dispersion(d["vehicle_lift_mean"])
        print(f"{label:<55} {pct(d['detection']):<16} {'/'.join(map(str, d['n_detected'])):<10} "
              f"{lead['mean']:>7.0f} ± {lead['std']:<6.0f} lift {lift['mean']:.3f} ± {lift['std']:.3f}")
    spread = evaluated["when_spread"]
    print(f"(a') {spread['when_delta']['mean']:+.4f} ± {spread['when_delta']['std']:.4f} | "
          f"PR-AUC fila {evaluated['oof']['pr_auc']:.4f} | Brier {evaluated['oof']['brier']:.4f}")
    if audit:
        print(f"(a0) PR-AUC {audit['a0']['pr_auc']:.4f} vs tasa base {audit['a0']['base_rate']:.4f} "
              f"({audit['a0']['delta']:+.4f}) — {'PASS' if audit['passed']['a0'] else 'FALLA'} | "
              f"(b) ROC {audit['b']['roc_reference']:.4f} → {audit['b']['roc_with_calendar']:.4f} "
              f"({audit['b']['delta_roc']:+.4f})")
    print(f"escrito: {out_dir.relative_to(repo_root())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
