#!/usr/bin/env python
"""Capa de decisión sobre una corrida existente: dónde se opera el score y si el punto se sostiene.

    python scripts/decision_layer.py --config configs/exp_decision_k2.yaml

Es el §6 del preregistro F7 (y F5 §3.6). No reentrena nada y no cambia el orden de los autos:
todo lo de acá es monótono en el score, así que no es un candidato y no consume presupuesto.

1. **Curva** de detección contra falsas alarmas a varios presupuestos (`budgets_per_1000`), por
   repetición, con la misma cuenta que `train.py` (`lead_time_curve` + `operating_point`), la
   anticipación mediana y el nulo de tamaño de bolsa en cada punto (regla 6: el score permutado
   entre filas, como `audit_detection_null.py`).
2. **¿El punto se sostiene fuera de muestra?** Hoy el umbral del 5% se elige mirando los mismos
   sanos con los que se mide, así que la tasa realizada es ≤ 5% por construcción. Acá, por fold
   de cada repetición, el umbral se fija con los sanos de los otros folds y se aplica al fold que
   queda afuera:
   - `empirical`: el umbral de la grilla que respeta alpha en los sanos de ajuste (lo de hoy);
   - `neyman_pearson`: el estadístico de orden de Tong, Feng & Li (2018, "umbrella"), que
     garantiza P(tasa real > alpha) ≤ delta si los sanos nuevos son intercambiables.
   Se reporta la tasa de falsas alarmas realizada en los sanos que no fijaron el umbral y la
   detección que queda, más el IC de Clopper-Pearson de la tasa dentro de muestra.

Un vehículo alerta a τ si y solo si `max_i min(s_i..s_{i+k−1}) ≥ τ` (`vehicle_levels` de
`audit_detection_null.py`, verificado ahí contra `operating_point`). Deja
`experiments/<name>/decision_layer.json` y `curve.csv`.
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
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from scripts.audit_detection_null import check_fast_detection, vehicle_levels  # noqa: E402
from scripts.ensemble_rank import n_repeats_of  # noqa: E402
from scripts.eval_window_label import attach_panel_columns  # noqa: E402
from scripts.train import select_dev  # noqa: E402
from src.config import ensure_dir, load_config, resolve_path  # noqa: E402
from src.eval.metrics import dispersion, first_alert_lead_times, lead_time_curve, operating_point  # noqa: E402

logger = logging.getLogger("decision_layer")

KEY = ["vehicle_id", "cut_odo"]


def grid_of(scores: np.ndarray, n_thresholds: int) -> np.ndarray:
    """La grilla de umbrales de `lead_time_curve` (cuantiles + "no alerto nunca")."""
    grid = np.quantile(scores, np.linspace(0.0, 1.0, n_thresholds))
    return np.unique(np.append(grid, np.nextafter(scores.max(), np.inf)))


def np_threshold(healthy_levels: np.ndarray, alpha: float, delta: float) -> float:
    """Umbral de Neyman-Pearson (umbrella): alerta si nivel > t_(k*), con
    k* = min{k : P(Bin(n, 1−α) ≥ k) ≤ δ}. Sin k* posible (n chico) no alerta nunca."""
    t = np.sort(healthy_levels)
    n = len(t)
    for k in range(1, n + 1):
        if stats.binom.sf(k - 1, n, 1.0 - alpha) <= delta:
            return float(np.nextafter(t[k - 1], np.inf))
    return float("inf")


# --- 1 · curva ----------------------------------------------------------------------

def curve_block(rows: pd.DataFrame, all_rows: pd.DataFrame, mask: np.ndarray, cfg: dict, eval_cfg: dict,
                n_repeats: int) -> tuple[list[dict], pd.DataFrame]:
    k, n_thr = int(eval_cfg.get("k_consecutive", 2)), int(eval_cfg.get("n_thresholds", 50))
    rng = np.random.default_rng(cfg["seed"])
    out, curves = [], []
    for r in range(n_repeats):
        col = f"score_r{r}"
        check_fast_detection(rows, col, eval_cfg)
        curve = lead_time_curve(rows.assign(score=rows[col]), n_thresholds=n_thr, k_consecutive=k)
        curves.append(curve.assign(repeat=r))
        # nulo: el score permutado entre todas las filas de dev, medido en las filas de la etiqueta
        null = {b: [] for b in cfg["budgets_per_1000"]}
        raw = all_rows[col].to_numpy(dtype=float)
        for _ in range(int(cfg["null_permutations"])):
            s = rng.permutation(raw)[mask]
            levels, event, _ = vehicle_levels(rows, s, k)
            thr = grid_of(s, n_thr)
            fa = 1000.0 * (levels[event == 0][None, :] >= thr[:, None]).mean(axis=1)
            det = (levels[event == 1][None, :] >= thr[:, None]).mean(axis=1)
            for b in null:
                ok = fa <= b
                null[b].append(float(det[ok].max()) if ok.any() else 0.0)
        for b in cfg["budgets_per_1000"]:
            p = operating_point(curve, max_false_alarms_per_1000=float(b))
            out.append({"repeat": r, "budget_per_1000": b,
                        "detection": float(p["detection_rate"]) if p else 0.0,
                        "n_detected": int(p["n_detected"]) if p else 0,
                        "n_event_vehicles": int(p["n_event_vehicles"]) if p else None,
                        "fa_realized_per_1000": float(p["false_alarms_per_1000"]) if p else 0.0,
                        "median_lead_km": float(p["median_lead_km"]) if p else float("nan"),
                        "null_mean": float(np.mean(null[b])), "null_p95": float(np.quantile(null[b], 0.95))})
    return out, pd.concat(curves, ignore_index=True)


# --- 2 · fuera de muestra -----------------------------------------------------------

def holdout_block(rows: pd.DataFrame, cfg: dict, eval_cfg: dict, n_repeats: int) -> list[dict]:
    k, n_thr = int(eval_cfg.get("k_consecutive", 2)), int(eval_cfg.get("n_thresholds", 50))
    ids = rows["vehicle_id"].to_numpy()
    first = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1]])
    vehicles = ids[first]
    out = []
    for r in range(n_repeats):
        col = f"score_r{r}"
        s = rows[col].to_numpy(dtype=float)
        levels, event, _ = vehicle_levels(rows, s, k)
        folds = rows[f"fold_r{r}"].to_numpy()[first]
        row_folds = rows[f"fold_r{r}"].to_numpy()
        n_healthy_all = int((event == 0).sum())
        for alpha in cfg["alphas"]:
            # dentro de muestra (lo de hoy): el umbral de la grilla con todos los sanos
            thr_all = grid_of(s, n_thr)
            fa_all = (levels[event == 0][None, :] >= thr_all[:, None]).sum(axis=1)
            feasible = fa_all <= alpha * n_healthy_all
            fa_in = int(fa_all[feasible].min() if not feasible.any() else fa_all[np.flatnonzero(feasible)[0]])
            ci_hi = float(stats.beta.ppf(0.95, fa_in + 1, n_healthy_all - fa_in)) if fa_in < n_healthy_all else 1.0
            for method in ("empirical", "neyman_pearson"):
                fa = det = n_h = n_f = 0
                leads: list[float] = []
                for f in np.unique(folds):
                    fit_h = (folds != f) & (event == 0)
                    if method == "empirical":
                        thr = grid_of(s[row_folds != f], n_thr)
                        ok = 1000.0 * (levels[fit_h][None, :] >= thr[:, None]).mean(axis=1) <= 1000.0 * alpha
                        tau = float(thr[np.flatnonzero(ok)[0]]) if ok.any() else float("inf")
                    else:
                        tau = np_threshold(levels[fit_h], alpha, float(cfg["np_delta"]))
                    held = folds == f
                    fa += int((levels[held & (event == 0)] >= tau).sum())
                    det += int((levels[held & (event == 1)] >= tau).sum())
                    n_h += int((held & (event == 0)).sum())
                    n_f += int((held & (event == 1)).sum())
                    if np.isfinite(tau):
                        sub = rows[np.isin(rows["vehicle_id"].to_numpy(), vehicles[held])].assign(score=s[np.isin(ids, vehicles[held])])
                        leads += first_alert_lead_times(sub, tau, k_consecutive=k)["lead_times_km"].tolist()
                out.append({"repeat": r, "alpha": alpha, "method": method,
                            "fa_heldout": fa / n_h, "n_fa_heldout": fa, "n_healthy": n_h,
                            "detection_heldout": det / n_f, "n_detected": det, "n_failed": n_f,
                            "median_lead_km": float(np.median(leads)) if leads else float("nan"),
                            "fa_in_sample": fa_in / n_healthy_all, "fa_in_sample_ci95_hi": ci_hi})
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(message)s")
    cfg = load_config(args.config)
    run_cfg = load_config(cfg["run_config"])
    eval_cfg = run_cfg.get("eval", {})
    evaluable = run_cfg["window_eval"]["evaluable_column"]
    experiments = resolve_path(run_cfg.get("output_dir", "experiments"))
    name = run_cfg["name"]

    preds = pd.read_parquet(experiments / name / "predictions.parquet")
    panel = select_dev(pd.read_parquet(resolve_path(run_cfg["data"]["panel"])), run_cfg)
    preds = attach_panel_columns(preds, panel, [evaluable]).sort_values(KEY, kind="stable").reset_index(drop=True)
    n_repeats = n_repeats_of(preds)
    masks = {"corrected": preds[evaluable].to_numpy(dtype=int) == 1, "hard": np.ones(len(preds), dtype=bool)}

    result: dict[str, Any] = {"run": name, "n_repeats": n_repeats, "labels": {}}
    curves = []
    pd.set_option("display.width", 200)
    for label in cfg["labels"]:
        mask = masks[label]
        rows = preds.loc[mask].reset_index(drop=True)
        points, curve = curve_block(rows, preds, mask, cfg, eval_cfg, n_repeats)
        holdout = holdout_block(rows, cfg, eval_cfg, n_repeats)
        curves.append(curve.assign(label=label))
        result["labels"][label] = {"curve_points": points, "holdout": holdout}

        pts = pd.DataFrame(points)
        summary = pts.groupby("budget_per_1000").agg(
            detection=("detection", "mean"), det_sd=("detection", lambda x: x.std(ddof=1)),
            detected=("n_detected", lambda x: "/".join(map(str, x))), of=("n_event_vehicles", "first"),
            fa_realized=("fa_realized_per_1000", "mean"), lead_km=("median_lead_km", "mean"),
            null=("null_mean", "mean"), null_p95=("null_p95", "mean")).reset_index()
        summary["excess"] = summary["detection"] - summary["null"]
        print(f"\n== {'corregida (V)' if label == 'corrected' else 'dura (D)'} · curva (media de {n_repeats} repeticiones) ==")
        print(summary.round(3).to_string(index=False))

        ho = pd.DataFrame(holdout)
        hs = ho.groupby(["alpha", "method"]).agg(
            fa_heldout=("fa_heldout", "mean"), fa_max=("fa_heldout", "max"),
            n_fa=("n_fa_heldout", lambda x: "/".join(map(str, x))), of_healthy=("n_healthy", "first"),
            detection=("detection_heldout", "mean"), det_sd=("detection_heldout", lambda x: x.std(ddof=1)),
            detected=("n_detected", lambda x: "/".join(map(str, x))), lead_km=("median_lead_km", "mean"),
            fa_in_sample=("fa_in_sample", "mean"), fa_in_ci95_hi=("fa_in_sample_ci95_hi", "mean")).reset_index()
        print(f"\n-- ¿se sostiene fuera de muestra? (umbral con los sanos de los otros folds) --")
        print(hs.round(3).to_string(index=False))
        result["labels"][label]["holdout_summary"] = hs.to_dict(orient="records")
        result["labels"][label]["curve_summary"] = summary.to_dict(orient="records")

    out_dir = ensure_dir(experiments / "decision-k2")
    pd.concat(curves, ignore_index=True).to_csv(out_dir / "curve.csv", index=False)
    (out_dir / "decision_layer.json").write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
    print(f"\nescrito: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
