#!/usr/bin/env python
"""La detección de un candidato contra un nulo que conserva el largo de cada historial (F6, §4–§5).

    python scripts/audit_detection_null.py --config configs/exp_ss_hw_r3.yaml

La alerta sostenida (`k_consecutive` cortes seguidos sobre el umbral) depende del largo del
historial: los fallados del panel tienen el doble de cortes que los sanos, así que un score al
azar ya detecta algo. CLAUDE.md (regla 6) pide comparar toda métrica así contra un nulo que
conserve esa magnitud. Este script hace lo del §4 del preregistro
`docs/reproducibilidad.md`:

* **Nulo.** Por repetición, el score **crudo** del modelo (antes de la post-transformación del
  candidato, si la tiene) se permuta entre todas las filas de dev, se le aplica la misma
  post-transformación (`smoothing:` del YAML) y se mide la detección con la etiqueta que decide
  (`window_eval.evaluable_column`). `detection_null.n_permutations` veces, con
  `detection_null.seed`. El **exceso** es la detección observada menos la media del nulo.
* **La referencia** se mide igual, con su propio nulo (sin post-transformación).
* **Informativo:** bootstrap pareado por vehículo (estratificado por fallado/sano) de la
  diferencia de detección candidato − referencia, sobre `score` (promedio de repeticiones).

La detección se calcula con una versión vectorizada de `lead_time_curve` + `operating_point`:
con `k` cortes seguidos, un vehículo alerta a un umbral τ si y solo si
`max_i min(s_i, …, s_{i+k−1}) ≥ τ`, y todo corte de un fallado es anterior a su evento. Antes de
usarla se verifica que reproduzca `operating_point` en las predicciones observadas.

Deja `experiments/<name>/detection_null.json`.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from scripts.ensemble_rank import align_members, n_repeats_of  # noqa: E402
from scripts.eval_window_label import attach_panel_columns  # noqa: E402
from scripts.smooth_scores import smooth  # noqa: E402
from scripts.train import select_dev  # noqa: E402
from src.config import load_config, resolve_path, set_seed  # noqa: E402
from src.eval.metrics import dispersion, lead_time_curve, operating_point  # noqa: E402

logger = logging.getLogger("audit_detection_null")

KEY = ["vehicle_id", "cut_odo"]


# --------------------------------------------------------------------------- #
# Detección vectorizada
# --------------------------------------------------------------------------- #
def vehicle_levels(frame: pd.DataFrame, score: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """Por vehículo: `max_i min(s_i..s_{i+k−1})` (−∞ con menos de k cortes), si falló, y sus scores.

    `frame` tiene que venir ordenado por `KEY`; `score` alineado con él.
    """
    ids = frame["vehicle_id"].to_numpy()
    starts = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1]])
    ends = np.r_[starts[1:], len(ids)]
    event = frame["event_observed"].to_numpy(dtype=int)[starts]
    levels = np.full(len(starts), -np.inf)
    pieces = []
    for j, (a, b) in enumerate(zip(starts, ends)):
        s = score[a:b]
        pieces.append(s)
        if b - a >= k:
            windows = np.lib.stride_tricks.sliding_window_view(s, k)
            levels[j] = windows.min(axis=1).max()
    return levels, event, pieces


def detection_from_levels(levels: np.ndarray, event: np.ndarray, row_scores: np.ndarray, *,
                          n_thresholds: int, budget_per_1000: float) -> tuple[float, int]:
    """La detección del punto de operación de `lead_time_curve` + `operating_point`."""
    grid = np.quantile(row_scores, np.linspace(0.0, 1.0, n_thresholds))
    thresholds = np.unique(np.append(grid, np.nextafter(row_scores.max(), np.inf)))
    failed, healthy = levels[event == 1], levels[event == 0]
    alerts_failed = (failed[None, :] >= thresholds[:, None]).sum(axis=1)
    alerts_healthy = (healthy[None, :] >= thresholds[:, None]).sum(axis=1)
    feasible = 1000.0 * alerts_healthy / max(len(healthy), 1) <= budget_per_1000
    if not feasible.any():
        return 0.0, 0
    best = int(alerts_failed[feasible].max())
    return best / max(len(failed), 1), best


def fast_detection(frame: pd.DataFrame, score: np.ndarray, eval_cfg: dict) -> tuple[float, int]:
    k = int(eval_cfg.get("k_consecutive", 2))
    levels, event, _ = vehicle_levels(frame, score, k)
    return detection_from_levels(levels, event, score, n_thresholds=int(eval_cfg.get("n_thresholds", 50)),
                                 budget_per_1000=float(eval_cfg.get("max_false_alarms_per_1000", 50)))


def check_fast_detection(frame: pd.DataFrame, column: str, eval_cfg: dict) -> None:
    """La versión vectorizada tiene que dar lo mismo que `operating_point` de la curva."""
    fast, _ = fast_detection(frame, frame[column].to_numpy(dtype=float), eval_cfg)
    curve = lead_time_curve(frame.assign(score=frame[column]), n_thresholds=int(eval_cfg.get("n_thresholds", 50)),
                            k_consecutive=int(eval_cfg.get("k_consecutive", 2)))
    point = operating_point(curve, max_false_alarms_per_1000=float(eval_cfg.get("max_false_alarms_per_1000", 50)))
    slow = float(point["detection_rate"]) if point else 0.0
    if abs(fast - slow) > 1e-12:
        raise AssertionError(f"La detección vectorizada ({fast}) no reproduce operating_point ({slow}) en `{column}`")


# --------------------------------------------------------------------------- #
# Nulo y bootstrap
# --------------------------------------------------------------------------- #
def permutation_null(raw: pd.DataFrame, mask: np.ndarray, method: str | None, eval_cfg: dict, *,
                     n_permutations: int, seed: int) -> dict[str, Any]:
    """Detección con el score crudo permutado entre filas (+ la post-transformación), por repetición."""
    n_repeats = n_repeats_of(raw)
    rng = np.random.default_rng(seed)
    rows = raw.loc[mask].reset_index(drop=True)
    by_repeat = []
    for r in range(n_repeats):
        values = []
        for _ in range(n_permutations):
            permuted = raw.copy()
            for rr in range(n_repeats):
                if rr == r:
                    permuted[f"score_r{rr}"] = rng.permutation(raw[f"score_r{rr}"].to_numpy(dtype=float))
            if method:
                permuted = smooth(permuted, method)
            score = permuted.loc[mask, f"score_r{r}"].to_numpy(dtype=float)
            values.append(fast_detection(rows, score, eval_cfg)[0])
        by_repeat.append(values)
    flat = np.concatenate([np.asarray(v) for v in by_repeat])
    return {"mean": float(flat.mean()), "std": float(flat.std(ddof=1)),
            "p95": float(np.quantile(flat, 0.95)),
            "by_repeat_mean": [float(np.mean(v)) for v in by_repeat],
            "n_permutations": n_permutations, "seed": seed, "transform": method}


def observed(frame: pd.DataFrame, mask: np.ndarray, eval_cfg: dict) -> list[float]:
    rows = frame.loc[mask].reset_index(drop=True)
    out = []
    for r in range(n_repeats_of(frame)):
        check_fast_detection(rows, f"score_r{r}", eval_cfg)
        out.append(fast_detection(rows, rows[f"score_r{r}"].to_numpy(dtype=float), eval_cfg)[0])
    return out


def paired_bootstrap(candidate: pd.DataFrame, reference: pd.DataFrame, mask: np.ndarray, eval_cfg: dict, *,
                     n_boot: int, seed: int) -> dict[str, Any]:
    """Diferencia de detección (sobre `score`) remuestreando vehículos, estratificado por fallado/sano."""
    k = int(eval_cfg.get("k_consecutive", 2))
    n_thresholds = int(eval_cfg.get("n_thresholds", 50))
    budget = float(eval_cfg.get("max_false_alarms_per_1000", 50))
    rows_c = candidate.loc[mask].reset_index(drop=True)
    rows_r = reference.loc[mask].reset_index(drop=True)
    lev_c, event, pieces_c = vehicle_levels(rows_c, rows_c["score"].to_numpy(dtype=float), k)
    lev_r, _, pieces_r = vehicle_levels(rows_r, rows_r["score"].to_numpy(dtype=float), k)
    failed, healthy = np.flatnonzero(event == 1), np.flatnonzero(event == 0)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        pick = np.concatenate([rng.choice(failed, len(failed)), rng.choice(healthy, len(healthy))])
        values = []
        for levels, pieces in ((lev_c, pieces_c), (lev_r, pieces_r)):
            scores = np.concatenate([pieces[i] for i in pick])
            values.append(detection_from_levels(levels[pick], event[pick], scores,
                                                n_thresholds=n_thresholds, budget_per_1000=budget)[0])
        diffs.append(values[0] - values[1])
    diffs = np.asarray(diffs)
    return {"mean": float(diffs.mean()), "ci90": [float(np.quantile(diffs, 0.05)), float(np.quantile(diffs, 0.95))],
            "p_diff_gt_0": float((diffs > 0).mean()), "n_boot": n_boot, "seed": seed, "score": "score"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))
    wcfg, ncfg, eval_cfg = cfg["window_eval"], cfg["detection_null"], cfg.get("eval", {})
    experiments = resolve_path(cfg.get("output_dir", "experiments"))
    name, reference_run = cfg["name"], wcfg["reference"]
    method = (cfg.get("smoothing") or {}).get("method")
    raw_run = (cfg.get("smoothing") or {}).get("member", {}).get("run", name)

    runs = [name, reference_run, raw_run]
    frames = align_members([pd.read_parquet(experiments / run / "predictions.parquet") for run in runs], runs)
    panel = select_dev(pd.read_parquet(resolve_path(cfg["data"]["panel"])), cfg)
    evaluable = wcfg["evaluable_column"]
    frames = [attach_panel_columns(f, panel, [evaluable]).sort_values(KEY, kind="stable").reset_index(drop=True)
              for f in frames]
    candidate, reference, raw = frames
    masks = {"corrected": candidate[evaluable].to_numpy(dtype=int) == 1,
             "hard": np.ones(len(candidate), dtype=bool)}
    decides = wcfg.get("decides", "corrected")
    n_perm, seed = int(ncfg.get("n_permutations", 200)), int(ncfg.get("seed", 0))

    out: dict[str, Any] = {"candidate": name, "reference": reference_run, "raw_member": raw_run,
                           "transform": method, "decides": decides, "labels": {}}
    for label, mask in masks.items():
        cand_obs, ref_obs = observed(candidate, mask, eval_cfg), observed(reference, mask, eval_cfg)
        cand_null = permutation_null(raw, mask, method, eval_cfg, n_permutations=n_perm, seed=seed)
        ref_null = permutation_null(reference, mask, None, eval_cfg, n_permutations=n_perm, seed=seed)
        cand_excess = float(np.mean(cand_obs) - cand_null["mean"])
        ref_excess = float(np.mean(ref_obs) - ref_null["mean"])
        out["labels"][label] = {
            "candidate": {"detection": cand_obs, "null": cand_null, "excess": cand_excess},
            "reference": {"detection": ref_obs, "null": ref_null, "excess": ref_excess},
            "excess_not_below_reference": bool(cand_excess >= ref_excess),
            "paired_bootstrap": paired_bootstrap(candidate, reference, mask, eval_cfg,
                                                 n_boot=int(ncfg.get("n_boot", 1000)), seed=seed),
        }

    path = experiments / name / "detection_null.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    for label in ("corrected", "hard"):
        block = out["labels"][label]
        print(f"\n== {'corregida (V)' if label == 'corrected' else 'dura (D)'}"
              f"{' · DECIDE' if label == decides else ''} ==")
        for who in ("candidate", "reference"):
            b = block[who]
            d = dispersion(b["detection"])
            print(f"  {who:<10} detección {100 * d['mean']:.1f}% ± {100 * d['std']:.1f} · nulo "
                  f"{100 * b['null']['mean']:.1f}% (p95 {100 * b['null']['p95']:.1f}%) · exceso {100 * b['excess']:+.1f} pts")
        bs = block["paired_bootstrap"]
        print(f"  exceso del candidato ≥ el de la referencia: {block['excess_not_below_reference']}")
        print(f"  bootstrap pareado (informativo): diferencia {100 * bs['mean']:+.1f} pts, IC90 "
              f"[{100 * bs['ci90'][0]:+.1f}; {100 * bs['ci90'][1]:+.1f}], P(dif > 0) = {bs['p_diff_gt_0']:.2f}")
    print(f"\nescrito: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
