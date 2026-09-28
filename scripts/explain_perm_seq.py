#!/usr/bin/env python
"""En qué se apoya un modelo secuencial para detectar autos: importancia por permutación de cada canal.

    WANDB_MODE=disabled python scripts/explain_perm_seq.py --config configs/explain_f11.yaml

Para cada experimento de la lista reentrena una repetición de su CV (mismo panel, folds, modelo y
target que `train.py`) y, en la validación de cada fold, permuta **una unidad por vez** entre las
filas: un canal de la secuencia con sus `T` bins juntos, o una estática con su one-hot. La
permutación es la misma para todas las columnas de la unidad, así la ventana de un canal queda
entera, pegada a otro corte.

Por unidad reporta dos cosas, promediadas sobre `n_permutations`:
* **caída de la detección** (puntos): la detección media por auto a `budgets` (umbral exacto,
  `k` cortes seguidos, la cuenta de `report_v2_models.py`) sin la unidad, contra la de referencia;
* **|Δlogit| medio**: cuánto se mueve el score de un corte cuando la unidad se permuta, y qué parte
  del total se lleva cada familia (`families` del YAML).

Es explicabilidad global del modelo, no del auto: para el porqué de una alerta está
`src/eval/explain_seq.py` (Shapley por permutación por canal). Deja
`experiments/<output_name>/`: `units.csv`, `families.csv` y `scores_<run>.parquet`.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from scripts.audit_detection_null import vehicle_levels  # noqa: E402
from scripts.report_v2_models import detection_at  # noqa: E402
from scripts.train import select_dev  # noqa: E402
from src.config import ensure_dir, load_config, resolve_path  # noqa: E402
from src.eval.splits import iter_repeats, load_splits  # noqa: E402
from src.models.registry import get_model  # noqa: E402
from src.training.cv import build_preprocessor, select_feature_columns  # noqa: E402
from src.training.targets import build_target  # noqa: E402

logger = logging.getLogger("explain_perm_seq")
KEY = ["vehicle_id", "cut_odo"]


def unit_columns(features: list[str], unit: str) -> list[str]:
    """Las columnas crudas de una unidad: `feat_seq_t{t}_c{k}_<canal>` o la estática por nombre."""
    return [c for c in features if c == unit or (c.startswith("feat_seq_") and c.split("_", 4)[4] == unit)]


def mean_detection(frame: pd.DataFrame, score: np.ndarray, k: int, budgets: list[float]) -> float:
    levels, event, _ = vehicle_levels(frame, score, k)
    return float(np.mean([detection_at(levels, event, b)[0] for b in budgets]))


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def explain_run(exp_path: str, cfg: dict, rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    exp = load_config(exp_path)
    panel = pd.read_parquet(resolve_path(exp["data"]["panel"]))
    panel = select_dev(panel, exp).sort_values(KEY).reset_index(drop=True)
    splits = load_splits(resolve_path(exp["splits"]["path"]))
    features = select_feature_columns(panel)
    X = panel[features]
    y = panel["label"].astype(int).to_numpy()
    target = exp.get("target") or {}
    statics = [c for c in features if c.startswith("static_")]
    channels = list(dict.fromkeys(c.split("_", 4)[4] for c in features if c.startswith("feat_seq_")))
    units = channels + statics
    n_perm = int(cfg["n_permutations"])

    base = np.full(len(panel), np.nan)
    permuted = {u: np.full((n_perm, len(panel)), np.nan) for u in units}
    for repeat, masks in iter_repeats(panel, splits, strict=True, min_valid_positives=exp["splits"].get("min_valid_positives")):
        if repeat != int(cfg["repeat"]):
            continue
        for fold, train_mask, valid_mask in masks:
            model = get_model(exp["model"]["name"], dict(exp["model"].get("params") or {}))
            pipe = Pipeline([("prep", build_preprocessor(X)), ("model", model)])
            fit_y = (build_target(target["name"], panel, train_mask, **(target.get("params") or {})).y
                     if target else y[train_mask])
            pipe.fit(X.loc[train_mask], fit_y)
            valid = X.loc[valid_mask].reset_index(drop=True)
            base[valid_mask] = pipe.predict_proba(valid)[:, 1]
            for unit in units:
                cols = unit_columns(features, unit)
                for i in range(n_perm):
                    shuffled = valid.copy()
                    shuffled[cols] = valid[cols].to_numpy()[rng.permutation(len(valid))]
                    permuted[unit][i, valid_mask] = pipe.predict_proba(shuffled)[:, 1]
            logger.info("%s | fold %d listo", exp["name"], fold)

    k, budgets = int(cfg["k_consecutive"]), [float(b) for b in cfg["budgets"]]
    ref = mean_detection(panel, base, k, budgets)
    families = cfg["families"]
    rows = []
    for unit in units:
        drop = np.mean([ref - mean_detection(panel, permuted[unit][i], k, budgets) for i in range(n_perm)])
        dlogit = np.mean([np.abs(logit(permuted[unit][i]) - logit(base)).mean() for i in range(n_perm)])
        family = next((f for f, members in families.items() if unit in members), "otra")
        rows.append({"run": exp["name"], "unit": unit, "family": family, "detection": 100 * ref,
                     "detection_drop_pts": 100 * drop, "mean_abs_dlogit": dlogit})
    table = pd.DataFrame(rows)
    table["share_dlogit"] = table["mean_abs_dlogit"] / table["mean_abs_dlogit"].sum()
    scores = panel[KEY + ["event_observed"]].assign(base=base)
    return table, scores


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cfg = load_config(args.config)
    out_dir = ensure_dir(resolve_path(cfg.get("experiments_dir", "experiments")) / cfg["output_name"])
    rng = np.random.default_rng(int(cfg["seed"]))
    tables = []
    for exp_path in cfg["experiments"]:
        table, scores = explain_run(exp_path, cfg, rng)
        scores.to_parquet(out_dir / f"scores_{table['run'].iloc[0]}.parquet", index=False)
        tables.append(table)
    units = pd.concat(tables, ignore_index=True)
    fam = (units.groupby(["run", "family"])[["detection_drop_pts", "share_dlogit"]].sum().reset_index())
    units.to_csv(out_dir / "units.csv", index=False)
    fam.to_csv(out_dir / "families.csv", index=False)
    print(units.round(3).to_string(index=False))
    print()
    print(fam.round(3).to_string(index=False))
    print(f"\n→ {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
