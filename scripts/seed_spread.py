#!/usr/bin/env python
"""Una semilla no es un resultado: la misma corrida con N semillas, media y rango.

    python scripts/seed_spread.py --config configs/exp_gru_seq.yaml
    python scripts/seed_spread.py --config configs/exp_cnn_lstm.yaml --seeds 5 --start 42

Corre exactamente lo que correría `train.py` —el mismo panel, el mismo recorte a dev,
los mismos folds, el mismo pipeline— y lo repite cambiando **solo**
`model.params.random_state`. Nada más se toca: es la dispersión que introduce la
inicialización de los pesos y el orden de los minibatches, no otro experimento.

Por qué hace falta acá: la CNN-LSTM movió ±0,01 de PR-AUC entre semillas
(docs/reproducibilidad.md) y la distancia entre los modelos que estamos
comparando es de ese orden. Reportar la mejor semilla de un modelo contra una semilla
sola de otro es elegir ruido. El número que va a la memoria es la media, con el rango
al lado.

No escribe nada y no loguea a wandb: imprime. Dev-only, igual que `train.py`.
Sirve para cualquier modelo del registry que acepte `random_state` (una red, un GBM);
para un modelo determinista las N corridas dan lo mismo, que también es una respuesta.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.config import load_config, resolve_path  # noqa: E402
from src.eval.metrics import classification_metrics  # noqa: E402
from src.eval.splits import load_test_split, test_split_masks  # noqa: E402
from src.training.cv import run_cv  # noqa: E402

METRICS = ("pr_auc", "pr_auc_lift", "roc_auc", "brier")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, help="El mismo YAML de experimento que usa train.py")
    parser.add_argument("--seeds", type=int, default=5, help="Cuántas semillas (>= 5 para reportar)")
    parser.add_argument("--start", type=int, default=None,
                        help="Primera semilla; por defecto la del YAML. Las siguientes son +1, +2, ...")
    parser.add_argument("--splits", default=None, help="Override del splits.json (p. ej. los de R=1)")
    return parser.parse_args()


def repeat_metrics(predictions: pd.DataFrame, n_repeats: int) -> list[dict[str, float]]:
    """Métricas de CADA repetición, como las calcula `train.py`.

    El número de la corrida es el promedio de las R repeticiones, no el PR-AUC de los
    scores promediados: eso último es un ensamble de R pasadas y da mejor de lo que el
    modelo es (ver el docstring de `run_cv`).
    """
    if n_repeats <= 1:
        return [classification_metrics(predictions["label"], predictions["score"])]
    return [classification_metrics(predictions["label"], predictions[f"score_r{r}"]) for r in range(n_repeats)]


def main() -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(message)s")
    warnings.filterwarnings("ignore")
    args = parse_args()

    cfg = load_config(args.config)
    panel = pd.read_parquet(resolve_path(cfg["data"]["panel"]))
    splits_path = resolve_path(args.splits or cfg["splits"]["path"])
    splits = json.loads(splits_path.read_text(encoding="utf-8"))

    # El mismo recorte a dev que hace `train.py`, con la misma clave obligatoria: sin
    # esto el nulo y la dispersión se medirían sobre el holdout (CLAUDE.md, regla 2).
    if "test_split" not in cfg.get("splits", {}):
        raise KeyError("El config no declara `splits.test_split` (ver scripts/train.py::select_dev)")
    if cfg["splits"]["test_split"]:
        dev_mask, _ = test_split_masks(panel, load_test_split(cfg["splits"]["test_split"]), strict=True)
        panel = panel.loc[dev_mask].reset_index(drop=True)

    model = cfg["model"]["name"]
    params = dict(cfg["model"].get("params", {}))
    base_seed = args.start if args.start is not None else int(params.get("random_state", cfg.get("seed", 42)))
    n_repeats = int(splits.get("n_repeats", 1))
    seeds = [base_seed + i for i in range(int(args.seeds))]

    print(f"\n== {cfg.get('name')} · modelo `{model}` ==")
    print(f"panel: {cfg['data']['panel']} · folds: {splits_path.name} (R={n_repeats}) · {len(panel)} filas dev")
    print(f"semillas: {seeds} (solo cambia model.params.random_state)\n")

    rows = []
    for seed in seeds:
        started = time.time()
        preds, _ = run_cv(panel, splits, model_name=model, model_params={**params, "random_state": seed})
        by_repeat = repeat_metrics(preds, n_repeats)
        row = {"seed": seed, **{k: float(np.mean([m[k] for m in by_repeat])) for k in METRICS}}
        rows.append(row)
        print(f"  seed {seed:>5} | PR-AUC {row['pr_auc']:.4f} | lift {row['pr_auc_lift']:.2f}x | "
              f"ROC {row['roc_auc']:.4f} | Brier {row['brier']:.4f} | {time.time() - started:.0f}s")

    table = pd.DataFrame(rows)
    print(f"\n  n = {len(table)} semillas")
    for metric in METRICS:
        column = table[metric]
        print(f"  {metric:<12} media {column.mean():.4f} ± {column.std(ddof=1):.4f} "
              f"· rango [{column.min():.4f}, {column.max():.4f}]")
    print("\n  Lo que va a la memoria es la media con el rango; la mejor semilla sola no es el modelo.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
