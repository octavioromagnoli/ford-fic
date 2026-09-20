#!/usr/bin/env python
"""¿El número del modelo es señal, o es lo que da este panel con una etiqueta al azar?

    python scripts/permutation_test.py --config configs/exp_logistic_pseudo.yaml
    python scripts/permutation_test.py --config configs/exp_logistic_l1.yaml --n-perm 50

Corre la CV completa —el mismo panel, los mismos folds, el mismo pipeline— con la
etiqueta **permutada a nivel vehículo**, y compara el resultado real contra esa
distribución. Es el complemento del piso `baserate`: el piso dice qué pasa sin modelo,
esto dice qué pasa **sin señal**, que no es lo mismo.

Por qué hace falta acá y no en cualquier proyecto: con 53 vehículos con evento el ruido
de estimación es grande, y además el panel tiene estructura propia (las filas positivas
son los últimos `H/Δ` cortes de su vehículo). Las dos cosas pueden levantar el nulo por
encima de 0,50, y entonces un ROC de 0,62 no significa lo que parece. Medido sobre dev:

    panel v1 canónico       nulo ROC 0,555 ± 0,059 · real 0,617 → p = 0,20
    panel del evento fict.  nulo ROC 0,501 ± 0,057 · real 0,703 → p < 0,001

Cómo se permuta, y por qué así: se sortea **qué vehículos** son positivos (no fila a
fila: las filas de un vehículo no son independientes y permutarlas destruiría la
estructura que justamente se quiere tener en el nulo) y se reproduce el patrón real
dentro del vehículo —los últimos cortes son los positivos—. Así el nulo tiene la misma
forma que los datos y lo único que se rompe es la relación entre features y etiqueta.

No escribe nada: imprime. Dev-only, igual que `train.py`.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
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

ID = "vehicle_id"


def relabel(frame: pd.DataFrame, positive_vehicles: set[str], n_tail: int) -> np.ndarray:
    """Etiqueta sintética con la forma de la real: los últimos `n_tail` cortes de cada
    vehículo sorteado como positivo."""
    ranked = (frame.sort_values([ID, "cut_odo"])
              .groupby(ID, sort=False).cumcount(ascending=False)
              .reindex(frame.index))
    is_tail = (ranked < n_tail).to_numpy()
    in_positive = frame[ID].isin(positive_vehicles).to_numpy()
    return (is_tail & in_positive).astype(int)


def main() -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(message)s")
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, help="El mismo YAML de experimento que usa train.py")
    parser.add_argument("--n-perm", type=int, default=30)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    cfg = load_config(args.config)
    panel = pd.read_parquet(resolve_path(cfg["data"]["panel"]))
    splits = json.loads(resolve_path(cfg["splits"]["path"]).read_text(encoding="utf-8"))
    test_split_path = cfg["splits"]["test_split"]
    if test_split_path:
        dev_mask, _ = test_split_masks(panel, load_test_split(test_split_path), strict=True)
        panel = panel.loc[dev_mask].reset_index(drop=True)

    model = cfg["model"]["name"]
    params = cfg["model"].get("params", {})
    features_cfg = cfg.get("features") or {}
    extra = tuple(features_cfg.get("extra_prefixes") or ())
    exclude = tuple(features_cfg.get("exclude_prefixes") or ())

    preds, _ = run_cv(panel, splits, model_name=model, model_params=params,
                      extra_prefixes=extra, exclude_prefixes=exclude)
    real = classification_metrics(preds["label"], preds["score"])

    by_vehicle = panel.groupby(ID)["label"].max()
    n_positive_vehicles = int(by_vehicle.sum())
    tail = int(panel[panel["label"].eq(1)].groupby(ID).size().median())
    vehicles = by_vehicle.index.to_numpy()

    rng = np.random.default_rng(args.seed)
    null_roc: list[float] = []
    null_norm: list[float] = []
    for _ in range(args.n_perm):
        fake = panel.copy()
        fake["label"] = relabel(fake, set(rng.choice(vehicles, size=n_positive_vehicles, replace=False)), tail)
        if fake["label"].sum() < 20:
            continue
        # `strict_splits=False`: los folds son los del panel real y la guarda de positivos
        # por fold no aplica a una etiqueta sorteada.
        p, _ = run_cv(fake, splits, model_name=model, model_params=params, strict_splits=False,
                      extra_prefixes=extra, exclude_prefixes=exclude)
        m = classification_metrics(p["label"], p["score"])
        null_roc.append(m["roc_auc"])
        null_norm.append(m["pr_auc_norm"])

    roc, norm = np.array(null_roc), np.array(null_norm)
    print(f"\n== {cfg.get('name')} · modelo `{model}` ==")
    print(f"panel: {cfg['data']['panel']} · {len(panel)} filas dev · "
          f"{n_positive_vehicles} vehículos positivos · cola de {tail} cortes")
    print(f"\n  real            ROC {real['roc_auc']:.3f} · PR-AUC norm {real['pr_auc_norm']:.3f}")
    print(f"  nulo (n={len(roc)})     ROC {roc.mean():.3f} ± {roc.std():.3f} · "
          f"p95 {np.percentile(roc, 95):.3f} · max {roc.max():.3f}")
    print(f"                  PR-AUC norm {norm.mean():.3f} ± {norm.std():.3f} · "
          f"p95 {np.percentile(norm, 95):.3f}")
    p_roc = float((roc >= real["roc_auc"]).mean())
    p_norm = float((norm >= real["pr_auc_norm"]).mean())
    print(f"\n  p-valor         ROC {p_roc:.4f} · PR-AUC norm {p_norm:.4f}")
    if roc.mean() > 0.55:
        print("\n  OJO: el nulo está por encima de 0,55. El panel tiene estructura que el modelo "
              "aprende sin señal — mirá el atajo de posición en docs/memoria/f3-posicion-en-la-serie.md")
    print("  (el nulo NO es 0,50 por definición: es lo que da ESTE panel con la etiqueta rota)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
