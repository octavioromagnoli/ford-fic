#!/usr/bin/env python
"""Re-mide una corrida ya entrenada desde su `predictions.parquet`, sin reentrenar.

    python scripts/rescore_run.py f3-cnn-lstm-r3-regen15 \
        --carry-from data/rebuild-0921/processed/panel_seq_v1.parquet

**Para qué.** Una corrida entrenada con un `scripts/train.py` anterior a una métrica
no la trae en su `metrics.json`: el aporte del *cuándo* por repetición, el techo de
cohorte, la detección por repetición o el eje vehículo aparecieron después que varias
corridas R=3. Reentrenar para eso es caro y además innecesario: todas esas métricas
salen de las predicciones out-of-fold, que ya están en disco. Esto las recalcula con
**la misma función que usa `train.py`** (`evaluate_predictions`), así que el número es
el que habría dado la corrida si se hubiera entrenado hoy.

**Qué verifica antes de escribir.** Lo que el `metrics.json` viejo ya traía (`oof`,
`oof_by_repeat`, `operating_point`) tiene que salir idéntico de las predicciones. Si no
sale, las predicciones no son las de ese `metrics.json` —o cambió una métrica que ya
estaba— y el script falla sin tocar nada.

`--carry-from` trae del panel las columnas de `src/training/cv.py::CARRY_COLUMNS` que
las predicciones no tengan (hoy, `aux_km_observed_after_cut`, que el C-index necesita y
que el panel viejo no emitía). El cruce es por `(vehicle_id, cut_odo)` y tiene que
coincidir en `label` y `event_observed` fila a fila, o falla.

Lo que describe el entrenamiento (`folds`, `folds_summary`, `stratify`) se conserva del
`metrics.json` viejo. El viejo queda como `metrics.pre-rescore.json` y el nuevo lleva un
bloque `rescored` que dice qué se agregó, con qué commit y cuándo. No loguea a wandb.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from scripts.train import evaluate_predictions, panel_build  # noqa: E402
from src.config import resolve_path  # noqa: E402
from src.training.cv import CARRY_COLUMNS  # noqa: E402

logger = logging.getLogger("rescore_run")

KEY = ["vehicle_id", "cut_odo"]
#: Lo que el metrics.json viejo ya traía y tiene que reproducirse exacto.
MUST_MATCH = ("oof", "oof_by_repeat", "operating_point")


def carry_columns(predictions: pd.DataFrame, panel_path: Path) -> tuple[pd.DataFrame, list[str]]:
    """Agrega las `CARRY_COLUMNS` que falten, desde el panel, verificando la clave."""
    missing = [c for c in CARRY_COLUMNS if c not in predictions.columns]
    if not missing:
        return predictions, []
    panel = pd.read_parquet(panel_path, columns=[*KEY, "label", "event_observed", *missing])
    merged = predictions.merge(
        panel, on=KEY, how="left", suffixes=("", "_panel"), validate="one_to_one", indicator=True
    )
    if (merged["_merge"] != "both").any():
        raise ValueError(
            f"{int((merged['_merge'] != 'both').sum())} fila(s) de las predicciones no están "
            f"en {panel_path.name}: no es el panel de esta corrida."
        )
    for column in ("label", "event_observed"):
        if not (merged[column] == merged[f"{column}_panel"]).all():
            raise ValueError(f"`{column}` no coincide con {panel_path.name}: no es el panel de esta corrida")
    out = merged.drop(columns=["_merge", "label_panel", "event_observed_panel"])
    return out[[*predictions.columns, *missing]], missing


def same(a: Any, b: Any, *, tol: float = 1e-12) -> bool:
    """Igualdad estructural con tolerancia numérica (JSON viejo contra dict nuevo)."""
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k], tol=tol) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(same(x, y, tol=tol) for x, y in zip(a, b))
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if math.isnan(float(a)) and math.isnan(float(b)):
            return True
        return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(a)))
    return a == b


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run", help="nombre de la corrida en experiments/")
    parser.add_argument("--dir", default="experiments")
    parser.add_argument("--carry-from", default=None,
                        help="panel del que traer las CARRY_COLUMNS que falten (mismas filas)")
    parser.add_argument("--panel", default=None,
                        help="panel que usó la corrida, para `panel_build` si el metrics.json no lo trae "
                             "(default: `data.panel` del config, resuelto con el FORD_DATA_DIR actual)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    run_dir = resolve_path(args.dir) / args.run
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    old = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    predictions = pd.read_parquet(run_dir / "predictions.parquet")
    carried: list[str] = []
    if args.carry_from:
        predictions, carried = carry_columns(predictions, resolve_path(args.carry_from))

    n_repeats = int(old.get("n_repeats", 1))
    evaluated, curve = evaluate_predictions(
        predictions, cfg, n_repeats=n_repeats, seed=int(cfg.get("seed", 42))
    )

    mismatched = [key for key in MUST_MATCH if key in old and not same(old[key], evaluated[key])]
    if mismatched:
        raise RuntimeError(
            f"Las predicciones de {args.run} no reproducen {mismatched} del metrics.json: o no "
            "son las de esa corrida o cambió una métrica que ya existía. No se escribe nada."
        )

    added = sorted(set(evaluated) - set(old))
    metrics = {**old, **evaluated}
    if not old.get("panel_build"):
        panel_path = resolve_path(args.panel or cfg["data"]["panel"])
        metrics["panel"] = str(panel_path)
        metrics["panel_build"] = panel_build(panel_path)
        added += ["panel", "panel_build"]
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                            text=True, cwd=resolve_path(".")).stdout.strip()
    metrics["rescored"] = {
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "by": "scripts/rescore_run.py",
        "commit": commit,
        "added": added,
        "carried_columns": carried,
        "reproduced": [key for key in MUST_MATCH if key in old],
    }

    backup = run_dir / "metrics.pre-rescore.json"
    if not backup.exists():
        backup.write_text(json.dumps(old, indent=2, default=float), encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float), encoding="utf-8")
    curve.to_csv(run_dir / "lead_time_curve.csv", index=False)

    spread = metrics["when_spread"]
    vehicle_mean = (metrics.get("vehicle") or {}).get("aggregations", {}).get("mean", {})
    logger.info("%s | reproduce %s | agrega %s", args.run, ", ".join(metrics["rescored"]["reproduced"]), added)
    logger.info("(a') aporte del cuándo: %+.4f ± %.4f | detección %.1f%% ± %.1f | anticipación %.0f ± %.0f km",
                spread["when_delta"]["mean"], spread["when_delta"]["std"],
                100 * spread["detection_rate"]["mean"], 100 * spread["detection_rate"]["std"],
                spread["median_lead_km"]["mean"], spread["median_lead_km"]["std"])
    if vehicle_mean:
        logger.info("lift por vehículo (mean): %.3f", vehicle_mean["pr_auc_lift"])
    if metrics.get("concordance"):
        logger.info("C-index: %.4f", metrics["concordance"]["c_index"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
