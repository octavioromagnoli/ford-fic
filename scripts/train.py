#!/usr/bin/env python
"""Entrypoint único de entrenamiento. Un experimento = un YAML.

    python scripts/train.py --config configs/exp_dummy.yaml

Hace siempre lo mismo, en este orden: levanta el config, fija la semilla, carga
el panel y los splits congelados, corre la CV agrupada por vehículo, calcula las
métricas (clasificación + anticipación), loguea todo a wandb y deja los outputs
en `experiments/<run_name>/`.

Nunca se edita este archivo para cambiar un hiperparámetro: se escribe otro YAML.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ensure_dir, load_config, repo_root, resolve_path, set_seed  # noqa: E402
from src.eval.metrics import (  # noqa: E402
    classification_metrics,
    lead_time_curve,
    operating_point,
    summarize_folds,
)
from src.eval.splits import load_splits, make_splits, save_splits  # noqa: E402
from src.training.cv import run_cv  # noqa: E402

logger = logging.getLogger("train")


def load_or_make_splits(panel: pd.DataFrame, cfg: dict) -> dict:
    """Usa los splits congelados; los genera solo si el config lo autoriza.

    Regenerar splits en silencio es la forma más fácil de que dos personas
    comparen números que no son comparables.
    """
    splits_cfg = cfg.get("splits", {})
    path = resolve_path(splits_cfg["path"])
    if path.exists():
        logger.info("Splits congelados: %s", path)
        return load_splits(path)
    if not splits_cfg.get("build_if_missing", False):
        raise FileNotFoundError(
            f"No existen los splits en {path} y `splits.build_if_missing` es false. "
            "Generalos una sola vez y compartilos como wandb Artifact."
        )
    logger.warning("Generando splits nuevos en %s (build_if_missing=true)", path)
    splits = make_splits(
        panel, n_splits=int(splits_cfg.get("n_splits", 5)), seed=int(splits_cfg.get("seed", 42))
    )
    save_splits(splits, path)
    return splits


def init_wandb(cfg: dict, run_name: str):
    """wandb opcional: `mode: disabled` para iterar rápido, `offline` sin red."""
    wandb_cfg = cfg.get("wandb", {})
    if not wandb_cfg.get("enabled", True):
        return None
    try:
        import wandb
    except ImportError:
        logger.warning("wandb no está instalado: la corrida no se loguea")
        return None
    return wandb.init(
        project=wandb_cfg.get("project", "ford-fic"),
        entity=wandb_cfg.get("entity"),
        name=run_name,
        group=wandb_cfg.get("group"),
        job_type=wandb_cfg.get("job_type", "train"),
        tags=wandb_cfg.get("tags"),
        mode=wandb_cfg.get("mode", "offline"),
        config=cfg,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Corre un experimento definido por un YAML")
    parser.add_argument("--config", required=True)
    parser.add_argument("--panel", default=None, help="Override del panel (ej.: pasar de dummy a real)")
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    cfg = load_config(args.config)
    seed = set_seed(int(cfg.get("seed", 42)))

    panel_path = resolve_path(args.panel or cfg["data"]["panel"])
    if not panel_path.exists():
        raise FileNotFoundError(
            f"No existe el panel {panel_path}. Si es el dummy: "
            "`python scripts/make_dummy.py --config configs/data/dummy_v1.yaml`"
        )
    panel = pd.read_parquet(panel_path)
    logger.info("Panel: %s | %d filas | %d vehículos", panel_path.name, len(panel), panel["vehicle_id"].nunique())

    splits = load_or_make_splits(panel, cfg)

    model_cfg = cfg["model"]
    run_name = args.run_name or cfg.get("name") or f"{model_cfg['name']}-{datetime.now():%Y%m%d-%H%M%S}"
    run = init_wandb(cfg, run_name)

    predictions, fold_metrics = run_cv(
        panel,
        splits,
        model_name=model_cfg["name"],
        model_params=model_cfg.get("params", {}),
        strict_splits=bool(cfg.get("splits", {}).get("strict", True)),
    )

    eval_cfg = cfg.get("eval", {})
    k_consecutive = int(eval_cfg.get("k_consecutive", 2))

    oof = classification_metrics(predictions["label"], predictions["score"])
    summary = summarize_folds([{k: v for k, v in m.items() if k != "fold"} for m in fold_metrics], seed=seed)
    curve = lead_time_curve(
        predictions,
        n_thresholds=int(eval_cfg.get("n_thresholds", 50)),
        k_consecutive=k_consecutive,
    )
    budget = float(eval_cfg.get("max_false_alarms_per_1000", 50))
    point = operating_point(curve, max_false_alarms_per_1000=budget)

    metrics = {
        "oof": oof,
        "folds": fold_metrics,
        "folds_summary": summary,
        "operating_point": point,
        "operating_point_budget_per_1000": budget,
        "k_consecutive": k_consecutive,
    }

    logger.info("OOF | PR-AUC=%.4f (tasa base %.4f) | ROC-AUC=%.4f | Brier=%.4f",
                oof["pr_auc"], oof["base_rate"], oof["roc_auc"], oof["brier"])
    if point:
        logger.info(
            "Punto de operación (<= %.0f falsas alarmas/1000 sanos): detección %.1f%% | "
            "anticipación mediana %.0f km | umbral %.4f",
            budget, 100 * point["detection_rate"], point["median_lead_km"], point["threshold"],
        )
    else:
        logger.info("Ningún umbral respeta el presupuesto de %.0f falsas alarmas/1000", budget)

    out_dir = ensure_dir(Path(resolve_path(cfg.get("output_dir", "experiments"))) / run_name)
    predictions.to_parquet(out_dir / "predictions.parquet", index=False)
    curve.to_csv(out_dir / "lead_time_curve.csv", index=False)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float), encoding="utf-8")
    (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    logger.info("Outputs: %s", out_dir.relative_to(repo_root()))

    if run is not None:
        import wandb

        flat = {f"oof/{k}": v for k, v in oof.items()}
        flat.update({f"cv/{k}": v for k, v in summary.items()})
        if point:
            flat.update(
                {
                    "pitch/detection_rate": point["detection_rate"],
                    "pitch/median_lead_km": point["median_lead_km"],
                    "pitch/false_alarms_per_1000": point["false_alarms_per_1000"],
                    "pitch/threshold": point["threshold"],
                }
            )
        run.log(flat)
        run.log({"lead_time_curve": wandb.Table(dataframe=curve)})
        run.log({"fold_metrics": wandb.Table(dataframe=pd.DataFrame(fold_metrics))})
        if cfg.get("wandb", {}).get("log_artifacts", False):
            artifact = wandb.Artifact(f"{run_name}-predictions", type="predictions")
            artifact.add_file(str(out_dir / "predictions.parquet"))
            artifact.add_file(str(out_dir / "lead_time_curve.csv"))
            run.log_artifact(artifact)
        run.finish()


if __name__ == "__main__":
    main()
