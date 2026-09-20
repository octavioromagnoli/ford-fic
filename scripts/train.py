#!/usr/bin/env python
"""Entrypoint único de entrenamiento. Un experimento = un YAML.

    python scripts/train.py --config configs/exp_dummy.yaml

Hace siempre lo mismo, en este orden: levanta el config, fija la semilla, carga
el panel, **lo recorta a dev con el holdout congelado**, carga los splits, corre
la CV agrupada por vehículo, calcula las métricas (clasificación + anticipación),
loguea todo a wandb y deja los outputs en `experiments/<run_name>/`.

El recorte a dev no es opcional ni implícito: el YAML tiene que declarar
`splits.test_split` (el path del holdout, o `null` explícito para un panel que no
tiene). Un config que se olvide de la clave falla acá y no entrena, porque el modo
de fallar de lo contrario es un número mejor de lo que corresponde y nadie
enterándose.

Nunca se edita este archivo para cambiar un hiperparámetro: se escribe otro YAML.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ensure_dir, load_config, repo_root, resolve_path, set_seed  # noqa: E402
from src.eval.metrics import (  # noqa: E402
    classification_metrics,
    dispersion,
    lead_time_curve,
    operating_point,
    summarize_folds,
)
from src.eval.splits import (  # noqa: E402
    MIN_VALID_POSITIVES,
    N_REPEATS,
    STRATIFY_COLUMN,
    STRATIFY_LEVEL,
    load_splits,
    load_test_split,
    make_splits,
    save_splits,
    split_options,
    splits_match_options,
    test_split_masks,
)
from src.training.cv import run_cv  # noqa: E402

logger = logging.getLogger("train")


def select_dev(panel: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Recorta el panel a dev con el holdout congelado. El test nunca llega a la CV.

    `splits.test_split` es obligatoria en el YAML: con el path del holdout recorta,
    con `null` explícito no recorta (el panel dummy no tiene holdout). Si la clave
    falta, esto revienta antes de entrenar. Es a propósito: el olvido no se nota en
    ninguna métrica —da PR-AUC más alto y listo—, así que el único momento en que se
    puede detectar es antes de correr.
    """
    splits_cfg = cfg.get("splits", {})
    if "test_split" not in splits_cfg:
        raise KeyError(
            "El config no declara `splits.test_split`. Poné el path del holdout congelado "
            "(data/processed/test_split.json) para entrenar solo con dev, o `null` explícito "
            "si el panel no tiene holdout (solo el dummy). Ver CLAUDE.md, regla 2."
        )
    if splits_cfg["test_split"] is None:
        logger.warning("`splits.test_split: null`: se entrena con el panel entero, sin holdout")
        return panel

    split = load_test_split(splits_cfg["test_split"])
    dev_mask, test_mask = test_split_masks(panel, split)
    dev = panel.loc[dev_mask].reset_index(drop=True)
    logger.info(
        "Holdout %s | dev: %d filas / %d vehículos | test reservado: %d filas / %d vehículos",
        Path(str(splits_cfg["test_split"])).name,
        len(dev),
        dev["vehicle_id"].nunique(),
        int(test_mask.sum()),
        panel.loc[test_mask, "vehicle_id"].nunique(),
    )
    if dev.empty:
        raise ValueError("El recorte a dev dejó el panel vacío: ¿el holdout es de otro dataset?")
    return dev


def load_or_make_splits(panel: pd.DataFrame, cfg: dict) -> dict:
    """Usa los splits congelados; los genera solo si el config lo autoriza.

    Regenerar splits en silencio es la forma más fácil de que dos personas
    comparen números que no son comparables. Por lo mismo, si el archivo congelado
    se armó con otra estratificación o con otro número de repeticiones que las que
    el YAML declara, esto falla en vez de entrenar: el archivo manda sobre el
    config —los folds son los del archivo—, así que un YAML que diga otra cosa está
    describiendo una corrida que no es la que va a pasar.
    """
    splits_cfg = cfg.get("splits", {})
    options = split_options(cfg)
    path = resolve_path(splits_cfg["path"])
    if path.exists():
        logger.info("Splits congelados: %s", path)
        splits = load_splits(path)
        mismatch = splits_match_options(splits, options)
        if mismatch:
            raise ValueError(
                f"El splits.json de {path.name} no coincide con lo que declara el YAML: "
                f"{mismatch}. Los folds salen del archivo, así que regeneralo con "
                "`python scripts/make_splits.py --config configs/data/panel_v1.yaml` "
                "(o alineá el YAML del experimento con el archivo)."
            )
        return splits
    if not splits_cfg.get("build_if_missing", False):
        raise FileNotFoundError(
            f"No existen los splits en {path} y `splits.build_if_missing` es false. "
            "Generalos una sola vez y compartilos como wandb Artifact."
        )
    logger.warning("Generando splits nuevos en %s (build_if_missing=true)", path)
    splits = make_splits(panel, **options)
    save_splits(splits, path)
    return splits


def repeat_metrics(predictions: pd.DataFrame, n_repeats: int) -> list[dict[str, float]]:
    """Métricas out-of-fold de CADA repetición, por separado.

    El número de selección de modelo es el promedio de estos, no el PR-AUC de los
    scores promediados: promediar scores entre repeticiones es un ensamble, y un
    ensamble de R pasadas da mejor que el modelo que se está evaluando.
    """
    if n_repeats <= 1:
        return [classification_metrics(predictions["label"], predictions["score"])]
    return [
        classification_metrics(predictions["label"], predictions[f"score_r{repeat}"])
        for repeat in range(n_repeats)
    ]


def init_wandb(cfg: dict, run_name: str):
    """wandb opcional: `mode: disabled` para iterar rápido, `offline` sin red.

    El entity y el modo salen del YAML, pero `WANDB_ENTITY` y `WANDB_MODE` los
    pisan: así se corre sin red (o contra una cuenta propia) sin tocar el config
    compartido, que es lo que hace comparables las corridas de los tres.
    """
    wandb_cfg = cfg.get("wandb", {})
    if not wandb_cfg.get("enabled", True):
        return None
    try:
        import wandb
    except ImportError:
        logger.warning("wandb no está instalado: la corrida no se loguea")
        return None
    entity = os.environ.get("WANDB_ENTITY") or wandb_cfg.get("entity")
    mode = os.environ.get("WANDB_MODE") or wandb_cfg.get("mode", "online")
    logger.info("wandb: %s/%s | mode=%s", entity or "<default>", wandb_cfg.get("project", "ford-fic"), mode)
    return wandb.init(
        project=wandb_cfg.get("project", "ford-fic"),
        entity=entity,
        name=run_name,
        group=wandb_cfg.get("group"),
        job_type=wandb_cfg.get("job_type", "train"),
        tags=wandb_cfg.get("tags"),
        mode=mode,
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

    # Antes que cualquier otra cosa: lo que sigue solo ve dev.
    panel = select_dev(panel, cfg)
    splits = load_or_make_splits(panel, cfg)

    features_cfg = cfg.get("features") or {}
    model_cfg = cfg["model"]
    run_name = args.run_name or cfg.get("name") or f"{model_cfg['name']}-{datetime.now():%Y%m%d-%H%M%S}"
    run = init_wandb(cfg, run_name)

    options = split_options(cfg)
    predictions, fold_metrics = run_cv(
        panel,
        splits,
        model_name=model_cfg["name"],
        model_params=model_cfg.get("params", {}),
        strict_splits=bool(cfg.get("splits", {}).get("strict", True)),
        min_valid_positives=options["min_valid_positives"],
        # Ablaciones desde el YAML (`features.extra_prefixes` / `features.exclude_prefixes`):
        # sacar o meter un grupo de columnas es otro config, no otra rama de código.
        extra_prefixes=tuple(features_cfg.get("extra_prefixes") or ()),
        exclude_prefixes=tuple(features_cfg.get("exclude_prefixes") or ()),
    )

    eval_cfg = cfg.get("eval", {})
    k_consecutive = int(eval_cfg.get("k_consecutive", 2))

    n_repeats = int(splits.get("n_repeats", 1))
    by_repeat = repeat_metrics(predictions, n_repeats)
    # `oof` es el promedio entre repeticiones. Con R=1 es exactamente la métrica de
    # siempre (una sola repetición, promedio de un elemento).
    # `n` y `n_positive` son los mismos en toda repetición (el conjunto out-of-fold es
    # siempre el panel de dev entero): promediarlos los volvería float sin motivo.
    oof = {
        key: value if key in ("n", "n_positive")
        else float(np.mean([m[key] for m in by_repeat]))
        for key, value in by_repeat[0].items()
    }
    repeats_spread = {
        key: dispersion([m[key] for m in by_repeat])
        for key in ("pr_auc", "roc_auc", "brier", "pr_auc_lift")
    }
    summary = summarize_folds(
        [{k: v for k, v in m.items() if k not in ("fold", "repeat")} for m in fold_metrics],
        seed=seed,
    )
    curve = lead_time_curve(
        predictions,
        n_thresholds=int(eval_cfg.get("n_thresholds", 50)),
        k_consecutive=k_consecutive,
    )
    budget = float(eval_cfg.get("max_false_alarms_per_1000", 50))
    point = operating_point(curve, max_false_alarms_per_1000=budget)

    metrics = {
        "oof": oof,
        "n_repeats": n_repeats,
        "oof_by_repeat": by_repeat,
        "repeats_spread": repeats_spread,
        "folds": fold_metrics,
        "folds_summary": summary,
        "operating_point": point,
        "operating_point_budget_per_1000": budget,
        "k_consecutive": k_consecutive,
        "stratify": splits.get("stratify"),
        "min_valid_positives": splits.get("min_valid_positives"),
    }

    logger.info("OOF | PR-AUC=%.4f (tasa base %.4f, norm %.4f) | ROC-AUC=%.4f | "
                "F1=%.4f (trivial %.4f, umbral %.3f) | Brier=%.4f",
                oof["pr_auc"], oof["base_rate"], oof.get("pr_auc_norm", float("nan")),
                oof["roc_auc"], oof.get("f1", float("nan")), oof.get("f1_trivial", float("nan")),
                oof.get("f1_threshold", float("nan")), oof["brier"])
    if n_repeats > 1:
        spread = repeats_spread["pr_auc"]
        logger.info(
            "CV repetida (%d pasadas) | PR-AUC por repetición: %.4f ± %.4f (min %.4f, max %.4f)",
            n_repeats, spread["mean"], spread["std"], spread["min"], spread["max"],
        )
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
        flat["cv/n_repeats"] = n_repeats
        flat.update({f"repeats/{k}_{stat}": v
                     for k, stats in repeats_spread.items()
                     for stat, v in stats.items()})
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
