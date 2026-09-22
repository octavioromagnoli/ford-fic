#!/usr/bin/env python
"""Publica el panel, sus splits y el holdout como wandb Artifacts (entregable de F2).

    python scripts/log_panel_artifact.py --config configs/data/panel_v1.yaml

Plan §4: "panel.parquet v1 + splits.json como wandb Artifacts, para que B y C se los
bajen en vez de reconstruirlos". Sube dos artefactos al proyecto compartido
(`wandb.project` / `wandb.entity` del YAML del panel; `WANDB_ENTITY` y `WANDB_MODE`
los pisan, igual que en `scripts/train.py`):

- `<wandb.artifact>` (tipo `dataset`): `panel.parquet`, `splits.json`,
  `panel_meta.json` y los dos YAML que lo definen (panel y features), con la terna,
  los conteos y la huella de vehículos como metadata.
- `test-split` (tipo `holdout`): `test_split.json`, para que `select_dev()` recorte
  igual en todas las máquinas.

El mismo script publica el panel de hitos del cure model (`panel-landmark-ps`, con sus
folds congelados y el YAML del reloj del evento):

    python scripts/log_panel_artifact.py --config configs/data/panel_landmark_ps.yaml

Para consumirlos desde otro lado:

    import wandb
    run = wandb.init(project="ford-fic", entity="oromagnoli-", job_type="download")
    path = run.use_artifact("panel-v1:latest").download()   # deja los archivos en `path`
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.config import load_config, resolve_path  # noqa: E402

logger = logging.getLogger("log_panel_artifact")


def describe(meta: dict) -> tuple[str, dict]:
    """`(descripción, metadata)` del Artifact: panel v1 (tiene `label`) o panel de hitos."""
    if "label" in meta:
        description = (
            f"Panel real F2 · W={meta['label']['window_km']:.0f} G={meta['label']['gap_km']:.0f} "
            f"H={meta['label']['horizon_km']:.0f} Δ={meta['label']['cut_step_km']:.0f} km · "
            f"{meta['panel']['rows']} filas antes del emparejado · dev {meta['dev']['rows']} filas / "
            f"{meta['dev']['vehicles']} vehículos · test {meta['test']['rows']} / {meta['test']['vehicles']}"
        )
        metadata = {
            "label": meta["label"], "anchor": meta["anchor"], "events": meta["events"],
            "dev": meta["dev"], "test": meta["test"], "sampling": meta["sampling"],
            "n_feat": len(meta["columns"]["feat"]), "n_static": len(meta["columns"]["static"]),
            "n_aux": len(meta["columns"]["aux"]), "splits": meta["splits"], "created_at": meta["created_at"],
        }
        return description, metadata
    landmark = meta["landmark"]
    description = (
        f"Panel de hitos post-venta (cure model) · hitos {landmark['landmarks_days']} d · "
        f"G={landmark['gap_days']} d · H={landmark['horizon_days']} d · dev {meta['dev']['rows']} filas / "
        f"{meta['dev']['vehicles']} vehículos · test {meta['test']['rows']} / {meta['test']['vehicles']}"
    )
    metadata = {
        "landmark": landmark, "anchor": meta["anchor"], "dev": meta["dev"], "test": meta["test"],
        "fleet_features": meta["fleet_features"], "n_feat_fm": meta["columns"]["feat_fm"],
        "n_aux": len(meta["columns"]["aux"]), "created_at": meta["created_at"],
    }
    return description, metadata


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/data/panel_v1.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    wandb_cfg = cfg.get("wandb") or {}
    if not wandb_cfg.get("artifact"):
        raise KeyError("El YAML del panel no declara `wandb.artifact` (el nombre del Artifact a publicar)")

    panel_path = resolve_path(cfg["output"]["panel"])
    splits_path = resolve_path(cfg["output"]["splits"])
    meta_path = resolve_path(cfg["output"]["meta"])
    holdout_path = resolve_path(cfg["test_split"])
    features_path = resolve_path((cfg.get("features") or {}).get("spec") or cfg["features_spec"])
    for path in (panel_path, splits_path, meta_path, holdout_path):
        if not path.exists():
            raise FileNotFoundError(f"Falta {path}. Construí el panel primero: "
                                    f"python scripts/build_dataset.py --config {args.config}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    import wandb

    entity = os.environ.get("WANDB_ENTITY") or wandb_cfg.get("entity")
    mode = os.environ.get("WANDB_MODE") or wandb_cfg.get("mode", "online")
    run = wandb.init(
        project=wandb_cfg.get("project", "ford-fic"),
        entity=entity,
        name=f"log-{wandb_cfg['artifact']}",
        group=wandb_cfg.get("group", "f2-panel"),
        job_type="dataset",
        tags=list(wandb_cfg.get("tags", ["f2", "panel"])),
        mode=mode,
        config={"panel_config": cfg.get("_config_path"), "label": meta.get("label") or meta.get("landmark"),
                "splits": meta.get("splits") or cfg.get("splits")},
    )
    logger.info("wandb: %s/%s | mode=%s", entity or "<default>", wandb_cfg.get("project", "ford-fic"), mode)

    description, metadata = describe(meta)
    panel_artifact = wandb.Artifact(wandb_cfg["artifact"], type="dataset", description=description,
                                    metadata=metadata)
    panel_artifact.add_file(str(panel_path), name="panel.parquet")
    panel_artifact.add_file(str(splits_path), name="splits.json")
    panel_artifact.add_file(str(meta_path), name="panel_meta.json")
    panel_artifact.add_file(str(resolve_path(args.config)), name="configs/panel.yaml")
    panel_artifact.add_file(str(features_path), name="configs/features.yaml")
    if cfg.get("event_clock"):
        panel_artifact.add_file(str(resolve_path(cfg["event_clock"])), name="configs/event_clock.yaml")
    run.log_artifact(panel_artifact)

    holdout = json.loads(holdout_path.read_text(encoding="utf-8"))
    holdout_artifact = wandb.Artifact(
        wandb_cfg.get("holdout_artifact", "test-split"),
        type="holdout",
        description=(
            f"Holdout dev/test congelado · {len(holdout['dev_vehicles'])} dev / "
            f"{len(holdout['test_vehicles'])} test / {len(holdout.get('excluded_vehicles', []))} excluidos · "
            f"semilla {holdout.get('seed')} · {holdout.get('created_at')}"
        ),
        metadata={k: holdout.get(k) for k in ("kind", "seed", "created_at", "n_vehicles", "n_event_vehicles",
                                              "dev", "test", "universe", "panel")},
    )
    holdout_artifact.add_file(str(holdout_path), name="test_split.json")
    run.log_artifact(holdout_artifact)
    run.finish()
    print(f"Publicados `{wandb_cfg['artifact']}` (dataset) y `{wandb_cfg.get('holdout_artifact', 'test-split')}` "
          f"(holdout) en {entity or '<default>'}/{wandb_cfg.get('project', 'ford-fic')} · mode={mode}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
