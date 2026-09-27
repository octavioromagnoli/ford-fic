#!/usr/bin/env python
"""Genera los YAML de la re-medición completa sobre v2 (`configs/exp_v2all_*.yaml`).

    python scripts/make_v2all_configs.py

Todas las corridas parten de `configs/exp_v2_lgbm_r3.yaml` (mismos folds `splits_r3.json`, mismo
holdout, misma evaluación) y cambian solo panel, target y modelo. La lista vive en
`configs/v2all_runs.yaml`: agregar una corrida es una entrada ahí, no una línea acá. Una entrada con
`seeds: [...]` genera además la misma corrida con cada semilla del modelo (`<name>-s<semilla>-r3`).
"""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def config_path(name: str) -> Path:
    return ROOT / "configs" / f"exp_{name.replace('-', '_')}.yaml"


def main() -> int:
    spec = yaml.safe_load((ROOT / "configs/v2all_runs.yaml").read_text(encoding="utf-8"))
    base = yaml.safe_load((ROOT / spec["base"]).read_text(encoding="utf-8"))
    runs = []
    for run in spec["runs"]:
        runs.append(run)
        # `seeds: [1, 2]` agrega la misma corrida con otra semilla del modelo (`<name>-s<semilla>`)
        for seed in run.get("seeds", []):
            extra = copy.deepcopy(run)
            extra.pop("seeds")
            extra["name"] = run["name"].replace("-r3", f"-s{seed}-r3")
            extra["what"] = f"{run['what']}, semilla del modelo {seed} (dispersión por semilla)"
            extra["model"]["params"]["random_state"] = seed
            runs.append(extra)
    for run in runs:
        cfg = copy.deepcopy(base)
        cfg["name"] = run["name"]
        cfg["data"]["panel"] = run["panel"]
        cfg.pop("target", None)
        if run.get("target"):
            cfg["target"] = run["target"]
        cfg["model"] = run["model"]
        cfg["eval"] = {**cfg["eval"], **run.get("eval", {})}
        cfg["wandb"]["group"] = "f9-v2all"
        cfg["wandb"]["tags"] = ["f9", "v2", "v2all", "r3"] + run.get("tags", [])
        out = config_path(run["name"])
        header = (
            f"# Re-medición completa sobre la ENTREGA v2 (26-09-2026): {run['what']}.\n"
            "# Generado por scripts/make_v2all_configs.py desde configs/v2all_runs.yaml (no editar a mano).\n"
            f"#   WANDB_MODE=disabled python scripts/train.py --config configs/{out.name}\n"
        )
        out.write_text(header + yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
        print(out.relative_to(ROOT).as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
