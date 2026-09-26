#!/usr/bin/env python
"""Baja el bundle de la demo (wandb Artifact) al arrancar el contenedor.

    WANDB_API_KEY=... python scripts/demo_app/fetch_bundle.py

La versión sale de `bundle.version` en configs/demo.yaml (o `DEMO_BUNDLE_VERSION`), y el destino de
`DEMO_BUNDLE_DIR`. Si el bundle ya está (reinicio del contenedor sin volumen nuevo), no baja nada.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.agents.bundle import REQUIRED, bundle_dir  # noqa: E402
from src.config import load_config  # noqa: E402


def main() -> int:
    cfg = load_config(os.environ.get("DEMO_CONFIG", "configs/demo.yaml"))
    root = bundle_dir(cfg)
    if all((root / f).exists() for f in REQUIRED) and not os.environ.get("DEMO_BUNDLE_FORCE"):
        print(f"bundle presente en {root}: no se baja")
        return 0
    if not os.environ.get("WANDB_API_KEY"):
        print("Falta WANDB_API_KEY: no se puede bajar el bundle de la demo.", file=sys.stderr)
        return 1
    import wandb

    b = cfg["bundle"]
    version = os.environ.get("DEMO_BUNDLE_VERSION", b["version"])
    ref = f"{b['entity']}/{b['project']}/{b['artifact']}:{version}"
    artifact = wandb.Api().artifact(ref, type="dataset")
    artifact.download(root=str(root))
    missing = [f for f in REQUIRED if not (root / f).exists()]
    if missing:
        print(f"El artifact {ref} no trae {missing}", file=sys.stderr)
        return 1
    print(f"bundle {ref} → {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
