"""Carga de configs YAML, resolución de rutas y semillas.

Regla del repo: ningún path ni hiperparámetro se hardcodea en el código; todo
entra por un YAML de `configs/` o por variable de entorno.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def repo_root() -> Path:
    """Raíz del repo (el directorio que contiene `src/`)."""
    return Path(__file__).resolve().parents[1]


def data_dir() -> Path:
    """Directorio de datos. Override por `FORD_DATA_DIR` para correr fuera del repo."""
    env = os.environ.get("FORD_DATA_DIR")
    return Path(env).expanduser().resolve() if env else repo_root() / "data"


def resolve_path(path: str | Path) -> Path:
    """Resuelve un path del config: absoluto tal cual, relativo contra la raíz del repo.

    Soporta los placeholders `${DATA_DIR}` y `${REPO_ROOT}`.
    """
    text = str(path)
    text = text.replace("${DATA_DIR}", str(data_dir()))
    text = text.replace("${REPO_ROOT}", str(repo_root()))
    text = os.path.expandvars(text)
    resolved = Path(text).expanduser()
    return resolved if resolved.is_absolute() else (repo_root() / resolved).resolve()


def load_config(path: str | Path) -> dict[str, Any]:
    """Lee un YAML de config y le agrega `_config_path` para trazabilidad en wandb."""
    cfg_path = resolve_path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"No existe el config: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"El config {cfg_path} no es un mapping YAML")
    cfg["_config_path"] = str(cfg_path.relative_to(repo_root()))
    return cfg


def set_seed(seed: int) -> int:
    """Fija las semillas de python, numpy y `PYTHONHASHSEED`. Devuelve la semilla usada."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    return seed


def ensure_dir(path: str | Path) -> Path:
    """Crea el directorio (y sus padres) si no existe, y lo devuelve resuelto."""
    target = resolve_path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target
