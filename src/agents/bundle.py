"""Lectura del bundle de la demo (lo arma `scripts/build_demo_bundle.py`)."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import resolve_path

REQUIRED = ("cuts.parquet", "vehicles.parquet", "waterfall.parquet", "meta.json")


@dataclass
class Bundle:
    cuts: pd.DataFrame        # un corte de dev por fila: score de la repetición, fecha, perfil de uso
    vehicles: pd.DataFrame    # un auto por fila, indexado por vehicle_id
    waterfall: pd.DataFrame   # escalones del "por qué" por auto
    meta: dict[str, Any]
    root: Path

    @property
    def threshold(self) -> float:
        return float(self.meta["threshold"])

    @property
    def k(self) -> int:
        return int(self.meta["k_consecutive"])

    def vehicle_cuts(self, vehicle_id: str) -> pd.DataFrame:
        return self.cuts.loc[self.cuts["vehicle_id"].eq(vehicle_id)].sort_values("cut_odo", kind="stable")


def bundle_dir(cfg: dict[str, Any]) -> Path:
    """`DEMO_BUNDLE_DIR` pisa el directorio del config (el contenedor baja el bundle a otro lado)."""
    env = os.environ.get("DEMO_BUNDLE_DIR")
    return Path(env).expanduser().resolve() if env else resolve_path(cfg["bundle"]["dir"])


def load_bundle(root: Path) -> Bundle:
    missing = [f for f in REQUIRED if not (root / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"El bundle de la demo está incompleto en {root} (faltan {missing}). Construilo con "
            "`python scripts/build_demo_bundle.py --config configs/demo.yaml` o bajalo con "
            "`python scripts/demo_app/fetch_bundle.py`."
        )
    cuts = pd.read_parquet(root / "cuts.parquet")
    cuts["cut_date"] = pd.to_datetime(cuts["cut_date"])
    vehicles = pd.read_parquet(root / "vehicles.parquet")
    for col in ("alert_first_date", "alert_confirm_date", "event_date"):
        vehicles[col] = pd.to_datetime(vehicles[col])
    vehicles["factors"] = vehicles["factors"].map(json.loads)
    vehicles["technician_signals"] = vehicles["technician_signals"].map(json.loads)
    meta = json.loads((root / "meta.json").read_text(encoding="utf-8"))
    return Bundle(cuts=cuts, vehicles=vehicles.set_index("vehicle_id", drop=False), waterfall=pd.read_parquet(root / "waterfall.parquet"),
                  meta=meta, root=root)
