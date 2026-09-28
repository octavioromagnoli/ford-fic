"""Lectura del bundle de la demo (lo arma `scripts/build_demo_bundle.py`).

El bundle trae varios puntos de operación (el presupuesto de falsas alarmas que elige la perilla de la app): los cortes
y los números oficiales son uno solo, y las alertas, los hábitos y el triage, uno por punto. `load_bundle` devuelve un
punto a la vez, con la meta de ese punto (umbral, presupuesto, conteos), así el resto del código no sabe que hay otros.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import resolve_path

#: El único "por qué" de la demo: comparación descriptiva con los autos sanos del mercado (no es atribución).
FLEET_PROFILE = "fleet_profile"
REQUIRED = ("cuts.parquet", "vehicles.parquet", "deviations.parquet", "meta.json")
#: La columna de `vehicles.parquet` y `deviations.parquet` que dice a qué punto de operación pertenece cada fila.
BUDGET = "budget_per_1000"


def op_key(budget_per_1000: float) -> str:
    """El nombre del directorio de un punto de operación: `fa050` es 50 falsas alarmas cada 1.000 sanos (5%)."""
    return f"fa{int(round(float(budget_per_1000))):03d}"


@dataclass
class Bundle:
    cuts: pd.DataFrame        # un corte de dev por fila: score de la repetición, fecha, perfil de uso
    vehicles: pd.DataFrame    # un auto por fila, indexado por vehicle_id (del punto de operación elegido)
    deviations: pd.DataFrame  # auto × hábito: dónde se aparta de los sanos de su mercado (el "por qué")
    meta: dict[str, Any]      # la meta común más la del punto elegido (`threshold`, `budget_per_1000`, `counts`)
    root: Path

    @property
    def threshold(self) -> float:
        return float(self.meta["threshold"])

    @property
    def budget(self) -> float:
        """Falsas alarmas toleradas cada 1.000 autos sanos en el punto de operación elegido."""
        return float(self.meta[BUDGET])

    @property
    def budgets(self) -> list[float]:
        """Los puntos de operación que trae el bundle, de menor a mayor: las opciones de la perilla."""
        return [float(p[BUDGET]) for p in self.meta["operating_points"]]

    @property
    def triage_dir(self) -> Path:
        """El triage guardado de este punto de operación (la caché del LLM es una sola, en `root`)."""
        return self.root / "triage" / op_key(self.budget)

    @property
    def k(self) -> int:
        return int(self.meta["k_consecutive"])

    @property
    def model(self) -> dict[str, Any]:
        """Nombre, familia y corrida del modelo que puntúa la flota: lo que la app muestra sale de acá."""
        return self.meta["model"]

    def vehicle_cuts(self, vehicle_id: str) -> pd.DataFrame:
        return self.cuts.loc[self.cuts["vehicle_id"].eq(vehicle_id)].sort_values("cut_odo", kind="stable")


def bundle_dir(cfg: dict[str, Any]) -> Path:
    """`DEMO_BUNDLE_DIR` pisa el directorio del config (el contenedor baja el bundle a otro lado)."""
    env = os.environ.get("DEMO_BUNDLE_DIR")
    return Path(env).expanduser().resolve() if env else resolve_path(cfg["bundle"]["dir"])


def check_meta(meta: dict[str, Any]) -> None:
    """El bundle tiene que decir qué modelo lleva, que su "por qué" es la comparación con la flota y qué puntos de
    operación trae."""
    model, explanation = meta.get("model") or {}, (meta.get("explanation") or {}).get("type")
    if not model.get("name") or explanation != FLEET_PROFILE or not meta.get("operating_points"):
        raise ValueError(
            "El bundle no declara el modelo, su porqué no es la comparación con la flota sana "
            f"(`explanation.type` = {explanation!r}) o no trae los puntos de operación de la perilla: es de una "
            "versión anterior de la demo. Reconstruilo con `python scripts/build_demo_bundle.py --config configs/demo.yaml`."
        )


def operating_point(meta: dict[str, Any], budget_per_1000: float | None = None) -> dict[str, Any]:
    """El punto de operación pedido, o el que abre la demo (`default_budget_per_1000`) si no se pide ninguno."""
    wanted = float(meta["default_budget_per_1000"] if budget_per_1000 is None else budget_per_1000)
    for point in meta["operating_points"]:
        if abs(float(point[BUDGET]) - wanted) < 1e-9:
            return point
    offered = ", ".join(f"{float(p[BUDGET]):g}" for p in meta["operating_points"])
    raise ValueError(f"El bundle no trae el punto de {wanted:g} falsas alarmas cada 1.000 (trae {offered})")


def load_bundle(root: Path, budget_per_1000: float | None = None) -> Bundle:
    missing = [f for f in REQUIRED if not (root / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"El bundle de la demo está incompleto en {root} (faltan {missing}). Construilo con "
            "`python scripts/build_demo_bundle.py --config configs/demo.yaml` o bajalo con "
            "`python scripts/demo_app/fetch_bundle.py`."
        )
    meta = json.loads((root / "meta.json").read_text(encoding="utf-8"))
    check_meta(meta)
    point = operating_point(meta, budget_per_1000)
    budget = float(point[BUDGET])
    meta = {**meta, BUDGET: budget, "threshold": float(point["threshold"]),
            "counts": {**meta["counts"], **point["counts"]}}
    cuts = pd.read_parquet(root / "cuts.parquet")
    cuts["cut_date"] = pd.to_datetime(cuts["cut_date"])
    vehicles = pd.read_parquet(root / "vehicles.parquet")
    vehicles = vehicles.loc[vehicles[BUDGET].eq(budget)].drop(columns=BUDGET).reset_index(drop=True)
    for col in ("alert_first_date", "alert_confirm_date", "event_date"):
        vehicles[col] = pd.to_datetime(vehicles[col])
    vehicles["factors"] = vehicles["factors"].map(json.loads)
    vehicles["technician_signals"] = vehicles["technician_signals"].map(json.loads)
    deviations = pd.read_parquet(root / "deviations.parquet")
    deviations = deviations.loc[deviations[BUDGET].eq(budget)].drop(columns=BUDGET).reset_index(drop=True)
    return Bundle(cuts=cuts, vehicles=vehicles.set_index("vehicle_id", drop=False), deviations=deviations, meta=meta,
                  root=root)
