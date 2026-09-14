"""Splits antileakage. Único lugar del repo donde se decide qué fila va a qué fold.

Dos reglas, centralizadas acá para que ningún modelo pueda saltearlas:

1. **Split agrupado por vehículo**: un mismo `vehicle_id` nunca aparece en train y
   en validación a la vez. Si se partiera, el modelo memoriza el vehículo y no
   el mecanismo.
2. **Estratificado por presencia de evento**: con pocos vehículos etiquetados,
   un fold sin eventos vuelve la métrica indefinida.

Los folds se serializan a `data/processed/splits.json` como listas de
`vehicle_id` (no de índices de fila) para que sigan siendo válidos cuando el
panel se regenere con otro Δ de corte, y se guarda una huella del panel para
detectar desalineaciones.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.config import ensure_dir, resolve_path

GROUP_COLUMN = "vehicle_id"
EVENT_COLUMN = "event_observed"


def panel_fingerprint(panel: pd.DataFrame) -> dict[str, Any]:
    """Huella del panel: sirve para detectar que los splits son de otro panel."""
    vehicles = sorted(panel[GROUP_COLUMN].astype(str).unique())
    digest = hashlib.sha256("|".join(vehicles).encode("utf-8")).hexdigest()[:16]
    return {
        "n_rows": int(len(panel)),
        "n_vehicles": int(len(vehicles)),
        "vehicles_sha256_16": digest,
    }


def make_splits(
    panel: pd.DataFrame,
    *,
    n_splits: int = 5,
    seed: int = 42,
    group_column: str = GROUP_COLUMN,
    event_column: str = EVENT_COLUMN,
) -> dict[str, Any]:
    """Construye los folds agrupados por vehículo y estratificados por evento.

    Devuelve un dict serializable: metadatos + `folds[i].valid_vehicles`.
    El train de cada fold es el complemento, así no hay riesgo de que las dos
    listas se desincronicen al editarlas a mano.
    """
    _validate_panel(panel, group_column, event_column)

    vehicle_level = (
        panel.groupby(group_column, observed=True)[event_column].max().astype(int).reset_index()
    )
    n_event_vehicles = int(vehicle_level[event_column].sum())
    if n_event_vehicles < n_splits:
        raise ValueError(
            f"Solo {n_event_vehicles} vehículos con evento para {n_splits} folds. "
            "Bajá n_splits o cambiá a CV repetida (ver plan §10, riesgo 'pocos eventos')."
        )

    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    groups = panel[group_column].astype(str).to_numpy()
    y = panel[event_column].astype(int).to_numpy()

    folds = []
    for fold, (_, valid_idx) in enumerate(splitter.split(np.zeros(len(panel)), y, groups)):
        valid_vehicles = sorted(pd.unique(groups[valid_idx]).tolist())
        valid_mask = np.isin(groups, valid_vehicles)
        folds.append(
            {
                "fold": fold,
                "valid_vehicles": valid_vehicles,
                "n_valid_rows": int(valid_mask.sum()),
                "n_train_rows": int((~valid_mask).sum()),
                "n_valid_event_vehicles": int(
                    vehicle_level.loc[
                        vehicle_level[group_column].astype(str).isin(valid_vehicles), event_column
                    ].sum()
                ),
                "valid_positive_rate": float(panel.loc[valid_mask, "label"].mean())
                if "label" in panel.columns
                else None,
            }
        )

    return {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_splits": n_splits,
        "seed": seed,
        "group_column": group_column,
        "event_column": event_column,
        "panel": panel_fingerprint(panel),
        "n_event_vehicles": n_event_vehicles,
        "folds": folds,
    }


def save_splits(splits: dict[str, Any], path: str | Path) -> Path:
    """Serializa los folds a JSON (por defecto `data/processed/splits.json`)."""
    target = resolve_path(path)
    ensure_dir(target.parent)
    with target.open("w", encoding="utf-8") as fh:
        json.dump(splits, fh, indent=2, ensure_ascii=False)
    return target


def load_splits(path: str | Path) -> dict[str, Any]:
    target = resolve_path(path)
    if not target.exists():
        raise FileNotFoundError(
            f"No existen los splits en {target}. Generalos con `scripts/make_splits.py` "
            "o dejá que `scripts/train.py` los cree si el config lo permite."
        )
    with target.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def iter_folds(
    panel: pd.DataFrame, splits: dict[str, Any], *, strict: bool = True
) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
    """Itera `(fold, train_mask, valid_mask)` sobre el panel dado.

    Con `strict=True` verifica que la huella del panel coincida con la que se usó
    para generar los splits: un panel regenerado con otro Δ cambia el número de
    filas y ahí es donde aparecen los bugs silenciosos.
    """
    group_column = splits.get("group_column", GROUP_COLUMN)
    if strict:
        current = panel_fingerprint(panel)
        stored = splits.get("panel", {})
        if stored.get("vehicles_sha256_16") != current["vehicles_sha256_16"]:
            raise ValueError(
                "Los splits no corresponden a este panel (cambió el set de vehículos). "
                "Regeneralos antes de entrenar."
            )

    groups = panel[group_column].astype(str).to_numpy()
    for fold in splits["folds"]:
        valid_mask = np.isin(groups, np.array(fold["valid_vehicles"], dtype=object).astype(str))
        train_mask = ~valid_mask
        assert_no_vehicle_leakage(groups[train_mask], groups[valid_mask])
        if valid_mask.sum() == 0 or train_mask.sum() == 0:
            raise ValueError(f"Fold {fold['fold']} vacío para este panel")
        yield int(fold["fold"]), train_mask, valid_mask


def assert_no_vehicle_leakage(train_groups: np.ndarray, valid_groups: np.ndarray) -> None:
    """Falla ruidosamente si algún vehículo cae de los dos lados del split."""
    overlap = set(train_groups) & set(valid_groups)
    if overlap:
        raise AssertionError(
            f"Leakage por vehículo: {len(overlap)} VIN en train y validación "
            f"(ej.: {sorted(overlap)[:3]})"
        )


def _validate_panel(panel: pd.DataFrame, group_column: str, event_column: str) -> None:
    missing = [c for c in (group_column, event_column) if c not in panel.columns]
    if missing:
        raise KeyError(f"El panel no tiene las columnas requeridas: {missing}")
    if panel.empty:
        raise ValueError("El panel está vacío")
