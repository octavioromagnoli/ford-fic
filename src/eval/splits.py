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

Dos niveles, con la misma mecánica y propósitos distintos:

- `make_test_split()` parte el universo de vehículos en **dev / test** una sola
  vez y congela el resultado. El test no se toca hasta el final: se congela antes
  de F2 para que ninguna decisión de diseño (features, W/G/H, umbrales) se tome
  mirándolo.
- `make_splits()` arma la CV de 5 folds **dentro de dev**, que es donde se
  compara y se elige modelo.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.config import ensure_dir, resolve_path

logger = logging.getLogger(__name__)

GROUP_COLUMN = "vehicle_id"
EVENT_COLUMN = "event_observed"


def _digest(vehicles: list[str]) -> str:
    """sha256 corto de una lista de ids, ordenada: la misma receta que `panel_fingerprint`."""
    return hashlib.sha256("|".join(sorted(vehicles)).encode("utf-8")).hexdigest()[:16]


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


def make_test_split(
    frame: pd.DataFrame,
    *,
    test_size: float = 0.2,
    seed: int = 42,
    group_column: str = GROUP_COLUMN,
    event_column: str = EVENT_COLUMN,
    tolerance: float = 0.02,
) -> dict[str, Any]:
    """Parte el universo de vehículos en dev/test, agrupado por vehículo y estratificado.

    Acepta un panel (muchas filas por vehículo) o una tabla a nivel vehículo: lo
    primero que hace es colapsar a una fila por `group_column`, así el split es por
    vehículo por construcción y no depende de cuántos cortes tenga cada uno.

    Usa el mismo `StratifiedGroupKFold` que `make_splits` y se queda con un fold
    como test: con `test_size=0.2` son 5 folds y el fold 0 es el test. Reusar el
    splitter en vez de muestrear a mano mantiene una sola implementación de la
    estratificación en el repo.

    A diferencia de los folds de CV, acá se serializan **las dos listas** de
    vehículos. En la CV el train es "el resto del panel" porque el panel ya existe;
    el test se congela *antes* de que exista el panel de F2, así que "el resto" no
    está definido al momento de usarlo: si F2 filtra vehículos o agrega otros, la
    única forma de detectarlo es tener el universo completo escrito.
    """
    _validate_panel(frame, group_column, event_column)

    if not 0.0 < test_size < 1.0:
        raise ValueError(f"test_size={test_size} tiene que estar entre 0 y 1")
    equivalent_folds = int(round(1.0 / test_size))
    if equivalent_folds < 2:
        raise ValueError(f"test_size={test_size} no deja folds: tiene que estar entre 0 y 0,5")
    achieved = 1.0 / equivalent_folds
    if abs(achieved - test_size) > tolerance:
        raise ValueError(
            f"test_size={test_size} no es representable como un fold de StratifiedGroupKFold "
            f"(el más cercano es {achieved:.4f}, con {equivalent_folds} folds). Elegí un "
            "valor de la forma 1/n (0,5 · 0,333 · 0,25 · 0,2 · 0,1) o subí `tolerance`."
        )

    vehicle_level = (
        frame.groupby(group_column, observed=True)[event_column].max().astype(int).reset_index()
    )
    n_event_vehicles = int(vehicle_level[event_column].sum())
    if n_event_vehicles < equivalent_folds:
        raise ValueError(
            f"Solo {n_event_vehicles} vehículos con evento: no alcanzan para un test del "
            f"{test_size:.0%} estratificado."
        )

    groups = vehicle_level[group_column].astype(str).to_numpy()
    y = vehicle_level[event_column].to_numpy()
    splitter = StratifiedGroupKFold(n_splits=equivalent_folds, shuffle=True, random_state=seed)
    _, test_idx = next(iter(splitter.split(np.zeros(len(vehicle_level)), y, groups)))

    test_mask = np.zeros(len(vehicle_level), dtype=bool)
    test_mask[test_idx] = True
    test_vehicles = sorted(groups[test_mask].tolist())
    dev_vehicles = sorted(groups[~test_mask].tolist())
    assert_no_vehicle_leakage(np.array(dev_vehicles), np.array(test_vehicles))

    return {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "kind": "dev_test_holdout",
        "test_size_requested": float(test_size),
        "test_size_achieved": float(test_mask.mean()),
        "equivalent_folds": equivalent_folds,
        "seed": seed,
        "group_column": group_column,
        "event_column": event_column,
        "panel": panel_fingerprint(frame.rename(columns={group_column: GROUP_COLUMN})),
        "n_vehicles": int(len(vehicle_level)),
        "n_event_vehicles": n_event_vehicles,
        "dev": _side_summary(vehicle_level, ~test_mask, event_column),
        "test": _side_summary(vehicle_level, test_mask, event_column),
        "test_vehicles": test_vehicles,
        "dev_vehicles": dev_vehicles,
    }


def restrict_test_split(
    split: dict[str, Any],
    keep_vehicles: set[str] | list[str],
    *,
    events: pd.Series | None = None,
    universe: dict[str, Any] | None = None,
    group_column: str = GROUP_COLUMN,
) -> dict[str, Any]:
    """Recorta un holdout ya congelado a un subconjunto de vehículos, **sin re-tirar el dado**.

    Cada vehículo que sobrevive conserva el lado que le tocó en el split original:
    esto filtra, no re-parte. Es la diferencia que importa. Volver a correr
    `make_test_split()` sobre el universo recortado sería un sorteo nuevo, hecho
    *después* de haber mirado los datos que motivaron el recorte —justo lo que el
    holdout existe para impedir—. Filtrar es determinista y no depende de ninguna
    métrica, así que la garantía de "elegido a ciegas" sigue en pie.

    El costo de filtrar es que las proporciones se mueven: la tasa de eventos de
    cada lado ya no queda pareja, porque la estratificación se hizo sobre la
    población vieja. Se reporta en el resumen y se acepta; forzarla de vuelta es
    elegir el test.

    Los vehículos que salen se guardan en `excluded_vehicles`. No es redundante:
    sin esa lista, `test_split_masks()` no puede distinguir un vehículo que
    **decidimos** sacar de uno que aparece porque alguien regeneró el panel con
    otro universo, y el segundo caso es un bug que hay que gritar.

    `events` es `event_observed` indexado por `vehicle_id`. Con él se recalculan
    los resúmenes por lado —cuántos vehículos, cuántos con evento, a qué tasa—; sin
    él el recorte igual sale, pero los resúmenes quedan los del holdout de origen y
    dejan de describir lo que hay.
    """
    keep = {str(v) for v in keep_vehicles}
    dev_before = [str(v) for v in split.get("dev_vehicles", [])]
    test_before = [str(v) for v in split["test_vehicles"]]
    known = set(dev_before) | set(test_before)

    forasteros = sorted(keep - known)
    if forasteros:
        raise ValueError(
            f"{len(forasteros)} vehículo(s) a conservar no están en el holdout de origen "
            f"(ej.: {forasteros[:3]}). El recorte es un subconjunto, no un universo nuevo."
        )

    dev_vehicles = sorted(v for v in dev_before if v in keep)
    test_vehicles = sorted(v for v in test_before if v in keep)
    if not dev_vehicles or not test_vehicles:
        raise ValueError(
            "El recorte deja un lado vacío "
            f"(dev={len(dev_vehicles)}, test={len(test_vehicles)}): revisá el criterio."
        )
    assert_no_vehicle_leakage(np.array(dev_vehicles), np.array(test_vehicles))

    excluded = sorted(known - keep)

    out = dict(split)
    out.update(
        {
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "kind": "dev_test_holdout_restricted",
            # La huella pasa a ser la del universo RECORTADO: es el conjunto con el
            # que se va a construir el panel, y por lo tanto contra el que tiene que
            # chequear `iter_folds`. La del universo completo queda en `parent`.
            "panel": {
                "n_rows": len(dev_vehicles) + len(test_vehicles),
                "n_vehicles": len(dev_vehicles) + len(test_vehicles),
                "vehicles_sha256_16": _digest(dev_vehicles + test_vehicles),
            },
            "dev_vehicles": dev_vehicles,
            "test_vehicles": test_vehicles,
            "excluded_vehicles": excluded,
            "n_vehicles": len(dev_vehicles) + len(test_vehicles),
            "n_excluded": len(excluded),
            "test_size_achieved": len(test_vehicles) / (len(dev_vehicles) + len(test_vehicles)),
            "universe": universe,
            "parent": {
                "kind": split.get("kind"),
                "created_at": split.get("created_at"),
                "seed": split.get("seed"),
                "n_vehicles": split.get("n_vehicles"),
                "n_event_vehicles": split.get("n_event_vehicles"),
                "panel": split.get("panel"),
                "dev": split.get("dev"),
                "test": split.get("test"),
                # Huella del REPARTO original, no del universo. Es lo que permite
                # verificar, sin guardar las 1081 ids, que el recorte salió del
                # sorteo de siempre y no de uno nuevo hecho a posteriori.
                "test_vehicles_sha256_16": _digest(test_before),
            },
        }
    )
    if events is not None:
        lookup = events.copy()
        lookup.index = lookup.index.astype(str)
        vehicle_level = pd.DataFrame(
            {
                group_column: dev_vehicles + test_vehicles,
                EVENT_COLUMN: [int(lookup[v]) for v in dev_vehicles + test_vehicles],
            }
        )
        is_test = np.array([False] * len(dev_vehicles) + [True] * len(test_vehicles))
        out["dev"] = _side_summary(vehicle_level, ~is_test, EVENT_COLUMN)
        out["test"] = _side_summary(vehicle_level, is_test, EVENT_COLUMN)
        out["n_event_vehicles"] = int(vehicle_level[EVENT_COLUMN].sum())
    return out


def test_split_masks(
    panel: pd.DataFrame, test_split: dict[str, Any], *, strict: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Máscaras `(dev, test)` para aplicar el holdout congelado sobre un panel.

    Con `strict=True` falla si el panel trae vehículos que el holdout no conoce:
    sin ese chequeo un vehículo nuevo caería en dev por descarte y nadie se
    enteraría. Que falten vehículos del holdout (F2 puede filtrar los que no tienen
    histórico suficiente) no es un error, pero se avisa por log.
    """
    group_column = test_split.get("group_column", GROUP_COLUMN)
    groups = panel[group_column].astype(str).to_numpy()
    test_vehicles = set(test_split["test_vehicles"])
    dev_vehicles = set(test_split.get("dev_vehicles", []))
    excluded = set(test_split.get("excluded_vehicles", []))

    # Un vehículo que el holdout no conoce puede ser dos cosas muy distintas, y la
    # lista de excluidos es lo único que las separa. Si está excluido, el panel se
    # construyó con el universo viejo: es un error de pipeline con arreglo conocido.
    # Si no figura en ninguna lista, el universo cambió abajo del holdout y lo que
    # hay que regenerar es el holdout. Sin distinguirlas, las dos terminan en dev
    # por descarte y ninguna métrica lo delata.
    presentes = set(groups)
    intrusos = sorted(presentes & excluded)
    if intrusos:
        message = (
            f"{len(intrusos)} vehículo(s) del panel están EXCLUIDOS del universo del "
            f"estudio (ej.: {intrusos[:3]}). El panel se construye con el universo del "
            "holdout: ver `universe` en test_split.json y `src/data/usable.py`."
        )
        if strict:
            raise ValueError(message)
        logger.warning(message)

    # Sin la lista de dev no se puede distinguir "vehículo nuevo" de "vehículo de dev",
    # así que el chequeo se hace solo cuando el holdout la trae.
    unknown = (
        sorted(presentes - (test_vehicles | dev_vehicles | excluded)) if dev_vehicles else []
    )
    if unknown:
        message = (
            f"{len(unknown)} vehículo(s) del panel no están en el holdout congelado "
            f"(ej.: {unknown[:3]}). Si el universo cambió, hay que regenerar el holdout."
        )
        if strict:
            raise ValueError(message)
        logger.warning(message)

    faltan_test = sorted(test_vehicles - set(groups))
    if faltan_test:
        logger.warning(
            "%d vehículo(s) de test no aparecen en el panel (ej.: %s)",
            len(faltan_test),
            faltan_test[:3],
        )

    test_mask = np.isin(groups, np.array(sorted(test_vehicles), dtype=object).astype(str))
    # `dev` es pertenencia explícita, no el complemento de test: con un universo
    # recortado, "no es test" incluiría a los excluidos y a cualquier forastero.
    if dev_vehicles:
        dev_mask = np.isin(groups, np.array(sorted(dev_vehicles), dtype=object).astype(str))
    else:
        dev_mask = ~test_mask
    assert_no_vehicle_leakage(groups[dev_mask], groups[test_mask])
    return dev_mask, test_mask


def split_balance(
    frame: pd.DataFrame,
    test_split: dict[str, Any],
    columns: list[str],
    *,
    group_column: str | None = None,
) -> pd.DataFrame:
    """Cómo quedó repartida cada variable entre dev y test, en proporciones.

    No estratifica por estas columnas —la estratificación es solo por evento—: las
    reporta para que un desbalance grosero se vea antes de congelar el split.
    """
    group_column = group_column or test_split.get("group_column", GROUP_COLUMN)
    vehicle_level = frame.drop_duplicates(subset=[group_column]).copy()
    is_test = vehicle_level[group_column].astype(str).isin(set(test_split["test_vehicles"]))

    rows = []
    for column in columns:
        if column not in vehicle_level.columns:
            logger.warning("split_balance: columna ausente, se omite: %s", column)
            continue
        dev_counts = vehicle_level.loc[~is_test, column].value_counts(dropna=False)
        test_counts = vehicle_level.loc[is_test, column].value_counts(dropna=False)
        for value in sorted(set(dev_counts.index) | set(test_counts.index), key=str):
            n_dev = int(dev_counts.get(value, 0))
            n_test = int(test_counts.get(value, 0))
            rows.append(
                {
                    "variable": column,
                    "value": value,
                    "n_dev": n_dev,
                    "n_test": n_test,
                    "frac_dev": n_dev / max(int((~is_test).sum()), 1),
                    "frac_test": n_test / max(int(is_test.sum()), 1),
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["diff"] = out["frac_test"] - out["frac_dev"]
    return out


def save_splits(splits: dict[str, Any], path: str | Path) -> Path:
    """Serializa los folds a JSON (por defecto `data/processed/splits.json`)."""
    return _write_json(splits, path)


def load_splits(path: str | Path) -> dict[str, Any]:
    target = resolve_path(path)
    if not target.exists():
        raise FileNotFoundError(
            f"No existen los splits en {target}. Generalos con `scripts/make_splits.py` "
            "o dejá que `scripts/train.py` los cree si el config lo permite."
        )
    return _read_json(target)


def save_test_split(test_split: dict[str, Any], path: str | Path) -> Path:
    """Serializa el holdout dev/test (por defecto `data/processed/test_split.json`)."""
    return _write_json(test_split, path)


def load_test_split(path: str | Path) -> dict[str, Any]:
    target = resolve_path(path)
    if not target.exists():
        raise FileNotFoundError(
            f"No existe el holdout dev/test en {target}. Generalo con "
            "`python scripts/make_test_split.py --config configs/data/test_split.yaml`."
        )
    return _read_json(target)


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


def _side_summary(
    vehicle_level: pd.DataFrame, mask: np.ndarray, event_column: str
) -> dict[str, Any]:
    """Resumen de un lado del holdout: cuántos vehículos, cuántos con evento y a qué tasa."""
    side = vehicle_level.loc[mask, event_column]
    return {
        "n_vehicles": int(len(side)),
        "n_event_vehicles": int(side.sum()),
        "event_rate": float(side.mean()) if len(side) else 0.0,
    }


def _write_json(payload: dict[str, Any], path: str | Path) -> Path:
    target = resolve_path(path)
    ensure_dir(target.parent)
    with target.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    return target


def _read_json(path: str | Path) -> dict[str, Any]:
    with resolve_path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _validate_panel(panel: pd.DataFrame, group_column: str, event_column: str) -> None:
    missing = [c for c in (group_column, event_column) if c not in panel.columns]
    if missing:
        raise KeyError(f"El panel no tiene las columnas requeridas: {missing}")
    if panel.empty:
        raise ValueError("El panel está vacío")
