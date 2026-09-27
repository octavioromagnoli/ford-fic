"""Splits antileakage. Único lugar del repo donde se decide qué fila va a qué fold.

Dos reglas, centralizadas acá para que ningún modelo pueda saltearlas:

1. **Split agrupado por vehículo**: un mismo `vehicle_id` nunca aparece en train y
   en validación a la vez. Si se partiera, el modelo memoriza el vehículo y no
   el mecanismo.
2. **Estratificado por la variable que mide la métrica**: por `label` colapsado a
   nivel vehículo (¿tiene al menos un corte positivo?), no por `event_observed`.
   Con pocos vehículos etiquetados, un fold sin positivos vuelve la métrica
   indefinida, y una guarda dura (`min_valid_positives`) falla antes de que eso
   pase en silencio.

Encima de las dos, **CV repetida**: `n_repeats` juegos de folds con semillas
distintas, para que la diferencia entre dos modelos no se confunda con la varianza
del sorteo (plan §10, riesgo "pocos eventos").

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
LABEL_COLUMN = "label"

# Estratificación por defecto: la etiqueta de la FILA (`label`), colapsada a nivel
# vehículo. El porqué está en el docstring de `make_splits`.
STRATIFY_COLUMN = LABEL_COLUMN
STRATIFY_LEVEL = "vehicle"
STRATIFY_LEVELS = ("vehicle", "row")

# Piso de filas positivas por fold en validación. Por debajo de esto el PR-AUC del
# fold no es una métrica mala: es ruido, y con cero positivos queda indefinido (NaN)
# y ensucia el promedio out-of-fold sin que nadie se entere.
MIN_VALID_POSITIVES = 5

N_REPEATS = 1
# Separación entre las semillas de dos repeticiones. Con paso 1, la repetición 1 de
# la semilla 42 sería la repetición 0 de la 43: dos corridas "independientes"
# compartirían folds y nadie lo notaría.
REPEAT_SEED_STEP = 10_000


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


MIN_STRATUM_SIZE = 5


def composite_strata(
    frame: pd.DataFrame, columns: list[str], *, min_size: int = MIN_STRATUM_SIZE
) -> np.ndarray:
    """Estrato compuesto por fila (`"1|BRA|ENG_3"`), fundiendo hacia arriba los chicos.

    El orden de `columns` es la prioridad: la primera (el evento) nunca se funde. Un
    estrato con menos de `min_size` filas pierde su última columna y se junta con el
    de nivel superior (`"1|BRA"`); si ese también queda chico, sube otra vez. Así cada
    estrato que llega al splitter tiene, con `min_size = n_folds`, al menos un
    representante por fold, y ninguno se descarta.

    Pensado para una fila por vehículo (el holdout) o para la tabla colapsada por
    vehículo de un panel; con muchas filas por vehículo, `min_size` cuenta filas.
    """
    if not columns:
        raise ValueError("`columns` vacío: hace falta al menos la columna del evento")
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise KeyError(f"Faltan columnas de estratificación: {missing}")
    parts = frame[columns].astype(str).fillna("NA").to_numpy()
    depth = np.full(len(frame), len(columns))
    while True:
        keys = np.array(["|".join(row[:d]) for row, d in zip(parts, depth)], dtype=object)
        counts = pd.Series(keys).value_counts()
        small = np.array([counts[k] < min_size for k in keys]) & (depth > 1)
        if not small.any():
            return keys
        depth[small] -= 1


def stratify_labels(
    panel: pd.DataFrame,
    *,
    group_column: str = GROUP_COLUMN,
    column: str = STRATIFY_COLUMN,
    level: str = STRATIFY_LEVEL,
    extra_columns: list[str] | tuple[str, ...] = (),
    min_stratum_size: int = MIN_STRATUM_SIZE,
) -> np.ndarray:
    """Vector por fila con el que se estratifican los folds.

    `level="row"` usa la columna tal cual. `level="vehicle"` la colapsa con `max`
    por vehículo y la reparte de vuelta a todas sus filas: con `column="label"` eso
    contesta *¿este vehículo tiene al menos un corte positivo?*.

    Con `extra_columns` (constantes dentro del vehículo, p. ej. mercado y motor) el
    estrato pasa a ser compuesto —`composite_strata` sobre una fila por vehículo— y
    el vector devuelto es un código entero por estrato. Sin ellas, es el binario de
    siempre, bit a bit.

    El nivel no cambia que el vehículo caiga entero de un lado —eso lo garantiza
    `groups=vehicle_id`, no la estratificación—: cambia qué se equilibra entre folds.
    """
    if level not in STRATIFY_LEVELS:
        raise ValueError(
            f"`stratify_level` desconocido: {level!r}. Opciones: {list(STRATIFY_LEVELS)}"
        )
    if column not in panel.columns:
        raise KeyError(
            f"El panel no tiene la columna de estratificación `{column}`. Sale del YAML "
            "(`splits.stratify.column`), así que revisá el config antes que el panel."
        )
    values = panel[column]
    if level == "vehicle":
        values = panel.groupby(group_column, observed=True)[column].transform("max")
    binary = (values.astype(float) > 0).astype(int)
    if not extra_columns:
        return binary.to_numpy()
    if level != "vehicle":
        raise ValueError("La estratificación compuesta (`extra_columns`) solo existe a nivel vehículo")
    extra = list(extra_columns)
    missing = [c for c in extra if c not in panel.columns]
    if missing:
        raise KeyError(f"El panel no tiene las columnas de estratificación {missing} (`splits.stratify.extra_columns`)")
    varying = [c for c in extra if panel.groupby(group_column, observed=True)[c].nunique(dropna=False).gt(1).any()]
    if varying:
        raise ValueError(f"Las columnas {varying} cambian dentro de un vehículo: no sirven para estratificar vehículos")
    per_vehicle = (
        panel.assign(_y=binary.to_numpy()).groupby(group_column, observed=True)
        .agg(_y=("_y", "max"), **{c: (c, "first") for c in extra})
    )
    keys = pd.Series(
        composite_strata(per_vehicle, ["_y", *extra], min_size=min_stratum_size), index=per_vehicle.index
    )
    codes = pd.Series(pd.factorize(keys, sort=True)[0], index=keys.index)
    return panel[group_column].map(codes).to_numpy()


def _thin_fold_message(
    *, repeat: int, fold: int, n_positives: int, minimum: int, n_valid_rows: int, where: str
) -> str:
    """Un solo texto para la guarda, la arme `make_splits` o la re-chequee `iter_repeats`."""
    return (
        f"Repetición {repeat}, fold {fold}: {n_positives} fila(s) `label=1` en validación "
        f"sobre {n_valid_rows} (mínimo {minimum}, {where}). Con menos positivos el PR-AUC "
        "del fold es ruido, y con cero queda indefinido y ensucia el promedio out-of-fold. "
        "Bajá `splits.n_splits`, agrandá H, o movés `splits.min_valid_positives` en el YAML "
        "sabiendo lo que eso implica."
    )


def _group_max(values: np.ndarray, groups: np.ndarray) -> int:
    """Cuántos grupos distintos tienen al menos un 1 en `values`."""
    return int(pd.Series(values).groupby(groups).max().sum())


def _build_folds(
    panel: pd.DataFrame,
    groups: np.ndarray,
    y: np.ndarray,
    *,
    n_splits: int,
    seed: int,
    repeat: int,
    label_column: str,
    min_valid_positives: int,
) -> list[dict[str, Any]]:
    """Los `n_splits` folds de UNA repetición, ya chequeados contra la guarda de positivos."""
    labels = panel[label_column].astype(int).to_numpy()
    events = (
        panel[EVENT_COLUMN].astype(int).to_numpy() if EVENT_COLUMN in panel.columns else None
    )
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    folds = []
    for fold, (_, valid_idx) in enumerate(splitter.split(np.zeros(len(panel)), y, groups)):
        valid_vehicles = sorted(pd.unique(groups[valid_idx]).tolist())
        folds.append(_fold_record(groups, labels, events, fold=fold, valid_vehicles=valid_vehicles,
                                  repeat=repeat, min_valid_positives=min_valid_positives,
                                  where="al armar los folds"))
    return folds


def _fold_record(
    groups: np.ndarray,
    labels: np.ndarray,
    events: np.ndarray | None,
    *,
    fold: int,
    valid_vehicles: list[str],
    repeat: int,
    min_valid_positives: int,
    where: str,
) -> dict[str, Any]:
    """El registro de un fold en el JSON, después de pasar la guarda de positivos."""
    valid_mask = np.isin(groups, valid_vehicles)
    n_positives = int(labels[valid_mask].sum())
    if n_positives < min_valid_positives:
        raise ValueError(
            _thin_fold_message(
                repeat=repeat,
                fold=fold,
                n_positives=n_positives,
                minimum=min_valid_positives,
                n_valid_rows=int(valid_mask.sum()),
                where=where,
            )
        )
    return {
        "fold": fold,
        "valid_vehicles": valid_vehicles,
        "n_valid_rows": int(valid_mask.sum()),
        "n_train_rows": int((~valid_mask).sum()),
        "n_valid_event_vehicles": (
            _group_max(events[valid_mask], groups[valid_mask]) if events is not None else None
        ),
        "valid_positive_rate": float(labels[valid_mask].mean()),
        # Lo que mira la guarda, escrito en el JSON para que la tabla de folds
        # salga del archivo y no haya que recalcularla contra el panel.
        "n_valid_positives": n_positives,
        "n_valid_positive_vehicles": _group_max(labels[valid_mask], groups[valid_mask]),
    }


def make_splits(
    panel: pd.DataFrame,
    *,
    n_splits: int = 5,
    seed: int = 42,
    group_column: str = GROUP_COLUMN,
    stratify_column: str = STRATIFY_COLUMN,
    stratify_level: str = STRATIFY_LEVEL,
    label_column: str = LABEL_COLUMN,
    min_valid_positives: int = MIN_VALID_POSITIVES,
    n_repeats: int = N_REPEATS,
    stratify_extra_columns: list[str] | tuple[str, ...] = (),
    min_stratum_size: int = MIN_STRATUM_SIZE,
) -> dict[str, Any]:
    """Construye los folds agrupados por vehículo y estratificados.

    Devuelve un dict serializable: metadatos + `repeats[r].folds[i].valid_vehicles`.
    El train de cada fold es el complemento, así no hay riesgo de que las dos listas
    se desincronicen al editarlas a mano.

    **Por qué la estratificación sale de `label` y no de `event_observed`.** El
    PR-AUC se calcula sobre `label` —¿el evento cae en el horizonte de ESTA fila?—,
    mientras que `event_observed` dice si el vehículo tuvo evento alguna vez. Hoy,
    con W=1000/G=500, los dos coinciden: los 53 vehículos de dev con evento que
    llegan al panel tienen al menos un corte positivo, porque los cortes de un
    vehículo con evento terminan en `E − G` y ese último corte es positivo por
    construcción (`src/data/panel.py::vehicle_cuts`). Pero nada lo garantiza:
    alcanza con que el QC de ventana (`min_trips_in_window`, `min_km_covered_frac`)
    descarte los cortes tardíos de un vehículo —más probable con W grande— para que
    entre al panel con cero positivos y la estratificación quede equilibrando una
    variable que la métrica no usa. Derivarla de `label` cierra esa puerta sin
    cambiar nada mientras las dos coincidan. `event_observed` sigue disponible como
    `stratify.column` para el panel dummy y para reproducir corridas viejas.

    `stratify_level="vehicle"` (default) colapsa la columna con `max` por vehículo;
    `"row"` la usa fila a fila, lo que equilibra el CONTEO de filas positivas por
    fold. Ninguno de los dos afecta la integridad del vehículo: eso lo da `groups`.

    **Estrato compuesto (entrega v2).** Con `stratify_extra_columns` (p. ej. mercado y
    motor) cada vehículo se estratifica por `label × mercado × motor`, fundiendo hacia
    arriba los estratos de menos de `min_stratum_size` vehículos (`composite_strata`).
    En v2 el riesgo cambia 8× entre mercados y ENG_3 falla casi solo en Brasil: sin
    esto, un fold puede quedarse con los fallados de un estrato entero.

    **Guarda de positivos.** Si un fold queda con menos de `min_valid_positives`
    filas `label=1` en validación, esto falla nombrando repetición, fold y conteo.
    Antes el caso pasaba en silencio y el PR-AUC del fold salía NaN.

    **CV repetida.** Con `n_repeats > 1` se sortean R juegos de folds con semillas
    `seed + r * REPEAT_SEED_STEP`. La repetición 0 usa `seed` tal cual, así que
    subir R **no** cambia los folds que ya existían: los suma. `folds` en la raíz
    del dict es siempre la repetición 0, para que lo que lea el formato viejo siga
    leyendo una CV válida.
    """
    _validate_panel(panel, group_column, stratify_column, label_column)
    if n_splits < 2:
        raise ValueError(f"n_splits={n_splits}: hacen falta al menos 2 folds")
    if n_repeats < 1:
        raise ValueError(f"n_repeats={n_repeats}: tiene que ser >= 1 (1 = CV simple)")
    if min_valid_positives < 1:
        raise ValueError(
            f"min_valid_positives={min_valid_positives}: con 0 la guarda no guarda nada. "
            "Si un fold sin positivos es aceptable, el problema es otro."
        )

    groups = panel[group_column].astype(str).to_numpy()
    binary = stratify_labels(
        panel, group_column=group_column, column=stratify_column, level=stratify_level
    )
    y = (
        stratify_labels(
            panel,
            group_column=group_column,
            column=stratify_column,
            level=stratify_level,
            extra_columns=list(stratify_extra_columns),
            min_stratum_size=min_stratum_size,
        )
        if stratify_extra_columns
        else binary
    )
    n_positive_vehicles = _group_max(binary, groups)
    if n_positive_vehicles < n_splits:
        raise ValueError(
            f"Solo {n_positive_vehicles} vehículo(s) positivos según `{stratify_column}` "
            f"({stratify_level}) para {n_splits} folds. Bajá n_splits o subí H "
            "(ver plan §10, riesgo 'pocos eventos')."
        )

    events = panel[EVENT_COLUMN] if EVENT_COLUMN in panel.columns else None
    n_event_vehicles = (
        _group_max(events.astype(int).to_numpy(), groups) if events is not None else None
    )

    repeats = []
    for repeat in range(n_repeats):
        repeat_seed = int(seed) + repeat * REPEAT_SEED_STEP
        repeats.append(
            {
                "repeat": repeat,
                "seed": repeat_seed,
                "folds": _build_folds(
                    panel,
                    groups,
                    y,
                    n_splits=n_splits,
                    seed=repeat_seed,
                    repeat=repeat,
                    label_column=label_column,
                    min_valid_positives=min_valid_positives,
                ),
            }
        )

    return {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_splits": n_splits,
        "seed": seed,
        "group_column": group_column,
        "stratify": {
            "column": stratify_column,
            "level": stratify_level,
            "n_positive_vehicles": n_positive_vehicles,
            **(
                {
                    "extra_columns": list(stratify_extra_columns),
                    "min_stratum_size": int(min_stratum_size),
                    "n_strata": int(len(np.unique(y))),
                }
                if stratify_extra_columns
                else {}
            ),
        },
        "label_column": label_column,
        "min_valid_positives": int(min_valid_positives),
        "n_repeats": int(n_repeats),
        "panel": panel_fingerprint(panel),
        "n_event_vehicles": n_event_vehicles,
        "repeats": repeats,
        # Espejo de la repetición 0: compatibilidad con todo lo que lee `splits["folds"]`.
        "folds": repeats[0]["folds"],
    }


def _file_digest(path: str | Path) -> str:
    return hashlib.sha256(resolve_path(path).read_bytes()).hexdigest()[:16]


def extend_splits(
    base_splits: dict[str, Any],
    panel: pd.DataFrame,
    *,
    seed: int,
    n_splits: int | None = None,
    n_repeats: int | None = None,
    group_column: str = GROUP_COLUMN,
    stratify_column: str = STRATIFY_COLUMN,
    stratify_level: str = STRATIFY_LEVEL,
    label_column: str = LABEL_COLUMN,
    min_valid_positives: int = MIN_VALID_POSITIVES,
    guard_by: str | None = None,
    base_path: str | Path | None = None,
) -> dict[str, Any]:
    """Folds para un panel nuevo que **conservan** los de un `splits.json` existente.

    Para que la comparación con el finalista quede pareada (cure model, preregistro §6):
    en cada repetición, un vehículo que ya tenía fold en `base_splits` lo conserva. Solo
    se sortean los vehículos que el panel base no tenía:

    * se estratifican por `stratify_column` a nivel vehículo (como `make_splits`) y, dentro
      de cada estrato, cada uno va al fold con menos vehículos de ese estrato, desempatando
      por menos vehículos en total y después al azar. El azar sale de
      `seed + r · REPEAT_SEED_STEP`, la misma receta de semillas que `make_splits`;
    * los vehículos del panel base que no están en este panel se descartan y se cuentan.

    **Guarda.** La de siempre (filas `label = 1` por fold) y, con `guard_by`, una más
    estricta: al menos `min_valid_positives` **vehículos** positivos en validación por
    cada valor de esa columna (en el panel de hitos, por hito). Si no se cumple, falla: la
    salida del preregistro es `make_splits` normal, con la comparación declarada no pareada.

    El resultado tiene el mismo formato que `make_splits` más `extended_from` (path, huella
    del archivo y del panel base, cuántos vehículos se conservaron, sortearon y descartaron)
    y, por fold, `n_valid_new_vehicles` y `n_valid_positive_vehicles_by`.
    """
    _validate_panel(panel, group_column, stratify_column, label_column, *([guard_by] if guard_by else []))
    base_repeats = _repeats_of(base_splits)
    declared = splits_declared(base_splits)
    n_splits = int(n_splits or declared["n_splits"])
    n_repeats = int(n_repeats or len(base_repeats))
    if n_splits != declared["n_splits"] or n_repeats != len(base_repeats):
        raise ValueError(
            f"El split base tiene {declared['n_splits']} folds × {len(base_repeats)} repeticiones y se "
            f"pidieron {n_splits} × {n_repeats}: extender solo conserva folds que existen."
        )
    if min_valid_positives < 1:
        raise ValueError(f"min_valid_positives={min_valid_positives}: con 0 la guarda no guarda nada")

    groups = panel[group_column].astype(str).to_numpy()
    labels = panel[label_column].astype(int).to_numpy()
    events = panel[EVENT_COLUMN].astype(int).to_numpy() if EVENT_COLUMN in panel.columns else None
    strata = pd.Series(stratify_labels(panel, group_column=group_column, column=stratify_column,
                                       level=stratify_level)).groupby(groups).max()
    vehicles = sorted(strata.index)
    base_vehicles = {str(v) for fold in base_repeats[0]["folds"] for v in fold["valid_vehicles"]}
    kept = [v for v in vehicles if v in base_vehicles]
    new = [v for v in vehicles if v not in base_vehicles]
    dropped = sorted(base_vehicles - set(vehicles))

    repeats = []
    for repeat, entry in enumerate(base_repeats):
        base_fold = {str(v): int(fold["fold"]) for fold in entry["folds"] for v in fold["valid_vehicles"]}
        if set(base_fold) != base_vehicles:
            raise ValueError(f"La repetición {repeat} del split base no reparte los mismos vehículos que la 0")
        assignment = {v: base_fold[v] for v in kept}
        rng = np.random.default_rng(int(seed) + repeat * REPEAT_SEED_STEP)
        for stratum in sorted(strata.unique(), reverse=True):
            in_stratum = np.zeros(n_splits, dtype=int)
            totals = np.zeros(n_splits, dtype=int)
            for vehicle, fold in assignment.items():
                totals[fold] += 1
                in_stratum[fold] += int(strata[vehicle] == stratum)
            candidates = [v for v in new if strata[v] == stratum]
            for vehicle in rng.permutation(np.asarray(candidates, dtype=object)):
                fold = int(np.lexsort((rng.random(n_splits), totals, in_stratum))[0])
                assignment[str(vehicle)] = fold
                in_stratum[fold] += 1
                totals[fold] += 1

        folds = []
        for fold in range(n_splits):
            valid_vehicles = sorted(v for v, f in assignment.items() if f == fold)
            record = _fold_record(groups, labels, events, fold=fold, valid_vehicles=valid_vehicles,
                                  repeat=repeat, min_valid_positives=min_valid_positives,
                                  where="al extender los folds")
            record["n_valid_new_vehicles"] = int(sum(v not in base_vehicles for v in valid_vehicles))
            if guard_by:
                valid_mask = np.isin(groups, valid_vehicles)
                by_group = (pd.DataFrame({"g": panel[guard_by].to_numpy()[valid_mask], "v": groups[valid_mask],
                                          "y": labels[valid_mask]})
                            .loc[lambda d: d["y"] == 1].groupby("g")["v"].nunique())
                values = sorted(pd.unique(panel[guard_by]))
                counts = {f"{value:g}" if isinstance(value, float) else str(value): int(by_group.get(value, 0))
                          for value in values}
                thin = {k: n for k, n in counts.items() if n < min_valid_positives}
                if thin:
                    raise ValueError(
                        f"Repetición {repeat}, fold {fold}: {thin} vehículo(s) positivos en validación por "
                        f"`{guard_by}` (mínimo {min_valid_positives}). Extender no es viable: la salida del "
                        "preregistro es `make_splits` normal, con la comparación declarada no pareada."
                    )
                record["n_valid_positive_vehicles_by"] = counts
            folds.append(record)
        covered = [v for fold in folds for v in fold["valid_vehicles"]]
        if sorted(covered) != vehicles:
            raise AssertionError(f"Repetición {repeat}: los folds no reparten cada vehículo del panel exactamente una vez")
        repeats.append({"repeat": repeat, "seed": int(seed) + repeat * REPEAT_SEED_STEP,
                        "base_seed": entry.get("seed"), "folds": folds})

    return {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_splits": n_splits,
        "seed": int(seed),
        "group_column": group_column,
        "stratify": {"column": stratify_column, "level": stratify_level,
                     "n_positive_vehicles": int(strata.sum())},
        "label_column": label_column,
        "min_valid_positives": int(min_valid_positives),
        "guard_by": guard_by,
        "n_repeats": n_repeats,
        "panel": panel_fingerprint(panel),
        "n_event_vehicles": _group_max(events, groups) if events is not None else None,
        "extended_from": {
            "path": None if base_path is None else str(base_path),
            "file_sha256_16": None if base_path is None else _file_digest(base_path),
            "panel": base_splits.get("panel"),
            "declared": declared,
            "n_kept": len(kept),
            "n_new": len(new),
            "n_new_positive": int(strata[new].sum()) if new else 0,
            "n_dropped": len(dropped),
            "dropped_vehicles": dropped,
        },
        "repeats": repeats,
        "folds": repeats[0]["folds"],
    }


# Con qué parámetros se armó un `splits.json` que no los declara: son los que el
# código usaba antes de que la estratificación fuera configurable.
LEGACY_OPTIONS = {"stratify_column": EVENT_COLUMN, "stratify_level": "row", "n_repeats": 1}


def split_options(cfg: dict[str, Any], *, key: str = "splits") -> dict[str, Any]:
    """Traduce el bloque `splits:` de un YAML a los kwargs de `make_splits` (regla 7).

    Vive acá y no en cada script para que los cuatro entrypoints que arman folds
    —`build_dataset.py`, `make_splits.py`, `make_dummy.py`, `train.py`— lean las
    mismas claves con los mismos defaults. Un default distinto entre dos scripts es
    un split distinto sin que ningún YAML lo diga.
    """
    block = cfg.get(key, {}) or {}
    stratify = block.get("stratify", {}) or {}
    options = {
        "n_splits": int(block.get("n_splits", 5)),
        "seed": int(block.get("seed", 42)),
        "stratify_column": str(stratify.get("column", STRATIFY_COLUMN)),
        "stratify_level": str(stratify.get("level", STRATIFY_LEVEL)),
        "min_valid_positives": int(block.get("min_valid_positives", MIN_VALID_POSITIVES)),
        "n_repeats": int(block.get("n_repeats", N_REPEATS)),
    }
    # Solo aparecen si el YAML las declara: un YAML viejo arma exactamente los mismos
    # kwargs que antes, y `make_splits` no las recibe.
    if stratify.get("extra_columns"):
        options["stratify_extra_columns"] = [str(c) for c in stratify["extra_columns"]]
        options["min_stratum_size"] = int(stratify.get("min_stratum_size", MIN_STRATUM_SIZE))
    return options


def splits_declared(splits: dict[str, Any]) -> dict[str, Any]:
    """Con qué parámetros se armó un `splits.json` (incluido uno del formato viejo)."""
    stratify = splits.get("stratify") or {}
    declared = {
        "n_splits": int(splits.get("n_splits", len(splits.get("folds", [])))),
        "seed": splits.get("seed"),
        "stratify_column": str(stratify.get("column", LEGACY_OPTIONS["stratify_column"])),
        "stratify_level": str(stratify.get("level", LEGACY_OPTIONS["stratify_level"])),
        "n_repeats": int(splits.get("n_repeats", len(_repeats_of(splits)))),
    }
    if stratify.get("extra_columns"):
        declared["stratify_extra_columns"] = list(stratify["extra_columns"])
        declared["min_stratum_size"] = int(stratify.get("min_stratum_size", MIN_STRATUM_SIZE))
    return declared


def splits_match_options(splits: dict[str, Any], options: dict[str, Any]) -> list[str]:
    """Diferencias entre lo que declara el YAML y con qué se armó el archivo congelado.

    Devuelve la lista de discrepancias (vacía si coinciden). No compara
    `min_valid_positives`: ese no cambia los folds, y el valor del YAML se aplica
    igual al entrenar, aunque el archivo sea viejo y no lo traiga.
    """
    declared = splits_declared(splits)
    diffs = []
    for key in ("n_splits", "seed", "stratify_column", "stratify_level", "n_repeats"):
        want, got = options.get(key), declared.get(key)
        if want is not None and got is not None and want != got:
            diffs.append(f"{key} (YAML={want!r}, archivo={got!r})")
    # Un estrato compuesto declarado de un solo lado también es una diferencia.
    for key in ("stratify_extra_columns", "min_stratum_size"):
        want, got = options.get(key), declared.get(key)
        if want != got:
            diffs.append(f"{key} (YAML={want!r}, archivo={got!r})")
    return diffs


def make_test_split(
    frame: pd.DataFrame,
    *,
    test_size: float = 0.2,
    seed: int = 42,
    group_column: str = GROUP_COLUMN,
    event_column: str = EVENT_COLUMN,
    tolerance: float = 0.02,
    stratify_columns: list[str] | None = None,
    min_stratum_size: int = MIN_STRATUM_SIZE,
) -> dict[str, Any]:
    """Parte el universo de vehículos en dev/test, agrupado por vehículo y estratificado.

    Con `stratify_columns` (entrega v2: `[event_observed, mercado, motor]`) el estrato
    es compuesto (`composite_strata`, fundiendo los chicos hacia arriba); sin él, es
    `event_column` como siempre, bit a bit.

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
    strata_info: dict[str, Any] | None = None
    if stratify_columns:
        if stratify_columns[0] != event_column:
            raise ValueError(
                f"La primera columna del estrato tiene que ser el evento (`{event_column}`): es la "
                "única que nunca se funde."
            )
        extra = list(stratify_columns[1:])
        per_vehicle = frame.groupby(group_column, observed=True)[extra].first()
        vehicle_level = vehicle_level.join(per_vehicle, on=group_column)
        keys = composite_strata(vehicle_level, stratify_columns, min_size=min_stratum_size)
        y = pd.factorize(pd.Series(keys), sort=True)[0]
        strata_info = {
            "columns": list(stratify_columns),
            "min_stratum_size": int(min_stratum_size),
            "strata": {str(k): int(v) for k, v in pd.Series(keys).value_counts().sort_index().items()},
        }
    splitter = StratifiedGroupKFold(n_splits=equivalent_folds, shuffle=True, random_state=seed)
    _, test_idx = next(iter(splitter.split(np.zeros(len(vehicle_level)), y, groups)))

    test_mask = np.zeros(len(vehicle_level), dtype=bool)
    test_mask[test_idx] = True
    test_vehicles = sorted(groups[test_mask].tolist())
    dev_vehicles = sorted(groups[~test_mask].tolist())
    assert_no_vehicle_leakage(np.array(dev_vehicles), np.array(test_vehicles))

    extra_fields: dict[str, Any] = {}
    if strata_info is not None:
        by_side = pd.DataFrame({"stratum": keys, "test": test_mask}).groupby("stratum")["test"].agg(["size", "sum"])
        strata_info["test_by_stratum"] = {str(k): int(v) for k, v in by_side["sum"].items()}
        extra_fields["stratify"] = strata_info

    return {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "kind": "dev_test_holdout",
        **extra_fields,
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


def holdout_on_universe(
    vehicles: pd.DataFrame,
    keep: pd.Series | np.ndarray,
    *,
    test_size: float = 0.2,
    seed: int = 42,
    group_column: str = GROUP_COLUMN,
    event_column: str = EVENT_COLUMN,
    stratify_columns: list[str] | None = None,
    min_stratum_size: int = MIN_STRATUM_SIZE,
    universe: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Holdout dev/test sorteado **sobre el universo ya definido**, en una sola etapa.

    Es el procedimiento de la entrega v2 (26-09-2026). En la entrega 1 se sorteaba la
    población completa y después se recortaba (`restrict_test_split`), para que el
    dado no se tirara después de mirar los datos que motivaron el recorte. En v2 el
    universo, las etiquetas y los estratos cambiaron enteros, y se decidió volver a
    sortear con la misma semilla: el universo se fija con criterios de calidad y de
    muestreo (sin mirar ninguna feature contra la etiqueta) y recién ahí se sortea
    una vez, estratificando por `stratify_columns`. Lo que se paga, y queda escrito
    en el JSON por `compare_holdouts`, es que algunos autos que se miraron en el dev
    viejo caen en el test nuevo.

    Devuelve el mismo formato que `restrict_test_split` (tres listas, `universe`,
    huella del universo), así `test_split_masks` lo consume igual.
    """
    keep = pd.Series(np.asarray(keep, dtype=bool), index=vehicles.index)
    inside = vehicles.loc[keep]
    split = make_test_split(
        inside,
        test_size=test_size,
        seed=seed,
        group_column=group_column,
        event_column=event_column,
        stratify_columns=stratify_columns,
        min_stratum_size=min_stratum_size,
    )
    everyone = set(vehicles[group_column].astype(str))
    excluded = sorted(everyone - set(split["dev_vehicles"]) - set(split["test_vehicles"]))
    split.update(
        {
            "kind": "dev_test_holdout_universe",
            "excluded_vehicles": excluded,
            "n_excluded": len(excluded),
            "universe": universe,
            "population": {
                "n_vehicles": len(everyone),
                "vehicles_sha256_16": _digest(sorted(everyone)),
            },
        }
    )
    return split


def extend_holdout(
    base: dict[str, Any],
    vehicles: pd.DataFrame,
    keep: pd.Series | np.ndarray,
    *,
    test_size: float = 0.2,
    seed: int = 42,
    group_column: str = GROUP_COLUMN,
    event_column: str = EVENT_COLUMN,
    stratify_columns: list[str] | None = None,
    min_stratum_size: int = MIN_STRATUM_SIZE,
    keep_sides_of: dict[str, Any] | None = None,
    universe: dict[str, Any] | None = None,
    base_path: str | Path | None = None,
) -> dict[str, Any]:
    """Agranda un holdout ya congelado a un universo más grande **sin mover a nadie**.

    Es lo que se hizo el 26-09-2026 al sacar la ventana de producción del universo v2
    (557 → 990 vehículos). Tres reglas, en este orden:

    1. Todo vehículo de `base` conserva su lado. Nadie que se haya mirado en dev pasa al
       test (la EDA del dev de `base` ya se hizo), y el test de `base` sigue intacto.
    2. Entre los que entran, los que estaban en dev o test de `keep_sides_of` (el holdout
       de la entrega 1) vuelven a ese lado. Así no se suma al test ningún auto que se
       miró durante F2–F8.
    3. El resto se sortea una vez con `make_test_split` (misma semilla, mismo estrato).

    Un universo más chico que el de `base` no se extiende: falla. Restringir es otra
    operación (`restrict_test_split`), y como la extensión no mueve a nadie, restringir el
    resultado al universo de `base` devuelve exactamente `base`.
    """
    keep = pd.Series(np.asarray(keep, dtype=bool), index=vehicles.index)
    inside = set(vehicles.loc[keep, group_column].astype(str))
    base_dev, base_test = set(base.get("dev_vehicles", [])), set(base["test_vehicles"])
    lost = sorted((base_dev | base_test) - inside)
    if lost:
        raise ValueError(
            f"{len(lost)} vehículo(s) del holdout base quedan fuera del universo nuevo (ej.: {lost[:3]}): "
            "extender no saca a nadie. Para achicar el universo se restringe, no se extiende."
        )
    new = inside - base_dev - base_test
    prev_dev = set((keep_sides_of or {}).get("dev_vehicles", []))
    prev_test = set((keep_sides_of or {}).get("test_vehicles", []))
    forced_dev, forced_test = sorted(new & prev_dev), sorted(new & prev_test)
    to_draw = sorted(new - set(forced_dev) - set(forced_test))

    drawn: dict[str, Any] = {"dev_vehicles": [], "test_vehicles": []}
    if to_draw:
        frame = vehicles[vehicles[group_column].astype(str).isin(to_draw)]
        drawn = make_test_split(frame, test_size=test_size, seed=seed, group_column=group_column,
                                event_column=event_column, stratify_columns=stratify_columns,
                                min_stratum_size=min_stratum_size)

    dev = sorted(base_dev | set(forced_dev) | set(drawn["dev_vehicles"]))
    test = sorted(base_test | set(forced_test) | set(drawn["test_vehicles"]))
    assert_no_vehicle_leakage(np.array(dev), np.array(test))
    everyone = set(vehicles[group_column].astype(str))
    excluded = sorted(everyone - set(dev) - set(test))

    per_vehicle = vehicles.drop_duplicates(group_column)
    events = pd.Series(per_vehicle[event_column].to_numpy(), index=per_vehicle[group_column].astype(str))
    level = pd.DataFrame({group_column: dev + test, event_column: [int(events[v]) for v in dev + test]})
    is_test = np.array([False] * len(dev) + [True] * len(test))
    return {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "kind": "dev_test_holdout_extended",
        "seed": seed,
        "group_column": group_column,
        "event_column": event_column,
        "test_size_requested": float(test_size),
        "test_size_achieved": len(test) / (len(dev) + len(test)),
        "panel": {"n_rows": len(dev) + len(test), "n_vehicles": len(dev) + len(test),
                  "vehicles_sha256_16": _digest(dev + test)},
        "population": {"n_vehicles": len(everyone), "vehicles_sha256_16": _digest(sorted(everyone))},
        "n_vehicles": len(dev) + len(test),
        "n_event_vehicles": int(level[event_column].sum()),
        "dev": _side_summary(level, ~is_test, event_column),
        "test": _side_summary(level, is_test, event_column),
        "dev_vehicles": dev,
        "test_vehicles": test,
        "excluded_vehicles": excluded,
        "n_excluded": len(excluded),
        "universe": universe,
        "extended_from": {
            "path": None if base_path is None else str(base_path),
            "created_at": base.get("created_at"),
            "kind": base.get("kind"),
            "dev_vehicles_sha256_16": _digest(sorted(base_dev)),
            "test_vehicles_sha256_16": _digest(sorted(base_test)),
            "n_base_dev": len(base_dev),
            "n_base_test": len(base_test),
            "n_new": len(new),
            "n_forced_dev": len(forced_dev),
            "n_forced_test": len(forced_test),
            "n_drawn": len(to_draw),
            "n_drawn_test": len(drawn["test_vehicles"]),
            "stratify": drawn.get("stratify"),
        },
    }


def compare_holdouts(new: dict[str, Any], old: dict[str, Any]) -> dict[str, Any]:
    """De qué lado estaba en el holdout viejo cada vehículo del nuevo.

    El número que importa es `dev->test`: autos que se miraron durante la selección
    de modelo del holdout viejo y ahora están en el test. La lista se guarda para que
    la evaluación final pueda reportar el test también sin ellos.
    """
    def side_map(split: dict[str, Any]) -> dict[str, str]:
        out = {v: "dev" for v in split.get("dev_vehicles", [])}
        out.update({v: "test" for v in split.get("test_vehicles", [])})
        out.update({v: "excl" for v in split.get("excluded_vehicles", [])})
        return out

    before, after = side_map(old), side_map(new)
    transitions: dict[str, int] = {}
    for vehicle, side in after.items():
        key = f"{before.get(vehicle, 'nuevo')}->{side}"
        transitions[key] = transitions.get(key, 0) + 1
    seen = sorted(v for v in new.get("test_vehicles", []) if before.get(v) == "dev")
    return {
        "old_created_at": old.get("created_at"),
        "old_kind": old.get("kind"),
        "old_counts": {k: len(old.get(k, [])) for k in ("dev_vehicles", "test_vehicles", "excluded_vehicles")},
        "transitions": dict(sorted(transitions.items())),
        "test_vehicles_in_old_dev": seen,
        "n_test_vehicles_in_old_dev": len(seen),
    }


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


def _repeats_of(splits: dict[str, Any]) -> list[dict[str, Any]]:
    """Normaliza el formato: un `splits.json` viejo (solo `folds`) es una repetición."""
    if "repeats" in splits:
        if not splits["repeats"]:
            raise ValueError("El splits.json no tiene ninguna repetición")
        return splits["repeats"]
    if "folds" in splits:
        return [{"repeat": 0, "seed": splits.get("seed"), "folds": splits["folds"]}]
    raise KeyError("El splits.json no tiene ni `repeats` ni `folds`: no es un split de este repo")


def iter_repeats(
    panel: pd.DataFrame,
    splits: dict[str, Any],
    *,
    strict: bool = True,
    min_valid_positives: int | None = None,
) -> Iterator[tuple[int, list[tuple[int, np.ndarray, np.ndarray]]]]:
    """Itera `(repeat, [(fold, train_mask, valid_mask), ...])` sobre el panel dado.

    Es el único camino para conseguir máscaras de fold: `run_cv` no arma splits, los
    pide (CLAUDE.md, regla 2).

    Con `strict=True` verifica que la huella del panel coincida con la que se usó
    para generar los splits: un panel regenerado con otro Δ cambia el set de
    vehículos y ahí es donde aparecen los bugs silenciosos.

    `min_valid_positives` re-chequea la guarda **al entrenar**, no solo al armar los
    folds. Importa porque un `splits.json` congelado antes de que la guarda existiera
    no la trae adentro: sin este segundo control, un fold flaco entra a la CV igual.
    Si no se pasa, se usa el valor que el archivo declare; si el archivo tampoco lo
    trae, no se chequea (y se avisa por log una vez).
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

    if min_valid_positives is None:
        min_valid_positives = splits.get("min_valid_positives")
    label_column = splits.get("label_column", LABEL_COLUMN)
    labels = (
        panel[label_column].astype(int).to_numpy() if label_column in panel.columns else None
    )
    if min_valid_positives and labels is None:
        logger.warning(
            "El panel no tiene `%s`: no se puede chequear el mínimo de positivos por fold",
            label_column,
        )

    groups = panel[group_column].astype(str).to_numpy()
    for entry in _repeats_of(splits):
        repeat = int(entry.get("repeat", 0))
        masks: list[tuple[int, np.ndarray, np.ndarray]] = []
        for fold in entry["folds"]:
            valid_mask = np.isin(groups, np.array(fold["valid_vehicles"], dtype=object).astype(str))
            train_mask = ~valid_mask
            assert_no_vehicle_leakage(groups[train_mask], groups[valid_mask])
            if valid_mask.sum() == 0 or train_mask.sum() == 0:
                raise ValueError(
                    f"Fold {fold['fold']} de la repetición {repeat} vacío para este panel"
                )
            if min_valid_positives and labels is not None:
                n_positives = int(labels[valid_mask].sum())
                if n_positives < int(min_valid_positives):
                    raise ValueError(
                        _thin_fold_message(
                            repeat=repeat,
                            fold=int(fold["fold"]),
                            n_positives=n_positives,
                            minimum=int(min_valid_positives),
                            n_valid_rows=int(valid_mask.sum()),
                            where="chequeado al entrenar contra este panel",
                        )
                    )
            masks.append((int(fold["fold"]), train_mask, valid_mask))
        yield repeat, masks


def iter_folds(
    panel: pd.DataFrame,
    splits: dict[str, Any],
    *,
    strict: bool = True,
    min_valid_positives: int | None = None,
) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
    """Itera `(fold, train_mask, valid_mask)` de la **repetición 0**.

    Existe para el código que no sabe de CV repetida (y para leer un `splits.json`
    del formato viejo). Lo que entrena de verdad usa `iter_repeats`: con R > 1 esto
    devolvería un quinto de los folds y el promedio saldría de menos pasadas de las
    que el config pidió.
    """
    for _, masks in iter_repeats(
        panel, splits, strict=strict, min_valid_positives=min_valid_positives
    ):
        yield from masks
        return


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


def _validate_panel(panel: pd.DataFrame, *columns: str) -> None:
    missing = [c for c in dict.fromkeys(columns) if c not in panel.columns]
    if missing:
        raise KeyError(f"El panel no tiene las columnas requeridas: {missing}")
    if panel.empty:
        raise ValueError("El panel está vacío")
