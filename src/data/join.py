"""Unión de las tres tablas crudas a nivel vehículo, con la auditoría del join.

El panel de F2 tiene una fila por `(vehículo, punto de corte)`, pero las tres
tablas crudas no se unen fila a fila: `vehicles` es una fila por vehículo,
`trips` una por viaje y `signals` una por mensaje del postratamiento.

**`trips` y `signals` no se mergean entre sí.** Comparten `VehicleCode`, pero no
una clave fila a fila: un merge por vehículo es muchos-a-muchos y el resultado es
el producto de los dos históricos. Para el vehículo mediano son 1971 viajes ×
7195 señales ≈ 14,2M de filas **de un solo vehículo**; sumando los 1081 da
3,2×10¹⁰ filas. No es que sea lento: no entra. Alinear señales contra viajes es
trabajo de F2 y se hace por odómetro/timestamp, no por clave (`trips` manda y
`signals` se alinea contra él, ver `docs/reproducibilidad.md`).

Lo que este módulo ofrece son las dos uniones que sí tienen sentido:

- **Enriquecer cada tabla grande por separado** con las estáticas del vehículo
  (`attach_static`, `iter_enriched_table`): left join 1-a-muchos que deja `trips`
  con sus filas y `signals` con las suyas, solo con columnas nuevas pegadas. Son
  dos tablas separadas, no una fusionada.
- **Bajar todo a nivel vehículo** (`build_vehicle_table`): una fila por
  `vehicle_id` con las estáticas, la etiqueta de cohorte y la cobertura de cada
  tabla. Es lo único que hace falta para auditar el join y congelar el holdout.

Todo se hace en **una sola pasada por tabla** (`scan_table`), porque `signals`
tiene 10,6M de filas: la misma pasada cuenta nulos por columna, detecta filas
duplicadas por hash y arma los agregados por vehículo.

El dedupe de los 13 vehículos clonados se aplica acá, antes de cualquier
agregado (`src/data/dedupe.py`). Ojo con el detalle que hace falta para que sea
exacto: las filas duplicadas de un clon viven en **archivos distintos** (una
copia en la cohorte failed y otra en la not_failed), así que nunca caen en el
mismo chunk y un `drop_duplicates` por chunk no las ve. Por eso las filas de los
26 códigos clonados se juntan aparte y se deduplican enteras al final.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import load_config
from src.data.dedupe import (
    DEFAULT_DEDUPE_CONFIG,
    VEHICLE_COL,
    apply_canonical_ids,
    canonical_map,
    dedupe_vehicles,
)
from src.data.loader import DEFAULT_SOURCES_CONFIG, iter_table, load_table

logger = logging.getLogger(__name__)

DEFAULT_PANEL_CONFIG = "configs/data/panel_v1.yaml"

ID_COL = "vehicle_id"
COHORT_COL = "cohort"
EVENT_DAY_COL = "IdentificationDate"
EVENT_COLUMN = "event_observed"
STATIC_PREFIX = "static_"

# Agregados por vehículo: `salida: (columna, agregador por chunk, agregador al combinar)`.
# Dos agregadores porque el cálculo es en dos etapas: primero dentro de cada chunk,
# después sobre los parciales (un conteo se suma, un mínimo se re-minimiza).
AGG_SPECS: dict[str, dict[str, tuple[str, str, str]]] = {
    "trips": {
        "n_trips": ("OdometerTripEnd", "size", "sum"),
        "trip_odo_min": ("OdometerTripStart", "min", "min"),
        "trip_odo_max": ("OdometerTripEnd", "max", "max"),
        "trip_date_min": ("TripDatetimeStart", "min", "min"),
        "trip_date_max": ("TripDatetimeStart", "max", "max"),
        "trip_km_negative": ("_km_negative", "sum", "sum"),
    },
    "signals": {
        "n_signals": ("OdometerValue", "size", "sum"),
        "signal_odo_null": ("_odo_null", "sum", "sum"),
        "signal_odo_max": ("OdometerValue", "max", "max"),
        "signal_date_min": ("eventTimestamp", "min", "min"),
        "signal_date_max": ("eventTimestamp", "max", "max"),
        "n_regenerations": ("Regenerations", "count", "sum"),
    },
}


@dataclass
class TableScan:
    """Resultado de una pasada por una tabla cruda."""

    table: str
    n_rows: int
    n_rows_unique_raw: int
    n_rows_unique_canonical: int
    n_rows_clones: int
    n_rows_clones_unique: int
    quality: pd.DataFrame
    per_vehicle: pd.DataFrame
    vehicles: list[str] = field(default_factory=list)

    @property
    def n_duplicated_raw(self) -> int:
        """Filas idénticas bajo el mismo código (la misma fila en las dos cohortes)."""
        return self.n_rows - self.n_rows_unique_raw

    @property
    def n_duplicated_canonical(self) -> int:
        """Filas idénticas una vez colapsados los clones: lo que se lleva el dedupe."""
        return self.n_rows - self.n_rows_unique_canonical

    @property
    def n_duplicated_clones(self) -> int:
        """La parte de los duplicados que explican los 13 vehículos clonados."""
        return self.n_rows_clones - self.n_rows_clones_unique

    @property
    def n_duplicated_otros(self) -> int:
        """Duplicados exactos que NO son clones: filas repetidas en el archivo de origen."""
        return self.n_duplicated_canonical - self.n_duplicated_clones


def scan_table(
    name: str,
    *,
    config_path: str | Path = DEFAULT_SOURCES_CONFIG,
    dedupe_config: str | Path = DEFAULT_DEDUPE_CONFIG,
    chunksize: int = 500_000,
) -> TableScan:
    """Una pasada por la tabla: calidad por columna, duplicados y agregados por vehículo.

    Los duplicados se cuentan por hash de la fila completa **sin la columna de
    cohorte**, dos veces: con los códigos tal como vienen (`raw`) y con los códigos
    canónicos (`canonical`). La diferencia entre las dos es exactamente lo que
    aporta el colapso de clones.
    """
    mapping = canonical_map(dedupe_config)
    clone_codes = set(mapping) | set(mapping.values())
    spec = AGG_SPECS.get(name, {})

    n_rows = 0
    null_counts: pd.Series | None = None
    dtypes: pd.Series | None = None
    hashes_raw: list[np.ndarray] = []
    hashes_canonical: list[np.ndarray] = []
    partials: list[pd.DataFrame] = []
    clone_rows: list[pd.DataFrame] = []
    vehicles: set[str] = set()

    for chunk in iter_table(name, config_path, chunksize=chunksize):
        n_rows += len(chunk)
        nulls = chunk.isna().sum()
        null_counts = nulls if null_counts is None else null_counts.add(nulls, fill_value=0)
        if dtypes is None:
            dtypes = chunk.dtypes.astype(str)

        payload = chunk.drop(columns=[COHORT_COL], errors="ignore")
        hashes_raw.append(pd.util.hash_pandas_object(payload, index=False).to_numpy())

        chunk[VEHICLE_COL] = chunk[VEHICLE_COL].replace(mapping)
        payload = chunk.drop(columns=[COHORT_COL], errors="ignore")
        hashes_canonical.append(pd.util.hash_pandas_object(payload, index=False).to_numpy())
        vehicles.update(chunk[VEHICLE_COL].unique().tolist())

        is_clone = chunk[VEHICLE_COL].isin(clone_codes)
        if is_clone.any():
            clone_rows.append(chunk.loc[is_clone].copy())
        rest = chunk.loc[~is_clone]
        if spec and len(rest):
            partials.append(_aggregate(rest, spec, stage="chunk"))

    if n_rows == 0:
        raise ValueError(f"Tabla `{name}`: no se leyó ninguna fila")

    # Las copias del clon están en archivos distintos: se deduplican enteras, no por
    # chunk. `apply_canonical_ids` es el mismo dedupe que va a usar F2.
    clones = (
        apply_canonical_ids(pd.concat(clone_rows, ignore_index=True), mapping, cohort_col=COHORT_COL)
        if clone_rows
        else pd.DataFrame()
    )
    n_rows_clones = int(sum(len(part) for part in clone_rows))

    per_vehicle = pd.DataFrame()
    if spec:
        if len(clones):
            partials.append(_aggregate(clones, spec, stage="chunk"))
        per_vehicle = _aggregate(pd.concat(partials), spec, stage="combine").reset_index()
        per_vehicle = per_vehicle.rename(columns={VEHICLE_COL: ID_COL})

    quality = pd.DataFrame(
        {
            "table": name,
            "column": null_counts.index,
            "dtype": [dtypes.get(c, "?") for c in null_counts.index],
            "n_null": null_counts.astype(int).to_numpy(),
            "null_frac": (null_counts / n_rows).to_numpy(),
        }
    ).reset_index(drop=True)

    scan = TableScan(
        table=name,
        n_rows=n_rows,
        n_rows_unique_raw=int(len(np.unique(np.concatenate(hashes_raw)))),
        n_rows_unique_canonical=int(len(np.unique(np.concatenate(hashes_canonical)))),
        n_rows_clones=n_rows_clones,
        n_rows_clones_unique=int(len(clones)),
        quality=quality,
        per_vehicle=per_vehicle,
        vehicles=sorted(vehicles),
    )
    logger.info(
        "Scan `%s`: %d filas, %d vehículos, %d filas duplicadas tras el dedupe",
        name,
        scan.n_rows,
        len(scan.vehicles),
        scan.n_duplicated_canonical,
    )
    return scan


def scan_raw_tables(
    *,
    config_path: str | Path = DEFAULT_SOURCES_CONFIG,
    dedupe_config: str | Path = DEFAULT_DEDUPE_CONFIG,
    chunksize: int = 500_000,
    tables: list[str] | None = None,
) -> dict[str, TableScan]:
    """Escanea las tres tablas crudas (una pasada cada una)."""
    names = tables if tables is not None else list(load_config(config_path)["tables"])
    return {
        name: scan_table(
            name, config_path=config_path, dedupe_config=dedupe_config, chunksize=chunksize
        )
        for name in names
    }


def load_vehicle_static(
    *,
    config_path: str | Path = DEFAULT_SOURCES_CONFIG,
    dedupe_config: str | Path = DEFAULT_DEDUPE_CONFIG,
    panel_config: str | Path = DEFAULT_PANEL_CONFIG,
    extra_static_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Tabla estática deduplicada: una fila por vehículo real, con `event_observed`.

    Las columnas estáticas se renombran a `static_*` según el contrato, y salen
    declaradas del YAML del panel (`features.static_columns` + `static_excluded`),
    no de una lista escrita acá. `Engine` viene incluida a propósito aunque esté
    excluida del set base: hace falta para **reportar** el balance del split, no
    para modelar (ver `docs/reproducibilidad.md`).

    `IdentificationDate` se conserva como `event_day_since_production` porque es la
    etiqueta —nunca una feature—: con ella se deriva `event_observed`.

    `extra_static_columns` suma estáticas que el panel no declara (p. ej. `SalesCity`
    de la entrega v2) para reportarlas o estratificar con ellas, sin tocar el panel.
    """
    vehicles = load_table("vehicles", config_path)
    deduped = dedupe_vehicles(vehicles, config_path=dedupe_config)

    static_columns = _declared_static_columns(panel_config)
    static_columns += [c for c in (extra_static_columns or []) if c not in static_columns]
    out = deduped.rename(columns={VEHICLE_COL: ID_COL})
    out[EVENT_COLUMN] = out[EVENT_DAY_COL].notna().astype(int)
    out = out.rename(columns={EVENT_DAY_COL: "event_day_since_production"})
    present = [c for c in static_columns if c in out.columns]
    missing = [c for c in static_columns if c not in out.columns]
    if missing:
        logger.warning("Columnas estáticas declaradas y ausentes en el crudo: %s", missing)
    out = out.rename(columns={c: f"{STATIC_PREFIX}{c}" for c in present})

    ordered = [ID_COL, COHORT_COL, EVENT_COLUMN, "event_day_since_production"]
    ordered += [f"{STATIC_PREFIX}{c}" for c in present]
    ordered += [c for c in out.columns if c not in ordered]
    return out[ordered].sort_values(ID_COL, ignore_index=True)


def build_vehicle_table(
    static: pd.DataFrame, scans: dict[str, TableScan], *, id_col: str = ID_COL
) -> pd.DataFrame:
    """Une la estática deduplicada con los agregados por vehículo de `trips` y `signals`.

    Una fila por `vehicle_id`. Las banderas `in_trips`/`in_signals` son la cobertura
    del join: si alguna queda en 0, el vehículo existe en una tabla y no en la otra.
    """
    out = static.copy()
    for name, scan in scans.items():
        if scan.per_vehicle.empty:
            continue
        out = out.merge(scan.per_vehicle, on=id_col, how="outer")
        out[f"in_{name}"] = out[id_col].isin(scan.vehicles).astype(int)
    return out.sort_values(id_col, ignore_index=True)


def attach_static(
    frame: pd.DataFrame,
    static: pd.DataFrame,
    *,
    columns: list[str] | None = None,
    id_col: str = VEHICLE_COL,
    validate: bool = True,
) -> pd.DataFrame:
    """Left join 1-a-muchos: le pega las estáticas del vehículo a `trips` o a `signals`.

    La tabla sale con **exactamente las mismas filas** con las que entró, solo con
    columnas nuevas pegadas: es un enriquecimiento, no una fusión. Los ids de
    `frame` tienen que venir ya canonizados (`canonical_map`), que es lo que hace
    `iter_enriched_table`; si no, los códigos clonados quedarían huérfanos —y con
    `validate=True` eso falla ruidosamente en vez de dejar filas con estáticas
    vacías.
    """
    payload = static if columns is None else static[[ID_COL, *columns]]
    out = frame.rename(columns={id_col: ID_COL}) if id_col != ID_COL else frame

    # Sin esto, `cohort` (que está en las dos) saldría como `cohort_x`/`cohort_y`.
    # Gana la de la tabla grande —la cohorte del archivo del que salió la fila—; la
    # etiqueta resuelta ya viaja como `event_observed`.
    chocan = [c for c in payload.columns if c != ID_COL and c in out.columns]
    if chocan:
        logger.info("attach_static: columnas ya presentes, no se pegan: %s", chocan)
        payload = payload.drop(columns=chocan)

    n_orphans = orphan_rows(out, payload)
    if n_orphans and validate:
        raise ValueError(
            f"{n_orphans} fila(s) sin estática: hay `{ID_COL}` en la tabla que no están "
            "en `vehicles`. ¿Los ids vienen sin canonizar?"
        )

    n_before = len(out)
    merged = out.merge(payload, on=ID_COL, how="left", validate="m:1")
    if len(merged) != n_before:
        raise ValueError(
            f"El join cambió el número de filas ({n_before} -> {len(merged)}): la estática "
            "tiene más de una fila por vehículo."
        )
    return merged


def orphan_rows(frame: pd.DataFrame, static: pd.DataFrame, *, id_col: str = ID_COL) -> int:
    """Filas cuyo vehículo no existe en la tabla estática."""
    return int((~frame[id_col].isin(set(static[id_col]))).sum())


def iter_enriched_table(
    name: str,
    static: pd.DataFrame,
    *,
    config_path: str | Path = DEFAULT_SOURCES_CONFIG,
    dedupe_config: str | Path = DEFAULT_DEDUPE_CONFIG,
    chunksize: int = 500_000,
    columns: list[str] | None = None,
    static_columns: list[str] | None = None,
    drop_clone_duplicates: bool = True,
) -> Iterator[pd.DataFrame]:
    """`trips` o `signals` con las estáticas pegadas, por chunks y con ids canónicos.

    Cada chunk sale con las filas con las que entró. El único lugar donde cambia el
    total es el colapso de los clones: sus copias viven en archivos distintos, así
    que las filas de los 26 códigos clonados se retienen y se emiten deduplicadas
    como último chunk (`drop_clone_duplicates=False` las deja pasar tal cual).
    """
    mapping = canonical_map(dedupe_config)
    clone_codes = set(mapping) | set(mapping.values())
    retenidas: list[pd.DataFrame] = []

    for chunk in iter_table(name, config_path, chunksize=chunksize, columns=columns):
        chunk[VEHICLE_COL] = chunk[VEHICLE_COL].replace(mapping)
        if drop_clone_duplicates:
            is_clone = chunk[VEHICLE_COL].isin(clone_codes)
            if is_clone.any():
                retenidas.append(chunk.loc[is_clone].copy())
            chunk = chunk.loc[~is_clone]
        if len(chunk):
            yield attach_static(chunk, static, columns=static_columns)

    if retenidas:
        clones = apply_canonical_ids(
            pd.concat(retenidas, ignore_index=True), mapping, cohort_col=COHORT_COL
        )
        yield attach_static(clones, static, columns=static_columns)


def enrichment_report(
    static: pd.DataFrame,
    scans: dict[str, TableScan],
    *,
    config_path: str | Path = DEFAULT_SOURCES_CONFIG,
    dedupe_config: str | Path = DEFAULT_DEDUPE_CONFIG,
    chunksize: int = 500_000,
    tables: tuple[str, ...] = ("trips", "signals"),
    static_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Verifica el enriquecimiento de cada tabla grande: filas que entran, filas que salen.

    Es la validación fila a fila que complementa a `join_coverage` (que mira ids):
    si `VehicleCode` cierra sin huérfanos, `filas_esperadas` y `filas_enriquecidas`
    tienen que ser iguales, y `huerfanas` cero.
    """
    rows = []
    for name in tables:
        scan = scans[name]
        n_out = 0
        n_orphans = 0
        n_cols_in = len(scan.quality)
        n_cols_out = n_cols_in
        for chunk in iter_enriched_table(
            name,
            static,
            config_path=config_path,
            dedupe_config=dedupe_config,
            chunksize=chunksize,
            static_columns=static_columns,
        ):
            n_out += len(chunk)
            n_orphans += orphan_rows(chunk, static)
            n_cols_out = chunk.shape[1]
        rows.append(
            {
                "table": name,
                "filas_crudas": scan.n_rows,
                "filas_esperadas": scan.n_rows - scan.n_duplicated_clones,
                "filas_enriquecidas": n_out,
                "huerfanas": n_orphans,
                "columnas": f"{n_cols_in} -> {n_cols_out}",
            }
        )
    return pd.DataFrame(rows)


def join_coverage(static: pd.DataFrame, scans: dict[str, TableScan], *, id_col: str = ID_COL) -> pd.DataFrame:
    """Cobertura del join entre las tres tablas, con los códigos ya canonizados."""
    sets = {"vehicles": set(static[id_col])}
    sets.update({name: set(scan.vehicles) for name, scan in scans.items() if name != "vehicles"})

    rows = [{"check": f"vehículos en `{name}`", "n": len(ids)} for name, ids in sets.items()]
    rows.append({"check": "vehículos en las tres tablas", "n": len(set.intersection(*sets.values()))})
    for name, ids in sets.items():
        if name == "vehicles":
            continue
        rows.append({"check": f"en `vehicles` y no en `{name}`", "n": len(sets["vehicles"] - ids)})
        rows.append({"check": f"en `{name}` y no en `vehicles`", "n": len(ids - sets["vehicles"])})
    return pd.DataFrame(rows)


def quality_report(scans: dict[str, TableScan]) -> pd.DataFrame:
    """Nulos y tipo por columna, para las tablas escaneadas."""
    frames = [scan.quality for scan in scans.values()]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def duplicate_report(scans: dict[str, TableScan]) -> pd.DataFrame:
    """Filas duplicadas, separando las que explican los clones de las que no.

    `dup_mismo_codigo` son filas idénticas bajo el mismo `VehicleCode` (la misma
    fila publicada en las dos cohortes). `dup_clones` es lo que colapsa el dedupe de
    los 13 vehículos duplicados. `dup_otros` es el remanente: filas exactamente
    repetidas dentro de un mismo archivo, que el dedupe de vehículos no toca.
    """
    return pd.DataFrame(
        [
            {
                "table": scan.table,
                "n_rows": scan.n_rows,
                "dup_mismo_codigo": scan.n_duplicated_raw,
                "dup_total_canonico": scan.n_duplicated_canonical,
                "dup_clones": scan.n_duplicated_clones,
                "dup_otros": scan.n_duplicated_otros,
                "dup_otros_frac": scan.n_duplicated_otros / scan.n_rows,
            }
            for scan in scans.values()
        ]
    )


def anchor_offset_days(
    vehicle_table: pd.DataFrame,
    *,
    first_trip_col: str = "trip_date_min",
    production_day_col: str = f"{STATIC_PREFIX}ProductionDay",
) -> pd.Series:
    """`primer_viaje − ProductionDay` en días: el puente entre el eje de días y el calendario.

    Si `ProductionDay` está en el mismo eje que el calendario con un origen común a
    la flota, esta diferencia es casi constante entre vehículos (F1 midió IQR de 0
    días). Devuelve la serie para que el llamador reporte su dispersión; el origen
    concreto lo congela F2, acá solo se verifica que el puente sigue en pie.
    Ver `docs/reproducibilidad.md`.
    """
    first_trip = pd.to_datetime(vehicle_table[first_trip_col], utc=True, errors="coerce")
    day = first_trip.dt.tz_convert("UTC").dt.floor("D")
    days_since_epoch = (day - pd.Timestamp("1970-01-01", tz="UTC")).dt.days
    return days_since_epoch - vehicle_table[production_day_col]


def _declared_static_columns(panel_config: str | Path) -> list[str]:
    """Estáticas declaradas en el YAML del panel: las del set base más las excluidas."""
    features = (load_config(panel_config).get("features") or {})
    columns = list(features.get("static_columns") or [])
    columns += [c for c in (features.get("static_excluded") or []) if c not in columns]
    return columns


def _aggregate(
    frame: pd.DataFrame, spec: dict[str, tuple[str, str, str]], *, stage: str
) -> pd.DataFrame:
    """Agrega por vehículo en la etapa pedida (`chunk` sobre crudo, `combine` sobre parciales)."""
    if stage == "chunk":
        frame = _with_derived(frame)
        named = {out: pd.NamedAgg(column=src, aggfunc=agg) for out, (src, agg, _) in spec.items()}
        grouped = frame.groupby(VEHICLE_COL, observed=True)
    else:
        named = {out: pd.NamedAgg(column=out, aggfunc=agg) for out, (_, _, agg) in spec.items()}
        grouped = frame.groupby(level=0, observed=True)
    return grouped.agg(**named)


def _with_derived(frame: pd.DataFrame) -> pd.DataFrame:
    """Columnas auxiliares que los agregados necesitan y el crudo no trae."""
    out = frame
    if "OdometerTripEnd" in out.columns and "OdometerTripStart" in out.columns:
        out = out.assign(
            _km_negative=(out["OdometerTripEnd"] - out["OdometerTripStart"] < 0).astype(int)
        )
    if "OdometerValue" in out.columns:
        out = out.assign(_odo_null=out["OdometerValue"].isna().astype(int))
    return out
