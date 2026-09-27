"""Carga de las tres tablas crudas (viajes, señales dinámicas, info de vehículo).

Nada de rutas ni nombres de columna hardcodeados: todo sale de un YAML
(`configs/data/raw_sources.yaml` por defecto), para que cuando F1 audite los
archivos reales solo haya que tocar el config.

El esquema declarado en ese YAML es *provisional* hasta que cierre F1: el loader
no rompe si una columna declarada no aparece, pero lo reporta en
`describe_tables()` para que el desvío quede a la vista y no se descubra tarde.

## Transformaciones por parte (entrega v2, 26-09-2026)

La segunda entrega de Ford no trae las seis partes con el mismo esquema, y cada
desvío se corrige acá, declarado en el YAML, para que ningún consumidor tenga que
acordarse (`docs/memoria/f9-entrega-v2.md`):

- `rename`: la estática de fallados llama `IdentificationDaysSinceProduction` a lo
  que la de sanos llama `IdentificationDate`.
- `offsets`: el `ProductionDay` de la estática de fallados está contado desde otro
  origen (+538 días). Sin el corrimiento, el anclaje al calendario se rompe solo
  para una cohorte.
- `truncate_after` (por tabla): los fallados se extrajeron diez días después que
  los sanos. Todo lo posterior al fin de la extracción de los sanos se descarta,
  para que ninguna cohorte tenga telemetría que la otra no puede tener.

Una parte se puede declarar como un path (formato viejo) o como un mapping con
`path` y esas claves. Sin ellas, el comportamiento es el de siempre.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import load_config, resolve_path

logger = logging.getLogger(__name__)

DEFAULT_SOURCES_CONFIG = "configs/data/raw_sources.yaml"

_READERS = {
    ".csv": pd.read_csv,
    ".txt": pd.read_csv,
    ".tsv": lambda p, **kw: pd.read_csv(p, sep="\t", **kw),
    ".parquet": pd.read_parquet,
    ".xlsx": pd.read_excel,
    ".xls": pd.read_excel,
}


@dataclass(frozen=True)
class PartSpec:
    """Una parte (archivo) de una tabla lógica, con las correcciones que necesita."""

    name: str | None
    path: Path
    # columna tal como viene en el archivo -> nombre del contrato
    rename: dict[str, str] = field(default_factory=dict)
    # columna del contrato -> corrimiento que se suma después del cast
    offsets: dict[str, float] = field(default_factory=dict)

    def raw_name(self, column: str) -> str:
        """Nombre en el archivo de una columna del contrato (para `usecols`)."""
        inverse = {canonical: raw for raw, canonical in self.rename.items()}
        return inverse.get(column, column)


def load_sources_config(config_path: str | Path = DEFAULT_SOURCES_CONFIG) -> dict[str, Any]:
    """Config de fuentes crudas: `{tables: {nombre: {path, dtypes, date_columns, ...}}}`."""
    cfg = load_config(config_path)
    if "tables" not in cfg:
        raise ValueError(f"El config {config_path} no declara la clave `tables`")
    return cfg


def load_table(
    name: str,
    config_path: str | Path = DEFAULT_SOURCES_CONFIG,
    *,
    nrows: int | None = None,
) -> pd.DataFrame:
    """Carga una tabla cruda por nombre, con dtypes explícitos y fechas parseadas.

    Args:
        name: clave de la tabla en el config (`trips`, `signals`, `vehicles`).
        config_path: YAML de fuentes crudas.
        nrows: si se pasa, lee solo las primeras N filas (para auditoría rápida).
    """
    cfg = load_sources_config(config_path)
    tables = cfg["tables"]
    if name not in tables:
        raise KeyError(f"Tabla `{name}` no declarada. Disponibles: {sorted(tables)}")

    spec = tables[name]
    read_kwargs = {**(cfg.get("defaults", {}) or {}).get("read_kwargs", {}),
                   **(spec.get("read_kwargs", {}) or {})}
    part_column = spec.get("part_column", cfg.get("part_column", "cohort"))

    declared_dtypes: dict[str, str] = spec.get("dtypes", {}) or {}
    date_columns: list[str] = spec.get("date_columns", []) or []

    frames = []
    parts = _table_parts(spec, table=name)
    for part in parts:
        frame = _read_one(part.path, table=name, read_kwargs=read_kwargs, nrows=nrows)
        frame = _apply_part_columns(frame, part, table=name)
        if part.name is not None:
            frame[part_column] = part.name
        frame = _apply_dtypes(frame, declared_dtypes, table=name)
        frame = _apply_offsets(frame, part, table=name)
        frames.append(frame)
    df = frames[0] if len(frames) == 1 else pd.concat(frames, ignore_index=True)

    df = _parse_dates(df, date_columns, table=name, fmt=spec.get("date_format"))
    df = _truncate(df, spec, table=name)
    logger.info(
        "Tabla `%s`: %d filas x %d columnas desde %s",
        name,
        len(df),
        df.shape[1],
        ", ".join(str(part.path) for part in parts),
    )
    return df


def iter_table(
    name: str,
    config_path: str | Path = DEFAULT_SOURCES_CONFIG,
    *,
    chunksize: int = 500_000,
    columns: list[str] | None = None,
) -> Iterator[pd.DataFrame]:
    """Itera una tabla por chunks, con los mismos dtypes/fechas/`cohort` que `load_table`.

    `trips` y `signals` suman 13M de filas: agregarlas por ventana (F2) o perfilarlas
    (EDA) no entra cómodo en memoria de una sola vez. `columns` limita la lectura a
    las columnas que se van a usar, con los nombres **del contrato** (el loader los
    traduce a los del archivo de cada parte).
    """
    cfg = load_sources_config(config_path)
    tables = cfg["tables"]
    if name not in tables:
        raise KeyError(f"Tabla `{name}` no declarada. Disponibles: {sorted(tables)}")

    spec = tables[name]
    read_kwargs = {
        **(cfg.get("defaults", {}) or {}).get("read_kwargs", {}),
        **(spec.get("read_kwargs", {}) or {}),
    }
    part_column = spec.get("part_column", cfg.get("part_column", "cohort"))
    all_dtypes: dict[str, str] = spec.get("dtypes", {}) or {}
    all_dates: list[str] = spec.get("date_columns", []) or []

    truncate_column = (spec.get("truncate_after") or {}).get("column")
    for part in _table_parts(spec, table=name):
        if part.path.suffix.lower() not in {".csv", ".txt", ".tsv"}:
            raise ValueError(f"`iter_table` solo soporta CSV/TSV; `{name}` es {part.path.suffix}")
        kwargs = dict(read_kwargs)
        declared_dtypes, date_columns = all_dtypes, all_dates
        extra: list[str] = []
        if columns is not None:
            wanted = [c for c in columns if c != part_column]
            # El recorte por fin de extracción necesita su fecha aunque no se la pida:
            # se lee, se recorta y se descarta, para que ninguna lectura parcial lo esquive.
            if truncate_column and truncate_column not in wanted:
                extra = [truncate_column]
            kwargs["usecols"] = [part.raw_name(c) for c in wanted + extra]
            # Sin esto, cada chunk avisa por cada columna declarada que no pedimos.
            declared_dtypes = {c: t for c, t in all_dtypes.items() if c in wanted}
            date_columns = [c for c in all_dates if c in wanted + extra]
        if not part.path.exists():
            _read_one(part.path, table=name, read_kwargs=kwargs, nrows=0)  # levanta el error de siempre
        for chunk in pd.read_csv(part.path, chunksize=chunksize, **kwargs):
            chunk = _apply_part_columns(chunk, part, table=name)
            if part.name is not None:
                chunk[part_column] = part.name
            chunk = _apply_dtypes(chunk, declared_dtypes, table=name)
            chunk = _apply_offsets(chunk, part, table=name)
            chunk = _parse_dates(chunk, date_columns, table=name, fmt=spec.get("date_format"))
            chunk = _truncate(chunk, spec, table=name, quiet=True)
            if extra:
                chunk = chunk.drop(columns=extra)
            yield chunk


def load_raw_tables(
    config_path: str | Path = DEFAULT_SOURCES_CONFIG,
    *,
    nrows: int | None = None,
) -> dict[str, pd.DataFrame]:
    """Carga todas las tablas declaradas en el config, en un dict por nombre."""
    cfg = load_sources_config(config_path)
    return {name: load_table(name, config_path, nrows=nrows) for name in cfg["tables"]}


def describe_tables(
    config_path: str | Path = DEFAULT_SOURCES_CONFIG,
    *,
    nrows: int | None = None,
) -> pd.DataFrame:
    """Resumen por tabla para la auditoría de F1: filas, columnas, faltantes y extras.

    `missing_declared` son columnas que el contrato provisional espera y el archivo
    no trae; `undeclared` son columnas del archivo que nadie documentó todavía.
    """
    cfg = load_sources_config(config_path)
    rows = []
    for name, spec in cfg["tables"].items():
        try:
            df = load_table(name, config_path, nrows=nrows)
        except FileNotFoundError as exc:
            rows.append({"table": name, "status": "missing_file", "detail": str(exc).splitlines()[0]})
            continue
        declared = set(spec.get("dtypes", {}) or {}) | set(spec.get("date_columns", []) or [])
        present = set(df.columns)
        rows.append(
            {
                "table": name,
                "status": "ok",
                "n_rows": len(df),
                "n_cols": df.shape[1],
                "missing_declared": sorted(declared - present),
                "undeclared": sorted(present - declared),
                "null_frac_max": float(df.isna().mean().max()) if df.shape[1] else 0.0,
            }
        )
    return pd.DataFrame(rows)


def table_parts(
    name: str, config_path: str | Path = DEFAULT_SOURCES_CONFIG
) -> list[PartSpec]:
    """Las partes declaradas de una tabla, con sus correcciones (para auditorías)."""
    tables = load_sources_config(config_path)["tables"]
    if name not in tables:
        raise KeyError(f"Tabla `{name}` no declarada. Disponibles: {sorted(tables)}")
    return _table_parts(tables[name], table=name)


def _table_parts(spec: dict[str, Any], *, table: str) -> list[PartSpec]:
    """Partes de una tabla: `parts: {cohorte: path | {path, rename, offsets}}` o un `path`.

    Los datos crudos vienen partidos por cohorte de muestreo (failed / not_failed);
    declararlas como `parts` mantiene una sola tabla lógica río abajo.
    """
    parts = spec.get("parts")
    if parts:
        out = []
        for part_name, part in parts.items():
            if isinstance(part, (str, Path)):
                out.append(PartSpec(name=str(part_name), path=resolve_path(part)))
                continue
            if not isinstance(part, dict) or "path" not in part:
                raise ValueError(f"Tabla `{table}`, parte `{part_name}`: se espera un path o un mapping con `path`")
            unknown = set(part) - {"path", "rename", "offsets"}
            if unknown:
                raise ValueError(f"Tabla `{table}`, parte `{part_name}`: claves desconocidas {sorted(unknown)}")
            out.append(
                PartSpec(
                    name=str(part_name),
                    path=resolve_path(part["path"]),
                    rename=dict(part.get("rename") or {}),
                    offsets={str(k): float(v) for k, v in (part.get("offsets") or {}).items()},
                )
            )
        return out
    if spec.get("path"):
        return [PartSpec(name=None, path=resolve_path(spec["path"]))]
    raise ValueError(f"Tabla `{table}`: el config no declara ni `path` ni `parts`")


def _read_one(
    path: Path, *, table: str, read_kwargs: dict[str, Any], nrows: int | None
) -> pd.DataFrame:
    """Lee un archivo suelto, con el reader que corresponda a su extensión."""
    if not path.exists():
        raise FileNotFoundError(
            f"No se encuentra el archivo de `{table}`: {path}\n"
            "Copiá los datos crudos a `data/raw/` (o apuntá FORD_DATA_DIR / el config "
            "a donde estén). No se versionan: `data/` está en .gitignore."
        )

    reader = _READERS.get(path.suffix.lower())
    if reader is None:
        raise ValueError(f"Extensión no soportada para `{table}`: {path.suffix}")

    kwargs: dict[str, Any] = dict(read_kwargs)
    if path.suffix.lower() in {".csv", ".txt", ".tsv"}:
        # Los dtypes se aplican después de leer: si una columna declarada no existe
        # o trae basura, queremos el aviso explícito, no un ValueError del parser.
        if nrows is not None:
            kwargs["nrows"] = nrows
        return reader(path, **kwargs)

    df = reader(path, **kwargs)
    return df.head(nrows) if nrows is not None else df


def _apply_part_columns(frame: pd.DataFrame, part: PartSpec, *, table: str) -> pd.DataFrame:
    """Renombra las columnas de la parte a los nombres del contrato."""
    return frame.rename(columns=part.rename) if part.rename else frame


def _apply_dtypes(df: pd.DataFrame, dtypes: dict[str, str], *, table: str) -> pd.DataFrame:
    for column, dtype in dtypes.items():
        if column not in df.columns:
            logger.warning("Tabla `%s`: columna declarada ausente: %s", table, column)
            continue
        try:
            if dtype.startswith(("float", "int", "Int", "Float")):
                df[column] = pd.to_numeric(df[column], errors="coerce").astype(dtype)
            else:
                df[column] = df[column].astype(dtype)
        except (TypeError, ValueError) as exc:
            logger.warning("Tabla `%s`: no se pudo castear %s a %s (%s)", table, column, dtype, exc)
    return df


def _apply_offsets(frame: pd.DataFrame, part: PartSpec, *, table: str) -> pd.DataFrame:
    """Suma los corrimientos declarados de la parte (después del cast numérico)."""
    for column, value in part.offsets.items():
        if column not in frame.columns:
            continue
        frame[column] = frame[column] + value
    return frame


def _parse_dates(
    df: pd.DataFrame, date_columns: list[str], *, table: str, fmt: str | None = None
) -> pd.DataFrame:
    for column in date_columns:
        if column not in df.columns:
            logger.warning("Tabla `%s`: columna de fecha declarada ausente: %s", table, column)
            continue
        df[column] = pd.to_datetime(df[column], format=fmt, errors="coerce")
    return df


def _truncate(
    frame: pd.DataFrame, spec: dict[str, Any], *, table: str, quiet: bool = False
) -> pd.DataFrame:
    """Descarta las filas posteriores al fin de extracción común (`truncate_after`)."""
    rule = spec.get("truncate_after")
    if not rule:
        return frame
    column = rule["column"]
    if column not in frame.columns:
        # Una lectura con `columns` que no pidió la fecha no se puede recortar: se avisa.
        logger.warning(
            "Tabla `%s`: `truncate_after` necesita `%s` y la lectura no la trae; filas SIN recortar",
            table, column,
        )
        return frame
    at = pd.Timestamp(rule["at"])
    values = frame[column]
    if getattr(values.dt, "tz", None) is None and at.tzinfo is not None:
        at = at.tz_convert(None)
    keep = ~(values > at)
    n_drop = int((~keep).sum())
    if n_drop and not quiet:
        logger.info("Tabla `%s`: %d filas posteriores a %s descartadas (fin de extracción)", table, n_drop, at)
    return frame.loc[keep] if n_drop else frame
