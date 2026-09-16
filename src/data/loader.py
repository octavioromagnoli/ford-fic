"""Carga de las tres tablas crudas (viajes, señales dinámicas, info de vehículo).

Nada de rutas ni nombres de columna hardcodeados: todo sale de un YAML
(`configs/data/raw_sources.yaml` por defecto), para que cuando F1 audite los
archivos reales solo haya que tocar el config.

El esquema declarado en ese YAML es *provisional* hasta que cierre F1: el loader
no rompe si una columna declarada no aparece, pero lo reporta en
`describe_tables()` para que el desvío quede a la vista y no se descubra tarde.
"""

from __future__ import annotations

import logging
from pathlib import Path
from collections.abc import Iterator
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
    for part_name, part_path in parts:
        frame = _read_one(part_path, table=name, read_kwargs=read_kwargs, nrows=nrows)
        if part_name is not None:
            frame[part_column] = part_name
        frames.append(frame)
    df = frames[0] if len(frames) == 1 else pd.concat(frames, ignore_index=True)

    df = _apply_dtypes(df, declared_dtypes, table=name)
    df = _parse_dates(df, date_columns, table=name, fmt=spec.get("date_format"))
    logger.info(
        "Tabla `%s`: %d filas x %d columnas desde %s",
        name,
        len(df),
        df.shape[1],
        ", ".join(str(part_path) for _, part_path in parts),
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
    las columnas que se van a usar.
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
    declared_dtypes: dict[str, str] = spec.get("dtypes", {}) or {}
    date_columns: list[str] = spec.get("date_columns", []) or []

    for part_name, part_path in _table_parts(spec, table=name):
        if part_path.suffix.lower() not in {".csv", ".txt", ".tsv"}:
            raise ValueError(f"`iter_table` solo soporta CSV/TSV; `{name}` es {part_path.suffix}")
        kwargs = dict(read_kwargs)
        if columns is not None:
            wanted = [c for c in columns if c != part_column]
            kwargs["usecols"] = wanted
            # Sin esto, cada chunk avisa por cada columna declarada que no pedimos.
            declared_dtypes = {c: t for c, t in declared_dtypes.items() if c in wanted}
            date_columns = [c for c in date_columns if c in wanted]
        for chunk in pd.read_csv(part_path, chunksize=chunksize, **kwargs):
            if part_name is not None:
                chunk[part_column] = part_name
            chunk = _apply_dtypes(chunk, declared_dtypes, table=name)
            chunk = _parse_dates(chunk, date_columns, table=name, fmt=spec.get("date_format"))
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


def _table_parts(spec: dict[str, Any], *, table: str) -> list[tuple[str | None, Path]]:
    """Partes de una tabla: `parts: {cohorte: path}` o un único `path`.

    Los datos crudos vienen partidos por cohorte de muestreo (failed / not_failed);
    declararlas como `parts` mantiene una sola tabla lógica río abajo.
    """
    parts = spec.get("parts")
    if parts:
        return [(str(part_name), resolve_path(part_path)) for part_name, part_path in parts.items()]
    if spec.get("path"):
        return [(None, resolve_path(spec["path"]))]
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


def _parse_dates(
    df: pd.DataFrame, date_columns: list[str], *, table: str, fmt: str | None = None
) -> pd.DataFrame:
    for column in date_columns:
        if column not in df.columns:
            logger.warning("Tabla `%s`: columna de fecha declarada ausente: %s", table, column)
            continue
        df[column] = pd.to_datetime(df[column], format=fmt, errors="coerce")
    return df
