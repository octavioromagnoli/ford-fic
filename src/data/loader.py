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
    path = resolve_path(spec["path"])
    if not path.exists():
        raise FileNotFoundError(
            f"No se encuentra el archivo de `{name}`: {path}\n"
            "Copiá los datos crudos a `data/raw/` (o apuntá FORD_DATA_DIR / el config "
            "a donde estén). No se versionan: `data/` está en .gitignore."
        )

    reader = _READERS.get(path.suffix.lower())
    if reader is None:
        raise ValueError(f"Extensión no soportada para `{name}`: {path.suffix}")

    kwargs: dict[str, Any] = {}
    declared_dtypes: dict[str, str] = spec.get("dtypes", {}) or {}
    date_columns: list[str] = spec.get("date_columns", []) or []

    if path.suffix.lower() in {".csv", ".txt", ".tsv"}:
        # Los dtypes se aplican después de leer: si una columna declarada no existe
        # o trae basura, queremos el aviso explícito, no un ValueError del parser.
        if nrows is not None:
            kwargs["nrows"] = nrows
        df = reader(path, **kwargs)
    else:
        df = reader(path)
        if nrows is not None:
            df = df.head(nrows)

    df = _apply_dtypes(df, declared_dtypes, table=name)
    df = _parse_dates(df, date_columns, table=name, fmt=spec.get("date_format"))
    logger.info("Tabla `%s`: %d filas x %d columnas desde %s", name, len(df), df.shape[1], path)
    return df


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
