"""Formato de números y fechas para textos en castellano: coma decimal, punto de miles.

Es el mismo estilo que `src/eval/explain.py::format_value` (el de los mensajes de la
explicabilidad), sin importar ese módulo: arrastra sklearn y el registry de modelos.
"""

from __future__ import annotations

import math
from datetime import date, datetime

import pandas as pd

MONTHS = ("enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre",
          "octubre", "noviembre", "diciembre")

#: formato → (escala, decimales, unidad). Los de la explicabilidad más los del taller.
FORMATS = {"pct": (100.0, 0, "%"), "km1": (1.0, 1, " km"), "kmh0": (1.0, 0, " km/h"), "degc0": (1.0, 0, " °C"),
           "per1000": (1.0, 0, " cada 1.000 km"), "int": (1.0, 0, ""), "min0": (1.0, 0, " min"),
           "pp1": (100.0, 1, " pp"), "dec1": (1.0, 1, ""), "int_km": (1.0, 0, " km")}


def number(value: float, digits: int = 0) -> str:
    """`12.345,6`: punto de miles y coma decimal."""
    return f"{value:,.{digits}f}".replace(",", "_").replace(".", ",").replace("_", ".")


def format_value(value: float, fmt: str) -> str:
    """El valor con su unidad; `—` si falta."""
    if value is None or not math.isfinite(value):
        return "—"
    scale, digits, unit = FORMATS.get(fmt, (1.0, 2, ""))
    return number(value * scale, digits) + unit


def long_date(value: date | datetime | pd.Timestamp | str) -> str:
    """`2 de octubre de 2025`."""
    ts = pd.Timestamp(value)
    return f"{ts.day} de {MONTHS[ts.month - 1]} de {ts.year}"


def short_date(value: date | datetime | pd.Timestamp | str) -> str:
    """`02-10-2025`."""
    return pd.Timestamp(value).strftime("%d-%m-%Y")
