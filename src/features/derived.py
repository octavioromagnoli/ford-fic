"""Features derivadas: relaciones ENTRE features de ventana, no agregados de una columna.

La primitiva de `windows.py` contesta "¿cuánto de X hay en la ventana?". Hay preguntas
que no se pueden escribir así porque son un cociente de dos agregados distintos:

    ¿cuántas regeneraciones hace el sistema **por cada km frío** que el conductor genera?

`regenerations_per_1000km` sola confunde dos cosas —un auto que regenera poco porque
está sano y anda en ruta, y uno que regenera poco porque no puede—. Dividida por la
carga que ese mismo uso genera, queda la **ganancia del lazo**: cuánto responde el
controlador por unidad de agresión. Esa es la versión medible de la hipótesis del §0 de
`docs/f3-features-candidatas-fisica.md`, y es la que la primera vuelta no pudo probar:
se midió el esfuerzo en absoluto (no dio) pero nunca condicionado por la carga.

Lo mismo con las conjunciones: "viaje corto" y "arranque en frío" están cada uno en el
panel, pero "viaje corto **y** en frío **y** después de una noche parado" es un caso
peor que la suma de los tres, y un modelo lineal no lo puede construir. Las conjunciones
se arman a nivel viaje (en `trips.py`); acá van las que son cociente o producto de dos
agregados de ventana.

**Por qué no leakea.** Todo entra desde columnas `feat_*`/`aux_*` de la MISMA fila, que
ya son solo hacia atrás (regla 3). No hay estadístico de train acá adentro: es aritmética
fila a fila, así que no hay nada que fitear por fold.

**Cómo se agrega una.** Una línea en el bloque `derived:` del YAML de features:

    derived:
      - {name: regen_per_cold_km, op: ratio, a: feat_regenerations_per_1000km,
         b: feat_cold_km_per_1000km, min_b: 0.5}

`op` es uno de `OPS`. `min_b` (solo en `ratio`) es el piso del denominador: por debajo,
la fila sale NaN en vez de un cociente que explota. Sin ese piso, un vehículo con 0,01 km
fríos en la ventana se convierte en el outlier que domina el modelo.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd

from src.features.windows import AUX_PREFIX, FEAT_PREFIX

logger = logging.getLogger(__name__)

DEFAULT_MIN_B = 1e-9


def _ratio(a: pd.Series, b: pd.Series, min_b: float) -> pd.Series:
    return a / b.where(b.abs() >= min_b)


OPS: dict[str, Callable[[pd.Series, pd.Series, float], pd.Series]] = {
    "ratio": _ratio,
    "product": lambda a, b, _: a * b,
    "diff": lambda a, b, _: a - b,
}


@dataclass(frozen=True)
class DerivedSpec:
    name: str
    op: str
    a: str
    b: str
    min_b: float = DEFAULT_MIN_B
    aux: bool = False

    @property
    def output(self) -> str:
        return f"{AUX_PREFIX if self.aux else FEAT_PREFIX}{self.name}"


def load_derived_specs(cfg: dict[str, Any]) -> list[DerivedSpec]:
    """Lee el bloque `derived:` del YAML de features. Vacío o ausente = lista vacía."""
    entries = cfg.get("derived") or []
    specs: list[DerivedSpec] = []
    seen: set[str] = set()
    for entry in entries:
        spec = DerivedSpec(
            name=str(entry["name"]), op=str(entry["op"]), a=str(entry["a"]), b=str(entry["b"]),
            min_b=float(entry.get("min_b", DEFAULT_MIN_B)), aux=bool(entry.get("aux", False)),
        )
        if spec.op not in OPS:
            raise ValueError(f"Derivada `{spec.name}`: operación desconocida `{spec.op}` "
                             f"(disponibles: {sorted(OPS)})")
        if spec.op != "ratio" and entry.get("min_b") is not None:
            raise ValueError(f"Derivada `{spec.name}`: `min_b` solo tiene sentido en `ratio`")
        if spec.output in seen:
            raise ValueError(f"Derivada repetida en el YAML: `{spec.output}`")
        seen.add(spec.output)
        specs.append(spec)
    return specs


def add_derived_features(panel: pd.DataFrame, specs: list[DerivedSpec]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Agrega una columna por spec. Devuelve `(panel, resumen)`.

    Falla si una spec pide una columna que el panel no tiene: es un YAML mal escrito,
    y descubrirlo al entrenar (con la columna en NaN) es peor que fallar acá.
    """
    if not specs:
        return panel, {"enabled": False, "n": 0}

    out = panel.copy()
    summary: dict[str, Any] = {"enabled": True, "n": len(specs), "columns": {}}
    for spec in specs:
        missing = [c for c in (spec.a, spec.b) if c not in out.columns]
        if missing:
            raise KeyError(f"Derivada `{spec.name}`: el panel no tiene {missing}. "
                           f"¿Está declarada la feature de base en `families:`?")
        values = OPS[spec.op](out[spec.a].astype("float64"), out[spec.b].astype("float64"), spec.min_b)
        out[spec.output] = values.replace([np.inf, -np.inf], np.nan)
        summary["columns"][spec.output] = {
            "op": spec.op, "a": spec.a, "b": spec.b, "min_b": spec.min_b,
            "null_frac": float(out[spec.output].isna().mean()),
        }
    worst = max(summary["columns"].items(), key=lambda kv: kv[1]["null_frac"])
    logger.info("Derivadas: %d columnas · la más nula es %s (%.1f%%)",
                len(specs), worst[0], worst[1]["null_frac"] * 100)
    return out, summary
