"""Secuencias por corte sobre el eje de odómetro: la entrada de los modelos secuenciales.

El panel v1 resume cada ventana `(c − W, c]` en un número por feature. Un modelo
secuencial necesita la ventana entera y en orden: acá se parte en `T = L / b` bins de
`b` km y cada bin se agrega en `C` canales. El resultado es un tensor `(filas, T, C)`
que se aplana a columnas `feat_seq_t{t}_c{k}_{canal}`, así viaja en un panel común
(mismas filas, mismo `splits.json`, mismo `scripts/train.py`).

Las reglas son las mismas que las de las features de ventana:

- **Solo hacia atrás** (regla 3): el bin más reciente termina en `c`. Una fila con
  odómetro > c no entra en la secuencia de c.
- **Nada que mida el calendario**: el marcador `Regenerations` y
  `DistanceBetweenRegenerations` se cortan el 25-05-2026 para toda la flota y no
  pueden ser canal (`FORBIDDEN_COLUMNS`).
- **El relleno de bins vacíos no mira otras filas**: ceros, o arrastre dentro de la
  misma secuencia (que es toda anterior al corte). Lo que se ajusta con datos
  —imputación, escalado— lo hace el `Pipeline` de cada fold.

Los bins se calculan una vez por vehículo sobre una grilla absoluta de múltiplos de
`b` (el bin `j` cubre `(j·b, (j+1)·b]`) y cada corte toma su tajada. Para eso `c` y `L`
tienen que ser múltiplos de `b`, y la grilla del panel (múltiplos de Δ) lo garantiza
si `b` divide a Δ y a `L`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.features.windows import FEAT_PREFIX

logger = logging.getLogger(__name__)

ID_COL = "vehicle_id"
SEQ_PREFIX = f"{FEAT_PREFIX}seq_"
ALL_ROWS = "*"                      # `column: "*"` cuenta filas, no valores de una columna
POSITION_COLUMN = {"signals": "OdometerValue", "trips": "OdometerTripEnd"}
AGGS = ("mean", "max", "sum", "count", "any")
FILLS = ("zero", "carry", "none")
TRANSFORMS = ("log1p",)
FORBIDDEN_COLUMNS = frozenset({"Regenerations", "DistanceBetweenRegenerations", "regen_marker"})


@dataclass(frozen=True)
class ChannelSpec:
    """Un canal de la secuencia: qué columna de qué tabla, cómo se agrega por bin y cómo se rellena.

    `fill` decide qué vale un bin sin filas: `zero` (conteos, fracciones), `carry`
    (niveles: el último valor visto dentro de la misma secuencia, y hacia atrás si
    la secuencia arranca vacía) o `none` (queda NaN para el imputador del fold).
    """

    name: str
    source: str
    column: str
    agg: str
    fill: str = "zero"
    transform: str | None = None

    def __post_init__(self) -> None:
        if self.source not in POSITION_COLUMN:
            raise ValueError(f"Canal `{self.name}`: source `{self.source}` desconocida ({sorted(POSITION_COLUMN)})")
        if self.agg not in AGGS:
            raise ValueError(f"Canal `{self.name}`: agg `{self.agg}` desconocido ({list(AGGS)})")
        if self.fill not in FILLS:
            raise ValueError(f"Canal `{self.name}`: fill `{self.fill}` desconocido ({list(FILLS)})")
        if self.transform is not None and self.transform not in TRANSFORMS:
            raise ValueError(f"Canal `{self.name}`: transform `{self.transform}` desconocido ({list(TRANSFORMS)})")
        if self.column in FORBIDDEN_COLUMNS:
            raise ValueError(
                f"Canal `{self.name}`: `{self.column}` se corta el 25-05-2026 y mide el calendario, "
                "no el vehículo (CLAUDE.md, docs/reproducibilidad.md)."
            )
        if self.column == ALL_ROWS and self.agg not in ("count", "any"):
            raise ValueError(f"Canal `{self.name}`: `column: \"*\"` solo admite agg count/any")


def load_channel_specs(entries: list[dict[str, Any]]) -> list[ChannelSpec]:
    """Canales desde el bloque `sequence.channels` del YAML, en el orden declarado."""
    specs = [ChannelSpec(**entry) for entry in entries]
    names = [s.name for s in specs]
    dup = sorted({n for n in names if names.count(n) > 1})
    if dup:
        raise ValueError(f"Canales repetidos en el YAML: {dup}")
    if not specs:
        raise ValueError("`sequence.channels` está vacío")
    return specs


def derive_signal_sequence_columns(signals: pd.DataFrame, *, regen_drop_points: float) -> pd.DataFrame:
    """Derivadas de `signals` que el panel v1 no necesita y la secuencia sí.

    Entra la salida de `src/features/signals.py::derive_signal_columns`. Las diferencias
    de `Acumulation` se toman en orden temporal dentro del vehículo: cada fila mira solo
    la anterior, así que agregarlas hacia atrás no filtra nada del futuro.

    - `acum_rise`: puntos que subió el nivel desde la fila anterior (carga del filtro).
    - `acum_regen_drop`: el nivel cayó más de `regen_drop_points` (una regeneración).
      Es la misma definición que `regen_drop` en `trips`, sin el marcador cortado.
    - `msg_severe`: `Overloaded` u `Over Limit`, los dos estados raros y malos juntos.
    """
    required = [ID_COL, "eventTimestamp", "OdometerValue", "Acumulation", "msg_overloaded", "msg_over_limit"]
    missing = [c for c in required if c not in signals.columns]
    if missing:
        raise KeyError(f"`signals` no tiene las columnas requeridas: {missing}")
    out = signals.sort_values([ID_COL, "eventTimestamp", "OdometerValue"], kind="stable", ignore_index=True)
    delta = out.groupby(ID_COL, observed=True)["Acumulation"].diff().astype("float64")
    out["acum_rise"] = delta.clip(lower=0)
    out["acum_regen_drop"] = delta.lt(-float(regen_drop_points))
    out["msg_severe"] = out["msg_overloaded"] | out["msg_over_limit"]
    return out


def sequence_column_names(seq_len: int, channels: list[ChannelSpec]) -> list[str]:
    """Nombres aplanados, en orden (t, canal).

    El orden lexicográfico de los nombres coincide con el del tensor: el panel ordena
    las `feat_*` alfabéticamente y el modelo reconstruye `(T, C)` con un reshape.
    """
    width_t = max(3, len(str(seq_len - 1)))
    width_c = max(2, len(str(len(channels) - 1)))
    return [
        f"{SEQ_PREFIX}t{t:0{width_t}d}_c{k:0{width_c}d}_{spec.name}"
        for t in range(seq_len)
        for k, spec in enumerate(channels)
    ]


def bin_source(frame: pd.DataFrame, *, source: str, bin_km: float, channels: list[ChannelSpec]) -> pd.DataFrame:
    """Agregado por `(vehicle_id, bin)` de los canales de una tabla. Solo aparecen bins con filas."""
    position = POSITION_COLUMN[source]
    mine = [c for c in channels if c.source == source]
    columns = sorted({c.column for c in mine if c.column != ALL_ROWS})
    missing = [c for c in [ID_COL, position, *columns] if c not in frame.columns]
    if missing:
        raise KeyError(f"`{source}` no tiene columnas que piden los canales: {missing}")

    work = frame.loc[frame[position].notna(), [ID_COL, position, *columns]].copy()
    work["_bin"] = (np.ceil(work[position].astype("float64") / bin_km) - 1).astype("int64")
    work["_one"] = 1.0
    for column in columns:
        work[column] = work[column].astype("float64")

    named = {}
    for spec in mine:
        column = "_one" if spec.column == ALL_ROWS else spec.column
        named[spec.name] = (column, "count" if spec.agg == "any" else spec.agg)
    binned = work.groupby([ID_COL, "_bin"], observed=True, sort=True).agg(**named)
    for spec in mine:
        if spec.agg == "any":
            binned[spec.name] = binned[spec.name].gt(0).astype("float64")
    return binned


def build_sequences(
    keys: pd.DataFrame,
    binned: pd.DataFrame,
    channels: list[ChannelSpec],
    *,
    lookback_km: float,
    bin_km: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Tensor `(len(keys), T, C)` con la secuencia de cada `(vehicle_id, cut_odo)`.

    `binned` es la unión de `bin_source` sobre las tablas que usen los canales. El bin
    `t = T − 1` es el más cercano al corte. Devuelve también un resumen de cobertura.
    """
    seq_len = int(round(lookback_km / bin_km))
    if not np.isclose(seq_len * bin_km, lookback_km):
        raise ValueError(f"lookback_km={lookback_km} no es múltiplo de bin_km={bin_km}")
    cuts = keys["cut_odo"].to_numpy(dtype="float64")
    start = (cuts - lookback_km) / bin_km
    if not np.allclose(start, np.round(start)):
        raise ValueError(
            f"Hay cortes que no caen en la grilla de {bin_km:g} km: bin_km tiene que dividir a Δ "
            "(sequence.bin_km en el YAML contra label.cut_step_km del panel)."
        )
    start = np.round(start).astype("int64")

    vehicles = np.repeat(keys[ID_COL].astype(str).to_numpy(), seq_len)
    bins = (start[:, None] + np.arange(seq_len)[None, :]).ravel()
    index = pd.MultiIndex.from_arrays([vehicles, bins], names=[ID_COL, "_bin"])
    aligned = binned.reindex(index)
    names = [c.name for c in channels]
    tensor = aligned[names].to_numpy(dtype="float64").reshape(len(keys), seq_len, len(channels))

    empty_bins = np.isnan(tensor).all(axis=2)
    for k, spec in enumerate(channels):
        plane = tensor[:, :, k]
        if spec.fill == "zero":
            plane = np.where(np.isnan(plane), 0.0, plane)
        elif spec.fill == "carry":
            plane = pd.DataFrame(plane).ffill(axis=1).bfill(axis=1).to_numpy()
        if spec.transform == "log1p":
            plane = np.log1p(plane)
        tensor[:, :, k] = plane

    coverage = {
        "seq_len": seq_len,
        "n_channels": len(channels),
        "empty_bin_frac": float(empty_bins.mean()),
        "rows_all_bins_empty": int(empty_bins.all(axis=1).sum()),
        "rows_with_nan_after_fill": int(np.isnan(tensor).any(axis=(1, 2)).sum()),
    }
    return tensor, coverage


def flatten_sequences(tensor: np.ndarray, channels: list[ChannelSpec], *, index: pd.Index | None = None) -> pd.DataFrame:
    """`(N, T, C)` → DataFrame con una columna por `(t, canal)`, en el orden de `sequence_column_names`."""
    n_rows, seq_len, n_channels = tensor.shape
    if n_channels != len(channels):
        raise ValueError(f"El tensor tiene {n_channels} canales y el YAML declara {len(channels)}")
    names = sequence_column_names(seq_len, channels)
    return pd.DataFrame(tensor.reshape(n_rows, seq_len * n_channels), columns=names, index=index)
