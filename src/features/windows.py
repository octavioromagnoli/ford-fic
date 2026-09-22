"""Primitiva de agregación de ventana móvil hacia atrás (F2, Track A).

Una sola función hace el trabajo: `compute_window_features` recibe los arrays de un
vehículo (viajes y señales ordenados por odómetro), una lista de puntos de corte y
la lista de features declarada en `configs/data/features_v1.yaml`, y devuelve una
fila por corte con las columnas `feat_*` (y `aux_*` para las que no entran al
modelo). **Agregar una feature es agregar una línea en ese YAML**, no tocar esto.

Cada entrada del YAML es `(name, source, column, agg)`:

- `source`: `trips` o `signals`;
- `column`: una columna derivada por `src/features/trips.py` / `signals.py`;
- `agg`: uno de `AGGREGATORS` (abajo). Los que terminan en `per_1000km` usan los km
  recorridos dentro de la ventana; los `half_*` parten la ventana en dos mitades de
  W/2 km y comparan la segunda contra la primera (tendencia robusta: diferencia de
  medianas, no una recta que un outlier pueda tumbar).

Regla no negociable (regla 3 de CLAUDE.md): la ventana de un corte `c` es
`(c − W, c]` sobre el odómetro, y ninguna fila posterior a `c` entra. Un viaje se
ubica por su `OdometerTripEnd`: si termina después del corte, no se ve.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from src.config import load_config

logger = logging.getLogger(__name__)

FEAT_PREFIX = "feat_"
AUX_PREFIX = "aux_"
MIN_HALF_POINTS = 3     # puntos mínimos por mitad para calcular una tendencia


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    source: str
    column: str
    agg: str
    family: str = ""
    aux: bool = False

    @property
    def output(self) -> str:
        return f"{AUX_PREFIX if self.aux else FEAT_PREFIX}{self.name}"


@dataclass
class WindowContext:
    """Lo que un agregador necesita además de la columna: dónde están las filas y cuánto cubren."""

    positions: np.ndarray        # odómetro de cada fila de la ventana (creciente)
    cut_odo: float
    window_km: float
    km_covered: float            # km recorridos dentro de la ventana (suma de trip_km > 0)
    days_covered: float          # días entre el primer y el último viaje de la ventana (>= 1)

    @property
    def half_odo(self) -> float:
        return self.cut_odo - self.window_km / 2.0

    def second_half(self) -> np.ndarray:
        return self.positions > self.half_odo


# --------------------------------------------------------------------------------------
# Agregadores: (valores de la columna en la ventana, contexto) -> float
# --------------------------------------------------------------------------------------
def _clean(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype="float64")
    return x[~np.isnan(x)]


def _nan_if_empty(fn: Callable[[np.ndarray], float]) -> Callable[[np.ndarray, WindowContext], float]:
    def wrapped(x: np.ndarray, ctx: WindowContext) -> float:
        clean = _clean(x)
        return float(fn(clean)) if clean.size else np.nan
    return wrapped


def _per_1000km(x: np.ndarray, ctx: WindowContext) -> float:
    if ctx.km_covered <= 0:
        return np.nan
    return float(np.nansum(x) / ctx.km_covered * 1000.0)


def _per_day(x: np.ndarray, ctx: WindowContext) -> float:
    if ctx.days_covered <= 0:
        return np.nan
    return float(np.nansum(x) / ctx.days_covered)


def _half_medians(x: np.ndarray, ctx: WindowContext) -> tuple[float, float] | None:
    x = np.asarray(x, dtype="float64")
    second = ctx.second_half()
    a, b = x[~second], x[second]
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if a.size < MIN_HALF_POINTS or b.size < MIN_HALF_POINTS:
        return None
    return float(np.median(a)), float(np.median(b))


def _half_delta(x: np.ndarray, ctx: WindowContext) -> float:
    halves = _half_medians(x, ctx)
    return np.nan if halves is None else halves[1] - halves[0]


def _half_slope_per_1000km(x: np.ndarray, ctx: WindowContext) -> float:
    delta = _half_delta(x, ctx)
    return np.nan if np.isnan(delta) else delta / (ctx.window_km / 2.0) * 1000.0


def _half_mean_delta(x: np.ndarray, ctx: WindowContext) -> float:
    """Diferencia de medias (para booleanos: diferencia de fracciones) entre mitades."""
    x = np.asarray(x, dtype="float64")
    second = ctx.second_half()
    a, b = x[~second], x[second]
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if a.size < MIN_HALF_POINTS or b.size < MIN_HALF_POINTS:
        return np.nan
    return float(b.mean() - a.mean())


def _half_count_delta_per_1000km(x: np.ndarray, ctx: WindowContext) -> float:
    """Eventos por 1.000 km en la segunda mitad menos la primera (mitades de W/2 km)."""
    x = np.nan_to_num(np.asarray(x, dtype="float64"))
    second = ctx.second_half()
    half_km = ctx.window_km / 2.0
    return float((x[second].sum() - x[~second].sum()) / half_km * 1000.0)


def _event_positions(x: np.ndarray, ctx: WindowContext) -> np.ndarray:
    x = np.nan_to_num(np.asarray(x, dtype="float64"))
    return ctx.positions[x > 0]


def _gap_mean_km(x: np.ndarray, ctx: WindowContext) -> float:
    pos = _event_positions(x, ctx)
    return float(np.diff(pos).mean()) if pos.size >= 2 else np.nan


def _gap_median_km(x: np.ndarray, ctx: WindowContext) -> float:
    pos = _event_positions(x, ctx)
    return float(np.median(np.diff(pos))) if pos.size >= 2 else np.nan


def _gap_trend_km(x: np.ndarray, ctx: WindowContext) -> float:
    """Distancia media entre eventos en la segunda mitad menos en la primera."""
    pos = _event_positions(x, ctx)
    if pos.size < 4:
        return np.nan
    gaps = np.diff(pos)
    ends = pos[1:]
    second = ends > ctx.half_odo
    if second.sum() < 1 or (~second).sum() < 1:
        return np.nan
    return float(gaps[second].mean() - gaps[~second].mean())


def _km_since_last_event(x: np.ndarray, ctx: WindowContext) -> float:
    """Km recorridos desde el último evento de la ventana hasta el corte (NaN si no hubo)."""
    pos = _event_positions(x, ctx)
    return float(ctx.cut_odo - pos[-1]) if pos.size else np.nan


AGGREGATORS: dict[str, Callable[[np.ndarray, WindowContext], float]] = {
    "mean": _nan_if_empty(np.mean),
    "median": _nan_if_empty(np.median),
    "max": _nan_if_empty(np.max),
    "min": _nan_if_empty(np.min),
    "std": _nan_if_empty(lambda v: np.std(v, ddof=1) if v.size > 1 else np.nan),
    "p25": _nan_if_empty(lambda v: np.percentile(v, 25)),
    "p75": _nan_if_empty(lambda v: np.percentile(v, 75)),
    "sum": _nan_if_empty(np.sum),
    "count": lambda x, ctx: float(_clean(x).size),
    "any": _nan_if_empty(lambda v: float(v.max() > 0)),
    "per_1000km": _per_1000km,
    "per_day": _per_day,
    "half_delta": _half_delta,
    "half_mean_delta": _half_mean_delta,
    "half_slope_per_1000km": _half_slope_per_1000km,
    "half_count_delta_per_1000km": _half_count_delta_per_1000km,
    "gap_mean_km": _gap_mean_km,
    "gap_median_km": _gap_median_km,
    "gap_trend_km": _gap_trend_km,
    "km_since_last": _km_since_last_event,
}


# --------------------------------------------------------------------------------------
# Spec
# --------------------------------------------------------------------------------------
def load_feature_specs(path: str | Path, *, enabled_families: list[str] | None = None) -> list[FeatureSpec]:
    """Lee `families:` del YAML de features y devuelve la lista plana, en orden."""
    cfg = load_config(path)
    families = cfg.get("families") or {}
    specs: list[FeatureSpec] = []
    seen: set[str] = set()
    for family, entries in families.items():
        if enabled_families is not None and family not in enabled_families:
            continue
        for entry in entries or []:
            spec = FeatureSpec(
                name=str(entry["name"]), source=str(entry["source"]), column=str(entry["column"]),
                agg=str(entry["agg"]), family=family, aux=bool(entry.get("aux", False)),
            )
            if spec.source not in {"trips", "signals"}:
                raise ValueError(f"Feature `{spec.name}`: source desconocido `{spec.source}`")
            if spec.agg not in AGGREGATORS:
                raise ValueError(f"Feature `{spec.name}`: agregador desconocido `{spec.agg}` "
                                 f"(disponibles: {sorted(AGGREGATORS)})")
            if spec.output in seen:
                raise ValueError(f"Feature repetida en el YAML: `{spec.output}`")
            seen.add(spec.output)
            specs.append(spec)
    if not specs:
        raise ValueError(f"El YAML {path} no declara ninguna feature (o ninguna familia habilitada)")
    return specs


def required_columns(specs: list[FeatureSpec], source: str) -> list[str]:
    return sorted({s.column for s in specs if s.source == source})


# --------------------------------------------------------------------------------------
# Arrays por vehículo y agregación por corte
# --------------------------------------------------------------------------------------
@dataclass
class VehicleArrays:
    """Columnas de un vehículo como arrays float64 ordenados por odómetro."""

    positions: np.ndarray
    columns: dict[str, np.ndarray] = field(default_factory=dict)
    trip_km: np.ndarray | None = None
    time_start_ns: np.ndarray | None = None
    time_end_ns: np.ndarray | None = None

    @classmethod
    def from_frame(cls, frame: pd.DataFrame, *, position_col: str, columns: list[str],
                   trip_km_col: str | None = None, time_cols: tuple[str, str] | None = None) -> "VehicleArrays":
        ordered = frame.sort_values(position_col)
        arrays = {c: ordered[c].to_numpy(dtype="float64", na_value=np.nan) for c in columns}
        trip_km = ordered[trip_km_col].to_numpy(dtype="float64", na_value=np.nan) if trip_km_col else None
        t0 = t1 = None
        if time_cols:
            t0 = _datetime_to_ns(ordered[time_cols[0]])
            t1 = _datetime_to_ns(ordered[time_cols[1]])
        return cls(positions=ordered[position_col].to_numpy(dtype="float64"), columns=arrays,
                   trip_km=trip_km, time_start_ns=t0, time_end_ns=t1)

    def window(self, lo: float, hi: float) -> tuple[int, int]:
        """Índices `[i0, i1)` de las filas con `lo < posición <= hi`."""
        i0 = int(np.searchsorted(self.positions, lo, side="right"))
        i1 = int(np.searchsorted(self.positions, hi, side="right"))
        return i0, i1


def _datetime_to_ns(series: pd.Series) -> np.ndarray:
    # pandas 3 parsea las fechas con resolución de microsegundos (`datetime64[us, UTC]`):
    # se fuerza a ns ANTES de leer los enteros, o los días salen mil veces más cortos.
    values = pd.to_datetime(series, utc=True, errors="coerce").dt.as_unit("ns")
    out = values.array.asi8.astype("float64")
    out[values.isna().to_numpy()] = np.nan
    return out


def compute_window_features(
    trips: VehicleArrays,
    signals: VehicleArrays | None,
    cuts: np.ndarray,
    *,
    window_km: float,
    specs: list[FeatureSpec],
) -> pd.DataFrame:
    """Una fila por corte: features `feat_*`/`aux_*` + control de la ventana.

    Las columnas de control salen siempre: `feat_n_trips_window`,
    `feat_window_km_covered` (reservadas por el panel dummy), y `aux_n_moving_window`,
    `aux_n_signals_window`, `aux_window_days_covered`, `cut_date` (fecha del último
    viaje de la ventana, NaT si no hay viaje con fecha).
    """
    trip_specs = [s for s in specs if s.source == "trips"]
    signal_specs = [s for s in specs if s.source == "signals"]
    rows: list[dict[str, float]] = []
    dates: list[pd.Timestamp] = []

    for cut in np.asarray(cuts, dtype="float64"):
        lo, hi = cut - window_km, cut
        i0, i1 = trips.window(lo, hi)
        n_trips = i1 - i0
        pos = trips.positions[i0:i1]
        km = trips.trip_km[i0:i1] if trips.trip_km is not None else np.zeros(n_trips)
        km_covered = float(np.nansum(np.clip(km, 0, None))) if n_trips else 0.0
        days_covered, cut_date = _time_coverage(trips, i0, i1)
        ctx = WindowContext(positions=pos, cut_odo=float(cut), window_km=float(window_km),
                            km_covered=km_covered, days_covered=days_covered)
        row: dict[str, float] = {
            f"{FEAT_PREFIX}n_trips_window": float(n_trips),
            f"{FEAT_PREFIX}window_km_covered": km_covered,
            f"{AUX_PREFIX}n_moving_window": float(np.nansum(km > 0)) if n_trips else 0.0,
            f"{AUX_PREFIX}window_days_covered": days_covered,
        }
        for spec in trip_specs:
            values = trips.columns[spec.column][i0:i1]
            row[spec.output] = AGGREGATORS[spec.agg](values, ctx) if n_trips else np.nan

        if signals is not None:
            j0, j1 = signals.window(lo, hi)
            n_signals = j1 - j0
            sctx = WindowContext(positions=signals.positions[j0:j1], cut_odo=float(cut), window_km=float(window_km),
                                 km_covered=km_covered, days_covered=days_covered)
            row[f"{AUX_PREFIX}n_signals_window"] = float(n_signals)
            for spec in signal_specs:
                values = signals.columns[spec.column][j0:j1]
                row[spec.output] = AGGREGATORS[spec.agg](values, sctx) if n_signals else np.nan
        else:
            row[f"{AUX_PREFIX}n_signals_window"] = 0.0
            for spec in signal_specs:
                row[spec.output] = np.nan
        rows.append(row)
        dates.append(cut_date)

    out = pd.DataFrame(rows)
    out["cut_date"] = pd.Series(dates, dtype="datetime64[ns, UTC]").dt.tz_convert(None).to_numpy()
    return out


def _time_coverage(trips: VehicleArrays, i0: int, i1: int) -> tuple[float, pd.Timestamp]:
    if trips.time_start_ns is None or i1 <= i0:
        return 1.0, pd.NaT
    starts = trips.time_start_ns[i0:i1]
    ends = trips.time_end_ns[i0:i1] if trips.time_end_ns is not None else starts
    valid_start = starts[~np.isnan(starts)]
    valid_end = ends[~np.isnan(ends)]
    if valid_start.size == 0:
        return 1.0, pd.NaT
    last = valid_end.max() if valid_end.size else valid_start.max()
    days = (last - valid_start.min()) / 86_400e9
    return max(float(days), 1.0), pd.Timestamp(int(last), unit="ns", tz="UTC")


# --------------------------------------------------------------------------------------
# Desvío respecto de la historia previa del propio vehículo
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class HistoryDeviationSpec:
    """`agg(historia previa) − agg(ventana)` de una columna de `trips`.

    La historia previa de un corte `c` es `(c − max_context_km, c − W]`: todo lo que el
    vehículo recorrió antes de la ventana, hasta un largo máximo. La ventana es la misma
    `(c − W, c]` de `compute_window_features`, así que el lado de la ventana reproduce la
    columna del panel v1 con el mismo agregador (y el builder lo verifica).
    """

    name: str
    column: str
    agg: str

    @property
    def output(self) -> str:
        return f"{FEAT_PREFIX}{self.name}"


@dataclass(frozen=True)
class HistoryDeviationConfig:
    max_context_km: float        # largo máximo mirado hacia atrás desde el corte (incluye W)
    min_history_km: float        # km recorridos mínimos en la historia previa; si no, NaN
    specs: tuple[HistoryDeviationSpec, ...]

    def __post_init__(self) -> None:
        if self.min_history_km <= 0:
            raise ValueError("`min_history_km` tiene que ser > 0: sin historia no hay desvío que medir")


def load_history_deviation_config(path: str | Path) -> HistoryDeviationConfig:
    """Lee `history:` y `features:` de un YAML como `configs/data/features_history.yaml`."""
    cfg = load_config(path)
    history = cfg["history"]
    specs = []
    for entry in cfg.get("features") or []:
        spec = HistoryDeviationSpec(name=str(entry["name"]), column=str(entry["column"]), agg=str(entry["agg"]))
        if spec.agg not in AGGREGATORS:
            raise ValueError(f"Feature `{spec.name}`: agregador desconocido `{spec.agg}`")
        if not spec.name.endswith("_hist_delta"):
            raise ValueError(f"Feature `{spec.name}`: las de desvío se llaman `*_hist_delta`")
        specs.append(spec)
    if not specs:
        raise ValueError(f"El YAML {path} no declara ninguna feature de desvío")
    return HistoryDeviationConfig(
        max_context_km=float(history["max_context_km"]),
        min_history_km=float(history["min_history_km"]),
        specs=tuple(specs),
    )


def _range_context(trips: VehicleArrays, lo: float, hi: float) -> tuple[int, int, WindowContext]:
    i0, i1 = trips.window(lo, hi)
    km = trips.trip_km[i0:i1] if trips.trip_km is not None else np.zeros(i1 - i0)
    km_covered = float(np.nansum(np.clip(km, 0, None))) if i1 > i0 else 0.0
    days_covered, _ = _time_coverage(trips, i0, i1)
    ctx = WindowContext(positions=trips.positions[i0:i1], cut_odo=float(hi), window_km=float(hi - lo),
                        km_covered=km_covered, days_covered=days_covered)
    return i0, i1, ctx


def compute_history_deviation(
    trips: VehicleArrays,
    cuts: np.ndarray,
    *,
    window_km: float,
    config: HistoryDeviationConfig,
) -> pd.DataFrame:
    """Una fila por corte: `feat_<name>` = agg(historia previa) − agg(ventana).

    Estrictamente hacia atrás (regla 3): las dos partes terminan en el corte, y un viaje
    se ubica por su `OdometerTripEnd` igual que en `compute_window_features`. Si la
    historia previa recorre menos de `min_history_km` —el corte está al principio del
    registro—, el desvío es NaN: no se inventa una historia de ceros.

    Sale también, fuera del modelo, lo necesario para auditar que esto no sea el largo
    del historial disfrazado: `aux_hist_km_covered` (km de la historia usada) y el valor
    de cada lado (`aux_<name>_hist`, `aux_<name>_window`).
    """
    if config.max_context_km <= window_km:
        raise ValueError(f"`max_context_km`={config.max_context_km} no deja historia antes de W={window_km}")
    rows: list[dict[str, float]] = []
    for cut in np.asarray(cuts, dtype="float64"):
        w0, w1, wctx = _range_context(trips, cut - window_km, cut)
        h0, h1, hctx = _range_context(trips, cut - config.max_context_km, cut - window_km)
        enough = hctx.km_covered >= config.min_history_km
        row: dict[str, float] = {f"{AUX_PREFIX}hist_km_covered": hctx.km_covered}
        for spec in config.specs:
            values = trips.columns[spec.column]
            win = AGGREGATORS[spec.agg](values[w0:w1], wctx) if w1 > w0 else np.nan
            hist = AGGREGATORS[spec.agg](values[h0:h1], hctx) if (enough and h1 > h0) else np.nan
            base = spec.output[len(FEAT_PREFIX):]
            row[f"{AUX_PREFIX}{base}_hist"] = hist
            row[f"{AUX_PREFIX}{base}_window"] = win
            row[spec.output] = hist - win
        rows.append(row)
    return pd.DataFrame(rows)
