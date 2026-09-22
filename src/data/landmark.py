"""Panel de hitos post-venta (F3, cure model, Track A).

Una fila por **(vehículo, hito L)**, con L en días desde la venta. Reemplaza la grilla de
km del panel v1 porque el evento sigue un reloj de días post-venta y se registra solo
dentro de una ventana de calendario (`docs/memoria/f3-reloj-y-ventana-del-evento.md`).
Lo que se fija acá está preregistrado en `docs/memoria/f3-preregistro-cure.md` §1–§3.

**Reloj.** venta = origen + `ProductionDay` + `daysUntilSale`; evento = origen +
`ProductionDay` + `IdentificationDate` (`src/data/anchor.py`). Todo en días desde la venta
(*dss*).

**Qué fila existe.** El vehículo tiene fecha de venta, llega al hito con telemetría
(venta + L ≤ su último viaje), no tuvo el evento en `≤ L + G` (regla 1: el gap) y tiene al
menos `min_trips` viajes en la ventana de features. Cada descarte se cuenta.

**Ventana de features:** los viajes que arrancan después de la venta y **terminan** a más
tardar en el hito, `(venta, venta + L]`. Ningún viaje posterior al hito entra en ninguna
columna `feat_`/`aux_raw_` (regla 3).

**Riesgo observable** (lo que consume el target `cure_window`):

- entrada `e = max(L + G, dss del inicio de la ventana del registro)`;
- salida `x = min(dss del evento, dss del fin de la ventana)`. **No** se acota por el
  último viaje: el registro anota eventos sin telemetría (preregistro §1);
- δ = 1 si el evento cae en `(e, x]`. Una fila con `x ≤ e` no informa sobre el evento,
  pero se conserva: entra a la referencia de flota y se puntúa.

**Ventana de features fija (opcional).** Con `feature_window_days` las features salen de
`(venta, venta + min(L, feature_window_days)]` en vez de `(venta, venta + L]`: el rasgo
temprano de los primeros días, igual en todos los hitos. Qué fila existe, `window_km`,
`aux_n_trips_window` y la temperatura ambiente siguen midiéndose hasta el hito, así las
filas y los pisos del cure model no cambian (F5 §3.2, la incidencia externa se entrena
con 30 días y se aplica con 30 días).

**Features para la normalización contra la flota.** Por cada feature de
`fleet_features` y cada mes calendario de la ventana, el valor del vehículo en ese mes
(`feat_fm__<feature>__<YYYY-MM>`) y su peso (`feat_fm__w_<feature>__<YYYY-MM>`). El valor
queda en NaN si el peso no llega a `min_weight`; **el peso se guarda igual**, así la suma
de los pesos reproduce los km o los viajes de la ventana. La referencia de la flota no se
calcula acá: es un parámetro ajustado con el train de cada fold
(`FleetReferenceNormalizer`, Track B).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd

from src.config import load_config

logger = logging.getLogger(__name__)

ID_COL = "vehicle_id"
DAY = pd.Timedelta(days=1)

FM_PREFIX = "feat_fm__"
FM_WEIGHT = "w_"
_FM_PATTERN = re.compile(rf"^{FM_PREFIX}(?P<weight>{FM_WEIGHT})?(?P<feature>.+)__(?P<month>\d{{4}}-\d{{2}})$")

AGG_WEIGHTS = {"per_1000km": "km", "mean": "n_valid"}
VEHICLE_REQUIRED = ("event_observed", "sale_date", "event_dss", "event_odo_km", "last_trip")
TRIP_REQUIRED = (ID_COL, "TripDatetimeStart", "TripDatetimeEnd", "OdometerTripEnd", "trip_km", "AirTemperatureAvg")


@dataclass(frozen=True)
class FleetFeature:
    """Una feature normalizada contra la flota: definición de `features_v1.yaml` + peso de la celda."""

    name: str
    column: str
    agg: str
    weight: str          # `km` (suma de trip_km) o `n_valid` (viajes con dato en `column`)
    min_weight: float

    def value_column(self, month: str) -> str:
        return f"{FM_PREFIX}{self.name}__{month}"

    def weight_column(self, month: str) -> str:
        return f"{FM_PREFIX}{FM_WEIGHT}{self.name}__{month}"

    @property
    def raw_column(self) -> str:
        return f"aux_raw_{self.name}"


@dataclass(frozen=True)
class LandmarkConfig:
    landmarks_days: tuple[float, ...]
    gap_days: float
    horizon_days: float
    min_trips: int
    window_start: pd.Timestamp       # inicio de la ventana del registro de eventos (UTC)
    window_end: pd.Timestamp         # fin de la ventana, inclusive
    resolved_from_landmark_days: float
    e_min_days: float
    e_min_sensitivity_days: tuple[float, ...] = ()
    # Ventana de features fija en días desde la venta (None: hasta el hito).
    feature_window_days: float | None = None
    # `feat_log1p_km_per_day`: log(1 + km/día) en la ventana de features (control de uso).
    usage_feature: bool = False


def landmark_config(cfg: dict[str, Any], event_window: dict[str, Any]) -> LandmarkConfig:
    """`LandmarkConfig` desde el YAML del panel y el `event_window` de `event_clock.yaml`."""
    lm, rn = cfg["landmark"], cfg["resolved_negative"]
    return LandmarkConfig(
        landmarks_days=tuple(float(x) for x in lm["landmarks_days"]),
        gap_days=float(lm["gap_days"]),
        horizon_days=float(lm["horizon_days"]),
        min_trips=int(lm["min_trips"]),
        window_start=pd.Timestamp(event_window["start"], tz="UTC"),
        window_end=pd.Timestamp(event_window["end"], tz="UTC"),
        resolved_from_landmark_days=float(rn["from_landmark_days"]),
        e_min_days=float(rn["e_min_days"]),
        e_min_sensitivity_days=tuple(float(x) for x in rn.get("sensitivity_days") or ()),
        feature_window_days=float(lm["feature_window_days"]) if lm.get("feature_window_days") is not None else None,
        usage_feature=bool(lm.get("usage_feature", False)),
    )


def load_fleet_features(entries: Iterable[dict[str, Any]], features_spec: str) -> list[FleetFeature]:
    """Las features de la flota con `column`/`agg` tomados de `features_v1.yaml`, no redeclarados.

    Falla si una no existe en el spec, si el peso no corresponde al agregador
    (`per_1000km` → km; `mean` → viajes con dato) o si el agregador no está soportado.
    """
    spec = load_config(features_spec)
    defined = {f["name"]: f for family in spec["families"].values() for f in family}
    out = []
    for entry in entries:
        name = entry["name"]
        if name not in defined:
            raise KeyError(f"`{name}` no está en {features_spec}: las features de la flota usan esas definiciones")
        base = defined[name]
        if base.get("source") != "trips":
            raise ValueError(f"`{name}` no sale de `trips`")
        agg = base["agg"]
        if agg not in AGG_WEIGHTS:
            raise ValueError(f"`{name}`: el agregador `{agg}` no tiene peso de celda definido ({sorted(AGG_WEIGHTS)})")
        if entry["weight"] != AGG_WEIGHTS[agg]:
            raise ValueError(f"`{name}`: con `{agg}` el peso de la celda es `{AGG_WEIGHTS[agg]}`, no `{entry['weight']}`")
        out.append(FleetFeature(name=name, column=base["column"], agg=agg, weight=entry["weight"],
                                min_weight=float(entry["min_weight"])))
    return out


def parse_fleet_column(column: str) -> tuple[str, str, bool] | None:
    """`(feature, mes, es_peso)` de una columna `feat_fm__*`; None si no es una."""
    match = _FM_PATTERN.match(column)
    if match is None:
        return None
    return match["feature"], match["month"], match["weight"] is not None


def production_tercile_edges(production_day: pd.Series, n: int) -> list[float]:
    """Bordes internos de los n-tiles de `ProductionDay` (se estiman sobre dev)."""
    values = pd.to_numeric(production_day, errors="coerce").dropna()
    return [float(values.quantile(k / n)) for k in range(1, n)]


def assign_tercile(production_day: pd.Series, edges: list[float]) -> pd.Series:
    """Tercil 0…n−1 con la convención de `pd.qcut`: (−∞, e1], (e1, e2], (e2, ∞)."""
    values = pd.to_numeric(production_day, errors="coerce")
    tercile = pd.Series(np.digitize(values.to_numpy(), edges, right=True), index=values.index)
    return tercile.where(values.notna()).astype("Int64")


def _cells(window: pd.DataFrame, features: list[FleetFeature], keys: list[str]) -> pd.DataFrame:
    """Valor y peso de cada feature por grupo `keys`. El valor es NaN bajo `min_weight`; el peso, nunca."""
    out = {}
    grouped_km = window.groupby(keys, observed=True)["trip_km"].sum()
    for f in features:
        x = window[f.column].astype("float64")
        frame = pd.DataFrame({"x": x, "n": x.notna().astype("float64")}, index=window.index)
        sums = frame.join(window[keys]).groupby(keys, observed=True)[["x", "n"]].sum()
        if f.agg == "per_1000km":
            weight = grouped_km.reindex(sums.index)
            value = sums["x"] / weight.where(weight > 0) * 1000.0
        else:
            weight = sums["n"]
            value = sums["x"] / weight.where(weight > 0)
        out[(f.name, "value")] = value.where(weight >= f.min_weight)
        out[(f.name, "weight")] = weight
    return pd.DataFrame(out)


def build_landmark_panel(
    vehicles: pd.DataFrame,
    trips: pd.DataFrame,
    cfg: LandmarkConfig,
    features: list[FleetFeature],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Arma el panel de hitos. Devuelve `(panel, reporte)`.

    `vehicles` va indexado por `vehicle_id`, con fechas en UTC: `sale_date`, `last_trip`
    (inicio del último viaje), `event_dss` (NaN sin evento), `event_odo_km` y
    `event_observed`. Sus columnas `static_*` y `aux_*` pasan tal cual a cada fila del
    vehículo. `trips` son los viajes derivados (`derive_trip_columns`) de esos vehículos.
    """
    missing = [c for c in VEHICLE_REQUIRED if c not in vehicles.columns]
    if missing:
        raise KeyError(f"`vehicles` no trae {missing}")
    needed = set(TRIP_REQUIRED) | {f.column for f in features}
    missing = sorted(needed - set(trips.columns))
    if missing:
        raise KeyError(f"`trips` no trae {missing}")
    if trips["TripDatetimeEnd"].isna().any():
        raise ValueError("Hay viajes sin `TripDatetimeEnd`: la ventana de features se corta por el fin del viaje")

    v = vehicles.copy()
    v["dss_w0"] = (cfg.window_start - v["sale_date"]) / DAY
    v["dss_w1"] = (cfg.window_end - v["sale_date"]) / DAY
    has_sale = v["sale_date"].notna()
    has_event = v["event_observed"].eq(1)
    # Exposición del vehículo desde el hito de referencia: define al negativo resuelto (D2).
    resolved_entry = np.maximum(cfg.resolved_from_landmark_days + cfg.gap_days, v["dss_w0"])
    potential = (v["dss_w1"] - resolved_entry).clip(lower=0).where(has_sale)
    v["resolved_exposure_days"] = potential.where(~has_event)
    # La misma cuenta sin mirar el desenlace: cuánta ventana del registro tuvo por delante
    # el vehículo según su fecha de venta. Es la elegibilidad simétrica del conjunto externo.
    v["potential_exposure_days"] = potential

    t = trips.loc[trips[ID_COL].isin(v.index)].join(v[["sale_date"]], on=ID_COL)
    t = t.loc[t["sale_date"].notna() & (t["TripDatetimeStart"] > t["sale_date"])]
    t["month"] = t["TripDatetimeStart"].dt.tz_convert(None).dt.to_period("M").astype(str)
    passthrough = [c for c in v.columns if c.startswith(("static_", "aux_"))]

    rows, report = [], {"landmarks": {}}
    for landmark in cfg.landmarks_days:
        land = v["sale_date"] + pd.Timedelta(days=landmark)
        reached = has_sale & (land <= v["last_trip"])
        before_gap = has_event & (v["event_dss"] <= landmark + cfg.gap_days)
        win = t.loc[t["TripDatetimeEnd"] <= t[ID_COL].map(land)]
        n_trips = win.groupby(ID_COL, observed=True).size().reindex(v.index, fill_value=0)
        enough = n_trips >= cfg.min_trips
        keep = reached & ~before_gap & enough
        report["landmarks"][f"{landmark:g}"] = {
            "sin_venta": int((~has_sale).sum()),
            "no_llega_al_hito": int((has_sale & ~reached).sum()),
            "evento_antes_o_en_gap": int((reached & before_gap).sum()),
            "pocos_viajes": int((reached & ~before_gap & ~enough).sum()),
            "filas": int(keep.sum()),
        }
        ids = v.index[keep]
        win = win.loc[win[ID_COL].isin(ids)]
        if cfg.feature_window_days is None:
            fwin, feature_days = win, landmark
        else:
            feature_days = min(landmark, cfg.feature_window_days)
            feature_end = v["sale_date"] + pd.Timedelta(days=feature_days)
            fwin = win.loc[win["TripDatetimeEnd"] <= win[ID_COL].map(feature_end)]
        d = v.loc[ids].copy()
        d["cut_date"] = land.loc[ids]
        # Odómetro del corte: el último viaje (de toda la historia) terminado a más tardar en el hito.
        done = trips.loc[trips[ID_COL].isin(ids)]
        done = done.loc[done["TripDatetimeEnd"] <= done[ID_COL].map(d["cut_date"])]
        d["cut_odo"] = done.groupby(ID_COL, observed=True)["OdometerTripEnd"].max().astype("float64")
        g = win.groupby(ID_COL, observed=True)
        d["window_km"] = g["trip_km"].sum()
        d["n_trips"] = g.size()
        d["air_temp"] = g["AirTemperatureAvg"].mean()
        raw = _cells(fwin, features, [ID_COL])
        monthly = _cells(fwin, features, [ID_COL, "month"])

        entry = np.maximum(landmark + cfg.gap_days, d["dss_w0"])
        exit_ = np.minimum(d["event_dss"].where(d["event_observed"].eq(1), np.inf), d["dss_w1"])
        delta = d["event_observed"].eq(1) & (d["event_dss"] > entry) & (d["event_dss"] <= exit_)
        in_horizon = (d["event_dss"] > landmark + cfg.gap_days) & (d["event_dss"] <= landmark + cfg.gap_days + cfg.horizon_days)

        cols: dict[str, Any] = {
            ID_COL: pd.Series(ids.astype(str), index=ids),
            "cut_odo": d["cut_odo"],
            "cut_date": d["cut_date"].dt.tz_convert(None).astype("datetime64[ns]"),
            "window_km": d["window_km"],
            "horizon_km": np.nan,
            "gap_km": np.nan,
            "label": (d["event_observed"].eq(1) & in_horizon).astype(int),
            "time_to_event_km": (d["event_odo_km"] - d["cut_odo"]).where(d["event_observed"].eq(1)),
            "event_observed": d["event_observed"].astype(int),
            "feat_landmark_day": float(landmark),
        }
        if cfg.usage_feature:
            feature_km = fwin.groupby(ID_COL, observed=True)["trip_km"].sum().reindex(ids, fill_value=0.0)
            cols["feat_log1p_km_per_day"] = np.log1p(feature_km.clip(lower=0) / feature_days)
        cols.update({c: d[c] for c in passthrough if c.startswith("static_")})
        for f in features:
            values = monthly[(f.name, "value")].unstack("month")
            weights = monthly[(f.name, "weight")].unstack("month")
            for month in values.columns:
                cols[f.value_column(month)] = values[month]
                cols[f.weight_column(month)] = weights[month]
        cols.update({
            "aux_landmark_day": float(landmark),
            "aux_gap_days": cfg.gap_days,
            "aux_horizon_days": cfg.horizon_days,
            "aux_dss_entry": entry,
            "aux_dss_exit": exit_,
            "aux_event_in_window": delta.astype(int),
            "aux_inwindow_exposure_days": (exit_ - entry).clip(lower=0),
            "aux_event_dss": d["event_dss"].where(d["event_observed"].eq(1)),
            "aux_days_to_event_after_cut": (d["event_dss"] - landmark).where(d["event_observed"].eq(1)),
            "aux_dss_window_start": d["dss_w0"],
            "aux_dss_window_end": d["dss_w1"],
            "aux_resolved_exposure_days": d["resolved_exposure_days"],
            "aux_potential_exposure_days": d["potential_exposure_days"],
            "aux_resolved_negative": d["resolved_exposure_days"].ge(cfg.e_min_days).astype(int),
        })
        for e_min in cfg.e_min_sensitivity_days:
            cols[f"aux_resolved_negative_e{e_min:g}"] = d["resolved_exposure_days"].ge(e_min).astype(int)
        # Mes del hito como índice lineal desde el mes de inicio de la ventana (sep-2025 = 0):
        # solo para la auditoría de calendario (A3).
        month_index = ((d["cut_date"].dt.year - cfg.window_start.year) * 12
                       + d["cut_date"].dt.month - cfg.window_start.month)
        cols.update({
            "aux_feature_window_days": float(feature_days),
            "aux_n_trips_window": d["n_trips"].astype(int),
            "aux_air_temp_window_mean": d["air_temp"],
            "aux_landmark_month": month_index.astype(int),
        })
        cols.update({f.raw_column: raw[(f.name, "value")] for f in features})
        cols.update({c: d[c] for c in passthrough if c.startswith("aux_")})
        row = pd.DataFrame(cols, index=ids)
        rows.append(row)

    panel = pd.concat(rows, ignore_index=True)
    fm = [c for c in panel.columns if c.startswith(FM_PREFIX)]
    fixed = [c for c in panel.columns if c not in fm]
    fm_sorted = sorted(fm, key=lambda c: (parse_fleet_column(c)[0], parse_fleet_column(c)[2], parse_fleet_column(c)[1]))
    head = [c for c in fixed if not c.startswith("aux_")]
    aux = [c for c in fixed if c.startswith("aux_")]
    panel = panel[head + fm_sorted + aux]
    panel = panel.sort_values([ID_COL, "aux_landmark_day"], ignore_index=True)
    report["rows"] = int(len(panel))
    report["months"] = sorted({parse_fleet_column(c)[1] for c in fm})
    report["duplicated_vehicle_cut_odo"] = int(panel.duplicated([ID_COL, "cut_odo"]).sum())
    return panel, report
