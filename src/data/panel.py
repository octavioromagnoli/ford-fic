"""Armado del panel: cortes, etiqueta, muestreo de los sanos y ensamblado.

Una fila por `(vehículo, punto de corte)`. Para cada vehículo:

- los cortes van sobre una grilla de múltiplos de Δ, desde el primer corte con una
  ventana completa (`first_odo + W`) hacia adelante;
- en un vehículo **con evento** en `E` los cortes llegan hasta `E − G`: los que caen
  dentro del gap `(E − G, E)` no se etiquetan (el evento está demasiado cerca para
  que la fila sea anticipación), y todo lo posterior a `E` se descarta —después de
  `IdentificationDate` el uso cambia (ver `docs/memoria/f2-eda-revision-y-features.md`)—.
  `label = 1` si `E ∈ [c + G, c + G + H]`, es decir si `c >= E − G − H`;
- en un vehículo **sano** el último corte verificable es `last_odo − G − H`: más allá
  no sabemos si el evento habría caído dentro del horizonte (`censored_policy:
  drop`). Con `keep_as_negative` se conservan como 0, asumiendo que no hubo evento.

**Muestreo de los sanos.** Los eventos de dev caen entre sep-2025 y mar-2026 y a
odómetros de 1.000–13.000 km; la exposición de los sanos se concentra en 2026 y va
hasta 80.000 km. Sin emparejar, el modelo aprende "odómetro alto o ventana en 2026
⇒ sano". `match_healthy_cuts` se queda con el subconjunto más grande de filas sanas
cuya distribución conjunta de (bin de odómetro, mes del corte) reproduce la de las
filas positivas **de dev** (el test no se mira ni para esto).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.features.windows import (
    AUX_PREFIX,
    FEAT_PREFIX,
    FeatureSpec,
    VehicleArrays,
    compute_window_features,
    required_columns,
)

logger = logging.getLogger(__name__)

ID_COL = "vehicle_id"
HEAD_COLUMNS = ["vehicle_id", "cut_odo", "cut_date", "window_km", "horizon_km", "gap_km",
                "label", "time_to_event_km", "event_observed"]


@dataclass(frozen=True)
class LabelConfig:
    window_km: float
    gap_km: float
    horizon_km: float
    cut_step_km: float
    censored_policy: str = "drop"          # "drop" | "keep_as_negative"
    min_trips_in_window: int = 5
    min_km_covered_frac: float = 0.5       # km recorridos en la ventana / W

    def __post_init__(self) -> None:
        for key in ("window_km", "gap_km", "horizon_km", "cut_step_km"):
            value = getattr(self, key)
            if value is None or float(value) <= 0:
                raise ValueError(f"label.{key} tiene que ser > 0 (vale {value}). F2 lo fija en panel_v1.yaml")
        if self.censored_policy not in {"drop", "keep_as_negative"}:
            raise ValueError(f"label.censored_policy desconocida: {self.censored_policy}")


def vehicle_cuts(first_odo: float, last_odo: float, event_odo: float | None, cfg: LabelConfig) -> pd.DataFrame:
    """Cortes de un vehículo con su etiqueta. Vacío si no entra ni una ventana completa."""
    step = cfg.cut_step_km
    start = np.ceil((first_odo + cfg.window_km) / step) * step
    has_event = event_odo is not None and np.isfinite(event_odo)
    if has_event:
        end = event_odo - cfg.gap_km
    elif cfg.censored_policy == "drop":
        end = last_odo - cfg.gap_km - cfg.horizon_km
    else:
        end = last_odo
    if end < start:
        return pd.DataFrame(columns=["cut_odo", "label", "time_to_event_km"])
    cuts = np.arange(start, end + 1e-6, step)
    if has_event:
        tte = event_odo - cuts
        label = (tte <= cfg.gap_km + cfg.horizon_km).astype(int)
    else:
        tte = np.full(cuts.shape, np.nan)
        label = np.zeros(cuts.shape, dtype=int)
    return pd.DataFrame({"cut_odo": cuts, "label": label, "time_to_event_km": tte})


def build_panel(
    trips: pd.DataFrame,
    signals: pd.DataFrame | None,
    vehicles: pd.DataFrame,
    specs: list[FeatureSpec],
    cfg: LabelConfig,
    *,
    static_columns: list[str],
    static_excluded: list[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Panel completo (sin muestrear los sanos) + informe de qué se descartó y por qué.

    `vehicles` va indexado por `vehicle_id` con `event_observed`, `event_odo_km`
    (NaN si censurado) y las `static_*`. Solo entran los vehículos de `vehicles`.
    """
    trip_cols = required_columns(specs, "trips")
    signal_cols = required_columns(specs, "signals")
    missing = [c for c in trip_cols if c not in trips.columns]
    if missing:
        raise KeyError(f"`trips` no tiene columnas que el YAML de features pide: {missing}")
    if signals is not None:
        missing = [c for c in signal_cols if c not in signals.columns]
        if missing:
            raise KeyError(f"`signals` no tiene columnas que el YAML de features pide: {missing}")

    trip_groups = {k: g for k, g in trips.groupby(ID_COL, observed=True, sort=False)}
    signal_groups = ({k: g for k, g in signals.groupby(ID_COL, observed=True, sort=False)}
                     if signals is not None else {})

    frames: list[pd.DataFrame] = []
    dropped: dict[str, list[str]] = {"sin_viajes": [], "sin_ventana_completa": [], "sin_corte_verificable": []}
    rows_dropped_qc = 0
    for vehicle_id, info in vehicles.iterrows():
        vtrips = trip_groups.get(vehicle_id)
        if vtrips is None or vtrips.empty:
            dropped["sin_viajes"].append(str(vehicle_id))
            continue
        first_odo = float(vtrips["OdometerTripEnd"].min())
        last_odo = float(vtrips["OdometerTripEnd"].max())
        event_odo = float(info["event_odo_km"]) if int(info["event_observed"]) == 1 else None
        cuts = vehicle_cuts(first_odo, last_odo, event_odo, cfg)
        if cuts.empty:
            key = "sin_ventana_completa" if (last_odo - first_odo) < cfg.window_km else "sin_corte_verificable"
            dropped[key].append(str(vehicle_id))
            continue

        t_arrays = VehicleArrays.from_frame(
            vtrips, position_col="OdometerTripEnd", columns=trip_cols, trip_km_col="trip_km",
            time_cols=("TripDatetimeStart", "TripDatetimeEnd"),
        )
        vsignals = signal_groups.get(vehicle_id)
        s_arrays = (VehicleArrays.from_frame(vsignals, position_col="OdometerValue", columns=signal_cols)
                    if vsignals is not None and not vsignals.empty else None)
        feats = compute_window_features(t_arrays, s_arrays, cuts["cut_odo"].to_numpy(),
                                        window_km=cfg.window_km, specs=specs)
        block = pd.concat([cuts.reset_index(drop=True), feats.reset_index(drop=True)], axis=1)

        ok = block[f"{FEAT_PREFIX}n_trips_window"].ge(cfg.min_trips_in_window) & block[
            f"{FEAT_PREFIX}window_km_covered"].ge(cfg.min_km_covered_frac * cfg.window_km)
        rows_dropped_qc += int((~ok).sum())
        block = block.loc[ok]
        if block.empty:
            dropped["sin_ventana_completa"].append(str(vehicle_id))
            continue
        block.insert(0, ID_COL, str(vehicle_id))
        block["event_observed"] = int(info["event_observed"])
        for column in static_columns:
            block[f"static_{column}"] = info.get(f"static_{column}")
        for column in static_excluded:
            block[f"{AUX_PREFIX}static_{column}"] = info.get(f"static_{column}")
        frames.append(block)

    if not frames:
        raise RuntimeError("El panel quedó vacío: revisá W/G/H/Δ contra el historial disponible")
    panel = pd.concat(frames, ignore_index=True)
    panel["window_km"] = float(cfg.window_km)
    panel["horizon_km"] = float(cfg.horizon_km)
    panel["gap_km"] = float(cfg.gap_km)
    panel = panel[_column_order(panel)]

    # El informe no cuenta positivos: el panel trae dev y test, y una cifra global permite
    # deducir la de test por diferencia. Los positivos se reportan sobre dev, en el script.
    report = {
        "n_vehicles_in": int(len(vehicles)),
        "n_vehicles_out": int(panel[ID_COL].nunique()),
        "rows": int(len(panel)),
        "rows_dropped_qc": rows_dropped_qc,
        "dropped_vehicles": {k: v for k, v in dropped.items()},
    }
    logger.info("Panel: %d filas, %d vehículos; QC descartó %d filas; vehículos sin cortes: %s",
                report["rows"], report["n_vehicles_out"], rows_dropped_qc,
                {k: len(v) for k, v in dropped.items()})
    return panel, report


def match_healthy_cuts(
    panel: pd.DataFrame,
    reference_vehicles: set[str],
    *,
    match_on: tuple[str, ...] = ("cut_odo", "cut_date"),
    odo_bin_km: float = 2000.0,
    date_freq: str = "M",
    max_shortfall: float = 0.1,
    seed: int = 42,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Máscara de filas a conservar: todas las de vehículos con evento + un subconjunto de sanos.

    El subconjunto sano es el más grande cuya distribución sobre las celdas de
    `match_on` reproduce la de las filas positivas de `reference_vehicles` (dev),
    tolerando un `max_shortfall` global en celdas donde no hay sanos suficientes.
    """
    if "cut_odo" not in match_on and "cut_date" not in match_on:
        raise ValueError("match_on tiene que incluir `cut_odo` y/o `cut_date`")
    cells = pd.Series("", index=panel.index, dtype="object")
    if "cut_odo" in match_on:
        cells = cells + "odo=" + (np.floor(panel["cut_odo"] / odo_bin_km) * odo_bin_km).astype(int).astype(str)
    if "cut_date" in match_on:
        period = pd.to_datetime(panel["cut_date"]).dt.to_period(date_freq).astype(str).fillna("NaT")
        cells = cells + "|date=" + period

    is_ref_positive = panel["label"].eq(1) & panel[ID_COL].astype(str).isin(reference_vehicles)
    is_healthy = panel["event_observed"].eq(0)
    if not is_ref_positive.any():
        raise ValueError("No hay filas positivas de referencia para emparejar los sanos")
    target = cells[is_ref_positive].value_counts(normalize=True)
    avail = cells[is_healthy].value_counts()
    avail = avail.reindex(target.index).fillna(0).astype(int)

    # n más grande tal que la suma de min(avail, n*target) cubre (1 - shortfall) de n.
    n_total = int(is_healthy.sum())
    chosen = 0
    for n in range(n_total, 0, -1):
        quota = np.minimum(avail.to_numpy(), np.round(n * target.to_numpy()))
        if quota.sum() >= (1.0 - max_shortfall) * n:
            chosen = n
            break
    quota = pd.Series(np.minimum(avail.to_numpy(), np.round(chosen * target.to_numpy())).astype(int), index=target.index)

    rng = np.random.default_rng(seed)
    keep = ~is_healthy.to_numpy()          # todo lo que tenga evento se conserva
    healthy_idx = panel.index[is_healthy]
    healthy_cells = cells[is_healthy]
    for cell, q in quota.items():
        if q <= 0:
            continue
        candidates = healthy_idx[healthy_cells.eq(cell).to_numpy()]
        take = rng.choice(candidates, size=min(int(q), len(candidates)), replace=False)
        keep[panel.index.get_indexer(take)] = True

    kept_healthy = int(keep[is_healthy.to_numpy()].sum())
    summary = {
        "match_on": list(match_on), "odo_bin_km": odo_bin_km, "date_freq": date_freq,
        "n_reference_positive_rows": int(is_ref_positive.sum()),
        "n_cells": int(len(target)),
        "cells_without_healthy": int((avail == 0).sum()),
        "healthy_rows_before": n_total, "healthy_rows_after": kept_healthy,
        "healthy_vehicles_before": int(panel.loc[is_healthy, ID_COL].nunique()),
        "healthy_vehicles_after": int(panel.loc[keep & is_healthy.to_numpy(), ID_COL].nunique()),
        "shortfall_frac": float(1.0 - quota.sum() / chosen) if chosen else None,
    }
    logger.info("Emparejado de sanos sobre %s: %d -> %d filas (%d -> %d vehículos), %d celdas, %d sin sanos",
                match_on, n_total, kept_healthy, summary["healthy_vehicles_before"],
                summary["healthy_vehicles_after"], summary["n_cells"], summary["cells_without_healthy"])
    return keep, summary


PANEL_KEY = ["vehicle_id", "cut_odo"]


def attach_columns(panel: pd.DataFrame, extra: pd.DataFrame, *, key: list[str] = PANEL_KEY) -> pd.DataFrame:
    """Variante del panel con columnas `feat_*`/`aux_*` nuevas: mismas filas, mismo orden.

    Es la forma de medir una familia de features como ablación limpia: mismas filas
    ⇒ misma huella ⇒ mismos folds de `splits.json`. `extra` trae `key` + las columnas
    nuevas; una fila del panel sin pareja queda en NaN (el imputador del `Pipeline` la
    maneja). Falla si una columna nueva pisa una existente o si la clave no es única.
    """
    new = [c for c in extra.columns if c not in key]
    clash = sorted(set(new) & set(panel.columns))
    if clash:
        raise ValueError(f"Las columnas nuevas pisan columnas del panel: {clash}")
    bad = [c for c in new if not c.startswith((FEAT_PREFIX, AUX_PREFIX))]
    if bad:
        raise ValueError(f"Toda columna agregada al panel es `feat_` o `aux_`: {bad}")
    out = panel.merge(extra[key + new], on=key, how="left", validate="1:1")
    if len(out) != len(panel):
        raise RuntimeError("El merge cambió la cantidad de filas del panel")
    return out[_column_order(out)]


def _column_order(panel: pd.DataFrame) -> list[str]:
    head = [c for c in HEAD_COLUMNS if c in panel.columns]
    feats = sorted(c for c in panel.columns if c.startswith(FEAT_PREFIX))
    statics = sorted(c for c in panel.columns if c.startswith("static_"))
    aux = sorted(c for c in panel.columns if c.startswith(AUX_PREFIX))
    rest = [c for c in panel.columns if c not in set(head + feats + statics + aux)]
    return head + feats + statics + aux + rest
