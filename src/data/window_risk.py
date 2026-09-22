"""El conjunto en riesgo del panel v1 en días, dentro de la ventana del registro (F5 §3.3).

El panel v1 mide el riesgo en km desde el corte y cuenta como negativo verificado todo
horizonte de un sano, aunque el registro de eventos no lo cubriera. La Fase 1 del cure
model (`docs/memoria/f3-reloj-y-ventana-del-evento.md`) mostró dos cosas:
- el evento se ordena por días desde la venta, no por km;
- el registro solo anota eventos entre `event_window.start` y `event_window.end`
  (`configs/data/event_clock.yaml`).

Este módulo reescribe **las mismas filas** en ese reloj. Para cada fila `(vehículo, c)`:

* **Origen del riesgo** `t0`: la fecha en que el odómetro llega a `c + G`. El gap sigue en
  km (regla 1). Sale de `src/data/anchor.py::project_odometer_to_dates`, la inversa exacta
  de la proyección que ubica el evento en km, así que en una fila con evento `t0` nunca
  pasa la fecha del evento.
* **Tramo en riesgo**, en días desde `t0`: entra en `max(t0, inicio de la ventana)` y sale
  en `min(evento, fin de la ventana)`. Un sano sale en el fin de la ventana aunque su
  telemetría termine antes: el registro habría anotado el evento igual (CLAUDE.md). Una
  fila cuyo tramo es vacío no está en riesgo y no informa el ajuste.
* **Horizonte en días** `t_H − t0`, con `t_H` = la fecha en que el odómetro llega a
  `c + G + H`. Usa los km **futuros** del vehículo: va solo como `aux_` y sirve para la
  etiqueta corregida, nunca para el score.
* **Fila evaluable con la etiqueta corregida**: el horizonte `[t0, t_H]` cae entero dentro
  de la ventana. Vale igual para positivas y negativas, así que no hay atajo de posición
  en el calendario: una negativa es un "sano verificado" solo si el registro cubría su
  horizonte, y una positiva se evalúa bajo la misma condición.
* **`feat_cut_dss`**: días desde la venta en la fecha del corte (`cut_date`, el último
  viaje de la ventana). Es información del pasado y reemplaza a `feat_cut_odo` como
  posición de la fila en el reloj del evento. NaN sin `daysUntilSale`.

Las fechas son UTC. El fin de la ventana es un instante (las 0 h del día que dice el YAML),
igual que en el panel de hitos: los eventos también están fechados a las 0 h, así que un
evento del último día queda adentro.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data.anchor import project_odometer_to_dates

ID_COL = "vehicle_id"
DAY = pd.Timedelta(days=1)

#: Columnas que agrega `window_risk_columns`, en orden.
CUT_DSS = "feat_cut_dss"
ORIGIN_DSS = "aux_risk_origin_dss"
ENTRY = "aux_risk_entry_days"
EXIT = "aux_risk_exit_days"
EVENT = "aux_risk_event"
HORIZON = "aux_horizon_days"
EVALUABLE = "aux_eval_in_window"
RISK_COLUMNS = (CUT_DSS, ORIGIN_DSS, ENTRY, EXIT, EVENT, HORIZON, EVALUABLE)


@dataclass(frozen=True)
class RegistryWindow:
    """La ventana del registro de eventos, en UTC."""

    start: pd.Timestamp
    end: pd.Timestamp

    @classmethod
    def from_config(cls, event_window: dict) -> "RegistryWindow":
        start = pd.Timestamp(event_window["start"], tz="UTC")
        end = pd.Timestamp(event_window["end"], tz="UTC")
        if end <= start:
            raise ValueError(f"La ventana del registro termina ({end}) antes de empezar ({start})")
        return cls(start=start, end=end)


def _utc(values: pd.Series) -> pd.Series:
    stamps = pd.to_datetime(values)
    return stamps.dt.tz_localize("UTC") if stamps.dt.tz is None else stamps.dt.tz_convert("UTC")


def window_risk_columns(
    panel: pd.DataFrame,
    trips: pd.DataFrame,
    vehicles: pd.DataFrame,
    window: RegistryWindow,
) -> pd.DataFrame:
    """Las columnas del reloj post-venta por fila del panel, alineadas con su índice.

    `vehicles` va indexado por `vehicle_id` con `event_observed`, `sale_date` (NaT sin
    `daysUntilSale`) y `event_date` (NaT en los sanos), en UTC. `trips` son los viajes del
    vehículo con `TripDatetimeStart` y `OdometerTripEnd`.
    """
    missing = [c for c in (ID_COL, "cut_odo", "cut_date", "gap_km", "horizon_km", "event_observed")
               if c not in panel]
    if missing:
        raise KeyError(f"El panel no trae {missing}")
    missing = [c for c in ("event_observed", "sale_date", "event_date") if c not in vehicles]
    if missing:
        raise KeyError(f"`vehicles` no trae {missing}")
    unknown = sorted(set(panel[ID_COL].astype(str)) - set(vehicles.index.astype(str)))
    if unknown:
        raise ValueError(f"{len(unknown)} vehículo(s) del panel sin fila en `vehicles` (ej.: {unknown[:3]})")

    ids = panel[ID_COL].astype(str)
    table = vehicles.copy()
    table.index = table.index.astype(str)
    observed = ids.map(table["event_observed"]).astype(int)
    if not (observed.to_numpy() == panel["event_observed"].to_numpy(dtype=int)).all():
        raise ValueError("`event_observed` del panel no coincide con el de `vehicles`")
    sale = _utc(ids.map(table["sale_date"]))
    event = _utc(ids.map(table["event_date"])).where(observed.eq(1))
    if observed.eq(1).any() and event[observed.eq(1)].isna().any():
        raise ValueError("Hay filas de vehículos con evento y sin fecha del evento")

    start_odo = panel["cut_odo"].astype(float) + panel["gap_km"].astype(float)
    t0 = project_odometer_to_dates(pd.DataFrame({ID_COL: ids, "odo": start_odo}), trips)
    t_h = project_odometer_to_dates(
        pd.DataFrame({ID_COL: ids, "odo": start_odo + panel["horizon_km"].astype(float)}), trips
    )

    # El sano sale en el fin de la ventana; el fallado, en su evento (todos caen adentro en
    # dev, y si alguno cayera después, el registro no lo habría visto: sale en el fin).
    end = pd.Series(window.end, index=panel.index)
    exit_date = event.where(event.notna() & (event < end), end)
    entry = ((window.start - t0) / DAY).clip(lower=0.0)
    exit_ = (exit_date - t0) / DAY
    in_window = event.notna() & (event >= window.start) & (event <= window.end)
    at_risk = t0.notna() & (exit_ > entry)
    risk_event = (in_window & at_risk).astype(int)

    out = pd.DataFrame(index=panel.index)
    out[CUT_DSS] = (_utc(panel["cut_date"]) - sale) / DAY
    out[ORIGIN_DSS] = (t0 - sale) / DAY
    out[ENTRY] = entry.where(t0.notna())
    out[EXIT] = exit_.where(t0.notna())
    out[EVENT] = risk_event
    out[HORIZON] = (t_h - t0) / DAY
    out[EVALUABLE] = (t0.notna() & t_h.notna() & (t0 >= window.start) & (t_h <= window.end)).astype(int)
    return out


def risk_summary(frame: pd.DataFrame, *, max_horizon_days: float | None = None) -> dict[str, float]:
    """Conteos del tramo en riesgo de un conjunto de filas (para el embudo, sin features)."""
    entry, exit_ = frame[ENTRY].to_numpy(float), frame[EXIT].to_numpy(float)
    if max_horizon_days is not None:
        exit_ = np.minimum(exit_, float(max_horizon_days))
    exposure = np.where(np.isfinite(entry) & np.isfinite(exit_), np.clip(exit_ - entry, 0.0, None), 0.0)
    at_risk = exposure > 0
    return {
        "rows": int(len(frame)),
        "origin_unknown": int(np.isnan(frame[ENTRY].to_numpy(float)).sum()),
        "at_risk": int(at_risk.sum()),
        "exposure_days_total": float(exposure.sum()),
        "exposure_days_median_at_risk": float(np.median(exposure[at_risk])) if at_risk.any() else float("nan"),
    }
