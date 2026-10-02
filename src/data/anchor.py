"""Traducción del evento al eje de odómetro (regla 4 de CLAUDE.md).

`IdentificationDate` está en días desde producción; `TripDatetimeStart` es
calendario. F1 encontró el puente (`docs/reproducibilidad.md`):
`ProductionDay` está en el eje del calendario con un origen común a la flota, así
que `primer_viaje − ProductionDay` es casi constante entre vehículos (IQR de 0 días).

    fecha_evento = origen + ProductionDay + IdentificationDate
    odo_evento   = interpolar esa fecha sobre los viajes del vehículo

El origen se **estima una vez** (mediana de `primer_viaje − ProductionDay` sobre los
vehículos de referencia) y se congela en la metadata del panel: no se recalcula
por corrida. Acá se estima sobre los vehículos que se le pasen —el script del panel
pasa dev— y se aplica a todos.

Lo que este módulo NO decide: qué lectura de `IdentificationDate` es la correcta.
El universo del estudio ya dejó afuera a los positivos con fecha por defecto
(`src/data/usable.py`), así que para los que quedan la lectura literal del anexo
7.3 (días desde producción) es la que se aplica.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ID_COL = "vehicle_id"
EPOCH = pd.Timestamp("1970-01-01", tz="UTC")


def estimate_origin_day(first_trip: pd.Series, production_day: pd.Series) -> tuple[float, dict[str, float]]:
    """Origen del calendario en días desde epoch, con la dispersión del puente.

    `first_trip` y `production_day` van indexadas por vehículo. Devuelve
    `(origen, {std, iqr, range, n})`; si el IQR se aleja de cero el puente dejó de
    valer y hay que volver a F1 antes de usar el resultado.
    """
    first = pd.to_datetime(first_trip, utc=True, errors="coerce")
    days = (first.dt.floor("D") - EPOCH).dt.days
    offset = (days - production_day).dropna()
    if offset.empty:
        raise ValueError("No hay vehículos con primer viaje y ProductionDay para estimar el origen")
    stats = {
        "std_days": float(offset.std()),
        "iqr_days": float(offset.quantile(0.75) - offset.quantile(0.25)),
        "range_days": float(offset.max() - offset.min()),
        "n": int(len(offset)),
    }
    origin = float(offset.median())
    logger.info("Origen del calendario: día %.0f desde epoch (IQR %.2f d, std %.2f d, n=%d)",
                origin, stats["iqr_days"], stats["std_days"], stats["n"])
    return origin, stats


def event_dates(
    vehicles: pd.DataFrame,
    origin_day: float,
    *,
    production_col: str = "static_ProductionDay",
    event_day_col: str = "event_day_since_production",
) -> pd.Series:
    """Fecha calendario del evento por vehículo (NaT para los censurados)."""
    days = origin_day + vehicles[production_col] + vehicles[event_day_col]
    return EPOCH + pd.to_timedelta(days, unit="D")


def project_dates_to_odometer(
    dates: pd.Series,
    trips: pd.DataFrame,
    *,
    date_col: str = "TripDatetimeStart",
    odo_col: str = "OdometerTripEnd",
) -> pd.DataFrame:
    """Odómetro de cada fecha, interpolando entre el viaje anterior y el posterior.

    `dates` va indexada por `vehicle_id`. Devuelve una tabla por vehículo con
    `event_odo_km`, `event_odo_source` (`interpolado` / `posterior_al_ultimo_viaje` /
    `previo_al_primer_viaje`) y los km entre los dos viajes usados, que es la
    incertidumbre de la posición.
    """
    events = dates.dropna().rename("event_date").reset_index().rename(columns={"index": ID_COL})
    events = events.sort_values("event_date")
    anchors = (
        trips[[ID_COL, date_col, odo_col]].dropna()
        .rename(columns={date_col: "trip_date", odo_col: "odo"})
        .sort_values("trip_date")
    )
    before = pd.merge_asof(events, anchors, left_on="event_date", right_on="trip_date", by=ID_COL,
                           direction="backward").rename(columns={"trip_date": "date_before", "odo": "odo_before"})
    after = pd.merge_asof(events, anchors, left_on="event_date", right_on="trip_date", by=ID_COL,
                          direction="forward")[[ID_COL, "trip_date", "odo"]]
    after = after.rename(columns={"trip_date": "date_after", "odo": "odo_after"})
    merged = before.merge(after, on=ID_COL, how="left")

    span = (merged["date_after"] - merged["date_before"]).dt.total_seconds()
    weight = ((merged["event_date"] - merged["date_before"]).dt.total_seconds() / span.where(span > 0)).clip(0, 1)
    interpolated = merged["odo_before"] + weight * (merged["odo_after"] - merged["odo_before"])

    out = pd.DataFrame({ID_COL: merged[ID_COL], "event_date": merged["event_date"]})
    out["event_odo_km"] = interpolated.where(interpolated.notna(), merged["odo_before"]).astype(float)
    out["event_odo_source"] = np.where(
        merged["odo_after"].notna() & merged["odo_before"].notna(), "interpolado",
        np.where(merged["odo_before"].notna(), "posterior_al_ultimo_viaje", "previo_al_primer_viaje"),
    )
    out["event_odo_uncertainty_km"] = (merged["odo_after"] - merged["odo_before"]).abs().astype(float)
    return out.set_index(ID_COL)


def project_odometer_to_dates(
    queries: pd.DataFrame,
    trips: pd.DataFrame,
    *,
    odo_query_col: str = "odo",
    date_col: str = "TripDatetimeStart",
    odo_col: str = "OdometerTripEnd",
) -> pd.Series:
    """Fecha en que el odómetro de cada vehículo llega a un valor: la inversa de la anterior.

    `queries` trae `vehicle_id` y `odo_query_col`. Devuelve las fechas (UTC) alineadas con
    su índice: NaT si el vehículo no tiene viajes o si su odómetro nunca llega a ese valor
    (antes del primer viaje o después del último).

    Usa la misma curva (fecha de inicio del viaje, odómetro al final) y la misma
    interpolación lineal que `project_dates_to_odometer`, sobre el odómetro acumulado para
    que sea monótona. Con una meseta (viajes de 0 km) devuelve la primera fecha en que el
    odómetro alcanza el valor. Por eso, si `x ≤ odómetro del evento`, la fecha de `x` nunca
    pasa la del evento: el origen del riesgo de una fila con evento no cae después de él.
    """
    anchors = trips[[ID_COL, date_col, odo_col]].dropna().sort_values([ID_COL, date_col], kind="stable")
    anchor_rows = anchors.groupby(ID_COL, observed=True, sort=False).indices
    days = ((anchors[date_col] - EPOCH) / pd.Timedelta(days=1)).to_numpy(dtype=float)
    odometer = anchors[odo_col].to_numpy(dtype=float)

    out = np.full(len(queries), np.nan)
    positions = queries.groupby(ID_COL, observed=True, sort=False).indices
    targets = queries[odo_query_col].to_numpy(dtype=float)
    for vehicle, rows in positions.items():
        idx = anchor_rows.get(vehicle)
        if idx is None:
            continue
        odo = np.maximum.accumulate(odometer[idx])
        t = days[idx]
        x = targets[rows]
        j = np.searchsorted(odo, x, side="left")
        # j = 0: solo vale si x es exactamente el primer odómetro. j = len: nunca llega.
        found = np.isfinite(x) & (j < len(odo)) & ((j > 0) | (x == odo[0]))
        jj = np.clip(j, 1, max(len(odo) - 1, 1))
        lo_o, hi_o = odo[jj - 1], odo[np.minimum(jj, len(odo) - 1)]
        lo_t, hi_t = t[jj - 1], t[np.minimum(jj, len(odo) - 1)]
        # Con j ≥ 1, odo[j−1] < x ≤ odo[j]: el cociente está en (0, 1] y nunca divide por 0.
        weight = np.where(hi_o > lo_o, (x - lo_o) / np.where(hi_o > lo_o, hi_o - lo_o, 1.0), 1.0)
        value = np.where(j == 0, t[0], lo_t + weight * (hi_t - lo_t))
        out[rows] = np.where(found, value, np.nan)
    return pd.Series(EPOCH + pd.to_timedelta(out, unit="D"), index=queries.index)
