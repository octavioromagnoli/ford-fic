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
    # Conservar solo la ventana de riesgo de cada vehículo: los H/Δ cortes previos al
    # evento (real o ficticio). Requiere `pseudo_event.enabled`, porque un sano sin
    # evento ficticio no tiene ventana de riesgo que conservar.
    #
    # Por qué existe: dentro de un vehículo, **la etiqueta ES la posición** —las filas
    # positivas son, por definición, los últimos H/Δ cortes de la serie—, así que la
    # posición relativa separa sola con P = 0,83 y ninguna feature es comparable entre
    # grupos. Cortarle la serie al sano en un evento ficticio (`assign_pseudo_events`)
    # arregla dónde TERMINA la serie, pero no esto: el sano sigue aportando filas en
    # todas las posiciones y el positivo solo en la última. Quedándose en los dos grupos
    # con la misma ventana, los dos aportan cortes en el mismo tramo y la posición deja
    # de decir de qué grupo es la fila.
    #
    # El precio: se pierden los negativos tempranos de los vehículos con evento (filas
    # legítimas), y el panel queda como un caso-control emparejado a nivel ventana.
    window_only: bool = False

    def __post_init__(self) -> None:
        for key in ("window_km", "gap_km", "horizon_km", "cut_step_km"):
            value = getattr(self, key)
            if value is None or float(value) <= 0:
                raise ValueError(f"label.{key} tiene que ser > 0 (vale {value}). F2 lo fija en panel_v1.yaml")
        if self.censored_policy not in {"drop", "keep_as_negative"}:
            raise ValueError(f"label.censored_policy desconocida: {self.censored_policy}")
        if self.window_only and self.horizon_km < self.cut_step_km:
            raise ValueError("label.window_only con H < Δ deja ventanas de riesgo vacías")


def vehicle_cuts(
    first_odo: float,
    last_odo: float,
    event_odo: float | None,
    cfg: LabelConfig,
    *,
    pseudo_event_odo: float | None = None,
) -> pd.DataFrame:
    """Cortes de un vehículo con su etiqueta. Vacío si no entra ni una ventana completa.

    `pseudo_event_odo` solo aplica a vehículos **sanos** (ver `assign_pseudo_events`): la
    serie se corta en `P − G`, igual que la de un vehículo con evento se corta en `E − G`,
    pero todas las etiquetas siguen siendo 0 y `time_to_event_km` sigue siendo NaN. No es
    un evento: es un punto de corte de la observación.

    Con `cfg.window_only` se conserva **solo la ventana de riesgo**: los `H/Δ` cortes
    anteriores al evento (real o ficticio). Es lo que iguala la posición entre los dos
    grupos; el porqué está en `LabelConfig.window_only`. Vive en el config y no en la
    firma a propósito: es una decisión del panel, no de cada llamada.
    """
    step = cfg.cut_step_km
    start = np.ceil((first_odo + cfg.window_km) / step) * step
    has_event = event_odo is not None and np.isfinite(event_odo)
    has_pseudo = pseudo_event_odo is not None and np.isfinite(pseudo_event_odo)
    if cfg.window_only and (has_event or has_pseudo):
        # Solo la ventana de riesgo: los `H/Δ` cortes previos al evento (real o ficticio).
        # Es lo que iguala la posición entre grupos, ver `LabelConfig.window_only`.
        anchor = event_odo if has_event else pseudo_event_odo
        start = max(start, np.ceil((anchor - cfg.gap_km - cfg.horizon_km) / step) * step)
    if has_event and has_pseudo:
        raise ValueError("Un vehículo con evento no puede tener además un evento ficticio")
    if has_event:
        end = event_odo - cfg.gap_km
    elif has_pseudo:
        end = pseudo_event_odo - cfg.gap_km
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


def assign_pseudo_events(
    vehicles: pd.DataFrame,
    spans: pd.DataFrame,
    cfg: LabelConfig,
    *,
    reference_vehicles: set[str],
    seed: int = 42,
) -> tuple[pd.Series, dict[str, Any]]:
    """Un **evento ficticio** para cada vehículo sano: dónde se le corta la serie.

    El problema que resuelve (medido en `docs/memoria/f3-emparejado-posicion-vs-calendario.md`):
    la serie de un vehículo con evento termina en `E − G` —temprano, y a pocos km— y la de
    un sano termina donde se acaba el historial, o sea tarde y con más kilómetros. Por eso
    **"estar al final de la propia serie" significa cosas distintas en cada grupo**: el 77%
    de las filas positivas cae en el último cuarto de su serie contra el 30% de las sanas, y
    esa asimetría separa sola con P = 0,83. No se puede arreglar emparejando celdas después
    —odómetro, mes y posición son incompatibles entre sí—, porque se genera **al construir
    las series**.

    La solución es de diseño y es la estándar del sesgo de tiempo inmortal (asignación de
    fecha índice / *risk-set sampling*): a cada sano se le sortea un punto `P` de la
    **distribución de odómetros de evento de los positivos** y se le corta la serie en
    `P − G`, exactamente como a un positivo. Con eso los dos grupos terminan igual por
    construcción: la posición dentro de la serie deja de ser informativa, y el emparejado
    por odómetro y mes puede seguir haciendo su trabajo sin competir con una tercera
    dimensión.

    `P` tiene que caer en el rango **factible** del vehículo:

    - `P >= first_odo + W + G`, o no entra ni un corte con ventana completa;
    - `P <= last_odo − H`, o el último corte no es verificable (hay que haber observado
      hasta `c + G + H` para afirmar que no hubo evento; es la misma exigencia que hoy
      impone `censored_policy: drop`, solo que ahora la ancla el punto sorteado).

    Se sortea **con reemplazo entre los valores de referencia que caen en ese rango**. Un
    vehículo sin ningún valor factible se descarta y se cuenta: preferimos perderlo antes
    que recortar la distribución a un borde, que es justamente el sesgo que se quiere sacar.

    La referencia se toma **solo de `reference_vehicles` (dev)**: el test no se mira ni
    para esto, igual que en `match_healthy_cuts` y en el anclaje del calendario.
    """
    is_event = vehicles["event_observed"].eq(1)
    reference = vehicles.loc[
        is_event & vehicles.index.astype(str).isin(reference_vehicles), "event_odo_km"
    ].dropna()
    if reference.empty:
        raise ValueError("No hay odómetros de evento de referencia (dev) para sortear eventos ficticios")

    rng = np.random.default_rng(seed)
    pool = np.sort(reference.to_numpy(dtype="float64"))
    healthy = vehicles.index[~is_event]
    out = pd.Series(np.nan, index=vehicles.index, dtype="float64")
    n_infeasible_range = 0
    n_no_reference = 0
    for vehicle_id in healthy:
        if vehicle_id not in spans.index:
            continue
        first_odo = float(spans.at[vehicle_id, "first_odo"])
        last_odo = float(spans.at[vehicle_id, "last_odo"])
        lo = first_odo + cfg.window_km + cfg.gap_km
        hi = last_odo - cfg.horizon_km
        if hi < lo:
            n_infeasible_range += 1
            continue
        feasible = pool[(pool >= lo) & (pool <= hi)]
        if feasible.size == 0:
            n_no_reference += 1
            continue
        out.at[vehicle_id] = float(rng.choice(feasible))

    assigned = out.notna()
    summary = {
        "enabled": True,
        "seed": seed,
        "reference": "dev_event_vehicles",
        "n_reference": int(len(reference)),
        "reference_odo_km_median": float(np.median(pool)),
        "n_healthy": int((~is_event).sum()),
        "n_assigned": int(assigned.sum()),
        "n_dropped_infeasible_range": n_infeasible_range,
        "n_dropped_no_reference_in_range": n_no_reference,
        "pseudo_odo_km_median": float(out[assigned].median()) if assigned.any() else None,
    }
    logger.info("Eventos ficticios: %d de %d sanos (%d sin rango factible, %d sin referencia en rango) "
                "· odómetro mediano %s km",
                summary["n_assigned"], summary["n_healthy"], n_infeasible_range, n_no_reference,
                f"{summary['pseudo_odo_km_median']:.0f}" if summary["pseudo_odo_km_median"] else "-")
    return out, summary


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
    dropped: dict[str, list[str]] = {"sin_viajes": [], "sin_ventana_completa": [],
                                     "sin_corte_verificable": [], "sin_evento_ficticio": []}
    rows_dropped_qc = 0
    for vehicle_id, info in vehicles.iterrows():
        vtrips = trip_groups.get(vehicle_id)
        if vtrips is None or vtrips.empty:
            dropped["sin_viajes"].append(str(vehicle_id))
            continue
        first_odo = float(vtrips["OdometerTripEnd"].min())
        last_odo = float(vtrips["OdometerTripEnd"].max())
        event_odo = float(info["event_odo_km"]) if int(info["event_observed"]) == 1 else None
        # Evento ficticio: solo en sanos, y solo si `vehicles` lo trae (lo asigna
        # `assign_pseudo_events`). Corta la serie como si fuera un evento; la etiqueta
        # sigue siendo 0.
        pseudo = info.get(PSEUDO_EVENT_COL) if PSEUDO_EVENT_COL in vehicles.columns else None
        pseudo_odo = float(pseudo) if pseudo is not None and pd.notna(pseudo) else None
        if pseudo_odo is not None and int(info["event_observed"]) == 1:
            pseudo_odo = None
        if PSEUDO_EVENT_COL in vehicles.columns and int(info["event_observed"]) == 0 and pseudo_odo is None:
            # Sano sin punto de corte factible: no entra al panel, y se cuenta aparte.
            dropped["sin_evento_ficticio"].append(str(vehicle_id))
            continue
        cuts = vehicle_cuts(first_odo, last_odo, event_odo, cfg, pseudo_event_odo=pseudo_odo)
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


AUX_POSITION = f"{AUX_PREFIX}cut_position"
# Columna de `vehicles` con el evento ficticio de cada sano (la escribe
# `assign_pseudo_events`; `build_panel` la lee si está).
PSEUDO_EVENT_COL = "pseudo_event_odo_km"


def add_cut_position(panel: pd.DataFrame) -> pd.DataFrame:
    """Posición relativa del corte dentro de la serie **completa** de su vehículo, en (0, 1].

    Se calcula una vez, **antes** de muestrear los sanos, y viaja en el panel como
    `aux_cut_position`. Las dos razones son la misma:

    - **Emparejar.** Es la tercera dimensión de `match_healthy_cuts`.
    - **Auditar.** `scripts/audit_sequence.py` venía recalculando la posición sobre el
      panel ya muestreado, y ahí está distorsionada: de un vehículo con evento se
      conservan todos los cortes, pero de un sano sobrevive un subconjunto, así que su
      rango se comprime y dos filas con el mismo `_frac` no están en el mismo punto de
      sus historias. La columna guarda la posición real.

    No entra al modelo (prefijo `aux_`), y no podría: la posición del corte en la serie
    separa sola con P = 0,83 y es la etiqueta por construcción
    (`docs/memoria/f3-posicion-en-la-serie.md`).
    """
    grouped = panel.groupby(ID_COL, observed=True)["cut_odo"]
    out = panel.copy()
    out[AUX_POSITION] = grouped.rank(method="first") / grouped.transform("size")
    return out


def match_healthy_cuts(
    panel: pd.DataFrame,
    reference_vehicles: set[str],
    *,
    match_on: tuple[str, ...] = ("cut_odo", "cut_date"),
    odo_bin_km: float = 2000.0,
    date_freq: str = "M",
    position_bins: int = 4,
    max_shortfall: float = 0.1,
    seed: int = 42,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Máscara de filas a conservar: todas las de vehículos con evento + un subconjunto de sanos.

    El subconjunto sano es el más grande cuya distribución sobre las celdas de
    `match_on` reproduce la de las filas positivas de `reference_vehicles` (dev),
    tolerando un `max_shortfall` global en celdas donde no hay sanos suficientes.

    **`cut_position` como tercera dimensión de emparejado (F3).** Las filas positivas de
    un vehículo son, por construcción, sus ÚLTIMOS cortes: su serie termina en `E − G` y
    las positivas son las últimas `H/Δ`. Así que emparejar solo por odómetro y mes deja
    una asimetría abierta —las positivas se apilan al final de su serie y los sanos que
    las acompañan vienen de series largas, que son otra población— y la medición se
    diluye: auditando por estratos de posición, la señal térmica separa con 0,74 en los
    cortes tempranos y con 0,56 en los últimos, contra 0,59 agrupado
    (`docs/memoria/f3-esfuerzo-de-control-y-dosis.md` §2b). Agregar `cut_position` a
    `match_on` empareja cada positiva con sanos que están **en el mismo punto de su
    propia serie**, que es la comparación que la auditoría venía exigiendo a mano.

    La posición se calcula sobre el panel COMPLETO (antes de muestrear), porque es la
    posición dentro de la serie real del vehículo, no dentro de lo que sobrevivió.
    """
    if "cut_odo" not in match_on and "cut_date" not in match_on:
        raise ValueError("match_on tiene que incluir `cut_odo` y/o `cut_date`")
    cells = pd.Series("", index=panel.index, dtype="object")
    if "cut_odo" in match_on:
        cells = cells + "odo=" + (np.floor(panel["cut_odo"] / odo_bin_km) * odo_bin_km).astype(int).astype(str)
    if "cut_date" in match_on:
        period = pd.to_datetime(panel["cut_date"]).dt.to_period(date_freq).astype(str).fillna("NaT")
        cells = cells + "|date=" + period
    if "cut_position" in match_on:
        if AUX_POSITION not in panel.columns:
            raise KeyError(
                f"Emparejar por `cut_position` necesita `{AUX_POSITION}` en el panel. "
                "Lo agrega `add_cut_position()` justo después de `build_panel()`."
            )
        rank_frac = panel[AUX_POSITION]
        # Bins fijos en [0, 1], no cuantiles: los cuantiles se calcularían sobre la
        # mezcla de positivas y sanas y moverían el corte según la composición.
        edges = np.linspace(0.0, 1.0, int(position_bins) + 1)
        binned = pd.cut(rank_frac, bins=edges, labels=False, include_lowest=True)
        cells = cells + "|pos=" + binned.astype("Int64").astype(str)

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
        "position_bins": int(position_bins) if "cut_position" in match_on else None,
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


def _column_order(panel: pd.DataFrame) -> list[str]:
    head = [c for c in HEAD_COLUMNS if c in panel.columns]
    feats = sorted(c for c in panel.columns if c.startswith(FEAT_PREFIX))
    statics = sorted(c for c in panel.columns if c.startswith("static_"))
    aux = sorted(c for c in panel.columns if c.startswith(AUX_PREFIX))
    rest = [c for c in panel.columns if c not in set(head + feats + statics + aux)]
    return head + feats + statics + aux + rest
