#!/usr/bin/env python
"""Genera `data/processed/panel_dummy.parquet` con el esquema exacto del contrato.

Este es el entregable que desbloquea a Track B y a Track C: tiene las mismas
columnas, dtypes y nombres de feature que va a tener el panel real de F2, pero
valores aleatorios. Cuando el panel real aterrice, B y C solo cambian el path del
config.

Los valores son ruido plausible a propósito: el modelo dummy tiene que dar basura.
`signal_strength > 0` inyecta una señal débil en la familia B (saturación del DPF)
para verificar que las curvas de anticipación no son degeneradas; no es un dato,
es un test.

    python scripts/make_dummy.py --config configs/data/dummy_v1.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ensure_dir, load_config, resolve_path, set_seed  # noqa: E402
from src.eval.splits import make_splits, save_splits, split_options  # noqa: E402

logger = logging.getLogger("make_dummy")

# Las features del panel REAL, tal como las materializa `scripts/build_dataset.py`
# desde `configs/data/features_v1.yaml`: (media, desvío, mínimo, máximo) plausibles,
# tomados de los rangos del panel v1 sobre dev. Si el YAML de features cambia, esta
# lista se actualiza con él: es lo que permite que B y C trabajen contra el mismo esquema.
FEATURE_SPECS: dict[str, tuple[float, float, float | None, float | None]] = {
    # A · régimen térmico y trayectos cortos
    "feat_idle_frac": (0.25, 0.15, 0.0, 1.0),
    "feat_idle_per_1000km": (20.0, 25.0, 0.0, None),
    "feat_idle_frac_trend": (0.0, 0.17, -1.0, 1.0),
    "feat_short_trip_frac_5km": (0.38, 0.16, 0.0, 1.0),
    "feat_trips_below_regime_temp_frac": (0.15, 0.13, 0.0, 1.0),
    "feat_below_regime_moving_frac": (0.06, 0.08, 0.0, 1.0),
    "feat_below_regime_frac_trend": (0.0, 0.17, -1.0, 1.0),
    "feat_engine_temp_avg_median": (69.0, 9.5, 20.0, 110.0),
    "feat_engine_temp_max_median": (96.0, 4.5, 20.0, 130.0),
    "feat_engine_temp_amplitude_mean": (60.0, 8.0, 0.0, None),
    "feat_coolant_temp_end_mean": (91.0, 4.7, 20.0, 110.0),
    "feat_coolant_temp_start_median": (30.0, 11.0, -20.0, 110.0),
    "feat_cold_start_frac": (0.64, 0.14, 0.0, 1.0),
    "feat_chained_trip_frac": (0.5, 0.13, 0.0, 1.0),
    "feat_trip_distance_median_km": (7.0, 6.0, 1.0, None),
    "feat_trip_distance_p25_km": (3.0, 2.5, 1.0, None),
    "feat_trip_duration_median_min": (22.0, 10.0, 1.0, None),
    # B · salud del ciclo de regeneración (desde trips.AirRegeneration*)
    "feat_regenerations_per_1000km": (3.1, 1.8, 0.0, None),
    "feat_regenerations_trend_per_1000km": (0.0, 2.3, None, None),
    "feat_distance_between_regen_mean_km": (290.0, 130.0, 10.0, None),
    "feat_distance_between_regen_trend": (30.0, 120.0, None, None),
    "feat_km_since_last_regen": (160.0, 150.0, 0.0, None),
    "feat_regen_residual_mean": (10.0, 12.0, 0.0, 100.0),
    "feat_regen_start_level_mean": (75.0, 15.0, 0.0, 100.0),
    "feat_dpf_end_mean": (42.0, 13.5, 0.0, 100.0),
    "feat_dpf_end_max": (90.0, 12.0, 0.0, 100.0),
    "feat_dpf_end_slope_per_1000km": (0.0, 50.0, None, None),
    "feat_dpf_positive_delta_frac": (0.47, 0.12, 0.0, 1.0),
    "feat_dpf_saturated_frac": (0.03, 0.09, 0.0, 1.0),
    "feat_filter_abnormal_end_frac": (0.02, 0.08, 0.0, 1.0),
    "feat_filter_cleaning_auto_end_frac": (0.002, 0.007, 0.0, 1.0),
    "feat_manual_regen_ever": (0.0, 0.05, 0.0, 1.0),
    # C · contexto de uso
    "feat_speed_kmh_mean": (25.0, 10.7, 3.0, 130.0),
    "feat_trips_below_30kmh_frac": (0.70, 0.20, 0.0, 1.0),
    "feat_moving_trips_per_1000km": (54.0, 40.0, 1.0, None),
    "feat_km_per_day": (44.0, 22.0, 0.5, None),
    "feat_trips_per_day": (3.2, 1.4, 0.1, None),
    "feat_hours_between_trips_median": (0.6, 0.5, 0.05, None),
    # D · severidad y proxies baratos
    "feat_msg_full_per_1000km": (12.0, 29.0, 0.0, None),
    "feat_msg_overloaded_per_1000km": (0.6, 4.3, 0.0, None),
    "feat_msg_overloaded_ever": (0.14, 0.35, 0.0, 1.0),
    "feat_msg_over_limit_per_1000km": (0.5, 18.0, 0.0, None),
    "feat_msg_over_limit_ever": (0.01, 0.10, 0.0, 1.0),
    "feat_msg_at_limit_ever": (0.002, 0.05, 0.0, 1.0),
    "feat_msg_cleaning_auto_per_1000km": (1.0, 3.3, 0.0, None),
    "feat_msg_cleaning_auto_trend_per_1000km": (0.0, 3.8, None, None),
    "feat_msg_abnormal_frac": (0.03, 0.07, 0.0, 1.0),
    "feat_msgs_per_1000km": (330.0, 255.0, 1.0, None),
    "feat_oil_life_mean": (70.0, 24.0, 0.0, 100.0),
    "feat_oil_life_slope_per_1000km": (-5.0, 19.0, None, None),
    "feat_fuel_pct_per_100km_median": (14.7, 3.3, 0.0, None),
    # cobertura de la ventana (control de calidad, no hipótesis física)
    "feat_n_trips_window": (85.0, 35.0, 5.0, None),
    "feat_window_km_covered": (970.0, 120.0, 500.0, None),
}

# Features por las que se filtra la señal sintética cuando `signal_strength > 0`.
SIGNAL_FEATURES = (
    "feat_dpf_end_slope_per_1000km",
    "feat_dpf_end_mean",
    "feat_distance_between_regen_trend",
    "feat_short_trip_frac_5km",
)

# Mismo esquema de estáticas que el panel real: `static_*` entra al modelo, `aux_static_*`
# queda en el panel solo para auditar (Engine, ModelSeries y ProductionDay están fuera
# del set base: ver `features.static_excluded` en configs/data/panel_v1.yaml).
STATIC_LEVELS = {
    "aux_static_Engine": (["ENG_1", "ENG_2", "ENG_3"], [0.2, 0.6, 0.2]),
    "aux_static_ModelSeries": (["MODEL_1", "MODEL_2", "MODEL_3", "MODEL_4"], [0.35, 0.25, 0.1, 0.3]),
    "static_SalesCountry_cd": (["CNTRY_3", "CNTRY_4"], [0.35, 0.65]),
}


def build_dummy_panel(cfg: dict) -> pd.DataFrame:
    """Arma el panel dummy: una fila por par (vehículo, punto de corte)."""
    rng = np.random.default_rng(cfg["seed"])

    n_vehicles = int(cfg["n_vehicles"])
    event_rate = float(cfg["event_rate"])
    window_km = float(cfg["window_km"])
    gap_km = float(cfg["gap_km"])
    horizon_km = float(cfg["horizon_km"])
    step_km = float(cfg["cut_step_km"])
    signal_strength = float(cfg.get("signal_strength", 0.0))
    missing_frac = float(cfg.get("missing_frac", 0.0))
    date_anchor_frac = float(cfg.get("date_anchor_frac", 1.0))

    vehicle_ids = [f"VIN{idx:05d}" for idx in range(n_vehicles)]
    has_event = rng.random(n_vehicles) < event_rate
    # Vida observada por vehículo: lognormal recortada, en km de odómetro.
    life_km = np.clip(rng.lognormal(mean=np.log(28_000), sigma=0.55, size=n_vehicles), 4_000, 180_000)
    start_odo = rng.uniform(0, 1_500, size=n_vehicles)
    km_per_day = np.clip(rng.normal(45, 18, size=n_vehicles), 5, 200)
    production_day = rng.integers(0, 900, size=n_vehicles)
    days_until_sale = rng.integers(5, 400, size=n_vehicles)

    statics = {
        name: rng.choice(levels, size=n_vehicles, p=probs)
        for name, (levels, probs) in STATIC_LEVELS.items()
    }

    records: list[dict] = []
    for i, vehicle_id in enumerate(vehicle_ids):
        last_odo = start_odo[i] + life_km[i]
        event_odo = (
            float(rng.uniform(start_odo[i] + window_km + gap_km + 500, last_odo))
            if has_event[i]
            else np.nan
        )
        # El vehículo deja de observarse en el evento: no hay cortes posteriores.
        horizon_end = event_odo if has_event[i] else last_odo
        cuts = np.arange(start_odo[i] + window_km, horizon_end, step_km)
        if cuts.size == 0:
            continue

        for cut_odo in cuts:
            time_to_event = float(event_odo - cut_odo) if has_event[i] else np.nan
            # Etiqueta del contrato: el evento cae en [corte+G, corte+G+H].
            label = int(
                has_event[i] and (gap_km <= time_to_event <= gap_km + horizon_km)
            )
            records.append(
                {
                    "vehicle_id": vehicle_id,
                    "cut_odo": float(cut_odo),
                    "days_offset": float((cut_odo - start_odo[i]) / km_per_day[i]),
                    "horizon_km": horizon_km,
                    "gap_km": gap_km,
                    "window_km": window_km,
                    "label": label,
                    "time_to_event_km": time_to_event,
                    "event_observed": int(has_event[i]),
                    "static_SalesCountry_cd": statics["static_SalesCountry_cd"][i],
                    "static_daysUntilSale": float(days_until_sale[i]),
                    "aux_static_Engine": statics["aux_static_Engine"][i],
                    "aux_static_ModelSeries": statics["aux_static_ModelSeries"][i],
                    "aux_static_ProductionDay": float(production_day[i]),
                }
            )

    panel = pd.DataFrame.from_records(records)
    if panel.empty:
        raise RuntimeError("El panel dummy quedó vacío: revisá window_km / cut_step_km.")

    panel = _add_features(panel, rng, signal_strength=signal_strength)
    panel = _add_cut_dates(panel, rng, date_anchor_frac=date_anchor_frac)
    if missing_frac > 0:
        panel = _inject_missing(panel, rng, missing_frac)

    panel = panel.sort_values(["vehicle_id", "cut_odo"]).reset_index(drop=True)
    return panel[_contract_column_order(panel)]


def _add_features(panel: pd.DataFrame, rng: np.random.Generator, *, signal_strength: float):
    """Ruido plausible por feature, con recorte al rango físico de cada variable."""
    n = len(panel)
    # Efecto vehículo: los cortes de un mismo vehículo se parecen entre sí.
    codes = pd.factorize(panel["vehicle_id"])[0]
    for name, (mean, std, lo, hi) in FEATURE_SPECS.items():
        vehicle_effect = rng.normal(0, std * 0.6, size=codes.max() + 1)[codes]
        values = mean + vehicle_effect + rng.normal(0, std * 0.8, size=n)
        if lo is not None or hi is not None:
            values = np.clip(values, lo if lo is not None else -np.inf, hi if hi is not None else np.inf)
        panel[name] = values.astype(float)

    if signal_strength > 0:
        # Señal sintética: crece al acercarse el evento y es nula en los sanos.
        proximity = np.where(
            panel["event_observed"].to_numpy() == 1,
            np.exp(-panel["time_to_event_km"].to_numpy() / 4_000.0),
            0.0,
        )
        proximity = np.nan_to_num(proximity)
        for name in SIGNAL_FEATURES:
            _, std, lo, hi = FEATURE_SPECS[name]
            shifted = panel[name].to_numpy() + signal_strength * std * proximity
            panel[name] = np.clip(
                shifted, lo if lo is not None else -np.inf, hi if hi is not None else np.inf
            )
    return panel


def _add_cut_dates(panel: pd.DataFrame, rng: np.random.Generator, *, date_anchor_frac: float):
    """`cut_date` es nullable por contrato: el anclaje temporal puede no existir (plan §0.3)."""
    base = pd.Timestamp("2023-01-01")
    dates = base + pd.to_timedelta(panel.pop("days_offset").to_numpy(), unit="D")
    if date_anchor_frac < 1.0:
        anchored_vehicles = set(
            pd.Series(panel["vehicle_id"].unique()).sample(
                frac=date_anchor_frac, random_state=int(rng.integers(0, 10_000))
            )
        )
        dates = pd.Series(dates).where(panel["vehicle_id"].isin(anchored_vehicles), pd.NaT)
    panel["cut_date"] = pd.Series(dates).astype("datetime64[ns]").to_numpy()
    return panel


def _inject_missing(panel: pd.DataFrame, rng: np.random.Generator, frac: float) -> pd.DataFrame:
    """Nulos en las features: el panel real los va a tener y el pipeline debe aguantarlos."""
    feature_columns = [c for c in panel.columns if c.startswith("feat_")]
    for column in feature_columns:
        mask = rng.random(len(panel)) < frac
        panel.loc[mask, column] = np.nan
    return panel


def _contract_column_order(panel: pd.DataFrame) -> list[str]:
    head = [
        "vehicle_id",
        "cut_odo",
        "cut_date",
        "window_km",
        "horizon_km",
        "gap_km",
        "label",
        "time_to_event_km",
        "event_observed",
    ]
    feats = sorted(c for c in panel.columns if c.startswith("feat_"))
    statics = sorted(c for c in panel.columns if c.startswith("static_"))
    aux = sorted(c for c in panel.columns if c.startswith("aux_"))
    return head + feats + statics + aux


def main() -> None:
    parser = argparse.ArgumentParser(description="Genera el panel dummy del contrato de datos")
    parser.add_argument("--config", default="configs/data/dummy_v1.yaml")
    parser.add_argument("--out", default=None, help="Override del path de salida")
    parser.add_argument("--no-splits", action="store_true", help="No generar splits.json")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    panel = build_dummy_panel(cfg)
    out_path = resolve_path(args.out or cfg["output"]["panel"])
    ensure_dir(out_path.parent)
    panel.to_parquet(out_path, index=False)

    n_event_vehicles = int(panel.groupby("vehicle_id")["event_observed"].max().sum())
    logger.info(
        "Panel dummy: %d filas | %d vehículos (%d con evento) | tasa base %.4f | %s",
        len(panel),
        panel["vehicle_id"].nunique(),
        n_event_vehicles,
        panel["label"].mean(),
        out_path,
    )

    if not args.no_splits:
        splits = make_splits(panel, **split_options(cfg))
        splits_path = save_splits(splits, cfg["output"]["splits"])
        logger.info("Splits: %d folds | %s", splits["n_splits"], splits_path)


if __name__ == "__main__":
    main()
