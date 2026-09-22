#!/usr/bin/env python
"""Smoke test de F0: verifica que el harness está sano antes de confiar en un número.

    python scripts/check_setup.py

Corre en segundos sobre un panel dummy chico y chequea las cuatro cosas que, si
fallan, invalidan todo lo que venga después:

1. El panel dummy cumple el contrato de datos (columnas, dtypes, prefijos).
2. Los splits no filtran vehículos entre train y validación, cubren el panel,
   se niegan a dejar un fold sin positivos y la CV repetida con R=1 es la simple.
3. Un modelo de tasa base saca PR-AUC ≈ tasa base y ROC-AUC ≈ 0,5 (si saca más,
   hay leakage o un bug en la evaluación).
4. Las métricas de anticipación premian a un ranker oráculo y castigan al ruido.
5. La agregación a nivel vehículo (MIL) colapsa bolsas sin romper el out-of-fold.
6. La descomposición cohorte/cuándo: el techo de cohorte es el piso real de un
   PR-AUC por fila, y (a') aísla lo que el modelo sabe del *cuándo*.
7. El panel de hitos post-venta (cure model): nada posterior al hito entra a una
   feature, el gap excluye la fila, el riesgo respeta la ventana del registro y los
   pesos por mes reproducen la ventana.
8. La normalización contra la flota se ajusta solo con los sanos del train del fold,
   y el hook `preprocessing` deja el default intacto.
9. El cure model: Firth contra referencias independientes, el EM sobre datos simulados
   con ventana (monótono, en el máximo de la verosimilitud, recuperando la verdad) y el
   par `cure_window` / `cure_mixture` por el loop de CV.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# La consola de Windows arranca en cp1252 y revienta con los ≈/á de los mensajes.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from scripts.make_dummy import build_dummy_panel  # noqa: E402
from scripts.build_eda_cache import resolved_eda_config  # noqa: E402
from scripts.audit_model import CALENDAR_AUX, promote_aux  # noqa: E402
from scripts.train import vehicle_block  # noqa: E402
from src.config import load_config, set_seed  # noqa: E402
from src.eval.metrics import (  # noqa: E402
    VEHICLE_AGGREGATIONS,
    classification_metrics,
    cohort_ceiling,
    cost_matrix_score,
    cost_ratio_breakeven,
    cost_ratio_sweep,
    lead_time_curve,
    operating_point,
    pr_auc_within_failed,
    vehicle_metrics,
    vehicle_scores,
    when_contribution,
)
from src.data.landmark import (  # noqa: E402
    assign_tercile,
    build_landmark_panel,
    landmark_config,
    load_fleet_features,
    parse_fleet_column,
    production_tercile_edges,
)
from src.eval.splits import iter_folds, iter_repeats, make_splits, test_split_masks  # noqa: E402
from src.features.trips import DEFAULT_THRESHOLDS  # noqa: E402
from src.models.cure import (  # noqa: E402
    UNIT_WEIGHT_SCORE,
    CureMixtureModel,
    fit_cure_em,
    fit_logistic,
    susceptible_weights,
)
from src.training.cv import build_preprocessor, run_cv, select_feature_columns  # noqa: E402
from src.training.transformers import FleetReferenceNormalizer  # noqa: E402
from src.training.targets import build_ordinal_target, build_target  # noqa: E402

CONTRACT_COLUMNS = {
    "vehicle_id": "object",
    "cut_odo": "float",
    "cut_date": "datetime",
    "horizon_km": "float",
    "gap_km": "float",
    "label": "int",
    "time_to_event_km": "float",
    "event_observed": "int",
}

SMALL_PANEL = {
    "seed": 7,
    "n_vehicles": 120,
    "event_rate": 0.25,
    "window_km": 1000,
    "gap_km": 500,
    "horizon_km": 3000,
    "cut_step_km": 750,
    "missing_frac": 0.05,
    "date_anchor_frac": 0.8,
    "signal_strength": 0.0,
}

_checks: list[tuple[str, bool, str]] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    _checks.append((name, bool(condition), detail))
    print(f"{'PASS' if condition else 'FAIL'} | {name}" + (f" — {detail}" if detail else ""))


def _raises(call, exception: type[Exception]) -> bool:
    try:
        call()
    except exception:
        return True
    except Exception:
        return False
    return False


def _synthetic_landmark_inputs(seed: int = 7) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Seis vehículos vendidos el 01-06-2025, uno por caso que el panel de hitos tiene que resolver.

    A sano (telemetría hasta el 01-01-2026, antes del fin de la ventana) · B fallado a 150 d
    post-venta, con un julio casi sin viajes · C fallado justo en L + G del primer hito (60 d) · D quieto (solo idle) ·
    E con menos viajes que el mínimo · F sin fecha de venta. Dos viajes por día: uno de
    30 km a la mañana y un idle a la tarde; A tiene además un viaje que cruza el hito de 30 d.
    """
    rng = np.random.default_rng(seed)
    sale = pd.Timestamp("2025-06-01", tz="UTC")
    rows = []
    for vid in "ABCDEF":
        days = pd.date_range("2025-05-01", "2026-01-01" if vid == "A" else "2026-05-01", freq="D", tz="UTC")
        if vid == "E":
            # Llega a todos los hitos (viaja de nuevo a los 200 d) pero con 10 viajes post-venta.
            days = (days[days <= sale].append(pd.date_range(sale + pd.Timedelta(days=1), periods=5, freq="D"))
                    .append(pd.date_range(sale + pd.Timedelta(days=200), periods=5, freq="D")))
        if vid == "B":
            # Julio con 2 días de viaje: 4 viajes y 60 km, bajo el mínimo de las cuatro features.
            july = (days >= pd.Timestamp("2025-07-01", tz="UTC")) & (days < pd.Timestamp("2025-08-01", tz="UTC"))
            days = days[~july | (days.day <= 2)]
        for day in days:
            rows.append((vid, day + pd.Timedelta(hours=8), 40.0, 0.0 if vid == "D" else 30.0))
            rows.append((vid, day + pd.Timedelta(hours=18), 5.0, 0.0))
    rows.append(("A", sale + pd.Timedelta(days=30) - pd.Timedelta(hours=1), 120.0, 500.0))
    trips = pd.DataFrame(rows, columns=["vehicle_id", "TripDatetimeStart", "minutes", "trip_km"])
    trips = trips.sort_values(["vehicle_id", "TripDatetimeStart"], ignore_index=True)
    trips["TripNumber"] = trips.groupby("vehicle_id").cumcount().astype(float)
    trips["TripDatetimeEnd"] = trips["TripDatetimeStart"] + pd.to_timedelta(trips["minutes"], unit="min")
    trips["OdometerTripEnd"] = trips.groupby("vehicle_id")["trip_km"].cumsum()
    trips["idle"] = trips["trip_km"].eq(0)
    moving = ~trips["idle"]
    trips["below_regime"] = rng.random(len(trips)) < 0.3
    trips["speed_kmh_moving"] = (trips["trip_km"] / trips["minutes"] * 60).where(moving)
    trips["CoolantTemperatureEnd_moving"] = pd.Series(rng.normal(85, 5, len(trips))).where(moving)
    trips["AirTemperatureAvg"] = rng.normal(15, 5, len(trips))

    vehicles = pd.DataFrame(index=pd.Index(list("ABCDEF"), name="vehicle_id"))
    vehicles["event_observed"] = [0, 1, 1, 0, 0, 0]
    vehicles["sale_date"] = pd.to_datetime([sale] * 5 + [pd.NaT], utc=True)
    vehicles["event_dss"] = [np.nan, 150.0, 60.0, np.nan, np.nan, np.nan]
    vehicles["event_odo_km"] = [np.nan, 8000.0, 3000.0, np.nan, np.nan, np.nan]
    vehicles["last_trip"] = trips.groupby("vehicle_id")["TripDatetimeStart"].max()
    vehicles["static_SalesCountry_cd"] = ["X", "Y", "X", "Y", "X", "Y"]
    vehicles["aux_static_ProductionDay"] = [100.0, 200.0, 300.0, 400.0, 500.0, 600.0]
    return vehicles, trips


def _fleet_columns(panel: pd.DataFrame, feature: str, *, weight: bool) -> list[str]:
    out = []
    for column in panel.columns:
        parsed = parse_fleet_column(column)
        if parsed is not None and parsed[0] == feature and parsed[2] == weight:
            out.append(column)
    return out


def landmark_panel_checks() -> None:
    """7 · El panel de hitos post-venta, sobre datos sintéticos y con los YAML reales."""
    cfg = load_config("configs/data/panel_landmark_ps.yaml")
    lm_cfg = landmark_config(cfg, load_config(cfg["event_clock"])["event_window"])
    features = load_fleet_features(cfg["fleet_features"], cfg["features_spec"])
    vehicles, trips = _synthetic_landmark_inputs()
    panel, report = build_landmark_panel(vehicles, trips, lm_cfg, features)
    first = min(lm_cfg.landmarks_days)
    at_first = report["landmarks"][f"{first:g}"]

    check(
        "hitos: una fila por (vehículo, hito)",
        not panel.duplicated(["vehicle_id", "aux_landmark_day"]).any()
        and set(panel["aux_landmark_day"]) == set(lm_cfg.landmarks_days),
    )
    check(
        "hitos: un evento en ≤ L + G excluye la fila (regla 1), y se cuenta",
        "C" not in set(panel["vehicle_id"]) and at_first["evento_antes_o_en_gap"] == 1
        and bool(panel.loc[panel["vehicle_id"].eq("B"), "label"].eq(1).all()),
        f"descartes en el hito {first:g}: {at_first}",
    )
    check(
        "hitos: sin fecha de venta o con pocos viajes no hay fila, y se cuentan",
        not {"E", "F"} & set(panel["vehicle_id"]) and at_first["sin_venta"] == 1 and at_first["pocos_viajes"] == 1,
    )

    # Nada posterior al hito: se perturban todos los viajes que terminan después del primer
    # hito y las filas de ese hito no pueden moverse (el viaje de A que cruza el hito incluido).
    land = vehicles["sale_date"] + pd.Timedelta(days=first)
    later = trips["TripDatetimeEnd"] > trips["vehicle_id"].map(land)
    perturbed = trips.copy()
    perturbed.loc[later, "trip_km"] *= 7
    perturbed.loc[later, "OdometerTripEnd"] += 1e5
    perturbed.loc[later, "idle"] = ~perturbed.loc[later, "idle"]
    perturbed.loc[later, "below_regime"] = ~perturbed.loc[later, "below_regime"]
    for column in ("speed_kmh_moving", "CoolantTemperatureEnd_moving", "AirTemperatureAvg"):
        perturbed.loc[later, column] = -perturbed.loc[later, column]
    moved, _ = build_landmark_panel(vehicles, perturbed, lm_cfg, features)
    watched = [c for c in panel.columns if c.startswith(("feat_", "aux_raw_"))]
    watched += ["cut_odo", "window_km", "aux_n_trips_window", "aux_air_temp_window_mean"]
    a = panel.loc[panel["aux_landmark_day"].eq(first)].set_index("vehicle_id")
    b = moved.loc[moved["aux_landmark_day"].eq(first)].set_index("vehicle_id").reindex(a.index)
    common = [c for c in watched if c in b.columns]
    extra = [c for c in b.columns if c.startswith(("feat_", "aux_raw_")) and c not in panel.columns]
    changed = [c for c in common if not a[c].equals(b[c])]
    check(
        "hitos: ningún viaje posterior al hito entra a una feature (perturbarlos no mueve la fila)",
        not changed and bool(b[extra].isna().all().all()) and len(common) == len(watched),
        f"cambian: {changed[:5]}" if changed else "",
    )

    km_ok = np.allclose(panel[_fleet_columns(panel, "idle_per_1000km", weight=True)].sum(axis=1), panel["window_km"])
    trips_ok = np.allclose(panel[_fleet_columns(panel, "trips_below_regime_temp_frac", weight=True)].sum(axis=1),
                           panel["aux_n_trips_window"])
    july = panel.loc[panel["vehicle_id"].eq("B") & panel["aux_landmark_day"].gt(first)]
    thin_ok = all(july[f.value_column("2025-07")].isna().all() and july[f.weight_column("2025-07")].gt(0).all()
                  for f in features)
    check(
        "hitos: los pesos por mes suman los km y los viajes de la ventana; bajo el mínimo, valor NaN y peso intacto",
        km_ok and trips_ok and len(july) > 0 and thin_ok,
    )

    entry_ok = np.allclose(panel["aux_dss_entry"],
                           np.maximum(panel["aux_landmark_day"] + lm_cfg.gap_days, panel["aux_dss_window_start"]))
    exit_ok = bool((panel["aux_dss_exit"] <= panel["aux_dss_window_end"]).all())
    b_rows = panel.loc[panel["vehicle_id"].eq("B")]
    check(
        "hitos: entrada = max(L + G, inicio de la ventana) y salida ≤ fin de la ventana",
        entry_ok and exit_ok and bool(b_rows["aux_event_in_window"].eq(1).all())
        and bool(b_rows["aux_dss_exit"].eq(150).all()),
    )
    a_rows = panel.loc[panel["vehicle_id"].eq("A")]
    check(
        "hitos: la salida de un sano es el fin de la ventana, no su último viaje",
        len(a_rows) > 0 and bool(a_rows["aux_dss_exit"].eq(a_rows["aux_dss_window_end"]).all())
        and bool(vehicles.loc["A", "last_trip"] < lm_cfg.window_end),
    )

    d_rows = panel.loc[panel["vehicle_id"].eq("D")]
    check(
        "hitos: un auto quieto queda en NaN en idle por km y conserva los viajes bajo régimen",
        len(d_rows) > 0
        and bool(d_rows[_fleet_columns(panel, "idle_per_1000km", weight=False)].isna().all().all())
        and bool(d_rows[_fleet_columns(panel, "trips_below_regime_temp_frac", weight=False)].notna().any(axis=1).all()),
    )

    split = {"dev_vehicles": ["A", "B", "C", "E", "F"], "test_vehicles": ["D"], "excluded_vehicles": ["Z"]}
    dev_mask, test_mask = test_split_masks(panel, split, strict=True)
    intruder = pd.concat([panel.iloc[:1].assign(vehicle_id="Z"), panel], ignore_index=True)
    check(
        "hitos: el recorte a dev no deja pasar vehículos de test, y un excluido hace fallar",
        not set(panel.loc[dev_mask, "vehicle_id"]) & set(split["test_vehicles"])
        and bool((dev_mask | test_mask).all())
        and _raises(lambda: test_split_masks(intruder, split, strict=True), ValueError),
    )

    production = pd.Series(np.random.default_rng(3).integers(0, 400, 90).astype(float))
    ours = assign_tercile(production, production_tercile_edges(production, 3))
    check(
        "hitos: los terciles de producción siguen la convención de pd.qcut",
        bool((ours.to_numpy() == pd.qcut(production, 3, labels=False).to_numpy()).all()),
    )


def _fleet_frame() -> tuple[pd.DataFrame, np.ndarray]:
    """Panel mínimo con una feature, dos mercados y tres meses, uno por nivel de la jerarquía.

    2025-09: 11 vehículos sanos por mercado, así que la celda mercado × mes alcanza.
    2025-10: 6 por mercado —ninguna celda llega a 10, pero el mes con los dos mercados
    juntos sí—. 2025-11: un solo vehículo, que tiene que caer a la mediana global.
    """
    rows = []
    for market, base in (("X", 10.0), ("Y", 30.0)):
        for i in range(12):
            rows.append({"vehicle": f"{market}{i}", "market": market, "healthy": i > 0,
                         "v09": base + i, "w09": 100.0})
        for i in range(6):
            rows.append({"vehicle": f"{market}oct{i}", "market": market, "healthy": True,
                         "v10": base + i, "w10": 50.0})
    rows.append({"vehicle": "solo", "market": "X", "healthy": True, "v11": 7.0, "w11": 20.0})
    # Un fallado con dos meses de peso muy distinto: es el que distingue el promedio
    # ponderado del promedio a secas. No entra a la referencia (no es sano).
    rows.append({"vehicle": "mix", "market": "X", "healthy": False,
                 "v09": 5.0, "w09": 300.0, "v10": 10.0, "w10": 50.0})
    frame = pd.DataFrame(rows).reindex(columns=["vehicle", "market", "healthy", "v09", "w09",
                                                "v10", "w10", "v11", "w11"])
    X = pd.DataFrame({
        "feat_fm__a__2025-09": frame["v09"], "feat_fm__w_a__2025-09": frame["w09"],
        "feat_fm__a__2025-10": frame["v10"], "feat_fm__w_a__2025-10": frame["w10"],
        "feat_fm__a__2025-11": frame["v11"], "feat_fm__w_a__2025-11": frame["w11"],
        "static_SalesCountry_cd": frame["market"], "feat_landmark_day": 30.0,
    })
    y = np.rec.fromarrays([frame["healthy"].to_numpy(), frame["vehicle"].to_numpy()],
                          names="healthy,group")
    return X, y


def fleet_normalizer_checks() -> None:
    """8 · La normalización contra la flota se ajusta con el train del fold y no filtra."""
    X, y = _fleet_frame()
    normalizer = FleetReferenceNormalizer(min_ref_vehicles=10)
    out = normalizer.fit_transform(X, y)

    # Referencia de X en 2025-09: mediana de los 11 sanos (i = 1..11) = 10 + 6 = 16.
    expected = X.loc[0, "feat_fm__a__2025-09"] - 16.0
    check(
        "flota: el desvío es (valor − mediana de los sanos comparables) de ese mercado y mes",
        np.isclose(out.loc[0, "fleet_a"], expected),
        f"{out.loc[0, 'fleet_a']:.2f} vs. {expected:.2f}",
    )
    check(
        "flota: la salida es fleet_* más lo que no es feat_fm__, y el mercado no pasa",
        list(out.columns) == ["fleet_a", "feat_landmark_day"]
        and list(normalizer.get_feature_names_out()) == list(out.columns),
        f"{list(out.columns)}",
    )
    # 2025-10 no llega a 10 vehículos en ningún mercado (cae al mes con los dos juntos) y
    # 2025-11 tiene uno solo (cae a la mediana global de la feature).
    october = X.loc[y["healthy"], "feat_fm__a__2025-10"].dropna()
    pooled = float(october.median())
    row_10 = int(october.index[0])
    row_11 = int(X["feat_fm__a__2025-11"].notna().to_numpy().nonzero()[0][-1])
    check(
        "flota: una celda sin vehículos suficientes cae al nivel de arriba, y una sola al global",
        normalizer.levels_used_ == {"market×month": 2, "month": 2, "global": 1, "sin_referencia": 0}
        and np.isclose(out.loc[row_10, "fleet_a"], X.loc[row_10, "feat_fm__a__2025-10"] - pooled),
        f"{normalizer.levels_used_}",
    )
    mix = len(X) - 1
    weighted = ((5.0 - 16.0) * 300.0 + (10.0 - pooled) * 50.0) / 350.0
    check(
        "flota: los meses se promedian ponderados por su peso, no a secas",
        np.isclose(out.loc[mix, "fleet_a"], weighted)
        and not np.isclose(weighted, ((5.0 - 16.0) + (10.0 - pooled)) / 2),
        f"{out.loc[mix, 'fleet_a']:.3f} vs. {weighted:.3f}",
    )

    value_columns = [c for c in X.columns if c.startswith("feat_fm__a__")]
    cells = pd.concat([X.loc[y["healthy"], c] for c in value_columns]).dropna()
    check(
        "flota: el último recurso es la mediana global de la feature entre los sanos",
        np.isclose(out.loc[row_11, "fleet_a"], 7.0 - float(cells.median())),
        f"{out.loc[row_11, 'fleet_a']:.2f} vs. {7.0 - float(cells.median()):.2f}",
    )
    check(
        "flota: sin `y` no se puede fitear (la referencia son los sanos del train)",
        _raises(lambda: FleetReferenceNormalizer().fit(X), ValueError),
    )

    # Un vehículo con tres hitos no puede pesar el triple en la mediana.
    repeated = pd.concat([X, X.iloc[[1]], X.iloc[[1]]], ignore_index=True)
    y_repeated = np.rec.fromarrays(
        [np.append(y["healthy"], [True, True]), np.append(y["group"], ["X1", "X1"])], names="healthy,group")
    out_repeated = FleetReferenceNormalizer(min_ref_vehicles=10).fit_transform(repeated, y_repeated)
    check(
        "flota: la mediana es entre vehículos, no entre filas (los hitos se deduplican)",
        np.isclose(out_repeated.loc[0, "fleet_a"], out.loc[0, "fleet_a"]),
        f"{out_repeated.loc[0, 'fleet_a']:.2f} vs. {out.loc[0, 'fleet_a']:.2f}",
    )

    # Fuga: ajustar con un subconjunto y perturbar las filas que NO se usaron.
    train = np.zeros(len(X), dtype=bool)
    train[:12] = True
    fitted = FleetReferenceNormalizer(min_ref_vehicles=10).fit(X.loc[train], (y["healthy"][train], y["group"][train]))
    perturbed = X.copy()
    fm = [c for c in X.columns if c.startswith("feat_fm__")]
    perturbed.loc[~train, fm] = perturbed.loc[~train, fm] * 100 + 1
    after = FleetReferenceNormalizer(min_ref_vehicles=10).fit(perturbed.loc[train], (y["healthy"][train], y["group"][train]))
    same_reference = all(a.equals(b) for a, b in zip(fitted.reference_, after.reference_))
    same_rows = fitted.transform(X.loc[train]).equals(fitted.transform(perturbed.loc[train]))
    check(
        "flota: perturbar las filas que no se usaron no mueve ni la referencia ni el train",
        same_reference and same_rows,
    )

    # El hook `preprocessing`: `none` no arma preprocesador y el default no cambia nada.
    panel = build_dummy_panel({**SMALL_PANEL, "seed": 11})
    splits = make_splits(panel, n_splits=3, seed=11, min_valid_positives=1)
    default, _ = run_cv(panel, splits, model_name="baserate")
    explicit, _ = run_cv(panel, splits, model_name="baserate", preprocessing="standard")
    check(
        "preprocessing: sin la clave es `standard` y da lo mismo, columna por columna",
        default.equals(explicit),
    )
    check(
        "preprocessing: un modo inventado falla en vez de entrenar otra cosa",
        _raises(lambda: run_cv(panel, splits, model_name="baserate", preprocessing="ninguno"), ValueError),
    )
    # Con `none` el pipeline es solo el modelo: un estimador que no tolera NaN ni strings
    # falla, que es la prueba de que nadie imputó ni codificó por atrás.
    check(
        "preprocessing: con `none` no hay preprocesador (el estimador recibe el panel crudo)",
        _raises(lambda: run_cv(panel, splits, model_name="lgbm", preprocessing="none"), ValueError),
    )


# El test de correctitud del cure model (prompt §8.2): 300 vehículos, 70% curados, dos
# covariables, latencia Weibull con k = 2 y λ = 200 d desde la venta, y una ventana de
# calendario de 300 d que trunca la exposición: entrada tardía para los vendidos antes de
# que abra y salida al cierre; a los vendidos muy tarde la ventana se les cierra antes de
# L + G y quedan fuera de riesgo. Con una ventana de 190 d, como la real, la MV sin
# penalizar degenera en 2 de 200 réplicas (π → 1 con una latencia larguísima: no hay
# seguimiento suficiente) y Firth en ninguna. Con 300 d no degenera ninguna.
CURE_SIM = {"beta": (-1.1, 0.8, -0.6), "shape": 2.0, "scale": 200.0, "landmark": 30.0,
            "gap": 30.0, "window": 300.0, "n": 300, "seed": 11}
# Tolerancia de la recuperación: 3 desvíos de cada estimador entre 200 réplicas del mismo
# diseño (piloto del 22-09): 0,23 / 0,26 / 0,22 en β, 0,42 en k y 24 d en λ.
CURE_SIM_TOLERANCE = (0.7, 0.8, 0.7, 1.3, 72.0)


def _simulate_cure(*, beta, shape, scale, landmark, gap, window, n, seed):
    """Datos del modelo del preregistro: Z ~ Bern(π(x)) y, si Z = 1, T | T > e ~ Weibull."""
    rng = np.random.default_rng(seed)
    design = np.column_stack([np.ones(n), rng.normal(size=(n, 2))])
    susceptible = rng.random(n) < expit(design @ np.asarray(beta))
    sale = rng.uniform(-200.0, 280.0, n)  # día de la venta respecto de la apertura de la ventana
    entry = np.maximum(landmark + gap, -sale)
    end = window - sale
    # Inversa de la condicional: S(T)/S(e) = U.
    t = scale * ((entry / scale) ** shape - np.log(rng.random(n))) ** (1.0 / shape)
    event = susceptible & (t <= end) & (end > entry)
    return design, entry, np.where(event, t, end), event.astype(int)


def _cure_negloglik(theta, design, entry, exit_, event, firth=False):
    """−ℓ del cure model escrita de nuevo, sin reusar nada de src/models/cure.py."""
    beta, k, lam = theta[:-2], np.exp(theta[-2]), np.exp(theta[-1])
    pi = expit(design @ beta)
    log_ratio = (entry / lam) ** k - (exit_ / lam) ** k
    log_h = np.log(k / lam) + (k - 1) * np.log(exit_ / lam)
    loglik = np.where(event == 1, np.log(pi) + log_h + log_ratio,
                      np.log(1 - pi + pi * np.exp(log_ratio))).sum()
    if firth:
        v = pi * (1 - pi)
        loglik += 0.5 * np.linalg.slogdet((design * v[:, None]).T @ design)[1]
    return -loglik


def _direct_max(design, entry, exit_, event, *, firth, start):
    """El máximo de la misma verosimilitud por BFGS, sin EM: `(β, k, λ)` y el objetivo."""
    theta = np.r_[start[:-2], np.log(start[-2:])]
    result = minimize(_cure_negloglik, theta, args=(design, entry, exit_, event, firth), method="BFGS",
                      options={"gtol": 1e-9, "maxiter": 5000})
    return np.r_[result.x[:-2], np.exp(result.x[-2:])], -result.fun


def _firth_by_augmentation(X, y, *, n_iter=500):
    """Firth por aumento de datos (Heinze & Schemper 2002), independiente del Newton del modelo.

    Cada fila se parte en (y, peso 1 + h/2) y (1 − y, peso h/2), se ajusta la logística por
    MV ponderada y se repite con el h nuevo hasta el punto fijo.
    """
    beta = np.zeros(X.shape[1])
    Xa, ya = np.vstack([X, X]), np.r_[y, 1 - y]
    for _ in range(n_iter):
        p = expit(X @ beta)
        v = p * (1 - p)
        h = v * np.einsum("ij,jk,ik->i", X, np.linalg.inv((X * v[:, None]).T @ X), X)
        wa = np.r_[1 + h / 2, h / 2]
        result = minimize(lambda b: -(wa * (ya * (Xa @ b) - np.logaddexp(0, Xa @ b))).sum(), beta,
                          jac=lambda b: -(Xa.T @ (wa * (ya - expit(Xa @ b)))), method="BFGS",
                          options={"gtol": 1e-12})
        if np.max(np.abs(result.x - beta)) < 1e-11:
            return result.x
        beta = result.x
    return beta


def _cure_panel(seed: int = 5, n: int = 160) -> pd.DataFrame:
    """Panel de hitos sintético (30 y 60 d) con dos features de flota y el desenlace del cure.

    Cada feature es un corrimiento por mercado × mes más el rasgo latente del vehículo: el
    normalizador tiene que sacar el corrimiento y dejar el rasgo. El riesgo sube con `a` y
    baja con `b`. Los eventos caen después de 90 d (ninguno dentro del gap) y solo se
    registran dentro de una ventana de 300 d.
    """
    rng = np.random.default_rng(seed)
    markets = np.where(np.arange(n) % 2 == 0, "X", "Y")
    latent = rng.normal(size=(n, 2))
    susceptible = rng.random(n) < expit(-0.9 + 1.0 * latent[:, 0] - 0.8 * latent[:, 1])
    sale = rng.uniform(-150.0, 250.0, n)
    window = 300.0
    t = 200.0 * ((90.0 / 200.0) ** 2 - np.log(rng.random(n))) ** 0.5  # Weibull(2, 200) | T > 90
    registered = susceptible & (t >= -sale) & (t <= window - sale)
    shift = {("X", "2025-09"): 10.0, ("X", "2025-10"): 12.0, ("Y", "2025-09"): 30.0, ("Y", "2025-10"): 33.0}
    rows = []
    for landmark in (30.0, 60.0):
        entry = np.maximum(landmark + 30.0, -sale)
        end = window - sale
        in_window = registered & (t > entry)
        for i in range(n):
            row = {
                "vehicle_id": f"V{i:03d}", "cut_odo": landmark * 40.0, "cut_date": pd.NaT,
                "horizon_km": np.nan, "gap_km": np.nan,
                "label": int(registered[i] and t[i] <= landmark + 30.0 + 240.0),
                "time_to_event_km": np.nan, "event_observed": int(registered[i]),
                "feat_landmark_day": landmark, "static_SalesCountry_cd": markets[i],
                "aux_landmark_day": landmark, "aux_gap_days": 30.0, "aux_horizon_days": 240.0,
                "aux_dss_entry": entry[i], "aux_dss_exit": t[i] if in_window[i] else end[i],
                "aux_event_in_window": int(in_window[i]),
            }
            for month in ("2025-09", "2025-10"):
                base = shift[(markets[i], month)]
                row[f"feat_fm__a__{month}"] = base + 2.0 * latent[i, 0] + rng.normal(scale=0.5)
                row[f"feat_fm__w_a__{month}"] = 100.0
                row[f"feat_fm__b__{month}"] = base + 1.5 * latent[i, 1] + rng.normal(scale=0.5)
                row[f"feat_fm__w_b__{month}"] = 20.0
            rows.append(row)
    return pd.DataFrame(rows)


def firth_checks() -> None:
    """9a · Firth y FLIC contra referencias que no comparten código con el modelo."""
    # Tabla 2×2 con una celda vacía: la MV diverge, y Firth en un modelo saturado es exactamente
    # el log-odds con ½ sumado a cada celda (Firth 1993). Filas colapsadas, con la cuenta como peso.
    X = np.array([[1, 0], [1, 0], [1, 1], [1, 1]], dtype=float)
    y = np.array([0, 1, 0, 1], dtype=float)
    fit = fit_logistic(X, y, weights=np.array([12, 3, 9, 0], dtype=float), firth=True)
    p0, p1 = 3.5 / 16, 0.5 / 10
    expected = np.array([np.log(p0 / (1 - p0)), np.log(p1 / (1 - p1)) - np.log(p0 / (1 - p0))])
    check(
        "firth: en una tabla 2×2 con una celda vacía es sumar ½ a cada celda (Firth 1993)",
        fit.converged and np.allclose(fit.coef, expected, atol=1e-8),
        f"{np.round(fit.coef, 5)} vs. {np.round(expected, 5)}",
    )

    rng = np.random.default_rng(3)
    X = np.column_stack([np.ones(120), rng.normal(size=(120, 3))])
    y = (rng.random(120) < expit(X @ np.array([-1.0, 1.5, -1.0, 0.5]))).astype(float)
    ours = fit_logistic(X, y, firth=True).coef
    reference = _firth_by_augmentation(X, y)
    check(
        "firth: el Newton coincide con Firth por aumento de datos (Heinze & Schemper 2002)",
        np.allclose(ours, reference, atol=1e-6),
        f"max |Δ| = {np.max(np.abs(ours - reference)):.1e}",
    )
    slopes = X[:, 1:] @ ours[1:]
    intercept = fit_logistic(X[:, :1], y, offset=slopes).coef[0]
    check(
        "FLIC: con las pendientes de Firth fijas, el intercepto de MV iguala la tasa observada",
        np.isclose(expit(intercept + slopes).mean(), y.mean(), atol=1e-9)
        and not np.isclose(expit(X @ ours).mean(), y.mean(), atol=1e-4),
        f"π̄ {expit(X @ ours).mean():.4f} → {expit(intercept + slopes).mean():.4f} (tasa {y.mean():.4f})",
    )


def cure_em_checks() -> None:
    """9b · El EM sobre datos simulados con ventana: monótono, en el máximo y recuperando la verdad."""
    design, entry, exit_, event = _simulate_cure(**CURE_SIM)
    risk = exit_ > entry
    d, e, x, ev = design[risk], entry[risk], exit_[risk], event[risk]
    truth = np.r_[CURE_SIM["beta"], CURE_SIM["shape"], CURE_SIM["scale"]]

    fit = fit_cure_em(d, e, x, ev)
    estimate = np.r_[fit.coef, fit.latency.shape, fit.latency.scale]
    check(
        "cure EM: la verosimilitud no baja en ninguna iteración y converge",
        fit.converged and np.diff(fit.history).min() >= 0,
        f"{fit.n_iter} iteraciones, {int(risk.sum())} en riesgo, {int(ev.sum())} eventos",
    )
    direct, direct_value = _direct_max(d, e, x, ev, firth=False, start=truth)
    check(
        "cure EM: llega al máximo de la verosimilitud (BFGS sobre la misma ℓ, escrita aparte)",
        np.allclose(estimate, direct, rtol=1e-3) and fit.loglik >= direct_value - 1e-5,
        f"ℓ {fit.loglik:.6f} vs. {direct_value:.6f}",
    )
    check(
        "cure EM: recupera β, k y λ de la simulación (3 desvíos del piloto)",
        bool(np.all(np.abs(estimate - truth) <= np.asarray(CURE_SIM_TOLERANCE))),
        f"{np.round(estimate, 2)} vs. {truth}",
    )

    penalized = fit_cure_em(d, e, x, ev, firth=True)
    direct_firth, direct_firth_value = _direct_max(d, e, x, ev, firth=True, start=truth)
    check(
        "cure EM + Firth: el objetivo penalizado no baja y llega a su máximo",
        penalized.converged and np.diff(penalized.history).min() >= 0
        and np.allclose(np.r_[penalized.coef, penalized.latency.shape, penalized.latency.scale],
                        direct_firth, rtol=1e-3)
        and penalized.history[-1] >= direct_firth_value - 1e-5,
        f"{penalized.history[-1]:.6f} vs. {direct_firth_value:.6f}",
    )
    flic = fit_cure_em(d, e, x, ev, firth=True, flic=True)
    eta = d @ flic.coef
    weights = susceptible_weights(eta, flic.latency, e, x, ev)
    check(
        "cure EM + FLIC: las pendientes son las de Firth y el intercepto iguala Σw con Σπ",
        np.allclose(flic.coef[1:], penalized.coef[1:]) and not np.isclose(flic.coef[0], penalized.coef[0])
        and np.isclose(weights.sum(), expit(eta).sum(), rtol=1e-4)
        and np.diff(flic.flic_history).min() >= 0 and flic.loglik >= penalized.loglik,
        f"intercepto {penalized.coef[0]:.4f} → {flic.coef[0]:.4f}",
    )
    check(
        "cure EM: una fila sin exposición (salida ≤ entrada) no entra a la verosimilitud",
        (~risk).any() and _raises(lambda: fit_cure_em(design, entry, exit_, event), ValueError),
        f"{int((~risk).sum())} filas fuera de riesgo en la simulación",
    )


def cure_model_checks() -> None:
    """9c · Target `cure_window`, modelo `cure_mixture` y su paso por el loop de CV."""
    panel = _cure_panel()
    spec = build_target("cure_window", panel, np.ones(len(panel), dtype=bool))
    y = spec.y
    check(
        "cure_window: `y` sale del panel, en riesgo = salida > entrada, score en (L+G, L+G+H]",
        np.array_equal(y["at_risk"], panel["aux_dss_exit"].to_numpy() > panel["aux_dss_entry"].to_numpy())
        and np.array_equal(y["healthy"], panel["event_observed"].to_numpy() == 0)
        and np.allclose(y["score_start"], panel["aux_landmark_day"] + 30.0)
        and np.allclose(y["score_end"], panel["aux_landmark_day"] + 270.0)
        and (~y["at_risk"]).any() and y["event"].sum() > 0,
        f"{spec.info['n_at_risk']} de {spec.info['n_rows']} filas en riesgo, {spec.info['n_events']} eventos",
    )
    broken = panel.copy()
    first_event = int(np.flatnonzero(panel["aux_event_in_window"].to_numpy() == 1)[0])
    broken.loc[first_event, "aux_dss_exit"] = broken.loc[first_event, "aux_dss_entry"]
    check(
        "cure_window: un evento sin exposición en la ventana es un panel roto y falla",
        _raises(lambda: build_target("cure_window", broken, np.ones(len(broken), dtype=bool)), ValueError),
    )

    X = panel[select_feature_columns(panel)]
    firth = CureMixtureModel(incidence="firth").fit(X, y)
    parts = firth.predict_components(X)
    landmarks = X["feat_landmark_day"].to_numpy()
    constant_horizon = all(np.unique(parts["p_horizon"][landmarks == value]).size == 1
                           for value in np.unique(landmarks))
    check(
        "cure_mixture: score = π·(1 − S(L+G+H)/S(L+G)), con el segundo factor fijo en cada hito",
        np.allclose(parts["score"], parts["pi_incidence"] * parts["p_horizon"]) and constant_horizon
        and np.allclose(firth.predict_proba(X)[:, 1], parts["score"]),
    )
    coef = firth.summary_
    check(
        "cure_mixture: Firth recupera el signo de las dos covariables en los dos hitos",
        all(c["coef"]["fleet_a"] > 0 and c["coef"]["fleet_b"] < 0 for c in coef.values()),
        "; ".join(f"L={k}: a {c['coef']['fleet_a']:+.2f}, b {c['coef']['fleet_b']:+.2f}" for k, c in coef.items()),
    )
    unit = CureMixtureModel(incidence="unit_weight", unit_weight_signs={"fleet_a": 1, "fleet_b": -1}).fit(X, y)
    unit_parts = unit.predict_components(X)
    standardized = [unit._standardized(value, unit.normalizer_.transform(X).loc[landmarks == value, unit.covariates_])
                    for value in np.unique(landmarks)]
    by_hand = np.concatenate([z[:, 0] - z[:, 1] for z in standardized])
    order = np.concatenate([np.flatnonzero(landmarks == value) for value in np.unique(landmarks)])
    monotone = all(
        np.array_equal(np.argsort(unit_parts[UNIT_WEIGHT_SCORE][landmarks == value], kind="stable"),
                       np.argsort(unit_parts["score"][landmarks == value], kind="stable"))
        for value in np.unique(landmarks))
    check(
        "cure_mixture P0: `unit_weight_score` = Σ signo·z por hito, y el cure es monótono en él",
        np.allclose(unit_parts[UNIT_WEIGHT_SCORE][order], by_hand) and monotone
        and all(c["coef"][UNIT_WEIGHT_SCORE] > 0 for c in unit.summary_.values()),
    )
    check(
        "cure_mixture P0: un signo que falta o que no es ±1 falla",
        _raises(lambda: CureMixtureModel(incidence="unit_weight", unit_weight_signs={"fleet_a": 1}).fit(X, y),
                ValueError)
        and _raises(lambda: CureMixtureModel(incidence="unit_weight",
                                             unit_weight_signs={"fleet_a": 1, "fleet_b": -0.5}).fit(X, y),
                    ValueError),
    )

    outside = ~y["at_risk"]
    moved = y.copy()
    moved["exit"][outside] = moved["entry"][outside] - 50.0
    check(
        "cure_mixture: el desenlace de una fila fuera de riesgo no mueve el ajuste",
        outside.any() and np.array_equal(CureMixtureModel(incidence="firth").fit(X, moved).predict_proba(X),
                                         firth.predict_proba(X)),
    )
    check(
        "cure_mixture: con `label` como `y` falla (va de a pares con `target: cure_window`)",
        _raises(lambda: CureMixtureModel().fit(X, panel["label"].to_numpy()), TypeError),
    )
    check(
        "cure_mixture: `incidence: tabpfn` (P2) no corre hasta que P1 le gane a P0",
        _raises(lambda: CureMixtureModel(incidence="tabpfn").fit(X, y), NotImplementedError),
    )
    first = landmarks == 30.0
    pooled = CureMixtureModel(incidence="firth", per_landmark=False).fit(X.loc[first], y[first])
    split = CureMixtureModel(incidence="firth").fit(X.loc[first], y[first])
    check(
        "cure_mixture: con un solo hito, `per_landmark: false` es el mismo ajuste",
        np.allclose(pooled.predict_proba(X.loc[first]), split.predict_proba(X.loc[first])),
    )

    splits = make_splits(panel, n_splits=3, seed=5, min_valid_positives=1)
    cv_args = dict(model_name="cure_mixture", model_params={"incidence": "firth"}, target={"name": "cure_window"})
    predictions, _ = run_cv(panel, splits, preprocessing="none", **cv_args)
    check(
        "cure por run_cv: `preprocessing: none` + `target: cure_window` trae score, π y p_horizon",
        {"pi_incidence", "p_horizon"} <= set(predictions.columns)
        and np.allclose(predictions["score"], predictions["pi_incidence"] * predictions["p_horizon"])
        and predictions["score"].between(0, 1).all(),
    )
    check(
        "cure por run_cv: con `preprocessing: standard` falla (necesita el panel crudo)",
        _raises(lambda: run_cv(panel, splits, preprocessing="standard", **cv_args), TypeError),
    )


def main() -> int:
    set_seed(7)
    panel = build_dummy_panel(SMALL_PANEL)

    # 0 · contrato de parsing y umbrales compartidos
    sources = load_config("configs/data/raw_sources.yaml")
    features = load_config("configs/data/features_v1.yaml")
    eda_cfg = resolved_eda_config(load_config("configs/data/eda_cache.yaml"))
    check(
        "loader: fechas mixtas usan ISO8601",
        all(sources["tables"][name].get("date_format") == "ISO8601" for name in ("trips", "signals")),
    )
    regen_threshold = float(features["thresholds"]["regen_drop_points"])
    check(
        "features: regeneración exige una caída de 15 puntos",
        regen_threshold == 15.0,
    )
    check(
        "EDA y fallback comparten el umbral de regeneración",
        float(eda_cfg["regen_drop_points"]) == regen_threshold
        and float(DEFAULT_THRESHOLDS["regen_drop_points"]) == regen_threshold,
    )

    # 1 · contrato de datos
    missing = [c for c in CONTRACT_COLUMNS if c not in panel.columns]
    check("panel: columnas del contrato presentes", not missing, f"faltan {missing}" if missing else "")
    check("panel: hay features con prefijo feat_/static_", len(select_feature_columns(panel)) > 0)
    check("panel: cut_date es datetime nullable", str(panel["cut_date"].dtype).startswith("datetime"))
    check(
        "panel: time_to_event_km es NaN exactamente en los censurados",
        bool(
            panel.loc[panel["event_observed"] == 0, "time_to_event_km"].isna().all()
            and panel.loc[panel["event_observed"] == 1, "time_to_event_km"].notna().all()
        ),
    )
    check(
        "panel: ningún label positivo dentro del gap de blanking",
        bool((panel.loc[panel["label"] == 1, "time_to_event_km"] >= panel.loc[panel["label"] == 1, "gap_km"]).all()),
    )
    check(
        "panel: aux_km_observed_after_cut está en toda fila",
        "aux_km_observed_after_cut" in panel.columns
        and bool(panel["aux_km_observed_after_cut"].notna().all()),
    )

    # 1b · el objetivo de supervivencia es la misma etiqueta escrita sobre el eje de km.
    # Si esto se rompe, el modelo entrena contra algo que no es lo que se mide.
    all_rows = np.ones(len(panel), dtype=bool)
    spec = build_target("discrete_survival", panel, all_rows)
    duration, event, at_risk = spec.y["duration_km"], spec.y["event"], spec.y["at_risk"]
    horizon = panel["horizon_km"].to_numpy()
    derived = (at_risk & (event == 1) & (duration <= horizon)).astype(int)
    check(
        "target: `discrete_survival` reconstruye exactamente `label`",
        bool((derived == panel["label"].to_numpy()).all()),
        f"{int((derived != panel['label'].to_numpy()).sum())} filas discrepan",
    )
    # El dummy genera cortes hasta el evento, así que sí tiene filas dentro del gap:
    # el chequeo verifica que queden fuera de riesgo y no que no existan.
    inside_gap = panel["time_to_event_km"].lt(panel["gap_km"]).fillna(False).to_numpy()
    check(
        "target: los cortes dentro del gap quedan fuera de riesgo (regla 1)",
        bool((~at_risk[inside_gap]).all()) and bool((duration[at_risk] >= 0).all())
        and spec.params["gap_km"] == float(panel["gap_km"].iloc[0]),
        f"{int(inside_gap.sum())} filas del dummy caen dentro del gap",
    )
    check(
        "target: el `y` está alineado con las filas de train, no con el panel",
        len(build_target("discrete_survival", panel, panel["label"].eq(0).to_numpy()).y)
        == int(panel["label"].eq(0).sum()),
    )
    check(
        "target: un modo que no existe falla en vez de entrenar con `label`",
        _raises(lambda: build_target("no_existe", panel, all_rows), KeyError),
    )

    # 1b · reformulación ordinal y costo, ambos gobernados por YAML. Se verifican las
    # DOS variantes registradas, porque la diferencia entre ellas es justamente dónde
    # cae el último bin respecto de G+H y eso cambia qué mide el target.
    label_array = panel["label"].to_numpy(dtype=int) == 1
    observed = panel["event_observed"].to_numpy(dtype=int) == 1
    ordinal_variants = {
        "restringidos": "configs/exp_ordinal_horizon.yaml",
        "extendidos": "configs/exp_ordinal_horizon_ext.yaml",
    }
    ordinal_targets = {}
    ordered_ok, sane_ok, score_ok = True, True, True
    for variant, path in ordinal_variants.items():
        cfg_variant = load_config(path)
        params = cfg_variant["target"]["params"]
        y = build_ordinal_target(panel, np.ones(len(panel), dtype=bool), **params).y
        ordinal_targets[variant] = (cfg_variant, y)
        visible = y > 0
        ordered = np.argsort(panel.loc[visible, "time_to_event_km"].to_numpy(dtype=float))
        # Invariantes que valen para cualquier grilla de bins.
        sane_ok &= bool((y[~observed] == 0).all())
        ordered_ok &= bool((np.diff(y[visible][ordered]) <= 0).all())
        # El score comparable es P(clase >= min_class) y tiene que reconstruir `label`
        # exactamente, cualquiera sea el último bin. Es lo que hace comparable la
        # corrida con el control binario.
        bins = np.asarray(params["bins_km"], dtype=float)
        min_class = len(bins) - int(np.flatnonzero(bins == params["score_max_tte_km"])[0])
        score_ok &= bool(np.array_equal(y >= min_class, label_array))
    check("target ordinal: los censurados son clase 0 en toda variante", sane_ok)
    check("target ordinal: la clase crece con la proximidad al evento", ordered_ok)
    check(
        "target ordinal: P(clase >= min_class) reconstruye `label` en toda variante",
        score_ok,
    )
    # La diferencia entre las dos variantes, medida y no supuesta: con el último bin en
    # G+H el target vive adentro de la clase positiva y no dice nada de las negativas;
    # extendiéndolo, las clases intermedias se llenan con filas `label = 0`.
    restricted = ordinal_targets["restringidos"][1]
    extended = ordinal_targets["extendidos"][1]
    check(
        "target ordinal: solo los bins extendidos informan sobre filas negativas",
        bool(
            np.array_equal(restricted > 0, label_array)
            and ((extended > 0) & ~label_array).sum() > 0
            # Y la clase 0 extendida conserva filas de vehículos con evento: si no,
            # el target sería la cohorte de muestreo disfrazada de horizonte.
            and ((extended == 0) & observed).sum() > 0
        ),
    )
    matrix = ordinal_targets["restringidos"][0]["eval"]["cost_matrix"]
    expected_cost = float(matrix[0][4] + matrix[4][0])
    check(
        "métricas: la matriz cobra por clase real/predicha y conserva la asimetría",
        cost_matrix_score([0, 4], [4, 0], matrix) == expected_cost
        and matrix[4][0] > matrix[0][4],
    )
    # 1c · el barrido de costo: lo que reemplaza a la matriz única en la variante nueva.
    sweep_ratios = load_config("configs/exp_ordinal_horizon_ext.yaml")["eval"]["cost_ratios"]
    sweep_label = np.r_[np.ones(20, dtype=int), np.zeros(140, dtype=int)]  # p = 0,125
    oracle = cost_ratio_sweep(sweep_label, sweep_label.astype(float), sweep_ratios)
    # Score constante: no ordena nada, así que ningún umbral separa y el óptimo es
    # siempre una de las dos políticas triviales.
    flat = cost_ratio_sweep(sweep_label, np.full(len(sweep_label), 0.5), sweep_ratios)
    check(
        "costo: el barrido cruza las políticas triviales en (1-p)/p",
        bool(np.isclose(cost_ratio_breakeven(oracle)["always_beats_never_above_ratio"], 7.0)),
    )
    check(
        "costo: el oráculo ahorra todo y un score constante no ahorra nada",
        bool(
            all(row["savings_frac"] == 1.0 for row in oracle)
            and all(row["savings_vs_trivial"] == 0.0 for row in flat)
        ),
    )
    check(
        "costo: ningún modelo puede costar más que la mejor política trivial",
        bool(all(row["model_cost"] <= row["trivial_cost"] + 1e-9 for row in oracle + flat)),
    )

    # 2 · splits antileakage
    splits = make_splits(panel, n_splits=5, seed=7)
    covered = np.zeros(len(panel), dtype=bool)
    leak_free = True
    for _, train_mask, valid_mask in iter_folds(panel, splits):
        covered |= valid_mask
        overlap = set(panel.loc[train_mask, "vehicle_id"]) & set(panel.loc[valid_mask, "vehicle_id"])
        leak_free &= not overlap
    check("splits: ningún vehículo en train y validación", leak_free)
    check("splits: los folds cubren todo el panel", bool(covered.all()))
    check(
        "splits: todos los folds tienen vehículos con evento",
        all(f["n_valid_event_vehicles"] > 0 for f in splits["folds"]),
    )

    # 2b · la estratificación nueva no cambia los folds mientras las dos variables
    # coincidan (en el panel dummy y en el real, todo vehículo con evento tiene al
    # menos un corte positivo). Es la compatibilidad hacia atrás, demostrada.
    legacy = make_splits(
        panel, n_splits=5, seed=7, stratify_column="event_observed", stratify_level="row"
    )
    same_folds = [f["valid_vehicles"] for f in legacy["folds"]] == [
        f["valid_vehicles"] for f in splits["folds"]
    ]
    check(
        "splits: estratificar por label/vehículo da los mismos folds que por event_observed",
        same_folds,
        "mientras todo vehículo con evento tenga un corte positivo",
    )

    # 2c · la guarda dura: un fold sin positivos suficientes tiene que hacer ruido.
    thin = panel.copy()
    thin["label"] = 0
    # Seis vehículos con evento, una sola fila positiva cada uno: alcanzan para los 5
    # folds (no dispara la guarda de "pocos vehículos"), pero reparten 6 positivos en
    # 5 folds, así que alguno queda con 1 < 2.
    positives = [g.index[-1] for _, g in
                 panel[panel["event_observed"] == 1].groupby("vehicle_id", observed=True)][:6]
    thin.loc[positives, "label"] = 1
    thin_error = ""
    try:
        make_splits(thin, n_splits=5, seed=7, min_valid_positives=2)
    except ValueError as exc:
        thin_error = str(exc)
    check(
        "splits: un fold con pocos positivos hace fallar el armado",
        "label=1" in thin_error and "fold" in thin_error.lower(),
        thin_error.split(".")[0] if thin_error else "no falló (debería)",
    )

    # 2d · y si el splits.json es viejo y se armó sin la guarda, el chequeo se
    # rehace al entrenar contra el panel que toca.
    late_error = ""
    try:
        list(iter_folds(panel, splits, min_valid_positives=10_000))
    except ValueError as exc:
        late_error = str(exc)
    check(
        "splits: la guarda se rechequea al entrenar, no solo al armar los folds",
        "chequeado al entrenar" in late_error,
        late_error.split(".")[0] if late_error else "no falló (debería)",
    )

    few_error = ""
    try:
        make_splits(thin.assign(label=0), n_splits=5, seed=7)
    except ValueError as exc:
        few_error = str(exc)
    check(
        "splits: sin vehículos positivos suficientes no se arman folds",
        "vehículo(s) positivos" in few_error,
        few_error.split(".")[0] if few_error else "no falló (debería)",
    )

    # 2e · CV repetida: la repetición 0 es, exactamente, la CV simple.
    repeated = make_splits(panel, n_splits=5, seed=7, n_repeats=3)
    check(
        "splits: R=3 conserva los folds de R=1 como repetición 0",
        [f["valid_vehicles"] for f in repeated["repeats"][0]["folds"]]
        == [f["valid_vehicles"] for f in splits["folds"]],
    )
    check(
        "splits: cada repetición tiene su propia semilla y sus propios folds",
        len({r["seed"] for r in repeated["repeats"]}) == 3
        and [f["valid_vehicles"] for f in repeated["repeats"][1]["folds"]]
        != [f["valid_vehicles"] for f in repeated["repeats"][0]["folds"]],
    )
    check(
        "splits: iter_repeats recorre R juegos de folds sin fugas",
        sum(len(masks) for _, masks in iter_repeats(panel, repeated)) == 15,
    )

    # 3 · el harness sobre un modelo que no sabe nada
    predictions, fold_metrics = run_cv(panel, splits, model_name="baserate")
    oof = classification_metrics(predictions["label"], predictions["score"])
    check(
        "cv: PR-AUC de la tasa base ≈ tasa base",
        abs(oof["pr_auc"] - oof["base_rate"]) < 0.01,
        f"PR-AUC={oof['pr_auc']:.4f} vs base={oof['base_rate']:.4f}",
    )
    check(
        "cv: ROC-AUC de la tasa base ≈ 0,5",
        abs(oof["roc_auc"] - 0.5) < 0.05,
        f"ROC-AUC={oof['roc_auc']:.4f}",
    )
    check("cv: hay predicción out-of-fold para toda fila", predictions["score"].notna().all())
    check("cv: una métrica por fold", len(fold_metrics) == splits["n_splits"])
    check(
        "cv: con R=1 las columnas son las de siempre (score, fold)",
        "score_std" not in predictions.columns
        and not [c for c in predictions.columns if c.startswith(("score_r", "fold_r"))],
    )

    # La CV repetida con R=1 tiene que dar exactamente la CV simple; con R=3, la
    # repetición 0 sigue siendo la misma corrida (lo que se suma no pisa lo que había).
    rep_predictions, rep_fold_metrics = run_cv(panel, repeated, model_name="baserate")
    check(
        "cv: R=3 reproduce la repetición 0 de la CV simple",
        bool(np.allclose(rep_predictions["score_r0"], predictions["score"])),
    )
    check("cv: R=3 corre 3 × 5 folds", len(rep_fold_metrics) == 15)
    check(
        "cv: con R>1 el score guardado es el promedio de las repeticiones",
        bool(
            np.allclose(
                rep_predictions["score"],
                rep_predictions[[f"score_r{r}" for r in range(3)]].mean(axis=1),
            )
        ),
    )
    check(
        "cv: lead_time_curve funciona sobre las predicciones promediadas",
        len(lead_time_curve(rep_predictions, k_consecutive=2)) > 1,
    )

    dup_error = ""
    try:
        lead_time_curve(pd.concat([predictions, predictions], ignore_index=True))
    except ValueError as exc:
        dup_error = str(exc)
    check(
        "métricas: un corte repetido no pasa como alerta sostenida",
        "repetido" in dup_error,
        dup_error.split(".")[0] if dup_error else "no falló (debería)",
    )

    # 4 · las métricas de anticipación distinguen señal de ruido
    rng = np.random.default_rng(7)
    oracle = predictions.copy()
    # Oráculo: score que crece al acercarse el evento (y cero en los sanos).
    oracle["score"] = np.where(
        oracle["event_observed"] == 1,
        1.0 / (1.0 + oracle["time_to_event_km"].to_numpy() / 1000.0),
        0.0,
    )
    oracle_curve = lead_time_curve(oracle, k_consecutive=2)
    oracle_point = operating_point(oracle_curve, max_false_alarms_per_1000=50)
    check(
        "métricas: el oráculo detecta con 0 falsas alarmas",
        oracle_point is not None and oracle_point["detection_rate"] > 0.8,
        f"detección={oracle_point['detection_rate']:.2f}, anticipación mediana="
        f"{oracle_point['median_lead_km']:.0f} km" if oracle_point else "sin punto factible",
    )

    noise = predictions.copy()
    noise["score"] = rng.random(len(noise))
    noise_curve = lead_time_curve(noise, k_consecutive=2)
    noise_point = operating_point(noise_curve, max_false_alarms_per_1000=50)
    check(
        "métricas: el ruido no detecta dentro del presupuesto",
        noise_point is None or noise_point["detection_rate"] < 0.2,
        f"detección={noise_point['detection_rate']:.2f}" if noise_point else "sin punto factible",
    )
    check(
        "métricas: K más alto no aumenta las falsas alarmas",
        float(lead_time_curve(noise, k_consecutive=3)["false_alarms_per_1000"].max())
        <= float(noise_curve["false_alarms_per_1000"].max()) + 1e-9,
    )

    # 5 · agregación a nivel vehículo (MIL): la capa de decisión, no un modelo nuevo
    bags = {how: vehicle_scores(predictions, how, k=3) for how in VEHICLE_AGGREGATIONS}
    check(
        "MIL: una bolsa por vehículo, etiquetada con event_observed",
        all(len(b) == predictions["vehicle_id"].nunique() for b in bags.values())
        and bool(
            (
                bags["max"].set_index("vehicle_id")["label"]
                == predictions.groupby("vehicle_id")["event_observed"].max()
            ).all()
        ),
    )
    check(
        "MIL: mean <= topk <= max <= noisy_or en toda bolsa",
        bool(
            (bags["mean"]["score"] <= bags["topk"]["score"] + 1e-12).all()
            and (bags["topk"]["score"] <= bags["max"]["score"] + 1e-12).all()
            and (bags["max"]["score"] <= bags["noisy_or"]["score"] + 1e-12).all()
        ),
    )

    # El oráculo de bolsa: score alto en los vehículos con evento y bajo en los sanos.
    # Si la agregación funciona, el PR-AUC por vehículo tiene que ser ~1 con cualquiera.
    vehicle_oracle = predictions.copy()
    vehicle_oracle["score"] = np.where(vehicle_oracle["event_observed"] == 1, 0.9, 0.1)
    oracle_metrics = {how: vehicle_metrics(vehicle_oracle, how, k=3) for how in VEHICLE_AGGREGATIONS}
    check(
        "MIL: el oráculo por vehículo saca PR-AUC ≈ 1 con max/mean/topk",
        all(oracle_metrics[how]["pr_auc"] > 0.99 for how in ("max", "mean", "topk")),
        ", ".join(f"{how}={m['pr_auc']:.3f}" for how, m in oracle_metrics.items()),
    )
    # noisy_or ni siquiera con el oráculo: 1 − Π(1 − p) satura con el tamaño de la
    # bolsa, así que un vehículo sano con muchos cortes supera a uno con evento y
    # pocos. No es un bug de la implementación, es la agregación: queda medido acá
    # para que nadie la elija sin saberlo (docs/memoria/f3-mil-agregacion-vehiculo.md).
    # Entre los sanos el riesgo por corte es constante (0,1), así que lo único que
    # queda ordenando sus bolsas es cuántos cortes tienen.
    healthy_bags = vehicle_scores(vehicle_oracle, "noisy_or").query("label == 0")
    size_rank_corr = float(healthy_bags["score"].corr(healthy_bags["n_cuts"], method="spearman"))
    check(
        "MIL: noisy_or satura con el tamaño de la bolsa (mide historia, no riesgo)",
        oracle_metrics["noisy_or"]["pr_auc"] < 0.99 and size_rank_corr > 0.99,
        f"PR-AUC del oráculo={oracle_metrics['noisy_or']['pr_auc']:.3f}, "
        f"corr de rango(score, n_cuts) entre sanos={size_rank_corr:.2f}",
    )
    noise_vehicle = vehicle_metrics(noise, "max", k=3)
    check(
        "MIL: el ruido por vehículo no le gana a la tasa base de bolsas",
        noise_vehicle["pr_auc_lift"] < 1.5,
        f"PR-AUC={noise_vehicle['pr_auc']:.3f} vs tasa base {noise_vehicle['base_rate']:.3f} "
        f"(la de filas es {oof['base_rate']:.3f}: no son el mismo número)",
    )
    check(
        "MIL: la tasa base de vehículos no es la de filas (no se comparan los PR-AUC)",
        abs(noise_vehicle["base_rate"] - oof["base_rate"]) > 0.05,
        f"{noise_vehicle['base_rate']:.3f} vs {oof['base_rate']:.3f}",
    )

    # noisy_or sobre algo que no es una probabilidad: se niega en vez de calibrar solo.
    margins = predictions.copy()
    margins["score"] = 4.0 * (predictions["score"] - 0.5)
    noisy_error = ""
    try:
        vehicle_scores(margins, "noisy_or")
    except ValueError as exc:
        noisy_error = str(exc)
    check(
        "MIL: noisy_or se niega si los scores no son probabilidades",
        "probabilidades" in noisy_error,
        noisy_error.split(".")[0] if noisy_error else "no falló (debería)",
    )

    # Y si un vehículo tuviera cortes en dos folds, la bolsa dejaría de ser out-of-fold.
    leaky = predictions.copy()
    leaky.loc[leaky.index[0], "fold"] = (int(leaky.loc[leaky.index[0], "fold"]) + 1) % 5
    leak_error = ""
    try:
        vehicle_scores(leaky, "max")
    except ValueError as exc:
        leak_error = str(exc)
    check(
        "MIL: una bolsa repartida entre dos folds hace fallar la agregación",
        "out-of-fold" in leak_error,
        leak_error.split(".")[0] if leak_error else "no falló (debería)",
    )

    # El bloque del YAML: sin la clave se miden las cuatro; con lista vacía, ninguna.
    default_block = vehicle_block(predictions, {}, 1)
    check(
        "MIL: sin `eval.vehicle_aggregation` se miden las cuatro agregaciones",
        default_block is not None
        and tuple(default_block["aggregations"]) == VEHICLE_AGGREGATIONS
        and default_block["n_vehicles"] == predictions["vehicle_id"].nunique(),
    )
    check(
        "MIL: `vehicle_aggregation: []` apaga el eje de vehículo",
        vehicle_block(predictions, {"vehicle_aggregation": []}, 1) is None
        and vehicle_block(predictions, {"vehicle_aggregation": None}, 1) is None,
    )
    repeated_block = vehicle_block(rep_predictions, {"vehicle_aggregation": ["max"]}, 3)
    check(
        "MIL: con R>1 se agrega cada repetición por separado, no el score promediado",
        repeated_block is not None
        and len(repeated_block["by_repeat"]["max"]) == 3
        and abs(
            repeated_block["aggregations"]["max"]["pr_auc"]
            - float(np.mean([m["pr_auc"] for m in repeated_block["by_repeat"]["max"]]))
        )
        < 1e-12,
    )

    # 6 · descomposición cohorte / cuándo: contra qué piso se lee un PR-AUC por fila
    #
    # El panel dummy no tiene señal, así que acá no se verifican valores del panel real
    # (eso lo hace `audit_model.py` sobre dev): se verifican las propiedades que hacen
    # que la descomposición signifique algo.
    ceiling = cohort_ceiling(predictions["label"], predictions["vehicle_id"])
    check(
        "cohorte: el techo es la precisión de marcar todos los cortes de los fallados",
        abs(ceiling["pr_auc"] - ceiling["n_positive"] / ceiling["n_failed_rows"]) < 1e-9,
        f"techo={ceiling['pr_auc']:.4f} = {ceiling['n_positive']}/{ceiling['n_failed_rows']}",
    )
    check(
        "cohorte: el techo está por encima de la tasa base (identificar cohorte ya paga)",
        ceiling["pr_auc"] > ceiling["base_rate"] and ceiling["lift"] > 1.0,
        f"techo={ceiling['pr_auc']:.4f} vs tasa base={ceiling['base_rate']:.4f} "
        f"(lift {ceiling['lift']:.2f}x)",
    )
    check(
        "cohorte: puntuar por tasa del vehículo es una cota más laxa que el indicador",
        ceiling["pr_auc_rate"] >= ceiling["pr_auc"] - 1e-9,
        f"por tasa={ceiling['pr_auc_rate']:.4f} >= indicador={ceiling['pr_auc']:.4f}",
    )
    # El propio indicador de cohorte, puntuado por `pr_auc_within_failed`, no puede
    # ordenar nada: dentro de los fallados es constante. Ese es el sentido de la métrica.
    cohort_score = (
        predictions.groupby("vehicle_id", observed=True)["label"].transform("max").astype(float)
    )
    within_cohort = pr_auc_within_failed(
        predictions["label"], cohort_score, predictions["vehicle_id"]
    )
    check(
        "entre fallados: el identificador de cohorte perfecto no ordena (lift ≈ 1)",
        abs(within_cohort["lift"] - 1.0) < 1e-9,
        f"PR-AUC={within_cohort['pr_auc']:.4f} sobre tasa {within_cohort['base_rate']:.4f}",
    )
    check(
        "entre fallados: el azar de referencia es la tasa del subconjunto, no la global",
        within_cohort["base_rate"] > float(predictions["label"].mean()),
        f"{within_cohort['base_rate']:.4f} vs {float(predictions['label'].mean()):.4f}",
    )
    # (a') sobre un oráculo del *cuándo* (score = -time_to_event_km) tiene que dar
    # positivo: es un modelo que ordena los cortes dentro del auto y nada más.
    when_oracle = predictions.copy()
    when_oracle["score"] = -when_oracle["time_to_event_km"].fillna(
        when_oracle["time_to_event_km"].max() + 1.0
    )
    oracle_when = when_contribution(when_oracle)
    check(
        "(a'): un oráculo del cuándo pierde PR-AUC al colapsarse por vehículo",
        oracle_when["delta"] > 0,
        f"{oracle_when['pr_auc']:.4f} → {oracle_when['pr_auc_vehicle_mean']:.4f} "
        f"({oracle_when['delta']:+.4f})",
    )
    # Y sobre un score que ya es constante por vehículo tiene que dar exactamente 0:
    # no había nada del *cuándo* que borrar.
    flat_when = predictions.copy()
    flat_when["score"] = cohort_score
    check(
        "(a'): un score constante por vehículo no pierde nada (no sabía el cuándo)",
        abs(when_contribution(flat_when)["delta"]) < 1e-12,
        f"delta={when_contribution(flat_when)['delta']:+.2e}",
    )

    # Auditoría (b): promover las `aux_` de calendario no puede tocar el orden de lo que
    # ya entraba al modelo. En el panel secuencial las `aux_` van antes de `feat_seq_*`, y
    # los modelos secuenciales leen las primeras T × C numéricas como la secuencia: si las
    # promovidas quedan en su lugar, la secuencia se corre sin que nada falle.
    seq_layout = pd.DataFrame(
        {
            "vehicle_id": ["a", "b", "c", "d"],
            "static_SalesCountry_cd": ["X", "Y", "X", "Y"],
            **{c: [1.0, 2.0, 3.0, 4.0] for c in CALENDAR_AUX},
            **{f"feat_seq_t{t:03d}_c00_x": [0.0, 1.0, 0.0, 1.0] for t in range(3)},
        }
    )
    promoted_seq, _ = promote_aux(seq_layout, CALENDAR_AUX)
    X_seq = promoted_seq[select_feature_columns(promoted_seq)]
    seq_names = list(build_preprocessor(X_seq).fit(X_seq).get_feature_names_out())
    check(
        "audit (b): en el panel secuencial la secuencia sigue primero tras promover las aux_",
        all(n.startswith("feat_seq_") for n in seq_names[:3]),
        f"{seq_names}",
    )
    promoted_tab, _ = promote_aux(panel.assign(**{c: 0.0 for c in CALENDAR_AUX}), CALENDAR_AUX)
    renamed_tab = panel.assign(**{c: 0.0 for c in CALENDAR_AUX}).rename(
        columns={c: f"feat_{c}" for c in CALENDAR_AUX}
    )
    check(
        "audit (b): en un panel tabular promover no cambia el orden de las columnas del modelo",
        select_feature_columns(promoted_tab) == select_feature_columns(renamed_tab),
    )

    landmark_panel_checks()
    fleet_normalizer_checks()
    firth_checks()
    cure_em_checks()
    cure_model_checks()

    failed = [name for name, ok, _ in _checks if not ok]
    print()
    if failed:
        print(f"{len(failed)} chequeo(s) fallaron: {failed}")
        return 1
    print(f"Todo verde ({len(_checks)} chequeos). El harness de F0 está sano.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
