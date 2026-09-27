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
10. D1 y D2 del panel de hitos: el C con entrada tardía contra un caso a mano y contra
    lifelines, el umbral de la primera alerta, el bootstrap por vehículo y el bloque
    `eval` (`landmark`, `legacy_blocks`, `carry_columns`) de punta a punta.
11. Extender folds (`extend_splits`): los vehículos del split base conservan su fold,
    los nuevos se reparten estratificados y la guarda por hito falla a tiempo.
12. E1/E2: el bagging por vehículo sortea autos enteros y pasa por el loop de CV, y el
    ensamble por rango alinea por clave, se niega a juntar folds distintos y aplica la
    regla de veredicto del preregistro.
13. La incidencia externa (F5 §3.2): el sorteo padre se reproduce y se verifica contra su
    huella, el conjunto externo no toca dev ni test, la ventana de features fija no mueve
    las filas ni mira después de venta + 30, el ajuste de la fuente recupera la señal y el
    scorer congelado no aprende nada en `fit`.
14. El reloj en días con la ventana del registro (F5 §3.3): la fecha de un odómetro es la
    inversa exacta de la del evento, el tramo en riesgo entra tarde y sale en el fin de la
    ventana, el PEM apila la exposición exacta y recupera la tasa, y la etiqueta corregida
    evalúa solo las filas con el horizonte adentro.
15. La explicabilidad de K2 (F4): la reconstrucción de los folds es `run_cv` bit a bit, V1 es
    aditivo sobre el log-odds medio del hazard, la agregación por vehículo usa el umbral y la
    `k` de la curva, la referencia de flota sale solo de los sanos del train, el mensaje nunca
    nombra contexto ni síntomas y ningún output lleva un vehículo de test.
16. La entrega v2 (26-09-2026): el loader corrige cada parte (renombre, corrimiento de
    `ProductionDay`, fin de extracción) también en lecturas parciales; el dedupe se queda con
    el primer evento; el estrato compuesto funde los chicos y el holdout por estrato es 1/5 ± 1;
    la ventana de producción del universo, el sorteo en una etapa, la extensión de un holdout y
    la comparación con el holdout viejo; y los folds sin `extra_columns` son los de siempre.
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
from scripts.train import evaluate_predictions, vehicle_block  # noqa: E402
from src.config import load_config, set_seed  # noqa: E402
from src.eval.metrics import (  # noqa: E402
    VEHICLE_AGGREGATIONS,
    classification_metrics,
    cohort_ceiling,
    cost_matrix_score,
    cost_ratio_breakeven,
    cost_ratio_sweep,
    _d2_values,
    _landmark_vehicles,
    _vehicle_max,
    landmark_bootstrap_draws,
    landmark_eval_config,
    landmark_detection,
    late_entry_concordance,
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
from src.eval.splits import (  # noqa: E402
    extend_splits,
    iter_folds,
    iter_repeats,
    make_splits,
    panel_fingerprint,
    save_splits,
    test_split_masks,
)
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
    registran dentro de una ventana de 300 d. Trae también lo que miden D1 y D2: negativo
    resuelto (exposición ≥ 90 d desde max(venta + 60, apertura)), días y km al evento
    (40 km/d) y el fin de la ventana.
    """
    rng = np.random.default_rng(seed)
    markets = np.where(np.arange(n) % 2 == 0, "X", "Y")
    latent = rng.normal(size=(n, 2))
    susceptible = rng.random(n) < expit(-0.9 + 1.0 * latent[:, 0] - 0.8 * latent[:, 1])
    sale = rng.uniform(-150.0, 250.0, n)
    window = 300.0
    t = 200.0 * ((90.0 / 200.0) ** 2 - np.log(rng.random(n))) ** 0.5  # Weibull(2, 200) | T > 90
    registered = susceptible & (t >= -sale) & (t <= window - sale)
    resolved_exposure = (window - sale) - np.maximum(60.0, -sale)
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
                "aux_event_in_window": int(in_window[i]), "aux_dss_window_end": end[i],
                "aux_days_to_event_after_cut": t[i] - landmark if registered[i] else np.nan,
                "aux_resolved_negative": int(not registered[i] and resolved_exposure[i] >= 90.0),
                "aux_resolved_negative_e60": int(not registered[i] and resolved_exposure[i] >= 60.0),
                "aux_resolved_negative_e120": int(not registered[i] and resolved_exposure[i] >= 120.0),
            }
            row["time_to_event_km"] = 40.0 * row["aux_days_to_event_after_cut"]
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
    raw = X.drop(columns=[c for c in X.columns if c.startswith("feat_fm__")]).assign(
        feat_raw_a=panel["feat_fm__a__2025-09"], feat_raw_b=panel["feat_fm__b__2025-09"])
    ablation = CureMixtureModel(incidence="firth", fleet_normalization=False).fit(raw, y)
    check(
        "cure_mixture A5: sin normalizador las covariables son las crudas (ni el mercado ni el hito)",
        ablation.normalizer_ is None and ablation.covariates_ == ["feat_raw_a", "feat_raw_b"]
        and ablation.predict_components(raw)["score"].shape == (len(raw),),
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


LANDMARK_CARRY = ["aux_landmark_day", "aux_dss_entry", "aux_dss_exit", "aux_event_in_window", "aux_gap_days",
                  "aux_dss_window_end", "aux_days_to_event_after_cut", "aux_resolved_negative",
                  "aux_resolved_negative_e60", "aux_resolved_negative_e120"]


def _landmark_frame(n_neg: int = 20) -> pd.DataFrame:
    """Predicciones a mano para D2: 20 negativos resueltos, 5 positivos y 1 sano sin resolver.

    Dos hitos (30 y 60). El máximo de los negativos va de 0,01 a 0,20, así que con el 5%
    (m = 1) el umbral es el segundo mayor, 0,19. P1 alerta en 30, P2 recién en 60, P3 no
    llega, P4 empata con el umbral (no alerta: tiene que superarlo) y P5 cae entre el
    primero y el segundo negativo (alerta solo si τ es el segundo).
    """
    rows = []
    for i in range(n_neg):
        for landmark in (30.0, 60.0):
            rows.append({"vehicle_id": f"N{i:02d}", "aux_landmark_day": landmark,
                         "score": (i + 1) / 100 - (0.005 if landmark == 30.0 else 0.0),
                         "aux_event_in_window": 0, "aux_resolved_negative": 1, "event_dss": np.nan})
    positives = {"P1": (0.50, 0.40, 200.0), "P2": (0.10, 0.30, 180.0), "P3": (0.05, 0.10, 150.0),
                 "P4": (0.19, 0.18, 170.0), "P5": (0.195, 0.10, 160.0)}
    for vid, (s30, s60, event_dss) in positives.items():
        for landmark, score in ((30.0, s30), (60.0, s60)):
            rows.append({"vehicle_id": vid, "aux_landmark_day": landmark, "score": score,
                         "aux_event_in_window": 1, "aux_resolved_negative": 0, "event_dss": event_dss})
    rows += [{"vehicle_id": "S0", "aux_landmark_day": landmark, "score": 0.99, "aux_event_in_window": 0,
              "aux_resolved_negative": 0, "event_dss": np.nan} for landmark in (30.0, 60.0)]
    frame = pd.DataFrame(rows)
    frame["aux_days_to_event_after_cut"] = frame["event_dss"] - frame["aux_landmark_day"]
    frame["time_to_event_km"] = 40.0 * frame["aux_days_to_event_after_cut"]
    return frame


def landmark_metric_checks() -> None:
    """10 · D1 y D2 del panel de hitos, y el bloque `eval` que los activa."""
    # D1 a mano. (0,1) concordante; (0,2) no es comparable (2 entró después del evento de 0);
    # (0,3) comparable por el empate en tiempo sin evento, con empate de score (½); 4 nunca
    # estuvo en riesgo; (0,5) y (5,1) concordantes; (5,2) discordante. C = 3,5 / 5.
    entry = np.array([10, 10, 60, 10, 40, 10], dtype=float)
    exit_ = np.array([50, 80, 100, 50, 30, 70], dtype=float)
    event = np.array([1, 0, 0, 0, 0, 1])
    score = np.array([0.9, 0.1, 0.95, 0.9, 0.0, 0.5])
    hand = late_entry_concordance(entry, exit_, event, score)
    check(
        "D1: el C con entrada tardía cuenta solo los pares en riesgo en x_i, con empates ½",
        hand["n_pairs"] == 5 and np.isclose(hand["c_index"], 0.7),
        f"{hand['n_pairs']} pares, C = {hand['c_index']:.3f} (esperado 5 y 0,700)",
    )

    from lifelines.utils import concordance_index

    rng = np.random.default_rng(8)
    times = rng.exponential(100.0, 80)
    events = (rng.random(80) < 0.6).astype(int)
    scores = rng.normal(size=80) - times / 100.0
    ours = late_entry_concordance(np.zeros(80), times, events, scores)["c_index"]
    reference = concordance_index(times, -scores, events)
    check(
        "D1: sin entrada tardía es el C de Harrell de lifelines",
        np.isclose(ours, reference, atol=1e-12),
        f"{ours:.6f} vs. {reference:.6f}",
    )
    late = rng.uniform(0.0, 60.0, 80)
    counts = rng.integers(0, 4, 80)
    weighted = late_entry_concordance(late, times + late, events, scores, weights=counts)["c_index"]
    copies = np.repeat(np.arange(80), counts)
    duplicated = late_entry_concordance(late[copies], (times + late)[copies], events[copies], scores[copies])["c_index"]
    check(
        "D1: pesar los pares por multiplicidad es duplicar las filas (el bootstrap por vehículo)",
        np.isclose(weighted, duplicated, atol=1e-12),
        f"{weighted:.6f} vs. {duplicated:.6f}",
    )

    frame = _landmark_frame()
    cfg = landmark_eval_config({"lead_columns": {"days": "aux_days_to_event_after_cut", "km": "time_to_event_km"}})
    d2 = landmark_detection(frame, cfg, score_column="score", budget=0.05, n_event_vehicles_total=6)
    check(
        "D2: τ es el (m+1)-ésimo mayor de los negativos, alerta quien lo supera y el sano sin resolver no cuenta",
        d2["alerts_allowed"] == 1 and np.isclose(d2["threshold"], 0.19) and d2["n_false_alarms"] == 1
        and d2["n_detected"] == 3 and d2["n_negative"] == 20 and d2["n_positive"] == 5
        and np.isclose(d2["detection_rate_all_events"], 3 / 6),
        f"m = {d2['alerts_allowed']}, τ = {d2.get('threshold')}, detectados {d2['n_detected']}/5",
    )
    check(
        "D2: la anticipación es evento − hito de la primera alerta, en días y en km",
        d2["first_alert_landmark"] == {"30": 2, "60": 1}
        and np.isclose(d2["lead_days"]["median"], np.median([200 - 30, 180 - 60, 160 - 30]))
        and np.isclose(d2["lead_km"]["median"], 40 * np.median([200 - 30, 180 - 60, 160 - 30])),
        f"{d2['first_alert_landmark']}, mediana {d2['lead_days']['median']:.0f} d",
    )
    table = _landmark_vehicles(frame, cfg, cfg["resolved_column"])
    index = pd.Index(table.index)
    fast = _d2_values(_vehicle_max(frame, ["score"], index), table["positive"].to_numpy(),
                      table["negative"].to_numpy(), 0.05, np.ones((1, len(index))))[0, 0]
    check(
        "D2: la cuenta del bootstrap (máximo por vehículo) da lo mismo que la primera alerta",
        np.isclose(fast, d2["detection_rate"]),
        f"{fast:.3f} vs. {d2['detection_rate']:.3f}",
    )
    check(
        "eval.landmark: una clave que no existe falla en vez de ignorarse",
        _raises(lambda: landmark_eval_config({"budget": 0.05}), KeyError),
    )

    # De punta a punta: run_cv con R = 2 → evaluate_predictions con legacy_blocks: false.
    panel = _cure_panel()
    splits = make_splits(panel, n_splits=3, seed=5, min_valid_positives=1, n_repeats=2)
    predictions, _ = run_cv(panel, splits, model_name="cure_mixture", target={"name": "cure_window"},
                            model_params={"incidence": "unit_weight",
                                          "unit_weight_signs": {"fleet_a": 1, "fleet_b": -1}},
                            preprocessing="none", carry_columns=LANDMARK_CARRY)
    check(
        "carry_columns: viajan a las predicciones, y los extras del decoder van por repetición",
        set(LANDMARK_CARRY) <= set(predictions.columns)
        and {"unit_weight_score_r0", "unit_weight_score_r1", "pi_incidence_r1", "latency_shape_r0"}
        <= set(predictions.columns),
    )
    check(
        "carry_columns: una columna que el panel no tiene es un error",
        _raises(lambda: run_cv(panel, splits, model_name="baserate", carry_columns=["aux_no_existe"]), KeyError),
    )
    oracle = predictions.copy()
    truth = np.where(oracle["aux_event_in_window"].eq(1), 1.0 / oracle["aux_dss_exit"], 0.0)
    for column in ("unit_weight_score_r0", "unit_weight_score_r1"):
        oracle[column] = truth
    cfg_eval = {"eval": {"legacy_blocks": False,
                         "landmark": {"score_column": "unit_weight_score", "n_boot": 200}},
                "target": {"name": "cure_window"}}
    metrics, legacy_curve = evaluate_predictions(oracle, cfg_eval, n_repeats=2, seed=5)
    block = metrics["landmark"]
    check(
        "landmark_metrics: un oráculo del orden de los eventos da D1 = 1 y detecta a todos al 5%",
        np.isclose(block["d1"]["mean"]["mean"], 1.0) and np.isclose(block["d2"]["detection_rate"]["mean"], 1.0)
        and legacy_curve.empty and metrics["cohort_ceiling"] is None,
        f"D1 {block['d1']['mean']['mean']:.3f}, D2 {block['d2']['detection_rate']['mean']:.3f}",
    )
    noise = predictions.copy()
    for r, column in enumerate(("unit_weight_score_r0", "unit_weight_score_r1")):
        noise[column] = np.random.default_rng(100 + r).random(len(noise))
    noise_block = evaluate_predictions(noise, cfg_eval, n_repeats=2, seed=5)[0]["landmark"]
    check(
        "landmark_metrics: con ruido D1 cae a ≈ 0,5 y su intervalo lo contiene",
        abs(noise_block["d1"]["mean"]["mean"] - 0.5) < 0.1
        and noise_block["d1"]["bootstrap_by_vehicle"]["lo"] < 0.5 < noise_block["d1"]["bootstrap_by_vehicle"]["hi"],
        f"D1 {noise_block['d1']['mean']['mean']:.3f} "
        f"[{noise_block['d1']['bootstrap_by_vehicle']['lo']:.3f}, {noise_block['d1']['bootstrap_by_vehicle']['hi']:.3f}]",
    )
    real = evaluate_predictions(predictions, cfg_eval, n_repeats=2, seed=5)[0]["landmark"]
    check(
        "landmark_metrics: mide P0 con `unit_weight_score` y calibra con la probabilidad del cure",
        real["config"]["score_column"] == "unit_weight_score"
        and {"calibration", "latency_by_fold", "d1_on_time_entry", "d2_e_min", "d2_other_budgets"}
        <= set(real["informative"]) and len(real["curve"]) > 1,
    )
    at_30 = predictions["aux_landmark_day"].eq(30.0) & (predictions["aux_dss_exit"] > predictions["aux_dss_entry"])
    on_time_30 = int((at_30 & np.isclose(predictions["aux_dss_entry"], 60.0)).sum())
    reported = real["informative"]["d1_on_time_entry"]["by_landmark"]["30"]["n_at_risk"]
    check(
        "D1 con entrada en L+G: usa solo las filas en riesgo que entran en L + G (informativa)",
        reported == on_time_30 < int(at_30.sum()),
        f"{reported} de {int(at_30.sum())} filas en riesgo del hito 30",
    )
    draws_a = landmark_bootstrap_draws(predictions, cfg_eval["eval"]["landmark"], n_repeats=2, n_boot=50, seed=9)
    draws_b = landmark_bootstrap_draws(noise, cfg_eval["eval"]["landmark"], n_repeats=2, n_boot=50, seed=9)
    again = landmark_bootstrap_draws(predictions, cfg_eval["eval"]["landmark"], n_repeats=2, n_boot=50, seed=9)
    check(
        "bootstrap pareado: dos corridas del mismo panel reciben los mismos remuestreos de vehículos",
        draws_a["vehicles"] == draws_b["vehicles"] and np.array_equal(draws_a["d1"], again["d1"])
        and not np.allclose(draws_a["d1"], draws_b["d1"]),
    )
    stuck = predictions.assign(cut_odo=0.0)  # autos quietos: el mismo odómetro en los dos hitos
    legacy_cfg = {"eval": {"landmark": {"score_column": "unit_weight_score", "n_boot": 50}},
                  "target": {"name": "cure_window"}}
    check(
        "legacy_blocks: false apaga lo que no aplica (con cortes repetidos la curva vieja falla)",
        _raises(lambda: evaluate_predictions(stuck, legacy_cfg, n_repeats=2, seed=5), ValueError)
        and evaluate_predictions(stuck, cfg_eval, n_repeats=2, seed=5)[0]["landmark"] is not None,
    )


def _external_source(seed: int = 11, n: int = 360) -> pd.DataFrame:
    """Fuente sintética: tres mercados, dos meses, dos features de flota y el uso.

    Cada feature es un corrimiento por mes más el rasgo del vehículo; el riesgo sube con
    `a`, baja con `b` y baja con el uso, y el intercepto cambia por mercado (el estrato).
    """
    rng = np.random.default_rng(seed)
    markets = np.array(["X", "Y", "Z"])[np.arange(n) % 3]
    latent = rng.normal(size=(n, 3))
    alpha = {"X": -0.6, "Y": 0.0, "Z": 0.5}
    eta = np.array([alpha[m] for m in markets]) + 0.9 * (latent[:, 0] - latent[:, 1]) - 0.7 * latent[:, 2]
    label = (rng.random(n) < expit(eta)).astype(int)
    frame = pd.DataFrame({"vehicle_id": [f"S{i:03d}" for i in range(n)], "label": label,
                          "event_observed": label, "static_SalesCountry_cd": markets,
                          "feat_landmark_day": 30.0, "feat_log1p_km_per_day": 3.0 + 0.8 * latent[:, 2],
                          "aux_eligible": (rng.random(n) < 0.9).astype(int)})
    for month, shift in (("2025-09", 10.0), ("2025-10", 14.0)):
        frame[f"feat_fm__a__{month}"] = shift + 2.0 * latent[:, 0] + rng.normal(scale=0.3, size=n)
        frame[f"feat_fm__w_a__{month}"] = 100.0
        frame[f"feat_fm__b__{month}"] = shift + 2.0 * latent[:, 1] + rng.normal(scale=0.3, size=n)
        frame[f"feat_fm__w_b__{month}"] = 20.0
    return frame


def external_incidence_checks() -> None:
    """13 · El conjunto externo, la ventana fija y la incidencia congelada (F5 §3.2)."""
    import json
    import tempfile

    from src.data.external import reproduce_parent_split, select_external
    from src.eval.splits import make_test_split, restrict_test_split
    from src.models.incidence import FrozenIncidenceScorer, fit_external_incidence, save_incidence

    # -- el sorteo padre y quién entra ----------------------------------------------------
    rng = np.random.default_rng(4)
    vehicles = pd.DataFrame({"vehicle_id": [f"V{i:03d}" for i in range(200)],
                             "event_observed": (rng.random(200) < 0.3).astype(int),
                             "static_SalesCountry_cd": np.array(["A", "B", "C"])[rng.integers(0, 3, 200)],
                             "static_daysUntilSale": 80.0})
    vehicles["event_day_since_production"] = np.where(vehicles["event_observed"].eq(1),
                                                      np.where(rng.random(200) < 0.5, 80.0, 200.0), np.nan)
    full = make_test_split(vehicles, test_size=0.2, seed=42)
    keep = vehicles.loc[vehicles["static_SalesCountry_cd"].eq("A") & ~(vehicles["event_day_since_production"] == 80.0),
                        "vehicle_id"]
    frozen = restrict_test_split(full, keep, events=vehicles.set_index("vehicle_id")["event_observed"])
    parent = reproduce_parent_split(vehicles, frozen)
    other_seed = {**frozen, "seed": 43}
    other_print = {**frozen, "parent": {**frozen["parent"], "test_vehicles_sha256_16": "0" * 16}}
    check(
        "externo: el sorteo padre se reproduce contra su huella; con otra semilla u otra huella falla",
        parent["dev"] == full["dev_vehicles"] and parent["test"] == full["test_vehicles"]
        and _raises(lambda: reproduce_parent_split(vehicles, other_seed), ValueError)
        and _raises(lambda: reproduce_parent_split(vehicles, other_print), ValueError),
    )
    external, report = select_external(vehicles, frozen, parent, side="dev", markets=["B", "C"])
    ids = set(external["vehicle_id"])
    kept = set(frozen["dev_vehicles"]) | set(frozen["test_vehicles"])
    try:
        every_market, _ = select_external(vehicles, frozen, parent, side="dev", markets=None)
        all_markets_ok = not set(every_market["vehicle_id"]) & kept and set(every_market["vehicle_id"]) <= set(
            frozen["excluded_vehicles"])
    except ValueError:
        all_markets_ok = False
    check(
        "externo: solo excluidos del lado dev del padre y de los mercados pedidos; nada de dev ni de test",
        ids <= set(frozen["excluded_vehicles"]) and ids <= set(parent["dev"]) and not ids & kept
        and set(external["static_SalesCountry_cd"]) <= {"B", "C"} and len(ids) > 0 and all_markets_ok
        and report["excluded_on_other_side_untouched"] == len(set(frozen["excluded_vehicles"]) - set(parent["dev"])),
    )

    # -- la ventana de features fija ---------------------------------------------------------
    cfg = load_config("configs/data/panel_landmark_ps.yaml")
    window = load_config(cfg["event_clock"])["event_window"]
    features = load_fleet_features(cfg["fleet_features"], cfg["features_spec"])
    cfg30 = {**cfg, "landmark": {**cfg["landmark"], "feature_window_days": 30, "usage_feature": True}}
    lm, lm30 = landmark_config(cfg, window), landmark_config(cfg30, window)
    veh, trips = _synthetic_landmark_inputs()
    base, _ = build_landmark_panel(veh, trips, lm, features)
    fixed, _ = build_landmark_panel(veh, trips, lm30, features)
    same_rows = ["vehicle_id", "aux_landmark_day", "cut_odo", "window_km", "label", "aux_dss_entry", "aux_dss_exit",
                 "aux_event_in_window", "aux_n_trips_window", "aux_resolved_negative"]
    first = fixed.loc[fixed["aux_landmark_day"].eq(30)].set_index("vehicle_id")
    fm_first = [c for c in first.columns if c.startswith("feat_fm__")]
    later_equal = all(
        np.allclose(fixed.loc[fixed["aux_landmark_day"].eq(L)].set_index("vehicle_id")
                    .reindex(index=first.index, columns=fm_first).to_numpy(float),
                    first[fm_first].to_numpy(float), equal_nan=True)
        for L in (60, 90))
    fm_base = [c for c in base.columns if c.startswith("feat_fm__")]
    check(
        "ventana fija: mismas filas y aux_ que el panel del cure; en el hito 30 las mismas features",
        fixed[same_rows].equals(base[same_rows])
        and np.allclose(base.loc[base["aux_landmark_day"].eq(30), fm_base].to_numpy(float),
                        fixed.loc[fixed["aux_landmark_day"].eq(30)].reindex(columns=fm_base).to_numpy(float),
                        equal_nan=True),
    )
    sale = veh["sale_date"]
    after = trips["TripDatetimeEnd"] > trips["vehicle_id"].map(sale + pd.Timedelta(days=30))
    moved = trips.copy()
    moved.loc[after, "trip_km"] *= 5
    moved.loc[after, "below_regime"] = ~moved.loc[after, "below_regime"]
    moved.loc[after, "speed_kmh_moving"] = -moved.loc[after, "speed_kmh_moving"]
    shaken, _ = build_landmark_panel(veh, moved, lm30, features)
    watched = [c for c in fixed.columns if c.startswith(("feat_fm__", "aux_raw_")) or c == "feat_log1p_km_per_day"]
    check(
        "ventana fija: en los hitos 60 y 90 las features son las de los primeros 30 días, y lo posterior no entra",
        later_equal and shaken.reindex(columns=watched).equals(fixed[watched]),
    )
    a30 = trips.loc[trips["vehicle_id"].eq("A") & (trips["TripDatetimeStart"] > sale["A"])
                    & (trips["TripDatetimeEnd"] <= sale["A"] + pd.Timedelta(days=30)), "trip_km"].sum()
    a_row = fixed.loc[fixed["vehicle_id"].eq("A") & fixed["aux_landmark_day"].eq(90)]
    check(
        "ventana fija: km/día es log1p(km en (venta, venta + 30] / 30) y la exposición potencial no mira el desenlace",
        np.isclose(float(a_row["feat_log1p_km_per_day"].iloc[0]), np.log1p(a30 / 30.0))
        and bool(fixed.loc[fixed["event_observed"].eq(1), "aux_potential_exposure_days"].notna().all())
        and np.allclose(fixed.loc[fixed["event_observed"].eq(0), "aux_potential_exposure_days"],
                        fixed.loc[fixed["event_observed"].eq(0), "aux_resolved_exposure_days"]),
    )

    # -- el ajuste de la fuente y el scorer congelado --------------------------------------
    source = _external_source()
    kwargs = {"fleet_signs": {"fleet_a": 1, "fleet_b": -1}, "usage_column": "feat_log1p_km_per_day",
              "normalizer_params": {"levels": [["month"], []], "min_ref_vehicles": 10}}
    model = fit_external_incidence(source, design="index", **kwargs)
    five = fit_external_incidence(source, design="separate", **kwargs)
    X = source[select_feature_columns(source)]
    z = model.standardized(X)
    by_weights = z[list(model.weights())].to_numpy() @ np.asarray(list(model.weights().values()))
    check(
        "incidencia: el índice recupera a > 0 y b < 0, y el score es Σ w·z sin intercepto",
        model.slopes[0] > 0 and model.slopes[1] < 0 and np.allclose(model.linear_predictor(X), by_weights)
        and five.slopes[0] > 0 and five.slopes[1] < 0 and five.slopes[2] < 0,
        f"a {model.slopes[0]:+.2f} · b {model.slopes[1]:+.2f}",
    )
    target = X.assign(static_SalesCountry_cd="W")
    check(
        "incidencia: un mercado que la fuente no vio cae a la referencia del mes (sin NaN, sin error)",
        bool(np.isfinite(model.linear_predictor(target)).all())
        and np.allclose(model.linear_predictor(target), model.linear_predictor(X)),
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = save_incidence(model, Path(tmp) / "m.joblib")
        one = FrozenIncidenceScorer(artifact=str(path)).fit(X.iloc[:50], source["label"].iloc[:50])
        two = FrozenIncidenceScorer(artifact=str(path)).fit(X.iloc[200:], 1 - source["label"].iloc[200:])
        missing = X.drop(columns=["feat_log1p_km_per_day"])
        check(
            "incidencia: el scorer congelado no aprende nada en `fit` y falla si falta una covariable",
            np.array_equal(one.predict_proba(X), two.predict_proba(X))
            and np.allclose(one.decision_function(X), model.linear_predictor(X))
            and _raises(lambda: FrozenIncidenceScorer(artifact=str(path)).fit(missing), KeyError),
        )
        panel = _cure_panel()
        spec = build_target("cure_window", panel, np.ones(len(panel), dtype=bool))
        Xc = panel[select_feature_columns(panel)]
        weights_path = Path(tmp) / "w.json"
        weights_path.write_text(json.dumps({"fleet_a": 1.0, "fleet_b": -1.0}), encoding="utf-8")
        unit = CureMixtureModel(incidence="unit_weight", unit_weight_signs={"fleet_a": 1, "fleet_b": -1}).fit(Xc, spec.y)
        fixed_w = CureMixtureModel(incidence="fixed_weight", fixed_weights_path=str(weights_path)).fit(Xc, spec.y)
        same = np.allclose(unit.predict_components(Xc)[UNIT_WEIGHT_SCORE],
                           fixed_w.predict_components(Xc)[UNIT_WEIGHT_SCORE])
        weights_path.write_text(json.dumps({"fleet_a": 1.0}), encoding="utf-8")
        check(
            "cure_mixture: `fixed_weight` con pesos ±1 es P0, y un peso faltante falla",
            same and _raises(lambda: CureMixtureModel(incidence="fixed_weight",
                                                      fixed_weights_path=str(weights_path)).fit(Xc, spec.y), ValueError),
        )


def _window_trips() -> pd.DataFrame:
    """Viajes de tres autos: uno que falla en la ventana, uno sano que deja de andar y uno tardío."""
    def trips(vid: str, start: str, n: int, step_days: float, km: float) -> pd.DataFrame:
        dates = pd.Timestamp(start, tz="UTC") + pd.to_timedelta(np.arange(n) * step_days, unit="D")
        return pd.DataFrame({"vehicle_id": vid, "TripDatetimeStart": dates,
                             "OdometerTripEnd": np.arange(1, n + 1) * km})
    return pd.concat([
        trips("F", "2025-07-01", 400, 1.0, 50.0),     # 50 km/día desde julio de 2025
        trips("S", "2025-08-01", 120, 1.0, 50.0),     # deja de andar a fines de noviembre
        trips("T", "2026-02-01", 200, 1.0, 50.0),     # arranca un mes antes del fin de la ventana
    ], ignore_index=True)


def km_window_detection_checks() -> None:
    """15 · F6: la ventana del registro en km (K2), la media acumulada causal (K1/K3) y el nulo de detección."""
    from scripts.audit_detection_null import fast_detection
    from scripts.build_km_window_panel import (ENTRY_KM, EVENT_KM, EXIT_KM, km_risk_columns,
                                               window_odometers)
    from scripts.smooth_scores import causal_cummean
    from src.data.window_risk import RegistryWindow
    from src.models.survival_stacking import DiscreteSurvivalStacker

    trips = _window_trips()
    window = RegistryWindow.from_config({"start": "2025-09-01", "end": "2026-03-11"})

    # -- el tramo en riesgo en km ----------------------------------------------------------------
    # F: 50 km/día desde julio (3.150 km al 01-09, evento el 10-12 en 8.150 km). S: deja de andar
    # en 6.000 km. T: arranca en febrero de 2026 (1.950 km al fin de la ventana).
    odometers = window_odometers(pd.Index(["F", "S", "T"]), trips, window)
    panel = pd.DataFrame({
        "vehicle_id": ["F", "F", "S", "T", "T"],
        "cut_odo": [1000.0, 5000.0, 2000.0, 1000.0, 2500.0],
        "gap_km": 500.0,
        "event_observed": [1, 1, 0, 0, 0],
        "time_to_event_km": [7150.0, 3150.0, np.nan, np.nan, np.nan],
        "aux_km_observed_after_cut": [7150.0, 3150.0, 4000.0, 9000.0, 7500.0],
    })
    in_window = pd.Series({"F": True, "S": False, "T": False})
    risk = km_risk_columns(panel, odometers, in_window)
    check(
        "ventana en km: entra en el odómetro del 01-09 (0 si arranca después); el fallado sale en su "
        "evento y el sano en min(último odómetro, odómetro del 11-03)",
        np.allclose(risk[ENTRY_KM], [1650.0, 0.0, 0.0, 0.0, 0.0])
        and np.allclose(risk[EXIT_KM], [6650.0, 2650.0, 3500.0, 450.0, -1050.0])
        and risk[EVENT_KM].tolist() == [1, 1, 0, 0, 0],
        f"entrada {risk[ENTRY_KM].tolist()} · salida {risk[EXIT_KM].round(1).tolist()}",
    )

    # -- el target -----------------------------------------------------------------------------
    full = pd.concat([panel, risk], axis=1)
    spec = build_target("window_km_survival", full, np.ones(len(full), dtype=bool))
    broken = full.assign(**{EXIT_KM: full[ENTRY_KM] - 1.0})
    check(
        "target: `window_km_survival` trae la entrada, deja fuera de riesgo el corte posterior a la "
        "ventana, y un evento sin tramo falla",
        spec.y["entry_km"].tolist() == [1650.0, 0.0, 0.0, 0.0, 0.0]
        and spec.y["at_risk"].tolist() == [True, True, True, True, False]
        and spec.y["event"].tolist() == [1, 1, 0, 0, 0]
        and _raises(lambda: build_target("window_km_survival", broken, np.ones(len(full), dtype=bool)), ValueError),
    )

    # -- el apilado con entrada tardía ---------------------------------------------------------
    model = DiscreteSurvivalStacker(horizon_km=1000, bin_km=500, max_horizon_km=3000)
    model._train_edges_ = model._bin_edges(3000)
    X = np.arange(5, dtype=float)[:, None]
    duration = np.array([1200.0, 2600.0, 700.0, 4000.0, 5000.0])
    event = np.array([1, 0, 1, 1, 1])
    at_risk = np.ones(5, dtype=bool)
    stacked, hazard, _, rows = model._stack(X, duration, event, at_risk, None,
                                            np.array([0.0, 1100.0, 700.0, 3200.0, 0.0]))
    per_row = pd.Series(rows).value_counts().sort_index().to_dict()
    bins = stacked[:, -1] / 500.0
    check(
        "stacker: con entrada tardía apila desde el bin de la entrada; el evento cae en su bin, el que "
        "entra después del final no genera filas y el que pasa el final se censura",
        per_row == {0: 3, 1: 4, 2: 1, 4: 6} and bins[rows == 1].min() == 2
        and sorted(zip(rows[hazard == 1].tolist(), bins[hazard == 1].tolist())) == [(0, 2.0), (2, 1.0)],
        f"filas por corte {per_row}",
    )
    plain = model._stack(X, duration, event, at_risk, None)
    zero = model._stack(X, duration, event, at_risk, None, np.zeros(5))
    check(
        "stacker: sin `entry_km` el apilado es el de siempre (entrada en 0)",
        np.array_equal(plain[0], zero[0]) and np.array_equal(plain[1], zero[1]),
    )

    # -- por el loop de CV ----------------------------------------------------------------------
    dummy = build_dummy_panel(SMALL_PANEL)
    observed = dummy["event_observed"].to_numpy(int) == 1
    tte = dummy["time_to_event_km"].to_numpy(float) - dummy["gap_km"].to_numpy(float)
    dummy[ENTRY_KM] = np.where(np.arange(len(dummy)) % 4 == 0, 400.0, 0.0)
    dummy[EXIT_KM] = np.where(observed, tte, 5000.0)
    dummy[EVENT_KM] = (observed & (tte >= dummy[ENTRY_KM])).astype(int)
    dummy[EXIT_KM] = np.where(observed & (tte < dummy[ENTRY_KM]), dummy[ENTRY_KM] - 1.0, dummy[EXIT_KM])
    splits = make_splits(dummy, n_splits=3, seed=1)
    predictions, _ = run_cv(dummy, splits, model_name="survival_stacking",
                            model_params={"horizon_km": 3000, "bin_km": 500, "max_horizon_km": 6000},
                            target={"name": "window_km_survival", "params": {}})
    check(
        "ventana en km: `window_km_survival` + `survival_stacking` pasan por el loop de CV con scores en [0, 1]",
        bool(predictions["score"].between(0, 1).all()) and predictions["score"].nunique() > 10,
    )

    # -- la media acumulada causal ---------------------------------------------------------------
    preds = pd.DataFrame({
        "vehicle_id": ["A", "B", "A", "A", "B"], "cut_odo": [3.0, 1.0, 1.0, 2.0, 2.0],
        "score_r0": [0.9, 0.2, 0.3, 0.6, 0.4], "fold_r0": [0, 1, 0, 0, 1],
        "score_r1": [0.1, 0.5, 0.7, 0.4, 0.1], "fold_r1": [2, 0, 2, 2, 0],
    })
    smoothed = causal_cummean(preds).set_index(["vehicle_id", "cut_odo"])
    later = causal_cummean(preds.assign(score_r0=np.where(preds["cut_odo"].eq(3.0), 0.0, preds["score_r0"])))
    later = later.set_index(["vehicle_id", "cut_odo"])
    check(
        "media acumulada: promedio de las filas del vehículo hasta la propia, ordenadas por odómetro; "
        "cambiar un corte posterior no toca los anteriores",
        np.allclose(smoothed.loc["A", "score_r0"], [0.3, 0.45, 0.6])
        and np.allclose(smoothed.loc["B", "score_r1"], [0.5, 0.3])
        and np.allclose(smoothed["score"], (smoothed["score_r0"] + smoothed["score_r1"]) / 2)
        and np.allclose(later.loc["A", "score_r0"].iloc[:2], smoothed.loc["A", "score_r0"].iloc[:2]),
        f"A r0 {smoothed.loc['A', 'score_r0'].round(3).tolist()}",
    )
    check(
        "media acumulada: un vehículo con filas en dos folds de la misma repetición falla",
        _raises(lambda: causal_cummean(preds.assign(fold_r0=[0, 1, 1, 0, 1])), ValueError),
    )

    # -- la detección vectorizada del nulo --------------------------------------------------------
    rng = np.random.default_rng(5)
    sizes = rng.integers(1, 12, 80)
    frame = pd.DataFrame({
        "vehicle_id": np.repeat([f"V{i:02d}" for i in range(80)], sizes),
        "cut_odo": np.concatenate([np.arange(s) * 500.0 for s in sizes]),
        "event_observed": np.repeat((np.arange(80) % 3 == 0).astype(int), sizes),
    })
    frame["time_to_event_km"] = np.where(frame["event_observed"].eq(1), 10000.0 - frame["cut_odo"], np.nan)
    frame["score"] = rng.random(len(frame)) + 0.3 * frame["event_observed"]
    frame = frame.sort_values(["vehicle_id", "cut_odo"]).reset_index(drop=True)
    eval_cfg = {"k_consecutive": 2, "n_thresholds": 50, "max_false_alarms_per_1000": 100}
    fast, _ = fast_detection(frame, frame["score"].to_numpy(), eval_cfg)
    point = operating_point(lead_time_curve(frame, n_thresholds=50, k_consecutive=2), max_false_alarms_per_1000=100)
    check(
        "nulo de detección: la versión vectorizada reproduce `operating_point` de la curva de anticipación",
        point is not None and abs(fast - float(point["detection_rate"])) < 1e-12,
        f"{fast:.4f} vs {float(point['detection_rate']) if point else float('nan'):.4f}",
    )


def window_survival_checks() -> None:
    """14 · El reloj en días con la ventana del registro (F5 §3.3): panel, target, PEM y evaluación."""
    import json
    import tempfile

    from scripts.eval_window_label import attach_panel_columns, floor_metrics_for, label_masks, read_a0, verdict
    from src.data.anchor import EPOCH, project_dates_to_odometer, project_odometer_to_dates
    from src.data.window_risk import (CUT_DSS, ENTRY, EVALUABLE, EVENT, EXIT, HORIZON, RegistryWindow,
                                      window_risk_columns)
    from src.models.window_stacking import WindowPEMStacker

    trips = _window_trips()
    window = RegistryWindow.from_config({"start": "2025-09-01", "end": "2026-03-11"})

    # -- la inversa de la proyección del evento ----------------------------------------------
    event = pd.Series([pd.Timestamp("2025-12-10 00:00", tz="UTC")], index=["F"])
    event_odo = project_dates_to_odometer(event, trips)["event_odo_km"]
    plateau = pd.concat([trips.loc[trips["vehicle_id"].eq("S")].iloc[:3],
                         pd.DataFrame({"vehicle_id": ["S"], "TripDatetimeStart": [pd.Timestamp("2025-08-10", tz="UTC")],
                                       "OdometerTripEnd": [150.0]})], ignore_index=True)
    back = project_odometer_to_dates(pd.DataFrame({"vehicle_id": ["F", "S", "S", "S"],
                                                   "odo": [event_odo["F"], 150.0, 1e6, 10.0]}), trips)
    first_of_plateau = project_odometer_to_dates(pd.DataFrame({"vehicle_id": ["S"], "odo": [150.0]}), plateau)
    check(
        "reloj: la fecha de un odómetro es la inversa exacta de la proyección del evento; nunca llegar da NaT",
        abs((back.iloc[0] - event["F"]) / pd.Timedelta(hours=1)) < 1e-3
        and back.iloc[2] is pd.NaT and back.iloc[3] is pd.NaT
        and first_of_plateau.iloc[0] == pd.Timestamp("2025-08-03", tz="UTC"),
        f"{back.tolist()} · meseta {first_of_plateau.iloc[0]}",
    )

    # -- las columnas del tramo en riesgo ----------------------------------------------------
    sale = {"F": "2025-07-01", "S": "2025-08-01", "T": "2026-02-01"}
    vehicles = pd.DataFrame({
        "event_observed": [1, 0, 0],
        "sale_date": [pd.Timestamp(sale[v], tz="UTC") for v in ("F", "S", "T")],
        "event_date": [event["F"], pd.NaT, pd.NaT],
    }, index=["F", "S", "T"])
    # F: un corte antes de la ventana y otro adentro; S: adentro, con horizonte que pasa su último
    # viaje; T: después de la ventana.
    panel = pd.DataFrame({
        "vehicle_id": ["F", "F", "S", "S", "T"],
        "cut_odo": [1000.0, 5000.0, 2000.0, 5000.0, 2500.0],
        "gap_km": 500.0, "horizon_km": 1000.0,
        "event_observed": [1, 1, 0, 0, 0],
    })
    panel["cut_date"] = project_odometer_to_dates(panel[["vehicle_id"]].assign(odo=panel["cut_odo"]),
                                                  trips).dt.tz_convert(None)
    risk = window_risk_columns(panel, trips, vehicles, window)
    t0 = project_odometer_to_dates(panel[["vehicle_id"]].assign(odo=panel["cut_odo"] + 500.0), trips)
    day = pd.Timedelta(days=1)
    expected_entry = ((window.start - t0) / day).clip(lower=0)
    expected_exit = pd.Series([(event["F"] - t0[0]) / day, (event["F"] - t0[1]) / day,
                               (window.end - t0[2]) / day, (window.end - t0[3]) / day,
                               (window.end - t0[4]) / day])
    check(
        "reloj: entrada tardía antes de la ventana; el sano sale en el fin de la ventana, no en su último viaje",
        np.allclose(risk[ENTRY], expected_entry) and np.allclose(risk[EXIT], expected_exit)
        and risk.loc[0, ENTRY] > 0 and risk.loc[1, ENTRY] == 0
        and risk.loc[3, EXIT] > (pd.Timestamp("2025-11-28", tz="UTC") - t0[3]) / day,
        f"entrada {risk[ENTRY].round(1).tolist()} · salida {risk[EXIT].round(1).tolist()}",
    )
    check(
        "reloj: evento solo en el fallado, y el corte posterior a la ventana queda fuera de riesgo",
        risk[EVENT].tolist() == [1, 1, 0, 0, 0] and risk.loc[4, EXIT] < risk.loc[4, ENTRY],
    )
    t_h = project_odometer_to_dates(panel[["vehicle_id"]].assign(odo=panel["cut_odo"] + 1500.0), trips)
    expected_eval = (t0.notna() & t_h.notna() & (t0 >= window.start) & (t_h <= window.end)).astype(int)
    check(
        "reloj: evaluable = horizonte entero en la ventana (igual para positivas y negativas); "
        "feat_cut_dss = días desde la venta en el corte",
        risk[EVALUABLE].tolist() == expected_eval.tolist() == [0, 1, 1, 0, 0]
        and np.isnan(risk.loc[3, HORIZON])
        and np.allclose(risk[CUT_DSS], (pd.to_datetime(panel["cut_date"]).dt.tz_localize("UTC")
                                         - panel["vehicle_id"].map(vehicles["sale_date"])) / day),
        f"evaluables {risk[EVALUABLE].tolist()}",
    )

    # -- el target ---------------------------------------------------------------------------
    full = pd.concat([panel, risk], axis=1)
    train = np.array([True, True, True, False, True])
    spec = build_target("window_survival", full, train)
    broken = full.assign(**{EXIT: full[ENTRY] - 1.0})
    check(
        "target: `window_survival` alineado con el train, fuera de riesgo el corte tardío, y un evento "
        "sin tramo falla",
        len(spec.y) == 4 and spec.y["at_risk"].tolist() == [True, True, True, False]
        and spec.y["event"].tolist() == [1, 1, 0, 0]
        and _raises(lambda: build_target("window_survival", broken, train), ValueError),
    )

    # -- el apilado con exposición exacta --------------------------------------------------------
    model = WindowPEMStacker(horizon_days=30, bin_days=5, max_horizon_days=60)
    model.n_bins_train_ = 12
    entry = np.array([0.0, 7.5, 3.0, 0.0, 10.0])
    exit_ = np.array([12.0, 22.0, 3.0, 80.0, 40.0])
    event = np.array([1, 1, 0, 1, 0])
    at_risk = exit_ > entry
    st = model.stack(np.arange(5, dtype=float)[:, None], entry, exit_, event, at_risk)
    per_row = pd.Series(st["exposure"]).groupby(st["row"]).sum()
    events_at = st["bin"][st["event"] == 1].tolist()
    check(
        "PEM: la exposición apilada suma el tramo en riesgo (cortado en 60 d), sin bins antes de la "
        "entrada; el evento cae en su bin y el que pasa 60 d se censura",
        np.allclose(per_row.reindex([0, 1, 3, 4]).to_numpy(), [12.0, 14.5, 60.0, 30.0])
        and 2 not in per_row.index
        and st["bin"][st["row"] == 1].min() == 1 and events_at == [2, 4]
        and st["X"][:, -1].tolist() == (st["bin"] * 5.0).tolist(),
        f"exposición {per_row.round(2).to_dict()} · eventos en bins {events_at}",
    )

    # -- el PEM recupera tasas y el score es 1 − S(H) ------------------------------------------
    rng = np.random.default_rng(3)
    n = 2500
    x = rng.integers(0, 2, n).astype(float)
    rate = np.where(x == 1, 0.02, 0.005)
    life = rng.exponential(1 / rate)
    enter = rng.uniform(0, 20, n)
    kept = life > enter
    x, life, enter = x[kept], life[kept], enter[kept]
    cens = rng.uniform(30, 90, len(life))
    y = np.empty(len(life), dtype=[("entry", "f8"), ("exit", "f8"), ("event", "i1"), ("at_risk", "?"), ("group", "U2")])
    y["entry"], y["exit"], y["event"] = enter, np.minimum(life, cens), (life <= cens).astype(int)
    y["at_risk"], y["group"] = y["exit"] > y["entry"], "g"
    X = np.column_stack([x, rng.normal(size=len(x))])
    fitted = WindowPEMStacker(horizon_days=30, bin_days=5, max_horizon_days=60).fit(X, y)
    query = np.array([[0.0, 0.0], [1.0, 0.0]])
    rates = np.exp(fitted.log_rate(query, np.array([10.0, 10.0])))
    by_hand = 1 - np.exp(-sum(np.exp(fitted.log_rate(query, np.full(2, 5.0 * k))) * 5.0 for k in range(6)))
    check(
        "PEM: recupera la tasa por día con el offset de exposición (0,005 y 0,02) y el score es "
        "1 − exp(−Σ λ·5) sobre los 6 bins de 30 d",
        0.003 < rates[0] < 0.009 and 0.014 < rates[1] < 0.026
        and np.allclose(fitted.predict_proba(query)[:, 1], by_hand),
        f"tasas {rates.round(4).tolist()}",
    )

    # -- por el loop de CV ----------------------------------------------------------------------
    dummy = build_dummy_panel(SMALL_PANEL)
    tte = dummy["time_to_event_km"].to_numpy(float)
    observed = dummy["event_observed"].to_numpy(int) == 1
    duration = (tte - dummy["gap_km"].to_numpy(float)) / 60.0
    dummy[ENTRY] = 0.0
    dummy[EXIT] = np.where(observed, duration, 90.0)
    dummy[EVENT] = (observed & (duration > 0)).astype(int)
    dummy[CUT_DSS] = dummy["cut_odo"] / 60.0
    splits = make_splits(dummy, n_splits=3, seed=1)
    predictions, _ = run_cv(dummy, splits, model_name="window_survival_stacking",
                            model_params={"horizon_days": 30, "bin_days": 5, "max_horizon_days": 60},
                            target={"name": "window_survival", "params": {}})
    check(
        "PEM: `window_survival` + `window_survival_stacking` pasan por el loop de CV con scores en [0, 1]",
        bool(predictions["score"].between(0, 1).all()) and predictions["score"].nunique() > 10,
    )

    # -- la evaluación con las dos etiquetas ---------------------------------------------------
    preds = pd.DataFrame({"vehicle_id": ["A", "A", "B", "B"], "cut_odo": [1.0, 2.0, 1.0, 2.0],
                          "score": 0.5, "label": [0, 1, 0, 0], "event_observed": [1, 1, 0, 0]})
    marks = pd.DataFrame({"vehicle_id": ["A", "A", "B", "B"], "cut_odo": [1.0, 2.0, 1.0, 2.0],
                          EVALUABLE: [0, 1, 1, 0], CUT_DSS: [5.0, np.nan, 3.0, 9.0]})
    joined = attach_panel_columns(preds, marks, [EVALUABLE, CUT_DSS])
    masks = label_masks(joined, EVALUABLE)
    check(
        "evaluación: V se queda solo con las filas evaluables, D con todas, y una fila sin par en el panel falla",
        masks["hard"].all() and masks["corrected"].tolist() == [False, True, True, False]
        and _raises(lambda: attach_panel_columns(preds, marks.iloc[:3], [EVALUABLE]), ValueError),
    )
    floor = pd.DataFrame({"vehicle_id": ["A", "A", "B", "B", "C", "C"], "cut_odo": [1.0, 2.0] * 3,
                          "label": [0, 1, 0, 0, 0, 0], "event_observed": [1, 1, 0, 0, 0, 0],
                          "time_to_event_km": [2000.0, 1000.0, np.nan, np.nan, np.nan, np.nan],
                          CUT_DSS: [np.nan, np.nan, 50.0, 60.0, 10.0, 20.0]})
    floored = floor_metrics_for(floor, CUT_DSS, {"n_thresholds": 20, "k_consecutive": 1,
                                                  "max_false_alarms_per_1000": 1000})
    lifted = floor_metrics_for(floor.assign(**{CUT_DSS: floor[CUT_DSS].fillna(100.0)}), CUT_DSS,
                               {"n_thresholds": 20, "k_consecutive": 1, "max_false_alarms_per_1000": 1000})
    win = {"verdict": "gana"}
    both = {"detection": {"wins": True}, "vehicle_lift_mean": {"wins": True}}
    half = {"detection": {"wins": True}, "vehicle_lift_mean": {"wins": False}}
    results = {"corrected": {"vs_reference": win, "vs_floors": {"a": both, "b": half}}}
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        missing_a0 = read_a0(run_dir, 0.02)
        (run_dir / "audit.json").write_text(json.dumps({"audits": [
            {"audit": "(a0) features permutadas", "pr_auc": 0.13, "base_rate": 0.125}]}), encoding="utf-8")
        passed_a0 = read_a0(run_dir, 0.02)
    check(
        "evaluación: el piso manda los NaN abajo; se adopta solo si gana, les gana a todos los pisos "
        "en las dos métricas y pasa (a0)",
        np.isclose(floored["vehicle_lift_mean"][0], 1.0) and np.isclose(lifted["vehicle_lift_mean"][0], 3.0)
        and not verdict(results, "corrected", ["a", "b"], passed_a0)["adopted"]
        and verdict(results, "corrected", ["a"], passed_a0)["adopted"]
        and not verdict(results, "corrected", ["a"], missing_a0)["adopted"],
        f"lift del piso {floored['vehicle_lift_mean'][0]:.3f}",
    )


def _fold_of(splits: dict, repeat: int) -> dict[str, int]:
    return {str(v): int(f["fold"]) for f in splits["repeats"][repeat]["folds"] for v in f["valid_vehicles"]}


def extend_splits_checks() -> None:
    """11 · Extender folds: los del split base no se mueven y solo se reparten los nuevos."""
    base_panel = build_dummy_panel({**SMALL_PANEL, "seed": 13})
    base = make_splits(base_panel, n_splits=3, seed=13, min_valid_positives=1, n_repeats=2)
    vehicles = sorted(base_panel["vehicle_id"].astype(str).unique())
    extra = build_dummy_panel({**SMALL_PANEL, "seed": 14, "n_vehicles": 40})
    extra["vehicle_id"] = "NEW_" + extra["vehicle_id"].astype(str)
    panel = pd.concat([base_panel.loc[base_panel["vehicle_id"].astype(str).isin(vehicles[5:])], extra],
                      ignore_index=True)
    panel["grupo"] = "a"
    extended = extend_splits(base, panel, seed=13, min_valid_positives=1, guard_by="grupo")

    kept = set(vehicles[5:])
    check(
        "extend_splits: los vehículos del split base conservan su fold en cada repetición",
        all(all(_fold_of(extended, r)[v] == _fold_of(base, r)[v] for v in kept) for r in range(2)),
    )
    panel_vehicles = sorted(panel["vehicle_id"].astype(str).unique())
    n_folds = sum(len(masks) for _, masks in iter_repeats(panel, extended, strict=True, min_valid_positives=1))
    check(
        "extend_splits: cada vehículo del panel cae en un solo fold por repetición e iter_repeats lo acepta",
        all(sorted(_fold_of(extended, r)) == panel_vehicles for r in range(2)) and n_folds == 6,
    )
    positive = panel.groupby(panel["vehicle_id"].astype(str))["label"].max()
    balanced = True
    for r in range(2):
        folds = pd.Series(_fold_of(extended, r))
        base_folds = folds.loc[sorted(kept)]
        for stratum in (0, 1):
            before = base_folds[positive.loc[base_folds.index] == stratum].value_counts().reindex(range(3), fill_value=0)
            after = folds[positive.loc[folds.index] == stratum].value_counts().reindex(range(3), fill_value=0)
            balanced &= int(after.max() - after.min()) <= max(1, int(before.max() - before.min()))
    check(
        "extend_splits: los nuevos van, dentro de su estrato, al fold con menos vehículos de ese estrato",
        balanced and extended["extended_from"]["n_new"] == 40,
    )
    again = extend_splits(base, panel, seed=13, min_valid_positives=1)
    other = extend_splits(base, panel, seed=99, min_valid_positives=1)
    new = sorted(set(panel_vehicles) - kept)
    check(
        "extend_splits: la semilla solo reparte a los nuevos (misma semilla, mismo reparto)",
        all(_fold_of(again, r) == _fold_of(extended, r) for r in range(2))
        and any(_fold_of(other, r)[v] != _fold_of(extended, r)[v] for r in range(2) for v in new)
        and all(_fold_of(other, r)[v] == _fold_of(extended, r)[v] for r in range(2) for v in kept),
    )
    import hashlib
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        base_path = Path(tmp) / "splits_base.json"
        save_splits(base, base_path)
        stamped = extend_splits(base, panel, seed=13, min_valid_positives=1, base_path=base_path)
        digest = hashlib.sha256(base_path.read_bytes()).hexdigest()[:16]
    origin = stamped["extended_from"]
    check(
        "extend_splits: la procedencia (archivo base, su huella, conservados, nuevos y descartados) queda en el JSON",
        origin["file_sha256_16"] == digest and origin["n_kept"] == len(kept) and origin["n_dropped"] == 5
        and origin["dropped_vehicles"] == vehicles[:5] and origin["panel"] == base["panel"]
        and stamped["panel"] == panel_fingerprint(panel),
    )
    # Un grupo con filas solo de vehículos sanos: la guarda por grupo tiene que fallar aunque
    # la de filas pase.
    thin = panel.copy()
    healthy_rows = thin.index[thin["vehicle_id"].astype(str).map(positive).eq(0)][:10]
    thin.loc[healthy_rows, "grupo"] = "b"
    try:
        extend_splits(base, thin, seed=13, min_valid_positives=1, guard_by="grupo")
        guard_message = ""
    except ValueError as exc:
        guard_message = str(exc)
    check(
        "extend_splits: la guarda por grupo (por hito) falla si un fold queda sin positivos en un grupo",
        "grupo" in guard_message and "no pareada" in guard_message,
        guard_message[:80],
    )
    check(
        "extend_splits: pedir otros folds o repeticiones que los del base falla (no se inventan folds)",
        _raises(lambda: extend_splits(base, panel, seed=13, n_splits=5), ValueError)
        and _raises(lambda: extend_splits(base, panel, seed=13, n_repeats=3), ValueError),
    )


def _km_window_dummy() -> pd.DataFrame:
    """El panel dummy con el tramo en riesgo en km de K2: uno de cada cuatro cortes entra tarde.

    La entrada es de 1.200 km, más allá del primer bin de 500: una entrada dentro del primer bin
    apila igual que sin entrada, y un chequeo de "le llegó la entrada" no la podría ver.
    """
    from scripts.build_km_window_panel import ENTRY_KM, EVENT_KM, EXIT_KM

    dummy = build_dummy_panel(SMALL_PANEL)
    observed = dummy["event_observed"].to_numpy(int) == 1
    tte = dummy["time_to_event_km"].to_numpy(float) - dummy["gap_km"].to_numpy(float)
    dummy[ENTRY_KM] = np.where(np.arange(len(dummy)) % 4 == 0, 1200.0, 0.0)
    dummy[EXIT_KM] = np.where(observed, tte, 5000.0)
    dummy[EVENT_KM] = (observed & (tte >= dummy[ENTRY_KM])).astype(int)
    dummy[EXIT_KM] = np.where(observed & (tte < dummy[ENTRY_KM]), dummy[ENTRY_KM] - 1.0, dummy[EXIT_KM])
    return dummy


def f7_variance_checks() -> None:
    """16 · F7: los tres candidatos que bajan la varianza de K2 (monótono, columnas, hazard logístico, bagging)."""
    import numpy.lib.recfunctions as rfn

    from src.models.bagging import VehicleBaggingClassifier, vehicle_bootstrap_rows
    from src.models.survival_stacking import (DiscreteSurvivalStacker, _monotone_vector,
                                              _resolve_columns, natural_spline_basis)

    # -- el mapa monótono, por nombre ----------------------------------------------------------
    check(
        "F7 monótono: el signo cae en la columna que se nombra, 0 en las demás; una feature que no "
        "existe o un signo que no es ±1 fallan",
        _monotone_vector({"b": 1, "c": -1}, ["a", "b", "c"]) == [0, 1, -1]
        and _raises(lambda: _monotone_vector({"z": 1}, ["a", "b"]), KeyError)
        and _raises(lambda: _monotone_vector({"a": 2}, ["a", "b"]), ValueError),
    )

    # Hazard sintético que sube con `x` y no depende de `ruido`. Con +1 el riesgo predicho no
    # puede bajar al barrer `x`; con −1 no puede subir. Un mapa que invirtiera el signo, o que
    # lo pusiera en otra columna, rompe una de las dos.
    rng = np.random.default_rng(11)
    n = 600
    X = pd.DataFrame({"x": rng.normal(size=n), "ruido": rng.normal(size=n)})
    duration = rng.exponential(4000.0 / np.exp(0.8 * X["x"].to_numpy()))
    y = np.empty(n, dtype=[("duration_km", "f8"), ("event", "i1"), ("at_risk", "?"), ("group", "U4")])
    y["duration_km"] = np.minimum(duration, 5000.0)
    y["event"] = (duration < 5000.0).astype(int)
    y["at_risk"] = True
    y["group"] = [f"v{i}" for i in range(n)]
    sweep = pd.DataFrame({"x": np.linspace(-3, 3, 61), "ruido": 0.0})
    base = {"horizon_km": 1000, "bin_km": 500, "max_horizon_km": 5000,
            "model_params": {"n_estimators": 60, "min_child_samples": 20}}
    up = DiscreteSurvivalStacker(**base, monotone={"x": 1}).fit(X, y).predict_proba(sweep)[:, 1]
    down = DiscreteSurvivalStacker(**base, monotone={"x": -1}).fit(X, y).predict_proba(sweep)[:, 1]
    check(
        "F7 monótono: con +1 el riesgo no baja al subir la feature, con −1 no sube (y sin nombres falla)",
        bool(np.all(np.diff(up) >= -1e-12)) and bool(np.all(np.diff(down) <= 1e-12)) and up[-1] > up[0]
        and _raises(lambda: DiscreteSurvivalStacker(**base, monotone={"x": 1}).fit(X.to_numpy(), y), TypeError),
        f"Δ mínimo con +1 {np.diff(up).min():.2e} · Δ máximo con −1 {np.diff(down).max():.2e}",
    )

    # -- columnas por nombre -------------------------------------------------------------------
    names = ["feat_idle_frac", "feat_idle_frac_trend", "static_pais_A", "static_pais_B", "feat_otra"]
    check(
        "F7 columnas: el nombre exacto no arrastra a otra que empieza igual; la categórica entra con "
        "sus dummies; una columna que no existe falla",
        _resolve_columns(["feat_idle_frac"], names).tolist() == [0]
        and _resolve_columns(["static_pais", "feat_otra"], names).tolist() == [2, 3, 4]
        and _raises(lambda: _resolve_columns(["feat_no_existe"], names), KeyError),
    )
    Xc = X.assign(fuera=rng.normal(size=n))
    for backend in ("lightgbm", "logistic"):
        params = base if backend == "lightgbm" else {**base, "model_params": None}
        model = DiscreteSurvivalStacker(**params, backend=backend, columns=["x", "ruido"]).fit(Xc, y)
        moved = Xc.assign(fuera=Xc["fuera"] * 100.0 + 5.0)
        check(
            f"F7 columnas ({backend}): una columna fuera de la lista no mueve la predicción, y la "
            "importancia sigue alineada con el preprocesado (0 en la que no ve)",
            np.array_equal(model.predict_proba(Xc), model.predict_proba(moved))
            and len(model.feature_importances_) == Xc.shape[1] + 1
            and model.feature_importances_[2] == 0.0,
        )

    # -- el hazard logístico ---------------------------------------------------------------------
    knots = np.array([500.0, 2000.0, 4000.0, 6000.0, 9000.0])
    grid = np.arange(0.0, 20000.0, 500.0)
    basis = natural_spline_basis(grid, knots)
    tail = basis[grid >= 9000.0]
    check(
        "F7 spline natural: K nodos dan K − 1 columnas, la primera es lineal y la base es lineal "
        "después del último nodo",
        basis.shape == (len(grid), len(knots) - 1)
        and np.allclose(basis[:, 0], grid / 1000.0)
        and np.allclose(np.diff(tail, n=2, axis=0), 0.0, atol=1e-9),
    )
    logistic = DiscreteSurvivalStacker(**{**base, "model_params": None}, backend="logistic").fit(X, y)
    risk = logistic.predict_proba(sweep)[:, 1]
    check(
        "F7 hazard logístico: aprende el signo del hazard sintético y da riesgos en (0, 1)",
        bool(np.all(np.diff(risk) > 0)) and risk.min() > 0 and risk.max() < 1
        and logistic.booster_.coef_.ravel()[0] > 0,
    )

    # -- por el loop de CV, con la ventana en km de K2 ---------------------------------------------
    dummy = _km_window_dummy()
    splits = make_splits(dummy, n_splits=3, seed=1)
    km = {"horizon_km": 3000, "bin_km": 500, "max_horizon_km": 6000}
    window = {"name": "window_km_survival", "params": {}}
    runs = {
        "P2 monótono": ("survival_stacking", {**km, "monotone": {"feat_idle_per_1000km": 1, "feat_speed_kmh_mean": -1}}),
        "P3 logístico": ("survival_stacking", {**km, "backend": "logistic", "columns": [
            "feat_idle_per_1000km", "feat_trips_below_regime_temp_frac", "feat_speed_kmh_mean",
            "feat_coolant_temp_end_mean", "feat_km_per_day", "static_SalesCountry_cd"]}),
        "P1 embolsado": ("vehicle_bagging", {"base_model": "survival_stacking", "base_params": km, "n_bags": 2}),
    }
    for label, (name, params) in runs.items():
        predictions, _ = run_cv(dummy, splits, model_name=name, model_params=params, target=window)
        check(
            f"F7 {label}: pasa por el loop de CV con la ventana en km y scores en [0, 1]",
            bool(predictions["score"].between(0, 1).all()) and predictions["score"].nunique() > 10,
        )
    plain, _ = run_cv(dummy, splits, model_name="survival_stacking", model_params=km, target=window)
    again, _ = run_cv(dummy, splits, model_name="survival_stacking", model_params={**km, "monotone": {}},
                      target=window)
    check(
        "F7: sin `monotone` ni `columns` el modelo recibe la matriz de siempre y predice lo mismo",
        np.array_equal(plain["score"].to_numpy(), again["score"].to_numpy()),
    )

    # -- P1: cada bolsa es un bootstrap de autos y le llega el target entero (con la entrada) -----
    full = build_target("window_km_survival", dummy, np.ones(len(dummy), dtype=bool)).y
    Xd = dummy[["feat_idle_per_1000km", "feat_speed_kmh_mean"]].to_numpy(float)
    bag = VehicleBaggingClassifier(base_model="survival_stacking", base_params=km, n_bags=1,
                                   random_state=5).fit(Xd, full)
    rows, _ = vehicle_bootstrap_rows(full["group"].astype(str), np.random.default_rng(5))
    alone = DiscreteSurvivalStacker(**km).fit(Xd[rows], full[rows])
    no_entry = DiscreteSurvivalStacker(**km).fit(
        Xd[rows], rfn.drop_fields(full[rows], "entry_km", usemask=False))
    check(
        "F7 P1: la bolsa entrena con los autos sorteados enteros y con su entrada tardía (el apilado "
        "coincide con el de esas filas, y no con el de las mismas filas sin entrada)",
        bag.estimators_[0].stacking_ == alone.stacking_
        and bag.estimators_[0].stacking_["rows_stacked"] != no_entry.stacking_["rows_stacked"],
        f"{bag.estimators_[0].stacking_['rows_stacked']} apiladas contra {alone.stacking_['rows_stacked']}",
    )


def bagging_ensemble_checks() -> None:
    """E1/E2: el bagging sortea autos enteros y el ensamble por rango solo junta lo comparable."""
    from scripts.ensemble_rank import align_members, compare, rank_ensemble
    from src.models.bagging import VehicleBaggingClassifier, vehicle_bootstrap_rows

    groups = np.array(["a"] * 3 + ["b"] + ["c"] * 5 + ["d"] * 2)
    sizes = pd.Series(groups).value_counts()
    rows, n_distinct = vehicle_bootstrap_rows(groups, np.random.default_rng(3))
    drawn = pd.Series(groups[rows]).value_counts()
    check(
        "bagging: el bootstrap sortea autos enteros, tantos como autos hay",
        bool((drawn % sizes.loc[drawn.index] == 0).all())
        and int((drawn / sizes.loc[drawn.index]).sum()) == len(sizes)
        and n_distinct == len(drawn),
        f"{dict(drawn)} sobre tamaños {dict(sizes)}",
    )
    again, _ = vehicle_bootstrap_rows(groups, np.random.default_rng(3))
    check("bagging: el sorteo es determinista con la semilla", bool(np.array_equal(rows, again)))
    X_small = np.random.default_rng(0).normal(size=(len(groups), 2))
    check(
        "bagging: sin el vehículo en el `y` no entrena (no remuestrea filas en silencio)",
        _raises(lambda: VehicleBaggingClassifier(base_model="baserate").fit(X_small, np.arange(len(groups)) % 2),
                TypeError),
    )

    panel = build_dummy_panel({**SMALL_PANEL, "seed": 13})
    train_mask = np.zeros(len(panel), dtype=bool)
    train_mask[: len(panel) // 2] = True
    spec = build_target("grouped_label", panel, train_mask)
    check(
        "grouped_label: la etiqueta es `label` tal cual y el vehículo viaja al lado",
        bool((spec.y["label"] == panel.loc[train_mask, "label"].to_numpy()).all()
             and (spec.y["group"] == panel.loc[train_mask, "vehicle_id"].astype(str).to_numpy()).all()),
    )
    splits = make_splits(panel, n_splits=3, seed=13, min_valid_positives=1)
    bagged, _ = run_cv(panel, splits, model_name="vehicle_bagging",
                       model_params={"base_model": "baserate", "n_bags": 3},
                       target={"name": "grouped_label"})
    check(
        "bagging: por el loop de CV, con `grouped_label`, deja una predicción OOF por fila",
        bool(bagged["score"].notna().all() and bagged["score"].between(0, 1).all()),
    )
    survival, _ = run_cv(panel, splits, model_name="vehicle_bagging",
                         model_params={"base_model": "survival_stacking", "n_bags": 2,
                                       "base_params": {"horizon_km": 3000, "bin_km": 500}},
                         target={"name": "discrete_survival"})
    check(
        "bagging: envuelve a survival stacking con su target estructurado",
        bool(survival["score"].notna().all() and survival["score"].between(0, 1).all()),
    )

    rng = np.random.default_rng(17)
    n = 40
    frame = pd.DataFrame({
        "vehicle_id": [f"v{i // 4}" for i in range(n)],
        "cut_odo": np.tile([500.0, 1000.0, 1500.0, 2000.0], n // 4),
        "label": rng.integers(0, 2, n),
        "event_observed": rng.integers(0, 2, n),
        "fold_r0": np.repeat(np.arange(5), n // 5),
        "fold_r1": np.tile(np.repeat(np.arange(5), 4), 2),
        "score_r0": rng.random(n),
        "score_r1": rng.random(n),
    })
    same = rank_ensemble(align_members([frame, frame.copy()], ["a", "b"]), [0.5, 0.5])
    check(
        "ensamble: dos copias del mismo modelo dan su rango percentil",
        bool(np.allclose(same["score_r0"].to_numpy(),
                         frame.sort_values(["vehicle_id", "cut_odo"])["score_r0"].rank(pct=True).to_numpy())),
    )
    shuffled = frame.sample(frac=1.0, random_state=3)
    check(
        "ensamble: alinea por (vehicle_id, cut_odo), no por posición",
        bool(np.allclose(rank_ensemble(align_members([frame, shuffled], ["a", "b"]), [0.5, 0.5])["score_r1"],
                         same["score_r1"])),
    )
    other_folds = frame.assign(fold_r1=(frame["fold_r1"] + 1) % 5)
    check(
        "ensamble: con folds distintos no arma nada (no sería out-of-fold)",
        _raises(lambda: align_members([frame, other_folds], ["a", "b"]), ValueError),
    )
    better = {"detection": [0.20, 0.21, 0.20], "vehicle_lift_mean": [1.5, 1.5, 1.5], "lead_km": [1.0, 1.0, 1.0]}
    worse = {"detection": [0.10, 0.11, 0.10], "vehicle_lift_mean": [1.5, 1.5, 1.5], "lead_km": [1.0, 1.0, 1.0]}
    check(
        "ensamble: el veredicto sigue la regla del preregistro (desvío combinado)",
        compare(better, worse)["verdict"] == "gana" and compare(worse, better)["verdict"] == "pierde"
        and compare(better, better)["verdict"] == "empata",
    )


def dashboard_checks() -> None:
    """F4 · La alerta por vehículo del dashboard es la misma cuenta que la curva de anticipación."""
    from src.eval.dashboard_data import vehicle_alerts

    rng = np.random.default_rng(11)
    sizes = rng.integers(1, 12, 60)
    frame = pd.DataFrame({
        "vehicle_id": np.repeat([f"V{i:02d}" for i in range(60)], sizes),
        "cut_odo": np.concatenate([np.arange(s) * 500.0 for s in sizes]),
        "event_observed": np.repeat((np.arange(60) % 3 == 0).astype(int), sizes),
        "static_SalesCountry_cd": "CNTRY_4",
    })
    frame["time_to_event_km"] = np.where(frame["event_observed"].eq(1), 4000.0 - frame["cut_odo"], np.nan)
    frame["event_odo_km"] = frame["cut_odo"] + frame["time_to_event_km"]
    frame["score_r0"] = rng.random(len(frame)) + 0.3 * frame["event_observed"]
    frame = frame.sort_values(["vehicle_id", "cut_odo"]).reset_index(drop=True)
    point = operating_point(lead_time_curve(frame.assign(score=frame["score_r0"]), n_thresholds=50, k_consecutive=2),
                            max_false_alarms_per_1000=100)
    alerts = vehicle_alerts(frame, 0, float(point["threshold"]), 2)
    detected = alerts["outcome"].eq("Detectado")
    check(
        "dashboard: la alerta por vehículo reproduce detectados, falsas alarmas y anticipación de la curva",
        int(detected.sum()) == int(point["n_detected"])
        and int(alerts["outcome"].eq("Falsa alarma").sum()) == int(point["n_false_alarm_vehicles"])
        and abs(float(alerts.loc[detected, "lead_km"].median()) - float(point["median_lead_km"])) < 1e-9,
        f"{int(detected.sum())} vs {int(point['n_detected'])} detectados",
    )

    from src.eval.dashboard_data import expected_cost

    kw = dict(pi=0.05, insp=150.0, prev=300.0, fail=2000.0, e=0.8)
    none, everyone = expected_cost(np.array([0.0, 1.0]), np.array([0.0, 1.0]), **kw)
    # a mano: no alertar = π·falla; alertar a todos = π·(diag + prev + (1 − e)·falla) + (1 − π)·diag
    by_hand = (0.05 * 2000.0, 0.05 * (150.0 + 300.0 + 0.2 * 2000.0) + 0.95 * 150.0)
    check(
        "costos: el costo esperado de las dos políticas triviales coincide con la cuenta a mano",
        abs(none - by_hand[0]) < 1e-9 and abs(everyone - by_hand[1]) < 1e-9,
        f"{none:.2f}/{everyone:.2f} vs {by_hand[0]:.2f}/{by_hand[1]:.2f}",
    )


def _alert_frame(seed: int = 11, n: int = 60) -> pd.DataFrame:
    """Cortes sintéticos con el esquema de las predicciones (como `dashboard_checks`)."""
    rng = np.random.default_rng(seed)
    sizes = rng.integers(1, 12, n)
    frame = pd.DataFrame({
        "vehicle_id": np.repeat([f"V{i:02d}" for i in range(n)], sizes),
        "cut_odo": np.concatenate([np.arange(s) * 500.0 for s in sizes]),
        "event_observed": np.repeat((np.arange(n) % 3 == 0).astype(int), sizes),
        "static_SalesCountry_cd": "CNTRY_4",
    })
    frame["time_to_event_km"] = np.where(frame["event_observed"].eq(1), 6000.0 - frame["cut_odo"], np.nan)
    frame["event_odo_km"] = frame["cut_odo"] + frame["time_to_event_km"]
    frame["score_r0"] = rng.random(len(frame)) + 0.3 * frame["event_observed"]
    return frame.sort_values(["vehicle_id", "cut_odo"]).reset_index(drop=True)


def explain_checks() -> None:
    """17 · F4, explicabilidad de K2: lo que tiene que valer para que el porqué explique a K2 y solo con dev."""
    import copy

    from src.eval import explain as ex
    from src.eval.dashboard_data import operating_threshold, vehicle_alerts

    # -- la reconstrucción es run_cv, y V1 es aditivo ---------------------------------------------------
    dummy = _km_window_dummy()
    # Holdout sintético: uno de cada cinco vehículos es de test y no entra a nada de lo que sigue.
    vehicles = sorted(dummy["vehicle_id"].astype(str).unique())
    holdout = {"test_vehicles": vehicles[::5], "dev_vehicles": [v for i, v in enumerate(vehicles) if i % 5]}
    dev_mask, _ = test_split_masks(dummy, holdout)
    dev = dummy.loc[dev_mask].reset_index(drop=True)
    splits = make_splits(dev, n_splits=3, seed=5)
    km = {"horizon_km": 3000, "bin_km": 500, "max_horizon_km": 6000, "random_state": 42,
          "model_params": {"n_estimators": 60, "min_child_samples": 20}}
    target = {"name": "window_km_survival", "params": {}}
    predictions, _ = run_cv(dev, splits, model_name="survival_stacking", model_params=km, target=target)
    columns, models = ex.refit_folds(dev, splits, model_name="survival_stacking", model_params=km, target=target)
    repro = ex.reproduction_report(models, dev, predictions, atol=0.0)
    check(
        "explicabilidad: `refit_folds` reproduce `run_cv` bit a bit (mismos folds, columnas, target y Pipeline)",
        repro["exact"] and len(models) == 3, f"máx |Δ| = {repro['max_abs_diff']:.1e}",
    )
    m = models[0]
    X = dev.loc[m.valid_mask, columns]
    v1 = ex.tree_shap_hazard(m.pipeline, X, columns)
    check(
        "explicabilidad V1: Σφ + base = log-odds medio del hazard sobre los tramos de H (< 1e-6) y el riesgo "
        "que se recompone desde el booster es predict_proba",
        float(v1.additivity_error.max()) < 1e-6 and np.allclose(v1.risk, m.predict(X), atol=1e-12)
        and v1.units[-1] == "surv_bin_start_km",
        f"error máx {v1.additivity_error.max():.1e}",
    )
    check(
        "explicabilidad V1: las dummies del one-hot se suman en una sola unidad (la columna del panel)",
        v1.units.count("static_SalesCountry_cd") == 1 and not any(u.startswith("static_SalesCountry_cd_") for u in v1.units)
        and set(v1.units) - {"surv_bin_start_km"} == set(columns),
    )

    # -- la agregación por vehículo es la alerta de la curva ----------------------------------------------
    frame = _alert_frame()
    for k in (2, 3):
        point = operating_threshold(frame, 0, 100, {"k_consecutive": k, "n_thresholds": 50})
        threshold = float(point["threshold"])
        cuts = ex.explained_cuts(frame, 0, threshold, k)
        alerts = vehicle_alerts(frame, 0, threshold, k).set_index("vehicle_id")
        ok = int(cuts.groupby("vehicle_id")["alerted"].first().sum()) == int(point["n_detected"] + point["n_false_alarm_vehicles"])
        for vid, g in cuts.groupby("vehicle_id"):
            seq = frame.loc[frame["vehicle_id"].eq(vid)].sort_values("cut_odo")
            if bool(alerts.loc[vid, "alerted"]):
                start = int(np.flatnonzero(seq["cut_odo"].to_numpy() == g["cut_odo"].iloc[0])[0])
                run = seq.iloc[start:start + k]
                ok &= (len(g) == k and g["cut_odo"].iloc[0] == alerts.loc[vid, "alert_cut_odo"]
                       and np.array_equal(run["cut_odo"].to_numpy(), g["cut_odo"].to_numpy())
                       and bool((run["score_r0"] >= threshold).all()))
            else:
                ok &= len(g) == 1 and g["cut_odo"].iloc[0] == seq.loc[seq["score_r0"].idxmax(), "cut_odo"]
        check(
            f"explicabilidad: con k={k} los cortes explicados son la racha que disparó la alerta (mismo umbral y k "
            "que la curva) y, sin alerta, el de score máximo",
            bool(ok), f"{int(point['n_detected'] + point['n_false_alarm_vehicles'])} autos alertan",
        )

    # -- la referencia de flota sale solo de los sanos del train ---------------------------------------
    features = ["feat_idle_frac", "feat_speed_kmh_mean"]
    train, valid = dev.loc[m.train_mask], dev.loc[m.valid_mask]
    ref = ex.fleet_reference(train, valid, features)
    moved_valid = ex.fleet_reference(train, valid.assign(**{f: valid[f] + 100.0 for f in features}), features)
    failed_train = train.assign(**{f: np.where(train["event_observed"].eq(1), train[f] + 100.0, train[f]) for f in features})
    healthy_train = train.assign(**{f: np.where(train["event_observed"].eq(0), train[f] + 100.0, train[f]) for f in features})
    check(
        "explicabilidad: la mediana sana comparable no mira validación ni fallados del train, y sí a los sanos del train",
        np.allclose(ref, moved_valid, equal_nan=True)
        and np.allclose(ref, ex.fleet_reference(failed_train, valid, features), equal_nan=True)
        and not np.allclose(ref, ex.fleet_reference(healthy_train, valid, features), equal_nan=True),
    )

    # -- estabilidad: top k positivo y Jaccard --------------------------------------------------------------
    vector = pd.Series({"a": 0.5, "b": -0.9, "c": 0.2, "ctx": 3.0})
    check(
        "explicabilidad: el top k solo toma candidatas con contribución positiva; Jaccard(∅, ∅) = 1",
        ex.top_k(vector, ["a", "b", "c"], 3) == frozenset({"a", "c"}) and ex.jaccard(frozenset(), frozenset()) == 1.0
        and ex.jaccard(frozenset({"a"}), frozenset({"a", "c"})) == 0.5,
    )

    # -- el mensaje nunca nombra contexto ni síntomas -------------------------------------------------------
    specs = ex.feature_specs({"features": {
        "feat_idle_frac": {"class": "accionable", "sign": 1},
        "feat_speed_kmh_mean": {"class": "accionable", "sign": -1},
        "feat_trip_duration_median_min": {"class": "accionable", "sign": 0},
        "feat_dpf_end_mean": {"class": "sintoma"},
        "feat_cut_odo": {"class": "contexto"},
    }}, {})
    texts = copy.deepcopy(load_config("configs/explain_texts.yaml"))
    for name, tag in (("feat_dpf_end_mean", "FUGA_SINTOMA"), ("feat_cut_odo", "FUGA_CONTEXTO"),
                      ("feat_trip_duration_median_min", "FUGA_SIN_HIPOTESIS")):
        texts["features"][name] = {"label": tag, "format": "pct", "phrase": tag + " {value}", "recommendation": "ralenti"}
    contribution = pd.Series({"feat_dpf_end_mean": 5.0, "feat_cut_odo": 4.0, "feat_trip_duration_median_min": 3.0,
                              "feat_idle_frac": 0.5, "feat_speed_kmh_mean": 0.2})
    values = pd.Series({"feat_dpf_end_mean": 0.9, "feat_cut_odo": 9000.0, "feat_trip_duration_median_min": 40.0,
                        "feat_idle_frac": 0.6, "feat_speed_kmh_mean": 15.0})
    references = pd.Series({"feat_dpf_end_mean": 0.3, "feat_cut_odo": 5000.0, "feat_trip_duration_median_min": 20.0,
                            "feat_idle_frac": 0.2, "feat_speed_kmh_mean": 20.0})
    everything = set(contribution.index)
    factors = ex.message_factors(contribution, values=values, references=references, specs=specs, allowed=everything)
    forced = factors + [ex.MessageFactor(n, 5.0, 1.0, 0.0) for n in ("feat_dpf_end_mean", "feat_cut_odo",
                                                                      "feat_trip_duration_median_min")]
    text = ex.render_vehicle_message(risk_level="alto", factors=forced, specs=specs, texts=texts, k=2, gap_km=500.0,
                                     horizon_km=3000.0, km_per_day=50.0, symptom_contribution=5.0)
    # El mismo auto con el idle por DEBAJO de la mediana sana: la frase "más idle que los sanos" sería falsa.
    incoherent = ex.message_factors(contribution, values=values.where(values.index != "feat_idle_frac", 0.1),
                                    references=references, specs=specs, allowed=everything)
    check(
        "explicabilidad: `render_vehicle_message` nunca nombra contexto, síntomas ni accionables sin hipótesis, "
        "aunque se los pasen; `message_factors` solo elige accionables coherentes con la física",
        [f.feature for f in factors] == ["feat_idle_frac", "feat_speed_kmh_mean"] and "FUGA" not in text
        and "Arranques sin moverse" in text and texts["symptom_line"]["up"] in text
        and [f.feature for f in incoherent] == ["feat_speed_kmh_mean"],
    )
    empty = ex.render_vehicle_message(risk_level="alto", factors=[], specs=specs, texts=texts, k=2, gap_km=500.0,
                                      horizon_km=3000.0, km_per_day=50.0, symptom_contribution=-1.0)
    low = ex.render_vehicle_message(risk_level="bajo", factors=factors, specs=specs, texts=texts, k=2, gap_km=500.0,
                                    horizon_km=3000.0, km_per_day=50.0, symptom_contribution=1.0)
    check(
        "explicabilidad: sin factores el mensaje dice que el riesgo no se explica por hábitos (no rellena); con riesgo "
        "bajo no lista factores; el horizonte va en km y en semanas al ritmo del auto",
        texts["no_factors"] in empty and texts["factors_intro"] not in low and "Arranques sin moverse" not in low
        and "500 a 3.500 km" in empty and "entre 1 y 10 semanas" in empty,
    )

    # -- ningún output lleva un vehículo de test ------------------------------------------------------------
    rows = pd.concat([ex.tree_shap_hazard(fm.pipeline, dev.loc[fm.valid_mask, columns], columns).frame()
                      .assign(vehicle_id=dev.loc[fm.valid_mask, "vehicle_id"].to_numpy()) for fm in models])
    scored = dev[["vehicle_id", "cut_odo", "event_observed", "time_to_event_km", "static_SalesCountry_cd"]].assign(
        event_odo_km=lambda f: f["cut_odo"] + f["time_to_event_km"], score_r0=predictions["score"].to_numpy())
    cut_table = ex.explained_cuts(scored, 0, float(scored["score_r0"].quantile(0.9)), 2)
    guard = ex.dev_only_guard([rows["vehicle_id"], cut_table["vehicle_id"]], holdout)
    check(
        "explicabilidad: ningún output (filas explicadas, cortes por auto) nombra un vehículo de test, y la guarda "
        "falla si uno se cuela",
        guard["test_vehicles_found"] == [] and guard["vehicles_in_outputs"] == dev["vehicle_id"].nunique()
        and _raises(lambda: ex.dev_only_guard([rows["vehicle_id"], pd.Series([holdout["test_vehicles"][0]])], holdout),
                    RuntimeError)
        and _raises(lambda: ex.dev_only_guard([pd.Series(["VEH_FORASTERO"])], holdout), RuntimeError),
        f"{guard['vehicles_in_outputs']} autos de dev, {len(holdout['test_vehicles'])} de test fuera",
    )


def delivery_v2_checks() -> None:
    """Entrega v2 (26-09-2026): correcciones por parte del loader, estrato compuesto, ventana
    de producción del universo, sorteo sobre el universo y eventos repetidos."""
    import shutil
    import tempfile

    import yaml

    from src.config import repo_root
    from src.data.dedupe import dedupe_vehicles
    from src.data.loader import iter_table, load_table
    from src.data.usable import select_universe
    from src.eval.splits import (
        compare_holdouts,
        composite_strata,
        extend_holdout,
        holdout_on_universe,
        make_test_split,
        split_options,
        splits_match_options,
    )

    tmp = Path(tempfile.mkdtemp(prefix="_check_v2_", dir=repo_root() / "experiments"))
    try:
        # --- loader: rename + offsets + truncate_after -----------------------------------
        pd.DataFrame({"VehicleCode": ["A", "B"], "IdentificationDaysSinceProduction": [300, 250],
                      "daysUntilSale": [50, 40], "ProductionDay": [600, 540]}).to_csv(tmp / "sf.csv", index=False)
        pd.DataFrame({"VehicleCode": ["C"], "IdentificationDate": [np.nan], "daysUntilSale": [30],
                      "ProductionDay": [10]}).to_csv(tmp / "sn.csv", index=False)
        pd.DataFrame({"VehicleCode": ["A", "A", "B"],
                      "TripDatetimeStart": ["2026-09-10T10:00:00+00:00", "2026-09-20T10:00:00+00:00",
                                            "2026-09-11T10:00:00+00:00"],
                      "OdometerTripEnd": [10.0, 20.0, 5.0]}).to_csv(tmp / "tf.csv", index=False)
        pd.DataFrame({"VehicleCode": ["C"], "TripDatetimeStart": ["2026-09-12T10:00:00+00:00"],
                      "OdometerTripEnd": [7.0]}).to_csv(tmp / "tn.csv", index=False)
        sources = {
            "tables": {
                "vehicles": {"parts": {
                    "failed": {"path": str(tmp / "sf.csv"), "rename": {"IdentificationDaysSinceProduction": "IdentificationDate"},
                               "offsets": {"ProductionDay": -538}},
                    "not_failed": str(tmp / "sn.csv")},
                    "dtypes": {"VehicleCode": "str", "IdentificationDate": "float64", "ProductionDay": "float64"}},
                "trips": {"date_format": "ISO8601",
                          "parts": {"failed": str(tmp / "tf.csv"), "not_failed": str(tmp / "tn.csv")},
                          "truncate_after": {"column": "TripDatetimeStart", "at": "2026-09-14T00:00:00+00:00"},
                          "dtypes": {"VehicleCode": "str", "OdometerTripEnd": "float64"},
                          "date_columns": ["TripDatetimeStart"]},
            }
        }
        cfg_path = tmp / "sources.yaml"
        cfg_path.write_text(yaml.safe_dump(sources), encoding="utf-8")
        rel = cfg_path.relative_to(repo_root())
        veh = load_table("vehicles", rel)
        check(
            "loader v2: `rename` unifica la columna del evento y `offsets` corre ProductionDay solo en su parte",
            "IdentificationDate" in veh.columns and "IdentificationDaysSinceProduction" not in veh.columns
            and veh.set_index("VehicleCode").loc[["A", "B", "C"], "ProductionDay"].tolist() == [62.0, 2.0, 10.0],
        )
        trips = load_table("trips", rel)
        check(
            "loader v2: `truncate_after` saca lo posterior al fin de extracción común",
            sorted(trips["VehicleCode"]) == ["A", "B", "C"]
            and trips["TripDatetimeStart"].max() <= pd.Timestamp("2026-09-14", tz="UTC"),
            f"{len(trips)} filas",
        )
        partial = pd.concat(list(iter_table("trips", rel, columns=["VehicleCode", "OdometerTripEnd"])))
        check(
            "loader v2: una lectura parcial que no pide la fecha igual se recorta (y no la devuelve)",
            len(partial) == 3 and "TripDatetimeStart" not in partial.columns,
        )
        # --- dedupe: una fila por evento -> se queda el primero ------------------------
        dup = pd.DataFrame({"VehicleCode": ["A", "A", "B", "C"], "IdentificationDate": [400.0, 250.0, np.nan, 90.0],
                            "cohort": ["failed", "failed", "not_failed", "failed"]})
        clones_cfg = tmp / "dedupe.yaml"
        clones_cfg.write_text(yaml.safe_dump({"clone_groups": []}), encoding="utf-8")
        d = dedupe_vehicles(dup, config_path=clones_cfg.relative_to(repo_root())).set_index("VehicleCode")
        check(
            "dedupe v2: con varios eventos se queda el primero y `n_events_recorded` los cuenta",
            d.loc["A", "IdentificationDate"] == 250.0 and d.loc["A", "n_events_recorded"] == 2
            and d.loc["B", "n_events_recorded"] == 0 and d.loc["B", "cohort"] == "not_failed",
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # --- estrato compuesto -------------------------------------------------------------
    rng = np.random.default_rng(11)
    n = 600
    vehicles = pd.DataFrame({
        "vehicle_id": [f"V{i:04d}" for i in range(n)],
        "event_observed": (rng.random(n) < 0.3).astype(int),
        "static_SalesCountry_cd": rng.choice(["ARG", "BRA", "CHL", "COL", "PER"], n, p=[0.2, 0.3, 0.2, 0.25, 0.05]),
        "static_Engine": rng.choice(["ENG_1", "ENG_2", "ENG_3"], n, p=[0.15, 0.5, 0.35]),
        "static_ProductionDay": rng.integers(-150, 340, n),
        "event_day_since_production": np.nan,
        "static_daysUntilSale": 60.0,
    })
    keys = composite_strata(vehicles, ["event_observed", "static_SalesCountry_cd", "static_Engine"], min_size=5)
    counts = pd.Series(keys).value_counts()
    full_depth = [k for k in counts.index if k.count("|") == 2]
    check(
        "estrato compuesto: todo estrato completo tiene al menos `min_size` y los chicos se funden hacia arriba",
        all(counts[k] >= 5 for k in full_depth) and len(counts) < len(pd.Series(
            ["|".join(r) for r in vehicles[["event_observed", "static_SalesCountry_cd", "static_Engine"]].astype(str).to_numpy()]
        ).unique()),
        f"{len(counts)} estratos",
    )
    split = make_test_split(vehicles, test_size=0.2, seed=42,
                            stratify_columns=["event_observed", "static_SalesCountry_cd", "static_Engine"])
    by = pd.DataFrame({"k": keys, "t": vehicles["vehicle_id"].isin(split["test_vehicles"])}).groupby("k")["t"].agg(["size", "sum"])
    check(
        "holdout estratificado: en cada estrato el test es 1/5 ± 1 vehículo",
        bool(((by["sum"] - by["size"] / 5).abs() <= 1).all()) and split["stratify"]["min_stratum_size"] == 5,
    )
    plain = make_test_split(vehicles, test_size=0.2, seed=42)
    check(
        "holdout: sin `stratify_columns` no hay bloque de estratos y el test sigue siendo 1/5 (el camino de la entrega 1)",
        "stratify" not in plain and abs(len(plain["test_vehicles"]) - n / 5) <= 5,
    )

    # --- universo: ventana de producción + sorteo en una etapa ----------------------------
    vehicles["event_day_since_production"] = np.where(vehicles["event_observed"].eq(1), 200.0, np.nan)
    vehicles.loc[vehicles["event_observed"].eq(1) & vehicles["static_ProductionDay"].gt(190), "event_observed"] = 0
    vehicles.loc[vehicles["event_observed"].eq(0), "event_day_since_production"] = np.nan
    keep, universe = select_universe(vehicles, require_usable_event_date=True, keep_markets=None,
                                     production_day_window=[1, 193])
    inside = vehicles["static_ProductionDay"].between(1, 193)
    check(
        "universo: la ventana de producción saca exactamente a los de afuera y lo cuenta por cohorte",
        bool((keep == inside).all()) and universe["n_dropped_production"] == int((~inside).sum())
        and sum(universe["dropped_production_by_cohort"].values()) == int((~inside).sum()),
    )
    holdout = holdout_on_universe(vehicles, keep, seed=42,
                                  stratify_columns=["event_observed", "static_SalesCountry_cd", "static_Engine"],
                                  universe=universe)
    in_universe = set(vehicles.loc[keep, "vehicle_id"])
    check(
        "sorteo sobre el universo: dev ∪ test = universo, excluidos = el resto, sin solapamiento",
        set(holdout["dev_vehicles"]) | set(holdout["test_vehicles"]) == in_universe
        and set(holdout["excluded_vehicles"]) == set(vehicles["vehicle_id"]) - in_universe
        and not set(holdout["dev_vehicles"]) & set(holdout["test_vehicles"]),
    )
    panel_ok = pd.DataFrame({"vehicle_id": sorted(in_universe)})
    panel_bad = pd.DataFrame({"vehicle_id": [holdout["excluded_vehicles"][0]]})
    dev_mask, test_mask = test_split_masks(panel_ok, holdout)
    check(
        "sorteo sobre el universo: test_split_masks lo consume y grita si aparece un excluido",
        int(dev_mask.sum()) == len(holdout["dev_vehicles"]) and int(test_mask.sum()) == len(holdout["test_vehicles"])
        and _raises(lambda: test_split_masks(panel_bad, holdout), ValueError),
    )
    old = {"dev_vehicles": holdout["test_vehicles"][:3] + holdout["dev_vehicles"][:5],
           "test_vehicles": holdout["dev_vehicles"][5:7], "excluded_vehicles": []}
    cmp_ = compare_holdouts(holdout, old)
    check(
        "compare_holdouts: cuenta las transiciones y lista los autos del test nuevo que estaban en el dev viejo",
        cmp_["n_test_vehicles_in_old_dev"] == 3 and cmp_["transitions"].get("dev->test") == 3
        and cmp_["transitions"].get("test->dev") == 2,
    )

    # --- extensión del holdout a un universo más grande (sin la ventana) ------------------
    everyone = pd.Series(True, index=vehicles.index)
    newcomers = sorted(set(vehicles["vehicle_id"]) - in_universe)
    prev = {"dev_vehicles": newcomers[:2], "test_vehicles": newcomers[2:4]}
    extended = extend_holdout(holdout, vehicles, everyone, seed=42,
                              stratify_columns=["event_observed", "static_SalesCountry_cd", "static_Engine"],
                              keep_sides_of=prev)
    check(
        "extend_holdout: nadie de la base cambia de lado, los del holdout previo vuelven a su lado y "
        "restringir al universo de la base devuelve la base",
        set(holdout["dev_vehicles"]) <= set(extended["dev_vehicles"])
        and set(holdout["test_vehicles"]) <= set(extended["test_vehicles"])
        and set(newcomers[:2]) <= set(extended["dev_vehicles"]) and set(newcomers[2:4]) <= set(extended["test_vehicles"])
        and set(extended["dev_vehicles"]) | set(extended["test_vehicles"]) == set(vehicles["vehicle_id"])
        and sorted(set(extended["test_vehicles"]) & in_universe) == sorted(holdout["test_vehicles"])
        and extended["extended_from"]["n_forced_dev"] == 2 and extended["extended_from"]["n_forced_test"] == 2,
    )
    check(
        "extend_holdout: un universo que deja afuera a un auto de la base falla (achicar es restringir)",
        _raises(lambda: extend_holdout(holdout, vehicles, vehicles["vehicle_id"].ne(holdout["dev_vehicles"][0])),
                ValueError),
    )

    # --- folds con estrato compuesto ----------------------------------------------------
    folds_panel = build_dummy_panel({**SMALL_PANEL, "seed": 3})
    vid = folds_panel["vehicle_id"].astype(str)
    folds_panel["aux_static_Engine"] = np.where(vid.str[-1].isin(list("0123")), "ENG_1", "ENG_2")
    base = make_splits(folds_panel, n_splits=3, seed=5, min_valid_positives=1)
    same = make_splits(folds_panel, n_splits=3, seed=5, min_valid_positives=1, stratify_extra_columns=())
    comp = make_splits(folds_panel, n_splits=3, seed=5, min_valid_positives=1,
                       stratify_extra_columns=["aux_static_Engine"], min_stratum_size=3)
    check(
        "folds: sin columnas extra el reparto es el de siempre, bit a bit",
        [f["valid_vehicles"] for f in base["folds"]] == [f["valid_vehicles"] for f in same["folds"]]
        and "extra_columns" not in base["stratify"],
    )
    check(
        "folds: con estrato compuesto quedan declaradas las columnas y cada vehículo en un solo fold",
        comp["stratify"]["extra_columns"] == ["aux_static_Engine"]
        and sorted(v for f in comp["folds"] for v in f["valid_vehicles"]) == sorted(vid.unique()),
    )
    opts_old = split_options({"splits": {"n_splits": 5}})
    opts_new = split_options({"splits": {"stratify": {"extra_columns": ["aux_static_Engine"], "min_stratum_size": 3}}})
    check(
        "folds: un YAML sin `extra_columns` arma los kwargs de siempre, y uno con ellas choca con un archivo sin",
        "stratify_extra_columns" not in opts_old
        and bool(splits_match_options(base, opts_new)) and not splits_match_options(comp, {**opts_new, "n_splits": 3, "seed": 5}),
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
    landmark_metric_checks()
    extend_splits_checks()
    bagging_ensemble_checks()
    external_incidence_checks()
    window_survival_checks()
    km_window_detection_checks()
    f7_variance_checks()
    dashboard_checks()
    explain_checks()
    delivery_v2_checks()

    failed =[name for name, ok, _ in _checks if not ok]
    print()
    if failed:
        print(f"{len(failed)} chequeo(s) fallaron: {failed}")
        return 1
    print(f"Todo verde ({len(_checks)} chequeos). El harness de F0 está sano.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
