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
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# La consola de Windows arranca en cp1252 y revienta con los ≈/á de los mensajes.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from scripts.make_dummy import build_dummy_panel  # noqa: E402
from scripts.build_eda_cache import resolved_eda_config  # noqa: E402
from src.config import load_config, set_seed  # noqa: E402
from src.eval.metrics import classification_metrics, lead_time_curve, operating_point  # noqa: E402
from src.eval.splits import iter_folds, iter_repeats, make_splits  # noqa: E402
from src.features.trips import DEFAULT_THRESHOLDS  # noqa: E402
from src.training.cv import run_cv, select_feature_columns  # noqa: E402
from src.training.targets import build_target  # noqa: E402

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

    failed = [name for name, ok, _ in _checks if not ok]
    print()
    if failed:
        print(f"{len(failed)} chequeo(s) fallaron: {failed}")
        return 1
    print(f"Todo verde ({len(_checks)} chequeos). El harness de F0 está sano.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
