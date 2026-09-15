#!/usr/bin/env python
"""Smoke test de F0: verifica que el harness está sano antes de confiar en un número.

    python scripts/check_setup.py

Corre en segundos sobre un panel dummy chico y chequea las cuatro cosas que, si
fallan, invalidan todo lo que venga después:

1. El panel dummy cumple el contrato de datos (columnas, dtypes, prefijos).
2. Los splits no filtran vehículos entre train y validación y cubren el panel.
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
from src.config import set_seed  # noqa: E402
from src.eval.metrics import classification_metrics, lead_time_curve, operating_point  # noqa: E402
from src.eval.splits import iter_folds, make_splits  # noqa: E402
from src.training.cv import run_cv, select_feature_columns  # noqa: E402

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


def main() -> int:
    set_seed(7)
    panel = build_dummy_panel(SMALL_PANEL)

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
