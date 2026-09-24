"""Datos del dashboard de F4 (K2): carga, curvas y la alerta por vehículo.

Sin Streamlit: la app (`scripts/dashboard_k2/`) cachea estas funciones y solo dibuja. Los
números oficiales salen de los JSON de las corridas (`window_eval.json`, `decision_layer.json`);
lo que depende de un vehículo o de un umbral se recalcula con `src/eval/metrics.py`, la misma
cuenta que `train.py`. Solo dev: `select_dev` recorta el panel antes de todo.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import load_config, resolve_path
from src.eval.metrics import _first_sustained_index, lead_time_curve, operating_point

KEY = ["vehicle_id", "cut_odo"]
LABEL_NAMES = {"corrected": "Corregida (V)", "hard": "Dura (D)"}


@dataclass
class K2Data:
    predictions: pd.DataFrame      # filas de dev, con score_r*, fold_r* y las columnas del panel
    n_repeats: int
    eval_cfg: dict[str, Any]
    evaluable_column: str
    run_name: str
    window_eval: dict[str, Any] | None
    decision: dict[str, Any] | None
    metrics: dict[str, Any] | None
    audit: dict[str, Any] | None


def _read_json(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def load_k2(config_path: str) -> K2Data:
    """Predicciones fuera de fold de K2 + las columnas del panel que el dashboard usa."""
    cfg = load_config(config_path)
    run_cfg = load_config(cfg["run_config"])
    evaluable = run_cfg["window_eval"]["evaluable_column"]
    run_dir = resolve_path(run_cfg.get("output_dir", "experiments")) / run_cfg["name"]
    preds_path = run_dir / "predictions.parquet"
    if not preds_path.exists():
        raise FileNotFoundError(
            f"No hay predicciones de K2 en {preds_path}. Corré `python scripts/train.py --config {cfg['run_config']}`."
        )
    preds = pd.read_parquet(preds_path)

    # Las predicciones ya son solo de dev (train.py recorta con `select_dev`); el merge es por
    # la izquierda, así que ninguna fila de test entra acá.
    panel = pd.read_parquet(resolve_path(run_cfg["data"]["panel"]))
    columns = [evaluable, "static_SalesCountry_cd", *cfg["profile_features"]]
    right = panel[KEY + [c for c in columns if c not in preds]]
    out = preds.merge(right, on=KEY, how="left", validate="one_to_one", indicator=True)
    if out["_merge"].ne("both").any():
        raise ValueError("Hay predicciones de K2 sin fila en el panel: el panel no es el de la corrida")
    out = out.drop(columns="_merge").sort_values(KEY, kind="stable", ignore_index=True)
    out["event_odo_km"] = out["cut_odo"] + out["time_to_event_km"]
    n_repeats = sum(c.startswith("score_r") for c in out.columns) or 1

    return K2Data(
        predictions=out,
        n_repeats=n_repeats,
        eval_cfg=run_cfg.get("eval", {}),
        evaluable_column=evaluable,
        run_name=run_cfg["name"],
        window_eval=_read_json(run_dir / "window_eval.json"),
        decision=_read_json(resolve_path(cfg["decision_layer"])),
        metrics=_read_json(run_dir / "metrics.json"),
        audit=_read_json(run_dir / "audit.json"),
    )


def rows_for(data: K2Data, label: str) -> pd.DataFrame:
    """D = todas las filas de dev; V = las que tienen el horizonte entero dentro de la ventana."""
    p = data.predictions
    if label == "hard":
        return p
    return p.loc[p[data.evaluable_column].eq(1)].reset_index(drop=True)


def repeat_curve(rows: pd.DataFrame, repeat: int, eval_cfg: dict[str, Any]) -> pd.DataFrame:
    """La curva de `train.py` para una repetición (misma grilla y misma regla de alerta)."""
    return lead_time_curve(rows.assign(score=rows[f"score_r{repeat}"]),
                           n_thresholds=int(eval_cfg.get("n_thresholds", 50)),
                           k_consecutive=int(eval_cfg.get("k_consecutive", 2)))


def vehicle_alerts(rows: pd.DataFrame, repeat: int, threshold: float, k: int) -> pd.DataFrame:
    """Una fila por vehículo: si alertó, en qué corte y con cuánta anticipación.

    Misma regla que `first_alert_lead_times`: la alerta es el primer corte que arranca una racha
    de `k` cortes con score ≥ umbral; una alerta posterior al evento no cuenta como detección.
    """
    records = []
    for vid, g in rows.groupby("vehicle_id", sort=False):
        g = g.sort_values("cut_odo")
        s = g[f"score_r{repeat}"].to_numpy(dtype=float)
        idx = _first_sustained_index(s >= threshold, k)
        failed = int(g["event_observed"].iloc[0]) == 1
        lead = float(g["time_to_event_km"].iloc[idx]) if idx is not None else float("nan")
        alerted = idx is not None and (not failed or (np.isfinite(lead) and lead > 0))
        records.append({
            "vehicle_id": vid,
            "market": g["static_SalesCountry_cd"].iloc[0],
            "failed": failed,
            "alerted": alerted,
            "alert_cut_odo": float(g["cut_odo"].iloc[idx]) if idx is not None else float("nan"),
            "lead_km": lead if failed and alerted else float("nan"),
            "event_odo_km": float(g["event_odo_km"].iloc[0]) if failed else float("nan"),
            "max_score": float(s.max()),
            "n_cuts": int(len(g)),
        })
    out = pd.DataFrame(records)
    out["outcome"] = np.select(
        [out["failed"] & out["alerted"], out["failed"], out["alerted"]],
        ["Detectado", "No detectado", "Falsa alarma"], default="Sano sin alerta")
    return out


def operating_threshold(rows: pd.DataFrame, repeat: int, budget: float, eval_cfg: dict[str, Any]) -> dict | None:
    """El punto de operación de la repetición al presupuesto dado (lo mismo que `train.py`)."""
    return operating_point(repeat_curve(rows, repeat, eval_cfg), max_false_alarms_per_1000=float(budget))


def fleet_profile(data: K2Data, feature: str, bin_km: float) -> pd.DataFrame:
    """Mediana y cuartiles de la feature entre los sanos de dev, por tramo de odómetro."""
    healthy = data.predictions.loc[data.predictions["event_observed"].eq(0), ["cut_odo", feature]].dropna()
    healthy = healthy.assign(bin=(healthy["cut_odo"] // bin_km) * bin_km + bin_km / 2)
    g = healthy.groupby("bin")[feature]
    out = pd.DataFrame({"median": g.median(), "p25": g.quantile(0.25), "p75": g.quantile(0.75), "n": g.size()})
    return out[out["n"] >= 5].reset_index().rename(columns={"bin": "cut_odo"})


def curve_summary(decision: dict[str, Any], label: str) -> pd.DataFrame:
    """Los puntos oficiales de la capa de decisión, en tabla (una fila por presupuesto)."""
    pts = pd.DataFrame(decision["labels"][label]["curve_points"])
    g = pts.groupby("budget_per_1000")
    out = pd.DataFrame({
        "detection": g["detection"].mean(),
        "detection_sd": g["detection"].std(ddof=0),
        "detection_min": g["detection"].min(),
        "detection_max": g["detection"].max(),
        "detected": g["n_detected"].apply(lambda x: " / ".join(map(str, x))),
        "n_event_vehicles": g["n_event_vehicles"].first(),
        "fa_realized_per_1000": g["fa_realized_per_1000"].mean(),
        "lead_km": g["median_lead_km"].mean(),
        "null": g["null_mean"].mean(),
        "null_p95": g["null_p95"].mean(),
    }).reset_index()
    out["excess"] = out["detection"] - out["null"]
    return out


def holdout_summary(decision: dict[str, Any], label: str) -> pd.DataFrame:
    return pd.DataFrame(decision["labels"][label]["holdout_summary"])
