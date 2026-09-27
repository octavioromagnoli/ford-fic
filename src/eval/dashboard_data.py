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
from typing import Any, Sequence

import numpy as np
import pandas as pd

from src.config import load_config, resolve_path
from src.eval.metrics import _first_sustained_index, lead_time_curve, operating_point

KEY = ["vehicle_id", "cut_odo"]
LABEL_NAMES = {"corrected": "Corregida (V)", "hard": "Dura (D)"}


@dataclass
class RunData:
    predictions: pd.DataFrame      # filas de dev, con score_r*, fold_r* y las columnas del panel
    n_repeats: int
    eval_cfg: dict[str, Any]
    evaluable_column: str
    run_name: str
    window_eval: dict[str, Any] | None
    decision: dict[str, Any] | None
    metrics: dict[str, Any] | None
    audit: dict[str, Any] | None


#: El dashboard de F4 la conoce con este nombre.
K2Data = RunData


def _read_json(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _panel_or_raise(path: Path, what: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"No existe {what} en {path}. ¿Está definida FORD_DATA_DIR en la terminal? En PowerShell: "
            "$env:FORD_DATA_DIR = \"$PWD\\data\\rebuild-0921\" · en bash: export FORD_DATA_DIR=$PWD/data/rebuild-0921. "
            "Si es Streamlit, cortá el servidor (Ctrl+C) y relanzalo en esa misma terminal."
        )
    return pd.read_parquet(path)


def load_run(run_config: str, *, decision_layer: str | Path | None = None, columns: Sequence[str] = (),
             columns_panel: str | Path | None = None) -> RunData:
    """Predicciones fuera de fold de una corrida + las columnas del panel que se piden.

    Las columnas salen del panel de la corrida; las que ese panel no trae, de `columns_panel`, que tiene que tener
    las mismas filas (un panel secuencial no trae el perfil de uso: lo trae el panel de agregados). Cada merge es 1:1
    y falla si alguna predicción no aparea.
    """
    run_cfg = load_config(run_config)
    evaluable = run_cfg["window_eval"]["evaluable_column"]
    run_dir = resolve_path(run_cfg.get("output_dir", "experiments")) / run_cfg["name"]
    preds_path = run_dir / "predictions.parquet"
    if not preds_path.exists():
        raise FileNotFoundError(
            f"No hay predicciones de {run_cfg['name']} en {preds_path}. Corré `python scripts/train.py --config {run_config}`."
        )
    out = pd.read_parquet(preds_path)

    # Las predicciones ya son solo de dev (train.py recorta con `select_dev`); el merge es por
    # la izquierda, así que ninguna fila de test entra acá.
    wanted = list(dict.fromkeys([evaluable, "static_SalesCountry_cd", *columns]))
    sources = [(resolve_path(run_cfg["data"]["panel"]), "el panel de la corrida")]
    if columns_panel is not None:
        sources.append((resolve_path(columns_panel), "el panel de las columnas del perfil"))
    for path, what in sources:
        missing = [c for c in wanted if c not in out]
        if not missing:
            break
        panel = _panel_or_raise(path, what)
        right = panel[KEY + [c for c in missing if c in panel]]
        out = out.merge(right, on=KEY, how="left", validate="one_to_one", indicator=True)
        if out["_merge"].ne("both").any():
            raise ValueError(f"Hay predicciones de {run_cfg['name']} sin fila en {path.name}: no tiene las filas de la corrida")
        out = out.drop(columns="_merge")
    missing = [c for c in wanted if c not in out]
    if missing:
        raise KeyError(f"Ningún panel trae {missing}")
    out = out.sort_values(KEY, kind="stable", ignore_index=True)
    out["event_odo_km"] = out["cut_odo"] + out["time_to_event_km"]
    n_repeats = sum(c.startswith("score_r") for c in out.columns) or 1

    return RunData(
        predictions=out,
        n_repeats=n_repeats,
        eval_cfg=run_cfg.get("eval", {}),
        evaluable_column=evaluable,
        run_name=run_cfg["name"],
        window_eval=_read_json(run_dir / "window_eval.json"),
        decision=_read_json(resolve_path(decision_layer)) if decision_layer else None,
        metrics=_read_json(run_dir / "metrics.json"),
        audit=_read_json(run_dir / "audit.json"),
    )


def load_k2(config_path: str) -> RunData:
    """Predicciones fuera de fold de K2 + las columnas del panel que el dashboard usa."""
    cfg = load_config(config_path)
    return load_run(cfg["run_config"], decision_layer=cfg["decision_layer"], columns=cfg["profile_features"])


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


# --- por qué (docs/memoria/f4-explicabilidad-k2.md) -----------------------------------------

@dataclass
class K2Explanations:
    """Lo que dejó `scripts/explain_k2.py`: el dashboard lo lee y no recalcula SHAP."""

    vehicle: pd.DataFrame          # shap_vehicle.parquet: un vector por (variante, etiqueta, réplica, auto)
    messages: pd.DataFrame         # messages.parquet: el mensaje al cliente por (auto, etiqueta, repetición)
    evaluation: dict[str, Any]     # explain_eval.json
    texts: dict[str, Any]          # configs/explain_texts.yaml
    classes: dict[str, str]        # unidad -> accionable | contexto | sintoma
    winner: str | None
    budget_per_1000: float


def load_explanations(config_path: str) -> K2Explanations | None:
    """Las explicaciones de K2, o None si todavía no se corrió `scripts/explain_k2.py`."""
    cfg = load_config(config_path)
    if not cfg.get("explain_config"):
        return None
    ecfg = load_config(cfg["explain_config"])
    out = resolve_path(ecfg["output_dir"])
    paths = {"vehicle": out / ecfg["outputs"]["shap_vehicle"], "messages": out / "messages.parquet",
             "evaluation": out / ecfg["outputs"]["eval"]}
    if not all(p.exists() for p in paths.values()):
        return None
    evaluation = json.loads(paths["evaluation"].read_text(encoding="utf-8"))
    cls = evaluation["classification"]
    classes = {**{u: "accionable" for u in cls["actionable"]}, **{u: "sintoma" for u in cls["symptom"]},
               **{u: "contexto" for u in cls["context"]}}
    return K2Explanations(vehicle=pd.read_parquet(paths["vehicle"]), messages=pd.read_parquet(paths["messages"]),
                          evaluation=evaluation, texts=load_config(ecfg["message"]["texts"]), classes=classes,
                          winner=evaluation["choice"]["winner"], budget_per_1000=float(evaluation["budget_per_1000"]))


def vehicle_why(expl: K2Explanations, vehicle_id: str, label: str, repeat: int, *, top: int) -> dict[str, Any] | None:
    """El waterfall y el mensaje de un auto con la variante elegida (misma cuenta que el informe).

    V3 promedia repeticiones, así que su vector no depende de `repeat`; el valor del auto, la mediana
    sana, el nivel de riesgo y el mensaje sí (salen de la repetición elegida).
    """
    from src.eval.explain import waterfall_steps

    if expl.winner is None:
        return None
    v = expl.vehicle
    replicate = 0 if expl.winner in ("V3", "V4") else repeat
    vec = v.loc[(v["variant"] == expl.winner) & (v["label"] == label) & (v["replicate"] == replicate)
                & (v["vehicle_id"] == vehicle_id)]
    ctx = v.loc[(v["variant"] == "V1") & (v["label"] == label) & (v["replicate"] == repeat) & (v["vehicle_id"] == vehicle_id)]
    msg = expl.messages.loc[(expl.messages["vehicle_id"] == vehicle_id) & (expl.messages["label"] == label)
                            & (expl.messages["repeat"] == repeat)]
    if vec.empty or ctx.empty:
        return None
    vec, ctx = vec.iloc[0], ctx.iloc[0]
    units = [c.removeprefix("phi__") for c in v.columns if c.startswith("phi__") and np.isfinite(vec[c])]
    vector = pd.Series({u: float(vec[f"phi__{u}"]) for u in units})
    shown = {f["feature"] for f in json.loads(msg["factors"].iloc[0])} if len(msg) else set()
    values = pd.Series({u: ctx.get(f"val__{u}", np.nan) for u in units}, dtype=float)
    references = pd.Series({u: ctx.get(f"ref__{u}", np.nan) for u in units}, dtype=float)
    steps = waterfall_steps(vector, expl.classes, top=top, allowed_in_message=shown, texts=expl.texts,
                            values=values, references=references)
    return {"steps": steps, "base": float(vec["base"]), "output": float(vec["output"]),
            "message": msg["message"].iloc[0] if len(msg) else None, "risk_level": ctx["risk_level"],
            "variant": expl.winner}


# --- costo esperado (docs/memoria/f8-costos-k2.md) ------------------------------------------

def curve_points(curve: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(FPR, TPR) de una curva, con las dos políticas triviales agregadas como extremos."""
    fpr = np.append(curve["false_alarms_per_1000"].to_numpy(dtype=float) / 1000.0, [0.0, 1.0])
    tpr = np.append(curve["detection_rate"].to_numpy(dtype=float), [0.0, 1.0])
    return fpr, tpr


def expected_cost(fpr: np.ndarray, tpr: np.ndarray, *, pi: float, insp: float, prev: float, fail: float,
                  e: float) -> np.ndarray:
    """Costo esperado por vehículo de la flota en cada punto (FPR, TPR).

    Falsa alarma = `insp`; detectado = `insp + prev + (1 − e)·fail`; no detectado = `fail`.
    Alertar a un auto que va a fallar ahorra `B = e·fail − prev − insp` contra no alertarlo.
    """
    benefit = e * fail - prev - insp
    return pi * fail - pi * np.asarray(tpr) * benefit + (1.0 - pi) * np.asarray(fpr) * insp


def null_and_k2_budget_points(decision: dict[str, Any], label: str) -> tuple[tuple, tuple]:
    """K2 y el nulo de tamaño de bolsa en los presupuestos de la capa de decisión (+ los triviales)."""
    pts = pd.DataFrame(decision["labels"][label]["curve_points"]).groupby("budget_per_1000").mean(numeric_only=True)
    budgets = pts.index.to_numpy(dtype=float) / 1000.0
    k2 = (np.r_[0.0, pts["fa_realized_per_1000"].to_numpy() / 1000.0, 1.0], np.r_[0.0, pts["detection"].to_numpy(), 1.0])
    null = (np.r_[0.0, budgets, 1.0], np.r_[0.0, pts["null_mean"].to_numpy(), 1.0])
    return k2, null


def cost_optimum(curves: list[tuple[np.ndarray, np.ndarray]], decision: dict[str, Any], label: str,
                 **costs: float) -> dict[str, Any]:
    """El punto de mínimo costo en cada repetición, las políticas triviales y el ahorro sobre el azar."""
    per_repeat = []
    for fpr, tpr in curves:
        cost = expected_cost(fpr, tpr, **costs)
        i = int(np.argmin(cost))
        per_repeat.append((fpr[i], tpr[i], cost[i]))
    a = np.array(per_repeat)
    none = float(expected_cost(np.array([0.0]), np.array([0.0]), **costs)[0])
    everyone = float(expected_cost(np.array([1.0]), np.array([1.0]), **costs)[0])
    best_trivial = min(none, everyone)
    k2_pts, null_pts = null_and_k2_budget_points(decision, label)
    return {
        "fa_opt": float(a[:, 0].mean()), "fa_opt_max": float(a[:, 0].max()), "det_opt": float(a[:, 1].mean()),
        "cost_model": float(a[:, 2].mean()), "cost_none": none, "cost_everyone": everyone,
        "trivial": "no alertar" if none <= everyone else "alertar a todos",
        "savings_vs_trivial": 1.0 - float(a[:, 2].mean()) / best_trivial if best_trivial > 0 else 0.0,
        "savings_per_1000": 1000.0 * (best_trivial - float(a[:, 2].mean())),
        "savings_over_null_per_1000": 1000.0 * float(expected_cost(*null_pts, **costs).min()
                                                     - expected_cost(*k2_pts, **costs).min()),
        "ratio_benefit_insp": (costs["e"] * costs["fail"] - costs["prev"] - costs["insp"]) / costs["insp"],
    }
