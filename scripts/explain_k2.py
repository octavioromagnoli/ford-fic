#!/usr/bin/env python
"""Explicabilidad de K2 con SHAP, de punta a punta (preregistro: configs/explain_k2.yaml).

    FORD_DATA_DIR=$PWD/data/rebuild-0921 WANDB_MODE=disabled \
      python scripts/explain_k2.py --config configs/explain_k2.yaml

**No es un candidato**: explica al finalista, no gasta presupuesto de comparaciones y no cambia
ninguna métrica oficial. Solo dev; el test no se toca (y al final se verifica que ningún vehículo
de test aparezca en ningún archivo).

1. **Reconstrucción (§3).** `train.py` no guarda los modelos: se reentrena cada fold de las 3
   repeticiones de `splits_r3.json` con `src/eval/explain.py::refit_folds` (el loop de `run_cv`) y
   se verifica que el score reproduzca `predictions.parquet` (tolerancia 1e-9, o ρ = 1 si LightGBM
   no fuera determinista). Si no, para.
2. **Seis repeticiones más** (`make_splits`, otra semilla), solo para las réplicas de V3/V4.
3. **V1** en todas las filas de dev y todas las repeticiones; **V3** y **V4** a nivel auto.
4. **La alerta de K2** al 5% por repetición (V y D), con la cuenta del dashboard; los cortes que
   se explican son los que la dispararon (o el de score máximo si el auto no alerta).
5. **V2** (Permutation) sobre la muestra preregistrada.
6. **Criterios** (§5): aditividad, ρ, borrado con su nulo, estabilidad, plausibilidad física (y
   su control de calendario), costo. Elección con la regla del YAML.
7. **Mensajes** al cliente (§6), **casos** por regla (§7), figuras y archivos en
   `experiments/explain-k2/`.

`--figures-only` rehace las figuras y `cases.md` desde los archivos ya guardados.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from scripts.train import load_or_make_splits, select_dev  # noqa: E402
from src.config import ensure_dir, load_config, resolve_path, set_seed  # noqa: E402
from src.eval import explain as ex  # noqa: E402
from src.eval.dashboard_data import operating_threshold  # noqa: E402
from src.eval.splits import load_test_split, make_splits, split_options  # noqa: E402

logger = logging.getLogger("explain_k2")

KEY = ex.KEY
MARKET = "static_SalesCountry_cd"
LABEL_ALIASES = {"corrected": "V", "hard": "D"}


# --------------------------------------------------------------------------------------------- #
# Carga
# --------------------------------------------------------------------------------------------- #
def load_inputs(cfg: dict) -> dict[str, Any]:
    """Panel de dev, splits congelados, predicciones y evaluación de la corrida de K2."""
    run_cfg = load_config(cfg["run_config"])
    set_seed(int(run_cfg.get("seed", 42)))
    panel_path = resolve_path(run_cfg["data"]["panel"])
    panel = select_dev(pd.read_parquet(panel_path), run_cfg)
    splits = load_or_make_splits(panel, run_cfg)
    run_dir = resolve_path(cfg["run_dir"])
    preds_path = run_dir / "predictions.parquet"
    if not preds_path.exists():
        raise FileNotFoundError(f"No hay predicciones de K2 en {preds_path}: `python scripts/train.py --config {cfg['run_config']}`")
    spec = features_spec_of(panel_path)
    return {
        "run_cfg": run_cfg, "panel": panel, "panel_path": panel_path, "splits": splits,
        "predictions": pd.read_parquet(preds_path),
        "window_eval": json.loads((run_dir / "window_eval.json").read_text(encoding="utf-8")),
        "features_v1": load_config(spec), "features_spec": spec,
        "test_split": load_test_split(run_cfg["splits"]["test_split"]),
    }


def features_spec_of(panel_path: Path) -> str:
    """El `features_spec` con que se armó el panel: del meta del derivado o, si no lo trae, del base.

    Los paneles derivados (`panel_survival_kmw`, …) dejan su propio meta sin el spec; el panel base
    del mismo build (`panel_meta.json`) sí lo trae. Es el mismo orden que `train.py::panel_build`.
    """
    for meta_path in (panel_path.with_name(f"{panel_path.stem}_meta.json"), panel_path.with_name("panel_meta.json")):
        if meta_path.exists():
            spec = json.loads(meta_path.read_text(encoding="utf-8")).get("features_spec")
            if spec:
                return str(spec)
    raise KeyError(f"Ningún meta del build de {panel_path.name} dice qué features_spec lo armó: no hay familias para V4")


def build_rows(panel: pd.DataFrame, predictions: pd.DataFrame, extra_oof: dict[int, np.ndarray], evaluable: str,
               n_official: int) -> pd.DataFrame:
    """Una fila por corte de dev con lo que necesitan la alerta y los mensajes.

    Las repeticiones oficiales usan el score de `predictions.parquet` (el mismo que lee el
    dashboard); las extra, el score fuera de fold de sus modelos reentrenados.
    """
    keep = KEY + ["event_observed", "time_to_event_km", "label"] + [f"score_r{r}" for r in range(n_official)]
    rows = predictions[keep].merge(panel[KEY + [evaluable, MARKET, "cut_date", "feat_km_per_day"]], on=KEY,
                                   how="left", validate="one_to_one")
    rows["event_odo_km"] = rows["cut_odo"] + rows["time_to_event_km"]
    order = panel[KEY].reset_index(drop=True)
    for r, scores in extra_oof.items():
        extra = order.assign(**{f"score_r{r}": scores})
        rows = rows.merge(extra, on=KEY, how="left", validate="one_to_one")
    return rows.sort_values(KEY, kind="stable", ignore_index=True)


def rows_for_label(rows: pd.DataFrame, label: str, evaluable: str) -> pd.DataFrame:
    """D = todas las filas de dev; V = las que tienen el horizonte entero dentro de la ventana."""
    return rows if label == "hard" else rows.loc[rows[evaluable].eq(1)].reset_index(drop=True)


# --------------------------------------------------------------------------------------------- #
# V1 y referencias por fila
# --------------------------------------------------------------------------------------------- #
def v1_rows(models: list[ex.FoldModel], panel: pd.DataFrame, feature_columns: list[str]) -> tuple[pd.DataFrame, dict]:
    """V1 de cada fila de dev, con el modelo del fold que no la vio, en cada repetición."""
    parts, additivity, risk_gap, t0 = [], 0.0, 0.0, time.perf_counter()
    for m in models:
        X = panel.loc[m.valid_mask, feature_columns]
        e = ex.tree_shap_hazard(m.pipeline, X, feature_columns)
        additivity = max(additivity, float(e.additivity_error.max()))
        risk_gap = max(risk_gap, float(np.abs(e.risk - m.valid_score).max()))
        frame = e.frame(panel.index[m.valid_mask])
        frame["base"], frame["output"], frame["risk"] = e.base, e.output, e.risk
        frame["repeat"], frame["fold"] = m.repeat, m.fold
        frame[KEY] = panel.loc[m.valid_mask, KEY]
        parts.append(frame)
    seconds = time.perf_counter() - t0
    out = pd.concat(parts).rename_axis("row").reset_index()
    return out, {"additivity_max": additivity, "risk_vs_predict_max": risk_gap, "seconds": seconds,
                 "rows": int(len(out)), "units": [c for c in parts[0].columns if c not in (*KEY, "base", "output",
                                                                                      "risk", "repeat", "fold")]}


def fleet_rows(models: list[ex.FoldModel], panel: pd.DataFrame, features: list[str], cal: dict) -> pd.DataFrame:
    """Mediana sana comparable (mercado × mes, train del fold) de cada fila de dev, por repetición."""
    parts = []
    for m in models:
        ref = ex.fleet_reference(panel.loc[m.train_mask], panel.loc[m.valid_mask], features,
                                 month_column=cal["month_column"], min_ref_vehicles=int(cal["min_ref_vehicles"]),
                                 levels=cal["levels"])
        ref["repeat"] = m.repeat
        ref[KEY] = panel.loc[m.valid_mask, KEY]
        parts.append(ref)
    return pd.concat(parts).rename_axis("row").reset_index()


# --------------------------------------------------------------------------------------------- #
# Alertas y vectores por auto
# --------------------------------------------------------------------------------------------- #
def alert_cuts(rows: pd.DataFrame, labels: list[str], repeats: list[int], budget: float, eval_cfg: dict,
               evaluable: str) -> dict[tuple[str, int], dict]:
    """Por (etiqueta, repetición): el punto de operación y los cortes explicados de cada auto."""
    k = int(eval_cfg.get("k_consecutive", 2))
    out = {}
    for label in labels:
        lab = rows_for_label(rows, label, evaluable)
        for r in repeats:
            point = operating_threshold(lab, r, budget, eval_cfg)
            if point is None:
                raise RuntimeError(f"Ningún umbral respeta {budget}/1000 con {label} en la repetición {r}")
            out[(label, r)] = {"point": point, "cuts": ex.explained_cuts(lab, r, float(point["threshold"]), k)}
    return out


def vehicle_frame(row_values: pd.DataFrame, cuts: pd.DataFrame, columns: list[str], rows: pd.DataFrame,
                  repeat: int) -> pd.DataFrame:
    """Promedio sobre los cortes explicados + el estado del auto (alerta, resultado, riesgo)."""
    means = ex.vehicle_means(row_values, cuts, columns)
    scored = cuts[KEY].merge(rows[KEY + [f"score_r{repeat}", "feat_km_per_day"]], on=KEY, how="left")
    info = cuts.groupby("vehicle_id", sort=True).agg(
        alerted=("alerted", "first"), failed=("failed", "first"), outcome=("outcome", "first"),
        risk_level=("risk_level", "first"), threshold=("threshold", "first"), max_score=("max_score", "first"),
        n_cuts=("cut_odo", "size"), explained_cuts=("cut_odo", lambda s: ",".join(f"{v:.0f}" for v in s)))
    info["score"] = scored.groupby("vehicle_id")[f"score_r{repeat}"].mean()
    info["km_per_day"] = scored.groupby("vehicle_id")["feat_km_per_day"].mean()
    return info.join(means)


def mean_frames(frames: list[pd.DataFrame], columns: list[str]) -> pd.DataFrame:
    """Promedio elemento a elemento de varias tablas por auto (mismos autos en todas)."""
    index = frames[0].index
    for f in frames[1:]:
        if not f.index.equals(index):
            raise ValueError("Las réplicas no tienen los mismos autos: no se pueden promediar")
    return pd.DataFrame(np.mean([f[columns].to_numpy(dtype=float) for f in frames], axis=0), index=index,
                        columns=columns)


# --------------------------------------------------------------------------------------------- #
# V2
# --------------------------------------------------------------------------------------------- #
def v2_sample(alerts: dict, rows: pd.DataFrame, official: list[int], decides: str, evaluable: str,
              sample_cfg: dict, seed: int) -> tuple[dict[int, set], dict]:
    """Los cortes que explica V2, por repetición (preregistro, `variants.V2.sample`)."""
    labels = sorted({label for label, _ in alerts})
    alerted_any = sorted({v for (label, r), a in alerts.items() if r in official
                          for v in a["cuts"].loc[a["cuts"]["alerted"], "vehicle_id"]})
    alerted_decides = {v for (label, r), a in alerts.items() if r in official and label == decides
                       for v in a["cuts"].loc[a["cuts"]["alerted"], "vehicle_id"]}
    lab_rows = rows_for_label(rows, decides, evaluable)
    status = lab_rows.groupby("vehicle_id")["event_observed"].first()
    never = status.loc[~status.index.isin(alerted_decides)]
    rng = np.random.default_rng(seed)
    n_failed, n_healthy = int(sample_cfg["others"]["n_failed"]), int(sample_cfg["others"]["n_healthy"])
    failed = sorted(never.index[never.eq(1)])
    healthy = sorted(never.index[never.eq(0)])
    others = sorted(list(rng.choice(failed, size=min(n_failed, len(failed)), replace=False))
                    + list(rng.choice(healthy, size=min(n_healthy, len(healthy)), replace=False)))
    keys: dict[int, set] = {r: set() for r in official}
    for r in official:
        for label in labels:
            cuts = alerts[(label, r)]["cuts"]
            chosen = cuts.loc[cuts["vehicle_id"].isin(alerted_any)]
            keys[r].update(map(tuple, chosen[KEY].itertuples(index=False, name=None)))
        cuts = alerts[(decides, r)]["cuts"]
        keys[r].update(map(tuple, cuts.loc[cuts["vehicle_id"].isin(others), KEY].itertuples(index=False, name=None)))
    info = {"alerted_any": alerted_any, "others": [str(v) for v in others],
            "rows_by_repeat": {str(r): len(v) for r, v in keys.items()}}
    return keys, info


def v2_rows(models: list[ex.FoldModel], panel: pd.DataFrame, feature_columns: list[str], keys: dict[int, set],
            v2_cfg: dict, seed: int) -> tuple[pd.DataFrame, dict]:
    """V2 de las filas de la muestra, con el modelo del fold que no las vio y su fondo de sanos."""
    markets = sorted(panel[MARKET].astype(str).unique())
    panel_keys = list(panel[KEY].itertuples(index=False, name=None))
    parts, additivity, t0, n_rows = [], 0.0, time.perf_counter(), 0
    for m in models:
        wanted = np.array([m.valid_mask[i] and panel_keys[i] in keys.get(m.repeat, ()) for i in range(len(panel))])
        if not wanted.any():
            continue
        train = panel.loc[m.train_mask]
        for market, sub in panel.loc[wanted].groupby(MARKET, sort=True):
            code = markets.index(str(market))
            rng = np.random.default_rng([seed, m.repeat, m.fold, code])
            background = ex.healthy_background(train, str(market), min_vehicles=int(v2_cfg["background"]["min_vehicles"]),
                                               rng=rng)
            e = ex.permutation_shap_score(m.pipeline, sub[feature_columns], panel.loc[background, feature_columns],
                                          feature_columns, max_evals=v2_cfg.get("max_evals", "auto"),
                                          seed=int(seed + 7919 * m.repeat + 104729 * m.fold + code))
            additivity = max(additivity, float(e.additivity_error.max()))
            frame = e.frame(sub.index)
            frame["base"], frame["output"], frame["risk"] = e.base, e.output, e.risk
            frame["repeat"], frame["fold"], frame["n_background"] = m.repeat, m.fold, len(background)
            frame[KEY] = sub[KEY]
            parts.append(frame)
            n_rows += len(sub)
    seconds = time.perf_counter() - t0
    out = pd.concat(parts).rename_axis("row").reset_index()
    return out, {"additivity_max": additivity, "seconds": seconds, "rows": n_rows,
                 "seconds_per_row": seconds / max(n_rows, 1)}


# --------------------------------------------------------------------------------------------- #
# Criterios
# --------------------------------------------------------------------------------------------- #
def spearman_block(vehicles: list[pd.DataFrame], units: list[str]) -> dict:
    """ρ entre autos de Σφ + base contra el score de los cortes explicados (y sin la base)."""
    with_base = [ex.spearman(v[units].sum(axis=1) + v["base"], v["score"]) for v in vehicles]
    without = [ex.spearman(v[units].sum(axis=1), v["score"]) for v in vehicles]
    return {"mean": float(np.nanmean(with_base)), "by_replicate": with_base,
            "without_base_mean": float(np.nanmean(without)), "n_vehicles": int(len(vehicles[0]))}


def erasure_units(vehicle_vectors: dict[int, pd.DataFrame], alerts: dict, label: str, repeats: list[int],
                  models: dict[tuple[int, str], ex.FoldModel], panel: pd.DataFrame, refs: pd.DataFrame,
                  feature_columns: list[str], actionables: list[str], k: int, *, pooled: pd.DataFrame | None = None,
                  top_override: dict[str, list[str]] | None = None) -> list[ex.ErasureUnit]:
    """Unidades del borrado. Por repetición (V1/V2) o, con `pooled`, un auto con las 3 (V3/V4)."""
    index_of = {key: i for i, key in enumerate(panel[KEY].itertuples(index=False, name=None))}

    def block(vid: str, r: int) -> ex.ErasureBlock:
        cuts = alerts[(label, r)]["cuts"]
        mine = cuts.loc[cuts["vehicle_id"].eq(vid), KEY]
        idx = [index_of[t] for t in mine.itertuples(index=False, name=None)]
        ref = refs.loc[refs["repeat"].eq(r)].set_index(KEY).loc[list(mine.itertuples(index=False, name=None)),
                                                                 actionables]
        return ex.ErasureBlock(models[(r, vid)].predict, panel.loc[idx, feature_columns].reset_index(drop=True),
                               ref.reset_index(drop=True))

    units = []
    if pooled is None:
        for r in repeats:
            vec = vehicle_vectors[r]
            for vid in vec.index[vec["alerted"]]:
                top = sorted(ex.top_k(vec.loc[vid], actionables, k), key=lambda f: -vec.loc[vid, f])
                units.append(ex.ErasureUnit((vid, r), top, [block(vid, r)]))
    else:
        alerted = sorted({v for r in repeats for v in vehicle_vectors[r].index[vehicle_vectors[r]["alerted"]]})
        for vid in alerted:
            top = (top_override or {}).get(vid)
            if top is None:
                top = sorted(ex.top_k(pooled.loc[vid], actionables, k), key=lambda f: -pooled.loc[vid, f])
            units.append(ex.ErasureUnit((vid,), list(top), [block(vid, r) for r in repeats]))
    return units


def plausibility_for(values_by_repeat: dict[int, pd.DataFrame], phi_by_repeat: dict[int, pd.DataFrame],
                     phi_global: pd.DataFrame, values_global: pd.DataFrame, specs: dict, actionables: list[str],
                     min_abs_rho: float) -> pd.DataFrame:
    """Signo global (sobre `phi_global`) y por repetición de cada accionable, contra el físico."""
    by_repeat = pd.DataFrame({r: ex.association(values_by_repeat[r].loc[p.index], p, actionables)
                              for r, p in phi_by_repeat.items()}).T
    rho = ex.association(values_global, phi_global, actionables)
    return ex.plausibility_table(rho, by_repeat, specs, min_abs_rho=min_abs_rho)


# --------------------------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------------------------- #
def main() -> int:  # noqa: C901 - es el guion del preregistro, en orden
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--figures-only", action="store_true", help="Rehace figuras y cases.md desde los archivos")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(message)s")
    logger.setLevel(logging.INFO)

    cfg = load_config(args.config)
    out_dir = ensure_dir(resolve_path(cfg["output_dir"]))
    if args.figures_only:
        from scripts.explain_k2_report import write_report
        write_report(cfg, out_dir)
        return 0

    timings: dict[str, float] = {}
    t_start = time.perf_counter()
    inputs = load_inputs(cfg)
    run_cfg, panel, predictions = inputs["run_cfg"], inputs["panel"], inputs["predictions"]
    eval_cfg = run_cfg.get("eval", {})
    k = int(eval_cfg.get("k_consecutive", 2))
    evaluable = run_cfg["window_eval"]["evaluable_column"]
    options = split_options(run_cfg)
    alert_cfg, crit = cfg["alert"], cfg["criteria"]
    decides, labels = alert_cfg["decides"], list(alert_cfg["labels"])
    seed = int(cfg["seed"])

    # -- §3 · reconstrucción y verificación ------------------------------------------------------
    t0 = time.perf_counter()
    feature_columns, models = ex.refit_folds(
        panel, inputs["splits"], model_name=run_cfg["model"]["name"], model_params=run_cfg["model"].get("params", {}),
        target=run_cfg.get("target"), strict_splits=bool(run_cfg["splits"].get("strict", True)),
        min_valid_positives=options["min_valid_positives"])
    timings["refit_official"] = time.perf_counter() - t0
    repro = ex.reproduction_report(models, panel, predictions, atol=float(cfg["reproduction"]["atol"]))
    logger.info("§3 reproducción | %d folds | máx |Δ| = %.3g | modo: %s", len(models), repro["max_abs_diff"], repro["mode"])
    if not repro["passed"]:
        (out_dir / "reproduction_failed.json").write_text(json.dumps(repro, indent=2), encoding="utf-8")
        logger.error("El modelo reconstruido NO reproduce predictions.parquet: el SHAP explicaría otro modelo. Paro.")
        return 1
    official = sorted({m.repeat for m in models})

    # -- réplicas extra para V3/V4 ---------------------------------------------------------------
    extra_cfg = crit["stability"]["extra_repeats"]
    extra_splits = make_splits(panel, n_splits=options["n_splits"], seed=int(extra_cfg["seed"]),
                               stratify_column=options["stratify_column"], stratify_level=options["stratify_level"],
                               min_valid_positives=options["min_valid_positives"], n_repeats=int(extra_cfg["n_repeats"]))
    t0 = time.perf_counter()
    _, extra_models = ex.refit_folds(
        panel, extra_splits, model_name=run_cfg["model"]["name"], model_params=run_cfg["model"].get("params", {}),
        target=run_cfg.get("target"), strict_splits=True, min_valid_positives=options["min_valid_positives"],
        repeat_offset=len(official))
    timings["refit_extra"] = time.perf_counter() - t0
    extra = sorted({m.repeat for m in extra_models})
    extra_oof = ex.oof_scores(extra_models, len(panel))
    triplets = [official] + [extra[i:i + len(official)] for i in range(0, len(extra), len(official))]
    all_models = models + extra_models
    model_of = {(m.repeat, v): m for m in all_models for v in panel.loc[m.valid_mask, "vehicle_id"].unique()}

    # -- clasificación ---------------------------------------------------------------------------
    specs = ex.feature_specs(cfg, ex.feature_families(inputs["features_v1"]))
    actionables = ex.names_of(specs, ex.ACTIONABLE)
    symptoms, contexts = ex.names_of(specs, ex.SYMPTOM), ex.names_of(specs, ex.CONTEXT)

    # -- V1 por fila, todas las repeticiones -------------------------------------------------------
    v1, v1_info = v1_rows(all_models, panel, feature_columns)
    units = v1_info["units"]
    ex.check_coverage(units, specs)
    v1_official_seconds = v1_info["seconds"] * sum(m.repeat in official for m in all_models) / len(all_models)
    timings["v1_all_repeats"] = v1_info["seconds"]
    logger.info("V1 | %d filas | aditividad máx %.2e | riesgo reconstruido vs predict %.2e",
                v1_info["rows"], v1_info["additivity_max"], v1_info["risk_vs_predict_max"])

    # -- alertas y referencias ---------------------------------------------------------------------
    rows = build_rows(panel, predictions, extra_oof, evaluable, len(official))
    alerts = alert_cuts(rows, labels, official + extra, float(alert_cfg["budget_per_1000"]), eval_cfg, evaluable)
    alert_summary: dict[str, Any] = {}
    for label in labels:
        per = [{"repeat": r, **{key: alerts[(label, r)]["point"][key] for key in
                                ("threshold", "n_detected", "n_event_vehicles", "n_false_alarm_vehicles",
                                 "n_healthy_vehicles", "detection_rate", "false_alarms_per_1000")}} for r in official]
        expected = inputs["window_eval"]["results"][label]["candidate"]["n_detected"]
        got = [int(p["n_detected"]) for p in per]
        alert_summary[label] = {"per_repeat": per, "window_eval_n_detected": expected, "matches_window_eval": got == expected}
        logger.info("alerta %s al 5%% | detectados %s (window_eval: %s)", LABEL_ALIASES[label], got, expected)
        if got != expected:
            raise RuntimeError(f"La alerta con {label} no reproduce window_eval.json: {got} contra {expected}")

    cal = crit["calendar_control"]
    t0 = time.perf_counter()
    refs = fleet_rows(models, panel, actionables, cal)
    timings["fleet_reference"] = time.perf_counter() - t0

    # -- vectores por auto: V1 por repetición, V3 por triplete, V4 ---------------------------------
    value_cols = [f"val__{f}" for f in actionables]
    ref_cols = [f"ref__{f}" for f in actionables]
    row_extras = panel[KEY + actionables].rename(columns={f: f"val__{f}" for f in actionables})
    v1_vehicle: dict[tuple[str, int], pd.DataFrame] = {}
    for (label, r), a in alerts.items():
        frame = v1.loc[v1["repeat"].eq(r)].merge(row_extras, on=KEY, how="left")
        cols = units + ["base", "output"]
        if r in official:
            ref = refs.loc[refs["repeat"].eq(r), KEY + actionables].rename(columns={f: f"ref__{f}" for f in actionables})
            frame = frame.merge(ref, on=KEY, how="left")
            cols += value_cols + ref_cols
        v1_vehicle[(label, r)] = vehicle_frame(frame, a["cuts"], cols, rows, r)
    v3_vehicle = {(label, t): mean_frames([v1_vehicle[(label, r)] for r in trip], units + ["base", "output", "score"])
                  for label in labels for t, trip in enumerate(triplets)}
    v4_vehicle = {key: ex.family_sums(v[units], specs).join(v[["base", "output", "score"]]) for key, v in v3_vehicle.items()}
    families = [c for c in v4_vehicle[(decides, 0)].columns if c not in ("base", "output", "score")]
    actionable_families = sorted({specs[f].family for f in actionables})

    # -- V2 ------------------------------------------------------------------------------------
    keys, sample_info = v2_sample(alerts, rows, official, decides, evaluable, cfg["variants"]["V2"]["sample"], seed)
    v2, v2_info = v2_rows(models, panel, feature_columns, keys, cfg["variants"]["V2"], seed)
    v2_units = [u for u in units if u in v2.columns]
    timings["v2"] = v2_info["seconds"]
    logger.info("V2 | %d filas en %.0f s (%.2f s/fila) | aditividad máx %.2e", v2_info["rows"], v2_info["seconds"],
                v2_info["seconds_per_row"], v2_info["additivity_max"])
    v2_vehicle: dict[tuple[str, int], pd.DataFrame] = {}
    for label in labels:
        for r in official:
            cuts = alerts[(label, r)]["cuts"]
            have = v2.loc[v2["repeat"].eq(r), KEY]
            covered = cuts.merge(have, on=KEY, how="left", indicator=True)
            full = covered.groupby("vehicle_id")["_merge"].apply(lambda s: bool(s.eq("both").all()))
            sub = cuts.loc[cuts["vehicle_id"].isin(full.index[full])]
            if sub.empty:
                continue
            frame = v2.loc[v2["repeat"].eq(r)]
            v2_vehicle[(label, r)] = vehicle_frame(frame, sub, v2_units + ["base", "output"], rows, r)

    # -- criterios con la etiqueta que decide ------------------------------------------------------
    fid, stab_cfg = crit["fidelity"], crit["stability"]
    er_cfg = fid["erasure"]
    results: dict[str, dict] = {name: {} for name in ("V1", "V2", "V3", "V4")}
    v1_dec = {r: v1_vehicle[(decides, r)] for r in official}
    v2_dec = {r: v2_vehicle[(decides, r)] for r in official}
    v3_dec = [v3_vehicle[(decides, t)] for t in range(len(triplets))]
    v4_dec = [v4_vehicle[(decides, t)] for t in range(len(triplets))]

    # aditividad
    results["V1"]["additivity_max"] = v1_info["additivity_max"]
    results["V2"]["additivity_max"] = v2_info["additivity_max"]
    results["V3"]["additivity_max"] = float(max(np.abs(v[units].sum(axis=1) + v["base"] - v["output"]).max() for v in v3_dec))
    results["V4"]["additivity_max"] = float(max(np.abs(v[families].sum(axis=1) + v["base"] - v["output"]).max() for v in v4_dec))

    # ρ con el score oficial
    results["V1"]["spearman"] = spearman_block([v1_dec[r] for r in official], units)
    results["V2"]["spearman"] = spearman_block([v2_dec[r] for r in official], v2_units)
    results["V3"]["spearman"] = spearman_block([v3_dec[0]], units)
    results["V4"]["spearman"] = spearman_block([v4_dec[0]], families)

    # borrado
    t0 = time.perf_counter()
    common = dict(alerts=alerts, label=decides, repeats=official, models=model_of, panel=panel, refs=refs,
                  feature_columns=feature_columns, actionables=actionables, k=int(er_cfg["top_k"]))
    er_seed = seed + 1
    results["V1"]["erasure"] = ex.erasure_test(erasure_units(v1_dec, **common), actionables,
                                               n_null=int(er_cfg["n_null"]), seed=er_seed)
    results["V2"]["erasure"] = ex.erasure_test(erasure_units(v2_dec, **common), actionables,
                                               n_null=int(er_cfg["n_null"]), seed=er_seed)
    results["V3"]["erasure"] = ex.erasure_test(erasure_units(v1_dec, **common, pooled=v3_dec[0]), actionables,
                                               n_null=int(er_cfg["n_null"]), seed=er_seed)
    top_family = {}
    for vid in v4_dec[0].index:
        fam = v4_dec[0].loc[vid, actionable_families]
        fam = fam[fam > 0]
        top_family[vid] = ([f for f in actionables if specs[f].family == fam.idxmax()] if len(fam) else [])
    results["V4"]["erasure"] = ex.erasure_test(erasure_units(v1_dec, **common, pooled=v3_dec[0], top_override=top_family),
                                               actionables, n_null=int(er_cfg["n_null"]), seed=er_seed)
    results["V4"]["erasure"]["note"] = "informativa: se reemplazan todas las accionables de la familia de mayor contribución"
    timings["erasure"] = time.perf_counter() - t0

    # estabilidad: autos que alertan con la etiqueta que decide en alguna repetición oficial
    alerted_any = sorted({v for r in official for v in v1_dec[r].index[v1_dec[r]["alerted"]]})
    top_k = int(stab_cfg["top_k"])
    stab = {
        "V1": ex.stability_table([v1_dec[r][units] for r in official], alerted_any, actionables, top_k),
        "V2": ex.stability_table([v2_dec[r][v2_units] for r in official], alerted_any, actionables, top_k),
        "V3": ex.stability_table([v[units] for v in v3_dec], alerted_any, actionables, top_k),
        "V4": ex.stability_table([v[families] for v in v4_dec], alerted_any, actionable_families, 1),
    }
    for name, table in stab.items():
        results[name]["stability"] = {"jaccard": float(table["jaccard"].mean()), "jaccard_sd": float(table["jaccard"].std(ddof=0)),
                                      "sign_agreement": float(table["sign_agreement"].mean()), "n_vehicles": int(len(table)),
                                      "top_k": 1 if name == "V4" else top_k}
    # Informativo (no entra a la regla): V1 con las 9 repeticiones, todos los pares. Dice si el
    # Jaccard de V1 con 3 réplicas es representativo del de una repetición cualquiera.
    v1_all = ex.stability_table([v1_vehicle[(decides, r)][units] for r in official + extra], alerted_any, actionables, top_k)
    results["V1"]["stability"]["all_repeats_informative"] = {
        "jaccard": float(v1_all["jaccard"].mean()), "sign_agreement": float(v1_all["sign_agreement"].mean()),
        "n_repeats": len(official) + len(extra)}

    # plausibilidad física (y su control de calendario)
    min_rho = float(crit["plausibility"]["min_abs_rho"])
    values = panel[actionables]
    raw_by_repeat = {r: values for r in official}
    v1_off = v1.loc[v1["repeat"].isin(official)]
    phi_rep = {r: v1_off.loc[v1_off["repeat"].eq(r)].set_index("row")[actionables] for r in official}
    pooled_idx = v1_off["row"].to_numpy()
    plaus = {"V1": plausibility_for(raw_by_repeat, phi_rep, v1_off[actionables].set_axis(pooled_idx),
                                    values.loc[pooled_idx].set_axis(pooled_idx), specs, actionables, min_rho)}
    v3_rows = v1_off.groupby("row")[units + ["base", "output", "risk"]].mean()
    v3_rows = v3_rows.join(panel[KEY])
    plaus["V3"] = plausibility_for(raw_by_repeat, phi_rep, v3_rows[actionables], values.loc[v3_rows.index], specs,
                                   actionables, min_rho)
    phi2 = {r: v2.loc[v2["repeat"].eq(r)].set_index("row")[actionables] for r in official}
    v2_idx = v2["row"].to_numpy()
    plaus["V2"] = plausibility_for(raw_by_repeat, phi2, v2[actionables].set_axis(v2_idx), values.loc[v2_idx].set_axis(v2_idx),
                                   specs, actionables, min_rho)
    # Control de calendario: el mismo ρ con el valor desviado contra la flota sana del mismo
    # mercado × mes (la referencia de cada repetición sale del train de su fold).
    deviation = {r: (values.loc[refs.loc[refs["repeat"].eq(r), "row"]] -
                     refs.loc[refs["repeat"].eq(r)].set_index("row")[actionables]).sort_index() for r in official}
    dev_mean = sum(deviation[r] for r in official) / len(official)
    calendar = {
        "V1": plausibility_for(deviation, phi_rep, v1_off[actionables].set_axis(pooled_idx),
                               pd.concat([deviation[r].loc[phi_rep[r].index] for r in official]).set_axis(pooled_idx),
                               specs, actionables, min_rho),
        "V3": plausibility_for(deviation, phi_rep, v3_rows[actionables], dev_mean.loc[v3_rows.index], specs,
                               actionables, min_rho),
    }
    for name in ("V1", "V2", "V3"):
        table = plaus[name]
        signed = table.loc[table["expected_sign"] != 0]
        results[name]["plausibility"] = {"n_signed": int(len(signed)), "n_aligned": int(table["aligned"].sum()),
                                         "agreement": float((signed["observed_sign"] == signed["expected_sign"]).mean()),
                                         "aligned": sorted(table.index[table["aligned"]])}

    # costo de producción por auto explicado (segundos de explicación, sin reentrenar)
    rows_per_rep = len(panel)
    cuts_per_vehicle = float(np.mean([alerts[(decides, r)]["cuts"].groupby("vehicle_id").size().mean() for r in official]))
    per_row_v1 = v1_official_seconds / (len(official) * rows_per_rep)
    results["V1"]["cost"] = {"seconds_per_vehicle": per_row_v1 * cuts_per_vehicle, "models": 1}
    results["V3"]["cost"] = {"seconds_per_vehicle": per_row_v1 * cuts_per_vehicle * len(official), "models": len(official)}
    results["V4"]["cost"] = dict(results["V3"]["cost"])
    results["V2"]["cost"] = {"seconds_per_vehicle": v2_info["seconds_per_row"] * cuts_per_vehicle, "models": 1}

    # fidelidad y elección
    tol, rho_min, alpha = float(fid["additivity_tol"]), float(fid["spearman_min"]), float(er_cfg["alpha"])
    for name, res in results.items():
        res["passes"] = {"additivity": bool(res["additivity_max"] < tol), "spearman": bool(res["spearman"]["mean"] >= rho_min),
                         "erasure": bool(res["erasure"]["p_value"] < alpha)}
        res["passes_fidelity"] = all(res["passes"].values())
    eligible = list(cfg["choice"]["eligible"])
    passing = [n for n in eligible if results[n]["passes_fidelity"]]
    winner, reason = None, "ninguna variante elegible pasa la fidelidad: no se publica mensaje al cliente"
    if passing:
        best = max(results[n]["stability"]["jaccard"] for n in passing)
        tied = [n for n in passing if best - results[n]["stability"]["jaccard"] < float(stab_cfg["tie_tolerance"])]
        winner = min(tied, key=lambda n: results[n]["cost"]["seconds_per_vehicle"])
        reason = (f"pasan la fidelidad {passing}; Jaccard máximo {best:.3f}; a menos de "
                  f"{stab_cfg['tie_tolerance']} quedan {tied}; la más barata es {winner}")
    logger.info("Elección: %s (%s)", winner, reason)

    # discrepancia V1 / V2 en los autos alertados
    common_units = [u for u in v2_units if u in units]
    disc = []
    for r in official:
        a, b = v1_dec[r], v2_dec[r]
        for vid in a.index[a["alerted"]]:
            if vid in b.index:
                disc.append({"vehicle_id": vid, "repeat": r,
                             "rho": ex.spearman(a.loc[vid, common_units].to_numpy(float), b.loc[vid, common_units].to_numpy(float))})
    disc_df = pd.DataFrame(disc)
    discrepancy = {"median_rho": float(disc_df["rho"].median()), "min_rho": float(disc_df["rho"].min()),
                   "n_pairs": int(len(disc_df)), "threshold": float(cfg["choice"]["v1_v2_discrepancy_rho"])}
    discrepancy["finding"] = bool(discrepancy["median_rho"] < discrepancy["threshold"])

    # -- mensajes al cliente (ganadora) -----------------------------------------------------------------
    texts = load_config(cfg["message"]["texts"])
    msg_cfg = cfg["message"]
    horizon_km = float(run_cfg["model"]["params"]["horizon_km"])
    gap_km = float(panel["gap_km"].iloc[0])
    win_table = plaus.get(winner) if winner else None
    allowed = set(win_table.index[win_table["aligned"]]) if win_table is not None else set()
    messages = []
    if winner:
        for label in labels:
            for r in official:
                ctx = v1_vehicle[(label, r)]
                if winner == "V1":
                    contrib, reps = ctx, [v1_vehicle[(label, q)] for q in official]
                elif winner == "V2":
                    if (label, r) not in v2_vehicle:
                        continue
                    contrib, reps = v2_vehicle[(label, r)], [v2_vehicle.get((label, q)) for q in official]
                else:
                    contrib, reps = v3_vehicle[(label, 0)], [v3_vehicle[(label, t)] for t in range(len(triplets))]
                for vid in contrib.index:
                    if vid not in ctx.index:
                        continue
                    replicate_vectors = [rep.loc[vid, actionables] for rep in reps if rep is not None and vid in rep.index]
                    vals = pd.Series(ctx.loc[vid, value_cols].to_numpy(float), index=actionables)
                    refv = pd.Series(ctx.loc[vid, ref_cols].to_numpy(float), index=actionables)
                    risk_level = str(ctx.loc[vid, "risk_level"])
                    # Con riesgo bajo el mensaje no lista factores: se guardan vacíos para que coincidan.
                    factors = [] if risk_level == "bajo" else ex.message_factors(
                        contrib.loc[vid, actionables], values=vals, references=refv, specs=specs, allowed=allowed,
                        replicate_contributions=replicate_vectors, max_factors=int(msg_cfg["max_factors"]))
                    symptom = float(contrib.loc[vid, [s for s in symptoms if s in contrib.columns]].sum())
                    text = ex.render_vehicle_message(
                        risk_level=risk_level, factors=factors, specs=specs, texts=texts, k=k,
                        gap_km=gap_km, horizon_km=horizon_km, km_per_day=float(ctx.loc[vid, "km_per_day"]),
                        symptom_contribution=symptom, symptom_min=float(msg_cfg["symptom_line_min"]))
                    messages.append({"vehicle_id": vid, "label": label, "repeat": r, "variant": winner,
                                     "risk_level": ctx.loc[vid, "risk_level"], "alerted": bool(ctx.loc[vid, "alerted"]),
                                     "outcome": ctx.loc[vid, "outcome"], "n_factors": len(factors),
                                     "factors": json.dumps([f.__dict__ for f in factors], ensure_ascii=False),
                                     "symptom_contribution": symptom, "message": text})
    messages_df = pd.DataFrame(messages)

    # -- casos por regla ---------------------------------------------------------------------------------
    case_cfg = cfg["cases"]
    ctx0 = v1_vehicle[(case_cfg["label"], int(case_cfg["repeat"]))]
    cases = []
    for rule in case_cfg["rules"]:
        pool = ctx0.loc[ctx0["outcome"].eq(rule["outcome"])]
        pool = pool.sort_values("max_score", ascending=rule["order"] == "max_score_asc", kind="stable")
        order_text = "score máx. más bajo" if rule["order"] == "max_score_asc" else "score máx. más alto"
        for vid in pool.index[: int(rule["n"])]:
            cases.append({"vehicle_id": vid, "rule": f"{rule['outcome']} ({order_text})"})

    # -- archivos --------------------------------------------------------------------------------------
    timings["total"] = time.perf_counter() - t_start
    shap_rows = pd.concat([
        v1.assign(variant="V1", set=np.where(v1["repeat"].isin(official), "official", "extra")),
        v2.assign(variant="V2", set="official"),
        # V3 lleva además el valor crudo de cada feature (el color del beeswarm).
        v3_rows.join(panel[[u for u in units if u.startswith("feat_")]].add_prefix("val__")).reset_index()
        .assign(variant="V3", set="official", repeat=-1, fold=-1),
    ], ignore_index=True)
    refs_wide = refs.rename(columns={f: f"ref__{f}" for f in actionables}).drop(columns="row")
    shap_rows = shap_rows.merge(refs_wide, on=KEY + ["repeat"], how="left")
    shap_rows = shap_rows.rename(columns={u: f"phi__{u}" for u in units})

    vehicle_parts = []
    for (label, r), v in v1_vehicle.items():
        vehicle_parts.append(v.reset_index().assign(variant="V1", label=label, replicate=r,
                                                    set="official" if r in official else "extra"))
    for (label, r), v in v2_vehicle.items():
        vehicle_parts.append(v.reset_index().assign(variant="V2", label=label, replicate=r, set="official"))
    for (label, t), v in v3_vehicle.items():
        vehicle_parts.append(v.reset_index().assign(variant="V3", label=label, replicate=t, set="triplet"))
    for (label, t), v in v4_vehicle.items():
        vehicle_parts.append(v.reset_index().rename(columns={f: f"fam__{f}" for f in families})
                             .assign(variant="V4", label=label, replicate=t, set="triplet"))
    shap_vehicle = pd.concat(vehicle_parts, ignore_index=True).rename(columns={u: f"phi__{u}" for u in units})

    # Guarda: ningún vehículo de test en ningún archivo, y todos son de dev (falla si no).
    guard = ex.dev_only_guard(
        [shap_rows["vehicle_id"], shap_vehicle["vehicle_id"], messages_df.get("vehicle_id", pd.Series(dtype=str)),
         pd.Series([c["vehicle_id"] for c in cases], dtype=str)], inputs["test_split"])

    shap_rows.to_parquet(out_dir / cfg["outputs"]["shap_rows"], index=False)
    shap_vehicle.to_parquet(out_dir / cfg["outputs"]["shap_vehicle"], index=False)
    messages_df.to_parquet(out_dir / "messages.parquet", index=False)
    (out_dir / "splits_extra.json").write_text(json.dumps(extra_splits, indent=1), encoding="utf-8")

    def records(table: pd.DataFrame) -> list[dict]:
        return json.loads(table.reset_index().to_json(orient="records", force_ascii=False))

    report = {
        "config": args.config, "run": run_cfg["name"], "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "panel": str(inputs["panel_path"]), "features_spec": inputs["features_spec"],
        "not_a_candidate": "Explica a K2: no reentrena otra cosa, no gasta presupuesto de comparaciones y no cambia ninguna métrica oficial.",
        "reproduction": repro, "timings_s": timings,
        "alerts": alert_summary, "decides": decides, "budget_per_1000": float(alert_cfg["budget_per_1000"]), "k_consecutive": k,
        "extra_repeats": {"seed": int(extra_cfg["seed"]), "n_repeats": int(extra_cfg["n_repeats"]),
                          "triplets": triplets, "file": "splits_extra.json"},
        "classification": {"actionable": actionables, "symptom": symptoms, "context": contexts,
                           "families": {f: specs[f].family for f in actionables}},
        "v1": {key: v1_info[key] for key in ("additivity_max", "risk_vs_predict_max", "seconds", "rows")},
        "v2": {**v2_info, **sample_info},
        "variants": results,
        "stability_by_vehicle": {name: records(t.set_index("vehicle_id")) for name, t in stab.items()},
        "plausibility": {name: records(t) for name, t in plaus.items()},
        "calendar_control": {name: records(t) for name, t in calendar.items()},
        "discrepancy_v1_v2": discrepancy,
        "choice": {"winner": winner, "reason": reason, "eligible": eligible, "passing": passing,
                   "rule": cfg["choice"]["rule"]},
        "allowed_in_text": sorted(allowed),
        "cases": cases,
        "test_guard": guard,
        "horizon": {"gap_km": gap_km, "horizon_km": horizon_km},
    }
    (out_dir / cfg["outputs"]["eval"]).write_text(json.dumps(report, indent=2, ensure_ascii=False, default=float),
                                                 encoding="utf-8")
    logger.info("Outputs en %s", out_dir)

    from scripts.explain_k2_report import write_report
    write_report(cfg, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
