#!/usr/bin/env python
"""Auditorías del cure model (preregistro §7) sobre una corrida de `scripts/train.py`.

    python scripts/train.py --config configs/exp_cure_p0_unitweight.yaml
    python scripts/audit_cure.py --config configs/exp_cure_p0_unitweight.yaml

Lee las predicciones OOF de `experiments/<name>/`, corre las auditorías que declara
`audit_cure.audits` en el YAML y deja `audit_cure.json` al lado. Con `wandb.enabled`
las loguea en una corrida `audit-<name>` del mismo grupo. Todo es dev: el panel se
recorta con `select_dev`, igual que al entrenar.

- **A0 · nulo global** (reentrena): se permutan las filas de features dentro de cada
  hito. D1 tiene que caer a ~0,5 y D2 al presupuesto; si no, hay fuga y no se lee nada
  más. Las tolerancias salen del YAML (`a0`).
- **C1 · nulo estratificado** (sin reentrenar): se permuta el score entre las filas del
  mismo hito × tercil de producción × mercado y se recalcula el D1 promedio. Da el
  p-valor que decide la adopción (y la parada después de P0).
- **C2 · piso de producción**: score = −`ProductionDay` y score = −fecha de venta, con D1
  y D2. Además, el ρ del score del modelo con `ProductionDay` y la AUC dentro de cada tercil.
- **C3 · piso de uso**: score = −(km en `(venta, venta + L]`)/L.
- **C6 · autos quietos**: D1 y D2 sin las filas que recorrieron menos de `immobile_km`
  hasta el hito (`event_clock.yaml`). Informativo.
- **A3 · calendario (b)** (reentrena): mes del hito, temperatura ambiente y `ProductionDay`
  como covariables. Marca si D1 o la AUC entre resueltos suben más que el umbral.
- **A5 · ablación de la normalización** (reentrena): los agregados crudos de la ventana
  en vez del desvío contra la flota.
- **A6 · el finalista**: su score del último corte con fecha ≤ la del hito, en la
  intersección de vehículos. Informativo (sus sanos están submuestreados).

Después, el **veredicto** contra `audit_cure.compare_to` (P1 contra P0) y la
**adopción** (preregistro §5): "le gana a X en M" exige que la diferencia de medias
entre repeticiones supere √(σ² + σ_X²) **y** que el IC95 del bootstrap pareado por
vehículo excluya el 0.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from scripts.train import load_or_make_splits, panel_meta, select_dev  # noqa: E402
from src.config import load_config, resolve_path, set_seed  # noqa: E402
from src.eval.metrics import (  # noqa: E402
    dispersion,
    landmark_bootstrap_draws,
    landmark_eval_config,
    landmark_metrics,
    late_entry_concordance,
    roc_auc,
)
from src.eval.splits import split_options  # noqa: E402
from src.training.cv import run_cv, select_feature_columns  # noqa: E402

logger = logging.getLogger("audit_cure")
AUDITS = ("A0", "C1", "C2", "C3", "C6", "A3", "A5", "A6")
AUDIT_SCORE = "audit_score"


# --------------------------------------------------------------------------- #
# Piezas comunes
# --------------------------------------------------------------------------- #
def repeat_columns(column: str, n_repeats: int) -> list[str]:
    return [column] if n_repeats <= 1 else [f"{column}_r{r}" for r in range(n_repeats)]


def with_score(predictions: pd.DataFrame, values: np.ndarray, n_repeats: int) -> pd.DataFrame:
    """Una copia de las predicciones con `audit_score` (y sus `_r{i}`) = `values` (un piso)."""
    out = predictions.copy()
    for column in {AUDIT_SCORE, *repeat_columns(AUDIT_SCORE, n_repeats)}:
        out[column] = np.asarray(values, dtype=float)
    return out


def measure(predictions: pd.DataFrame, cfg: dict, n_repeats: int, seed: int, meta: dict | None,
            **overrides: Any) -> dict[str, Any]:
    """`landmark_metrics` con el bloque `eval.landmark` del YAML y lo que se pise."""
    lm_cfg = {**(cfg["eval"].get("landmark") or {}), **overrides}
    block, _ = landmark_metrics(predictions, lm_cfg, n_repeats=n_repeats, seed=seed,
                                n_event_vehicles_total=((meta or {}).get("dev") or {}).get("event_vehicles"))
    return block


def headline(block: dict[str, Any]) -> dict[str, Any]:
    """Lo que se lee de un bloque: D1 y D2 (media ± desvío entre repeticiones) y la AUC entre resueltos."""
    auc = [v["roc_auc"]["mean"] for v in block["informative"]["discrimination_resolved"].values()]
    return {
        "d1": block["d1"]["mean"]["mean"], "d1_std": block["d1"]["mean"]["std"],
        "d1_by_landmark": {k: v["c_index"]["mean"] for k, v in block["d1"]["by_landmark"].items()},
        "d1_boot": block["d1"]["bootstrap_by_vehicle"],
        "d2": block["d2"]["detection_rate"]["mean"], "d2_std": block["d2"]["detection_rate"]["std"],
        "d2_boot": block["d2"]["bootstrap_by_vehicle"],
        "d2_alerts_allowed": block["d2"]["alerts_allowed"], "d2_n_negative": block["d2"]["n_negative"],
        "d2_n_positive": block["d2"]["n_positive"],
        "median_lead_days": block["d2"]["median_lead_days"]["mean"],
        "auc_resolved_mean": float(np.mean(auc)),
    }


def beats(a: tuple[pd.DataFrame, dict], b: tuple[pd.DataFrame, dict], block_a: dict, block_b: dict,
          *, n_repeats: int, n_boot: int, seed: int, alpha: float = 0.05) -> dict[str, Any]:
    """¿A le gana a B en D1 y en D2? La regla del preregistro §5, con sus dos condiciones.

    Los dos bootstrap usan la misma semilla y la misma lista de vehículos (mismo panel),
    así que reciben los mismos remuestreos: la resta es el bootstrap pareado.
    """
    draws_a = landmark_bootstrap_draws(a[0], a[1], n_repeats=n_repeats, n_boot=n_boot, seed=seed)
    draws_b = landmark_bootstrap_draws(b[0], b[1], n_repeats=n_repeats, n_boot=n_boot, seed=seed)
    if draws_a["vehicles"] != draws_b["vehicles"]:
        raise ValueError("Las dos corridas no tienen los mismos vehículos: el bootstrap no sería pareado")
    out = {}
    for metric, key in (("d1", ("d1", "mean")), ("d2", ("d2", "detection_rate"))):
        sa, sb = block_a[key[0]][key[1]], block_b[key[0]][key[1]]
        diff = sa["mean"] - sb["mean"]
        threshold = float(np.sqrt(sa["std"] ** 2 + sb["std"] ** 2))
        paired = draws_a[metric] - draws_b[metric]
        paired = paired[np.isfinite(paired)]
        lo, hi = (float(np.quantile(paired, alpha / 2)), float(np.quantile(paired, 1 - alpha / 2)))
        by_repeat = (np.asarray(block_a[metric]["by_repeat"], dtype=float) if metric == "d1" else
                     np.asarray([r["detection_rate"] for r in block_a["d2"]["by_repeat"]], dtype=float))
        by_repeat_b = (np.asarray(block_b[metric]["by_repeat"], dtype=float) if metric == "d1" else
                       np.asarray([r["detection_rate"] for r in block_b["d2"]["by_repeat"]], dtype=float))
        out[metric] = {
            "a": sa["mean"], "b": sb["mean"], "diff": diff, "threshold_sqrt_var": threshold,
            "paired_boot_lo": lo, "paired_boot_hi": hi, "n_boot": int(paired.size),
            "diff_by_repeat": (by_repeat - by_repeat_b).tolist() if len(by_repeat) == len(by_repeat_b) else None,
            "wins": bool(diff > threshold and lo > 0), "loses": bool(diff < -threshold and hi < 0),
        }
    wins = [m for m in ("d1", "d2") if out[m]["wins"]]
    loses = [m for m in ("d1", "d2") if out[m]["loses"]]
    out["verdict"] = "pierde" if loses else ("gana" if wins else "empata")
    return out


def cv_from_cfg(panel: pd.DataFrame, splits: dict, cfg: dict, *, model_params: dict | None = None) -> pd.DataFrame:
    """La misma `run_cv` que `train.py`, con lo que declara el YAML (y params pisados)."""
    model = cfg["model"]
    predictions, _ = run_cv(
        panel, splits, model_name=model["name"], model_params={**(model.get("params") or {}), **(model_params or {})},
        target=cfg.get("target") or None, strict_splits=bool(cfg["splits"].get("strict", True)),
        min_valid_positives=split_options(cfg)["min_valid_positives"],
        preprocessing=str(cfg.get("preprocessing", "standard")),
        carry_columns=(cfg.get("eval") or {}).get("carry_columns"),
    )
    return predictions


# --------------------------------------------------------------------------- #
# Auditorías
# --------------------------------------------------------------------------- #
def permute_features_within(panel: pd.DataFrame, landmark_column: str, rng: np.random.Generator) -> pd.DataFrame:
    """Las filas de features (todas las `feat_`/`static_` juntas) permutadas dentro de cada hito."""
    features = select_feature_columns(panel)
    out = panel.copy()
    indices = panel.groupby(landmark_column).indices
    for column in features:
        values = panel[column].to_numpy().copy()
        for idx in indices.values():
            values[idx] = values[rng.permutation(idx)]
        out[column] = values
    return out


def audit_a0(dev: pd.DataFrame, splits: dict, cfg: dict, audit: dict, n_repeats: int, seed: int,
             meta: dict | None) -> dict[str, Any]:
    spec = audit.get("a0") or {}
    budget = float(cfg["eval"]["landmark"].get("max_false_alarms_per_1000", 50)) / 1000.0
    rng = np.random.default_rng(seed)
    runs = []
    for i in range(int(spec.get("permutations", 5))):
        permuted = permute_features_within(dev, spec.get("landmark_column", "feat_landmark_day"), rng)
        block = measure(cv_from_cfg(permuted, splits, cfg), cfg, n_repeats, seed, meta, n_boot=50)
        runs.append(headline(block))
        logger.info("A0 permutación %d: D1 %.4f · D2 %.3f", i, runs[-1]["d1"], runs[-1]["d2"])
    d1 = dispersion([r["d1"] for r in runs])
    d2 = dispersion([r["d2"] for r in runs])
    passed = (abs(d1["mean"] - 0.5) <= float(spec.get("d1_tolerance", 0.05))
              and abs(d2["mean"] - budget) <= float(spec.get("d2_tolerance", 0.05)))
    return {"d1": d1, "d2": d2, "budget": budget, "runs": runs, "passed": bool(passed),
            "reading": "sin fuga" if passed else "FALLA: hay fuga o el nulo no es nulo; no se lee ningún otro número"}


def audit_c1(predictions: pd.DataFrame, cfg: dict, audit: dict, n_repeats: int, seed: int,
             observed: float) -> dict[str, Any]:
    spec = audit.get("c1") or {}
    lm = landmark_eval_config(cfg["eval"].get("landmark"))
    strata = list(spec.get("strata", [lm["landmark_column"], "aux_production_tercile", "static_SalesCountry_cd"]))
    missing = [c for c in strata if c not in predictions]
    if missing:
        raise KeyError(f"C1: las predicciones no traen {missing}; agregalas a `eval.carry_columns`")
    columns = repeat_columns(lm["score_column"], n_repeats)
    parts = []
    for _, frame in predictions.groupby(lm["landmark_column"], sort=True):
        groups = [np.asarray(ix) for ix in frame.groupby(strata[1:] if len(strata) > 1 else strata,
                                                         dropna=False).indices.values()]
        parts.append({"entry": frame[lm["entry_column"]].to_numpy(float), "exit": frame[lm["exit_column"]].to_numpy(float),
                      "event": frame[lm["event_column"]].to_numpy(int), "scores": frame[columns].to_numpy(float),
                      "groups": groups})
    def d1_mean(permute: bool, rng: np.random.Generator | None = None) -> float:
        """D1 promedio de los hitos en cada repetición, promediado entre repeticiones."""
        per_repeat = np.zeros(len(columns))
        for part in parts:
            order = np.arange(len(part["event"]))
            if permute:
                for group in part["groups"]:
                    order[group] = group[rng.permutation(len(group))]
            scores = part["scores"][order]
            for r in range(len(columns)):
                per_repeat[r] += late_entry_concordance(part["entry"], part["exit"], part["event"], scores[:, r])["c_index"]
        return float(np.mean(per_repeat / len(parts)))

    # Sin permutar, la cuenta tiene que dar el D1 de metrics.json: si no, el nulo mide otra cosa.
    identity = d1_mean(False)
    if not np.isclose(identity, observed, atol=1e-9):
        raise RuntimeError(f"C1: sin permutar da {identity:.6f} y el D1 observado es {observed:.6f}")
    rng = np.random.default_rng(int(spec.get("seed", seed)))
    n_perm = int(spec.get("permutations", 2000))
    null = np.array([d1_mean(True, rng) for _ in range(n_perm)])
    p_value = float((1 + np.sum(null >= observed)) / (n_perm + 1))
    return {"observed_d1": observed, "null_mean": float(null.mean()), "null_std": float(null.std(ddof=0)),
            "null_p95": float(np.quantile(null, 0.95)), "p_value": p_value, "permutations": n_perm,
            "strata": strata, "passed": bool(p_value < float(spec.get("alpha", 0.05)))}


def audit_c2(predictions: pd.DataFrame, cfg: dict, audit: dict, n_repeats: int, seed: int, meta: dict | None,
             block: dict) -> dict[str, Any]:
    spec = audit.get("c2") or {}
    lm = landmark_eval_config(cfg["eval"].get("landmark"))
    n_boot = int(audit.get("n_boot", 2000))
    out: dict[str, Any] = {"floors": {}}
    for name, column in (spec.get("floors") or {}).items():
        floor = with_score(predictions, -predictions[column].to_numpy(float), n_repeats)
        floor_block = measure(floor, cfg, n_repeats, seed, meta, score_column=AUDIT_SCORE, n_boot=n_boot)
        out["floors"][name] = {
            "score": f"−{column}", **headline(floor_block),
            "model_vs_floor": beats((predictions, {**cfg["eval"]["landmark"], "n_boot": n_boot}),
                                    (floor, {**cfg["eval"]["landmark"], "score_column": AUDIT_SCORE}),
                                    block, floor_block, n_repeats=n_repeats, n_boot=n_boot, seed=seed),
        }
    # ρ del score del modelo con ProductionDay y AUC dentro de cada tercil, por hito.
    production = spec.get("production_column", "aux_static_ProductionDay")
    tercile = spec.get("tercile_column", "aux_production_tercile")
    resolved = predictions.groupby("vehicle_id")[lm["resolved_column"]].transform("max").eq(1)
    by_landmark = {}
    for landmark, frame in predictions.groupby(lm["landmark_column"], sort=True):
        score = frame[lm["score_column"]]
        keep = frame[lm["event_column"]].eq(1) | resolved.loc[frame.index]
        within = {}
        for t, part in frame.loc[keep].groupby(tercile):
            within[str(t)] = roc_auc(part[lm["event_column"]].to_numpy(int), part[lm["score_column"]].to_numpy(float))
        by_landmark[f"{landmark:g}"] = {
            "spearman_with_production": float(score.rank().corr(frame[production].rank())),
            "auc_within_production_tercile": within,
        }
    rhos = [v["spearman_with_production"] for v in by_landmark.values()]
    out["model"] = {"by_landmark": by_landmark, "max_abs_spearman": float(np.max(np.abs(rhos))),
                    "spearman_ok": bool(np.max(np.abs(rhos)) < float(spec.get("max_abs_rho", 0.2)))}
    out["passed"] = bool(all(f["model_vs_floor"]["d1"]["wins"] for f in out["floors"].values()))
    return out


def audit_c3(predictions: pd.DataFrame, cfg: dict, audit: dict, n_repeats: int, seed: int, meta: dict | None,
             block: dict) -> dict[str, Any]:
    spec = audit.get("c3") or {}
    lm = landmark_eval_config(cfg["eval"].get("landmark"))
    n_boot = int(audit.get("n_boot", 2000))
    usage = predictions[spec.get("km_column", "window_km")].to_numpy(float) / predictions[lm["landmark_column"]].to_numpy(float)
    floor = with_score(predictions, -usage, n_repeats)
    floor_block = measure(floor, cfg, n_repeats, seed, meta, score_column=AUDIT_SCORE, n_boot=n_boot)
    return {"score": "−km/L en (venta, venta + L]", **headline(floor_block),
            "model_vs_floor": beats((predictions, {**cfg["eval"]["landmark"], "n_boot": n_boot}),
                                    (floor, {**cfg["eval"]["landmark"], "score_column": AUDIT_SCORE}),
                                    block, floor_block, n_repeats=n_repeats, n_boot=n_boot, seed=seed)}


def audit_c6(predictions: pd.DataFrame, cfg: dict, audit: dict, n_repeats: int, seed: int, meta: dict | None,
             block: dict) -> dict[str, Any]:
    spec = audit.get("c6") or {}
    clock = load_config(audit.get("event_clock", "configs/data/event_clock.yaml"))
    immobile_km = float(clock["early_trait"]["immobile_km"])
    moving = predictions[spec.get("km_column", "window_km")].to_numpy(float) >= immobile_km
    subset = predictions.loc[moving].reset_index(drop=True)
    sub_block = measure(subset, cfg, n_repeats, seed, meta, n_boot=int(audit.get("n_boot", 2000)))
    lm = landmark_eval_config(cfg["eval"].get("landmark"))
    removed = (predictions.loc[~moving].groupby(lm["landmark_column"]).size())
    return {"immobile_km": immobile_km, "all": headline(block), "without_immobile": headline(sub_block),
            "rows_removed_by_landmark": {f"{k:g}": int(v) for k, v in removed.items()}}


def audit_retrain(dev: pd.DataFrame, splits: dict, cfg: dict, n_repeats: int, seed: int, meta: dict | None,
                  block: dict, *, panel: pd.DataFrame, model_params: dict | None = None) -> dict[str, Any]:
    variant = measure(cv_from_cfg(panel, splits, cfg, model_params=model_params), cfg, n_repeats, seed, meta,
                      n_boot=int((cfg.get("audit_cure") or {}).get("n_boot", 2000)))
    base, new = headline(block), headline(variant)
    return {"reference": base, "variant": new, "delta_d1": new["d1"] - base["d1"],
            "delta_auc_resolved": new["auc_resolved_mean"] - base["auc_resolved_mean"],
            "delta_d2": new["d2"] - base["d2"]}


def audit_a3(dev, splits, cfg, audit, n_repeats, seed, meta, block) -> dict[str, Any]:
    spec = audit.get("a3") or {}
    columns = list(spec.get("covariates", ["aux_landmark_month", "aux_air_temp_window_mean", "aux_static_ProductionDay"]))
    # Copias `feat_aux_*`: las `aux_` originales se quedan, porque `eval.carry_columns` las arrastra.
    names = [f"feat_{c}" for c in columns]
    promoted = dev.assign(**{name: dev[c].astype(float) for name, c in zip(names, columns)})
    out = audit_retrain(dev, splits, cfg, n_repeats, seed, meta, block, panel=promoted)
    threshold = float(spec.get("threshold", 0.02))
    out.update({"covariates": names, "threshold": threshold,
                "flag": bool(out["delta_d1"] > threshold or out["delta_auc_resolved"] > threshold)})
    return out


def audit_a5(dev, splits, cfg, audit, n_repeats, seed, meta, block) -> dict[str, Any]:
    spec = audit.get("a5") or {}
    prefix = spec.get("raw_prefix", "aux_raw_")
    raw = dev.drop(columns=[c for c in dev.columns if c.startswith("feat_fm__")])
    raw_columns = [c for c in dev.columns if c.startswith(prefix)]
    for column in raw_columns:
        raw[f"feat_raw_{column[len(prefix):]}"] = dev[column].astype(float)
    out = audit_retrain(dev, splits, cfg, n_repeats, seed, meta, block, panel=raw,
                        model_params={"fleet_normalization": False})
    out["raw_columns"] = raw_columns
    return out


def audit_a6(predictions: pd.DataFrame, cfg: dict, audit: dict, n_repeats: int, seed: int,
             meta: dict | None) -> dict[str, Any]:
    spec = audit.get("a6") or {}
    ref_dir = resolve_path(cfg.get("output_dir", "experiments")) / spec["reference_run"]
    reference = pd.read_parquet(ref_dir / "predictions.parquet")
    ref_metrics = json.loads((ref_dir / "metrics.json").read_text(encoding="utf-8"))
    ref_repeats = int(ref_metrics.get("n_repeats", 1))
    if ref_repeats != n_repeats:
        raise ValueError(f"A6: el finalista tiene R={ref_repeats} y esta corrida R={n_repeats}")
    ref_columns = repeat_columns("score", ref_repeats)
    left = predictions.assign(_row=np.arange(len(predictions)), vehicle_id=predictions["vehicle_id"].astype(str))
    right = reference[["vehicle_id", "cut_date", *ref_columns]].assign(vehicle_id=reference["vehicle_id"].astype(str))
    right = right.rename(columns={c: f"ref_{c}" for c in ref_columns}).dropna(subset=["cut_date"])
    merged = pd.merge_asof(left.sort_values("cut_date"), right.sort_values("cut_date"), on="cut_date",
                           by="vehicle_id", direction="backward").sort_values("_row")
    matched = merged[f"ref_{ref_columns[0]}"].notna().to_numpy()
    both = predictions.loc[matched].reset_index(drop=True)
    finalist = both.copy()
    for column in ref_columns:
        finalist[f"ref_{column}"] = merged.loc[matched, f"ref_{column}"].to_numpy(float)
    for r, column in enumerate(repeat_columns(AUDIT_SCORE, n_repeats)):
        finalist[column] = finalist[f"ref_{ref_columns[r]}"]
    finalist[AUDIT_SCORE] = finalist[[f"ref_{c}" for c in ref_columns]].mean(axis=1)
    n_boot = int(audit.get("n_boot", 2000))
    model_block = measure(both, cfg, n_repeats, seed, meta, n_boot=n_boot)
    finalist_block = measure(finalist, cfg, n_repeats, seed, meta, score_column=AUDIT_SCORE, n_boot=n_boot)
    lm = landmark_eval_config(cfg["eval"].get("landmark"))
    rows = predictions.groupby(lm["landmark_column"]).size()
    kept = both.groupby(lm["landmark_column"]).size()
    return {
        "reference_run": spec["reference_run"],
        "rows_matched_by_landmark": {f"{k:g}": f"{int(kept.get(k, 0))}/{int(v)}" for k, v in rows.items()},
        "n_vehicles": int(both["vehicle_id"].nunique()),
        "model": headline(model_block), "finalist": headline(finalist_block),
        "finalist_first_cut_vehicle_auc": spec.get("first_cut_auc"),
        "note": "Informativo: los sanos del finalista están submuestreados por el emparejado del panel v1.",
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def flatten(prefix: str, value: Any, out: dict[str, float]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            flatten(f"{prefix}/{key}" if prefix else str(key), item, out)
    elif isinstance(value, (bool, int, float, np.floating, np.integer)) and not (
            isinstance(value, float) and np.isnan(value)):
        out[prefix] = float(value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--only", nargs="*", choices=AUDITS, help="Correr solo estas auditorías")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    for noisy in ("src.training.cv", "src.models.cure", "src.training.transformers", "src.eval.splits"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    cfg = load_config(args.config)
    audit = cfg.get("audit_cure") or {}
    seed = set_seed(int(cfg.get("seed", 42)))
    run_dir = resolve_path(cfg.get("output_dir", "experiments")) / cfg["name"]
    predictions = pd.read_parquet(run_dir / "predictions.parquet")
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    n_repeats = int(metrics.get("n_repeats", 1))
    block = metrics["landmark"]
    panel_path = resolve_path(cfg["data"]["panel"])
    meta = panel_meta(panel_path)
    wanted = args.only or list(audit.get("audits", []))
    unknown = [a for a in wanted if a not in AUDITS]
    if unknown:
        raise KeyError(f"`audit_cure.audits` pide auditorías que no existen: {unknown}")

    dev = select_dev(pd.read_parquet(panel_path), cfg)
    splits = load_or_make_splits(dev, cfg) if {"A0", "A3", "A5"} & set(wanted) else None
    results: dict[str, Any] = {"run": cfg["name"], "score_column": block["config"]["score_column"],
                               "observed": headline(block)}
    logger.info("%s | D1 %.4f ± %.4f | D2 %.3f ± %.3f (score `%s`)", cfg["name"], results["observed"]["d1"],
                results["observed"]["d1_std"], results["observed"]["d2"], results["observed"]["d2_std"],
                results["score_column"])
    for name in wanted:
        logger.info("== %s ==", name)
        if name == "A0":
            results["A0"] = audit_a0(dev, splits, cfg, audit, n_repeats, seed, meta)
        elif name == "C1":
            results["C1"] = audit_c1(predictions, cfg, audit, n_repeats, seed, results["observed"]["d1"])
        elif name == "C2":
            results["C2"] = audit_c2(predictions, cfg, audit, n_repeats, seed, meta, block)
        elif name == "C3":
            results["C3"] = audit_c3(predictions, cfg, audit, n_repeats, seed, meta, block)
        elif name == "C6":
            results["C6"] = audit_c6(predictions, cfg, audit, n_repeats, seed, meta, block)
        elif name == "A3":
            results["A3"] = audit_a3(dev, splits, cfg, audit, n_repeats, seed, meta, block)
        elif name == "A5":
            results["A5"] = audit_a5(dev, splits, cfg, audit, n_repeats, seed, meta, block)
        elif name == "A6":
            results["A6"] = audit_a6(predictions, cfg, audit, n_repeats, seed, meta)

    compare_to = audit.get("compare_to")
    if compare_to:
        other_dir = resolve_path(cfg.get("output_dir", "experiments")) / compare_to
        other_pred = pd.read_parquet(other_dir / "predictions.parquet")
        other_metrics = json.loads((other_dir / "metrics.json").read_text(encoding="utf-8"))
        n_boot = int(audit.get("n_boot", 2000))
        results["verdict"] = {
            "against": compare_to,
            **beats((predictions, {**cfg["eval"]["landmark"], "n_boot": n_boot}),
                    (other_pred, {**other_metrics["landmark"]["config"], "n_boot": n_boot}),
                    block, other_metrics["landmark"], n_repeats=n_repeats, n_boot=n_boot, seed=seed),
        }

    adoption = {}
    if "C1" in results:
        adoption["beats_stratified_null_c1"] = results["C1"]["passed"]
    if "C2" in results:
        adoption["beats_production_floor_c2"] = results["C2"]["passed"]
    adoption["calendar_a3_not_flagged"] = (not results["A3"]["flag"]) if "A3" in results else "no aplica"
    if "A0" in results:
        adoption["no_leakage_a0"] = results["A0"]["passed"]
    results["adoption"] = adoption
    if "C1" in results and not results["C1"]["passed"] and audit.get("stop_if_c1_fails", False):
        results["stop"] = ("P0 no le gana a su nulo estratificado en D1: el rasgo temprano era ruido. "
                           "Se documenta el negativo y P1/P2 no corren (preregistro §8).")

    out_path = run_dir / "audit_cure.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print(json.dumps({k: v for k, v in results.items() if k not in ("A0",)}, indent=1, ensure_ascii=False,
                     default=float)[:6000])
    logger.info("Auditorías: %s", out_path)

    wandb_cfg = cfg.get("wandb") or {}
    if wandb_cfg.get("enabled", True):
        import wandb

        run = wandb.init(project=wandb_cfg.get("project", "ford-fic"),
                         entity=os.environ.get("WANDB_ENTITY") or wandb_cfg.get("entity"),
                         name=f"audit-{cfg['name']}", group=wandb_cfg.get("group"), job_type="audit",
                         tags=[*(wandb_cfg.get("tags") or []), "audit"],
                         mode=os.environ.get("WANDB_MODE") or wandb_cfg.get("mode", "online"),
                         config={"experiment": cfg["name"], "audit_cure": audit})
        flat: dict[str, float] = {}
        flatten("", {k: v for k, v in results.items() if k not in ("run", "score_column")}, flat)
        run.log(flat)
        artifact = wandb.Artifact(f"{cfg['name']}-audit", type="audit")
        artifact.add_file(str(out_path))
        run.log_artifact(artifact)
        run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
