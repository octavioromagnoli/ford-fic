#!/usr/bin/env python
"""Ensamble por rango de corridas que ya existen, medido con la misma cuenta que `train.py`.

    python scripts/ensemble_rank.py --config configs/exp_ens_e1.yaml            # arma y mide
    python scripts/ensemble_rank.py --config configs/exp_ens_e1.yaml --audit    # + (a0) y (b)

Es E1 y E2 del preregistro `docs/memoria/f3-preregistro-landmark-ensamble.md` (9b7bfb4),
y está escrito para aplicarlo tal cual:

* **Score.** Para cada repetición `r`, el rango percentil (empates promediados) del
  `score_r{r}` de cada miembro sobre las filas de dev, y el promedio con los pesos del
  YAML (50/50, fijos: no se ajustan). No se reentrena nada: los miembros son corridas de
  `experiments/` con sus predicciones out-of-fold.
* **Antes de medir**, las filas tienen que coincidir por `(vehicle_id, cut_odo)` —y en
  `label` y `event_observed`— y `fold_r{r}` tiene que ser idéntico en todos los miembros.
  Si no, el script falla: un ensamble de folds distintos no es out-of-fold.
* **Métricas:** `scripts/train.py::evaluate_predictions`, la misma función que usa toda
  corrida. Deciden la detección a ≤ 50 falsas alarmas cada 1.000 sanos (media ± desvío
  entre las 3 repeticiones) y el lift por vehículo con `mean`.
* **Veredicto** contra la referencia (reglas del preregistro): "le gana a X en M" = la
  diferencia de medias supera `√(σ² + σ_X²)`. Gana si le gana en una y no pierde en la
  otra; pierde si pierde en cualquiera; si no, empata. Para reemplazar a la referencia
  tiene que ganarle también al piso posicional en las dos.
* **Calibración.** El rango no es una probabilidad: su Brier no se declara. Se reporta el
  de una Platt **fuera de fold** (para cada fold, la logística se ajusta con las filas de
  los otros folds). Es monótona, así que no cambia detección ni lift.

`--audit` agrega lo que no sale de las predicciones: reentrena cada miembro con las
features permutadas entre todas las filas (a0) y con las `aux_` de calendario como
`feat_` (b), usando las mismas funciones que `scripts/audit_model.py`, y ensambla esas
predicciones con la misma regla. (a') sale de `evaluate_predictions`. El piso posicional
y el nulo de tamaño de bolsa se corren aparte sobre la salida
(`audit_positional_floor.py`, `audit_mil_bagsize.py`).

Deja en `experiments/<name>/` lo mismo que `train.py` (`predictions.parquet`,
`metrics.json`, `config.yaml`, `lead_time_curve.csv`), así que `compare.py`,
`results.py` y las auditorías lo leen como a cualquier corrida.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from scripts.audit_model import CALENDAR_AUX, evaluate, permute_all, promote_aux  # noqa: E402
from scripts.train import (  # noqa: E402
    evaluate_predictions,
    init_wandb,
    load_or_make_splits,
    select_dev,
)
from src.config import ensure_dir, load_config, repo_root, resolve_path, set_seed  # noqa: E402
from src.eval.metrics import (  # noqa: E402
    classification_metrics,
    dispersion,
    lead_time_curve,
    operating_point,
    vehicle_metrics,
)
from src.training.cv import CARRY_COLUMNS, ID_COLUMNS  # noqa: E402

logger = logging.getLogger("ensemble_rank")

KEY = ["vehicle_id", "cut_odo"]
#: Escala del piso posicional (la de `audit_positional_floor.py`): no toca el orden.
FLOOR_SCALE_KM = 100_000.0


# --------------------------------------------------------------------------- #
# Armado
# --------------------------------------------------------------------------- #
def n_repeats_of(frame: pd.DataFrame) -> int:
    repeats = sorted(int(c[len("score_r"):]) for c in frame.columns if c.startswith("score_r"))
    if not repeats:
        raise ValueError("Las predicciones no traen `score_r{i}`: el ensamble del preregistro es con R > 1")
    if repeats != list(range(len(repeats))):
        raise ValueError(f"Repeticiones salteadas en las predicciones: {repeats}")
    return len(repeats)


def align_members(frames: list[pd.DataFrame], names: list[str]) -> list[pd.DataFrame]:
    """Mismas filas, mismo orden, misma etiqueta y mismos folds, o falla."""
    base = frames[0].sort_values(KEY).reset_index(drop=True)
    if base.duplicated(KEY).any():
        raise ValueError(f"{names[0]}: hay pares {KEY} repetidos")
    n_repeats = n_repeats_of(base)
    aligned = [base]
    for frame, name in zip(frames[1:], names[1:]):
        if len(frame) != len(base):
            raise ValueError(f"{name}: {len(frame)} filas contra {len(base)} de {names[0]}")
        other = frame.sort_values(KEY).reset_index(drop=True)
        if not other[KEY].equals(base[KEY]):
            raise ValueError(f"{name}: las filas no coinciden por {KEY} con {names[0]}")
        for column in ("label", "event_observed"):
            if not (other[column].to_numpy() == base[column].to_numpy()).all():
                raise ValueError(f"{name}: `{column}` no coincide con {names[0]} fila a fila")
        if n_repeats_of(other) != n_repeats:
            raise ValueError(f"{name}: {n_repeats_of(other)} repeticiones contra {n_repeats}")
        for r in range(n_repeats):
            if not (other[f"fold_r{r}"].to_numpy() == base[f"fold_r{r}"].to_numpy()).all():
                raise ValueError(
                    f"{name}: `fold_r{r}` no es idéntico al de {names[0]}. Un ensamble de folds "
                    "distintos no es out-of-fold."
                )
        aligned.append(other)
    return aligned


def rank_ensemble(frames: list[pd.DataFrame], weights: list[float]) -> pd.DataFrame:
    """Rango percentil por repetición, promedio ponderado. Las filas ya vienen alineadas."""
    base = frames[0]
    n_repeats = n_repeats_of(base)
    total = float(sum(weights))
    columns = [c for c in (*ID_COLUMNS, *CARRY_COLUMNS, "label") if c in base.columns]
    out = base[columns].copy()
    # Columnas que el primer miembro no trae y otro sí (el C-index necesita
    # `aux_km_observed_after_cut`, que el panel secuencial no arrastra).
    for frame in frames[1:]:
        for column in CARRY_COLUMNS:
            if column not in out.columns and column in frame.columns:
                out[column] = frame[column].to_numpy()

    stacked = []
    for r in range(n_repeats):
        ranks = [
            frame[f"score_r{r}"].rank(method="average", pct=True).to_numpy(dtype=float)
            for frame in frames
        ]
        score_r = sum(w * rk for w, rk in zip(weights, ranks)) / total
        out[f"score_r{r}"] = score_r
        out[f"fold_r{r}"] = base[f"fold_r{r}"].to_numpy()
        stacked.append(score_r)
    stacked = np.vstack(stacked)
    out["score"] = stacked.mean(axis=0)
    out["score_std"] = stacked.std(axis=0, ddof=0)
    return out


def member_correlations(frames: list[pd.DataFrame], names: list[str]) -> list[dict[str, Any]]:
    """Correlación de rango entre miembros, por fila y por vehículo (`mean`), por repetición."""
    out = []
    n_repeats = n_repeats_of(frames[0])
    for i in range(len(frames)):
        for j in range(i + 1, len(frames)):
            by_row, by_vehicle = [], []
            for r in range(n_repeats):
                a, b = frames[i][f"score_r{r}"], frames[j][f"score_r{r}"]
                by_row.append(float(a.rank().corr(b.rank())))
                va = frames[i].groupby("vehicle_id")[f"score_r{r}"].mean()
                vb = frames[j].groupby("vehicle_id")[f"score_r{r}"].mean()
                by_vehicle.append(float(va.rank().corr(vb.loc[va.index].rank())))
            out.append({"members": [names[i], names[j]], "spearman_row": by_row,
                        "spearman_vehicle": by_vehicle})
    return out


# --------------------------------------------------------------------------- #
# Calibración Platt fuera de fold (solo para reportar el Brier)
# --------------------------------------------------------------------------- #
def platt_oof(predictions: pd.DataFrame, n_repeats: int) -> dict[str, Any]:
    """Brier de una logística sobre el score, ajustada con los otros folds de cada repetición."""
    from sklearn.linear_model import LogisticRegression

    y = predictions["label"].to_numpy(dtype=int)
    briers, means = [], []
    for r in range(n_repeats):
        score = predictions[f"score_r{r}"].to_numpy(dtype=float).reshape(-1, 1)
        folds = predictions[f"fold_r{r}"].to_numpy()
        calibrated = np.full(len(y), np.nan)
        for fold in np.unique(folds):
            train, valid = folds != fold, folds == fold
            model = LogisticRegression(C=1e6, max_iter=1000)  # ~sin penalización: Platt
            model.fit(score[train], y[train])
            calibrated[valid] = model.predict_proba(score[valid])[:, 1]
        briers.append(float(np.mean((calibrated - y) ** 2)))
        means.append(float(calibrated.mean()))
    return {
        "method": "platt_out_of_fold",
        "brier_by_repeat": briers,
        "brier": dispersion(briers),
        "mean_predicted": dispersion(means),
        "base_rate": float(y.mean()),
    }


# --------------------------------------------------------------------------- #
# Veredicto (reglas del preregistro)
# --------------------------------------------------------------------------- #
def decision_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """Detección, anticipación y lift por vehículo (`mean`) de cada repetición."""
    detect = [p for p in metrics.get("detection_by_repeat") or [] if p is not None]
    vehicle = (metrics.get("vehicle") or {})
    by_repeat = (vehicle.get("by_repeat") or {}).get("mean")
    if by_repeat:
        lifts = [float(m["pr_auc_lift"]) for m in by_repeat]
    else:
        lifts = [float(vehicle["aggregations"]["mean"]["pr_auc_lift"])]
    return {
        "detection": [float(p["detection_rate"]) for p in detect],
        "n_detected": [int(p["n_detected"]) for p in detect],
        "n_event_vehicles": int(detect[0]["n_event_vehicles"]) if detect else None,
        "lead_km": [float(p["median_lead_km"]) for p in detect],
        "vehicle_lift_mean": lifts,
    }


def floor_metrics(predictions: pd.DataFrame, eval_cfg: dict) -> dict[str, Any]:
    """El piso posicional (score = odómetro del corte) sobre las mismas filas. Sin repeticiones."""
    frame = predictions.assign(score=predictions["cut_odo"] / FLOOR_SCALE_KM)
    curve = lead_time_curve(frame, n_thresholds=int(eval_cfg.get("n_thresholds", 50)),
                            k_consecutive=int(eval_cfg.get("k_consecutive", 2)))
    point = operating_point(curve, max_false_alarms_per_1000=float(eval_cfg.get("max_false_alarms_per_1000", 50)))
    return {
        "detection": [float(point["detection_rate"])] if point else [0.0],
        "n_detected": [int(point["n_detected"])] if point else [0],
        "n_event_vehicles": int(point["n_event_vehicles"]) if point else None,
        "lead_km": [float(point["median_lead_km"])] if point else [float("nan")],
        "vehicle_lift_mean": [float(vehicle_metrics(frame, how="mean")["pr_auc_lift"])],
    }


def compare(candidate: dict[str, Any], other: dict[str, Any]) -> dict[str, Any]:
    """Diferencia de medias contra el desvío combinado, en las dos métricas que deciden."""
    out: dict[str, Any] = {}
    for key in ("detection", "vehicle_lift_mean"):
        a, b = dispersion(candidate[key]), dispersion(other[key])
        diff = a["mean"] - b["mean"]
        pooled = math.sqrt(a["std"] ** 2 + b["std"] ** 2)
        paired = (
            [x - y for x, y in zip(candidate[key], other[key])]
            if len(candidate[key]) == len(other[key]) > 1 else None
        )
        out[key] = {
            "candidate": a["mean"], "candidate_std": a["std"],
            "other": b["mean"], "other_std": b["std"],
            "diff": diff, "pooled_std": pooled,
            "wins": bool(diff > pooled), "loses": bool(-diff > pooled),
            "paired_by_repeat": paired,
        }
    wins = [k for k in out if out[k]["wins"]]
    loses = [k for k in out if out[k]["loses"]]
    if loses:
        verdict = "pierde"
    elif wins:
        verdict = "gana"
    else:
        verdict = "empata"
        cand_det, other_det = dispersion(candidate["detection"]), dispersion(other["detection"])
        cand_lead, other_lead = dispersion(candidate["lead_km"]), dispersion(other["lead_km"])
        if cand_det["std"] > other_det["std"] or cand_lead["std"] > other_lead["std"]:
            verdict = "empata, con más desvío (cuenta como peor)"
    out["verdict"] = verdict
    out["lead_km"] = {"candidate": dispersion(candidate["lead_km"]), "other": dispersion(other["lead_km"])}
    return out


def load_metrics(run: str, experiments: Path) -> dict[str, Any]:
    path = experiments / run / "metrics.json"
    if not path.exists():
        raise FileNotFoundError(f"No existe {path}: ¿la corrida `{run}` está entrenada en este checkout?")
    return json.loads(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# Auditorías que necesitan reentrenar a los miembros
# --------------------------------------------------------------------------- #
def audit_members(members: list[dict], weights: list[float], *, seed: int) -> dict[str, Any]:
    """(a0) y (b) del ensamble: cada miembro reentrenado igual que en `audit_model.py`."""
    permuted, promoted, reference = [], [], []
    for member in members:
        cfg = load_config(member["config"])
        set_seed(int(cfg.get("seed", 42)))
        panel = select_dev(pd.read_parquet(resolve_path(cfg["data"]["panel"])), cfg)
        splits = load_or_make_splits(panel, cfg)
        logger.info("auditando %s (%s)", member["run"], member["config"])
        _, pred_a0 = evaluate(permute_all(panel, seed=seed), splits, cfg,
                              label=f"(a0) {member['run']}")
        promoted_panel, present = promote_aux(panel, CALENDAR_AUX)
        _, pred_b = evaluate(promoted_panel, splits, cfg,
                             label=f"(b) {member['run']} +{len(present)} aux_")
        permuted.append(pred_a0)
        promoted.append(pred_b)
        reference.append(pd.read_parquet(resolve_path("experiments") / member["run"] / "predictions.parquet"))

    names = [m["run"] for m in members]
    ens_a0 = rank_ensemble(align_members(permuted, names), weights)
    ens_b = rank_ensemble(align_members(promoted, names), weights)
    ens_ref = rank_ensemble(align_members(reference, names), weights)
    # Misma convención que `audit_model.py`: se mide sobre `score` (promedio de repeticiones).
    null = classification_metrics(ens_a0["label"], ens_a0["score"])
    ref = classification_metrics(ens_ref["label"], ens_ref["score"])
    cal = classification_metrics(ens_b["label"], ens_b["score"])
    delta_null = null["pr_auc"] - null["base_rate"]
    delta_roc = cal["roc_auc"] - ref["roc_auc"]
    by_repeat_roc = [
        classification_metrics(ens_b["label"], ens_b[f"score_r{r}"])["roc_auc"]
        - classification_metrics(ens_ref["label"], ens_ref[f"score_r{r}"])["roc_auc"]
        for r in range(n_repeats_of(ens_ref))
    ]
    return {
        "a0": {"pr_auc": null["pr_auc"], "base_rate": null["base_rate"], "delta": delta_null,
               "passed": bool(abs(delta_null) < 0.02), "seed": seed},
        "b": {"roc_reference": ref["roc_auc"], "roc_with_calendar": cal["roc_auc"],
              "delta_roc": delta_roc, "delta_roc_by_repeat": by_repeat_roc,
              "marks": bool(delta_roc > 0.02), "columns": CALENDAR_AUX},
    }


# --------------------------------------------------------------------------- #
def fmt_pct(values: list[float]) -> str:
    d = dispersion(values)
    return f"{100 * d['mean']:.1f}% ± {100 * d['std']:.1f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--audit", action="store_true", help="reentrena los miembros para (a0) y (b)")
    parser.add_argument("--audit-seed", type=int, default=0, help="semilla de la permutación de (a0)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    cfg = load_config(args.config)
    seed = set_seed(int(cfg.get("seed", 42)))
    ens_cfg = cfg["ensemble"]
    if ens_cfg.get("method", "rank_mean") != "rank_mean":
        raise ValueError(f"method `{ens_cfg.get('method')}`: el preregistro es `rank_mean`")
    members = ens_cfg["members"]
    names = [m["run"] for m in members]
    weights = [float(m.get("weight", 1.0)) for m in members]
    experiments = resolve_path(cfg.get("output_dir", "experiments"))

    frames = [pd.read_parquet(experiments / run / "predictions.parquet") for run in names]
    aligned = align_members(frames, names)
    n_repeats = n_repeats_of(aligned[0])
    predictions = rank_ensemble(aligned, weights)
    logger.info("Ensamble %s | %d filas | %d vehículos | R=%d | pesos %s",
                " + ".join(names), len(predictions), predictions["vehicle_id"].nunique(),
                n_repeats, weights)

    evaluated, curve = evaluate_predictions(predictions, cfg, n_repeats=n_repeats, seed=seed)
    calibration = platt_oof(predictions, n_repeats)

    # Veredictos: contra la referencia, cada miembro, lo que el YAML pida y el piso.
    eval_cfg = cfg.get("eval", {})
    candidate = decision_metrics(evaluated)
    comparisons: dict[str, Any] = {}
    others = list(dict.fromkeys([cfg["compare"]["reference"], *names, *cfg["compare"].get("against", [])]))
    for run in others:
        comparisons[run] = {"decision": decision_metrics(load_metrics(run, experiments))}
        comparisons[run]["vs"] = compare(candidate, comparisons[run]["decision"])
    floor = floor_metrics(predictions, eval_cfg)
    floor_vs = compare(candidate, floor)
    beats_floor = floor_vs["detection"]["wins"] and floor_vs["vehicle_lift_mean"]["wins"]
    reference_vs = comparisons[cfg["compare"]["reference"]]["vs"]
    stability = None
    if cfg["compare"].get("stability_against"):
        other = decision_metrics(load_metrics(cfg["compare"]["stability_against"], experiments))
        stability = {
            "against": cfg["compare"]["stability_against"],
            "detection_std": {"candidate": dispersion(candidate["detection"])["std"],
                              "other": dispersion(other["detection"])["std"]},
            "lead_km_std": {"candidate": dispersion(candidate["lead_km"])["std"],
                            "other": dispersion(other["lead_km"])["std"]},
            "vs": compare(candidate, other),
        }
        stability["lowers_detection_std"] = stability["detection_std"]["candidate"] < stability["detection_std"]["other"]
        stability["lowers_lead_std"] = stability["lead_km_std"]["candidate"] < stability["lead_km_std"]["other"]

    audit = audit_members(members, weights, seed=args.audit_seed) if args.audit else None

    member_meta = []
    for run in names:
        m = load_metrics(run, experiments)
        member_meta.append({"run": run, "panel_build": m.get("panel_build"), "n_repeats": m.get("n_repeats")})
    metrics = {
        **evaluated,
        "panel_build": member_meta[0]["panel_build"],
        "ensemble": {
            "method": "rank_mean",
            "members": [{**m, "weight": w} for m, w in zip(members, weights)],
            "member_meta": member_meta,
            "checks": {"rows": len(predictions), "same_rows_labels_folds": True},
            "correlations": member_correlations(aligned, names),
            "calibration": calibration,
            "brier_note": "El Brier de `oof` es el del rango, que no es una probabilidad: no se "
                          "declara. El que vale es `calibration.brier` (Platt fuera de fold).",
        },
        "decision": {
            "candidate": candidate,
            "comparisons": comparisons,
            "floor": {"decision": floor, "vs": floor_vs, "beaten_in_both": bool(beats_floor)},
            "verdict_vs_reference": reference_vs["verdict"],
            "can_replace_reference": bool(reference_vs["verdict"] == "gana" and beats_floor),
            "stability": stability,
        },
        "ensemble_audit": audit,
    }

    run_name = cfg.get("name")
    out_dir = ensure_dir(experiments / run_name)
    predictions.to_parquet(out_dir / "predictions.parquet", index=False)
    if not curve.empty:
        curve.to_csv(out_dir / "lead_time_curve.csv", index=False)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float), encoding="utf-8")
    # `compare.py`/`results.py` leen el nombre del modelo de `model.name`. Un ensamble no es
    # un modelo del registry: se deja escrito qué es para que la tabla no diga `?`.
    saved_cfg = {**cfg, "model": cfg.get("model") or {"name": "rank_ensemble", "members": names}}
    (out_dir / "config.yaml").write_text(yaml.safe_dump(saved_cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    if audit is not None:
        (out_dir / "audit.json").write_text(json.dumps(audit, indent=2, default=float), encoding="utf-8")

    # ---- reporte
    n_ev = candidate["n_event_vehicles"]
    print(f"\n== {run_name}: {' + '.join(names)} (rango percentil, pesos {weights}) ==")
    print(f"{'corrida':<40} {'detección @≤50 FA/1000':<24} {'autos':<10} {'anticip. km':<16} {'lift veh. mean':<16} veredicto del ensamble")
    rows = [(run_name, candidate, "—")]
    rows += [(run, c["decision"], c["vs"]["verdict"]) for run, c in comparisons.items()]
    rows += [("piso posicional (cut_odo)", floor, floor_vs["verdict"])]
    for label, d, verdict in rows:
        lead = dispersion(d["lead_km"])
        lift = dispersion(d["vehicle_lift_mean"])
        print(f"{label:<40} {fmt_pct(d['detection']):<24} {'/'.join(map(str, d['n_detected'])):<10} "
              f"{lead['mean']:>7.0f} ± {lead['std']:<6.0f} {lift['mean']:.3f} ± {lift['std']:.3f}   {verdict}")
    print(f"(de {n_ev} vehículos con evento; veredicto = el ensamble contra esa fila, regla del preregistro)")
    ref = reference_vs
    print(f"\ncontra la referencia ({cfg['compare']['reference']}):")
    for key, label in (("detection", "detección"), ("vehicle_lift_mean", "lift veh.")):
        c = ref[key]
        print(f"  {label:<10} diferencia {c['diff']:+.4f} contra desvío combinado {c['pooled_std']:.4f} "
              f"| pareada por repetición {['%+.4f' % x for x in (c['paired_by_repeat'] or [])]}")
    print(f"  veredicto: {ref['verdict']} | le gana al piso en las dos: {beats_floor} | "
          f"puede reemplazar a la referencia: {metrics['decision']['can_replace_reference']}")
    if stability:
        print(f"\nestabilidad contra {stability['against']}: desvío de detección "
              f"{100 * stability['detection_std']['candidate']:.1f} vs {100 * stability['detection_std']['other']:.1f} pts | "
              f"de anticipación {stability['lead_km_std']['candidate']:.0f} vs {stability['lead_km_std']['other']:.0f} km")
    spread = evaluated["when_spread"]
    print(f"\n(a') aporte del cuándo: {spread['when_delta']['mean']:+.4f} ± {spread['when_delta']['std']:.4f} | "
          f"PR-AUC fila {evaluated['oof']['pr_auc']:.4f} | ROC {evaluated['oof']['roc_auc']:.4f} | "
          f"C-index {(evaluated.get('concordance') or {}).get('c_index', float('nan')):.4f}")
    print(f"Brier (Platt fuera de fold): {calibration['brier']['mean']:.4f} ± {calibration['brier']['std']:.4f} "
          f"(riesgo medio {calibration['mean_predicted']['mean']:.3f}, tasa real {calibration['base_rate']:.3f})")
    for corr in metrics["ensemble"]["correlations"]:
        print(f"ρ de rango {corr['members'][0]} / {corr['members'][1]}: por fila "
              f"{min(corr['spearman_row']):.2f}–{max(corr['spearman_row']):.2f}, por vehículo "
              f"{min(corr['spearman_vehicle']):.2f}–{max(corr['spearman_vehicle']):.2f}")
    if audit:
        print(f"\n(a0) features permutadas: PR-AUC {audit['a0']['pr_auc']:.4f} vs tasa base "
              f"{audit['a0']['base_rate']:.4f} ({audit['a0']['delta']:+.4f}) — "
              f"{'PASS' if audit['a0']['passed'] else 'FALLA: hay fuga'}")
        print(f"(b) aux_ de calendario: ROC {audit['b']['roc_reference']:.4f} → {audit['b']['roc_with_calendar']:.4f} "
              f"({audit['b']['delta_roc']:+.4f}) — {'marca' if audit['b']['marks'] else 'no marca'}")
    print(f"\nescrito: {out_dir.relative_to(repo_root())}")

    run = init_wandb(cfg, run_name)
    if run is not None:
        flat = {
            "pitch/detection_rate": dispersion(candidate["detection"])["mean"],
            "pitch/detection_rate_std": dispersion(candidate["detection"])["std"],
            "pitch/median_lead_km": dispersion(candidate["lead_km"])["mean"],
            "vehicle/mean/pr_auc_lift": dispersion(candidate["vehicle_lift_mean"])["mean"],
            "when/delta": spread["when_delta"]["mean"],
            "oof/pr_auc": evaluated["oof"]["pr_auc"],
            "oof/roc_auc": evaluated["oof"]["roc_auc"],
            "calibration/brier_platt": calibration["brier"]["mean"],
        }
        run.log(flat)
        run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
