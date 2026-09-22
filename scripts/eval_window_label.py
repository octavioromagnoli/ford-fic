#!/usr/bin/env python
"""Una corrida contra la referencia con las dos etiquetas: la dura (D) y la corregida por ventana (V).

    python scripts/eval_window_label.py --config configs/exp_ss_post_venta_r3.yaml

Es el §3–§4 del preregistro `docs/memoria/f5-preregistro-ss-post-venta.md`, escrito para
aplicarlo tal cual:

* **D:** todas las filas de dev, como la tabla de `results/`.
* **V:** solo las filas cuyo horizonte `[c+G, c+G+H]` cae entero dentro de la ventana del
  registro (`window_eval.evaluable_column` del panel). La regla es la misma para positivas y
  negativas. Un vehículo sin filas evaluables sale de la cuenta.
* **Métricas:** `scripts/train.py::evaluate_predictions`, la misma función que usa toda
  corrida, sobre las filas de cada etiqueta. Deciden la detección a ≤ 50 falsas alarmas cada
  1.000 sanos y el lift por vehículo con `mean`, cada una por repetición.
* **Antes de medir**, las dos corridas tienen que coincidir por `(vehicle_id, cut_odo)`, en
  `label` y `event_observed`, y en `fold_r{r}` (`ensemble_rank.py::align_members`).
* **Veredicto** con la etiqueta que diga `window_eval.decides`, con la regla de
  `ensemble_rank.py::compare`: gana si le gana en una y no pierde en la otra (diferencia de
  medias contra el desvío combinado). Para reemplazar a la referencia tiene que ganarle
  también, en las dos métricas, a cada piso de `window_eval.floors` (score = esa columna,
  sin ajustar nada) y pasar (a0), que se lee del `audit.json` de `audit_model.py`.

No reentrena nada: lee `experiments/<corrida>/predictions.parquet` de las dos. Deja
`experiments/<name>/window_eval.json`.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from scripts.ensemble_rank import align_members, compare, decision_metrics, n_repeats_of  # noqa: E402
from scripts.train import evaluate_predictions, select_dev  # noqa: E402
from src.config import load_config, resolve_path, set_seed  # noqa: E402
from src.eval.metrics import dispersion, lead_time_curve, operating_point, vehicle_metrics  # noqa: E402

logger = logging.getLogger("eval_window_label")

KEY = ["vehicle_id", "cut_odo"]
LABELS = ("hard", "corrected")


def attach_panel_columns(predictions: pd.DataFrame, panel: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Las columnas del panel pegadas por `(vehicle_id, cut_odo)`; falla si alguna fila no aparea."""
    missing = [c for c in columns if c not in panel]
    if missing:
        raise KeyError(f"El panel no trae {missing}")
    right = panel[KEY + columns].copy()
    if right.duplicated(KEY).any():
        raise ValueError(f"El panel tiene pares {KEY} repetidos")
    out = predictions.drop(columns=[c for c in columns if c in predictions]).merge(
        right, on=KEY, how="left", validate="one_to_one", indicator=True
    )
    unmatched = int(out["_merge"].ne("both").sum())
    if unmatched:
        raise ValueError(f"{unmatched} fila(s) de las predicciones sin fila en el panel")
    return out.drop(columns="_merge")


def label_masks(predictions: pd.DataFrame, evaluable_column: str) -> dict[str, np.ndarray]:
    """D = todas las filas; V = las evaluables con la etiqueta corregida."""
    values = predictions[evaluable_column]
    if values.isna().any() or not set(values.unique()) <= {0, 1}:
        raise ValueError(f"`{evaluable_column}` tiene que ser 0/1 en toda fila")
    return {"hard": np.ones(len(predictions), dtype=bool), "corrected": values.to_numpy(dtype=int) == 1}


def floor_metrics_for(predictions: pd.DataFrame, column: str, eval_cfg: dict) -> dict[str, Any]:
    """El piso: score = la columna cruda (sin ajustar nada). Un NaN va por debajo de todas."""
    values = predictions[column].astype(float)
    score = values.fillna(values.min() - 1.0 if values.notna().any() else 0.0)
    # Escala monótona a [0, 1]: no toca el orden, y así `vehicle_metrics` lo acepta como score.
    span = float(score.max() - score.min()) or 1.0
    frame = predictions.assign(score=(score - score.min()) / span)
    curve = lead_time_curve(frame, n_thresholds=int(eval_cfg.get("n_thresholds", 50)),
                            k_consecutive=int(eval_cfg.get("k_consecutive", 2)))
    point = operating_point(curve, max_false_alarms_per_1000=float(eval_cfg.get("max_false_alarms_per_1000", 50)))
    return {
        "detection": [float(point["detection_rate"])] if point else [0.0],
        "n_detected": [int(point["n_detected"])] if point else [0],
        "n_event_vehicles": int(point["n_event_vehicles"]) if point else None,
        "lead_km": [float(point["median_lead_km"])] if point else [float("nan")],
        "vehicle_lift_mean": [float(vehicle_metrics(frame, how="mean", fold_column=None)["pr_auc_lift"])],
    }


def measure(predictions: pd.DataFrame, cfg: dict, *, n_repeats: int, seed: int) -> dict[str, Any]:
    """Las dos métricas que deciden, por repetición, y las informativas de la misma cuenta."""
    metrics, _ = evaluate_predictions(predictions, cfg, n_repeats=n_repeats, seed=seed)
    decision = decision_metrics(metrics)
    spread = metrics.get("when_spread") or {}
    return {
        **decision,
        "rows": int(len(predictions)),
        "positive_rows": int(predictions["label"].sum()),
        "vehicles": int(predictions["vehicle_id"].nunique()),
        "event_vehicles": int(predictions.loc[predictions["event_observed"].eq(1), "vehicle_id"].nunique()),
        "healthy_vehicles": int(predictions.loc[predictions["event_observed"].eq(0), "vehicle_id"].nunique()),
        "pr_auc": metrics["oof"]["pr_auc"],
        "pr_auc_spread": metrics["repeats_spread"]["pr_auc"],
        "brier": metrics["oof"]["brier"],
        "roc_auc": metrics["oof"]["roc_auc"],
        "when_delta": spread.get("when_delta"),
        "lead_km_spread": dispersion(decision["lead_km"]) if decision["lead_km"] else None,
    }


def rank_correlation(a: pd.DataFrame, b: pd.DataFrame, n_repeats: int) -> dict[str, list[float]]:
    """ρ de Spearman entre dos corridas por repetición, por fila y por vehículo (`mean`)."""
    by_row, by_vehicle = [], []
    for r in range(n_repeats):
        column = f"score_r{r}"
        by_row.append(float(spearmanr(a[column], b[column]).statistic))
        va = a.groupby("vehicle_id", observed=True)[column].mean()
        vb = b.groupby("vehicle_id", observed=True)[column].mean().reindex(va.index)
        by_vehicle.append(float(spearmanr(va, vb).statistic))
    return {"by_row": by_row, "by_vehicle": by_vehicle}


def diagnostics(candidate: pd.DataFrame, reference: pd.DataFrame, n_repeats: int, *,
                position: str, usage: str, early_days: float) -> dict[str, Any]:
    """Por qué ordena autos como ordena. **No preregistrado:** se agregó después del veredicto.

    - ρ de Spearman del score con la posición en días (`position`), por fila y por vehículo;
    - AUC por vehículo (`mean`) con todas las filas y solo con las de `position ≤ early_days`,
      para comparar autos en posiciones parecidas;
    - el lift por vehículo de `−usage` solo (el uso como score, sin ajustar nada).
    """
    from sklearn.metrics import roc_auc_score

    out: dict[str, Any] = {"note": "diagnóstico posterior al veredicto, no preregistrado; no decide nada",
                           "early_days": early_days}
    for name, frame in (("candidate", candidate), ("reference", reference)):
        y = frame.groupby("vehicle_id", observed=True)["event_observed"].max()
        pos = frame.groupby("vehicle_id", observed=True)[position].mean()
        early = frame.loc[frame[position].le(early_days)]
        y_early = early.groupby("vehicle_id", observed=True)["event_observed"].max()
        block: dict[str, list[float]] = {"rho_row": [], "rho_vehicle": [], "auc_vehicle": [], "auc_vehicle_early": []}
        for r in range(n_repeats):
            column = f"score_r{r}"
            means = frame.groupby("vehicle_id", observed=True)[column].mean()
            block["rho_row"].append(float(spearmanr(frame[column], frame[position], nan_policy="omit").statistic))
            block["rho_vehicle"].append(float(spearmanr(means.reindex(pos.index), pos, nan_policy="omit").statistic))
            block["auc_vehicle"].append(float(roc_auc_score(y, means.reindex(y.index))))
            early_means = early.groupby("vehicle_id", observed=True)[column].mean()
            block["auc_vehicle_early"].append(float(roc_auc_score(y_early, early_means.reindex(y_early.index))))
        out[name] = {k: float(np.mean(v)) for k, v in block.items()}
        out[name]["early_vehicles"] = {"event": int(y_early.sum()), "healthy": int((y_early == 0).sum())}
    values = candidate[usage].astype(float)
    score = -values.fillna(values.median())
    frame = candidate.assign(score=(score - score.min()) / (float(score.max() - score.min()) or 1.0))
    out["usage_floor_vehicle_lift"] = float(vehicle_metrics(frame, how="mean", fold_column=None)["pr_auc_lift"])
    return out


def beats_in_both(result: dict[str, Any]) -> bool:
    return bool(result["detection"]["wins"] and result["vehicle_lift_mean"]["wins"])


def read_a0(run_dir: Path, tolerance: float) -> dict[str, Any]:
    """(a0) del `audit.json` de `audit_model.py`: |PR-AUC del nulo − tasa base| < tolerancia."""
    path = run_dir / "audit.json"
    if not path.exists():
        return {"available": False, "passed": False, "path": str(path)}
    audit = json.loads(path.read_text(encoding="utf-8"))
    null = next((a for a in audit["audits"] if str(a.get("audit", "")).startswith("(a0)")), None)
    if null is None:
        return {"available": False, "passed": False, "path": str(path)}
    delta = float(null["pr_auc"]) - float(null["base_rate"])
    return {"available": True, "passed": bool(abs(delta) < tolerance), "pr_auc": float(null["pr_auc"]),
            "base_rate": float(null["base_rate"]), "delta": delta, "tolerance": tolerance}


def verdict(results: dict[str, Any], decides: str, floors: list[str], a0: dict[str, Any]) -> dict[str, Any]:
    """La regla del §4: gana contra la referencia, les gana a los pisos en las dos y pasa (a0)."""
    block = results[decides]
    vs_reference = block["vs_reference"]["verdict"]
    floors_ok = {f: beats_in_both(block["vs_floors"][f]) for f in floors}
    adopted = vs_reference == "gana" and all(floors_ok.values()) and a0["passed"]
    return {"decides": decides, "vs_reference": vs_reference, "beats_floors": floors_ok,
            "a0_passed": a0["passed"], "adopted": bool(adopted)}


def pct(values: list[float]) -> str:
    d = dispersion(values)
    return f"{100 * d['mean']:.1f}% ± {100 * d['std']:.1f}"


def num(values: list[float], fmt: str = "{:.3f}") -> str:
    d = dispersion(values)
    return (fmt + " ± " + fmt).format(d["mean"], d["std"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--diagnose", action="store_true",
                        help="agrega el diagnóstico posterior al veredicto (no preregistrado)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(message)s")

    cfg = load_config(args.config)
    seed = set_seed(int(cfg.get("seed", 42)))
    wcfg = cfg["window_eval"]
    decides = wcfg.get("decides", "corrected")
    if decides not in LABELS:
        raise ValueError(f"`window_eval.decides` tiene que ser uno de {LABELS}")
    floors = list(wcfg.get("floors") or [])
    evaluable = wcfg["evaluable_column"]
    experiments = resolve_path(cfg.get("output_dir", "experiments"))
    name, reference = cfg["name"], wcfg["reference"]

    frames = [pd.read_parquet(experiments / run / "predictions.parquet") for run in (name, reference)]
    candidate, ref = align_members(frames, [name, reference])
    n_repeats = n_repeats_of(candidate)
    panel = select_dev(pd.read_parquet(resolve_path(cfg["data"]["panel"])), cfg)
    diagnose_cfg = wcfg.get("diagnose") or {}
    extra = [evaluable, *[f for f in floors if f not in candidate]]
    if args.diagnose:
        extra += [c for c in (diagnose_cfg["position"], diagnose_cfg["usage"]) if c not in extra and c not in candidate]
    candidate = attach_panel_columns(candidate, panel, extra)
    ref = attach_panel_columns(ref, panel, extra)
    masks = label_masks(candidate, evaluable)

    results: dict[str, Any] = {}
    for label, mask in masks.items():
        cand_rows = candidate.loc[mask].reset_index(drop=True)
        ref_rows = ref.loc[mask].reset_index(drop=True)
        mc = measure(cand_rows, cfg, n_repeats=n_repeats, seed=seed)
        mr = measure(ref_rows, cfg, n_repeats=n_repeats, seed=seed)
        mf = {f: floor_metrics_for(cand_rows, f, cfg.get("eval", {})) for f in floors}
        results[label] = {
            "candidate": mc,
            "reference": mr,
            "floors": mf,
            "vs_reference": compare(mc, mr),
            "vs_floors": {f: compare(mc, m) for f, m in mf.items()},
            "reference_vs_floors": {f: compare(mr, m) for f, m in mf.items()},
            "rank_correlation": rank_correlation(cand_rows, ref_rows, n_repeats),
        }
        if args.diagnose:
            results[label]["diagnostics"] = diagnostics(
                cand_rows, ref_rows, n_repeats, position=diagnose_cfg["position"],
                usage=diagnose_cfg["usage"], early_days=float(diagnose_cfg["early_days"]))

    a0 = read_a0(experiments / name, float(wcfg.get("a0_tolerance", 0.02)))
    decision = verdict(results, decides, floors, a0)
    out = {"config": cfg.get("_config_path"), "candidate": name, "reference": reference,
           "evaluable_column": evaluable, "n_repeats": n_repeats, "results": results,
           "a0": a0, "decision": decision}
    path = experiments / name / "window_eval.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=float), encoding="utf-8")

    for label in LABELS:
        block = results[label]
        mc, mr = block["candidate"], block["reference"]
        print(f"\n== Etiqueta {'dura (D)' if label == 'hard' else 'corregida (V)'}"
              f"{' · DECIDE' if label == decides else ''} ==")
        print(f"filas {mc['rows']} ({mc['positive_rows']} positivas) · {mc['event_vehicles']} fallados · "
              f"{mc['healthy_vehicles']} sanos")
        rows = [
            ("candidato", pct(mc["detection"]), "/".join(map(str, mc["n_detected"])), num(mc["vehicle_lift_mean"]),
             num(mc["lead_km"], "{:.0f}"), f"{mc['pr_auc']:.4f}", f"{mc['brier']:.4f}"),
            ("referencia", pct(mr["detection"]), "/".join(map(str, mr["n_detected"])), num(mr["vehicle_lift_mean"]),
             num(mr["lead_km"], "{:.0f}"), f"{mr['pr_auc']:.4f}", f"{mr['brier']:.4f}"),
            *[(f"piso {f}", pct(m["detection"]), "/".join(map(str, m["n_detected"])),
               num(m["vehicle_lift_mean"]), num(m["lead_km"], "{:.0f}"), "", "")
              for f, m in block["floors"].items()],
        ]
        print(pd.DataFrame(rows, columns=["", "detección", "detectados", "lift veh.", "anticip. km",
                                          "PR-AUC fila", "Brier"]).to_string(index=False))
        vs = block["vs_reference"]
        for key in ("detection", "vehicle_lift_mean"):
            r = vs[key]
            print(f"  {key}: diff {r['diff']:+.4f} contra desvío combinado {r['pooled_std']:.4f} · "
                  f"pareado {[round(x, 4) for x in (r['paired_by_repeat'] or [])]}")
        print(f"  contra la referencia: {vs['verdict']} · "
              + " · ".join(f"le gana al piso {f} en las dos: {beats_in_both(block['vs_floors'][f])}" for f in floors))
        rc = block["rank_correlation"]
        print(f"  ρ con la referencia: por fila {[round(x, 2) for x in rc['by_row']]} · "
              f"por vehículo {[round(x, 2) for x in rc['by_vehicle']]}")
        if "diagnostics" in block:
            d = block["diagnostics"]
            print(f"  diagnóstico (no preregistrado): lift de −uso solo {d['usage_floor_vehicle_lift']:.3f}")
            for who in ("candidate", "reference"):
                x = d[who]
                print(f"    {who}: ρ(score, posición) fila {x['rho_row']:+.2f} · vehículo {x['rho_vehicle']:+.2f} | "
                      f"AUC veh. {x['auc_vehicle']:.3f} · con posición ≤ {d['early_days']:g} d "
                      f"{x['auc_vehicle_early']:.3f} ({x['early_vehicles']})")

    print(f"\n(a0): {a0}")
    print(f"\n== Veredicto (etiqueta {decides}) == {decision}")
    print(f"\nescrito: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
