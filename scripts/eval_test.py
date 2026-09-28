#!/usr/bin/env python
"""La medición en test de un modelo ya elegido, contra su referencia (un solo tiro).

    # 1 · entrenar y puntuar (una vez por experimento; se pueden correr en paralelo)
    WANDB_MODE=disabled python scripts/eval_test.py --config configs/eval_test_f11.yaml --fit configs/exp_f11_gru_suave_conf_s101.yaml
    # 2 · medir
    python scripts/eval_test.py --config configs/eval_test_f11.yaml --report

El protocolo es el de docs/memoria/f11-preregistro-test.md. Por experimento (`--fit`):

* **Modelo final:** el `Pipeline` del experimento (preprocesado + modelo + target) entrenado con
  **todo dev**, que puntúa los cortes de test.
* **Modelos de fold:** los de la CV del experimento (sus `splits`), re-entrenados; dejan la
  predicción fuera de fold de dev y la de cada uno sobre test. Sirven para fijar el umbral en dev
  y aplicarlo en test (punto de operación fuera de muestra).

El reporte (`--report`) mide con la cuenta de `report_v2_models.py`: detección por auto con
umbral exacto a cada presupuesto de sanos **de test** con falsa alarma, el primario y el bootstrap
pareado contra la referencia, el nulo que conserva la bolsa, el piso de la celda mercado × motor
(la tasa de dev aplicada a test) y todo de nuevo sin los autos de test que estaban en el dev viejo.
Deja `experiments/<output_name>/`: `curve.csv`, `paired.csv`, `operating.csv`, `report.json`.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from scripts.audit_detection_null import vehicle_levels  # noqa: E402
from scripts.report_v2_models import detection_at, exact_tau, paired_bootstrap  # noqa: E402
from src.config import ensure_dir, load_config, resolve_path  # noqa: E402
from src.eval.metrics import first_alert_lead_times  # noqa: E402
from src.eval.splits import iter_repeats, load_splits, load_test_split, test_split_masks  # noqa: E402
from src.models.registry import get_model  # noqa: E402
from src.training.cv import build_preprocessor, select_feature_columns  # noqa: E402
from src.training.targets import build_target  # noqa: E402

logger = logging.getLogger("eval_test")
KEY = ["vehicle_id", "cut_odo"]
CARRY = KEY + ["event_observed", "label", "time_to_event_km"]


# --- 1 · entrenar y puntuar -------------------------------------------------------------------

def load_dev_test(exp: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    panel = pd.read_parquet(resolve_path(exp["data"]["panel"]))
    split = load_test_split(exp["splits"]["test_split"])
    dev_mask, test_mask = test_split_masks(panel, split)
    dev = panel.loc[dev_mask].sort_values(KEY).reset_index(drop=True)
    test = panel.loc[test_mask].sort_values(KEY).reset_index(drop=True)
    return dev, test, split


def fit_pipeline(exp: dict, frame: pd.DataFrame, train_mask: np.ndarray, X: pd.DataFrame) -> Pipeline:
    target = exp.get("target") or {}
    y = (build_target(target["name"], frame, train_mask, **(target.get("params") or {})).y
         if target else frame.loc[train_mask, "label"].astype(int).to_numpy())
    pipe = Pipeline([("prep", build_preprocessor(X)),
                     ("model", get_model(exp["model"]["name"], dict(exp["model"].get("params") or {})))])
    pipe.fit(X.loc[train_mask], y)
    return pipe


def fit(exp_path: str, out_dir: Path) -> None:
    exp = load_config(exp_path)
    dev, test, _ = load_dev_test(exp)
    features = select_feature_columns(dev)
    X_dev, X_test = dev[features], test[features]
    logger.info("%s | dev %d filas / %d autos | test %d filas / %d autos", exp["name"], len(dev),
                dev["vehicle_id"].nunique(), len(test), test["vehicle_id"].nunique())

    test_out = test[CARRY].copy()
    final = fit_pipeline(exp, dev, np.ones(len(dev), dtype=bool), X_dev)
    test_out["score_final"] = final.predict_proba(X_test)[:, 1]
    logger.info("%s | modelo final listo", exp["name"])

    dev_out = dev[CARRY].copy()
    splits = load_splits(resolve_path(exp["splits"]["path"]))
    for repeat, masks in iter_repeats(dev, splits, strict=True,
                                      min_valid_positives=exp["splits"].get("min_valid_positives")):
        oof = np.full(len(dev), np.nan)
        for fold, train_mask, valid_mask in masks:
            pipe = fit_pipeline(exp, dev, train_mask, X_dev)
            oof[valid_mask] = pipe.predict_proba(X_dev.loc[valid_mask])[:, 1]
            test_out[f"score_r{repeat}_f{fold}"] = pipe.predict_proba(X_test)[:, 1]
            logger.info("%s | rep %d fold %d listo", exp["name"], repeat, fold)
        dev_out[f"oof_r{repeat}"] = oof
    test_out.to_parquet(out_dir / f"test_{exp['name']}.parquet", index=False)
    dev_out.to_parquet(out_dir / f"devoof_{exp['name']}.parquet", index=False)


# --- 2 · medir ----------------------------------------------------------------------------------

def rank(x: np.ndarray) -> np.ndarray:
    return pd.Series(x).rank(pct=True).to_numpy()


def curve_row(frame: pd.DataFrame, score: np.ndarray, k: int, budgets: list[float]) -> tuple[list[float], dict]:
    levels, event, _ = vehicle_levels(frame, score, k)
    return [detection_at(levels, event, b)[0] for b in budgets], {"levels": [levels], "event": event}


def null_band(frame: pd.DataFrame, score: np.ndarray, k: int, budgets: list[float], n: int,
              rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    draws = np.array([curve_row(frame, rng.permutation(score), k, budgets)[0] for _ in range(n)])
    return draws.mean(axis=0), np.quantile(draws, 0.95, axis=0)


def cell_rate(dev: pd.DataFrame, test: pd.DataFrame, cell_columns: list[str]) -> np.ndarray:
    """La tasa de fallas por celda con los autos de dev, repetida en las filas de test (sin modelo)."""
    def cells(frame: pd.DataFrame) -> pd.Series:
        per = frame.groupby("vehicle_id")[cell_columns + ["event_observed"]].first()
        return per[cell_columns].astype(str).agg("×".join, axis=1), per["event_observed"]
    dev_cell, dev_event = cells(dev)
    rate = dev_event.groupby(dev_cell).mean()
    test_cell, _ = cells(test)
    per_vehicle = test_cell.map(rate).fillna(dev_event.mean())
    return test["vehicle_id"].map(per_vehicle).to_numpy(dtype=float)


def operating_point(dev_frames: list[pd.DataFrame], test_frames: list[pd.DataFrame], k: int,
                    budgets: list[float]) -> list[dict]:
    """τ fijado con los sanos de dev fuera de fold, aplicado a lo que cada modelo de fold puntúa en test.

    Las semillas se promedian en probabilidad (mismo fold ⇒ mismo train). Detección y falsa alarma
    realizada se promedian sobre los modelos de fold.
    """
    dev0, test0 = dev_frames[0], test_frames[0]
    repeats = sorted({int(c.split("_r")[1]) for c in dev0.columns if c.startswith("oof_r")})
    rows = []
    for b in budgets:
        det, fa = [], []
        for r in repeats:
            dev_score = np.mean([d[f"oof_r{r}"].to_numpy() for d in dev_frames], axis=0)
            lv_dev, ev_dev, _ = vehicle_levels(dev0, dev_score, k)
            tau = exact_tau(lv_dev[ev_dev == 0], b)
            folds = sorted(c for c in test0.columns if c.startswith(f"score_r{r}_f"))
            for col in folds:
                test_score = np.mean([t[col].to_numpy() for t in test_frames], axis=0)
                lv, ev, _ = vehicle_levels(test0, test_score, k)
                det.append(float((lv[ev == 1] >= tau).mean()))
                fa.append(float((lv[ev == 0] >= tau).mean()))
        rows.append({"budget": b, "detection": float(np.mean(det)), "detection_sd": float(np.std(det)),
                     "fa_realized": float(np.mean(fa)), "fa_sd": float(np.std(fa)), "n_models": len(det)})
    return rows


def report(cfg: dict, out_dir: Path) -> None:
    k = int(cfg["k_consecutive"])
    budgets = [float(b) for b in cfg["budgets"]]
    primary = [float(b) for b in cfg["primary_budgets"]]
    rng = np.random.default_rng(int(cfg["seed"]))
    groups = cfg["groups"]

    tests, devs = {}, {}
    for group, exp_paths in groups.items():
        names = [load_config(p)["name"] for p in exp_paths]
        tests[group] = [pd.read_parquet(out_dir / f"test_{n}.parquet") for n in names]
        devs[group] = [pd.read_parquet(out_dir / f"devoof_{n}.parquet") for n in names]
    base = next(iter(tests.values()))[0]
    dev_base = next(iter(devs.values()))[0]
    if not all(f[KEY].equals(base[KEY]) for frames in tests.values() for f in frames) or \
            not all(f[KEY].equals(dev_base[KEY]) for frames in devs.values() for f in frames):
        raise ValueError("los experimentos no puntuaron las mismas filas")

    exp0 = load_config(next(iter(groups.values()))[0])
    dev_panel, test_panel, split = load_dev_test(exp0)
    scores = {g: np.mean([rank(t["score_final"].to_numpy()) for t in frames], axis=0) for g, frames in tests.items()}
    scores["piso-celda"] = cell_rate(dev_panel, test_panel, cfg["cell_columns"])

    seen = set((split.get("previous") or {}).get("test_vehicles_in_old_dev", []))
    subsets = {"test": np.ones(len(base), dtype=bool),
               "test_sin_vistos": ~base["vehicle_id"].isin(seen).to_numpy()}

    curve, paired, summary = [], [], {}
    for subset, mask in subsets.items():
        frame = base.loc[mask].reset_index(drop=True)
        n_fail = int(frame.groupby("vehicle_id")["event_observed"].first().sum())
        n_veh = int(frame["vehicle_id"].nunique())
        summary[subset] = {"vehicles": n_veh, "failed": n_fail, "healthy": n_veh - n_fail}
        states = {}
        for name, score in scores.items():
            det, state = curve_row(frame, score[mask], k, budgets)
            states[name] = state
            null_mean, null_p95 = null_band(frame, score[mask], k, budgets, int(cfg["null_permutations"]), rng)
            for b, d, nm, n95 in zip(budgets, det, null_mean, null_p95):
                curve.append({"subset": subset, "model": name, "budget": b, "detection": d,
                              "null_mean": nm, "null_p95": n95})
            curve.append({"subset": subset, "model": name, "budget": "primary",
                          "detection": float(np.mean([d for b, d in zip(budgets, det) if b in primary]))})
        comparisons = [(cfg["candidate"], cfg["reference"]), (cfg["candidate"], "piso-celda"),
                       (cfg["reference"], "piso-celda")]
        for name, against in comparisons:
            for row in paired_bootstrap(states[against], states[name], budgets, int(cfg["n_boot"]), rng, primary):
                paired.append({"subset": subset, "model": name, "reference": against, **row})
        tau10 = exact_tau(states[cfg["candidate"]]["levels"][0][states[cfg["candidate"]]["event"] == 0], 0.10)
        leads = first_alert_lead_times(frame.assign(score=scores[cfg["candidate"]][mask]), tau10,
                                       k_consecutive=k)["lead_times_km"]
        summary[subset]["candidate_median_lead_km_at_10"] = float(np.median(leads)) if len(leads) else None

    operating = []
    for group in groups:
        for row in operating_point(devs[group], tests[group], k, [float(b) for b in cfg["operating_budgets"]]):
            operating.append({"model": group, **row})

    curve, paired, operating = pd.DataFrame(curve), pd.DataFrame(paired), pd.DataFrame(operating)
    curve.to_csv(out_dir / "curve.csv", index=False)
    paired.to_csv(out_dir / "paired.csv", index=False)
    operating.to_csv(out_dir / "operating.csv", index=False)
    (out_dir / "report.json").write_text(json.dumps(
        {"config": cfg.get("_config_path"), "summary": summary, "curve": curve.to_dict(orient="records"),
         "paired": paired.to_dict(orient="records"), "operating": operating.to_dict(orient="records")},
        indent=2, default=float), encoding="utf-8")

    for subset, info in summary.items():
        print(f"\n== {subset}: {info}")
        wide = curve[curve["subset"] == subset].pivot(index="model", columns="budget", values="detection")
        print((100 * wide).round(1).to_string())
        nulls = curve[(curve["subset"] == subset) & (curve["budget"] != "primary")]
        print("nulo p95:", (100 * nulls.groupby("budget")["null_p95"].mean()).round(1).to_dict())
        sub = paired[paired["subset"] == subset].copy()
        sub[["diff", "ci_lo", "ci_hi"]] = (100 * sub[["diff", "ci_lo", "ci_hi"]]).round(1)
        print(sub.drop(columns="subset").to_string(index=False))
    print("\n== punto de operación fijado en dev (modelos de fold)")
    print(operating.round(3).to_string(index=False))
    print(f"\n→ {out_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--fit", default=None, help="YAML del experimento a entrenar y puntuar")
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cfg = load_config(args.config)
    out_dir = ensure_dir(resolve_path(cfg.get("experiments_dir", "experiments")) / cfg["output_name"])
    if args.fit:
        fit(args.fit, out_dir)
    if args.report:
        report(cfg, out_dir)
    if not (args.fit or args.report):
        parser.error("pedí --fit <yaml> o --report")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
