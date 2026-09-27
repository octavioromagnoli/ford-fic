#!/usr/bin/env python
"""Comparación de todas las corridas sobre dev v2: detección por auto a varios niveles de falsas alarmas.

    python scripts/report_v2_models.py --config configs/report_v2_models.yaml

No reentrena nada: lee `predictions.parquet` (fuera de fold, R repeticiones) de cada corrida de la
lista y mide todo con la misma cuenta. Es un reporte, no una selección: el finalista sobre v2 sale
de un preregistro (CLAUDE.md, presupuesto de comparaciones).

Por corrida y repetición:

* **Curva por auto con umbral exacto.** Un auto alerta a τ si tiene `k` cortes seguidos ≥ τ
  (`vehicle_levels` de `audit_detection_null.py`, la misma regla que `lead_time_curve`). Para cada
  presupuesto `b` (fracción de autos sanos con falsa alarma) τ es el menor umbral que deja ≤ b de los
  sanos alertados; la detección es la fracción de fallados alertados. La grilla de 50 cuantiles de
  `train.py` salta mucho con ~27 cortes por sano (f9-remedicion-v2.md §7), así que acá manda el
  exacto y la grilla va al lado a 5%.
* **Nulo de tamaño de bolsa** (regla 6): el score permutado entre todas las filas, misma cuenta.
* **Fuera de muestra**: τ se fija con los sanos de los otros folds y se aplica al fold que queda
  afuera; se reporta la tasa de falsas alarmas realizada y la detección.
* **Anticipación** mediana de la primera alerta (km), con `first_alert_lead_times`.
* **AUC por auto** (score medio del auto): agrupado, dentro del mercado y dentro de mercado × motor
  (solo pares de la misma celda).
* **Bootstrap pareado por vehículo** contra la referencia (`reference`, y las de `extra_references`):
  diferencia de detección a cada presupuesto, remuestreando autos (fallados y sanos por separado) y
  re-fijando τ en cada réplica. Las semillas sueltas (`family: semilla`) no se comparan.

Pisos que no son corridas: la **tasa de la celda mercado × motor** (fuera de fold) como
score constante por auto, y el **nulo**. Diagnóstico de "¿por qué da menos que en la entrega 1?":
la detección con el monitoreo de los sanos recortado a sus primeros `bag_diag.healthy_cuts` cortes
(el tamaño de bolsa medio de los sanos en v1). `rank_mixes` promedia el rango de corridas con el piso
de celda (¿el uso le suma a la composición?); los folds son los mismos, así que siguen fuera de fold.

Deja `experiments/<output_name>/`: `summary.csv`, `curve.csv`, `paired.csv`, `report.json`, `report.md`.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from scripts.audit_detection_null import detection_from_levels, vehicle_levels  # noqa: E402
from src.config import ensure_dir, load_config, resolve_path  # noqa: E402
from src.eval.metrics import first_alert_lead_times  # noqa: E402

logger = logging.getLogger("report_v2_models")
KEY = ["vehicle_id", "cut_odo"]


# --- piezas -------------------------------------------------------------------------------

def exact_tau(healthy_levels: np.ndarray, budget: float) -> float:
    """El menor umbral que deja ≤ budget de los sanos con alerta (alerta ⇔ nivel ≥ τ)."""
    h = np.sort(healthy_levels)[::-1]
    allowed = int(np.floor(budget * len(h) + 1e-9))
    if allowed >= len(h):
        return -np.inf
    return float(np.nextafter(h[allowed], np.inf))


def detection_at(levels: np.ndarray, event: np.ndarray, budget: float) -> tuple[float, float, float]:
    tau = exact_tau(levels[event == 0], budget)
    det = float((levels[event == 1] >= tau).mean())
    fa = float((levels[event == 0] >= tau).mean())
    return det, fa, tau


def pairs_auc(score: np.ndarray, y: np.ndarray, strata: np.ndarray | None) -> float:
    """AUC con los pares (fallado, sano) de la misma celda; sin celdas, el AUC agrupado."""
    if strata is None:
        return float(roc_auc_score(y, score))
    num = den = 0.0
    for s in np.unique(strata):
        m = strata == s
        pos, neg = score[m & (y == 1)], score[m & (y == 0)]
        if len(pos) and len(neg):
            cmp = pos[:, None] - neg[None, :]
            num += (cmp > 0).sum() + 0.5 * (cmp == 0).sum()
            den += len(pos) * len(neg)
    return float(num / den) if den else float("nan")


def load_run(name: str, root: Path) -> tuple[pd.DataFrame, dict, int]:
    run_dir = root / name
    pred = pd.read_parquet(run_dir / "predictions.parquet").sort_values(KEY).reset_index(drop=True)
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    n_rep = sum(1 for c in pred.columns if c.startswith("score_r"))
    if n_rep == 0:
        pred = pred.assign(score_r0=pred["score"], fold_r0=pred.get("fold", 0))
        n_rep = 1
    return pred, metrics, n_rep


def cell_rate_scores(pred: pd.DataFrame, cells: pd.Series, r: int) -> np.ndarray:
    """Tasa de eventos de la celda mercado × motor fuera de fold (con los autos de los otros folds de la
    repetición `r`), repetida en las filas del auto: el piso de composición, sin modelo."""
    veh = pred.groupby("vehicle_id", sort=False).agg(event=("event_observed", "first"), fold=(f"fold_r{r}", "first"))
    veh["cell"] = cells.reindex(veh.index).to_numpy()
    rate = pd.Series(np.nan, index=veh.index)
    for f in veh["fold"].unique():
        train = veh[veh["fold"] != f]
        by_cell = train.groupby("cell")["event"].mean()
        held = veh["fold"] == f
        rate[held] = veh.loc[held, "cell"].map(by_cell).fillna(train["event"].mean()).to_numpy()
    return pred["vehicle_id"].map(rate).to_numpy(dtype=float)


# --- una corrida ----------------------------------------------------------------------------

def measure(name: str, pred: pd.DataFrame, n_rep: int, cfg: dict, strata: dict[str, pd.Series],
            rng: np.random.Generator) -> tuple[dict, list[dict], dict]:
    k = int(cfg["k_consecutive"])
    budgets = [float(b) for b in cfg["budgets"]]
    ids = pred["vehicle_id"].to_numpy()
    first = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1]])
    vehicles = ids[first]
    market = strata["market"].reindex(vehicles).to_numpy()
    cell = strata["cell"].reindex(vehicles).to_numpy()

    curve_rows, levels_by_rep = [], []
    auc = {"auc_vehicle": [], "auc_within_market": [], "auc_within_cell": []}
    grid5 = []
    bag = []
    for r in range(n_rep):
        s = pred[f"score_r{r}"].to_numpy(dtype=float)
        levels, event, _ = vehicle_levels(pred, s, k)
        levels_by_rep.append(levels)
        folds = pred[f"fold_r{r}"].to_numpy()[first] if f"fold_r{r}" in pred else np.zeros(len(first))
        vmean = pd.Series(s).groupby(ids).mean().reindex(vehicles).to_numpy()
        auc["auc_vehicle"].append(pairs_auc(vmean, event, None))
        auc["auc_within_market"].append(pairs_auc(vmean, event, market))
        auc["auc_within_cell"].append(pairs_auc(vmean, event, cell))
        grid5.append(detection_from_levels(levels, event, s, n_thresholds=int(cfg["n_thresholds"]),
                                           budget_per_1000=50.0)[0])
        # nulo: el score permutado entre todas las filas
        # diagnóstico de bolsa: sanos recortados a sus primeros N cortes (los fallados, enteros)
        n_cut = int(cfg["bag_diag"]["healthy_cuts"])
        pos_in_bag = pred.groupby("vehicle_id", sort=False).cumcount().to_numpy()
        keep = (pred["event_observed"].to_numpy() == 1) | (pos_in_bag < n_cut)
        short = pred[keep].reset_index(drop=True)
        lv_b, ev_b, _ = vehicle_levels(short, s[keep], k)
        bag.append([detection_at(lv_b, ev_b, b)[0] for b in budgets])
        # nulo: el score permutado entre todas las filas (con la bolsa entera y con la recortada)
        null = np.zeros((int(cfg["null_permutations"]), len(budgets)))
        null_b = np.zeros_like(null)
        for i in range(null.shape[0]):
            perm = rng.permutation(s)
            lv, ev, _ = vehicle_levels(pred, perm, k)
            null[i] = [detection_at(lv, ev, b)[0] for b in budgets]
            lv, ev, _ = vehicle_levels(short, perm[keep], k)
            null_b[i] = [detection_at(lv, ev, b)[0] for b in budgets]
        for j, b in enumerate(budgets):
            det, fa, tau = detection_at(levels, event, b)
            leads = first_alert_lead_times(pred.assign(score=s), tau, k_consecutive=k)["lead_times_km"] \
                if np.isfinite(tau) else np.array([])
            # fuera de muestra: τ con los sanos de los otros folds
            fa_o = det_o = 0
            for f in np.unique(folds):
                tau_f = exact_tau(levels[(folds != f) & (event == 0)], b)
                held = folds == f
                fa_o += int((levels[held & (event == 0)] >= tau_f).sum())
                det_o += int((levels[held & (event == 1)] >= tau_f).sum())
            curve_rows.append({
                "run": name, "repeat": r, "budget": b, "detection": det, "fa_realized": fa,
                "n_detected": int((levels[event == 1] >= tau).sum()), "n_failed": int((event == 1).sum()),
                "n_healthy": int((event == 0).sum()),
                "median_lead_km": float(np.median(leads)) if len(leads) else float("nan"),
                "null_mean": float(null[:, j].mean()), "null_p95": float(np.quantile(null[:, j], 0.95)),
                "oos_detection": det_o / max(int((event == 1).sum()), 1),
                "oos_fa": fa_o / max(int((event == 0).sum()), 1),
                "bagdiag_detection": bag[-1][j], "bagdiag_null_mean": float(null_b[:, j].mean()),
            })
    summary = {k2: float(np.mean(v)) for k2, v in auc.items()}
    summary["detection_grid_5pct"] = float(np.mean(grid5))
    summary["detection_grid_5pct_sd"] = float(np.std(grid5))
    return summary, curve_rows, {"levels": levels_by_rep, "event": event}


def paired_bootstrap(ref: dict, other: dict, budgets: list[float], n_boot: int,
                     rng: np.random.Generator) -> list[dict]:
    """Diferencia de detección (otra − referencia) con autos remuestreados, promedio sobre repeticiones."""
    event = ref["event"]
    if not np.array_equal(event, other["event"]):
        raise ValueError("las corridas no tienen los mismos autos en el mismo orden")
    fail, heal = np.flatnonzero(event == 1), np.flatnonzero(event == 0)

    def diff(fi: np.ndarray, hi: np.ndarray) -> np.ndarray:
        out = np.zeros(len(budgets))
        for la, lb in zip(ref["levels"], other["levels"]):
            ev = np.r_[np.ones(len(fi)), np.zeros(len(hi))]
            a, b_ = np.r_[la[fi], la[hi]], np.r_[lb[fi], lb[hi]]
            out += np.array([detection_at(b_, ev, b)[0] - detection_at(a, ev, b)[0] for b in budgets])
        return out / len(ref["levels"])

    point = diff(fail, heal)
    boots = np.array([diff(rng.choice(fail, len(fail)), rng.choice(heal, len(heal))) for _ in range(n_boot)])
    return [{"budget": b, "diff": float(point[j]), "ci_lo": float(np.quantile(boots[:, j], 0.025)),
             "ci_hi": float(np.quantile(boots[:, j], 0.975)), "p_le_0": float((boots[:, j] <= 0).mean())}
            for j, b in enumerate(budgets)]


def state_family(summaries: list[dict], name: str) -> str | None:
    return next((s["family"] for s in summaries if s["run"] == name), None)


def row_metrics(metrics: dict) -> dict:
    oof = metrics.get("oof", {})
    return {
        "pr_auc": oof.get("pr_auc"), "pr_auc_lift": oof.get("pr_auc_lift"), "roc_row": oof.get("roc_auc"),
        "brier": oof.get("brier"),
        "a_prime": (metrics.get("when_contribution") or {}).get("delta"),
        "lift_within_failed": (metrics.get("pr_auc_within_failed") or {}).get("lift"),
    }


def markdown_table(frame: pd.DataFrame) -> str:
    """Tabla markdown sin depender de `tabulate`."""
    def fmt(v) -> str:
        if isinstance(v, (float, np.floating)):
            return f"{v:.3f}" if np.isfinite(v) else ""
        return "" if v is None else str(v)
    lines = ["| " + " | ".join(frame.columns) + " |", "|" + "---|" * len(frame.columns)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join(lines)


# --- main -----------------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cfg = load_config(args.config)
    root = resolve_path(cfg.get("experiments_dir", "experiments"))
    out_dir = ensure_dir(root / cfg["output_name"])
    rng = np.random.default_rng(int(cfg["seed"]))
    budgets = [float(b) for b in cfg["budgets"]]

    panel = pd.read_parquet(resolve_path(cfg["strata_panel"]),
                            columns=["vehicle_id", "static_SalesCountry_cd", "aux_static_Engine"])
    per_vehicle = panel.groupby("vehicle_id").first()
    strata = {"market": per_vehicle["static_SalesCountry_cd"].astype(str),
              "cell": per_vehicle["static_SalesCountry_cd"].astype(str) + "×" + per_vehicle["aux_static_Engine"].astype(str)}

    summaries, curves, state = [], [], {}
    reference_pred = None
    for spec in cfg["runs"]:
        name = spec["run"]
        if not (root / name / "predictions.parquet").exists():
            logger.warning("falta %s: se saltea", name)
            continue
        pred, metrics, n_rep = load_run(name, root)
        if reference_pred is None:
            reference_pred = pred
        elif not reference_pred[KEY].equals(pred[KEY]):
            raise ValueError(f"{name} no tiene las mismas filas que {cfg['runs'][0]['run']}")
        summ, rows, st = measure(name, pred, n_rep, cfg, strata, rng)
        summaries.append({"run": name, "label": spec["label"], "family": spec["family"], "n_repeats": n_rep,
                          **row_metrics(metrics), **summ})
        curves += rows
        state[name] = st
        logger.info("%s listo", name)

    # piso: tasa de la celda mercado × motor fuera de fold (sin modelo)
    ref_pred = load_run(cfg["reference"], root)[0]   # los folds de sus R repeticiones
    floor_pred = ref_pred.assign(**{f"score_r{r}": cell_rate_scores(ref_pred, strata["cell"], r)
                                          for r in range(3)})
    summ, rows, st = measure("piso-celda", floor_pred, 3, cfg, strata, rng)
    summaries.append({"run": "piso-celda", "label": "Tasa de la celda mercado × motor (sin modelo)",
                      "family": "piso", "n_repeats": 3, **summ})
    curves += rows
    state["piso-celda"] = st

    # mezclas por rango con el piso de celda: ¿el uso le suma algo a la composición?
    preds = {s["run"]: load_run(s["run"], root)[0] for s in cfg["runs"] if s["run"] in state and s["run"] != "piso-celda"}
    preds["piso-celda"] = floor_pred
    for mix in cfg.get("rank_mixes", []):
        if not all(m in preds for m in mix["members"]):
            logger.warning("mezcla %s: falta un miembro, se saltea", mix["name"])
            continue
        mixed = ref_pred.copy()
        for r in range(3):
            ranks = [preds[m][f"score_r{r}"].rank(pct=True).to_numpy() for m in mix["members"]]
            mixed[f"score_r{r}"] = np.mean(ranks, axis=0)
        summ, rows, st = measure(mix["name"], mixed, 3, cfg, strata, rng)
        summaries.append({"run": mix["name"], "label": mix["label"], "family": "mezcla con celda", "n_repeats": 3, **summ})
        curves += rows
        state[mix["name"]] = st

    curve = pd.DataFrame(curves)
    agg = (curve.groupby(["run", "budget"])
           .agg(detection=("detection", "mean"), detection_sd=("detection", "std"),
                fa_realized=("fa_realized", "mean"), median_lead_km=("median_lead_km", "mean"),
                null_mean=("null_mean", "mean"), null_p95=("null_p95", "mean"),
                oos_detection=("oos_detection", "mean"), oos_fa=("oos_fa", "mean"),
                bagdiag_detection=("bagdiag_detection", "mean"),
                bagdiag_null_mean=("bagdiag_null_mean", "mean"))
           .reset_index())
    summary = pd.DataFrame(summaries)
    wide = agg.pivot(index="run", columns="budget", values="detection")
    wide.columns = [f"det_{int(round(100 * b))}" for b in wide.columns]
    summary = summary.merge(wide, left_on="run", right_index=True, how="left")

    ref = cfg["reference"]
    paired = []
    for against in [ref, *cfg.get("extra_references", [])]:
        for name, st in state.items():
            if name == against or against not in state or state_family(summaries, name) == "semilla":
                continue
            for row in paired_bootstrap(state[against], st, budgets, int(cfg["n_boot"]), rng):
                paired.append({"run": name, "reference": against, **row})
    paired = pd.DataFrame(paired)

    summary.to_csv(out_dir / "summary.csv", index=False)
    agg.to_csv(out_dir / "curve.csv", index=False)
    curve.to_csv(out_dir / "curve_by_repeat.csv", index=False)
    paired.to_csv(out_dir / "paired.csv", index=False)
    (out_dir / "report.json").write_text(json.dumps({
        "config": cfg.get("_config_path"), "budgets": budgets, "reference": ref,
        "summary": summary.to_dict(orient="records"), "curve": agg.to_dict(orient="records"),
        "paired": paired.to_dict(orient="records")}, indent=2, default=float), encoding="utf-8")

    cols = ["label", "pr_auc", "roc_row", "a_prime", "lift_within_failed", "auc_vehicle", "auc_within_cell"] + \
        [c for c in summary.columns if c.startswith("det_")]
    md = markdown_table(summary.sort_values("det_10", ascending=False)[cols])
    (out_dir / "report.md").write_text(md + "\n", encoding="utf-8")
    print(md)
    print(f"\n→ {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
