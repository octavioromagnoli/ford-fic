#!/usr/bin/env python
"""Ajusta la incidencia con el conjunto externo y la congela (F5 §3.2, preregistro §4 y §6-I1).

    python scripts/build_external_panel.py --config configs/data/panel_external_incidence.yaml
    python scripts/fit_external_incidence.py --config configs/exp_ext_incidence_a30.yaml
    python scripts/train.py --config configs/exp_ext_incidence_a30.yaml       # A-solo sobre dev

Lee el bloque `source_fit` del YAML del experimento. Todo lo que hace es con la fuente;
dev no se abre:

1. **G1 · qué es "fallado" sin fecha** (descriptiva). Por mercado, la razón fallados/sanos de
   los mensajes de filtro por 1.000 km sobre toda la telemetría post-venta, y los mismos
   números para los fallados con fecha por defecto de los mercados con fecha. Los km salen
   del odómetro de `signals`.
2. **G2 · el rasgo temprano apunta igual** (para). AUC del índice de pesos unitarios `s`
   entre fallados y sanos elegibles, con IC95 por bootstrap. Si el límite inferior no
   supera 0,5, escribe el informe con `stop` y no congela nada.
3. **El ajuste** (`src/models/incidence.py`), congelado en `artifact`, y sus pesos por
   covariable estandarizada en `weights` (los usa I2).
4. **I1 · la fuente por dentro:** pendientes con IC por bootstrap, encogimiento heurístico,
   AUC con CV interna, el piso de calendario en la fuente y las pendientes con otras
   elegibilidades.

El informe queda en `experiments/<name>/source_fit.json`.
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

from scripts.build_landmark_panel import _json_default  # noqa: E402
from src.config import ensure_dir, load_config, resolve_path, set_seed  # noqa: E402
from src.data.anchor import EPOCH  # noqa: E402
from src.data.external import reproduce_parent_split, select_external  # noqa: E402
from src.data.join import load_vehicle_static  # noqa: E402
from src.data.subset import read_table_for_vehicles  # noqa: E402
from src.eval.metrics import roc_auc  # noqa: E402
from src.eval.splits import load_test_split  # noqa: E402
from src.features.signals import derive_signal_columns  # noqa: E402
from src.models.cure import fit_logistic  # noqa: E402
from src.models.incidence import _strata, design_matrix, fit_external_incidence, save_incidence  # noqa: E402

logger = logging.getLogger("fit_external_incidence")
ID = "vehicle_id"


# --------------------------------------------------------------------------- #
# G1 · mensajes de filtro por mercado (descriptiva)
# --------------------------------------------------------------------------- #
def g1_messages(spec: dict[str, Any], data_cfg: dict[str, Any]) -> dict[str, Any]:
    """Tasa de mensajes de filtro post-venta por mercado × cohorte, en la fuente y en los fechados por defecto de CNTRY_3/4."""
    cache = resolve_path(spec["cache"])
    messages = list(spec["messages"])
    if cache.exists():
        per_vehicle = pd.read_parquet(cache)
    else:
        split = load_test_split(data_cfg["test_split"])
        static = load_vehicle_static(config_path=data_cfg["sources"], dedupe_config=data_cfg["dedupe"],
                                     panel_config=data_cfg["panel_config"])
        parent = reproduce_parent_split(static, split)
        everyone, _ = select_external(static, split, parent, side=data_cfg["external"]["side"], markets=None)
        keep = (everyone["static_SalesCountry_cd"].isin(data_cfg["external"]["markets"])
                | everyone["default_event_date"])
        everyone = everyone.loc[keep].set_index(ID)
        meta = json.loads(resolve_path(data_cfg["anchor_meta"]).read_text(encoding="utf-8"))
        origin = float(meta["anchor"]["origin_day_since_epoch"])
        sale = (EPOCH + pd.to_timedelta(origin + pd.to_numeric(everyone["static_ProductionDay"])
                                        + pd.to_numeric(everyone["static_daysUntilSale"]), unit="D"))
        raw, _ = read_table_for_vehicles("signals", None, set(everyone.index.astype(str)), sources=data_cfg["sources"],
                                         dedupe=data_cfg["dedupe"],
                                         chunksize=int((data_cfg.get("scan") or {}).get("chunksize", 500_000)))
        signals, _ = derive_signal_columns(raw)
        signals = signals.loc[signals["eventTimestamp"] > signals[ID].map(sale)]
        grouped = signals.groupby(ID)
        per_vehicle = pd.DataFrame({
            "km_post_sale": grouped["OdometerValue"].max() - grouped["OdometerValue"].min(),
            **{f"n_{m}": grouped[f"msg_{m}"].sum() for m in messages},
        }).reindex(everyone.index)
        per_vehicle["market"] = everyone["static_SalesCountry_cd"].astype(str)
        per_vehicle["event_observed"] = everyone["event_observed"].astype(int)
        per_vehicle["default_event_date"] = everyone["default_event_date"].astype(int)
        per_vehicle = per_vehicle.reset_index()
        ensure_dir(cache.parent)
        per_vehicle.to_parquet(cache, index=False)

    in_source = per_vehicle["market"].isin(spec["source_markets"])
    groups = {
        **{f"{m}": per_vehicle.loc[per_vehicle["market"].eq(m)] for m in spec["source_markets"]},
        "fuente": per_vehicle.loc[in_source],
        "CNTRY_3/4 fecha por defecto": per_vehicle.loc[~in_source],
    }
    rows = []
    for name, frame in groups.items():
        frame = frame.loc[frame["km_post_sale"].gt(0)]
        for event, part in frame.groupby("event_observed"):
            row = {"grupo": name, "cohorte": "fallado" if event else "sano", "n": int(len(part)),
                   "km_post_venta_mediana": float(part["km_post_sale"].median())}
            for m in messages:
                row[f"{m}_por_1000km"] = float(part[f"n_{m}"].sum() / part["km_post_sale"].sum() * 1000)
            rows.append(row)
    table = pd.DataFrame(rows)
    ratios = {}
    for name in [*spec["source_markets"], "fuente"]:
        part = table.loc[table["grupo"].eq(name)].set_index("cohorte")
        if {"fallado", "sano"} <= set(part.index):
            ratios[name] = {m: float(part.loc["fallado", f"{m}_por_1000km"] / part.loc["sano", f"{m}_por_1000km"])
                            for m in messages}
    flagged = [name for name, r in ratios.items() if name != "fuente" and all(v <= 1 for v in r.values())]
    return {"table": table.to_dict(orient="records"), "ratio_failed_over_healthy": ratios,
            "flagged_markets": flagged, "f1_reference_pooled": spec.get("f1_reference")}


# --------------------------------------------------------------------------- #
# G2 e I1
# --------------------------------------------------------------------------- #
def bootstrap_auc(y: np.ndarray, score: np.ndarray, n_boot: int, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max():
            continue
        draws.append(roc_auc(y[idx], score[idx]))
    draws = np.asarray(draws)
    return {"auc": float(roc_auc(y, score)), "lo": float(np.quantile(draws, 0.025)),
            "hi": float(np.quantile(draws, 0.975)), "n_boot": int(draws.size)}


def source_design(model, panel: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Las z de la fuente con el pipeline ajustado y la matriz de diseño (sin estratos)."""
    features = panel[[c for c in panel.columns if c.startswith(("feat_", "static_"))]]
    z = model.standardized(features)
    return z, model.design_matrix(z)


def gate_g2(model, panel: pd.DataFrame, spec: dict[str, Any], seed: int) -> dict[str, Any]:
    rows = panel["aux_eligible"].eq(1).to_numpy()
    y = panel["label"].to_numpy(dtype=int)[rows]
    z, _ = source_design(model, panel)
    z = z.loc[rows]
    fleet = [c for c in model.covariates if c != model.usage_column]
    s = z[fleet].to_numpy() @ np.asarray([model.signs[c] for c in fleet])
    n_boot = int(spec.get("n_boot", 2000))
    out = {"index_s": bootstrap_auc(y, s, n_boot, seed)}
    out["by_feature_signed"] = {c: float(roc_auc(y, model.signs[c] * z[c].to_numpy())) for c in fleet}
    out["minus_usage"] = float(roc_auc(y, -z[model.usage_column].to_numpy()))
    markets = panel.loc[rows, "static_SalesCountry_cd"].astype(str).to_numpy()
    out["index_s_by_market"] = {m: {"auc": float(roc_auc(y[markets == m], s[markets == m])),
                                    "n_failed": int(y[markets == m].sum()), "n_healthy": int((y[markets == m] == 0).sum())}
                                for m in sorted(set(markets))}
    out["passed"] = bool(out["index_s"]["lo"] > 0.5)
    return out


def refit_slopes(X: np.ndarray, y: np.ndarray, n_strata: int, firth: bool) -> np.ndarray:
    return fit_logistic(X, y, firth=firth).coef[n_strata:]


def informative_i1(model, panel: pd.DataFrame, fit_kwargs: dict[str, Any], spec: dict[str, Any], seed: int) -> dict[str, Any]:
    rows = panel["aux_eligible"].eq(1).to_numpy()
    y_all = panel["label"].to_numpy(dtype=int)
    _, d = source_design(model, panel)
    strata = _strata(panel["static_SalesCountry_cd"], model.market_levels) if set(
        panel.loc[rows, "static_SalesCountry_cd"].astype(str)) <= set(model.market_levels) else None
    X = np.hstack([strata, d])[rows]
    y = y_all[rows]
    k = len(model.market_levels)
    out: dict[str, Any] = {"slopes": dict(zip(model.slope_names, map(float, model.slopes))),
                           "intercepts_by_market": model.intercepts}

    # Pendientes con IC por bootstrap de vehículos (pipeline de features fijo).
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(int(spec.get("n_boot_coef", 1000))):
        idx = rng.integers(0, len(y), len(y))
        try:
            draws.append(refit_slopes(X[idx], y[idx], k, fit_kwargs["firth"]))
        except np.linalg.LinAlgError:
            continue
    draws = np.asarray(draws)
    out["slopes_bootstrap"] = {name: {"lo": float(np.quantile(draws[:, j], 0.025)),
                                      "hi": float(np.quantile(draws[:, j], 0.975)),
                                      "frac_positive": float((draws[:, j] > 0).mean())}
                               for j, name in enumerate(model.slope_names)}
    out["slopes_bootstrap_n"] = int(len(draws))

    # Encogimiento heurístico (van Houwelingen & le Cessie 1990), con verosimilitudes sin penalizar.
    full = fit_logistic(X, y, firth=False)
    null = fit_logistic(X[:, :k], y, firth=False)
    chi2 = 2.0 * (full.objective - null.objective)
    df = X.shape[1] - k
    out["heuristic_shrinkage"] = {"lr_chi2": float(chi2), "df": int(df),
                                  "shrinkage": float((chi2 - df) / chi2) if chi2 > 0 else float("nan")}

    # AUC con CV interna: todo el pipeline dentro del fold, estratificada por mercado × etiqueta.
    cv = spec.get("cv") or {}
    from sklearn.model_selection import RepeatedStratifiedKFold

    eligible_idx = np.flatnonzero(rows)
    strata_label = (panel["static_SalesCountry_cd"].astype(str) + "_" + panel["label"].astype(str)).to_numpy()
    splitter = RepeatedStratifiedKFold(n_splits=int(cv.get("n_splits", 5)), n_repeats=int(cv.get("n_repeats", 3)),
                                       random_state=int(cv.get("seed", seed)))
    oof = np.full((int(cv.get("n_repeats", 3)), len(panel)), np.nan)
    n_splits = int(cv.get("n_splits", 5))
    for i, (train_e, valid_e) in enumerate(splitter.split(eligible_idx, strata_label[eligible_idx])):
        repeat = i // n_splits
        valid_rows = eligible_idx[valid_e]
        train_mask = np.ones(len(panel), dtype=bool)
        train_mask[valid_rows] = False          # los no elegibles del train quedan para la referencia
        train = panel.loc[train_mask]
        fold_model = fit_external_incidence(train, **fit_kwargs)
        oof[repeat, valid_rows] = fold_model.linear_predictor(
            panel.iloc[valid_rows][[c for c in panel.columns if c.startswith(("feat_", "static_"))]])
    aucs = [roc_auc(y_all[eligible_idx], oof[r, eligible_idx]) for r in range(oof.shape[0])]
    out["internal_cv_auc"] = {"mean": float(np.mean(aucs)), "std": float(np.std(aucs)), "by_repeat": aucs}
    out["apparent_auc"] = float(roc_auc(y, d[rows] @ model.slopes))

    # ¿Hay atajo de calendario en la fuente? El piso de producción entre los elegibles.
    elig = panel.loc[rows]
    out["calendar_floor_auc"] = {
        "minus_production_day": float(roc_auc(y, -elig["aux_static_ProductionDay"].to_numpy(float))),
        "minus_sale_day": float(roc_auc(y, -elig["aux_sale_day"].to_numpy(float))),
    }

    # Pendientes con otras elegibilidades (el pipeline se reajusta entero).
    sensitivity = {}
    label = panel["label"].to_numpy(dtype=int)
    for name in spec.get("eligibility_sensitivity", []):
        mask = (panel["aux_eligible"].eq(1) | panel["label"].eq(1)).to_numpy() if name == "all_positives" else \
            panel[name].eq(1).to_numpy()
        variant = fit_external_incidence(panel, **{**fit_kwargs, "eligible": mask})
        sensitivity[name] = {"n_rows": int(mask.sum()), "n_events": int(label[mask].sum()),
                             "slopes": dict(zip(variant.slope_names, map(float, variant.slopes)))}
    out["eligibility_sensitivity"] = sensitivity
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    for noisy in ("src.training.transformers", "src.models.incidence", "src.data.subset"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    cfg = load_config(args.config)
    seed = set_seed(int(cfg.get("seed", 42)))
    spec = cfg["source_fit"]
    data_cfg = load_config(spec["data_config"])
    panel = pd.read_parquet(resolve_path(spec["panel"]))
    fit_kwargs = {
        "fleet_signs": dict(spec["fleet_signs"]), "usage_column": spec["usage_column"], "design": spec["design"],
        "normalizer_params": dict(spec.get("normalizer") or {}), "firth": bool(spec.get("firth", True)),
    }
    out_dir = resolve_path(cfg.get("output_dir", "experiments")) / cfg["name"]
    ensure_dir(out_dir)
    report: dict[str, Any] = {"run": cfg["name"], "design": spec["design"], "panel": spec["panel"],
                              "n_rows": int(len(panel)), "n_eligible": int(panel["aux_eligible"].sum()),
                              "n_events_eligible": int(panel.loc[panel["aux_eligible"].eq(1), "label"].sum())}

    model = fit_external_incidence(panel, **fit_kwargs)
    report["reference_levels_used"] = model.info["reference_levels_used"]
    gates = spec.get("gates") or {}
    if "g1" in gates:
        report["G1"] = g1_messages(gates["g1"], data_cfg)
    if "g2" in gates:
        report["G2"] = gate_g2(model, panel, gates["g2"], seed)
        if not report["G2"]["passed"]:
            report["stop"] = ("G2: el índice del rasgo temprano no separa fallados de sanos en la fuente "
                              "(IC95 de la AUC toca 0,5). Se para y no se congela nada (preregistro §4).")
    if "stop" not in report:
        report["model"] = {"slope_names": model.slope_names, "slopes": model.slopes.tolist(),
                           "intercepts": model.intercepts, "weights": model.weights(), "info": model.info}
        artifact = save_incidence(model, resolve_path(spec["output"]["artifact"]))
        weights_path = resolve_path(spec["output"]["weights"])
        weights_path.write_text(json.dumps(model.weights(), indent=2), encoding="utf-8")
        report["artifact"] = str(spec["output"]["artifact"])
        report["weights_path"] = str(spec["output"]["weights"])
        if spec.get("informative"):
            report["I1"] = informative_i1(model, panel, fit_kwargs, spec["informative"], seed)

    out_path = out_dir / "source_fit.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    print(json.dumps(report, indent=1, ensure_ascii=False, default=_json_default)[:7000])
    logger.info("Informe: %s", out_path)

    wandb_cfg = cfg.get("wandb") or {}
    if wandb_cfg.get("enabled", True):
        import wandb

        from scripts.audit_cure import flatten

        run = wandb.init(project=wandb_cfg.get("project", "ford-fic"),
                         entity=os.environ.get("WANDB_ENTITY") or wandb_cfg.get("entity"),
                         name=f"source-{cfg['name']}", group=wandb_cfg.get("group"), job_type="source_fit",
                         tags=[*(wandb_cfg.get("tags") or []), "source"],
                         mode=os.environ.get("WANDB_MODE") or wandb_cfg.get("mode", "online"),
                         config={"experiment": cfg["name"], "source_fit": spec})
        flat: dict[str, float] = {}
        flatten("", {k: v for k, v in report.items() if k not in ("run", "design", "panel")}, flat)
        run.log(flat)
        run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
