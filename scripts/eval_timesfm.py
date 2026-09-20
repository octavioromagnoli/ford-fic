#!/usr/bin/env python
"""TimesFM zero-shot en los cortes del panel v1. No pasa por `train.py`: no hay nada que entrenar.

    python scripts/eval_timesfm.py --config configs/exp_timesfm3_zeroshot.yaml

Por qué es un script aparte y no un builder del registry: ver el docstring de
`src/models/timesfm_zeroshot.py`. Lo que comparte con `train.py`:

* **Los cortes son los del panel v1** (`panel_config`): universo de 364, etiqueta,
  gap, horizonte, QC y sanos emparejados por odómetro y mes. Nada de eso se decide acá.
* **Solo dev se mide.** El recorte es `test_split_masks()` con el holdout de
  `splits.test_split`; los folds salen de `iter_folds()` sobre el `splits.json` del
  panel. Del test no se calcula ninguna métrica.
* **Mismas métricas** (`src/eval/metrics.py`) y mismo formato en `experiments/<run>/`.

Lo que sí se calcula para todas las filas del panel, test incluido, es el pronóstico
(`forecasts.parquet`): no ajusta nada y no mira la etiqueta, igual que una feature de
ventana. Es el insumo de `scripts/build_timesfm_panel.py`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_dataset import SIGNAL_COLUMNS, TRIP_COLUMNS  # noqa: E402  (mismas columnas que el panel)
from scripts.train import init_wandb  # noqa: E402
from src.config import ensure_dir, load_config, repo_root, resolve_path, set_seed  # noqa: E402
from src.data.subset import read_table_for_vehicles  # noqa: E402
from src.eval.metrics import (  # noqa: E402
    classification_metrics,
    lead_time_curve,
    operating_point,
    summarize_folds,
)
from src.eval.splits import iter_folds, load_splits, load_test_split, test_split_masks  # noqa: E402
from src.features.signals import derive_signal_columns  # noqa: E402
from src.features.trips import derive_trip_columns  # noqa: E402
from src.models.timesfm_zeroshot import ID_COL, build_km_series, load_forecaster, score_cuts  # noqa: E402

logger = logging.getLogger("eval_timesfm")

# Subirlo cada vez que cambia cómo se construyen las series: invalida el cache.
SERIES_VERSION = 3
KEY = [ID_COL, "cut_odo"]
LABEL_COLUMNS = ["label", "time_to_event_km", "event_observed"]
SCORES = (("timesfm", "score"), ("naive", "score_naive"), ("km", "score_km"))


def load_series(cfg: dict, panel_cfg: dict, vehicles: set[str]) -> pd.DataFrame:
    """Series por km de `vehicles`, cacheadas: leer 13M de filas tarda."""
    series_cfg = cfg["series"]
    bin_km = float(series_cfg["bin_km"])
    slugs = list(series_cfg["bad_messages"])
    # La clave incluye el set de vehículos y los mensajes: un cache nunca se confunde con otro.
    key = hashlib.sha256(("|".join(sorted(vehicles)) + "#" + ",".join(slugs)).encode()).hexdigest()[:10]
    path = resolve_path(series_cfg["cache_dir"]) / f"v{SERIES_VERSION}_bin{int(bin_km)}_{key}" / "series.parquet"
    if path.exists():
        logger.info("Series desde cache: %s", path.parent)
        return pd.read_parquet(path)

    spec_cfg = load_config(panel_cfg["features"]["spec"])
    common = dict(sources=panel_cfg["sources"], dedupe=panel_cfg["dedupe"],
                  chunksize=int(series_cfg.get("chunksize", 500_000)))
    trips, _ = read_table_for_vehicles("trips", None, vehicles, **common)
    trips, _ = derive_trip_columns(trips[[c for c in trips.columns if c in {ID_COL, *TRIP_COLUMNS}]],
                                   thresholds=spec_cfg.get("thresholds"), clip=spec_cfg.get("clip"))
    signals, _ = read_table_for_vehicles("signals", None, vehicles, **common)
    signals, _ = derive_signal_columns(signals[[c for c in signals.columns if c in {ID_COL, *SIGNAL_COLUMNS}]])
    series = build_km_series(trips, signals, bin_km=bin_km, bad_slugs=slugs)
    ensure_dir(path.parent)
    series.to_parquet(path, index=False)
    return series


def ranking_metrics(y: pd.Series, score: pd.Series) -> dict[str, float]:
    """`classification_metrics` sobre el rango percentil del score.

    El score es un pronóstico (ej.: regeneraciones esperadas), no una probabilidad.
    El rango es monótono, así que PR-AUC y ROC-AUC no cambian; el Brier sí dependería
    de una calibración que no hay, y se reporta como NaN en vez de un número engañoso.
    """
    metrics = classification_metrics(y, score.rank(pct=True))
    metrics["brier"] = float("nan")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="TimesFM zero-shot en los cortes del panel v1")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--reuse-forecasts", action="store_true",
                        help="Recalcula métricas desde experiments/<run>/forecasts.parquet sin pronosticar")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    cfg = load_config(args.config)
    seed = set_seed(int(cfg.get("seed", 42)))
    panel_cfg = load_config(cfg["panel_config"])
    model_cfg, score_cfg = cfg["model"], cfg["score"]

    splits_cfg = cfg.get("splits", {})
    if "test_split" not in splits_cfg or splits_cfg["test_split"] is None:
        raise KeyError("El config tiene que declarar `splits.test_split` con el holdout (CLAUDE.md, regla 2).")
    panel = pd.read_parquet(resolve_path(panel_cfg["output"]["panel"]))
    dev_mask, test_mask = test_split_masks(panel, load_test_split(splits_cfg["test_split"]), strict=True)
    logger.info("Panel: %d filas · dev %d · test %d (se pronostica, no se mide)",
                len(panel), int(dev_mask.sum()), int(test_mask.sum()))

    run_name = args.run_name or cfg.get("name") or f"timesfm-{datetime.now():%Y%m%d-%H%M%S}"
    out_dir = ensure_dir(Path(resolve_path(cfg.get("output_dir", "experiments"))) / run_name)
    forecasts_path = out_dir / "forecasts.parquet"
    if args.reuse_forecasts:
        forecasts = pd.read_parquet(forecasts_path)
    else:
        series = load_series(cfg, panel_cfg, set(panel[ID_COL].astype(str)))
        terna = panel[["window_km", "gap_km", "horizon_km"]].drop_duplicates()
        if len(terna) != 1:
            raise ValueError(f"El panel mezcla ternas W/G/H: {terna.to_dict('records')}")
        w, g, h = (float(terna.iloc[0][c]) for c in ("window_km", "gap_km", "horizon_km"))
        forecasts = score_cuts(
            panel[KEY], series, load_forecaster(model_cfg),
            bin_km=float(cfg["series"]["bin_km"]), window_km=w, gap_km=g, horizon_km=h,
            target=score_cfg["target"], quantile=float(score_cfg["quantile"]), agg=score_cfg["agg"],
            max_context_bins=int(model_cfg.get("max_context_bins", 256)),
            batch_size=int(model_cfg.get("batch_size", 64)),
            forecast_kwargs=model_cfg.get("forecast_kwargs") or {},
        )
        # Antes de las métricas: si algo falla después, el pronóstico no se pierde.
        forecasts.to_parquet(forecasts_path, index=False)
        (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True),
                                             encoding="utf-8")

    dev_panel = panel.loc[dev_mask].reset_index(drop=True)
    scored = dev_panel[KEY + LABEL_COLUMNS].merge(forecasts, on=KEY, how="left", validate="1:1")
    missing = int(scored["score"].isna().sum())
    if missing:
        raise RuntimeError(f"{missing} filas de dev sin pronóstico: revisar las series de esos vehículos")
    scored["score_km"] = -scored["cut_odo"]
    evaluate_and_log(scored, dev_panel, cfg, panel_cfg, run_name, out_dir, seed)


def evaluate_and_log(scored: pd.DataFrame, dev_panel: pd.DataFrame, cfg: dict, panel_cfg: dict,
                     run_name: str, out_dir: Path, seed: int) -> None:
    """Tres scores sobre las filas de dev del panel v1.

    * TimesFM; naive (persistencia de la ventana); y `km` (= −cut_odo), el control:
      un score que no le gana a "menos km ⇒ más riesgo" no está viendo degradación.
      Con el emparejado por odómetro y mes, `km` tiene que quedar cerca de la tasa base.
    * Intervalos: los folds de `splits.json` por `iter_folds` (no se ajusta nada por fold).
    * La curva de anticipación se calcula sobre las mismas filas que `train.py`.
    """
    eval_cfg = cfg.get("eval", {})
    k_consecutive = int(eval_cfg.get("k_consecutive", 2))
    budget = float(eval_cfg.get("max_false_alarms_per_1000", 50))
    splits = load_splits(panel_cfg["output"]["splits"])
    folds = [valid for _, _, valid in iter_folds(dev_panel, splits, strict=True)]

    metrics: dict = {"k_consecutive": k_consecutive, "operating_point_budget_per_1000": budget,
                     "target": cfg["score"]["target"]}
    curves = {}
    for name, column in SCORES:
        fold_metrics = [ranking_metrics(scored.loc[valid, "label"], scored.loc[valid, column]) for valid in folds]
        oof = ranking_metrics(scored["label"], scored[column])
        curve = lead_time_curve(scored.assign(score=scored[column]),
                                n_thresholds=int(eval_cfg.get("n_thresholds", 50)), k_consecutive=k_consecutive)
        curves[name] = curve
        metrics[name] = {
            "oof": oof,
            "folds": fold_metrics,
            "folds_summary": summarize_folds(fold_metrics, seed=seed),
            "operating_point": operating_point(curve, max_false_alarms_per_1000=budget),
        }
        logger.info("%-7s | PR-AUC=%.4f (tasa base %.4f, lift %.2f) | ROC-AUC=%.4f | n=%d",
                    name, oof["pr_auc"], oof["base_rate"], oof["pr_auc_lift"], oof["roc_auc"], oof["n"])

    # Dentro de los vehículos con evento: ¿el score sube cuando el evento se acerca?
    events = scored[scored["event_observed"] == 1]
    metrics["within_event_vehicles"] = {name: ranking_metrics(events["label"], events[column])
                                        for name, column in SCORES}
    for name, m in metrics["within_event_vehicles"].items():
        logger.info("%-7s | solo vehículos con evento | PR-AUC=%.4f (base %.4f) | ROC-AUC=%.4f",
                    name, m["pr_auc"], m["base_rate"], m["roc_auc"])

    scored.to_parquet(out_dir / "predictions.parquet", index=False)  # solo dev, como train.py
    for name, curve in curves.items():
        curve.to_csv(out_dir / f"lead_time_curve_{name}.csv", index=False)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float), encoding="utf-8")
    logger.info("Outputs: %s", out_dir.relative_to(repo_root()) if out_dir.is_relative_to(repo_root()) else out_dir)

    run = init_wandb(cfg, run_name)
    if run is not None:
        import wandb

        flat = {}
        for name, _ in SCORES:
            flat.update({f"{name}/oof/{k}": v for k, v in metrics[name]["oof"].items()})
            flat.update({f"{name}/cv/{k}": v for k, v in metrics[name]["folds_summary"].items()})
        point = metrics["timesfm"]["operating_point"]
        if point:
            flat.update({f"pitch/{k}": point[k] for k in
                         ("detection_rate", "median_lead_km", "false_alarms_per_1000", "threshold")})
        run.log(flat)
        run.log({"lead_time_curve": wandb.Table(dataframe=curves["timesfm"])})
        run.finish()


if __name__ == "__main__":
    main()
