#!/usr/bin/env python
"""Anticipación de la primera alerta y trayectoria del score en los fallados, sobre corridas de dev v2.

    python scripts/report_v2_leads.py --config configs/report_v2_leads.yaml

Complementa a `scripts/report_v2_models.py` (misma regla de alerta y mismo umbral exacto por
presupuesto de falsas alarmas). No reentrena nada. Por corrida y presupuesto:

* **Anticipación** de la primera alerta sostenida (k cortes seguidos ≥ τ) de cada fallado detectado:
  percentiles 25/50/75 en km hasta el evento y en días. Los días son aproximados: km / (km por día del
  auto), con los km por día estimados como la pendiente odómetro ~ fecha de sus cortes.
* **Dónde cae la primera alerta**: fracción de detectados que alertan en su primer corte posible y
  posición media de la alerta dentro de su historial (0 = primer corte, 1 = último).
* **Trayectoria del score**: rango percentil medio del score (promedio de las R repeticiones) en los
  cortes de fallados a más de `far_km`, entre `near_km` y `far_km`, y a menos de `near_km` del evento,
  contra el de los sanos. Si sube al acercarse el evento, el modelo sabe algo del *cuándo*; el nivel
  lejano sobre los sanos es el *qué auto*.

Deja `experiments/<output_name>/leads.csv` y `trend.csv`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from scripts.audit_detection_null import vehicle_levels  # noqa: E402
from scripts.report_v2_models import exact_tau, load_run  # noqa: E402
from src.config import ensure_dir, load_config, resolve_path  # noqa: E402


def km_per_day(pred: pd.DataFrame) -> pd.Series:
    """Pendiente odómetro ~ días de los cortes de cada auto (NaN con menos de 3 fechas distintas)."""
    t = (pd.to_datetime(pred["cut_date"]) - pd.Timestamp("2024-01-01")).dt.total_seconds() / 86400.0
    frame = pred.assign(t=t)

    def slope(g: pd.DataFrame) -> float:
        if g["t"].nunique() < 3:
            return float("nan")
        return float(np.polyfit(g["t"], g["cut_odo"], 1)[0])

    return frame.groupby("vehicle_id")[["t", "cut_odo"]].apply(slope)


def leads_for(pred: pd.DataFrame, n_rep: int, budget: float, k: int, kmd: pd.Series) -> dict:
    leads, days, first, pos = [], [], [], []
    failed = pred[pred["event_observed"] == 1]
    for r in range(n_rep):
        s = pred[f"score_r{r}"].to_numpy(dtype=float)
        levels, event, _ = vehicle_levels(pred, s, k)
        tau = exact_tau(levels[event == 0], budget)
        for vid, g in failed.groupby("vehicle_id", sort=False):
            flags = (g[f"score_r{r}"].to_numpy() >= tau).astype(int)
            hits = np.flatnonzero(np.convolve(flags, np.ones(k, dtype=int), "valid") == k)
            if not hits.size:
                continue
            lead = float(g["time_to_event_km"].to_numpy()[hits[0]])
            leads.append(lead)
            rate = kmd.get(vid, np.nan)
            days.append(lead / rate if np.isfinite(rate) and rate > 0 else np.nan)
            first.append(hits[0] == 0)
            pos.append(hits[0] / max(len(g) - k, 1))
    lead_km, lead_d = np.asarray(leads), np.asarray(days, dtype=float)
    lead_d = lead_d[np.isfinite(lead_d)]
    q = (lambda a, p: float(np.percentile(a, p)) if a.size else float("nan"))
    return {"detected_per_repeat": len(leads) / n_rep,
            "km_p25": q(lead_km, 25), "km_p50": q(lead_km, 50), "km_p75": q(lead_km, 75),
            "days_p25": q(lead_d, 25), "days_p50": q(lead_d, 50), "days_p75": q(lead_d, 75),
            "frac_first_cut": float(np.mean(first)) if first else float("nan"),
            "mean_alert_position": float(np.mean(pos)) if pos else float("nan")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    root = resolve_path(cfg.get("experiments_dir", "experiments"))
    out_dir = ensure_dir(root / cfg["output_name"])
    k = int(cfg["k_consecutive"])
    far, near = float(cfg["far_km"]), float(cfg["near_km"])

    lead_rows, trend_rows = [], []
    for spec in cfg["runs"]:
        pred, _, n_rep = load_run(spec["run"], root)
        kmd = km_per_day(pred)
        for b in cfg["budgets"]:
            lead_rows.append({"run": spec["run"], "label": spec["label"], "budget": float(b),
                              **leads_for(pred, n_rep, float(b), k, kmd)})
        rank = np.mean([pred[f"score_r{r}"].rank(pct=True).to_numpy() for r in range(n_rep)], axis=0)
        tte = pred["time_to_event_km"].to_numpy()
        is_f = pred["event_observed"].to_numpy() == 1
        trend_rows.append({"run": spec["run"], "label": spec["label"],
                           "healthy": float(rank[~is_f].mean()),
                           f"failed_gt_{int(far)}": float(rank[is_f & (tte > far)].mean()),
                           f"failed_{int(near)}_{int(far)}": float(rank[is_f & (tte > near) & (tte <= far)].mean()),
                           f"failed_le_{int(near)}": float(rank[is_f & (tte <= near)].mean())})
    leads, trend = pd.DataFrame(lead_rows), pd.DataFrame(trend_rows)
    leads.to_csv(out_dir / "leads.csv", index=False)
    trend.to_csv(out_dir / "trend.csv", index=False)
    pd.set_option("display.width", 250)
    print(leads.drop(columns="run").round(2).to_string(index=False))
    print()
    print(trend.drop(columns="run").round(3).to_string(index=False))
    print(f"\n→ {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
