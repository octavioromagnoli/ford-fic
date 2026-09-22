#!/usr/bin/env python
"""Panel v1 en el reloj del evento, con el conjunto en riesgo de la ventana del registro (F5 §3.3).

    python scripts/build_window_survival_panel.py --config configs/data/panel_survival_ps.yaml --counts-only
    python scripts/build_window_survival_panel.py --config configs/data/panel_survival_ps.yaml

Toma el panel v1 **tal cual** (mismas filas, mismas features, mismos vehículos: los folds del
finalista sirven sin tocarlos) y le agrega:
- `feat_cut_dss`, los días desde la venta en el corte. Reemplaza a `feat_cut_odo`, que este
  panel no trae;
- las `aux_` del tramo en riesgo en días desde `c + G` (`src/data/window_risk.py`), que lee el
  target `window_survival`;
- `aux_eval_in_window`, la marca de fila evaluable con la etiqueta corregida.

Antes de escribir verifica tres cosas y falla si no se cumplen:
1. el origen del calendario re-estimado sobre dev es el congelado;
2. el odómetro del evento que se recalcula con estos viajes es el del panel (el panel lo
   usa para `time_to_event_km`);
3. ninguna fila con evento tiene el origen del riesgo después del evento.

`--counts-only` imprime el embudo sin escribir el panel. Todo conteo es de dev; del test solo
se dice cuántas filas hay.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from scripts.build_landmark_panel import _json_default, load_trips  # noqa: E402
from src.config import ensure_dir, load_config, repo_root, resolve_path, set_seed  # noqa: E402
from src.data.anchor import EPOCH, estimate_origin_day, project_dates_to_odometer  # noqa: E402
from src.data.join import load_vehicle_static  # noqa: E402
from src.data.window_risk import (  # noqa: E402
    CUT_DSS,
    ENTRY,
    EVALUABLE,
    EVENT,
    EXIT,
    HORIZON,
    RISK_COLUMNS,
    RegistryWindow,
    risk_summary,
    window_risk_columns,
)
from src.eval.splits import iter_repeats, load_splits, load_test_split, panel_fingerprint, test_split_masks  # noqa: E402

logger = logging.getLogger("build_window_survival_panel")

ID = "vehicle_id"
TRIP_COLUMNS = {ID, "TripNumber", "TripDatetimeStart", "OdometerTripEnd"}
#: Tolerancia del chequeo del odómetro del evento: el panel lo guardó como float.
EVENT_ODO_TOLERANCE_KM = 1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/data/panel_survival_ps.yaml")
    parser.add_argument("--counts-only", action="store_true", help="imprime el embudo sin escribir el panel")
    parser.add_argument("--refresh", action="store_true", help="vuelve a leer los viajes crudos")
    return parser.parse_args()


def build_vehicles(cfg: dict[str, Any], universe: set[str], dev: set[str],
                   trips: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Una fila por vehículo del universo con las fechas de venta y del evento (UTC)."""
    static = load_vehicle_static(config_path=cfg["sources"], dedupe_config=cfg["dedupe"],
                                 panel_config=cfg["panel_config"])
    static = static[static[ID].astype(str).isin(universe)].set_index(ID)
    missing = sorted(universe - set(static.index.astype(str)))
    if missing:
        raise ValueError(f"{len(missing)} vehículo(s) del holdout sin fila estática (ej.: {missing[:3]})")

    meta = json.loads(resolve_path(cfg["anchor_meta"]).read_text(encoding="utf-8"))
    origin = float(meta["anchor"]["origin_day_since_epoch"])
    first = trips.groupby(ID, observed=True)["TripDatetimeStart"].min()
    dev_index = static.index[static.index.astype(str).isin(dev)]
    check, stats = estimate_origin_day(first.reindex(dev_index), static.loc[dev_index, "static_ProductionDay"])
    if abs(check - origin) > 0.5:
        raise ValueError(f"El origen re-estimado sobre dev ({check}) no coincide con el congelado ({origin})")

    production = pd.to_numeric(static["static_ProductionDay"], errors="coerce")
    until_sale = pd.to_numeric(static["static_daysUntilSale"], errors="coerce")
    event_age = pd.to_numeric(static["event_day_since_production"], errors="coerce")
    v = pd.DataFrame(index=static.index)
    v["event_observed"] = static["event_observed"].astype(int)
    prod_date = EPOCH + pd.to_timedelta(origin + production, unit="D")
    v["sale_date"] = prod_date + pd.to_timedelta(until_sale, unit="D")
    v["event_date"] = (prod_date + pd.to_timedelta(event_age, unit="D")).where(v["event_observed"].eq(1))
    return v, {"origin_day_since_epoch": origin, "origin_reestimated_on_dev": check, **stats}


def check_event_odometer(panel: pd.DataFrame, vehicles: pd.DataFrame, trips: pd.DataFrame) -> float:
    """Máxima diferencia entre el odómetro del evento de estos viajes y el del panel (filas con evento)."""
    failed = panel.loc[panel["event_observed"].eq(1)]
    if failed.empty:
        return 0.0
    ids = failed[ID].astype(str).unique()
    events = vehicles.loc[vehicles.index.astype(str).isin(ids), "event_date"]
    recomputed = project_dates_to_odometer(events, trips)["event_odo_km"]
    recomputed.index = recomputed.index.astype(str)
    from_panel = failed["cut_odo"].astype(float) + failed["time_to_event_km"].astype(float)
    diff = (from_panel - failed[ID].astype(str).map(recomputed)).abs()
    if diff.isna().any():
        raise ValueError(f"{int(diff.isna().sum())} fila(s) con evento sin odómetro del evento recalculado")
    return float(diff.max())


def quantiles(values: pd.Series) -> dict[str, float]:
    values = pd.Series(values, dtype=float).dropna()
    if values.empty:
        return {}
    return {f"p{int(q * 100)}": float(values.quantile(q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)}


def stacked_rows(frame: pd.DataFrame, bin_days: float, max_days: float) -> tuple[int, int]:
    """Filas apiladas (fila × bin con exposición > 0) y eventos dentro del horizonte de entrenamiento."""
    entry = frame[ENTRY].to_numpy(float)
    exit_ = np.minimum(frame[EXIT].to_numpy(float), max_days)
    ok = np.isfinite(entry) & np.isfinite(exit_) & (exit_ > entry)
    first = np.floor(entry[ok] / bin_days)
    last = np.ceil(exit_[ok] / bin_days) - 1
    n = int((last - first + 1).sum())
    events = int((frame[EVENT].to_numpy(int)[ok] == 1)[frame[EXIT].to_numpy(float)[ok] <= max_days].sum())
    return n, events


def dev_counts(dev: pd.DataFrame, window: RegistryWindow, clock: dict[str, float],
               splits: dict[str, Any]) -> dict[str, Any]:
    """El embudo sobre dev. Conteos del objetivo, nunca features contra la etiqueta."""
    failed = dev["event_observed"].eq(1)
    known = dev[ENTRY].notna()
    entry, exit_ = dev[ENTRY], dev[EXIT]
    at_risk = known & (exit_ > entry)
    t0_before = known & entry.gt(0)
    t0_after = known & exit_.le(0) & ~failed
    evaluable = dev[EVALUABLE].eq(1)
    max_days = float(clock["max_horizon_days"])

    by_side = {}
    for name, mask in (("sanas", ~failed), ("de fallados", failed)):
        part = dev.loc[mask]
        by_side[name] = {
            "filas": int(mask.sum()),
            "vehiculos": int(part[ID].nunique()),
            "origen_desconocido": int((~known & mask).sum()),
            "origen_antes_de_la_ventana": int((t0_before & mask).sum()),
            "origen_despues_de_la_ventana": int((t0_after & mask).sum()),
            "en_riesgo": int((at_risk & mask).sum()),
            "vehiculos_con_alguna_fila_en_riesgo": int(part.loc[at_risk[mask], ID].nunique()),
            **{f"riesgo_{k}": v for k, v in risk_summary(part, max_horizon_days=max_days).items()
               if k.startswith("exposure")},
        }
    event_rows = dev[EVENT].eq(1)
    stacked, hazard_events = stacked_rows(dev, float(clock["bin_days"]), max_days)

    positive = dev["label"].eq(1)
    evaluable_block = {
        "filas": int(evaluable.sum()),
        "filas_label_1": int((evaluable & positive).sum()),
        "filas_label_0": int((evaluable & ~positive).sum()),
        "filas_label_0_de_fallados": int((evaluable & ~positive & failed).sum()),
        "sanos_con_fila": int(dev.loc[evaluable & ~failed, ID].nunique()),
        "fallados_con_fila": int(dev.loc[evaluable & failed, ID].nunique()),
        "fallados_con_fila_positiva": int(dev.loc[evaluable & positive, ID].nunique()),
        "positivas_perdidas": int((~evaluable & positive).sum()),
        "horizonte_desconocido": int(dev[HORIZON].isna().sum()),
    }
    healthy = ~failed
    h9 = {
        "sanas_enteras_en_la_ventana": float(evaluable[healthy].mean()),
        "sanas_origen_antes": float(t0_before[healthy].mean()),
    }

    folds = []
    for repeat, masks in iter_repeats(dev, splits, strict=True):
        for fold, _, valid in masks:
            v = dev.loc[valid]
            ev = v[EVALUABLE].eq(1)
            folds.append({
                "rep": repeat, "fold": fold,
                "fallados_valid": int(v.loc[v["event_observed"].eq(1), ID].nunique()),
                "fallados_evaluables": int(v.loc[ev & v["event_observed"].eq(1), ID].nunique()),
                "positivas_evaluables": int((ev & v["label"].eq(1)).sum()),
                "sanos_evaluables": int(v.loc[ev & v["event_observed"].eq(0), ID].nunique()),
                "eventos_en_riesgo_train": int(dev.loc[~valid, EVENT].sum()),
            })

    return {
        "filas": int(len(dev)),
        "vehiculos": int(dev[ID].nunique()),
        "fallados": int(dev.loc[failed, ID].nunique()),
        "positivas": int(positive.sum()),
        "por_lado": by_side,
        "filas_con_evento_en_riesgo": int(event_rows.sum()),
        "fallados_con_evento_en_riesgo": int(dev.loc[event_rows, ID].nunique()),
        "filas_con_evento_fuera_de_riesgo": int((failed & ~event_rows).sum()),
        "apiladas": {"bin_days": float(clock["bin_days"]), "max_horizon_days": max_days,
                     "filas_apiladas": stacked, "eventos_dentro_del_horizonte": hazard_events},
        "horizonte_en_dias": quantiles(dev[HORIZON]),
        "evaluables": evaluable_block,
        "contra_h9": h9,
        "cut_dss_nan": {"filas": int(dev[CUT_DSS].isna().sum()),
                        "vehiculos": int(dev.loc[dev[CUT_DSS].isna(), ID].nunique())},
        "por_fold": folds,
    }


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    args = parse_args()
    cfg = load_config(args.config)
    seed = set_seed(int(cfg["seed"]))
    window = RegistryWindow.from_config(load_config(cfg["event_clock"])["event_window"])
    clock = cfg["clock"]

    panel = pd.read_parquet(resolve_path(cfg["panel_in"]))
    clash = [c for c in RISK_COLUMNS if c in panel.columns]
    if clash:
        raise ValueError(f"El panel de entrada ya trae {clash}: ¿es el panel v1?")
    split = load_test_split(cfg["test_split"])
    dev_mask, test_mask = test_split_masks(panel, split, strict=True)
    dev_ids = {str(v) for v in split["dev_vehicles"]}
    universe = dev_ids | {str(v) for v in split["test_vehicles"]}

    trips = load_trips(cfg, universe, TRIP_COLUMNS, refresh=args.refresh)
    vehicles, anchor = build_vehicles(cfg, universe, dev_ids, trips)
    gap = check_event_odometer(panel.loc[dev_mask], vehicles, trips)
    if gap > EVENT_ODO_TOLERANCE_KM:
        raise ValueError(f"El odómetro del evento recalculado difiere del panel en {gap:.2f} km")

    risk = window_risk_columns(panel, trips, vehicles, window)
    out = pd.concat([panel, risk], axis=1)
    failed = out["event_observed"].eq(1) & out[ENTRY].notna()
    late = failed & (out[EXIT] < 0)
    if (late & dev_mask).any():
        raise ValueError(f"{int((late & dev_mask).sum())} fila(s) de dev con el origen del riesgo después del evento")
    if panel_fingerprint(out) != panel_fingerprint(panel):
        raise RuntimeError("El panel nuevo cambió de filas o de vehículos")

    dev = out.loc[dev_mask].reset_index(drop=True)
    splits = load_splits(resolve_path(cfg["splits"]))
    counts = dev_counts(dev, window, clock, splits)

    pd.set_option("display.width", 200)
    print("\n== Panel v1 en días post-venta, con la ventana del registro ==")
    print(f"ventana {window.start.date()} → {window.end.date()} · bins {clock['bin_days']} d · "
          f"score a {clock['horizon_days']} d · entrena hasta {clock['max_horizon_days']} d")
    print(f"filas {len(out)} (dev {int(dev_mask.sum())}, test {int(test_mask.sum())}) · "
          f"odómetro del evento contra el panel: {gap:.3f} km")
    print(f"dev: {counts['filas']} filas · {counts['vehiculos']} vehículos · {counts['fallados']} fallados · "
          f"{counts['positivas']} positivas")
    print("\n-- dev · origen y tramo en riesgo por lado --")
    print(pd.DataFrame(counts["por_lado"]).to_string())
    print(f"\nfilas con evento en riesgo: {counts['filas_con_evento_en_riesgo']} "
          f"({counts['fallados_con_evento_en_riesgo']} fallados) · filas de fallados fuera de riesgo: "
          f"{counts['filas_con_evento_fuera_de_riesgo']}")
    print(f"apiladas (bins de {clock['bin_days']} d hasta {clock['max_horizon_days']} d): "
          f"{counts['apiladas']['filas_apiladas']} filas, {counts['apiladas']['eventos_dentro_del_horizonte']} eventos")
    print(f"horizonte de 3.000 km en días (todas las filas de dev): {counts['horizonte_en_dias']}")
    print("\n-- dev · etiqueta corregida (horizonte entero dentro de la ventana) --")
    print(pd.Series(counts["evaluables"]).to_string())
    print(f"contra H9 (71,8% de las sanas enteras adentro, 8,7% empiezan antes): {counts['contra_h9']}")
    print(f"feat_cut_dss sin fecha de venta: {counts['cut_dss_nan']}")
    print("\n-- dev · por fold (validación) --")
    print(pd.DataFrame(counts["por_fold"]).to_string(index=False))

    meta = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": cfg.get("_config_path"),
        "panel_in": str(cfg["panel_in"]),
        "seed": seed,
        "window": {"start": str(window.start), "end": str(window.end)},
        "clock": clock,
        "anchor": anchor,
        "event_odometer_max_diff_km": gap,
        "fingerprint": panel_fingerprint(out),
        "dev": counts,
        "test": {"rows": int(test_mask.sum()), "vehicles": int(out.loc[test_mask, ID].nunique())},
        "columns": {"added": list(RISK_COLUMNS),
                    "feat": [c for c in out.columns if c.startswith("feat_")].__len__()},
    }
    if args.counts_only:
        print("\n--counts-only: no se escribió el panel")
        return 0

    out_path = resolve_path(cfg["output"]["panel"])
    ensure_dir(out_path.parent)
    out.to_parquet(out_path, index=False)
    meta_path = resolve_path(cfg["output"]["meta"])
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    rel = out_path.relative_to(repo_root()) if out_path.is_relative_to(repo_root()) else out_path
    print(f"\nescrito: {rel} · {meta_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
