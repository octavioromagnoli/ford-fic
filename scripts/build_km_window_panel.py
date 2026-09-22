#!/usr/bin/env python
"""Panel del finalista con el tramo en riesgo de la ventana del registro, en km (F6, K2).

    python scripts/build_km_window_panel.py --config configs/data/panel_survival_kmw.yaml --counts-only
    python scripts/build_km_window_panel.py --config configs/data/panel_survival_kmw.yaml

Toma `panel_survival.parquet` **tal cual** (mismas filas, las mismas 53 features más
`feat_cut_odo`, mismos vehículos: los folds del finalista sirven sin tocarlos) y le agrega
solo columnas `aux_`, que ningún modelo ve:

- `aux_odo_window_start` / `aux_odo_window_end`: el odómetro del vehículo en el inicio y en
  el fin de la ventana del registro (`configs/data/event_clock.yaml`), con la misma
  interpolación que ubica el evento en km (`src/data/anchor.py::project_dates_to_odometer`).
- El tramo en riesgo de cada fila **en km desde `c + G`**, que lee el target
  `window_km_survival`:
  * entra en `max(0, odómetro al inicio de la ventana − (c + G))`: antes del 01-09-2025 el
    registro no anotaba eventos, así que esos km no son supervivencia verificada;
  * un fallado sale en su evento (todos los de dev caen dentro de la ventana);
  * un sano sale en `min(último odómetro, odómetro al fin de la ventana) − (c + G)`. Después
    del 11-03-2026 el registro ya no anotaba nada, y sin telemetría no se sabe cuántos km
    anduvo.
- `aux_eval_in_window` (la fila evaluable con la etiqueta corregida, V) y `aux_cut_dss`
  (días desde la venta en el corte, para el piso) salen de `src/data/window_risk.py`, la
  misma función que el panel de F5 §3.3. `feat_cut_dss` se renombra a `aux_`: acá no es
  feature.

Antes de escribir verifica que el origen del calendario y el odómetro del evento sean los
congelados, que ninguna fila con evento quede fuera de riesgo y que la huella del panel no
cambie. `--counts-only` imprime el embudo sin escribir. Todo conteo es de dev; del test solo
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
from scripts.build_window_survival_panel import (  # noqa: E402
    EVENT_ODO_TOLERANCE_KM,
    TRIP_COLUMNS,
    build_vehicles,
    check_event_odometer,
    quantiles,
)
from src.config import ensure_dir, load_config, repo_root, resolve_path, set_seed  # noqa: E402
from src.data.anchor import project_dates_to_odometer  # noqa: E402
from src.data.window_risk import CUT_DSS, EVALUABLE, RegistryWindow, window_risk_columns  # noqa: E402
from src.eval.splits import iter_repeats, load_splits, load_test_split, panel_fingerprint, test_split_masks  # noqa: E402

logger = logging.getLogger("build_km_window_panel")

ID = "vehicle_id"
ODO_START = "aux_odo_window_start"
ODO_END = "aux_odo_window_end"
ENTRY_KM = "aux_risk_entry_km"
EXIT_KM = "aux_risk_exit_km"
EVENT_KM = "aux_risk_event_km"
CUT_DSS_AUX = "aux_cut_dss"
KM_COLUMNS = (ODO_START, ODO_END, ENTRY_KM, EXIT_KM, EVENT_KM, EVALUABLE, CUT_DSS_AUX)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/data/panel_survival_kmw.yaml")
    parser.add_argument("--counts-only", action="store_true", help="imprime el embudo sin escribir el panel")
    parser.add_argument("--refresh", action="store_true", help="vuelve a leer los viajes crudos")
    return parser.parse_args()


def window_odometers(vehicle_ids: pd.Index, trips: pd.DataFrame, window: RegistryWindow) -> pd.DataFrame:
    """Odómetro de cada vehículo en el inicio y en el fin de la ventana.

    `project_dates_to_odometer` interpola entre el viaje anterior y el posterior. Sin viaje
    anterior (la telemetría arranca después de esa fecha) devuelve NaN; sin viaje posterior,
    el último odómetro visto.
    """
    out = pd.DataFrame(index=pd.Index(vehicle_ids.astype(str), name=ID))
    for column, date in ((ODO_START, window.start), (ODO_END, window.end)):
        dates = pd.Series(date, index=out.index)
        projected = project_dates_to_odometer(dates, trips)
        projected.index = projected.index.astype(str)
        out[column] = projected["event_odo_km"].reindex(out.index).astype(float)
        out[f"{column}_source"] = projected["event_odo_source"].reindex(out.index)
    return out


def km_risk_columns(panel: pd.DataFrame, odometers: pd.DataFrame, event_in_window: pd.Series) -> pd.DataFrame:
    """El tramo en riesgo de cada fila, en km desde `c + G`, dentro de la ventana del registro."""
    ids = panel[ID].astype(str)
    start_odo = panel["cut_odo"].astype(float) + panel["gap_km"].astype(float)
    odo_start = ids.map(odometers[ODO_START])
    end_source = ids.map(odometers[f"{ODO_END}_source"])
    # Sin viaje antes del fin de la ventana, el vehículo no estuvo expuesto en ella.
    odo_end = ids.map(odometers[ODO_END]).where(end_source.ne("previo_al_primer_viaje"), -np.inf)

    # Sin viaje antes del inicio: la telemetría arranca adentro de la ventana y la fila entra en c + G.
    entry = (odo_start - start_odo).clip(lower=0.0).fillna(0.0)
    failed = panel["event_observed"].eq(1)
    in_window = failed & ids.map(event_in_window).fillna(False).astype(bool)
    last_seen = panel["cut_odo"].astype(float) + panel["aux_km_observed_after_cut"].astype(float)
    healthy_exit = np.minimum(last_seen, odo_end) - start_odo
    event_exit = panel["time_to_event_km"].astype(float) - panel["gap_km"].astype(float)
    exit_ = pd.Series(np.where(in_window, event_exit, healthy_exit), index=panel.index)
    at_risk = (exit_ >= 0) & ((exit_ > entry) | (in_window & (exit_ >= entry)))

    out = pd.DataFrame(index=panel.index)
    out[ODO_START] = odo_start
    out[ODO_END] = ids.map(odometers[ODO_END])
    out[ENTRY_KM] = entry
    out[EXIT_KM] = exit_.where(np.isfinite(exit_))
    out[EVENT_KM] = (in_window & at_risk).astype(int)
    return out


def stacked_count(frame: pd.DataFrame, bin_km: float, max_km: float) -> dict[str, int]:
    """Filas apiladas (fila × bin en riesgo) y eventos dentro del horizonte de entrenamiento."""
    entry = frame[ENTRY_KM].to_numpy(float)
    exit_ = frame[EXIT_KM].to_numpy(float)
    event = frame[EVENT_KM].to_numpy(int) == 1
    ok = np.isfinite(exit_) & (exit_ >= 0) & ((exit_ > entry) | (event & (exit_ >= entry))) & (entry < max_km)
    first = np.floor(entry[ok] / bin_km)
    last = np.minimum(np.floor(np.minimum(exit_[ok], max_km) / bin_km), max_km / bin_km - 1)
    return {"filas_apiladas": int((last - first + 1).sum()),
            "eventos": int((event[ok] & (exit_[ok] <= max_km)).sum())}


def dev_counts(dev: pd.DataFrame, clock: dict[str, float], splits: dict[str, Any]) -> dict[str, Any]:
    """El embudo sobre dev: conteos del objetivo, nunca features contra la etiqueta."""
    failed = dev["event_observed"].eq(1)
    entry, exit_ = dev[ENTRY_KM], dev[EXIT_KM]
    event = dev[EVENT_KM].eq(1)
    at_risk = exit_.notna() & (exit_ >= 0) & ((exit_ > entry) | (event & (exit_ >= entry)))
    reference_max = float(clock["reference_max_horizon_km"])
    by_side = {}
    for name, mask in (("sanas", ~failed), ("de fallados", failed)):
        risk = at_risk & mask
        exposure = (exit_ - entry).where(risk)
        by_side[name] = {
            "filas": int(mask.sum()),
            "vehiculos": int(dev.loc[mask, ID].nunique()),
            "en_riesgo": int(risk.sum()),
            "vehiculos_con_fila_en_riesgo": int(dev.loc[risk, ID].nunique()),
            "entran_tarde": int((risk & entry.gt(0)).sum()),
            "sin_exposicion_en_la_ventana": int((mask & ~at_risk).sum()),
            "km_en_riesgo_mediana": float(exposure.median()),
            "km_en_riesgo_total": float(exposure.sum()),
        }
    # Lo que la referencia entrena distinto: sale del mismo panel con su propia cuenta.
    tte = dev["time_to_event_km"] - dev["gap_km"]
    followup = dev["aux_km_observed_after_cut"] - dev["gap_km"]
    reference = {
        "eventos_mas_alla_de_su_horizonte": int((failed & tte.gt(reference_max)).sum()),
        "sanas_con_supervivencia_despues_de_la_ventana": int((~failed & followup.gt(exit_.fillna(-np.inf))).sum()),
        "km_sanos_despues_de_la_ventana": float((np.minimum(followup, reference_max)
                                                 - np.minimum(exit_.fillna(0).clip(lower=0), reference_max))
                                                .where(~failed).clip(lower=0).sum()),
    }
    candidates = {f"{int(m)}": stacked_count(dev, float(clock["bin_km"]), float(m))
                  for m in clock["max_horizon_km_options"]}
    folds = []
    for repeat, masks in iter_repeats(dev, splits, strict=True):
        for fold, train, valid in masks:
            folds.append({"rep": repeat, "fold": fold,
                          "eventos_en_riesgo_train": int((event & train).sum()),
                          "fallados_train": int(dev.loc[train & failed, ID].nunique()),
                          "fallados_valid": int(dev.loc[valid & failed, ID].nunique())})
    return {
        "filas": int(len(dev)),
        "vehiculos": int(dev[ID].nunique()),
        "fallados": int(dev.loc[failed, ID].nunique()),
        "por_lado": by_side,
        "filas_con_evento_en_riesgo": int(event.sum()),
        "filas_de_fallados_fuera_de_riesgo": int((failed & ~event).sum()),
        "salida_km_en_riesgo": quantiles(exit_.where(at_risk)) | {"max": float(exit_.where(at_risk).max())},
        "entrada_km_en_riesgo": quantiles(entry.where(at_risk & entry.gt(0))),
        "contra_la_referencia": reference,
        "apiladas_por_horizonte": candidates,
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
    clash = [c for c in (*KM_COLUMNS, CUT_DSS) if c in panel.columns]
    if clash:
        raise ValueError(f"El panel de entrada ya trae {clash}: ¿es `panel_survival.parquet`?")
    if "feat_cut_odo" not in panel:
        raise ValueError("El panel de entrada no trae `feat_cut_odo`: tiene que ser el del finalista")
    split = load_test_split(cfg["test_split"])
    dev_mask, test_mask = test_split_masks(panel, split, strict=True)
    dev_ids = {str(v) for v in split["dev_vehicles"]}
    universe = dev_ids | {str(v) for v in split["test_vehicles"]}

    trips = load_trips(cfg, universe, TRIP_COLUMNS, refresh=args.refresh)
    vehicles, anchor = build_vehicles(cfg, universe, dev_ids, trips)
    gap = check_event_odometer(panel.loc[dev_mask], vehicles, trips)
    if gap > EVENT_ODO_TOLERANCE_KM:
        raise ValueError(f"El odómetro del evento recalculado difiere del panel en {gap:.2f} km")

    event_date = vehicles["event_date"]
    event_in_window = event_date.notna() & (event_date >= window.start) & (event_date <= window.end)
    event_in_window.index = event_in_window.index.astype(str)
    odometers = window_odometers(vehicles.index, trips, window)
    km = km_risk_columns(panel, odometers, event_in_window)
    days = window_risk_columns(panel, trips, vehicles, window)
    out = pd.concat([panel, km, days[[EVALUABLE]], days[[CUT_DSS]].rename(columns={CUT_DSS: CUT_DSS_AUX})], axis=1)

    failed_dev = dev_mask & out["event_observed"].eq(1)
    lost = failed_dev & out[EVENT_KM].ne(1)
    if lost.any():
        raise ValueError(f"{int(lost.sum())} fila(s) de fallados de dev fuera de riesgo o con el evento fuera de la ventana")
    if panel_fingerprint(out) != panel_fingerprint(panel):
        raise RuntimeError("El panel nuevo cambió de filas o de vehículos")
    if any(c.startswith(("feat_", "static_")) for c in out.columns.difference(panel.columns)):
        raise RuntimeError("Una columna nueva entraría al modelo: tienen que ser todas `aux_`")

    dev = out.loc[dev_mask].reset_index(drop=True)
    splits = load_splits(resolve_path(cfg["splits"]))
    counts = dev_counts(dev, clock, splits)
    sources = odometers.loc[odometers.index.isin(dev_ids)]

    pd.set_option("display.width", 200)
    print("\n== Panel del finalista con el tramo en riesgo de la ventana, en km ==")
    print(f"ventana {window.start.date()} → {window.end.date()} · bins {clock['bin_km']} km")
    print(f"filas {len(out)} (dev {int(dev_mask.sum())}, test {int(test_mask.sum())}) · "
          f"odómetro del evento contra el panel: {gap:.3f} km")
    print(f"dev: {counts['filas']} filas · {counts['vehiculos']} vehículos · {counts['fallados']} fallados")
    print("origen del odómetro de la ventana (dev):")
    print(pd.DataFrame({c: sources[f"{c}_source"].value_counts() for c in (ODO_START, ODO_END)}).fillna(0).astype(int))
    print("\n-- dev · tramo en riesgo por lado --")
    print(pd.DataFrame(counts["por_lado"]).to_string())
    print(f"\nfilas con evento en riesgo: {counts['filas_con_evento_en_riesgo']} · "
          f"filas de fallados fuera de riesgo: {counts['filas_de_fallados_fuera_de_riesgo']}")
    print(f"salida en km (filas en riesgo): {counts['salida_km_en_riesgo']}")
    print(f"entrada en km (filas que entran tarde): {counts['entrada_km_en_riesgo']}")
    print(f"lo que la referencia (hasta {clock['reference_max_horizon_km']} km, sin ventana) entrena distinto: "
          f"{counts['contra_la_referencia']}")
    print(f"apiladas por horizonte de entrenamiento: {counts['apiladas_por_horizonte']}")
    print("\n-- dev · por fold --")
    print(pd.DataFrame(counts["por_fold"]).to_string(index=False))

    if args.counts_only:
        print("\n--counts-only: no se escribió el panel")
        return 0

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
        "columns_added": list(KM_COLUMNS),
    }
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
