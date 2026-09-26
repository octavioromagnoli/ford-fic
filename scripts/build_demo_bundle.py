#!/usr/bin/env python
"""Bundle de la demo de producto: lo que la app y los agentes leen, precalculado y solo de dev.

    FORD_DATA_DIR=$PWD/data/rebuild-0921 python scripts/build_demo_bundle.py --config configs/demo.yaml
    python scripts/build_demo_bundle.py --config configs/demo.yaml --publish-only   # wandb Artifact

La app desplegada no importa LightGBM, SHAP ni sklearn: lee este directorio. Acá se hace todo lo que
necesita el stack completo (la corrida de K2, la explicabilidad, la capa de decisión) y nada se
recalcula distinto: el umbral, la alerta, el mensaje y el waterfall salen de las mismas funciones que
el dashboard de F4 (`src/eval/dashboard_data.py`) y la explicabilidad (`src/eval/explain.py`).

Deja en `bundle.dir`:
- cuts.parquet      una fila por corte de dev evaluable con la etiqueta elegida: score de la
                    repetición, fecha, mercado y perfil de uso
- vehicles.parquet  una fila por auto: evento (fecha y odómetro), alerta de la repetición, factores y
                    mensaje de la explicabilidad, señales del filtro contra la flota sana (taller)
- waterfall.parquet los escalones del "por qué" de cada auto
- meta.json         umbral, regla, números oficiales de la capa de decisión, ventana del replay,
                    textos fijos y procedencia (commit y hash de cada insumo)

Un directorio `llm_cache/` o `triage/` que ya esté en `bundle.dir` (lo deja
`scripts/warm_demo_cache.py`) se conserva y viaja con `--publish-only`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.agents.formatting import format_value  # noqa: E402
from src.config import ensure_dir, load_config, repo_root, resolve_path  # noqa: E402
from src.data.anchor import event_dates  # noqa: E402
from src.data.join import load_vehicle_static  # noqa: E402
from src.eval import explain as ex  # noqa: E402
from src.eval.dashboard_data import (KEY, curve_summary, holdout_summary, load_explanations, load_k2,  # noqa: E402
                                     operating_threshold, rows_for, vehicle_alerts, vehicle_why)
from src.eval.metrics import _first_sustained_index  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger("build_demo_bundle")

BUNDLE_FILES = ("cuts.parquet", "vehicles.parquet", "waterfall.parquet", "meta.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/demo.yaml")
    parser.add_argument("--publish-only", action="store_true",
                        help="no reconstruye: publica el bundle que ya está en disco como wandb Artifact")
    return parser.parse_args()


def sha256_16(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=repo_root(), capture_output=True, text=True)
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=repo_root(), capture_output=True, text=True)
        return out.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")
    except OSError:
        return "desconocido"


def dev_guard(vehicle_ids: pd.Series, test_split_path: Path) -> dict[str, Any]:
    """Falla si algún vehículo no es de dev. El bundle se publica: el test no puede viajar."""
    split = json.loads(test_split_path.read_text(encoding="utf-8"))
    dev, test = set(map(str, split["dev_vehicles"])), set(map(str, split["test_vehicles"]))
    ids = set(vehicle_ids.astype(str))
    if ids & test:
        raise RuntimeError(f"El bundle trae {len(ids & test)} vehículo(s) de test: {sorted(ids & test)[:3]}")
    if ids - dev:
        raise RuntimeError(f"El bundle trae {len(ids - dev)} vehículo(s) fuera de dev: {sorted(ids - dev)[:3]}")
    return {"n_vehicles": len(ids), "all_dev": True, "test_vehicles_present": 0}


def alert_timing(rows: pd.DataFrame, repeat: int, threshold: float, k: int) -> pd.DataFrame:
    """Cuándo se ve la alerta en el calendario: arranca en el primer corte de la racha, pero recién se
    confirma en el `k`-ésimo. El replay la muestra al confirmarse (lo que se sabía ese día)."""
    records = []
    for vid, g in rows.groupby("vehicle_id", sort=False):
        g = g.sort_values("cut_odo", kind="stable")
        idx = _first_sustained_index(g[f"score_r{repeat}"].to_numpy(dtype=float) >= threshold, k)
        if idx is None:
            continue
        first, confirm = g.iloc[idx], g.iloc[idx + k - 1]
        records.append({"vehicle_id": vid, "alert_first_date": first["cut_date"], "alert_confirm_date": confirm["cut_date"],
                        "alert_confirm_odo": float(confirm["cut_odo"]),
                        "alert_confirm_km_to_event": float(confirm["time_to_event_km"])})
    return pd.DataFrame(records)


def event_calendar(cfg: dict[str, Any], vehicle_ids: pd.Series) -> pd.Series:
    """Fecha del evento por vehículo (NaT en los sanos), con el origen congelado del build."""
    ecfg = cfg["event_dates"]
    static = load_vehicle_static(config_path=ecfg["sources"], dedupe_config=ecfg["dedupe"],
                                 panel_config=ecfg["panel_config"]).set_index("vehicle_id")
    static = static.loc[static.index.astype(str).isin(set(vehicle_ids.astype(str)))]
    meta = json.loads(resolve_path(ecfg["anchor_meta"]).read_text(encoding="utf-8"))
    origin = float(meta["anchor"]["origin_day_since_epoch"])
    dates = event_dates(static.loc[static["event_observed"].eq(1)], origin)
    # El ancla está en UTC y las fechas de los cortes son ingenuas: se comparan sin zona.
    return dates.dt.tz_localize(None).reindex(static.index).rename("event_date")


def factor_records(factors_json: str, texts: dict[str, Any]) -> list[dict[str, Any]]:
    """Los factores del mensaje con su texto: nombre, valor y referencia ya formateados."""
    out = []
    for f in json.loads(factors_json):
        t = (texts.get("features") or {}).get(f["feature"]) or {}
        fmt = t.get("format", "")
        out.append({"feature": f["feature"], "label": t.get("label", f["feature"]), "format": fmt,
                    "value": float(f["value"]), "reference": float(f["reference"]),
                    "value_text": format_value(float(f["value"]), fmt),
                    "reference_text": format_value(float(f["reference"]), fmt),
                    "recommendation": t.get("recommendation"), "contribution": float(f["contribution"])})
    return out


def technician_signals(rows: pd.DataFrame, explained: pd.DataFrame, signals: dict[str, Any]) -> dict[str, list]:
    """Las señales del filtro de cada auto en sus cortes explicados, contra los sanos de su mercado.

    La referencia es la mediana, entre los autos sanos de dev del mismo mercado, de la mediana de
    cada auto (así un auto con muchos cortes no pesa más). Es descriptiva: la ve el taller.
    """
    cols = list(signals)
    healthy = rows.loc[rows["event_observed"].eq(0)]
    per_vehicle = healthy.groupby(["static_SalesCountry_cd", "vehicle_id"])[cols].median()
    reference = per_vehicle.groupby(level=0).median()
    values = explained.merge(rows[KEY + ["static_SalesCountry_cd"] + cols], on=KEY, how="left")
    means = values.groupby("vehicle_id").agg({"static_SalesCountry_cd": "first", **{c: "mean" for c in cols}})
    out: dict[str, list] = {}
    for vid, r in means.iterrows():
        ref = reference.loc[r["static_SalesCountry_cd"]]
        items = []
        for col, spec in signals.items():
            scale = float(spec.get("scale", 1.0))
            value, refv = float(r[col]), float(ref[col])
            items.append({"feature": col, "label": spec["label"], "value": value, "reference": refv,
                          "value_text": format_value(value * scale, spec["format"]),
                          "reference_text": format_value(refv * scale, spec["format"])})
        out[str(vid)] = items
    return out


def later_explained_cuts(rows: pd.DataFrame, timing: pd.DataFrame, label_threshold: dict[int, float],
                         k: int, repeat: int) -> list[str]:
    """Autos cuyo vector V3 promedia cortes explicados posteriores a la alerta de la repetición.

    V3 promedia las 3 repeticiones y cada una explica su propia racha; si otra repetición alerta
    más tarde, el ranking de los factores usa cortes que el replay todavía no vio. Se reporta.
    """
    confirm = timing.set_index("vehicle_id")["alert_confirm_date"]
    dates = rows.set_index(KEY)["cut_date"]
    later = set()
    for r, thr in label_threshold.items():
        if r == repeat:
            continue
        cuts = ex.explained_cuts(rows, r, thr, k)
        for vid, g in cuts.groupby("vehicle_id"):
            if vid in confirm.index:
                last = max(dates.loc[(vid, c)] for c in g["cut_odo"])
                if last > confirm.loc[vid]:
                    later.add(str(vid))
    return sorted(later)


def build(cfg: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    dash_path = cfg["dashboard_config"]
    dcfg = load_config(dash_path)
    d = load_k2(dash_path)
    run_cfg = load_config(dcfg["run_config"])
    label, repeat, budget = cfg["label"], int(cfg["repeat"]), float(cfg["budget_per_1000"])
    k = int(d.eval_cfg.get("k_consecutive", 2))
    rows = rows_for(d, label)

    # -- el umbral y la alerta: los del dashboard, en la repetición elegida ------------------------
    point = operating_threshold(rows, repeat, budget, d.eval_cfg)
    if point is None:
        raise RuntimeError(f"Ningún umbral respeta {budget:g} falsas alarmas cada 1.000 en R{repeat + 1}")
    threshold = float(point["threshold"])
    alerts = vehicle_alerts(rows, repeat, threshold, k)
    timing = alert_timing(rows, repeat, threshold, k)
    logger.info("R%d · %s · %.0f‰ | umbral %.4f | %s", repeat + 1, label, budget, threshold,
                alerts["outcome"].value_counts().to_dict())

    # -- cortes: score de la repetición + perfil de uso y señales del filtro del panel -------------
    panel = pd.read_parquet(resolve_path(run_cfg["data"]["panel"]))
    extra = [c for c in dict.fromkeys([*cfg["profile_features"], *cfg["technician_signals"]]) if c not in rows]
    rows = rows.merge(panel[KEY + extra], on=KEY, how="left", validate="one_to_one")
    cuts = rows[KEY + ["cut_date", "static_SalesCountry_cd", "event_observed", "time_to_event_km"]
                + list(dict.fromkeys(cfg["profile_features"]))].copy()
    cuts["score"] = rows[f"score_r{repeat}"].to_numpy()
    cuts = cuts.rename(columns={"static_SalesCountry_cd": "market"}).sort_values(KEY, ignore_index=True)

    # -- explicabilidad: mensaje, factores, waterfall ----------------------------------------------
    expl = load_explanations(dash_path)
    if expl is None:
        raise FileNotFoundError("Falta la explicabilidad: corré scripts/explain_k2.py --config configs/explain_k2.yaml")
    msgs = expl.messages.loc[expl.messages["label"].eq(label) & expl.messages["repeat"].eq(repeat)].set_index("vehicle_id")
    ctx = expl.vehicle.loc[expl.vehicle["variant"].eq("V1") & expl.vehicle["label"].eq(label)
                           & expl.vehicle["replicate"].eq(repeat)].set_index("vehicle_id")
    ecfg = load_config(dcfg["explain_config"])
    symptom_min = float(ecfg["message"]["symptom_line_min"])
    gap_km, horizon_km = float(rows["gap_km"].iloc[0]), float(rows["horizon_km"].iloc[0])

    explained = ex.explained_cuts(rows, repeat, threshold, k)
    signals = technician_signals(rows, explained, cfg["technician_signals"])

    events = event_calendar(cfg, alerts["vehicle_id"])
    vehicles = alerts.merge(timing, on="vehicle_id", how="left")
    vehicles["event_date"] = vehicles["vehicle_id"].map(events)
    records, steps = [], []
    for vid in vehicles["vehicle_id"]:
        m = msgs.loc[vid]
        kmpd = float(ctx.loc[vid, "km_per_day"])
        weeks = ex.horizon_weeks(gap_km, horizon_km, kmpd)
        records.append({
            "vehicle_id": vid, "risk_level": m["risk_level"], "n_factors": int(m["n_factors"]),
            "factors": json.dumps(factor_records(m["factors"], expl.texts), ensure_ascii=False),
            "symptom_contribution": float(m["symptom_contribution"]),
            "symptoms_up": bool(np.isfinite(m["symptom_contribution"]) and m["symptom_contribution"] > symptom_min),
            "template_message": m["message"], "km_per_day": kmpd,
            "horizon_weeks_lo": weeks[0] if weeks else None, "horizon_weeks_hi": weeks[1] if weeks else None,
            "technician_signals": json.dumps(signals.get(str(vid), []), ensure_ascii=False),
        })
        why = vehicle_why(expl, vid, label, repeat, top=int(cfg["waterfall_top"]))
        if why is not None:
            s = why["steps"].assign(vehicle_id=vid, base=why["base"], output=why["output"])
            steps.append(s)
    vehicles = vehicles.merge(pd.DataFrame(records), on="vehicle_id", how="left", validate="one_to_one")
    waterfall = pd.concat(steps, ignore_index=True) if steps else pd.DataFrame()

    guard = dev_guard(pd.concat([cuts["vehicle_id"], vehicles["vehicle_id"]]), resolve_path(run_cfg["splits"]["test_split"]))

    # -- auditoría: ¿el porqué usa cortes posteriores a la alerta? ---------------------------------
    thresholds = {r: float(operating_threshold(rows, r, budget, d.eval_cfg)["threshold"]) for r in range(d.n_repeats)}
    later = later_explained_cuts(rows, timing, thresholds, k, repeat)

    # -- números oficiales: la capa de decisión y window_eval, sin recalcular ----------------------
    curve = curve_summary(d.decision, label)
    holdout = holdout_summary(d.decision, label)
    texts = expl.texts
    run_dir = resolve_path(run_cfg.get("output_dir", "experiments")) / run_cfg["name"]
    inputs = {
        "predictions": run_dir / "predictions.parquet", "window_eval": run_dir / "window_eval.json",
        "decision_layer": resolve_path(dcfg["decision_layer"]),
        "messages": resolve_path(ecfg["output_dir"]) / "messages.parquet",
        "shap_vehicle": resolve_path(ecfg["output_dir"]) / ecfg["outputs"]["shap_vehicle"],
        "panel": resolve_path(run_cfg["data"]["panel"]),
    }
    replay = cfg["replay"]
    meta = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "config": cfg["_config_path"],
        "product_name": cfg["product_name"],
        "run": run_cfg["name"],
        "label": label, "repeat": repeat, "budget_per_1000": budget,
        "threshold": threshold, "k_consecutive": k, "gap_km": gap_km, "horizon_km": horizon_km,
        "replay": {"start": replay["start"], "end": replay["end"], "first_week": replay["first_week"]},
        "counts": {
            "vehicles": int(len(vehicles)), "failed": int(vehicles["failed"].sum()),
            "healthy": int((~vehicles["failed"]).sum()),
            "outcomes": {k_: int(v) for k_, v in vehicles["outcome"].value_counts().items()},
            "cuts": int(len(cuts)),
        },
        "official": {
            "curve": curve.to_dict(orient="records"),
            "holdout": holdout.to_dict(orient="records"),
            "n_repeats": d.n_repeats,
        },
        "texts": {"recommendations": texts["recommendations"], "disclaimer": texts["disclaimer"],
                  "symptom_line": texts["symptom_line"], "no_factors": texts["no_factors"]},
        "explain_variant": expl.winner,
        "audit": {
            "dev_guard": guard,
            "v3_uses_cuts_after_alert": later,
            "note_v3": ("El ranking de factores es V3 (promedio de repeticiones). En estos autos otra "
                        "repetición explica cortes posteriores a la alerta de la repetición mostrada: "
                        "el orden de los hábitos usa información que el replay todavía no vio. Los "
                        "valores y referencias sí son los de los cortes de la alerta."),
            "note_reference": ("Las señales del filtro del taller se comparan con la mediana de los sanos "
                               "de dev del mismo mercado sobre toda la ventana (descriptivo)."),
        },
        "inputs": {name: {"path": str(p.relative_to(repo_root())) if p.is_relative_to(repo_root()) else str(p),
                          "sha256_16": sha256_16(p)} for name, p in inputs.items()},
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    cuts.to_parquet(out_dir / "cuts.parquet", index=False)
    vehicles.to_parquet(out_dir / "vehicles.parquet", index=False)
    waterfall.to_parquet(out_dir / "waterfall.parquet", index=False)
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return meta


def publish(cfg: dict[str, Any], out_dir: Path) -> str:
    """Sube el bundle (y la caché de los agentes, si está) como wandb Artifact."""
    import wandb

    missing = [f for f in BUNDLE_FILES if not (out_dir / f).exists()]
    if missing:
        raise FileNotFoundError(f"El bundle está incompleto en {out_dir}: faltan {missing}")
    meta = json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))
    bcfg = cfg["bundle"]
    run = wandb.init(entity=bcfg["entity"], project=bcfg["project"], job_type="demo-bundle",
                     config={"demo_config": cfg["_config_path"], "git_commit": meta["git_commit"]})
    artifact = wandb.Artifact(bcfg["artifact"], type="dataset",
                              description="Bundle de la demo de producto (solo dev): scripts/build_demo_bundle.py",
                              metadata={"run": meta["run"], "label": meta["label"], "repeat": meta["repeat"],
                                        "threshold": meta["threshold"], "git_commit": meta["git_commit"],
                                        "created_at": meta["created_at"]})
    artifact.add_dir(str(out_dir))
    logged = run.log_artifact(artifact)
    logged.wait()
    run.finish()
    return f"{bcfg['entity']}/{bcfg['project']}/{bcfg['artifact']}:{logged.version}"


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    out_dir = ensure_dir(cfg["bundle"]["dir"])
    if args.publish_only:
        ref = publish(cfg, out_dir)
        print(f"publicado: {ref}\nFijá la versión en {cfg['_config_path']} (bundle.version) antes del deploy.")
        return 0
    meta = build(cfg, out_dir)
    c = meta["counts"]
    print(f"\n== Bundle de la demo ({meta['product_name']}) ==")
    print(f"corrida        : {meta['run']} · R{meta['repeat'] + 1} · {meta['label']} · "
          f"{meta['budget_per_1000']:g}‰ · umbral {meta['threshold']:.4f} · k = {meta['k_consecutive']}")
    print(f"autos          : {c['vehicles']} ({c['failed']} fallan, {c['healthy']} sanos) · {c['cuts']} cortes")
    print(f"desenlaces     : {c['outcomes']}")
    print(f"solo dev       : {meta['audit']['dev_guard']}")
    print(f"V3 posterior   : {meta['audit']['v3_uses_cuts_after_alert'] or 'ninguno'}")
    print(f"escrito        : {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
