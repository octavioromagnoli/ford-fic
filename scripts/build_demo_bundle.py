#!/usr/bin/env python
"""Bundle de la demo de producto: lo que la app y los agentes leen, precalculado y solo de dev.

    FORD_DATA_DIR=$PWD/data/rebuild-0921 python scripts/build_demo_bundle.py --config configs/demo.yaml
    python scripts/build_demo_bundle.py --config configs/demo.yaml --publish-only   # wandb Artifact

La app desplegada no importa torch ni sklearn: lee este directorio. Acá se hace todo lo que necesita el stack
completo (la corrida de `model.run_config` y su capa de decisión) y nada se recalcula distinto: el umbral y la alerta
salen de las mismas funciones que el dashboard de F4 (`src/eval/dashboard_data.py`), y el mensaje, del mismo render
que la explicabilidad (`src/eval/explain.py::render_vehicle_message`).

El "por qué" es `fleet_profile` (`src/eval/fleet_profile.py`): el modelo de la demo no tiene atribución, así que se
describe en qué hábitos se aparta el auto de los autos sanos de dev de su mercado. Es una comparación con la flota,
no lo que el modelo usó (`docs/memoria/f9-demo-gru.md`).

Deja en `bundle.dir`:
- cuts.parquet        una fila por corte de dev evaluable con la etiqueta elegida: score de la repetición, fecha,
                      mercado y perfil de uso
- vehicles.parquet    una fila por auto: evento (fecha y odómetro), alerta de la repetición, hábitos que se apartan
                      de la flota sana y mensaje, señales del filtro contra la flota sana (taller)
- deviations.parquet  una fila por (auto, hábito nombrable): valor, mediana de los sanos del mercado, qué parte de
                      esos sanos supera hacia el lado riesgoso y si se nombra
- meta.json           modelo y tipo de "por qué", umbral, regla, números oficiales de la capa de decisión, ventana
                      del replay, textos fijos y procedencia (commit y hash de cada insumo)

Un directorio `llm_cache/` o `triage/` que ya esté en `bundle.dir` (lo deja `scripts/warm_demo_cache.py`) se
conserva y viaja con `--publish-only`.
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

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.agents.bundle import FLEET_PROFILE, REQUIRED  # noqa: E402
from src.agents.formatting import format_value  # noqa: E402
from src.agents.policy import monday  # noqa: E402
from src.config import ensure_dir, load_config, repo_root, resolve_path  # noqa: E402
from src.data.anchor import event_dates  # noqa: E402
from src.data.join import load_vehicle_static  # noqa: E402
from src.eval import explain as ex  # noqa: E402
from src.eval import fleet_profile as fp  # noqa: E402
from src.eval.dashboard_data import (KEY, curve_summary, holdout_summary, load_run, operating_threshold,  # noqa: E402
                                     rows_for, vehicle_alerts)
from src.eval.metrics import _first_sustained_index  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger("build_demo_bundle")

#: El ritmo del auto: no es un hábito que se nombre, convierte el horizonte en km a semanas.
KM_PER_DAY = "feat_km_per_day"


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


def technician_signals(rows: pd.DataFrame, explained: pd.DataFrame, signals: dict[str, Any]) -> dict[str, list]:
    """Las señales del filtro de cada auto en sus cortes explicados, contra los sanos de su mercado.

    La referencia es la mediana, entre los autos sanos de dev del mismo mercado, de la mediana de
    cada auto (así un auto con muchos cortes no pesa más). Es descriptiva: la ve el taller.
    """
    cols = list(signals)
    _, reference = fp.healthy_market_reference(rows, cols)
    means = fp.window_values(rows, explained, cols)
    out: dict[str, list] = {}
    for vid, r in means.iterrows():
        ref = reference.loc[r[fp.MARKET]]
        items = []
        for col, spec in signals.items():
            scale = float(spec.get("scale", 1.0))
            value, refv = float(r[col]), float(ref[col])
            items.append({"feature": col, "label": spec["label"], "value": value, "reference": refv,
                          "value_text": format_value(value * scale, spec["format"]),
                          "reference_text": format_value(refv * scale, spec["format"])})
        out[str(vid)] = items
    return out


def describe_deviations(deviations: pd.DataFrame, texts: dict[str, Any]) -> pd.DataFrame:
    """Los desvíos con el nombre del hábito, su recomendación y los valores ya formateados (lo que muestra la app)."""
    feature_texts = texts.get("features") or {}
    t = deviations["feature"].map(lambda f: feature_texts.get(f) or {})
    out = deviations.assign(label=t.map(lambda x: x.get("label")), format=t.map(lambda x: x.get("format", "")),
                            recommendation=t.map(lambda x: x.get("recommendation")))
    out["value_text"] = [format_value(v, f) for v, f in zip(out["value"], out["format"])]
    out["reference_text"] = [format_value(v, f) for v, f in zip(out["reference"], out["format"])]
    return out


def habit_records(named: pd.DataFrame) -> list[dict[str, Any]]:
    """Los hábitos que nombra el mensaje, en orden: lo que leen los hechos del agente y la ficha."""
    return [{"feature": r.feature, "label": r.label, "format": r.format, "value": float(r.value),
             "reference": float(r.reference), "value_text": r.value_text, "reference_text": r.reference_text,
             "recommendation": r.recommendation, "share": float(r.share)}
            for r in named.sort_values("rank").itertuples()]


def weeks_with_alerts(vehicles: pd.DataFrame) -> dict[str, int]:
    """Alertas nuevas confirmadas por semana del replay (lunes): con esto se elige `replay.first_week`."""
    alerted = vehicles.loc[vehicles["alerted"], "alert_confirm_date"].dropna()
    counts = alerted.map(monday).value_counts().sort_index()
    return {w.date().isoformat(): int(n) for w, n in counts.items()}


def build(cfg: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    mcfg, ecfg = cfg["model"], cfg["explanation"]
    if ecfg.get("type") != FLEET_PROFILE:
        raise ValueError(f"`explanation.type` tiene que ser `{FLEET_PROFILE}` (es `{ecfg.get('type')}`): el modelo de "
                         "la demo no tiene atribución, el porqué es una comparación con la flota sana")
    run_cfg = load_config(mcfg["run_config"])
    label, repeat, budget = cfg["label"], int(cfg["repeat"]), float(cfg["budget_per_1000"])
    profile = list(dict.fromkeys(cfg["profile_features"]))
    if KM_PER_DAY not in profile:
        raise ValueError(f"`profile_features` tiene que traer `{KM_PER_DAY}`: pasa el horizonte a semanas")
    d = load_run(mcfg["run_config"], decision_layer=mcfg["decision_layer"],
                 columns=[*profile, *cfg["technician_signals"]], columns_panel=mcfg.get("profile_panel"))
    if d.decision is None:
        raise FileNotFoundError(f"Falta la capa de decisión en {mcfg['decision_layer']}: corré scripts/decision_layer.py")
    k = int(d.eval_cfg.get("k_consecutive", 2))
    rows = rows_for(d, label)

    # -- el umbral y la alerta: los del dashboard, en la repetición elegida ------------------------
    point = operating_threshold(rows, repeat, budget, d.eval_cfg)
    if point is None:
        raise RuntimeError(f"Ningún umbral respeta {budget:g} falsas alarmas cada 1.000 en R{repeat + 1}")
    threshold = float(point["threshold"])
    alerts = vehicle_alerts(rows, repeat, threshold, k)
    timing = alert_timing(rows, repeat, threshold, k)
    logger.info("%s · R%d · %s · %.0f‰ | umbral %.4f | %s", run_cfg["name"], repeat + 1, label, budget, threshold,
                alerts["outcome"].value_counts().to_dict())

    # -- cortes: score de la repetición + perfil de uso --------------------------------------------
    cuts = rows[KEY + ["cut_date", "static_SalesCountry_cd", "event_observed", "time_to_event_km"] + profile].copy()
    cuts["score"] = rows[f"score_r{repeat}"].to_numpy()
    cuts = cuts.rename(columns={"static_SalesCountry_cd": "market"}).sort_values(KEY, ignore_index=True)

    # -- el porqué: en qué hábitos se aparta de los sanos de su mercado (no lo que usó el modelo) --
    gap_km, horizon_km = float(rows["gap_km"].iloc[0]), float(rows["horizon_km"].iloc[0])
    classification = load_config(ecfg["classification"])
    texts = load_config(ecfg["texts"])
    habits = fp.nameable_habits(profile, classification["features"], texts)
    if not habits:
        raise ValueError("Ningún hábito de `profile_features` es accionable, con signo físico y con texto")
    explained = ex.explained_cuts(rows, repeat, threshold, k)
    per_vehicle, reference = fp.healthy_market_reference(rows, list(habits))
    values = fp.window_values(rows, explained, [*habits, KM_PER_DAY])
    deviations = fp.fleet_factors(
        fp.fleet_deviations(values, per_vehicle, reference, habits, min_healthy=int(ecfg["min_healthy_vehicles"])),
        min_share=float(ecfg["min_healthy_share"]), max_factors=int(ecfg["max_factors"]))
    deviations = describe_deviations(deviations, texts)
    signals = technician_signals(rows, explained, cfg["technician_signals"])
    specs = ex.feature_specs(classification, {})
    message_texts = {**texts, **ecfg["message"]}
    risk = explained.groupby("vehicle_id")["risk_level"].first()

    events = event_calendar(cfg, alerts["vehicle_id"])
    vehicles = alerts.merge(timing, on="vehicle_id", how="left")
    vehicles["event_date"] = vehicles["vehicle_id"].map(events)
    records = []
    for vid in vehicles["vehicle_id"]:
        named = deviations.loc[deviations["vehicle_id"].eq(vid) & deviations["named"]].sort_values("rank")
        kmpd = float(values.loc[vid, KM_PER_DAY])
        weeks = ex.horizon_weeks(gap_km, horizon_km, kmpd)
        message = ex.render_vehicle_message(
            risk_level=risk[vid], specs=specs, texts=message_texts, k=k, gap_km=gap_km, horizon_km=horizon_km,
            km_per_day=kmpd, symptom_contribution=None,
            factors=[ex.MessageFactor(r.feature, float(r.share), float(r.value), float(r.reference))
                     for r in named.itertuples()])
        records.append({
            "vehicle_id": vid, "risk_level": risk[vid], "n_factors": int(len(named)),
            "factors": json.dumps(habit_records(named), ensure_ascii=False),
            "template_message": message, "km_per_day": kmpd,
            "horizon_weeks_lo": weeks[0] if weeks else None, "horizon_weeks_hi": weeks[1] if weeks else None,
            "technician_signals": json.dumps(signals.get(str(vid), []), ensure_ascii=False),
        })
    vehicles = vehicles.merge(pd.DataFrame(records), on="vehicle_id", how="left", validate="one_to_one")

    guard = dev_guard(pd.concat([cuts["vehicle_id"], vehicles["vehicle_id"], deviations["vehicle_id"]]),
                      resolve_path(run_cfg["splits"]["test_split"]))

    # -- números oficiales: la capa de decisión, sin recalcular -------------------------------------
    curve = curve_summary(d.decision, label)
    holdout = holdout_summary(d.decision, label)
    run_dir = resolve_path(run_cfg.get("output_dir", "experiments")) / run_cfg["name"]
    inputs = {
        "predictions": run_dir / "predictions.parquet", "window_eval": run_dir / "window_eval.json",
        "decision_layer": resolve_path(mcfg["decision_layer"]), "panel": resolve_path(run_cfg["data"]["panel"]),
        "classification": resolve_path(ecfg["classification"]), "texts": resolve_path(ecfg["texts"]),
    }
    if mcfg.get("profile_panel"):
        inputs["profile_panel"] = resolve_path(mcfg["profile_panel"])
    replay = cfg["replay"]
    meta = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "config": cfg["_config_path"],
        "product_name": cfg["product_name"],
        "model": {"name": mcfg["name"], "family": mcfg["family"], "run": run_cfg["name"],
                  "run_config": mcfg["run_config"]},
        "explanation": {
            "type": FLEET_PROFILE, "habits": list(habits), "min_healthy_share": float(ecfg["min_healthy_share"]),
            "max_factors": int(ecfg["max_factors"]),
            "reference": "mediana de los autos sanos de dev del mismo mercado (una mediana por auto)",
            "cuts": "los cortes que dispararon la alerta; el de score máximo si el auto no alertó",
        },
        "run": run_cfg["name"],
        "label": label, "repeat": repeat, "budget_per_1000": budget,
        "threshold": threshold, "k_consecutive": k, "gap_km": gap_km, "horizon_km": horizon_km,
        "replay": {"start": replay["start"], "end": replay["end"], "first_week": replay["first_week"]},
        "counts": {
            "vehicles": int(len(vehicles)), "failed": int(vehicles["failed"].sum()),
            "healthy": int((~vehicles["failed"]).sum()),
            "outcomes": {k_: int(v) for k_, v in vehicles["outcome"].value_counts().items()},
            "cuts": int(len(cuts)),
            "alerted_with_habits": int((vehicles["alerted"] & vehicles["n_factors"].gt(0)).sum()),
        },
        "official": {
            "curve": curve.to_dict(orient="records"),
            "holdout": holdout.to_dict(orient="records"),
            "n_repeats": d.n_repeats,
        },
        "texts": {"recommendations": texts["recommendations"], "disclaimer": message_texts["disclaimer"],
                  "factors_intro": message_texts["factors_intro"], "no_factors": message_texts["no_factors"]},
        "audit": {
            "dev_guard": guard,
            "note_explanation": ("El porqué compara el perfil de uso del auto en los cortes de la alerta con la mediana "
                                 "de los autos sanos de dev de su mercado: no es lo que usó el modelo, que no tiene "
                                 "atribución. Solo nombra hábitos accionables del lado que la física del DPF señala "
                                 "como riesgoso."),
            "note_reference": ("Los hábitos y las señales del filtro del taller se comparan con la mediana de los sanos "
                               "de dev del mismo mercado sobre toda la ventana (descriptivo)."),
        },
        "inputs": {name: {"path": str(p.relative_to(repo_root())) if p.is_relative_to(repo_root()) else str(p),
                          "sha256_16": sha256_16(p)} for name, p in inputs.items()},
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    stale = out_dir / "waterfall.parquet"   # el bundle de la demo anterior (con SHAP): no viaja más
    if stale.exists():
        stale.unlink()
    cuts.to_parquet(out_dir / "cuts.parquet", index=False)
    vehicles.to_parquet(out_dir / "vehicles.parquet", index=False)
    deviations.to_parquet(out_dir / "deviations.parquet", index=False)
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    meta["_weeks_with_alerts"] = weeks_with_alerts(vehicles)
    return meta


def publish(cfg: dict[str, Any], out_dir: Path) -> str:
    """Sube el bundle (y la caché de los agentes, si está) como wandb Artifact."""
    import wandb

    missing = [f for f in REQUIRED if not (out_dir / f).exists()]
    if missing:
        raise FileNotFoundError(f"El bundle está incompleto en {out_dir}: faltan {missing}")
    meta = json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))
    bcfg = cfg["bundle"]
    run = wandb.init(entity=bcfg["entity"], project=bcfg["project"], job_type="demo-bundle",
                     config={"demo_config": cfg["_config_path"], "git_commit": meta["git_commit"]})
    artifact = wandb.Artifact(bcfg["artifact"], type="dataset",
                              description="Bundle de la demo de producto (solo dev): scripts/build_demo_bundle.py",
                              metadata={"model": meta["model"]["name"], "run": meta["run"],
                                        "explanation": meta["explanation"]["type"], "label": meta["label"],
                                        "repeat": meta["repeat"], "threshold": meta["threshold"],
                                        "git_commit": meta["git_commit"], "created_at": meta["created_at"]})
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
    print(f"modelo         : {meta['model']['name']} ({meta['model']['family']}) · corrida {meta['run']}")
    print(f"punto          : R{meta['repeat'] + 1} · {meta['label']} · {meta['budget_per_1000']:g}‰ · "
          f"umbral {meta['threshold']:.4f} · k = {meta['k_consecutive']}")
    print(f"autos          : {c['vehicles']} ({c['failed']} fallan, {c['healthy']} sanos) · {c['cuts']} cortes")
    print(f"desenlaces     : {c['outcomes']}")
    print(f"porqué         : {meta['explanation']['type']} · alertas con hábitos que nombrar: {c['alerted_with_habits']}")
    print(f"alertas nuevas por semana: {meta['_weeks_with_alerts'] or 'ninguna'} "
          f"(first_week = {meta['replay']['first_week']})")
    print(f"solo dev       : {meta['audit']['dev_guard']}")
    print(f"escrito        : {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
