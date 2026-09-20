#!/usr/bin/env python
"""Tabla comparativa de corridas, en markdown, para pegar en el chat o en el informe.

    python scripts/compare.py                        # todas las corridas de experiments/
    python scripts/compare.py --runs exp_a exp_b     # solo estas corridas
    python scripts/compare.py --model lgbm baserate  # solo las corridas de estos modelos

Lee `experiments/<run>/metrics.json` y `config.yaml`. No recalcula nada: muestra
lo que dejó `scripts/train.py`, así la tabla no puede discrepar de wandb.

Avisa cuando las corridas no son comparables (panel distinto, splits distintos o
presupuesto de falsas alarmas distinto): comparar PR-AUC entre folds distintos es
comparar cualquier cosa.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import resolve_path  # noqa: E402


def _fmt(value: Any, digits: int = 3) -> str:
    """Número corto; `—` para lo que no existe (NaN, None)."""
    if value is None:
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return "—" if math.isnan(number) else f"{number:.{digits}f}"


def _panel_mtime(panel: Any) -> float | None:
    """Cuándo se escribió por última vez el panel que la corrida declara usar."""
    if not panel:
        return None
    try:
        path = resolve_path(str(panel))
    except Exception:  # noqa: BLE001 - un path con variables sin resolver no es un error
        return None
    return path.stat().st_mtime if path.exists() else None


def load_run(run_dir: Path) -> dict[str, Any] | None:
    """Levanta una corrida; devuelve None si el directorio no es una corrida."""
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        return None
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    config_path = run_dir / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    point = metrics.get("operating_point") or {}
    summary = metrics.get("folds_summary") or {}
    oof = metrics.get("oof", {})
    # Claves nuevas (descomposición cohorte/cuándo y eje de vehículo). Las corridas
    # anteriores no las tienen: quedan en None y `_fmt` las imprime como `—`. Nunca
    # falla por una corrida vieja, porque entonces la tabla dejaría de incluirla y
    # faltar una fila cambia la conclusión.
    ceiling = metrics.get("cohort_ceiling") or {}
    within = metrics.get("pr_auc_within_failed") or {}
    when = metrics.get("when_contribution") or {}
    vehicle = metrics.get("vehicle") or {}
    aggregation = (config.get("eval") or {}).get("vehicle_read_aggregation", "mean")
    vehicle_agg = (vehicle.get("aggregations") or {}).get(aggregation) or {}
    panel = (config.get("data") or {}).get("panel")
    return {
        "run": run_dir.name,
        "model": (config.get("model") or {}).get("name", "?"),
        # Internas del aviso de panel viejo: no son métricas y no se versionan
        # (`results.py` descarta las claves con guion bajo al armar el registro).
        "_measured_at": metrics_path.stat().st_mtime,
        "_panel_built_at": _panel_mtime(panel),
        "pr_auc": oof.get("pr_auc"),
        "cohort_ceiling": ceiling.get("pr_auc"),
        "cohort_lift": ceiling.get("lift"),
        "within_failed": within.get("pr_auc"),
        "within_failed_lift": within.get("lift"),
        "when_delta": when.get("delta"),
        "vehicle_agg": aggregation,
        "vehicle_pr_auc": vehicle_agg.get("pr_auc"),
        "vehicle_lift": vehicle_agg.get("pr_auc_lift"),
        "vehicle_base_rate": vehicle.get("base_rate"),
        "pr_auc_lo": summary.get("pr_auc_lo"),
        "pr_auc_hi": summary.get("pr_auc_hi"),
        "base_rate": oof.get("base_rate"),
        "lift": oof.get("pr_auc_lift"),
        "roc_auc": oof.get("roc_auc"),
        "brier": oof.get("brier"),
        "detection_rate": point.get("detection_rate"),
        "median_lead_km": point.get("median_lead_km"),
        "budget": metrics.get("operating_point_budget_per_1000"),
        "panel": (config.get("data") or {}).get("panel"),
        "splits": (config.get("splits") or {}).get("path"),
        "n_positive": oof.get("n_positive"),
    }


def comparability_warnings(runs: list[dict[str, Any]]) -> list[str]:
    """Todo lo que invalida una comparación entre estas corridas."""
    warnings = []
    for field, label in (
        ("panel", "panel"),
        ("splits", "splits"),
        ("budget", "presupuesto de falsas alarmas"),
    ):
        values = {r[field] for r in runs if r[field] is not None}
        if len(values) > 1:
            warnings.append(f"{label} distinto entre corridas ({sorted(map(str, values))})")

    # El mismo *path* de panel no es el mismo panel: el fix del umbral de regeneraciones
    # (commit 1eab4a1) reconstruyó `panel.parquet` con las mismas filas, los mismos
    # vehículos y los mismos folds, pero otros valores en las 7 columnas `feat_*regen*`.
    # La huella de `splits.json` no lo ve —es sobre ids de vehículo, no sobre features—,
    # así que la única señal que queda es cuándo se midió cada corrida contra cuándo se
    # escribió el panel. Importa: el control se movió de 0,165 a 0,1612 y la tabla
    # separa corridas por menos que eso.
    stale = sorted(
        r["run"]
        for r in runs
        if r["_measured_at"] is not None
        and r["_panel_built_at"] is not None
        and r["_measured_at"] < r["_panel_built_at"]
    )
    if stale:
        warnings.append(
            f"{len(stale)} corrida(s) medidas contra una versión anterior del panel que "
            f"declaran usar ({', '.join(stale)}): el archivo se reescribió después. Si el "
            "cambio fue 1eab4a1 (umbral de regeneraciones), sus números no se comparan "
            "con los del resto — hay que re-correrlas"
        )
    return warnings


def render_table(runs: list[dict[str, Any]]) -> str:
    """Markdown: una fila por corrida, ordenadas por PR-AUC descendente.

    El orden lo sigue fijando el PR-AUC por fila, pero **no es el número que decide**:
    en este panel el techo de cohorte (0,2627 sobre dev) está por encima de todo lo
    medido, así que el PR-AUC por fila ordena modelos sin decir si alguno anticipa.
    Eso lo dicen las tres columnas del medio: PR-AUC entre fallados (con su lift sobre
    la tasa del subconjunto, no sobre la global), (a') y la anticipación.
    """
    ranked = sorted(
        runs,
        key=lambda r: (r["pr_auc"] is None or math.isnan(float(r["pr_auc"])), -(float(r["pr_auc"] or 0))),
    )
    budget = next((r["budget"] for r in ranked if r["budget"] is not None), None)
    lines = [
        "| Corrida | Modelo | PR-AUC (oof) | Lift vs. base | Techo cohorte | "
        "PR-AUC entre fallados | Aporte del cuándo (a') | Vehículo (lift) | "
        "ROC-AUC | Brier | Detección | Anticip. mediana |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in ranked:
        detection = "—" if r["detection_rate"] is None else f"{100 * float(r['detection_rate']):.0f}%"
        lead = "—" if r["median_lead_km"] is None or math.isnan(float(r["median_lead_km"] or math.nan)) \
            else f"{float(r['median_lead_km']):,.0f} km"
        # El techo se muestra con una marca de si el PR-AUC por fila lo supera: es la
        # lectura que la tabla vieja no permitía hacer de un vistazo.
        ceiling = _fmt(r["cohort_ceiling"])
        if r["cohort_ceiling"] is not None and r["pr_auc"] is not None:
            ceiling += " ✓" if float(r["pr_auc"]) > float(r["cohort_ceiling"]) else " ✗"
        within = _fmt(r["within_failed"])
        if r["within_failed_lift"] is not None:
            within += f" ({_fmt(r['within_failed_lift'], 2)}×)"
        when = "—" if r["when_delta"] is None else f"{float(r['when_delta']):+.4f}"
        vehicle = "—" if r["vehicle_lift"] is None else f"{_fmt(r['vehicle_lift'], 2)}×"
        if r["vehicle_lift"] is not None:
            vehicle += f" ({r['vehicle_agg']})"
        lines.append(
            f"| `{r['run']}` | {r['model']} | {_fmt(r['pr_auc'])} | {_fmt(r['lift'], 2)}× | "
            f"{ceiling} | {within} | {when} | {vehicle} | "
            f"{_fmt(r['roc_auc'])} | {_fmt(r['brier'])} | {detection} | {lead} |"
        )
    base = next((r["base_rate"] for r in ranked if r["base_rate"] is not None), None)
    budgets = {r["budget"] for r in ranked if r["budget"] is not None}
    budget_text = (
        f"≤ {_fmt(budget, 0)} falsas alarmas/1000 sanos"
        if len(budgets) <= 1
        else "un presupuesto de falsas alarmas distinto por corrida (ver aviso)"
    )
    ceiling = next((r["cohort_ceiling"] for r in ranked if r["cohort_ceiling"] is not None), None)
    ceiling_lift = next((r["cohort_lift"] for r in ranked if r["cohort_lift"] is not None), None)
    vehicle_base = next(
        (r["vehicle_base_rate"] for r in ranked if r["vehicle_base_rate"] is not None), None
    )
    foot = [
        "",
        f"Tasa base: {_fmt(base)} · eventos positivos: {ranked[0]['n_positive'] or '—'} · "
        f"detección y anticipación al punto de operación de {budget_text}.",
    ]
    if ceiling is not None:
        foot += [
            "",
            f"**Techo de cohorte: {_fmt(ceiling)}** (lift {_fmt(ceiling_lift, 2)}× sobre la tasa "
            "base). Es lo que saca un modelo que solo sabe *qué* vehículos fallan, sin nada "
            "del *cuándo*: `✓`/`✗` en la columna marca si el PR-AUC por fila lo supera. Un "
            "`✗` no invalida la corrida, pero significa que **ese PR-AUC no demuestra "
            "anticipación** — y que un lift de 1,6–2× tampoco, porque el techo ya está ahí.",
            "",
            "«PR-AUC entre fallados» se mide solo sobre los cortes de vehículos con evento, "
            f"donde el azar es {_fmt(ceiling)} y no la tasa base: se lee el lift entre "
            "paréntesis, no el número pelado. «Aporte del "
            "cuándo (a')» es lo que se pierde al colapsar el score al promedio de cada "
            "vehículo sin reentrenar; negativo = el orden dentro del vehículo resta.",
        ]
    if vehicle_base is not None:
        foot += [
            "",
            f"El eje de vehículo tiene su propia tasa base ({_fmt(vehicle_base)}), así que se "
            "compara el lift y nunca el PR-AUC contra el de filas.",
        ]
    foot += [
        "",
        "Intervalo bootstrap de PR-AUC por fold: "
        + " · ".join(f"`{r['run']}` [{_fmt(r['pr_auc_lo'])}, {_fmt(r['pr_auc_hi'])}]" for r in ranked),
    ]
    for warning in comparability_warnings(ranked):
        foot.append(f"\n**No comparable:** {warning}.")
    return "\n".join(lines + foot)


def main() -> int:
    parser = argparse.ArgumentParser(description="Tabla comparativa de corridas en markdown")
    parser.add_argument("--dir", default="experiments", help="Directorio de corridas")
    parser.add_argument("--runs", nargs="*", default=None, help="Comparar solo estas corridas (nombre del directorio)")
    parser.add_argument("--model", nargs="*", default=None, help="Comparar solo las corridas de estos modelos")
    args = parser.parse_args()

    root = resolve_path(args.dir)
    if not root.exists():
        print(f"No existe {root}: todavía no hay corridas.", file=sys.stderr)
        return 1

    if args.runs:
        # Un nombre mal escrito no se ignora: faltar una fila cambia la conclusión.
        missing = [name for name in args.runs if load_run(root / name) is None]
        if missing:
            available = sorted(p.name for p in root.iterdir() if (p / "metrics.json").exists())
            print(f"No existen estas corridas en {root}: {missing}", file=sys.stderr)
            print(f"Disponibles: {available}", file=sys.stderr)
            return 1
        candidates = [root / name for name in args.runs]
    else:
        candidates = sorted(p for p in root.iterdir() if p.is_dir())

    runs = [run for run in (load_run(path) for path in candidates) if run is not None]

    if args.model:
        wanted = set(args.model)
        found = {r["model"] for r in runs}
        if not wanted <= found:
            print(f"Sin corridas para estos modelos: {sorted(wanted - found)}", file=sys.stderr)
            print(f"Modelos con corridas en {root}: {sorted(found)}", file=sys.stderr)
            return 1
        runs = [r for r in runs if r["model"] in wanted]

    if not runs:
        print(f"Ningún metrics.json en {root}. Corré `scripts/train.py` primero.", file=sys.stderr)
        return 1

    print(render_table(runs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
