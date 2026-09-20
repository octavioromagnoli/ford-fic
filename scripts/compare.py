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

# La consola de Windows es cp1252 y la tabla lleva "≤": sin esto, `compare.py`
# muere con UnicodeEncodeError en vez de imprimir. Mismo arreglo que el resto de
# los scripts del repo.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

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
    return {
        "run": run_dir.name,
        "model": (config.get("model") or {}).get("name", "?"),
        "pr_auc": oof.get("pr_auc"),
        "pr_auc_lo": summary.get("pr_auc_lo"),
        "pr_auc_hi": summary.get("pr_auc_hi"),
        "base_rate": oof.get("base_rate"),
        "lift": oof.get("pr_auc_lift"),
        "pr_auc_norm": oof.get("pr_auc_norm"),
        "f1": oof.get("f1"),
        "f1_trivial": oof.get("f1_trivial"),
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
    return warnings


def render_table(runs: list[dict[str, Any]]) -> str:
    """Markdown: una fila por corrida, ordenadas por PR-AUC descendente."""
    ranked = sorted(
        runs,
        key=lambda r: (r["pr_auc"] is None or math.isnan(float(r["pr_auc"])), -(float(r["pr_auc"] or 0))),
    )
    budget = next((r["budget"] for r in ranked if r["budget"] is not None), None)
    lines = [
        # `PR-AUC norm` y `F1 trivial` están porque PR-AUC y F1 dependen de la prevalencia:
        # entre paneles con tasas base distintas, las columnas crudas no se comparan.
        f"| Corrida | Modelo | PR-AUC (oof) | Lift vs. base | PR-AUC norm | F1 | F1 trivial | ROC-AUC | Brier | Detección | Anticip. mediana |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in ranked:
        detection = "—" if r["detection_rate"] is None else f"{100 * float(r['detection_rate']):.0f}%"
        lead = "—" if r["median_lead_km"] is None or math.isnan(float(r["median_lead_km"] or math.nan)) \
            else f"{float(r['median_lead_km']):,.0f} km"
        lines.append(
            f"| `{r['run']}` | {r['model']} | {_fmt(r['pr_auc'])} | {_fmt(r['lift'], 2)}× | "
            f"{_fmt(r['pr_auc_norm'])} | {_fmt(r['f1'])} | {_fmt(r['f1_trivial'])} | "
            f"{_fmt(r['roc_auc'])} | {_fmt(r['brier'])} | {detection} | {lead} |"
        )
    base = next((r["base_rate"] for r in ranked if r["base_rate"] is not None), None)
    budgets = {r["budget"] for r in ranked if r["budget"] is not None}
    budget_text = (
        f"≤ {_fmt(budget, 0)} falsas alarmas/1000 sanos"
        if len(budgets) <= 1
        else "un presupuesto de falsas alarmas distinto por corrida (ver aviso)"
    )
    foot = [
        "",
        f"Tasa base: {_fmt(base)} · eventos positivos: {ranked[0]['n_positive'] or '—'} · "
        f"detección y anticipación al punto de operación de {budget_text}.",
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
