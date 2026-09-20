#!/usr/bin/env python
"""Registro versionado de resultados: una corrida = un archivo en `results/`.

`experiments/` y wandb no se versionan (regla 8), así que un resultado que no se
anota acá se pierde con la máquina. Cada registro guarda las métricas resumen, la
**config resuelta completa** (no solo su path: el YAML puede vivir en otra rama o
haber cambiado después) y de dónde salió (rama, commit, si el YAML está versionado).

    python scripts/results.py log f3-lgbm-panel-v1 --note "control de las ablaciones"
    python scripts/results.py log --all              # anota lo que falte en results/
    python scripts/results.py table                  # tabla markdown ordenada por PR-AUC
    python scripts/results.py table --out results/README.md
    python scripts/results.py show f3-lgbm-panel-v1  # config para reproducir la corrida

Un archivo por corrida evita conflictos de merge entre los tres tracks. La tabla
se regenera; no se edita a mano. Las notas sí: son del registro (`--note`, o
editando `note:` en el YAML).
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from compare import _fmt, load_run  # noqa: E402
from src.config import resolve_path  # noqa: E402

RESULTS_DIR = "results"


def _git(*args: str) -> str:
    try:
        out = subprocess.run(["git", *args], capture_output=True, text=True, check=True,
                             cwd=resolve_path("."))
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def _config_versioned(config_path: str | None) -> bool:
    """¿El YAML de la corrida está commiteado en este checkout y sin cambios?"""
    if not config_path:
        return False
    tracked = _git("ls-files", "--error-unmatch", config_path) != ""
    return tracked and _git("status", "--porcelain", "--", config_path) == ""


def build_record(run_dir: Path, note: str | None, previous: dict[str, Any] | None) -> dict[str, Any]:
    summary = load_run(run_dir)
    if summary is None:
        raise SystemExit(f"{run_dir} no tiene metrics.json: no hay nada que anotar.")
    config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8")) or {}
    config_path = config.get("_config_path")
    return {
        "run": summary["run"],
        "logged_on": date.today().isoformat(),
        "note": note if note is not None else (previous or {}).get("note", ""),
        "source": {
            "config_path": config_path,
            "config_versioned": _config_versioned(config_path),
            "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "commit": _git("rev-parse", "--short", "HEAD"),
            "note": "rama y commit del checkout donde se anotó, no necesariamente donde se entrenó",
        },
        "summary": {k: v for k, v in summary.items() if k != "run"},
        "config": config,
    }


def _record_path(name: str) -> Path:
    return resolve_path(RESULTS_DIR) / f"{name}.yaml"


def _load_records() -> list[dict[str, Any]]:
    root = resolve_path(RESULTS_DIR)
    if not root.exists():
        return []
    return [yaml.safe_load(p.read_text(encoding="utf-8")) for p in sorted(root.glob("*.yaml"))]


def cmd_log(args: argparse.Namespace) -> int:
    experiments = resolve_path(args.dir)
    if args.all:
        names = sorted(p.name for p in experiments.iterdir() if (p / "metrics.json").exists())
        names = [n for n in names if not _record_path(n).exists()]
        if not names:
            print("Todas las corridas de experiments/ ya están en results/.")
            return 0
    else:
        if not args.run:
            print("Indicá una corrida o usá --all.", file=sys.stderr)
            return 1
        names = [args.run]

    for name in names:
        path = _record_path(name)
        previous = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else None
        if previous and not args.force and not args.all:
            print(f"{path} ya existe: --force para reescribirlo (la nota se conserva).", file=sys.stderr)
            return 1
        record = build_record(experiments / name, args.note, previous)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(record, sort_keys=False, allow_unicode=True), encoding="utf-8")
        flag = "" if record["source"]["config_versioned"] else "  ⚠ el YAML no está versionado en este checkout (queda la copia embebida)"
        print(f"anotada: {path.relative_to(resolve_path('.'))}{flag}")
    return 0


def cmd_table(args: argparse.Namespace) -> int:
    records = _load_records()
    if not records:
        print(f"No hay registros en {RESULTS_DIR}/. Corré `python scripts/results.py log --all`.", file=sys.stderr)
        return 1

    def pr_auc(r: dict[str, Any]) -> float:
        v = r["summary"].get("pr_auc")
        return -1.0 if v is None or math.isnan(float(v)) else float(v)

    lines = [
        "# Resultados por corrida",
        "",
        "Generado por `python scripts/results.py table`; no editar a mano (las notas viven en cada `results/<corrida>.yaml`).",
        "PR-AUC out-of-fold. Solo son comparables las corridas con la misma tasa base (mismas filas), splits y presupuesto de falsas alarmas:",
        "un PR-AUC más bajo con otra tasa base puede ser un lift mayor. Las corridas `timesfm3` (zero-shot) no producen estas métricas.",
        "",
        "| Corrida | Modelo | Panel | Tasa base | PR-AUC [IC fold] | Lift | Detección | Anticip. mediana | Config | Nota |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in sorted(records, key=pr_auc, reverse=True):
        s = r["summary"]
        det = "—" if s.get("detection_rate") is None else f"{100 * float(s['detection_rate']):.0f}%"
        lead = s.get("median_lead_km")
        lead = "—" if lead is None or math.isnan(float(lead)) else f"{float(lead):,.0f} km"
        panel = Path(str(s.get("panel") or "—")).name
        cfg = r["source"]["config_path"] or "—"
        cfg_cell = f"`{cfg}`" + ("" if r["source"]["config_versioned"] else " ⚠")
        lines.append(
            f"| `{r['run']}` | {s.get('model', '?')} | {panel} | {_fmt(s.get('base_rate'))} | {_fmt(s.get('pr_auc'))} "
            f"[{_fmt(s.get('pr_auc_lo'))}, {_fmt(s.get('pr_auc_hi'))}] | {_fmt(s.get('lift'), 2)}× | "
            f"{det} | {lead} | {cfg_cell} | {r.get('note') or ''} |"
        )
    lines += [
        "",
        "⚠ = el YAML no estaba commiteado al anotar; la config completa está embebida en `results/<corrida>.yaml`",
        "(`python scripts/results.py show <corrida>` la imprime lista para guardarse como `configs/exp_*.yaml`).",
    ]
    text = "\n".join(lines) + "\n"
    if args.out:
        out = resolve_path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(f"escrito: {args.out}")
    else:
        print(text)
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    path = _record_path(args.run)
    if not path.exists():
        print(f"No hay registro de {args.run} en {RESULTS_DIR}/.", file=sys.stderr)
        return 1
    record = yaml.safe_load(path.read_text(encoding="utf-8"))
    config = {k: v for k, v in record["config"].items() if not k.startswith("_")}
    print(f"# {record['run']} · anotada {record['logged_on']} · fuente {record['source']['config_path']}")
    print(yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Registro versionado de resultados por corrida")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_log = sub.add_parser("log", help="anotar una corrida (o todas las que falten)")
    p_log.add_argument("run", nargs="?", help="nombre del directorio en experiments/")
    p_log.add_argument("--all", action="store_true", help="anotar todas las corridas sin registro")
    p_log.add_argument("--note", default=None, help="qué se probó y qué se concluyó")
    p_log.add_argument("--force", action="store_true", help="reescribir un registro existente")
    p_log.add_argument("--dir", default="experiments")
    p_log.set_defaults(func=cmd_log)

    p_table = sub.add_parser("table", help="tabla markdown de todos los registros")
    p_table.add_argument("--out", default=None, help="escribir en este archivo en vez de stdout")
    p_table.set_defaults(func=cmd_table)

    p_show = sub.add_parser("show", help="imprimir la config de una corrida")
    p_show.add_argument("run")
    p_show.set_defaults(func=cmd_show)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
