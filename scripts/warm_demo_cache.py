#!/usr/bin/env python
"""Corre el agente de triage en todas las semanas con eventos y deja el resultado en el bundle.

    python scripts/warm_demo_cache.py --config configs/demo.yaml            # modo de configs/agents.yaml
    python scripts/warm_demo_cache.py --config configs/demo.yaml --weeks 2025-09-29   # una semana
    python scripts/warm_demo_cache.py --config configs/demo.yaml --prune   # antes de publicar el bundle

Guarda `triage/<lunes>.json` y la caché de cada llamada (`llm_cache/`) dentro del bundle, así la
app desplegada muestra los textos del agente sin llamar a la API (y la demo funciona sin red). Las
semanas sin eventos no se corren: su resumen es la plantilla.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.agents.bundle import bundle_dir, load_bundle  # noqa: E402
from src.agents.facts import events_for  # noqa: E402
from src.agents.llm import LLM  # noqa: E402
from src.agents.triage import run_triage, save_triage  # noqa: E402
from src.config import load_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/demo.yaml")
    parser.add_argument("--agents", default="configs/agents.yaml")
    parser.add_argument("--mode", choices=["cache_first", "cache_only", "live"], help="pisa llm.mode")
    parser.add_argument("--weeks", help="lunes separados por coma (por defecto, todas las semanas con eventos)")
    parser.add_argument("--prune", action="store_true",
                        help="borra de la caché lo que esta corrida no usó (respuestas de prompts viejos)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg, acfg = load_config(args.config), load_config(args.agents)
    bundle = load_bundle(bundle_dir(cfg))
    events = events_for(bundle, acfg)
    llm = LLM(acfg["llm"], bundle.root / acfg["llm"]["cache_subdir"], mode=args.mode)
    if not llm.available and llm.mode != "cache_only":
        print("No hay OPENAI_API_KEY (entorno o .env): no se puede correr el agente.")
        return 1
    weeks = ([pd.Timestamp(w) for w in args.weeks.split(",")] if args.weeks
             else sorted({e.week for e in events}))
    print(f"modelo {llm.model} · modo {llm.mode} · {len(weeks)} semana(s) · bundle {bundle.root}")
    failures = 0
    for week in weeks:
        before = dict(llm.calls)
        result = run_triage(bundle, events, week, acfg, llm)
        save_triage(bundle, result)
        sources = [r["draft"]["source"] for r in result["events"]]
        problems = sum(bool(p) for r in result["events"] for p in r["draft"]["problems"])
        api = llm.calls["api"] - before["api"]
        cache = llm.calls["cache"] - before["cache"]
        failures += result["summary"]["source"] != "agente" or any(s != "agente" for s in sources)
        print(f"{week.date()} | {len(result['events'])} evento(s) | textos {sources} | resumen {result['summary']['source']} "
              f"| rechazos del verificador {problems} | API {api}, caché {cache}")
    if args.prune and not args.weeks:
        stale = [p for p in llm.cache_dir.glob("*.json") if p.name not in llm.used]
        for p in stale:
            p.unlink()
        print(f"caché podada: {len(stale)} respuesta(s) sin usar, quedan {len(llm.used)}")
    print(f"\nlisto: {llm.calls['api']} llamadas a la API, {llm.calls['cache']} desde la caché. "
          f"Semanas con alguna plantilla: {failures}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
