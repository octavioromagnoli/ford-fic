#!/usr/bin/env python
"""Corre el agente de triage en todas las semanas con eventos, en cada punto de operación, y deja el resultado en el bundle.

    python scripts/warm_demo_cache.py --config configs/demo.yaml            # modo de configs/agents.yaml
    python scripts/warm_demo_cache.py --config configs/demo.yaml --budgets 100 --weeks 2025-09-29   # una semana
    python scripts/warm_demo_cache.py --config configs/demo.yaml --prune   # antes de publicar el bundle

Guarda `triage/faNNN/<lunes>.json` (uno por punto de operación de la perilla) y la caché de cada llamada
(`llm_cache/`, una sola) dentro del bundle, así la app desplegada muestra los textos del agente sin llamar a la API
(y la demo funciona sin red). Las semanas sin eventos no se corren: su resumen es la plantilla.

Los puntos se pueden correr en paralelo con `--budgets` (un proceso por punto), pero la caché es una sola: si dos
procesos piden lo mismo a la vez, el segundo pisa la respuesta del primero y el resto de ese recorrido queda
huérfano. Por eso, al final, una pasada con todos en `--mode cache_only --prune`: verifica que cada semana salga
entera de la caché y borra lo que ningún punto usó. Una semana a la que le falta algo no se guarda (el triage que ya
estaba queda) y se lista para rehacerla en un solo proceso; con alguna faltante, no se poda.
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


def agent_missing(result: dict) -> bool:
    """El agente no corrió entero: sin key, o en `cache_only` sin la respuesta guardada (no es un rechazo del
    verificador, que también deja la plantilla pero con el agente corriendo)."""
    return result["mode"] != "agente" or any(r["draft"]["note"].startswith("Sin agente") for r in result["events"])
from src.config import load_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/demo.yaml")
    parser.add_argument("--agents", default="configs/agents.yaml")
    parser.add_argument("--mode", choices=["cache_first", "cache_only", "live"], help="pisa llm.mode")
    parser.add_argument("--budgets", help="puntos de operación (falsas alarmas cada 1.000 sanos) separados por coma; "
                                          "por defecto, todos los del bundle")
    parser.add_argument("--weeks", help="lunes separados por coma (por defecto, todas las semanas con eventos)")
    parser.add_argument("--prune", action="store_true",
                        help="borra de la caché lo que esta corrida no usó (respuestas de prompts viejos)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg, acfg = load_config(args.config), load_config(args.agents)
    root = bundle_dir(cfg)
    offered = load_bundle(root).budgets
    budgets = [float(b) for b in args.budgets.split(",")] if args.budgets else offered
    llm = LLM(acfg["llm"], root / acfg["llm"]["cache_subdir"], mode=args.mode)
    if not llm.available and llm.mode != "cache_only":
        print("No hay OPENAI_API_KEY (entorno o .env): no se puede correr el agente.")
        return 1
    failures, missing = 0, []
    for budget in budgets:
        bundle = load_bundle(root, budget)
        events = events_for(bundle, acfg)
        weeks = ([pd.Timestamp(w) for w in args.weeks.split(",")] if args.weeks
                 else sorted({e.week for e in events}))
        print(f"\n== {budget / 10:g}% de falsas alarmas · modelo {llm.model} · modo {llm.mode} · {len(weeks)} semana(s) "
              f"· {bundle.triage_dir}")
        for week in weeks:
            before = dict(llm.calls)
            result = run_triage(bundle, events, week, acfg, llm)
            if agent_missing(result):
                missing.append(f"--budgets {budget:g} --weeks {week.date()}")
                print(f"{week.date()} | falta en la caché: no se guarda", flush=True)
                continue
            save_triage(bundle, result)
            sources = [r["draft"]["source"] for r in result["events"]]
            problems = sum(bool(p) for r in result["events"] for p in r["draft"]["problems"])
            api = llm.calls["api"] - before["api"]
            cache = llm.calls["cache"] - before["cache"]
            failures += result["summary"]["source"] != "agente" or any(s != "agente" for s in sources)
            print(f"{week.date()} | {len(result['events'])} evento(s) | textos {sources} | resumen "
                  f"{result['summary']['source']} | rechazos del verificador {problems} | API {api}, caché {cache}",
                  flush=True)
    # Podar con un solo punto borraría la caché de los otros: solo con todos los puntos y todas las semanas.
    if missing:
        print("\nSemanas a rehacer en un solo proceso (sin --mode cache_only):\n" + "\n".join(missing))
    elif args.prune and not args.weeks and set(budgets) >= set(offered):
        stale = [p for p in llm.cache_dir.glob("*.json") if p.name not in llm.used]
        for p in stale:
            p.unlink()
        print(f"caché podada: {len(stale)} respuesta(s) sin usar, quedan {len(llm.used)}")
    print(f"\nlisto: {llm.calls['api']} llamadas a la API, {llm.calls['cache']} desde la caché. "
          f"Semanas con alguna plantilla: {failures}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
