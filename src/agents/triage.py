"""Agente 2 · triage semanal: un loop de herramientas que deja la bandeja lista para aprobar.

El LLM orquesta: pide los eventos de la semana, la acción de la política para cada auto, la
redacción (agente 1) y el contexto de la flota, y escribe el resumen para la gerente. No decide nada
que importe: `redactar_mensajes` rechaza una acción distinta de la política, y si al terminar quedó
algún evento sin redactar, el código lo completa. El resumen pasa por el verificador contra lo que
devolvieron las herramientas; si no pasa, se reintenta una vez y si no, queda la plantilla.

Sin LLM (sin key o en `cache_only` sin caché) el resultado tiene la misma forma, con plantillas.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.agents.bundle import Bundle
from src.agents.drafter import DraftResult, draft_event
from src.agents.facts import fleet_facts, vehicle_facts
from src.agents.formatting import long_date
from src.agents.llm import LLM, LLMUnavailable
from src.agents.policy import Event, check_action, events_in_week, expected_action
from src.agents.verifier import verify_text

_NO_ARGS = {"type": "object", "properties": {}, "required": [], "additionalProperties": False}
_VEHICLE = {"type": "string", "description": "id del vehículo, p. ej. VEH_0513"}

TOOLS = [
    {"type": "function", "name": "eventos_de_la_semana", "strict": True, "parameters": _NO_ARGS,
     "description": "Las alertas nuevas y los escalamientos de la semana, con su fecha."},
    {"type": "function", "name": "accion_de_la_politica", "strict": True,
     "parameters": {"type": "object", "properties": {"vehiculo": _VEHICLE}, "required": ["vehiculo"],
                    "additionalProperties": False},
     "description": "La acción que la política de posventa asigna al evento de un auto esta semana, con el motivo."},
    {"type": "function", "name": "redactar_mensajes", "strict": True,
     "parameters": {"type": "object", "properties": {
         "vehiculo": _VEHICLE,
         "accion": {"type": "string", "enum": ["aviso_conductor", "turno_concesionario"]}},
         "required": ["vehiculo", "accion"], "additionalProperties": False},
     "description": ("Redacta el mensaje al conductor, el resumen del taller y la línea de la bandeja, y los pasa por "
                     "el verificador. La acción tiene que ser la de la política.")},
    {"type": "function", "name": "contexto_de_la_flota", "strict": True, "parameters": _NO_ARGS,
     "description": "Autos monitoreados hasta hoy, alertas acumuladas y el punto de operación del sistema."},
]


def triage_path(bundle: Bundle, week: pd.Timestamp) -> Path:
    return bundle.root / "triage" / f"{pd.Timestamp(week).date().isoformat()}.json"


def load_triage(bundle: Bundle, week: pd.Timestamp) -> dict[str, Any] | None:
    path = triage_path(bundle, week)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def save_triage(bundle: Bundle, result: dict[str, Any]) -> Path | None:
    path = triage_path(bundle, pd.Timestamp(result["week"]))
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        return path
    except OSError:
        return None


def template_summary(week: pd.Timestamp, events: list[Event], drafts: dict[str, DraftResult],
                     agents_cfg: dict[str, Any]) -> str:
    tpl = agents_cfg["templates"]
    if not events:
        return tpl["week_empty"].format(semana=long_date(week))
    lines = [tpl["week_intro"].format(semana=long_date(week), n_alertas=sum(e.kind == "alerta_nueva" for e in events),
                                      n_escaladas=sum(e.kind != "alerta_nueva" for e in events))]
    lines += [drafts[e.vehicle_id].drafts["linea_bandeja"] for e in events if e.vehicle_id in drafts]
    return "\n".join(lines)


class _Tools:
    """Las herramientas del agente, cerradas sobre la semana. Guardan lo que devolvieron."""

    def __init__(self, bundle: Bundle, events: list[Event], week: pd.Timestamp, agents_cfg: dict[str, Any], llm: LLM):
        self.bundle, self.events, self.week, self.cfg, self.llm = bundle, events, pd.Timestamp(week), agents_cfg, llm
        self.drafts: dict[str, DraftResult] = {}
        self.outputs: list[Any] = []

    def run(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        try:
            out = getattr(self, f"_{name}")(**args)
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            out = {"error": str(exc)}
        self.outputs.append(out)
        return out

    def _eventos_de_la_semana(self) -> dict[str, Any]:
        week_events = events_in_week(self.events, self.week)
        return {"semana": long_date(self.week),
                "eventos": [{"vehiculo": e.vehicle_id, "tipo": e.kind, "fecha": long_date(e.date)} for e in week_events]}

    def _accion_de_la_politica(self, vehiculo: str) -> dict[str, Any]:
        e = expected_action(self.events, vehiculo, self.week)
        if e is None:
            raise ValueError(f"{vehiculo} no tiene eventos esta semana")
        action = self.cfg["policy"]["actions"][e.action]
        return {"vehiculo": vehiculo, "accion": e.action, "nombre": action["label"], "motivo": e.reason}

    def _redactar_mensajes(self, vehiculo: str, accion: str) -> dict[str, Any]:
        e = check_action(self.events, vehiculo, self.week, accion)
        result = draft_event(self.bundle, e, self.cfg, self.llm)
        self.drafts[vehiculo] = result
        facts = vehicle_facts(self.bundle, e, self.cfg)
        return {"vehiculo": vehiculo, "accion": accion, "fuente": result.source,
                "verificador": "aprobado" if result.source == "agente" else "no aprobado: queda la plantilla",
                "linea_bandeja": result.drafts["linea_bandeja"],
                "habitos": [f"{h['nombre']}: {h['tu_valor']} (sanos: {h['autos_sanos_comparables']})" for h in facts["habitos"]],
                "horizonte": facts["horizonte"]}

    def _contexto_de_la_flota(self) -> dict[str, Any]:
        return fleet_facts(self.bundle, self.events, self.week)


def _event_record(bundle: Bundle, e: Event, draft: DraftResult, agents_cfg: dict[str, Any]) -> dict[str, Any]:
    return {"event": e.as_dict(), "facts": vehicle_facts(bundle, e, agents_cfg), "draft": draft.as_dict()}


def run_triage(bundle: Bundle, events: list[Event], week: pd.Timestamp, agents_cfg: dict[str, Any],
               llm: LLM | None) -> dict[str, Any]:
    week = pd.Timestamp(week)
    week_events = events_in_week(events, week)
    trace: list[dict[str, Any]] = []
    summary, summary_source, summary_problems = None, "plantilla", []
    tools = None
    mode = "plantilla"

    if llm is not None:
        tools = _Tools(bundle, events, week, agents_cfg, llm)
        instructions = agents_cfg["prompts"]["triage"]
        items: list[dict[str, Any]] = [{"role": "user", "content": f"Semana del {long_date(week)}. Prepará el triage."}]
        try:
            retries = int(agents_cfg["llm"].get("max_verification_retries", 1))
            for _ in range(int(agents_cfg["llm"]["max_tool_steps"])):
                out, meta = llm.respond(instructions=instructions, items=items, tools=TOOLS)
                if not out["tool_calls"]:
                    text = out["text"].strip()
                    problems = verify_text(text, tools.outputs, agents_cfg, field="resumen")
                    summary_problems.append(problems)
                    trace.append({"step": "resumen", "cached": meta["cached"], "problems": problems})
                    if not problems:
                        summary, summary_source = text, "agente"
                        break
                    if len(summary_problems) > retries:
                        break
                    items.append({"role": "assistant", "content": text})
                    items.append({"role": "user", "content": "El resumen no pasó el verificador:\n"
                                  + "\n".join(f"- {p}" for p in problems) + "\nReescribilo corrigiendo esos puntos."})
                    continue
                for call in out["tool_calls"]:
                    args = json.loads(call["arguments"] or "{}")
                    result = tools.run(call["name"], args)
                    trace.append({"step": "tool", "tool": call["name"], "args": args, "cached": meta["cached"],
                                  "error": result.get("error"),
                                  "result": result if call["name"] != "contexto_de_la_flota" else "(contexto)"})
                    items.append({"type": "function_call", "call_id": call["call_id"], "name": call["name"],
                                  "arguments": call["arguments"]})
                    items.append({"type": "function_call_output", "call_id": call["call_id"],
                                  "output": json.dumps(result, ensure_ascii=False)})
            mode = "agente"
        except LLMUnavailable as exc:
            trace.append({"step": "sin agente", "error": str(exc)})

    drafts = tools.drafts if tools else {}
    for e in week_events:
        if e.vehicle_id not in drafts:
            drafts[e.vehicle_id] = draft_event(bundle, e, agents_cfg, llm if mode == "agente" else None)
            if mode == "agente":
                trace.append({"step": "completado por el sistema", "vehicle_id": e.vehicle_id})
    if summary is None:
        summary = template_summary(week, week_events, drafts, agents_cfg)

    return {
        "week": week.date().isoformat(),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mode": mode,
        "model": llm.model if llm is not None and mode == "agente" else None,
        "summary": {"text": summary, "source": summary_source, "problems": summary_problems},
        "events": [_event_record(bundle, e, drafts[e.vehicle_id], agents_cfg) for e in week_events],
        "fleet": fleet_facts(bundle, events, week),
        "trace": trace,
        "llm_calls": dict(llm.calls) if llm is not None else {},
    }
