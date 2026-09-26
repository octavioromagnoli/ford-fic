"""Agente 1 · redactor + verificador: los textos de un evento (conductor, taller, bandeja).

El LLM devuelve una salida estructurada; el verificador (código) la controla contra los hechos. Si no
pasa, se reintenta una vez con la lista de problemas; si tampoco, queda la plantilla. Las
recomendaciones se eligen por id y su texto lo pone el código: el LLM no puede inventar un consejo.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from pydantic import BaseModel, Field

from src.agents.bundle import Bundle
from src.agents.facts import driver_view, vehicle_facts
from src.agents.llm import LLM, LLMUnavailable
from src.agents.policy import DRIVER, Event
from src.agents.verifier import facts_text, verify_drafts


class DriverDraft(BaseModel):
    asunto: str = Field(description="Asunto corto del mensaje al conductor")
    cuerpo: str = Field(description="De 2 a 4 oraciones, sin las recomendaciones (las agrega el sistema)")
    recomendaciones: list[str] = Field(description="ids de recomendaciones_permitidas")


class WorkshopDraft(BaseModel):
    resumen: str = Field(description="De 3 a 5 oraciones para el asesor de servicio")
    chequeos: list[str] = Field(description="ids de chequeos_permitidos")


class Drafts(BaseModel):
    conductor: DriverDraft
    taller: WorkshopDraft
    linea_bandeja: str = Field(description="Una línea para la bandeja de la gerente")


@dataclass
class DraftResult:
    vehicle_id: str
    source: str                     # agente | plantilla
    drafts: dict[str, Any]
    problems: list[list[str]] = field(default_factory=list)   # problemas del verificador, por intento
    attempts: int = 0
    note: str = ""
    driver_text: str = ""
    workshop_text: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def render_driver(drafts: dict[str, Any], facts: dict[str, Any], bundle: Bundle) -> str:
    """El mensaje final al conductor: lo del LLM + las recomendaciones y el aviso fijos."""
    d = drafts["conductor"]
    texts = bundle.meta["texts"]
    lines = [d["asunto"], "", d["cuerpo"]]
    recs = [facts["recomendaciones_permitidas"][r] for r in d.get("recomendaciones", [])
            if r in facts["recomendaciones_permitidas"]]
    if recs:
        lines += ["", "Qué podés hacer:"] + [f"- {r}" for r in recs]
    lines += ["", texts["disclaimer"]]
    return "\n".join(lines)


def render_workshop(drafts: dict[str, Any], facts: dict[str, Any]) -> str:
    t = drafts["taller"]
    checks = [facts["chequeos_permitidos"][c] for c in t.get("chequeos", []) if c in facts["chequeos_permitidos"]]
    return "\n".join([t["resumen"], "", "Chequeos sugeridos:"] + [f"- {c}" for c in checks])


def template_drafts(bundle: Bundle, event: Event, facts: dict[str, Any], agents_cfg: dict[str, Any]) -> dict[str, Any]:
    """Los textos sin LLM: el mensaje preregistrado de la explicabilidad y un resumen armado."""
    tpl = agents_cfg["templates"]
    v = bundle.vehicles.loc[event.vehicle_id]
    body = v["template_message"]
    if event.action != DRIVER:
        body = f"{body}\n{tpl['dealer_invite']}"
    lines = [tpl["workshop_intro"].format(fecha=facts["alerta"]["confirmada"], k=facts["alerta"]["revisiones_seguidas"])]
    if facts["habitos"]:
        lines.append(tpl["workshop_habits"])
        lines += [f"- {h['nombre']}: {h['tu_valor']} (sanos comparables: {h['autos_sanos_comparables']})"
                  for h in facts["habitos"]]
    else:
        lines.append(tpl["workshop_no_habits"])
    lines.append(tpl["workshop_signals"])
    lines += [f"- {s['nombre']}: {s['valor']} (sanos del mercado: {s['flota_sana_del_mercado']})"
              for s in facts["senales_filtro"]]
    return {
        "conductor": {"asunto": tpl["driver_subject"] if event.action == DRIVER else tpl["dealer_subject"],
                      "cuerpo": body, "recomendaciones": []},
        "taller": {"resumen": "\n".join(lines), "chequeos": list(facts["chequeos_permitidos"])},
        "linea_bandeja": f"{event.vehicle_id}: {facts['accion']['nombre']}. {event.reason}",
    }


def draft_event(bundle: Bundle, event: Event, agents_cfg: dict[str, Any], llm: LLM | None) -> DraftResult:
    facts = vehicle_facts(bundle, event, agents_cfg)
    dview = driver_view(facts)
    result = DraftResult(vehicle_id=event.vehicle_id, source="plantilla", drafts={})
    if llm is not None:
        instructions = agents_cfg["prompts"]["drafter"]
        payload = "HECHOS:\n" + facts_text(facts)
        retries = int(agents_cfg["llm"].get("max_verification_retries", 1))
        try:
            for attempt in range(retries + 1):
                out, _ = llm.parse(instructions=instructions, payload=payload, schema=Drafts)
                result.attempts = attempt + 1
                problems = verify_drafts(out, facts, dview, agents_cfg)
                result.problems.append(problems)
                if not problems:
                    result.source, result.drafts = "agente", out
                    break
                payload = ("HECHOS:\n" + facts_text(facts) + "\n\nTu versión anterior no pasó el verificador:\n"
                           + "\n".join(f"- {p}" for p in problems) + "\nCorregí esos puntos y devolvé los tres textos.")
            else:
                result.note = "La redacción no pasó el verificador: queda la plantilla."
        except LLMUnavailable as exc:
            result.note = f"Sin agente ({exc}): plantilla."
    if result.source == "plantilla":
        result.drafts = template_drafts(bundle, event, facts, agents_cfg)
        result.driver_text = result.drafts["conductor"]["asunto"] + "\n\n" + result.drafts["conductor"]["cuerpo"]
    else:
        result.driver_text = render_driver(result.drafts, facts, bundle)
    result.workshop_text = render_workshop(result.drafts, facts)
    return result
