"""Verificador determinista de los textos que escribe un agente. Es código, no otro LLM.

Tres controles, todos sobre el texto generado (las frases fijas las agrega el código después):
1. **Números.** Todo número del texto tiene que aparecer en los hechos que recibió el agente, tal
   cual (`3.500` y `3500` valen lo mismo; `3,5` no es `3`). Los identificadores (`VEH_0513`,
   `CNTRY_4`) no cuentan como números.
2. **Frases prohibidas** (`verifier.banned` de configs/agents.yaml): causa, certeza, promesas de que
   el riesgo baja, probabilidades. En el mensaje al conductor, además, los síntomas del filtro.
3. **Listas cerradas y largo**: recomendaciones y chequeos solo de los ids permitidos.

Devuelve la lista de problemas en castellano, vacía si el texto pasa.
"""

from __future__ import annotations

import json
import re
from typing import Any, Iterable

ID_PATTERN = re.compile(r"\b(?:VEH|CNTRY)_\d+\b", re.IGNORECASE)
NUMBER_PATTERN = re.compile(r"(?<![\w.,])(\d{1,3}(?:\.\d{3})+|\d+)(?:,(\d+))?(?![\w])")


def numbers_in(text: str) -> list[tuple[str, float]]:
    """Los números de un texto en castellano, con su forma original y su valor."""
    clean = ID_PATTERN.sub(" ", text)
    out = []
    for m in NUMBER_PATTERN.finditer(clean):
        integer, decimals = m.group(1).replace(".", ""), m.group(2)
        value = float(f"{integer}.{decimals}") if decimals else float(integer)
        out.append((m.group(0), value))
    return out


def allowed_numbers(*sources: Any) -> set[float]:
    """Todo número que aparece en los hechos (recorre dicts, listas y strings)."""
    found: set[float] = set()

    def walk(obj: Any) -> None:
        if isinstance(obj, dict):
            for v in obj.values():
                walk(v)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                walk(v)
        elif isinstance(obj, bool) or obj is None:
            return
        elif isinstance(obj, (int, float)):
            found.add(round(float(obj), 6))
        else:
            found.update(round(v, 6) for _, v in numbers_in(str(obj)))

    for s in sources:
        walk(s)
    return found


def number_problems(text: str, allowed: set[float], field: str) -> list[str]:
    bad = [raw for raw, value in numbers_in(text) if round(value, 6) not in allowed]
    return [f"{field}: el número «{raw}» no está en los hechos" for raw in dict.fromkeys(bad)]


def banned_problems(text: str, rules: Iterable[dict[str, str]], field: str) -> list[str]:
    out = []
    for rule in rules:
        m = re.search(rule["pattern"], text, flags=re.IGNORECASE)
        if m:
            out.append(f"{field}: «{m.group(0)}» ({rule['why']})")
    return out


def length_problems(text: str, limit: int | None, field: str) -> list[str]:
    if limit and len(text) > limit:
        return [f"{field}: {len(text)} caracteres, el máximo es {limit}"]
    if not text.strip():
        return [f"{field}: está vacío"]
    return []


def verify_text(text: str, facts: Any, cfg: dict[str, Any], *, field: str, driver: bool = False) -> list[str]:
    """Un texto suelto (p. ej. el resumen de la semana) contra los hechos que tuvo a mano."""
    vcfg = cfg["verifier"]
    rules = list(vcfg["banned"]) + (list(vcfg["driver_banned"]) if driver else [])
    return (length_problems(text, vcfg["max_chars"].get(field), field)
            + number_problems(text, allowed_numbers(facts), field)
            + banned_problems(text, rules, field))


def verify_drafts(drafts: dict[str, Any], facts: dict[str, Any], driver_facts: dict[str, Any],
                  cfg: dict[str, Any]) -> list[str]:
    """Los tres textos del redactor: conductor (con sus reglas extra), taller y línea de la bandeja."""
    problems: list[str] = []
    driver, workshop = drafts["conductor"], drafts["taller"]
    problems += verify_text(driver["asunto"], driver_facts, cfg, field="asunto", driver=True)
    problems += verify_text(driver["cuerpo"], driver_facts, cfg, field="cuerpo", driver=True)
    problems += verify_text(workshop["resumen"], facts, cfg, field="taller")
    problems += verify_text(drafts["linea_bandeja"], facts, cfg, field="linea_bandeja")

    allowed_recs = set(facts.get("recomendaciones_permitidas", {}))
    extra = [r for r in driver.get("recomendaciones", []) if r not in allowed_recs]
    if extra:
        problems.append(f"recomendaciones: {extra} no están entre las permitidas {sorted(allowed_recs)}")
    if facts["accion"]["id"] == "aviso_conductor" and allowed_recs and not driver.get("recomendaciones"):
        problems.append("recomendaciones: un aviso al conductor con hábitos tiene que llevar al menos una")
    fields = {"asunto": driver["asunto"], "cuerpo": driver["cuerpo"], "taller": workshop["resumen"],
              "linea_bandeja": drafts["linea_bandeja"]}
    for rule in (cfg["verifier"].get("action_rules") or {}).get(facts["accion"]["id"], []):
        text = fields[rule["field"]]
        if "forbid" in rule and (m := re.search(rule["forbid"], text, flags=re.IGNORECASE)):
            problems.append(f"{rule['field']}: «{m.group(0)}» ({rule['why']})")
        if "require" in rule and not re.search(rule["require"], text, flags=re.IGNORECASE):
            problems.append(f"{rule['field']}: falta decirlo ({rule['why']})")
    allowed_checks = set(facts.get("chequeos_permitidos", {}))
    extra = [c for c in workshop.get("chequeos", []) if c not in allowed_checks]
    if extra:
        problems.append(f"chequeos: {extra} no están entre los permitidos")
    if not workshop.get("chequeos"):
        problems.append("chequeos: el resumen del taller tiene que sugerir al menos uno")
    return problems


def facts_text(facts: Any) -> str:
    """Los hechos como los ve el LLM (y como se guardan en la traza)."""
    return json.dumps(facts, ensure_ascii=False, indent=1)
