"""Los hechos que ven los agentes: un JSON por evento, armado solo con lo que se sabía ese día.

El agente no ve el desenlace (si el auto falló después): los hechos se arman con la alerta, los
hábitos en los que el auto se aparta de los sanos de su mercado (una comparación con la flota, no lo
que usó el modelo) y las señales del filtro de los cortes que la dispararon. Cada número va ya
formateado, igual que lo tiene que escribir el texto: el verificador compara contra estos strings.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.agents.bundle import Bundle
from src.agents.formatting import format_value, long_date
from src.agents.policy import DRIVER, Event, all_events, week_end

#: Claves que el conductor no ve (señales del propio filtro y la lista del taller).
WORKSHOP_ONLY = ("senales_filtro", "chequeos_permitidos")


def vehicle_facts(bundle: Bundle, event: Event, agents_cfg: dict[str, Any]) -> dict[str, Any]:
    v = bundle.vehicles.loc[event.vehicle_id]
    policy = agents_cfg["policy"]
    recs_text = bundle.meta["texts"]["recommendations"]
    habits = [{"nombre": f["label"], "tu_valor": f["value_text"], "autos_sanos_comparables": f["reference_text"],
               "recomendacion": f["recommendation"]} for f in v["factors"]]
    recs = {h["recomendacion"]: recs_text[h["recomendacion"]] for h in habits if h["recomendacion"] in recs_text}
    gap, horizon = float(bundle.meta["gap_km"]), float(bundle.meta["horizon_km"])
    horizonte = {"desde_km": format_value(gap, "int"), "hasta_km": format_value(gap + horizon, "int")}
    if pd.notna(v["horizon_weeks_lo"]):
        horizonte.update({"semanas_desde": int(v["horizon_weeks_lo"]), "semanas_hasta": int(v["horizon_weeks_hi"]),
                          "km_por_dia": format_value(float(v["km_per_day"]), "int")})
    action = policy["actions"][event.action]
    facts = {
        "vehiculo": event.vehicle_id,
        "mercado": v["market"],
        "evento": event.kind,
        "fecha": long_date(event.date),
        "accion": {"id": event.action, "nombre": action["label"], "canal": action["channel"], "motivo": event.reason},
        "alerta": {"confirmada": long_date(v["alert_confirm_date"]), "revisiones_seguidas": bundle.k},
        "horizonte": horizonte,
        "habitos": habits,
        "recomendaciones_permitidas": recs,
        "senales_filtro": [{"nombre": s["label"], "valor": s["value_text"], "flota_sana_del_mercado": s["reference_text"]}
                           for s in v["technician_signals"]],
        "chequeos_permitidos": dict(agents_cfg["workshop_checks"]),
    }
    if event.kind != "alerta_nueva":
        facts["revisiones_despues_del_aviso"] = int(policy["persist_cuts"])
    return facts


def driver_view(facts: dict[str, Any]) -> dict[str, Any]:
    """Lo que el mensaje al conductor puede citar."""
    return {k: v for k, v in facts.items() if k not in WORKSHOP_ONLY}


def fleet_facts(bundle: Bundle, events: list[Event], week: pd.Timestamp) -> dict[str, Any]:
    """El contexto de la semana para el resumen: la flota vista hasta el domingo y el punto de operación."""
    end = week_end(week)
    seen = bundle.cuts.loc[bundle.cuts["cut_date"] < end, "vehicle_id"].nunique()
    past = [e for e in events if e.date < end]
    this_week = [e for e in past if e.week == pd.Timestamp(week)]
    curve = {round(float(p["budget_per_1000"])): p for p in bundle.meta["official"]["curve"]}
    point = curve.get(round(float(bundle.meta["budget_per_1000"])), {})
    return {
        "semana": long_date(week),
        "autos_monitoreados_hasta_hoy": int(seen),
        "alertas_nuevas_esta_semana": sum(e.kind == "alerta_nueva" for e in this_week),
        "escalamientos_esta_semana": sum(e.kind != "alerta_nueva" for e in this_week),
        "alertas_acumuladas_en_la_temporada": sum(e.kind == "alerta_nueva" for e in past),
        "avisos_al_conductor_acumulados": sum(e.kind == "alerta_nueva" and e.action == DRIVER for e in past),
        "punto_de_operacion": {
            "falsas_alarmas_toleradas": format_value(bundle.meta["budget_per_1000"] / 1000, "pct"),
            "deteccion_medida_en_desarrollo": format_value(float(point.get("detection", float("nan"))), "pct"),
            "anticipacion_mediana_km": format_value(float(point.get("lead_km", float("nan"))), "int"),
        },
    }


def events_for(bundle: Bundle, agents_cfg: dict[str, Any]) -> list[Event]:
    return all_events(bundle, agents_cfg["policy"])
