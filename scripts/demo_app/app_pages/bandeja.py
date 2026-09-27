"""Bandeja: lo que el agente de triage dejó listo para la semana, para aprobar o descartar."""

from html import escape

import streamlit as st

from scripts.demo_app.common import (ACTION_ICONS, KIND_LABELS, agents_cfg, bundle, current_week, events, llm,
                                     set_week, store_triage, triage_for)
from src.agents.llm import LLM
from src.agents.policy import events_in_week
from src.agents.triage import run_triage, save_triage

from scripts.demo_app.presentation import agent_trace, heading, metrics, message
from scripts.demo_app.wording import HABITS_CAPTION, model_name

b, acfg = bundle(), agents_cfg()


def _ids(ids: list[str]) -> str:
    return ids[0] if len(ids) == 1 else ", ".join(ids[:-1]) + " y " + ids[-1]


def _count(n: int, one: str, many: str) -> str:
    return f"{n} {one if n == 1 else many}"


def _events_line(result) -> str:
    found = result.get("eventos", []) if isinstance(result, dict) else []
    new = [e["vehiculo"] for e in found if e["tipo"] == "alerta_nueva"]
    esc = [e["vehiculo"] for e in found if e["tipo"] != "alerta_nueva"]
    parts = ([f"{_count(len(new), 'alerta nueva', 'alertas nuevas')}: {_ids(new)}"] if new else []) + \
            ([f"{_count(len(esc), 'escalamiento', 'escalamientos')}: {_ids(esc)}"] if esc else [])
    return " · ".join(parts) or "Ninguno esta semana."


def trace_steps(trace: list[dict], actions: dict) -> list[dict]:
    """La traza del triage en castellano: qué hizo el agente, qué le devolvió cada herramienta y qué aprobó el
    verificador. Todo sale de lo que el triage guardó; nada se recalcula."""
    steps, summaries = [], 0
    for t in trace:
        if t["step"] == "tool":
            tool, args, res, err = t["tool"], t["args"], t.get("result"), t.get("error")
            vid = args.get("vehiculo", "")
            if tool == "eventos_de_la_semana":
                step = {"title": "Revisó los eventos de la semana", "detail": "" if err else _events_line(res)}
            elif tool == "contexto_de_la_flota":
                step = {"title": "Leyó el contexto de la flota",
                        "detail": "Los autos revisados, las alertas de la semana y de la temporada y el punto de "
                                  "operación del modelo: los únicos números que puede citar."}
            elif tool == "accion_de_la_politica":
                step = {"title": f"Le preguntó a la política qué hacer con {vid}"}
                if not err:
                    step |= {"result": ("policy", res["nombre"]), "quote": res["motivo"]}
            elif tool == "redactar_mensajes":
                label = actions.get(args.get("accion"), {}).get("label")
                step = {"title": f"Redactó los mensajes de {vid}", "detail": f"Para la acción: {label}" if label else ""}
                if not err:
                    step["result"] = (("ok", "El verificador los aprobó") if res.get("fuente") == "agente"
                                      else ("fail", "El verificador no los aprobó: quedó la plantilla"))
            else:
                step = {"title": tool}
            if err:
                step["result"] = ("fail", f"La herramienta lo rechazó: {err}")
        elif t["step"] == "resumen":
            summaries += 1
            step = {"title": "Escribió el resumen de la semana" if summaries == 1 else "Reescribió el resumen",
                    "result": ("ok", "El verificador lo aprobó") if not t["problems"]
                    else ("fail", "El verificador lo rechazó: " + "; ".join(t["problems"]))}
        elif t["step"] == "completado por el sistema":
            step = {"title": f"El sistema completó los mensajes de {t['vehicle_id']}",
                    "detail": "El agente no los pidió: los redactó el redactor, con el mismo verificador."}
        else:
            step = {"title": "El agente no estuvo disponible", "detail": t.get("error", ""),
                    "result": ("fail", "Quedaron las plantillas")}
        steps.append(step)
    return steps
week = current_week()
week_events = events_in_week(events(), week)
result = triage_for(week)
if result is None:
    # Sin triage guardado: la misma estructura con plantillas (no llama a la API).
    result = run_triage(b, events(), week, acfg, None)

fleet = result["fleet"]
# La semana ya está en el calendario de arriba: el subtítulo dice qué se hace en esta pantalla.
heading("Cada alerta, una acción a tiempo.",
        "Lo que el agente de triage dejó listo esta semana: revisá cada acción y aprobala o descartala.")
with st.container(key="week_metrics"):
    metrics([
        ("Autos monitoreados", fleet["autos_monitoreados_hasta_hoy"], "Flota revisada hasta hoy"),
        ("Alertas nuevas", fleet["alertas_nuevas_esta_semana"], "Esta semana"),
        ("Escalamientos al concesionario", fleet["escalamientos_esta_semana"], "Esta semana"),
        ("Alertas en la temporada", fleet["alertas_acumuladas_en_la_temporada"], "Desde el inicio del registro"),
    ])

# --- resumen del agente ------------------------------------------------------------------------
with st.container(border=True, key="summary"):
    with st.container(horizontal=True, vertical_alignment="center", gap="small"):
        st.markdown("#### Resumen de la semana")
        if result["summary"]["source"] == "agente":
            st.badge("Agente · verificado", icon=":material/verified:", color="green")
        else:
            st.badge("Plantilla", icon=":material/description:", color="gray")
    st.markdown(result["summary"]["text"].replace("\n", "  \n"))
    can_run = llm().available and week_events
    if result["mode"] != "agente" and can_run:
        if st.button("Correr el agente de triage", icon=":material/smart_toy:", type="primary"):
            with st.spinner("El agente está revisando la semana…"):
                result = run_triage(b, events(), week, acfg, llm())
            save_triage(b, result)
            store_triage(result)
            st.rerun()
    elif result["mode"] != "agente" and week_events:
        st.caption("Los agentes no están disponibles en este entorno: se muestran las plantillas del bundle.")
    elif result["mode"] == "agente" and can_run:
        if st.button("Regenerar en vivo", icon=":material/refresh:", type="tertiary",
                     help="Vuelve a correr el agente contra la API en lugar de mostrar la versión guardada. "
                          "El texto nuevo pasa por el mismo verificador."):
            with st.spinner("El agente está revisando la semana…"):
                live = LLM(acfg["llm"], llm().cache_dir, mode="live")
                result = run_triage(b, events(), week, acfg, live)
            store_triage(result)   # solo en la sesión: no pisa la versión guardada en el bundle
            st.rerun()
    if result["trace"]:
        with st.expander("Cómo llegó el agente a este resumen", icon=":material/account_tree:"):
            # Solo las llamadas en vivo: los triages del bundle guardaron un conteo de caché acumulado entre
            # semanas (el precalentado compartía el LLM), así que ese número no se muestra.
            api = result.get("llm_calls", {}).get("api", 0)
            note = (f"Modelo {result['model']} · "
                    + (f"{api} {'llamada' if api == 1 else 'llamadas'} a la API en vivo." if api
                       else "respuestas guardadas en la caché de la demo, sin llamar a la API.")
                    ) if result.get("model") else ""
            agent_trace(trace_steps(result["trace"], acfg["policy"]["actions"]),
                        lead="La acción de cada auto no la elige el agente: se la pregunta a la política, que es fija. "
                             "Todo lo que escribe pasa por el verificador antes de llegar a esta bandeja.",
                        note=note)

if not week_events:
    later = [e.week for e in events() if e.week > week]
    with st.container(key="empty_week"):
        st.info(f"Sin alertas nuevas ni escalamientos esta semana: {model_name(b.meta)} sigue revisando la flota cada "
                "500 km.", icon=":material/check_circle:")
        if later:
            st.button("Ir a la próxima semana con alertas", icon=":material/skip_next:", on_click=set_week,
                      args=(later[0],))
    st.stop()

# --- tarjetas: una por evento ------------------------------------------------------------------
# Colores de los badges, uno por significado: ámbar/rojo el tipo de evento, azul la acción de la política,
# verde lo que pasó un control (verificador, aprobación), gris lo neutro.
approvals = st.session_state.setdefault("approvals", {})
disclaimer = b.meta["texts"]["disclaimer"]
st.markdown("### Acciones de la semana")
for card_index, rec in enumerate(result["events"]):
    ev, facts, draft = rec["event"], rec["facts"], rec["draft"]
    vid = ev["vehicle_id"]
    key = f"{result['week']}:{vid}:{ev['kind']}"
    with st.container(key=f"alertcard_{card_index}_{ev['kind']}"):
        with st.container(horizontal=True, vertical_alignment="center", key=f"cardhead_{card_index}"):
            st.markdown(f"#### {vid}")
            st.badge(KIND_LABELS[ev["kind"]], color="red" if ev["kind"] == "persistencia" else "orange")
            st.badge(facts["accion"]["nombre"], icon=ACTION_ICONS.get(ev["action"]), color="blue")
            st.caption(f"{facts['mercado']} · {facts['fecha']}")
        st.markdown(f'<p class="card-lead">{escape(draft["drafts"]["linea_bandeja"])}</p>', unsafe_allow_html=True)
        if facts["habitos"]:
            st.caption(HABITS_CAPTION + " · ".join(
                f"{h['nombre']} {h['tu_valor']} (sanos {h['autos_sanos_comparables']})" for h in facts["habitos"]))
        tab_driver, tab_shop, tab_check = st.tabs(["Al conductor", "Al taller", "Hechos y verificador"])
        with tab_driver:
            message(draft["driver_text"], subject=True, note=disclaimer)
        with tab_shop:
            message(draft["workshop_text"])
        with tab_check:
            if draft["source"] == "agente":
                st.success(f"Redactado por el agente y aprobado por el verificador (intento {draft['attempts']}).",
                           icon=":material/verified:")
            else:
                st.info(draft["note"] or "Plantilla del bundle.", icon=":material/description:")
            for i, probs in enumerate(draft["problems"], start=1):
                if probs:
                    st.warning(f"Intento {i}, rechazado:\n\n" + "\n".join(f"- {p}" for p in probs))
            st.caption("Los hechos que recibió el agente (lo único que puede citar):")
            st.json(facts, expanded=1)
        # Decidir a la izquierda; ir a la ficha, que no decide nada, a la derecha y sin peso de botón.
        with st.container(horizontal=True, vertical_alignment="center", key=f"cardactions_{card_index}"):
            decided = approvals.get(key)
            if decided:
                st.badge(decided, icon=":material/task_alt:" if decided == "Aprobado" else ":material/block:",
                         color="green" if decided == "Aprobado" else "gray")
                if st.button("Deshacer", key=f"undo:{key}", icon=":material/undo:", type="tertiary"):
                    approvals.pop(key)
                    st.rerun()
            else:
                if st.button("Aprobar y enviar", key=f"ok:{key}", icon=":material/send:", type="primary"):
                    approvals[key] = "Aprobado"
                    st.rerun()
                if st.button("Descartar", key=f"no:{key}", icon=":material/close:"):
                    approvals[key] = "Descartado"
                    st.rerun()
            if st.button("Ver ficha", key=f"go:{key}", icon=":material/directions_car:", type="tertiary"):
                st.session_state["vehicle"] = vid
                st.switch_page("app_pages/vehiculo.py")
