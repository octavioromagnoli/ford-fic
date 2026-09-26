"""Bandeja: lo que el agente de triage dejó listo para la semana, para aprobar o descartar."""

from html import escape

import streamlit as st

from scripts.demo_app.common import (ACTION_ICONS, KIND_LABELS, agents_cfg, bundle, current_week, events, llm,
                                     set_week, store_triage, triage_for)
from src.agents.llm import LLM
from src.agents.policy import events_in_week
from src.agents.triage import run_triage, save_triage

from scripts.demo_app.presentation import heading, metrics, message, table

b, acfg = bundle(), agents_cfg()
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
        st.caption("Los agentes no están disponibles en este entorno: se muestran las plantillas preregistradas.")
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
        with st.expander("Qué hizo el agente", icon=":material/account_tree:"):
            rows = []
            for t in result["trace"]:
                if t["step"] == "tool":
                    detail = ", ".join(f"{k}={v}" for k, v in t["args"].items()) or "—"
                    status = "error: " + t["error"] if t.get("error") else "ok"
                    rows.append([t["tool"], detail, status])
                elif t["step"] == "resumen":
                    rows.append(["resumen", "—", "verificado" if not t["problems"] else "; ".join(t["problems"])])
                else:
                    rows.append([t["step"], t.get("vehicle_id", "—"), t.get("error", "")])
            table(["paso", "argumentos", "resultado"], rows, stack=True)
            if result.get("model"):
                calls = result.get("llm_calls", {})
                st.caption(f"Modelo {result['model']} · {calls.get('api', 0)} llamadas a la API, "
                           f"{calls.get('cache', 0)} desde la caché.")

if not week_events:
    later = [e.week for e in events() if e.week > week]
    with st.container(key="empty_week"):
        st.info("Sin alertas nuevas ni escalamientos esta semana: K2 sigue revisando la flota cada 500 km.",
                icon=":material/check_circle:")
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
            st.caption("Hábitos que se parecen a los de autos que fallaron: " + " · ".join(
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
                st.info(draft["note"] or "Plantilla preregistrada de la explicabilidad.", icon=":material/description:")
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
