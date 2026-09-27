"""Ficha del vehículo: el riesgo del modelo en el tiempo, el umbral, la alerta y en qué se aparta de los sanos."""

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from scripts.demo_app.common import (DEVIATION_COLORS, KIND_LABELS, MODEL_COLOR, MUTED, agents_cfg, bundle,
                                     current_week, events, triage_for, week_label)
from src.agents.formatting import format_value, short_date
from src.agents.policy import week_end

from scripts.demo_app.presentation import heading, chart_style, message, table
from scripts.demo_app.wording import (WHY_CAPTION, WHY_CHART_AXIS, WHY_CHART_CAPTION, WHY_CHART_TITLE, WHY_INTRO,
                                      WHY_NONE, model_name)

b = bundle()
week = current_week()
today = week_end(week)
v_all = b.vehicles
seen = b.cuts.loc[b.cuts["cut_date"] < today, "vehicle_id"].unique()
name = model_name(b.meta)


def _label(vid: str) -> str:
    v = v_all.loc[vid]
    if bool(v["alerted"]) and v["alert_confirm_date"] < today:
        return f"{vid} · alerta del {short_date(v['alert_confirm_date'])}"
    return f"{vid} · sin alerta"


alerted_now = [vid for vid in v_all.index if bool(v_all.loc[vid, "alerted"]) and v_all.loc[vid, "alert_confirm_date"] < today]
options = alerted_now + sorted(set(seen) - set(alerted_now))
if not options:
    st.info("Todavía no hay revisiones de ningún auto esta semana.")
    st.stop()
default = st.session_state.get("vehicle")
heading("Una historia detrás de cada señal.",
        f"El riesgo de {name} en cada revisión del auto, la alerta y en qué hábitos se aparta de los autos sanos de su "
        "mercado.")
with st.container(key="vehicle_pick"):
    vid = st.selectbox("Vehículo", options, index=options.index(default) if default in options else 0,
                       format_func=_label,
                       help="Primero los que ya alertaron; después el resto de la flota vista hasta hoy.")
st.session_state["vehicle"] = vid
v = v_all.loc[vid]
cuts = b.vehicle_cuts(vid)
past = cuts.loc[cuts["cut_date"] < today]
alerted = bool(v["alerted"]) and v["alert_confirm_date"] < today

with st.container(key="vehicle_metrics"):
    market, reviews, pace, state = st.columns([1, 1, 1.1, 1.6])
    market.metric("Mercado", v["market"], border=True)
    reviews.metric("Revisiones hasta hoy", len(past), border=True, help="Un punto de corte cada 500 km recorridos.")
    pace.metric("Ritmo de uso", format_value(float(v["km_per_day"]), "int") + " km/día", border=True)
    state.metric("Estado", f"Alerta desde el {short_date(v['alert_confirm_date'])}" if alerted else "Sin alerta", border=True)

# --- el riesgo en el tiempo --------------------------------------------------------------------
# Solo lo que se sabía hasta esta semana: lo que pasó después está en «Qué pasó después».
data = past.assign(km=lambda f: f["cut_odo"].map(lambda x: format_value(x, "int") + " km"))
base = alt.Chart(data).encode(x=alt.X("cut_date:T", title=None, axis=alt.Axis(format="%d-%m-%y", tickCount=6)))
line = base.mark_line(color=MODEL_COLOR, strokeWidth=2).encode(y=alt.Y("score:Q", title=None, scale=alt.Scale(domain=[0, 1])))
points = base.mark_point(filled=True, size=70, color=MODEL_COLOR).encode(
    y="score:Q",
    tooltip=[alt.Tooltip("cut_date:T", title="Fecha", format="%d-%m-%Y"), alt.Tooltip("km:N", title="Odómetro"),
             alt.Tooltip("score:Q", title="Puntaje", format=".2f")])
rules = pd.DataFrame({"y": [b.threshold], "t": [f"umbral ({b.meta['budget_per_1000'] / 10:g}% de falsas alarmas)"]})
threshold = alt.Chart(rules).mark_rule(color=MUTED, strokeDash=[5, 3]).encode(y="y:Q", tooltip=alt.Tooltip("t:N", title=""))
layers = [threshold, line, points]
layers.append(alt.Chart(pd.DataFrame([{"x": today - pd.Timedelta(days=1), "t": "hoy"}])).mark_rule(
    color=MUTED, strokeWidth=2, strokeDash=[2, 2]).encode(x="x:T", tooltip=alt.Tooltip("t:N", title="")))
# El título va arriba y horizontal: rotado en el eje, en un celular se cortaba por los dos lados.
with st.container(key="risk_chart"):
    horizon = (f"próximos {format_value(float(b.meta['gap_km']), 'int')} a "
               f"{format_value(float(b.meta['gap_km']) + float(b.meta['horizon_km']), 'int')} km")
    st.altair_chart(chart_style(alt.layer(*layers).properties(height=300,
                                                              title=f"Puntaje de riesgo de {name} ({horizon})")),
                    width="stretch", theme=None)
    st.caption(f"Semana del {week_label(week)}. La alerta se confirma cuando {b.k} revisiones seguidas quedan sobre "
               f"el umbral. {b.model.get('score_note', '')}".strip())

# --- por qué ----------------------------------------------------------------------------------
# No es lo que usó el modelo: es en qué hábitos se aparta el auto de los sanos de su mercado (src/eval/fleet_profile.py).
why = st.container(key="why")
why.subheader("Por qué", anchor=False)
if not alerted:
    why.caption(f"El porqué se muestra cuando el auto alerta: por ahora {name} no lo marcó.")
    st.stop()
left, right = why.container(), why.container()
with left:
    if v["factors"]:
        st.markdown(f"**{WHY_INTRO}**")
        table(["Hábito", "Este auto", "Sanos de su mercado"],
              [[f["label"], f["value_text"], f["reference_text"]] for f in v["factors"]])
    else:
        st.markdown(WHY_NONE)
    st.caption(WHY_CAPTION)
with right:
    dv = b.deviations.loc[b.deviations["vehicle_id"].eq(vid)].copy()
    if len(dv):
        # Solo hacia el lado que la física del filtro señala como riesgoso: del lado de los sanos no hay barra.
        dv["efecto"] = np.select([dv["named"], dv["risky"]], list(DEVIATION_COLORS)[:2], default=list(DEVIATION_COLORS)[2])
        dv["start"] = 50.0
        dv["end"] = np.where(dv["risky"], 100.0 * dv["share"], 50.0)
        dv["step"] = (dv["label"] + " (" + dv["value_text"] + " vs " + dv["reference_text"] + ")"
                      + np.where(dv["risky"], "", " · del lado de los sanos"))
        dv["supera"] = [f"{format_value(s_, 'pct')} de los sanos del mercado" if r_ else "—"
                        for s_, r_ in zip(dv["share"], dv["risky"])]
        dv = dv.sort_values(["named", "risky", "share"], ascending=[False, False, False], kind="stable")
        min_share = float(b.meta["explanation"]["min_healthy_share"])
        # Cada hábito escribe su nombre arriba de la barra, no en el eje: en un celular el eje se comía el ancho.
        rows = alt.Chart(dv).encode(y=alt.Y("step:N", sort=None, title=None, axis=None))
        bars = rows.mark_bar(cornerRadiusEnd=4, height=12, yOffset=9).encode(
            x=alt.X("start:Q", title=WHY_CHART_AXIS, scale=alt.Scale(domain=[50, 100])), x2="end:Q",
            color=alt.Color("efecto:N", scale=alt.Scale(domain=list(DEVIATION_COLORS), range=list(DEVIATION_COLORS.values())),
                            legend=alt.Legend(orient="bottom", title=None, columns=1)),
            tooltip=[alt.Tooltip("label:N", title=""), alt.Tooltip("value_text:N", title="Este auto"),
                     alt.Tooltip("reference_text:N", title="Mediana de los sanos"), alt.Tooltip("supera:N", title="Supera al")])
        names = rows.mark_text(align="left", baseline="middle", yOffset=-8, color="#c6d6ec", font="Manrope",
                               fontSize=12, limit=alt.ExprRef("width")).encode(x=alt.value(0), text="step:N")
        cut = alt.Chart(pd.DataFrame({"x": [100.0 * min_share]})).mark_rule(color=MUTED, strokeDash=[4, 3]).encode(x="x:Q")
        st.altair_chart(chart_style(alt.layer(cut, bars, names).properties(height=44 * len(dv), title=WHY_CHART_TITLE)),
                        width="stretch", theme=None)
        st.caption(WHY_CHART_CAPTION.format(share=format_value(min_share, "pct")))

# --- lo que se le mandó ------------------------------------------------------------------------
sent = st.container(key="sent")
sent.subheader("Lo que se le comunicó", anchor=False)
records = []
for e in [e for e in events() if e.vehicle_id == vid and e.date < today]:
    t = triage_for(e.week)
    rec = next((r for r in (t or {}).get("events", []) if r["event"]["vehicle_id"] == vid
                and r["event"]["kind"] == e.kind), None)
    records.append((e, rec))
for e, rec in records:
    action = agents_cfg()["policy"]["actions"][e.action]["label"]
    with sent.expander(f"{short_date(e.date)} · {KIND_LABELS[e.kind]} · {action}"):
        if rec is None:
            st.caption("El triage de esa semana todavía no se corrió: abrí la bandeja de esa semana.")
            continue
        a, c = st.columns(2)
        a.markdown("**Al conductor**")
        with a:
            message(rec["draft"]["driver_text"], subject=True, note=b.meta["texts"]["disclaimer"])
        c.markdown("**Al taller**")
        with c:
            message(rec["draft"]["workshop_text"])
