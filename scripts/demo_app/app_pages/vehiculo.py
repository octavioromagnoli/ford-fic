"""Ficha del vehículo: el riesgo de K2 en el tiempo, el umbral, la alerta y el porqué."""

import altair as alt
import pandas as pd
import streamlit as st

from scripts.demo_app.common import (EFFECT_COLORS, K2_COLOR, KIND_LABELS, MUTED, agents_cfg, bundle, current_week,
                                     events, triage_for, week_label)
from src.agents.formatting import format_value, short_date
from src.agents.policy import week_end

from scripts.demo_app.presentation import heading, chart_style, message, table

b = bundle()
week = current_week()
today = week_end(week)
v_all = b.vehicles
seen = b.cuts.loc[b.cuts["cut_date"] < today, "vehicle_id"].unique()


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
        "El riesgo de K2 en cada revisión del auto, la alerta y los hábitos que se parecen a los de autos que fallaron.")
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
line = base.mark_line(color=K2_COLOR, strokeWidth=2).encode(y=alt.Y("score:Q", title=None, scale=alt.Scale(domain=[0, 1])))
points = base.mark_point(filled=True, size=70, color=K2_COLOR).encode(
    y="score:Q",
    tooltip=[alt.Tooltip("cut_date:T", title="Fecha", format="%d-%m-%Y"), alt.Tooltip("km:N", title="Odómetro"),
             alt.Tooltip("score:Q", title="Riesgo", format=".2f")])
rules = pd.DataFrame({"y": [b.threshold], "t": [f"umbral ({b.meta['budget_per_1000'] / 10:g}% de falsas alarmas)"]})
threshold = alt.Chart(rules).mark_rule(color=MUTED, strokeDash=[5, 3]).encode(y="y:Q", tooltip=alt.Tooltip("t:N", title=""))
layers = [threshold, line, points]
layers.append(alt.Chart(pd.DataFrame([{"x": today - pd.Timedelta(days=1), "t": "hoy"}])).mark_rule(
    color=MUTED, strokeWidth=2, strokeDash=[2, 2]).encode(x="x:T", tooltip=alt.Tooltip("t:N", title="")))
# El título va arriba y horizontal: rotado en el eje, en un celular se cortaba por los dos lados.
with st.container(key="risk_chart"):
    st.altair_chart(chart_style(alt.layer(*layers).properties(height=300,
                                                              title="Riesgo de K2 (próximos 500 a 3.500 km)")),
                    width="stretch", theme=None)
    st.caption(f"Semana del {week_label(week)}. La alerta se confirma cuando {b.k} revisiones seguidas quedan sobre "
               "el umbral.")

# --- por qué ----------------------------------------------------------------------------------
why = st.container(key="why")
why.subheader("Por qué", anchor=False)
if not alerted:
    why.caption("El porqué se explica cuando el auto alerta: por ahora K2 no lo marcó.")
    st.stop()
left, right = why.container(), why.container()
with left:
    if v["factors"]:
        st.markdown("**Su uso se parece al de autos que fallaron en:**")
        table(["Hábito", "Este auto", "Sanos comparables"],
              [[f["label"], f["value_text"], f["reference_text"]] for f in v["factors"]])
    else:
        st.markdown("**El riesgo no se explica por hábitos de uso**: ningún factor pasa los filtros de la explicabilidad.")
    st.caption("Es asociación, no causa: el modelo marca parecido con autos que fallaron. «Sanos comparables» es la "
               "mediana de los autos sanos del mismo mercado y mes.")
with right:
    w = b.waterfall.loc[b.waterfall["vehicle_id"].eq(vid)]
    if len(w):
        w = w.assign(efecto=lambda f: [("no accionable" if k != "accionable" else
                                        ("sube el riesgo" if val > 0 else "baja el riesgo"))
                                       for k, val in zip(f["kind"], f["value"])])
        # Cada factor escribe su nombre arriba de la barra, no en el eje: en un celular el eje se comía el ancho.
        rows = alt.Chart(w).encode(y=alt.Y("step:N", sort=None, title=None, axis=None))
        bars = rows.mark_bar(cornerRadiusEnd=4, height=12, yOffset=9).encode(
            x=alt.X("value:Q", title="Contribución al riesgo (log-odds)"),
            color=alt.Color("efecto:N", scale=alt.Scale(domain=list(EFFECT_COLORS), range=list(EFFECT_COLORS.values())),
                            legend=alt.Legend(orient="bottom", title=None, columns=2)),
            tooltip=[alt.Tooltip("step:N", title=""), alt.Tooltip("value:Q", title="Contribución", format="+.2f")])
        names = rows.mark_text(align="left", baseline="middle", yOffset=-8, color="#c6d6ec", font="Manrope",
                               fontSize=12, limit=alt.ExprRef("width")).encode(x=alt.value(0), text="step:N")
        zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color="#30445f").encode(x="x:Q")
        st.altair_chart(chart_style(alt.layer(zero, bars, names).properties(height=44 * len(w))),
                        width="stretch", theme=None)

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
