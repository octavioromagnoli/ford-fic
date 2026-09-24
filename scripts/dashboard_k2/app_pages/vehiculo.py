"""Vehículo: el score de K2 a lo largo del odómetro, el umbral, la alerta y el perfil de uso."""

import altair as alt
import pandas as pd
import streamlit as st

from scripts.dashboard_k2.common import (FLEET_COLOR, K2_COLOR, MUTED, OUTCOME_COLORS, alerts, budget,
                                         budget_label, cfg, data, km, label)
from src.eval.dashboard_data import fleet_profile, rows_for

d, settings = data(), cfg()
lab, b = label(), budget()

with st.container(horizontal=True, vertical_alignment="bottom"):
    repeat = st.segmented_control("Repetición de la CV", options=list(range(d.n_repeats)),
                                  format_func=lambda r: f"R{r + 1}", default=0, required=True, key="repeat",
                                  help="Cada repetición sortea otros folds: el umbral y las alertas cambian un poco.")
    outcomes = st.pills("Mostrar", options=list(OUTCOME_COLORS), default=["Detectado", "No detectado", "Falsa alarma"],
                        selection_mode="multi", key="outcomes")

point, table = alerts(lab, repeat, b)
if point is None:
    st.warning(f"Ningún umbral respeta {budget_label(b)} de falsas alarmas en esta repetición.")
    st.stop()

counts = table["outcome"].value_counts()
st.caption(f"Umbral de R{repeat + 1} a {budget_label(b)}: {point['threshold']:.4f} · "
           + " · ".join(f"{name}: {int(counts.get(name, 0))}" for name in OUTCOME_COLORS))

shown = table[table["outcome"].isin(outcomes or [])].sort_values(["outcome", "lead_km", "max_score"], ascending=[True, False, False])
if shown.empty:
    st.info("Elegí al menos un grupo en **Mostrar**.")
    st.stop()
vid = st.selectbox("Vehículo", shown["vehicle_id"].tolist(),
                   format_func=lambda v: f"{v} · {shown.set_index('vehicle_id').at[v, 'outcome']}", key="vehicle")
info = table.set_index("vehicle_id").loc[vid]

with st.container(horizontal=True):
    st.metric("Estado", info["outcome"], border=True)
    st.metric("Mercado", info["market"], border=True)
    st.metric("Evento en el odómetro", km(info["event_odo_km"]) if info["failed"] else "sin evento", border=True)
    st.metric("Primera alerta sostenida", km(info["alert_cut_odo"]) if info["alerted"] else "no alertó", border=True)
    if info["failed"]:
        st.metric("Anticipación", km(info["lead_km"]), border=True)

# --- score contra el umbral ----------------------------------------------------------------
rows = rows_for(d, lab)
v = rows.loc[rows["vehicle_id"].eq(vid)].sort_values("cut_odo").assign(score=lambda f: f[f"score_r{repeat}"])
base = alt.Chart(v).encode(x=alt.X("cut_odo:Q", title="Odómetro del corte [km]"))
score_line = base.mark_line(color=K2_COLOR, strokeWidth=2).encode(y=alt.Y("score:Q", title="Riesgo a 3.000 km (K2)"))
score_pts = base.mark_point(color=K2_COLOR, filled=True, size=70).encode(
    y="score:Q", tooltip=[alt.Tooltip("cut_odo:Q", title="Odómetro", format=",.0f"),
                          alt.Tooltip("score:Q", title="Riesgo", format=".3f"),
                          alt.Tooltip("cut_date:T", title="Fecha del corte")])
layers = [score_line, score_pts,
          alt.Chart(pd.DataFrame({"y": [point["threshold"]], "t": [f"umbral {budget_label(b)}"]}))
          .mark_rule(color=MUTED, strokeDash=[5, 3]).encode(y="y:Q", tooltip=alt.Tooltip("t:N", title=""))]
if info["failed"]:
    layers.append(alt.Chart(pd.DataFrame({"x": [info["event_odo_km"]], "t": ["evento"]}))
                  .mark_rule(color=FLEET_COLOR, strokeWidth=2).encode(x="x:Q", tooltip=alt.Tooltip("t:N", title="")))
if info["alerted"]:
    layers.append(alt.Chart(v.loc[v["cut_odo"].eq(info["alert_cut_odo"])])
                  .mark_point(shape="diamond", size=220, color=K2_COLOR, strokeWidth=2)
                  .encode(x="cut_odo:Q", y="score:Q", tooltip=alt.value("primera alerta sostenida")))

with st.container(border=True):
    st.markdown("**Riesgo que K2 le asigna en cada corte**")
    st.altair_chart(alt.layer(*layers), width="stretch")
    st.caption(f"Alerta sostenida = {int(d.eval_cfg.get('k_consecutive', 2))} cortes seguidos sobre el umbral (rombo). "
               "La línea naranja es el evento: el modelo nunca ve los últimos 500 km antes (gap de blanking).")

# --- perfil de uso -------------------------------------------------------------------------
features = settings["profile_features"]
with st.container(border=True):
    st.markdown("**Cómo se usa este auto, comparado con los sanos**")
    feature = st.selectbox("Variable", list(features), format_func=features.get, key="feature")
    fleet = fleet_profile(d, feature, float(settings["profile_bin_km"]))
    mine = d.predictions.loc[d.predictions["vehicle_id"].eq(vid), ["cut_odo", feature]].dropna()
    x = alt.X("cut_odo:Q", title="Odómetro del corte [km]")
    series = alt.Scale(domain=["Este auto", "Mediana de los sanos"], range=[K2_COLOR, FLEET_COLOR])
    fleet_long = fleet.assign(serie="Mediana de los sanos", value=fleet["median"])
    mine_long = mine.assign(serie="Este auto", value=mine[feature])
    chart = alt.layer(
        alt.Chart(fleet).mark_area(color=FLEET_COLOR, opacity=0.12).encode(x=x, y="p25:Q", y2="p75:Q"),
        alt.Chart(fleet_long).mark_line(strokeDash=[5, 3], strokeWidth=2).encode(
            x=x, y="value:Q", color=alt.Color("serie:N", scale=series, legend=alt.Legend(title=None, orient="top-left"))),
        alt.Chart(mine_long).mark_line(strokeWidth=2, point=alt.OverlayMarkDef(filled=True, size=50)).encode(
            x=x, y=alt.Y("value:Q", title=features[feature]), color=alt.Color("serie:N", scale=series),
            tooltip=[alt.Tooltip("cut_odo:Q", title="Odómetro", format=",.0f"),
                     alt.Tooltip("value:Q", title=features[feature], format=".3f")]),
    )
    st.altair_chart(chart, width="stretch")
    st.caption("Cada punto agrega la ventana de 1.000 km previa al corte. La banda naranja va del cuartil 25 al 75 "
               "de los sanos de dev en ese tramo de odómetro. Es contexto: el modelo no compara contra la flota.")

with st.expander("Todos los autos de esta repetición"):
    st.dataframe(table.sort_values(["outcome", "lead_km"], ascending=[True, False]), hide_index=True, column_config={
        "vehicle_id": "Vehículo", "market": "Mercado", "failed": "Falla", "alerted": "Alertó",
        "alert_cut_odo": st.column_config.NumberColumn("Primera alerta [km]", format="%,.0f"),
        "lead_km": st.column_config.NumberColumn("Anticipación [km]", format="%,.0f"),
        "event_odo_km": st.column_config.NumberColumn("Evento [km]", format="%,.0f"),
        "max_score": st.column_config.NumberColumn("Riesgo máximo", format="%.3f"),
        "n_cuts": "Cortes", "outcome": "Resultado"})
