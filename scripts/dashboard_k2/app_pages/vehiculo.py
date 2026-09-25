"""Vehículo: el score de K2 a lo largo del odómetro, el umbral, la alerta, el porqué y el perfil de uso."""

import altair as alt
import pandas as pd
import streamlit as st

from scripts.dashboard_k2.common import (EFFECT_COLORS, FLEET_COLOR, K2_COLOR, MUTED, OUTCOME_COLORS, alerts, budget,
                                         budget_label, cfg, data, explanations, km, label)
from src.eval.dashboard_data import LABEL_NAMES, fleet_profile, rows_for, vehicle_why

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

# --- por qué -------------------------------------------------------------------------------
VARIANT_NAMES = {"V1": "TreeSHAP del hazard (una repetición)", "V2": "SHAP del score (Permutation)",
                 "V3": "TreeSHAP del hazard, promedio de 3 repeticiones", "V4": "V3 por familia"}
expl = explanations()
with st.container(border=True):
    st.markdown("**Por qué: qué parte del uso explica el riesgo**")
    why = None if expl is None else vehicle_why(expl, vid, lab, repeat, top=int(settings.get("explain_waterfall_top", 6)))
    if expl is None:
        st.info("Todavía no hay explicaciones. Corré `python scripts/explain_k2.py --config configs/explain_k2.yaml`.")
    elif why is None:
        st.info("No hay explicación para este auto con esta etiqueta (o ninguna variante pasó la fidelidad).")
    else:
        steps = why["steps"]
        records = [{"paso": "base (hazard medio del entrenamiento)", "inicio": why["base"], "fin": why["base"],
                    "efecto": "base", "valor": why["base"]}]
        running = why["base"]
        for _, s in steps.iterrows():
            effect = ("sube el riesgo" if s["value"] > 0 else "baja el riesgo") if s["kind"] == "accionable" else "no accionable"
            records.append({"paso": ("✓ " if s["in_message"] else "") + s["step"], "inicio": running,
                            "fin": running + s["value"], "efecto": effect, "valor": s["value"]})
            running += s["value"]
        records.append({"paso": "salida del auto", "inicio": why["output"], "fin": why["output"], "efecto": "salida",
                        "valor": why["output"]})
        wf = pd.DataFrame(records)
        wf["tope"] = wf[["inicio", "fin"]].max(axis=1)
        wf["etiqueta"] = wf["valor"].map(lambda x: f"{x:+.2f}".replace(".", ","))
        order = wf["paso"].tolist()
        y = alt.Y("paso:N", sort=order, title=None, axis=alt.Axis(labelLimit=420))
        x_title = "log-odds del hazard (promedio de los 6 tramos de H)"
        steps_df = wf[~wf["efecto"].isin(["base", "salida"])]
        ends_df = wf[wf["efecto"].isin(["base", "salida"])]
        chart = alt.layer(
            alt.Chart(steps_df).mark_bar(size=16, cornerRadius=2).encode(
                y=y, x=alt.X("inicio:Q", title=x_title, scale=alt.Scale(zero=False)), x2="fin:Q",
                color=alt.Color("efecto:N", scale=alt.Scale(domain=list(EFFECT_COLORS), range=list(EFFECT_COLORS.values())),
                                legend=alt.Legend(title=None, orient="top")),
                tooltip=[alt.Tooltip("paso:N", title=""), alt.Tooltip("valor:Q", title="contribución", format="+.3f")]),
            alt.Chart(steps_df).mark_text(align="left", dx=4, color=MUTED).encode(y=y, x="tope:Q", text="etiqueta:N"),
            alt.Chart(ends_df).mark_point(filled=True, size=90, color=MUTED).encode(
                y=y, x="inicio:Q", tooltip=[alt.Tooltip("paso:N", title=""), alt.Tooltip("valor:Q", title="log-odds", format=".3f")]),
            alt.Chart(ends_df).mark_text(align="left", dx=8, color=MUTED).encode(y=y, x="inicio:Q", text="etiqueta:N"),
        ).properties(height=alt.Step(28))  # una fila por paso: sin esto Vega comprime y oculta etiquetas
        st.altair_chart(chart, width="stretch")
        st.markdown("**Mensaje al cliente**")
        if why["message"]:
            st.code(why["message"], language=None, wrap_lines=True)
        prereg = expl.budget_per_1000
        st.caption(
            f"Variante elegida por la regla preregistrada: {why['variant']} ({VARIANT_NAMES.get(why['variant'], '')}), "
            f"al {budget_label(prereg)} de falsas alarmas con la etiqueta {LABEL_NAMES.get(lab, lab)}. "
            "Rojo: la accionable empuja el riesgo hacia arriba; azul, hacia abajo; gris: contexto y síntomas, que "
            "nunca llegan al mensaje. ✓: el factor pasa todos los filtros (física, estabilidad, valor del lado riesgoso "
            "de la mediana sana) y aparece en el mensaje. SHAP explica al modelo, no al auto: es parecido con autos que "
            "fallaron, no una causa."
            + ("" if float(b) == prereg else f" Con {budget_label(b)} cambian las alertas de arriba; la explicación "
               f"y el mensaje siguen siendo los del {budget_label(prereg)}."))

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
