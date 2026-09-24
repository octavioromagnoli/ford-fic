"""Resumen: el número del pitch, el dial de falsas alarmas y si el punto se sostiene."""

import altair as alt
import pandas as pd
import streamlit as st

from scripts.dashboard_k2.common import K2_COLOR, MUTED, budget, budget_label, cfg, data, km, label, pct
from src.eval.dashboard_data import LABEL_NAMES, curve_summary, holdout_summary

d = data()
if d.decision is None:
    st.error(f"Falta la capa de decisión. Corré `python scripts/decision_layer.py --config {cfg()['decision_config']}`.")
    st.stop()

lab, b = label(), budget()
summary = curve_summary(d.decision, lab)
row = summary.loc[summary["budget_per_1000"].eq(b)].iloc[0]
holdout = holdout_summary(d.decision, lab)
held = holdout.loc[holdout["alpha"].sub(b / 1000).abs().lt(1e-9) & holdout["method"].eq("empirical")]

with st.container(horizontal=True):
    st.metric("Autos que van a fallar, detectados", pct(row["detection"]),
              f"{100 * row['excess']:+.1f} pts sobre el azar".replace(".", ","), border=True,
              help="Media de las 3 repeticiones de la CV. 'Azar' = un score permutado que conserva "
                   "cuántos cortes tiene cada auto (regla 6).")
    st.metric("Detectados por repetición", f"{row['detected']} de {int(row['n_event_vehicles'])}",
              f"± {100 * row['detection_sd']:.1f} pts".replace(".", ","), delta_color="off", border=True)
    st.metric("Anticipación mediana", km(row["lead_km"]),
              "desde la primera alerta sostenida", delta_color="off", border=True)
    if not held.empty:
        h = held.iloc[0]
        st.metric("Falsas alarmas fuera de muestra", pct(h["fa_heldout"]),
                  f"objetivo {budget_label(b)}", delta_color="off", border=True,
                  help="Umbral fijado con los sanos de otros folds y medido en los que no participaron.")

if row["excess"] <= 0:
    st.warning(f"A {budget_label(b)} de falsas alarmas K2 detecta lo mismo que un score al azar. "
               "No es un punto de operación defendible.", icon=":material/warning:")
else:
    st.info(f"Con **{budget_label(b)} de falsas alarmas** sobre los autos sanos, K2 detecta **{pct(row['detection'])}** "
            f"de los autos que van a fallar ({row['detected']} de {int(row['n_event_vehicles'])} en las tres "
            f"repeticiones), con una mediana de **{km(row['lead_km'])}** de anticipación. Etiqueta "
            f"{LABEL_NAMES[lab].lower()}.", icon=":material/campaign:")

# --- curva: detección y anticipación contra el presupuesto de falsas alarmas --------------
plot = summary.assign(fa_pct=summary["budget_per_1000"] / 10)
k2 = plot.assign(serie="K2", y=plot["detection"] * 100, lo=plot["detection_min"] * 100, hi=plot["detection_max"] * 100)
null = plot.assign(serie="Azar (mismo largo de historial)", y=plot["null"] * 100, lo=plot["null"] * 0, hi=plot["null_p95"] * 100)
long = pd.concat([k2, null], ignore_index=True)
colors = alt.Scale(domain=["K2", "Azar (mismo largo de historial)"], range=[K2_COLOR, MUTED])
x = alt.X("fa_pct:Q", title="Falsas alarmas sobre los sanos [%]", scale=alt.Scale(domain=[0, 21]))
tooltip = [alt.Tooltip("serie:N", title="Serie"), alt.Tooltip("fa_pct:Q", title="Falsas alarmas [%]"),
           alt.Tooltip("y:Q", title="Detección [%]", format=".1f"), alt.Tooltip("detected:N", title="Detectados (K2)")]
band = alt.Chart(long).mark_area(opacity=0.15).encode(x=x, y="lo:Q", y2="hi:Q", color=alt.Color("serie:N", scale=colors, legend=None))
line = alt.Chart(long).mark_line(strokeWidth=2).encode(
    x=x, y=alt.Y("y:Q", title="Autos que fallan detectados [%]"),
    color=alt.Color("serie:N", scale=colors, legend=alt.Legend(title=None, orient="top-left")),
    strokeDash=alt.condition(alt.datum.serie == "K2", alt.value([1, 0]), alt.value([5, 3])))
points = alt.Chart(long).mark_point(size=80, filled=True).encode(x=x, y="y:Q", color=alt.Color("serie:N", scale=colors, legend=None), tooltip=tooltip)
rule = alt.Chart(pd.DataFrame({"fa_pct": [b / 10]})).mark_rule(color=MUTED, strokeDash=[2, 2]).encode(x="fa_pct:Q")

left, right = st.columns([3, 2])
with left.container(border=True):
    st.markdown("**Cuántos autos se detectan según cuántas falsas alarmas se aceptan**")
    st.altair_chart(band + line + points + rule, width="stretch")
    st.caption("Banda de K2: mínimo y máximo de las 3 repeticiones. Banda del azar: hasta su percentil 95.")
with right.container(border=True):
    st.markdown("**Anticipación mediana en cada punto**")
    bars = alt.Chart(plot).mark_bar(color=K2_COLOR, cornerRadiusTopLeft=4, cornerRadiusTopRight=4, size=28).encode(
        x=alt.X("fa_pct:O", title="Falsas alarmas [%]"),
        y=alt.Y("lead_km:Q", title="Km antes del evento"),
        tooltip=[alt.Tooltip("fa_pct:O", title="Falsas alarmas [%]"), alt.Tooltip("lead_km:Q", title="Km", format=",.0f")])
    st.altair_chart(bars, width="stretch")
    st.caption("Aceptar más falsas alarmas suma autos, pero los nuevos se alertan más cerca del evento.")

# --- fuera de muestra -------------------------------------------------------------------
with st.container(border=True):
    st.markdown("**¿El punto se sostiene con autos sanos que no se usaron para fijarlo?**")
    table = holdout.assign(
        objetivo=holdout["alpha"].map(lambda a: budget_label(a * 1000)),
        metodo=holdout["method"].map({"empirical": "Empírico (lo de hoy)",
                                      "neyman_pearson": "Con garantía (Neyman-Pearson, 95%)"}),
    )[["objetivo", "metodo", "fa_heldout", "fa_max", "detection", "detected", "lead_km", "fa_in_ci95_hi"]]
    st.dataframe(table, hide_index=True, column_config={
        "objetivo": st.column_config.TextColumn("Objetivo de falsas alarmas"),
        "metodo": st.column_config.TextColumn("Cómo se fija el umbral"),
        "fa_heldout": st.column_config.NumberColumn("Falsas alarmas realizadas", format="percent"),
        "fa_max": st.column_config.NumberColumn("Peor repetición", format="percent"),
        "detection": st.column_config.NumberColumn("Detección", format="percent"),
        "detected": st.column_config.TextColumn("Detectados por repetición"),
        "lead_km": st.column_config.NumberColumn("Anticipación mediana [km]", format="%,.0f"),
        "fa_in_ci95_hi": st.column_config.NumberColumn("Tope del IC95 dentro de muestra", format="percent",
                                                       help="Clopper-Pearson sobre los sanos de dev: hasta dónde "
                                                            "podría llegar la tasa real sin que dev lo vea."),
    })
    st.caption("Neyman-Pearson elige el umbral para que la tasa real de falsas alarmas no supere el objetivo con 95% "
               "de confianza; con 95 sanos eso lo vuelve conservador. "
               f"Detalle: `{cfg()['docs']['decision']}`.")
