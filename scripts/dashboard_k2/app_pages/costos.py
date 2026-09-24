"""Costos: dónde operar K2 según lo que cuesta una falla contra una alerta, y cuánto ahorra."""

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from scripts.dashboard_k2.common import (FLEET_COLOR, K2_COLOR, MUTED, cfg, cost_curves, cost_scenarios, data, dec,
                                         label, pct)
from src.eval.dashboard_data import LABEL_NAMES, cost_optimum, expected_cost, null_and_k2_budget_points

d, settings, sc = data(), cfg(), cost_scenarios()
lab = label()
if d.decision is None:
    st.error(f"Falta la capa de decisión. Corré `python scripts/decision_layer.py --config {settings['decision_config']}`.")
    st.stop()

scenarios = sc["scenarios"]
KEYS = {"insp": "cost_insp", "prev": "cost_prev", "fail": "cost_fail"}


def apply_preset() -> None:
    preset = scenarios[st.session_state["cost_preset"]]
    for field, key in KEYS.items():
        st.session_state[key] = int(preset[field])


if "cost_insp" not in st.session_state:
    st.session_state["cost_preset"] = "medio"
    apply_preset()


def usd(value: float) -> str:
    return f"USD {value:,.0f}".replace(",", ".")


st.markdown("Elegí un escenario precargado o cargá tus propios costos. **La prevalencia de dev (~32%) no sirve acá**: "
            "está inflada por el muestreo en dos cohortes; poné la de la flota real.")

with st.container(border=True):
    st.segmented_control("Escenario", options=list(scenarios), key="cost_preset", on_change=apply_preset,
                         format_func=lambda s: s.replace("_", " ").capitalize(),
                         help=f"Rangos de precios públicos de EE. UU., no costos de Ford. Fuentes en `{settings['cost_scenarios']}`.")
    with st.container(horizontal=True):
        st.number_input("Falsa alarma: diagnóstico [USD]", min_value=1, step=10, key="cost_insp",
                        help="Lo que cuesta alertar a un auto que no iba a fallar.")
        st.number_input("Acción preventiva [USD]", min_value=0, step=50, key="cost_prev",
                        help="Lo que se hace, además del diagnóstico, cuando la alerta es correcta (p. ej. regeneración forzada).")
        st.number_input("Falla no anticipada [USD]", min_value=0, step=100, key="cost_fail",
                        help="Reparación (limpieza o reemplazo del DPF), grúa y parada.")
    with st.container(horizontal=True):
        e = st.slider("Fallas que evita la prevención", 0.0, 1.0, 0.8, 0.05, format="percent", key="cost_e")
        pi = st.slider("Prevalencia real en la flota", 0.005, 0.20, 0.05, 0.005, format="percent", key="cost_pi")

costs = dict(pi=pi, insp=float(st.session_state["cost_insp"]), prev=float(st.session_state["cost_prev"]),
             fail=float(st.session_state["cost_fail"]), e=e)
curves = cost_curves(lab)
opt = cost_optimum(curves, d.decision, lab, **costs)
validated = float(sc["validated_max_fa"])

with st.container(horizontal=True):
    st.metric("Dónde operar", pct(opt["fa_opt"]) + " de falsas alarmas", border=True,
              help="El umbral de mínimo costo esperado, promedio de las 3 repeticiones.")
    st.metric("Detección en ese punto", pct(opt["det_opt"]), border=True)
    st.metric("Ahorro contra la mejor política trivial", pct(opt["savings_vs_trivial"], 0),
              f"la trivial es {opt['trivial']}", delta_color="off", border=True)
    st.metric("Ahorro cada 1.000 autos", usd(opt["savings_per_1000"]),
              f"{usd(opt['savings_over_null_per_1000'])} sobre el azar", delta_color="off", border=True,
              help="'Sobre el azar' = lo que K2 ahorra por encima de un score permutado que conserva el largo de cada "
                   "historial, en los presupuestos de 2/5/10/20%. Es la parte que sale del modelo.")

benefit = e * costs["fail"] - costs["prev"] - costs["insp"]
if opt["savings_per_1000"] <= 0.5:
    st.warning(f"Con estos costos, **{opt['trivial']}** es tan bueno como usar K2: el modelo no agrega valor.",
               icon=":material/money_off:")
elif opt["fa_opt_max"] > validated:
    st.warning(f"El óptimo cae por encima del {pct(validated, 0)} de falsas alarmas, fuera de lo verificado fuera de "
               "muestra: el ahorro es optimista.", icon=":material/warning:")
else:
    st.success(f"K2 paga: operar a ~{pct(opt['fa_opt'])} de falsas alarmas ahorra {usd(opt['savings_per_1000'])} "
               "cada 1.000 autos contra la mejor política trivial.", icon=":material/savings:")
st.caption(f"Alertar a un auto que va a fallar ahorra B = e·falla − prevención − diagnóstico = {usd(benefit)}; "
           f"B / diagnóstico = {dec(opt['ratio_benefit_insp'], 1)}. El modelo suele pagar con B / diagnóstico entre ~5 "
           "y ~15 y prevalencias de 2–5%.")

# --- costo esperado a lo largo de la curva ------------------------------------------------
frames = []
for r, (fpr, tpr) in enumerate(curves):
    order = np.argsort(fpr, kind="stable")
    frames.append(pd.DataFrame({"fa": fpr[order] * 100, "cost": expected_cost(fpr[order], tpr[order], **costs) * 1000,
                                "repeat": f"R{r + 1}"}))
k2_long = pd.concat(frames, ignore_index=True)
_, null_pts = null_and_k2_budget_points(d.decision, lab)
null = pd.DataFrame({"fa": null_pts[0] * 100, "cost": expected_cost(*null_pts, **costs) * 1000})
x = alt.X("fa:Q", title="Falsas alarmas sobre los sanos [%]")
y = alt.Y("cost:Q", title="Costo esperado cada 1.000 autos [USD]")
zone = alt.Chart(pd.DataFrame({"a": [0], "b": [validated * 100]})).mark_rect(color=K2_COLOR, opacity=0.06).encode(x="a:Q", x2="b:Q")
k2_lines = alt.Chart(k2_long).mark_line(color=K2_COLOR, strokeWidth=2, opacity=0.6).encode(
    x=x, y=y, detail="repeat:N", tooltip=[alt.Tooltip("repeat:N", title="Repetición"),
                                          alt.Tooltip("fa:Q", title="Falsas alarmas [%]", format=".1f"),
                                          alt.Tooltip("cost:Q", title="USD / 1.000", format=",.0f")])
null_line = alt.Chart(null).mark_line(color=MUTED, strokeDash=[5, 3], strokeWidth=2, point=True).encode(
    x=x, y=y, tooltip=[alt.Tooltip("fa:Q", title="Falsas alarmas [%]"), alt.Tooltip("cost:Q", title="USD / 1.000 (azar)", format=",.0f")])
trivial = pd.DataFrame({"cost": [opt["cost_none"] * 1000, opt["cost_everyone"] * 1000],
                        "policy": ["No alertar a nadie", "Alertar a todos"]})
rules = alt.Chart(trivial).mark_rule(strokeDash=[2, 2], color=FLEET_COLOR).encode(
    y="cost:Q", tooltip=[alt.Tooltip("policy:N", title="Política"), alt.Tooltip("cost:Q", title="USD / 1.000", format=",.0f")])
rule_labels = alt.Chart(trivial).mark_text(align="left", dx=4, dy=-6, color=FLEET_COLOR).encode(
    x=alt.value(0), y="cost:Q", text="policy:N")

with st.container(border=True):
    st.markdown(f"**Costo esperado según el punto de operación** · etiqueta {LABEL_NAMES[lab].lower()}")
    st.altair_chart(zone + k2_lines + null_line + rules + rule_labels, width="stretch")
    st.caption(f"Azul: K2 en cada repetición. Gris punteado: un score al azar del mismo largo de historial. La banda "
               f"marca hasta {pct(validated, 0)} de falsas alarmas, lo verificado fuera de muestra. Naranja: las dos "
               f"políticas triviales. Detalle y fuentes: `{settings['docs']['costs']}`.")
