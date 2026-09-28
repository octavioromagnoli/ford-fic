"""Qué pasó después: las alertas de la temporada contra las fallas registradas, con los números oficiales."""

import altair as alt
import pandas as pd
import streamlit as st

from scripts.demo_app.common import EVENT_COLOR, MODEL_COLOR, MUTED, bundle
from src.agents.formatting import format_value, number, short_date

from scripts.demo_app.presentation import heading, chart_style, table
from scripts.demo_app.wording import model_name, promises

b = bundle()
meta = b.meta
v = b.vehicles
alerted = v.loc[v["alerted"]]
detected = alerted.loc[alerted["failed"]]
false_alarms = alerted.loc[~alerted["failed"]]
failed = v.loc[v["failed"]]
lead_weeks = ((detected["event_date"] - detected["alert_confirm_date"]).dt.days / 7)
lead_km = detected["alert_confirm_km_to_event"]

heading("Qué pasó después.", "Las alertas frente a las fallas registradas.")
st.caption(f"Del {short_date(meta['replay']['start'])} al {short_date(meta['replay']['end'])}: "
           f"{meta['replay']['description']}. Cada auto lo puntúa un modelo que no lo vio al entrenar.")

with st.container(horizontal=True, key="season_metrics"):
    st.metric("Fallas anticipadas", f"{len(detected)} de {len(failed)}", border=True,
              help="Autos que fallaron y recibieron la alerta confirmada antes de la falla.")
    st.metric("Anticipación mediana", f"{lead_weeks.median():.0f} semanas", border=True,
              help=f"Desde la alerta confirmada hasta la falla registrada: {format_value(lead_km.median(), 'int')} km de mediana.")
    st.metric("Alertas de más", f"{len(false_alarms)} de {meta['counts']['healthy']} sanos", border=True,
              help="Autos sanos que recibieron una alerta: es el costo del sistema.")
    st.metric("Fallas sin alerta", f"{len(failed) - len(detected)}", border=True,
              help=f"{model_name(meta)} no las vio: el diagnóstico actual de Ford sigue siendo necesario.")

# --- línea de tiempo: alerta → falla ----------------------------------------------------------
rows = []
for vid, r in alerted.sort_values("alert_confirm_date").iterrows():
    rows.append({"auto": vid, "desde": r["alert_confirm_date"],
                 "hasta": r["event_date"] if r["failed"] else pd.Timestamp(meta["replay"]["end"]),
                 "tipo": "Alerta → falla" if r["failed"] else "Alerta sin falla (falsa alarma)",
                 "detalle": (f"falló {((r['event_date'] - r['alert_confirm_date']).days / 7):.0f} semanas después"
                             if r["failed"] else "no falló en el período del replay")})
tl = pd.DataFrame(rows)
colors = alt.Scale(domain=["Alerta → falla", "Alerta sin falla (falsa alarma)"], range=[MODEL_COLOR, EVENT_COLOR])
order = tl["auto"].tolist()
# Una fila de 28 px por auto alertado: son decenas, así que el eje de fechas y la leyenda van arriba (lo primero que
# se lee) y el eje se repite abajo. Los meses, en castellano: Vega los escribe en inglés.
months = "['Ene','Feb','Mar','Abr','May','Jun','Jul','Ago','Sep','Oct','Nov','Dic'][month(datum.value)] + ' ' + year(datum.value)"
axis = dict(labelExpr=months, tickCount="month", labelOverlap=True)
segments = alt.Chart(tl).mark_rule(strokeWidth=3).encode(
    y=alt.Y("auto:N", sort=order, title=None),
    x=alt.X("desde:T", title=None, axis=alt.Axis(orient="top", **axis)),
    x2="hasta:T", color=alt.Color("tipo:N", scale=colors, legend=alt.Legend(orient="top", title=None, columns=1)),
    strokeDash=alt.condition(alt.datum.tipo == "Alerta → falla", alt.value([1, 0]), alt.value([3, 3])),
    tooltip=[alt.Tooltip("auto:N", title="Auto"), alt.Tooltip("desde:T", title="Alerta", format="%d-%m-%Y"),
             alt.Tooltip("detalle:N", title="")])
starts = alt.Chart(tl).mark_point(filled=True, size=80).encode(
    y=alt.Y("auto:N", sort=order), x=alt.X("desde:T", title=None, axis=alt.Axis(orient="bottom", **axis)),
    color=alt.Color("tipo:N", scale=colors, legend=None),
    tooltip=[alt.Tooltip("auto:N", title="Auto"), alt.Tooltip("desde:T", title="Alerta", format="%d-%m-%Y")])
ends = alt.Chart(tl.loc[tl["tipo"].eq("Alerta → falla")]).mark_point(shape="cross", size=110, color=EVENT_COLOR,
                                                                     strokeWidth=2).encode(
    y=alt.Y("auto:N", sort=order), x=alt.X("hasta:T", axis=None),
    tooltip=[alt.Tooltip("auto:N", title="Auto"), alt.Tooltip("hasta:T", title="Falla", format="%d-%m-%Y")])
# Un eje por capa (arriba el de los segmentos, abajo el de los círculos) sobre la misma escala, y una leyenda por capa:
# si se comparten, Vega avisa que una capa la oculta y otra no.
timeline_chart = alt.layer(segments, starts, ends).resolve_axis(x="independent").resolve_legend(color="independent")
with st.container(key="season_timeline"):
    st.altair_chart(chart_style(timeline_chart.properties(height=28 * len(tl)), fit="fit-x"),
                    width="stretch", theme=None)
    st.caption("Círculo: la alerta confirmada. Cruz: la falla registrada. Línea punteada: una alerta a un auto que no "
               "falló en el período del replay.")

# --- los números oficiales -------------------------------------------------------------------
official = st.container(key="official")
official.subheader("Los números medidos", anchor=False)
curve = pd.DataFrame(meta["official"]["curve"])
hold = pd.DataFrame(meta["official"]["holdout"])
emp = hold.loc[hold["method"].eq("empirical")].set_index("alpha")
points = []
shown = curve.loc[curve["budget_per_1000"].isin([50, 100])]
# El total de autos que fallan es el mismo en las dos columnas: va en la etiqueta, y los valores entran en un celular.
n_failed = int(shown["n_event_vehicles"].iloc[0])
for _, p in shown.iterrows():
    alpha = p["budget_per_1000"] / 1000
    h = emp.loc[alpha] if alpha in emp.index else None
    points.append({
        "Falsas alarmas toleradas": format_value(alpha, "pct"),
        "Detección (3 repeticiones)": f"{number(100 * p['detection'], 1)}% ± {number(100 * p['detection_sd'], 1)}",
        f"Detectados por repetición (de {n_failed})": p["detected"],
        "Anticipación mediana": format_value(p["lead_km"], "int") + " km",
        "Azar con el mismo historial": f"{number(100 * p['null'], 1)}%",
        "Fuera de muestra: detección / falsas alarmas": (f"{number(100 * h['detection'], 1)}% / "
                                                         f"{number(100 * h['fa_heldout'], 1)}%") if h is not None else "—",
    })
# Una columna por punto de operación y una fila por medida: entra en el ancho de un celular.
measures = [k for k in points[0] if k != "Falsas alarmas toleradas"]
with official:
    table(["Falsas alarmas toleradas"] + [p["Falsas alarmas toleradas"] for p in points],
          [[m] + [p[m] for p in points] for m in measures], nowrap_values=True)
official.caption(f"Muestra de desarrollo: {meta['counts']['failed']} autos que fallaron y {meta['counts']['healthy']} sanos. "
           f"La demo muestra la repetición {meta['repeat'] + 1}; la tabla, el promedio de las "
           f"{meta['official']['n_repeats']}. «Azar» es un score permutado que conserva el largo de cada historial.")

with st.expander("Qué no promete esta demo", icon=":material/info:"):
    st.markdown(promises(meta))
