"""Demo de producto: la flota de dev reproducida en el calendario con el modelo del bundle y los agentes de triage.

    streamlit run scripts/demo_app/app.py

Lee solo el bundle (`scripts/build_demo_bundle.py`, o `scripts/demo_app/fetch_bundle.py` en el
contenedor). Variables: `DEMO_PASSWORD` (si está, pide clave), `OPENAI_API_KEY` (para correr los
agentes en vivo), `DEMO_LLM_MODE` (cache_first | cache_only | live), `DEMO_BUNDLE_DIR`, `DEMO_CONFIG`. El modelo
que se muestra (nombre y familia) sale de la meta del bundle.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

import streamlit as st  # noqa: E402

from scripts.demo_app.common import (all_weeks, bundle, cfg, check_password, current_week, events,  # noqa: E402
                                     llm, set_week, timeline_data, week_label_short)
from scripts.demo_app.guide import tour_steps  # noqa: E402
from scripts.demo_app.wording import model_note  # noqa: E402

from scripts.demo_app.presentation import brand_block, brand_html, inject_theme, timeline, tour  # noqa: E402

st.set_page_config(page_title=cfg()["brand"]["page_title"], page_icon=":material/local_gas_station:", layout="wide")

inject_theme()

if not check_password():
    st.stop()

page = st.navigation(
    [
        st.Page(HERE / "app_pages" / "bandeja.py", title="Bandeja", icon=":material/inbox:", default=True),
        st.Page(HERE / "app_pages" / "vehiculo.py", title="Vehículo", icon=":material/directions_car:"),
        st.Page(HERE / "app_pages" / "resultados.py", title="Qué pasó después", icon=":material/timeline:"),
    ],
    position="hidden",
)

# st.page_link does not expose aria-current; mark the selected route explicitly.
st.markdown(f'''<style>[data-testid="stSidebar"] a[href="{page.url_path}"] {{background:var(--active);border-color:var(--active-line)}}
[data-testid="stSidebar"] a[href="{page.url_path}"] p {{color:var(--text)!important;font-weight:700}}
.st-key-dock_nav a[href="{page.url_path}"] [data-testid="stIconMaterial"] {{background:var(--active);color:var(--accent-text)}}
.st-key-dock_nav a[href="{page.url_path}"] p {{color:var(--text)!important}}</style>''', unsafe_allow_html=True)

b = bundle()
week_list = all_weeks()
week = current_week()
event_weeks = sorted({e.week for e in events()})


def _move(delta: int) -> None:
    i = week_list.index(current_week()) + delta
    set_week(week_list[max(0, min(len(week_list) - 1, i))])


def _next_with_events() -> None:
    later = [w for w in event_weeks if w > current_week()]
    if later:
        set_week(later[0])


def _from_timeline() -> None:
    chosen = st.session_state["timeline"].get("week")
    if chosen:
        set_week(pd.Timestamp(chosen))


c = b.meta["counts"]
FLEET_NOTE = (f"Flota de la muestra de desarrollo: {c['vehicles']} autos diésel conectados, {c['failed']} con falla "
              "registrada (la muestra está enriquecida en fallas; en una flota real son muchos menos). El test "
              "no se muestra.")
MODEL_NOTE = model_note(b.meta, llm().model, llm().mode)

with st.sidebar:
    st.markdown(brand_block(cfg()["brand"]["team"]), unsafe_allow_html=True)
    with st.container(key="side_nav"):
        st.page_link("app_pages/bandeja.py", label="Bandeja", icon=":material/inbox:")
        st.page_link("app_pages/vehiculo.py", label="Vehículo", icon=":material/directions_car:")
        st.page_link("app_pages/resultados.py", label="Qué pasó después", icon=":material/timeline:")
    st.divider()
    st.caption(FLEET_NOTE)
    st.caption(MODEL_NOTE)

# Arriba de toda página: qué es esto y cómo se recorre («Cómo usar»), y el calendario de la flota, que es el control
# que más cambia lo que se ve. «Qué pasó después» muestra la temporada entera, así que ahí no va. La marca solo
# aparece acá cuando la barra lateral está colapsada (celular).
page_id = {"vehiculo": "vehiculo", "resultados": "resultados"}.get(page.url_path, "bandeja")
with st.container(horizontal=True, vertical_alignment="center", key="topbar"):
    st.markdown('<div class="topbar-lead">' + brand_html(cfg()["brand"]["team"], "masthead-brand") +
                '<div class="status">Replay · muestra de desarrollo</div></div>', unsafe_allow_html=True)
    with st.container(key="tour", width="content"):
        tour(tour_steps(page_id))
if page_id != "resultados":
    with st.container(key="weekbar"):
        timeline(timeline_data(week), on_week=_from_timeline)

# Controles compactos. En 768 px o menos Streamlit colapsa la barra lateral, y con el header oculto no hay
# cómo abrirla: lo que vive ahí se repite acá (la semana y las páginas en un dock abajo, al alcance del
# pulgar; las notas al final). theme.css los muestra solo con la barra colapsada. La semana no va en «Qué pasó
# después», igual que el calendario: esa página muestra la temporada entera.
with st.container(key="compact_dock"):
    if page_id != "resultados":
        with st.container(horizontal=True, vertical_alignment="center", gap="small", key="dock_week"):
            st.button("Semana anterior", key="dock_prev", icon=":material/chevron_left:", on_click=_move, args=(-1,),
                      disabled=week == week_list[0])
            st.markdown(f'<div class="dock-week"><span>Semana</span><strong>{week_label_short(week)}</strong></div>',
                        unsafe_allow_html=True)
            st.button("Semana siguiente", key="dock_next", icon=":material/chevron_right:", on_click=_move, args=(1,),
                      disabled=week == week_list[-1])
            st.button("Próxima", key="dock_skip", icon=":material/skip_next:", on_click=_next_with_events,
                      help="Próxima semana con alertas", disabled=not any(w > week for w in event_weeks))
    with st.container(horizontal=True, gap=None, key="dock_nav"):
        st.page_link("app_pages/bandeja.py", label="Bandeja", icon=":material/inbox:")
        st.page_link("app_pages/vehiculo.py", label="Vehículo", icon=":material/directions_car:")
        st.page_link("app_pages/resultados.py", label="Qué pasó después", icon=":material/timeline:")
with st.container(key="compact_notes"):
    st.caption(FLEET_NOTE)
    st.caption(MODEL_NOTE)

page.run()
