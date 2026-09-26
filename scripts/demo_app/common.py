"""Lo compartido entre las páginas de la demo: configs, bundle, eventos, semana elegida y formato."""

from __future__ import annotations

import hmac
import os
from collections import Counter

import pandas as pd
import streamlit as st

from src.agents.bundle import Bundle, bundle_dir, load_bundle
from src.agents.facts import events_for
from src.agents.formatting import MONTHS, long_date
from src.agents.llm import LLM
from src.agents.policy import Event, weeks
from src.agents.triage import load_triage
from src.config import load_config

from scripts.demo_app.presentation import brand_block

DEMO_CONFIG = os.environ.get("DEMO_CONFIG", "configs/demo.yaml")
AGENTS_CONFIG = os.environ.get("AGENTS_CONFIG", "configs/agents.yaml")

# Paleta de presentación: azul para K2, ámbar para eventos y rojo para riesgo creciente.
K2_COLOR = "#4da3ff"
EVENT_COLOR = "#ffc16b"
MUTED = "#a3b8d2"
OUTCOME_COLORS = {"Detectado": "#4da3ff", "No detectado": "#a3b8d2", "Falsa alarma": "#ffc16b",
                  "Sano sin alerta": "#b7c5d9"}
EFFECT_COLORS = {"sube el riesgo": "#ff8585", "baja el riesgo": "#4da3ff", "no accionable": "#b7c5d9"}
KIND_LABELS = {"alerta_nueva": "Alerta nueva", "persistencia": "Escalamiento"}
# El canal de cada acción de la política: la app del vehículo o el concesionario.
ACTION_ICONS = {"aviso_conductor": ":material/notifications:", "turno_concesionario": ":material/car_repair:"}


@st.cache_data(show_spinner=False)
def cfg() -> dict:
    return load_config(DEMO_CONFIG)


@st.cache_data(show_spinner=False)
def agents_cfg() -> dict:
    return load_config(AGENTS_CONFIG)


@st.cache_resource(show_spinner="Cargando el bundle de la demo…")
def bundle() -> Bundle:
    return load_bundle(bundle_dir(cfg()))


@st.cache_resource(show_spinner=False)
def events() -> list[Event]:
    return events_for(bundle(), agents_cfg())


@st.cache_resource(show_spinner=False)
def llm() -> LLM:
    b = bundle()
    return LLM(agents_cfg()["llm"], b.root / agents_cfg()["llm"]["cache_subdir"])


def all_weeks() -> list[pd.Timestamp]:
    return weeks(bundle().meta)


# La semana vive en una clave que no es de un widget: Streamlit descarta el estado de los widgets al cambiar
# de página con st.switch_page («Ver ficha»), y la ficha abría en la primera semana. Los controles (el
# calendario y el dock) solo reflejan esta clave.

def current_week() -> pd.Timestamp:
    if "_week" not in st.session_state:
        st.session_state["_week"] = bundle().meta["replay"]["first_week"]
    return pd.Timestamp(st.session_state["_week"])


def set_week(week: pd.Timestamp) -> None:
    st.session_state["_week"] = pd.Timestamp(week).date().isoformat()


def week_label(week: pd.Timestamp) -> str:
    end = pd.Timestamp(week) + pd.Timedelta(days=6)
    return f"{week.day} de {long_date(week).split(' de ')[1]} al {long_date(end)}"


def week_label_short(week: pd.Timestamp) -> str:
    """`29 sep – 5 oct 2025`: la semana en el ancho de un celular."""
    start, end = pd.Timestamp(week), pd.Timestamp(week) + pd.Timedelta(days=6)
    first = f"{start.day} {MONTHS[start.month - 1][:3]}" + (f" {start.year}" if start.year != end.year else "")
    return f"{first} – {end.day} {MONTHS[end.month - 1][:3]} {end.year}"


def timeline_data(current: pd.Timestamp) -> dict:
    """Lo que dibuja el calendario de la flota: una columna por semana con sus alertas nuevas y escalamientos, y
    el mes en el eje donde empieza uno (con el año en el primero y en enero)."""
    per_week = Counter((e.week.date().isoformat(), e.kind) for e in events())
    rows, last_month = [], None
    for i, week in enumerate(all_weeks()):
        iso = week.date().isoformat()
        new_month = week.month != last_month
        last_month = week.month
        rows.append({
            "iso": iso,
            "title": f"Semana del {week_label(week)}",
            "range": week_label_short(week),
            "new": per_week[(iso, "alerta_nueva")],
            "esc": per_week[(iso, "persistencia")],
            "month": MONTHS[week.month - 1][:3] if new_month else "",
            "year": str(week.year) if new_month and (i == 0 or week.month == 1) else "",
        })
    return {"weeks": rows, "current": pd.Timestamp(current).date().isoformat()}


def triage_for(week: pd.Timestamp) -> dict | None:
    """El triage de la semana: el que se corrió en esta sesión o el que viaja en el bundle."""
    key = f"triage:{pd.Timestamp(week).date()}"
    return st.session_state.get(key) or load_triage(bundle(), week)


def store_triage(result: dict) -> None:
    st.session_state[f"triage:{result['week']}"] = result


def check_password() -> bool:
    """Si `DEMO_PASSWORD` está definida, nada se muestra sin la clave."""
    expected = os.environ.get("DEMO_PASSWORD")
    if not expected or st.session_state.get("auth_ok"):
        return True
    with st.container(key="login"):
        st.markdown(brand_block(cfg()["product_name"]), unsafe_allow_html=True)
        with st.form("login_form"):
            pwd = st.text_input("Contraseña de la demo", type="password")
            if st.form_submit_button("Entrar", type="primary"):
                if hmac.compare_digest(pwd.encode(), expected.encode()):
                    st.session_state["auth_ok"] = True
                    st.rerun()
                st.error("Contraseña incorrecta.")
    return False
