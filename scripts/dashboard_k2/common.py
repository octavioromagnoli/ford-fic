"""Lo compartido entre las páginas: config, carga cacheada y formato."""

from __future__ import annotations

import argparse
import os

import streamlit as st

from src.config import load_config
from src.eval.dashboard_data import (K2Data, curve_points, load_k2, operating_threshold, repeat_curve, rows_for,
                                     vehicle_alerts)

DEFAULT_CONFIG = "configs/dashboard_k2.yaml"

# Paleta categórica validada (skill dataviz, slots 1 y 2) + gris para nulos y referencias.
K2_COLOR = "#2a78d6"
FLEET_COLOR = "#eb6834"
MUTED = "#8a8984"
OUTCOME_COLORS = {"Detectado": "#2a78d6", "No detectado": "#8a8984",
                  "Falsa alarma": "#eb6834", "Sano sin alerta": "#c3c2b7"}


def config_path() -> str:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=os.environ.get("DASHBOARD_K2_CONFIG", DEFAULT_CONFIG))
    known, _ = parser.parse_known_args()
    return known.config


@st.cache_data(show_spinner="Cargando K2…", max_entries=2)
def cfg() -> dict:
    return load_config(config_path())


@st.cache_data(show_spinner="Cargando las predicciones de K2…", max_entries=2)
def data() -> K2Data:
    return load_k2(config_path())


@st.cache_data(show_spinner=False, max_entries=4)
def cost_scenarios() -> dict:
    return load_config(cfg()["cost_scenarios"])


@st.cache_data(show_spinner=False, max_entries=4)
def cost_curves(label: str) -> list:
    d = data()
    rows = rows_for(d, label)
    return [curve_points(repeat_curve(rows, r, d.eval_cfg)) for r in range(d.n_repeats)]


@st.cache_data(show_spinner=False, max_entries=16)
def alerts(label: str, repeat: int, budget: float) -> tuple[dict | None, object]:
    d = data()
    rows = rows_for(d, label)
    point = operating_threshold(rows, repeat, budget, d.eval_cfg)
    if point is None:
        return None, None
    return point, vehicle_alerts(rows, repeat, point["threshold"], int(d.eval_cfg.get("k_consecutive", 2)))


def budget_label(budget: float) -> str:
    return f"{budget / 10:g}%"


def km(value: float) -> str:
    return "—" if value != value else f"{value:,.0f} km".replace(",", ".")


def pct(value: float, digits: int = 1) -> str:
    return f"{100 * value:.{digits}f}%".replace(".", ",")


def dec(value: float, digits: int = 3, sign: bool = False) -> str:
    return f"{value:{'+' if sign else ''}.{digits}f}".replace(".", ",")


def label() -> str:
    return st.session_state.get("label") or cfg()["default_label"]


def budget() -> int:
    return st.session_state.get("budget") or cfg()["default_budget_per_1000"]
