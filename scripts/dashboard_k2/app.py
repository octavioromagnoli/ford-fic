"""Dashboard de F4 contra el panel real, con el finalista K2 (solo dev).

    streamlit run scripts/dashboard_k2/app.py [-- --config configs/dashboard_k2.yaml]

Con FORD_DATA_DIR apuntando al build de K2 (data/rebuild-0921). Necesita K2 entrenado
(`scripts/train.py --config configs/exp_ss_hw_r3.yaml`), su `eval_window_label.py` y la capa
de decisión (`scripts/decision_layer.py --config configs/exp_decision_k2.yaml`).
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

import streamlit as st  # noqa: E402

from scripts.dashboard_k2.common import budget_label, cfg, data  # noqa: E402
from src.eval.dashboard_data import LABEL_NAMES, rows_for  # noqa: E402

st.set_page_config(page_title="Ford FIC · K2", page_icon=":material/local_gas_station:", layout="wide")

page = st.navigation(
    [
        st.Page(HERE / "app_pages" / "resumen.py", title="Resumen", icon=":material/dashboard:", default=True),
        st.Page(HERE / "app_pages" / "vehiculo.py", title="Vehículo", icon=":material/directions_car:"),
        st.Page(HERE / "app_pages" / "costos.py", title="Costos", icon=":material/payments:"),
        st.Page(HERE / "app_pages" / "modelo.py", title="Modelo", icon=":material/fact_check:"),
    ],
    position="top",
)

settings = cfg()
with st.sidebar:
    st.subheader("Punto de operación")
    st.segmented_control(
        "Falsas alarmas (sobre los sanos)", options=[20, 50, 100, 200], format_func=budget_label,
        default=settings["default_budget_per_1000"], required=True, key="budget",
        help="Qué fracción de autos sanos se acepta alertar. Elegirla es una decisión de costo de Ford.",
    )
    st.segmented_control(
        "Etiqueta", options=["corrected", "hard"], format_func=LABEL_NAMES.get,
        default=settings["default_label"], required=True, key="label",
        help="V cuenta solo los horizontes que el registro de eventos cubría (decide); D, todos.",
    )
    rows = rows_for(data(), st.session_state.get("label") or settings["default_label"])
    st.caption(f"Solo dev: {rows['vehicle_id'].nunique()} autos con esta etiqueta, "
               f"{rows.loc[rows['event_observed'].eq(1), 'vehicle_id'].nunique()} que fallan. "
               "El test no se muestra.")

st.title("Detección temprana de degradación de combustión", anchor=False)
st.caption("Finalista K2 · survival stacking con la ventana del registro, en km")
page.run()
