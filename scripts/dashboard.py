#!/usr/bin/env python
"""Esqueleto del dashboard (Track C, F0). Levanta un parquet de predicciones y dibuja.

    streamlit run scripts/dashboard.py -- --run experiments/f0-dummy-baserate

No calcula nada propio: reusa `src/eval/metrics.py` y `src/eval/plots.py`, así lo
que se ve acá es lo mismo que se loguea en wandb y lo mismo que va al informe.
Contra el panel dummy sirve para verificar que la app funciona; cuando llegue el
panel real solo cambia el directorio de la corrida.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import repo_root, resolve_path  # noqa: E402
from src.eval.metrics import classification_metrics, lead_time_curve, operating_point  # noqa: E402
from src.eval.plots import (  # noqa: E402
    plot_detection_vs_false_alarms,
    plot_lead_time_curve,
    plot_vehicle_score,
)

DEFAULT_RUN = "experiments/f0-dummy-baserate"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default=DEFAULT_RUN, help="Directorio de la corrida")
    known, _ = parser.parse_known_args()
    return known


@st.cache_data(show_spinner=False)
def load_predictions(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


def main() -> None:
    args = parse_args()
    st.set_page_config(page_title="Ford FIC · Degradación de combustión", layout="wide")
    st.title("Predicción temprana de degradación de eficiencia de combustión")

    run_dir = resolve_path(st.sidebar.text_input("Corrida", args.run))
    predictions_path = run_dir / "predictions.parquet"
    if not predictions_path.exists():
        st.error(
            f"No hay predicciones en {predictions_path}. Corré primero "
            "`python scripts/train.py --config configs/exp_dummy.yaml`."
        )
        st.stop()

    predictions = load_predictions(str(predictions_path))
    k_consecutive = st.sidebar.slider("Cortes consecutivos para alerta sostenida (K)", 1, 5, 2)
    budget = st.sidebar.slider("Presupuesto de falsas alarmas por 1.000 sanos", 0, 500, 50, step=10)

    if run_dir.name.startswith("f0-") or "dummy" in str(run_dir):
        st.warning(
            "Corrida sobre el **panel dummy**: los números son ruido por construcción. "
            "Sirve para validar la app, no para decidir nada."
        )

    oof = classification_metrics(predictions["label"], predictions["score"])
    curve = lead_time_curve(predictions, k_consecutive=k_consecutive)
    point = operating_point(curve, max_false_alarms_per_1000=budget)

    cols = st.columns(4)
    cols[0].metric("PR-AUC (OOF)", f"{oof['pr_auc']:.3f}", f"tasa base {oof['base_rate']:.3f}")
    cols[1].metric("ROC-AUC", f"{oof['roc_auc']:.3f}")
    cols[2].metric(
        "Detección en el presupuesto",
        f"{100 * point['detection_rate']:.0f}%" if point else "—",
    )
    cols[3].metric(
        "Anticipación mediana",
        f"{point['median_lead_km']:.0f} km" if point and pd.notna(point["median_lead_km"]) else "—",
    )

    if point:
        st.info(
            f"A {point['false_alarms_per_1000']:.0f} falsas alarmas cada 1.000 vehículos sanos, "
            f"detectamos el {100 * point['detection_rate']:.0f}% de los casos con una mediana de "
            f"{point['median_lead_km']:.0f} km de anticipación (umbral {point['threshold']:.4f}, "
            f"K={k_consecutive})."
        )

    left, right = st.columns(2)
    left.pyplot(plot_lead_time_curve(curve, budget=budget))
    right.pyplot(plot_detection_vs_false_alarms(curve, budget=budget))

    st.subheader("Caso individual")
    event_vehicles = sorted(
        predictions.loc[predictions["event_observed"] == 1, "vehicle_id"].unique().tolist()
    )
    other = sorted(
        predictions.loc[predictions["event_observed"] == 0, "vehicle_id"].unique().tolist()
    )
    vehicle_id = st.selectbox("Vehículo", event_vehicles + other)
    st.pyplot(plot_vehicle_score(predictions, vehicle_id, threshold=point["threshold"] if point else None))

    with st.expander("Curva completa"):
        st.dataframe(curve, width="stretch")

    st.caption(f"Repo: {repo_root()} · corrida: {run_dir}")


if __name__ == "__main__":
    main()
