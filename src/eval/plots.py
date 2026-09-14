"""Figuras del pitch. Las usan tanto el dashboard como la exportación headless.

Todas reciben DataFrames ya calculados (nunca recalculan métricas) y devuelven
una figura de matplotlib, para que la misma curva que se ve en el dashboard sea
exactamente la que va al informe.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

FORD_BLUE = "#1c3f94"
ALERT_RED = "#c0392b"


def plot_lead_time_curve(curve: pd.DataFrame, *, ax=None, budget: float | None = None):
    """Anticipación vs. falsas alarmas: la pieza central del pitch.

    Eje x: falsas alarmas por cada 1.000 vehículos sanos. Eje y: mediana de km de
    anticipación. El color codifica qué fracción de los casos se detecta, porque
    una anticipación enorme sobre el 5% de los casos no es un buen punto.
    """
    data = curve.dropna(subset=["false_alarms_per_1000", "median_lead_km"]).sort_values(
        "false_alarms_per_1000"
    )
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(data["false_alarms_per_1000"], data["median_lead_km"], color=FORD_BLUE, lw=1.5, zorder=1)
    scatter = ax.scatter(
        data["false_alarms_per_1000"],
        data["median_lead_km"],
        c=data["detection_rate"],
        cmap="viridis",
        vmin=0,
        vmax=1,
        s=38,
        zorder=2,
    )
    if budget is not None:
        ax.axvline(budget, color=ALERT_RED, ls="--", lw=1, label=f"presupuesto: {budget:g}/1000")
        ax.legend(loc="lower right", fontsize=8)
    ax.set_xlabel("Falsas alarmas por cada 1.000 vehículos sanos")
    ax.set_ylabel("Anticipación mediana [km]")
    ax.set_title("Anticipación vs. falsas alarmas")
    ax.grid(alpha=0.25)
    plt.colorbar(scatter, ax=ax, label="fracción de casos detectados")
    return ax.figure


def plot_detection_vs_false_alarms(curve: pd.DataFrame, *, ax=None, budget: float | None = None):
    """La otra mitad de la frase del pitch: cuántos casos se detectan a cada costo."""
    data = curve.dropna(subset=["false_alarms_per_1000", "detection_rate"]).sort_values(
        "false_alarms_per_1000"
    )
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(data["false_alarms_per_1000"], 100 * data["detection_rate"], color=FORD_BLUE, lw=1.8)
    if budget is not None:
        ax.axvline(budget, color=ALERT_RED, ls="--", lw=1)
    ax.set_xlabel("Falsas alarmas por cada 1.000 vehículos sanos")
    ax.set_ylabel("Casos detectados [%]")
    ax.set_title("Cobertura vs. costo")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.25)
    return ax.figure


def plot_vehicle_score(
    predictions: pd.DataFrame, vehicle_id: str, *, threshold: float | None = None, ax=None
):
    """Score de riesgo de un vehículo a lo largo de su odómetro, con su umbral.

    Es la vista de demo: un caso concreto, su alerta y cuánto anticipó.
    """
    data = predictions[predictions["vehicle_id"] == vehicle_id].sort_values("cut_odo")
    if data.empty:
        raise ValueError(f"No hay predicciones para el vehículo {vehicle_id}")
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    ax.plot(data["cut_odo"], data["score"], color=FORD_BLUE, lw=1.6, marker="o", ms=3)
    if threshold is not None:
        ax.axhline(threshold, color=ALERT_RED, ls="--", lw=1, label=f"umbral {threshold:.3f}")
        ax.legend(loc="upper left", fontsize=8)
    if int(data["event_observed"].max()) == 1 and data["time_to_event_km"].notna().any():
        last = data.iloc[-1]
        event_odo = float(last["cut_odo"] + last["time_to_event_km"])
        ax.axvline(event_odo, color="black", lw=1.2)
        ax.annotate("evento", xy=(event_odo, ax.get_ylim()[1]), fontsize=8, ha="right", va="top")
    ax.set_xlabel("Odómetro [km]")
    ax.set_ylabel("Score de riesgo")
    ax.set_title(f"Vehículo {vehicle_id}")
    ax.grid(alpha=0.25)
    return ax.figure


def plot_feature_profile(panel: pd.DataFrame, vehicle_id: str, features: list[str], *, ax=None):
    """Perfil de uso de un vehículo: cómo evolucionan sus features de ventana."""
    data = panel[panel["vehicle_id"] == vehicle_id].sort_values("cut_odo")
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    for feature in features:
        if feature not in data.columns:
            continue
        values = data[feature]
        spread = values.std()
        normalized = (values - values.mean()) / spread if spread and spread > 0 else values * 0
        ax.plot(data["cut_odo"], normalized, lw=1.4, label=feature.replace("feat_", ""))
    ax.set_xlabel("Odómetro [km]")
    ax.set_ylabel("Feature (z-score dentro del vehículo)")
    ax.set_title(f"Perfil de uso · {vehicle_id}")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.25)
    return ax.figure
