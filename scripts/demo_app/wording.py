"""Los textos de la demo que dependen del bundle: el modelo, la lectura de la perilla y el marco del "por qué".

Sin Streamlit, para poder chequearlos (`scripts/check_setup.py`). Nada de acá nombra un modelo a mano: el nombre y
la familia salen de la meta del bundle (`model`), y los números, de la capa de decisión que viaja en él.
"""

from __future__ import annotations

from typing import Any

from src.agents.formatting import number

#: El "por qué" de la demo, dicho en la app. Es una comparación con la flota sana, no lo que usó el modelo.
WHY_INTRO = "Comparado con autos sanos de su mercado, este auto:"
WHY_NONE = ("**Ningún hábito que el conductor pueda cambiar se aparta de los autos sanos de su mercado.** "
            "Lo revisa el taller.")
WHY_CAPTION = ("Es una comparación con la flota, no lo que usó el modelo: el modelo no dice por qué alerta. «Sanos de su "
               "mercado» es la mediana de los autos sanos de la muestra de desarrollo del mismo mercado; el auto se "
               "mide en las revisiones de la alerta.")
WHY_CHART_TITLE = "Comparado con los autos sanos de su mercado"
#: En dos renglones: en una sola línea no entra en el ancho de un celular.
WHY_CHART_AXIS = ("Autos sanos del mercado que supera,", "del lado que perjudica al filtro (%)")
WHY_CHART_CAPTION = ("Cada barra va de la mediana de los sanos (50%) a la parte de ellos que el auto supera hacia el lado "
                     "que la física del filtro señala como riesgoso. Un hábito del lado de los sanos no se dibuja. Se "
                     "nombran los que superan al {share} de los sanos (línea punteada), hasta {max_factors}: los que más "
                     "se apartan.")
HABITS_CAPTION = "Comparado con autos sanos de su mercado: "


def model_name(meta: dict[str, Any]) -> str:
    return str(meta["model"]["name"])


def budget_label(budget_per_1000: float) -> str:
    """Un punto de operación como lo muestra la perilla: 50 falsas alarmas cada 1.000 sanos es `5%`."""
    return f"{float(budget_per_1000) / 10:g}%"


def operating_line(meta: dict[str, Any]) -> str:
    """Lo que la perilla cambia, dicho en el punto elegido: cuánto se anticipa (el número oficial, promedio de las
    repeticiones) y cuántos sanos reciben una alerta de más (el umbral exacto deja a lo sumo esa parte)."""
    point = _curve(meta).get(round(float(meta["budget_per_1000"])))
    if point is None:
        return ""
    return (f"Anticipa el {_pct(point['detection'])} de las fallas, medido en desarrollo. Hasta el "
            f"{budget_label(meta['budget_per_1000'])} de los autos sanos recibe una alerta de más.")


def model_note(meta: dict[str, Any], llm_model: str, llm_mode: str) -> str:
    m = meta["model"]
    return (f"Modelo: {m['name']} ({m['family']}) · umbral al {meta['budget_per_1000'] / 10:g}% de falsas alarmas · "
            f"agentes: {llm_model} ({llm_mode})")


def _pct(x: float) -> str:
    return f"{number(100 * float(x), 1)}%"


def _curve(meta: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {round(float(p["budget_per_1000"])): p for p in meta["official"]["curve"]}
