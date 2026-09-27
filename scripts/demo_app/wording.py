"""Los textos de la demo que dependen del bundle: el modelo, los números del pitch y el marco del "por qué".

Sin Streamlit, para poder chequearlos (`scripts/check_setup.py`). Nada de acá nombra un modelo a mano: el nombre y
la familia salen de la meta del bundle (`model`), y los números, de la capa de decisión que viaja en él.
"""

from __future__ import annotations

from typing import Any

from src.agents.formatting import format_value, number

#: El "por qué" de la demo, dicho en la app. Es una comparación con la flota sana, no lo que usó el modelo.
WHY_INTRO = "Comparado con autos sanos de su mercado, este auto:"
WHY_NONE = ("**Ningún hábito que el conductor pueda cambiar se aparta de los autos sanos de su mercado.** "
            "Lo revisa el taller.")
WHY_CAPTION = ("Es una comparación con la flota, no lo que usó el modelo: el modelo no dice por qué alerta. «Sanos de su "
               "mercado» es la mediana de los autos sanos de la muestra de desarrollo del mismo mercado; el auto se "
               "mide en las revisiones de la alerta.")
WHY_CHART_TITLE = "Comparado con los autos sanos de su mercado"
WHY_CHART_AXIS = "Autos sanos del mercado que supera, del lado que perjudica al filtro (%)"
WHY_CHART_CAPTION = ("Cada barra va de la mediana de los sanos (50%) a la parte de ellos que el auto supera hacia el lado "
                     "que la física del filtro señala como riesgoso. Un hábito del lado de los sanos no se dibuja. Se "
                     "nombra si supera al {share} de los sanos (línea punteada).")
HABITS_CAPTION = "Comparado con autos sanos de su mercado: "


def model_name(meta: dict[str, Any]) -> str:
    return str(meta["model"]["name"])


def model_note(meta: dict[str, Any], llm_model: str, llm_mode: str) -> str:
    m = meta["model"]
    return (f"Modelo: {m['name']} ({m['family']}) · umbral al {meta['budget_per_1000'] / 10:g}% de falsas alarmas · "
            f"agentes: {llm_model} ({llm_mode})")


def _pct(x: float) -> str:
    return f"{number(100 * float(x), 1)}%"


def _curve(meta: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(p["budget_per_1000"]): p for p in meta["official"]["curve"]}


def detection_line(meta: dict[str, Any], budgets: tuple[int, ...] = (50, 100)) -> str:
    """La detección oficial a cada presupuesto, dentro de muestra y con el umbral fijado fuera de muestra."""
    curve = _curve(meta)
    held = {round(float(h["alpha"]) * 1000): h for h in meta["official"]["holdout"] if h["method"] == "empirical"}
    parts = []
    for i, b in enumerate(x for x in budgets if x in curve):
        out = held.get(b)
        oos = "" if out is None else (f" ({_pct(out['detection'])} con el umbral fijado fuera de muestra)" if i == 0
                                      else f" ({_pct(out['detection'])})")
        lead = f"Al {b / 10:g}% de falsas alarmas anticipa {_pct(curve[b]['detection'])} de los autos que fallan" if i == 0 \
            else f"al {b / 10:g}%, {_pct(curve[b]['detection'])}"
        parts.append(lead + oos)
    return "; ".join(parts) + "." if parts else ""


def chance_line(meta: dict[str, Any]) -> str:
    """Contra el azar que conserva el largo de cada historial (regla 6), en el punto de la demo y alrededor."""
    name, curve, op = model_name(meta), _curve(meta), int(meta["budget_per_1000"])
    point = curve.get(op)
    if point is None:
        return ""
    if point["detection"] <= point["null"]:
        above = [b for b in sorted(curve) if b > op and curve[b]["detection"] > curve[b]["null"]]
        tail = (f" Al {above[0] / 10:g}% detecta {_pct(curve[above[0]]['detection'])} contra "
                f"{_pct(curve[above[0]]['null'])} del azar.") if above else ""
        return (f"Al {op / 10:g}%, el punto de esta demo, {name} detecta {_pct(point['detection'])} y el azar con el mismo "
                f"historial {_pct(point['null'])}: en ese punto no le gana al azar.{tail}")
    below = [b for b in sorted(curve) if b < op and curve[b]["detection"] <= curve[b]["null"]]
    return f"Por debajo del {op / 10:g}%, {name} no le gana al azar." if below else ""


def promises(meta: dict[str, Any]) -> str:
    """«Qué no promete esta demo», con todo número sacado del bundle."""
    gap, end = format_value(float(meta["gap_km"]), "int"), format_value(float(meta["gap_km"]) + float(meta["horizon_km"]), "int")
    share = format_value(float(meta["explanation"]["min_healthy_share"]), "pct")
    return (
        f"- **Detecta una fracción, no todas las fallas.** {detection_line(meta)} {chance_line(meta)}\n"
        "- **El porqué es una comparación con la flota, no lo que usó el modelo.** «Comparado con autos sanos de tu "
        "mercado, tu auto…», nunca «falla porque…» ni «el sistema lo marcó por…». Se nombra un hábito solo si el auto "
        f"supera al {share} de los sanos de su mercado hacia el lado que perjudica al filtro. Los agentes no pueden "
        "escribir otra cosa: un verificador en código lo controla.\n"
        f"- **El horizonte está en km** ({gap} a {end} km desde la revisión); las semanas son una conversión al ritmo "
        "del auto.\n"
        "- **La muestra está enriquecida en fallas.** En una flota real la prevalencia es mucho menor: los conteos "
        "absolutos de la bandeja no se trasladan tal cual.\n"
        "- **La referencia de los sanos es descriptiva:** la mediana de los autos sanos de desarrollo del mismo mercado "
        "en toda la ventana, no una referencia de producción."
    )
