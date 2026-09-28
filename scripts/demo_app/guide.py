"""Los pasos de «Cómo usar»: qué parte de la app resalta cada uno y qué dice (tour.js los recorre).

`targets` son selectores CSS, en orden de preferencia: gana el primero visible, así el mismo paso apunta a la
barra lateral en una notebook y al dock en un celular. Un paso sin `targets` es una tarjeta centrada, y uno cuyo
blanco no está en la página (una semana sin alertas no tiene tarjeta) se saltea. Todo número sale del bundle.
"""

from __future__ import annotations

from src.agents.formatting import long_date

from scripts.demo_app.common import bundle, cfg
from scripts.demo_app.wording import model_name


def _step(title: str, body: str, *targets: str) -> dict:
    return {"title": title, "body": body, "targets": list(targets) or None}


def tour_steps(page: str) -> list[dict]:
    """`page`: `bandeja`, `vehiculo` o `resultados`."""
    b = bundle()
    meta = b.meta
    pct = f"{meta['budget_per_1000'] / 10:g}%"
    name = model_name(meta)
    start, end = long_date(meta["replay"]["start"]), long_date(meta["replay"]["end"])

    intro = _step(
        f"Así se usa la demo de {cfg()['brand']['team']}",
        f"Estás en el lugar de quien gestiona la posventa. {name} revisa cada auto cada 500 km y, cuando el riesgo del "
        "filtro de partículas se sostiene, lo alerta. Los agentes convierten cada alerta en una acción lista para "
        "aprobar. Este recorrido te muestra cada parte.")
    calendar = _step(
        "El calendario de la flota",
        f"La demo reproduce la flota de desarrollo semana a semana, del {start} al {end}: "
        f"{meta['replay']['description']}. Cada columna es una semana y su barra, las alertas que llegaron. "
        "<b>Elegí una columna para abrir esa semana.</b>",
        ".st-key-weekbar")
    moving = _step(
        "Moverte por las semanas",
        "Las flechas cambian de semana. La mayoría de las semanas no trae alertas: <b>«Próxima»</b> salta directo a "
        "la siguiente que sí.",
        ".st-key-weekbar .tl-ctrl", ".st-key-dock_week")
    pages = _step(
        "Tres pantallas",
        "<b>Bandeja</b>: lo que hay que decidir esta semana. <b>Vehículo</b>: el riesgo de cada auto y por qué "
        "alertó. <b>Qué pasó después</b>: la temporada entera contra las fallas registradas.",
        ".st-key-side_nav", ".st-key-dock_nav")
    done = _step(
        "Listo",
        "Podés volver a este recorrido cuando quieras desde <b>«Cómo usar»</b>.",
        ".st-key-tour")

    by_page = {
        "bandeja": [
            _step("La semana en números",
                  f"Cuántos autos revisó {name} hasta hoy, las alertas nuevas y los escalamientos al concesionario de "
                  "esta semana, y las alertas acumuladas en la temporada.",
                  ".st-key-week_metrics"),
            _step("El resumen del agente",
                  "El agente de triage ordena la semana y la resume. <b>«Verificado»</b> quiere decir que el texto "
                  "pasó un verificador en código: solo cita números de los hechos y no afirma causas. La acción no "
                  "la elige el agente: la decide una política fija.",
                  ".st-key-summary"),
            _step("Una tarjeta por alerta",
                  "Qué pasó (alerta nueva o escalamiento), qué acción corresponde y los textos listos: el aviso al "
                  "conductor, el resumen para el taller y los hechos que recibió el agente.",
                  '[class*="st-key-alertcard_0_"]'),
            _step("Vos decidís",
                  "Aprobá y enviá, o descartá. <b>«Ver ficha»</b> abre la historia del auto: su riesgo en cada "
                  "revisión y por qué alertó.",
                  ".st-key-cardactions_0"),
            _step("Una semana sin alertas",
                  f"{name} siguió revisando la flota, pero ningún auto confirmó una alerta. Saltá a la próxima semana "
                  "con alertas para ver una tarjeta.",
                  ".st-key-empty_week"),
        ],
        "vehiculo": [
            _step("Elegí un auto",
                  "Primero aparecen los que ya alertaron; después, el resto de la flota revisada hasta esta semana.",
                  ".st-key-vehicle_pick"),
            _step("La ficha",
                  "El mercado, cuántas revisiones lleva, su ritmo de uso y si tiene una alerta activa.",
                  ".st-key-vehicle_metrics"),
            _step("El riesgo en el tiempo",
                  f"Cada punto es una revisión, un corte cada 500 km. La línea punteada es el umbral al {pct} de "
                  f"falsas alarmas: la alerta se confirma cuando {b.k} revisiones seguidas quedan por encima.",
                  ".st-key-risk_chart"),
            _step("Por qué",
                  "En qué hábitos de uso, si los hay, este auto se aparta de los autos sanos de su mercado. Es una "
                  "comparación con la flota, no lo que usó el modelo.",
                  ".st-key-why"),
            _step("Lo que se le comunicó",
                  "Los textos de cada alerta, tal como los recibieron el conductor y el taller.",
                  ".st-key-sent"),
        ],
        "resultados": [
            _step("La temporada completa",
                  "Cuántas fallas se anticiparon y con cuánto margen, cuántas alertas fueron de más y cuántas fallas "
                  f"{name} no vio.",
                  ".st-key-season_metrics"),
            _step("De la alerta a la falla",
                  "Una fila por auto alertado: de la alerta a la falla registrada, o punteada si el auto no falló "
                  "en el período del replay.",
                  ".st-key-season_timeline"),
            _step("Los números medidos",
                  f"Los oficiales: el promedio de las {meta['official']['n_repeats']} repeticiones de la validación, "
                  "contra un azar que conserva el largo de cada historial. El replay muestra una sola repetición.",
                  ".st-key-official"),
        ],
    }
    # «Qué pasó después» muestra la temporada entera: ahí no hay calendario.
    week_steps = [] if page == "resultados" else [calendar, moving]
    return [intro, *week_steps, *by_page[page], pages, done]
