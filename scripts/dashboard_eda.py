#!/usr/bin/env python
"""Dashboard del EDA de los datos crudos, **solo sobre dev**.

    streamlit run scripts/dashboard_eda.py

Es la versión navegable de `notebooks/eda-exhaustivo-dev.ipynb`: las mismas
secciones, los mismos números, filtrables. No calcula nada propio sobre los CSV
crudos —lee el cache que deja `scripts/build_eda_cache.py`— para que el filtro a
dev se decida **en un solo lugar** y el dashboard levante en segundos en vez de
volver a leer 1,2 GB.

No es `scripts/dashboard.py`: ese consume predicciones de modelo (Track C, F4+) y
todavía corre contra el panel dummy. Este mira el dato crudo, que es lo único real
que hay antes de F2.

**Alcance:** los 290 vehículos de dev del holdout congelado
(`data/processed/test_split.json`). Los 74 de test no están en el cache, y tampoco
los 717 que quedaron **fuera del universo del estudio** —positivos cuyo evento no
se puede ubicar en el tiempo, y mercados donde ningún evento es observable, ver
`src/data/usable.py`—. El banner de arriba lo dice en cada carga.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import repo_root  # noqa: E402
from scripts.build_eda_cache import (  # noqa: E402
    COHORT_LABELS,
    DEFAULT_CONFIG,
    MISSING_FAMILIES,
    PDF_COLUMNS,
    THEME,
    cache_dir,
    cache_exists,
    dictionary_comparison,
    feature_feasibility,
    load_eda_cache,
    raw_columns,
    vehicle_matrix,
)

CONFIG = DEFAULT_CONFIG
CATEGORICAS = ["static_Engine", "static_ModelSeries", "static_SalesCountry_cd"]


# ======================================================================================
# Tema
# ======================================================================================
def modo() -> str:
    """Claro u oscuro, según el tema activo de Streamlit."""
    try:
        return "dark" if st.context.theme.type == "dark" else "light"
    except Exception:
        return "light"


def paleta() -> dict[str, Any]:
    return THEME[modo()]


def estilar(chart, *, altura: int | None = 300):
    """Cromo recesivo y tipografía del sistema. Una sola vez, para todos los gráficos.

    `altura=None` para los compuestos (facet, concat): ahí la altura va en cada
    sub-gráfico y `config` solo es válido en el objeto de más afuera.
    """
    tinta = paleta()["ink"]
    if altura is not None:
        chart = chart.properties(height=altura)
    return (
        chart.configure_view(strokeWidth=0)
        .configure_axis(
            grid=True, gridColor=tinta["grid"], gridWidth=0.7, domainColor=tinta["muted"],
            tickColor=tinta["muted"], labelColor=tinta["muted"], titleColor=tinta["secondary"],
            labelFontSize=11, titleFontSize=11, titleFontWeight="normal",
        )
        .configure_legend(labelColor=tinta["secondary"], titleColor=tinta["secondary"],
                          labelFontSize=11, titleFontSize=11, symbolStrokeWidth=0)
        .configure_title(color=tinta["primary"], fontSize=13, fontWeight=600, anchor="start")
    )


def escala_cohorte() -> alt.Scale:
    colores = paleta()["cohort"]
    return alt.Scale(domain=[COHORT_LABELS[0], COHORT_LABELS[1]], range=[colores[0], colores[1]])


def color_cohorte(leyenda: str | None = "cohorte") -> alt.Color:
    return alt.Color("cohorte:N", scale=escala_cohorte(),
                     legend=alt.Legend(title=leyenda, orient="top") if leyenda else None)


def mostrar(chart: alt.Chart, *, altura: int = 300) -> None:
    st.altair_chart(estilar(chart, altura=altura), width="stretch", theme=None)


def con_etiqueta(frame: pd.DataFrame) -> pd.DataFrame:
    """Agrega la columna legible `cohorte` a partir de `event_observed`."""
    return frame.assign(cohorte=frame["event_observed"].map(COHORT_LABELS))


# ======================================================================================
# Datos
# ======================================================================================
@st.cache_data(show_spinner="Cargando el cache del EDA…")
def cargar() -> dict[str, Any]:
    cache = load_eda_cache(CONFIG)
    cache["veh"] = vehicle_matrix(cache)
    return cache


@st.cache_data(show_spinner=False)
def cargar_diccionario() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, list[str]]]:
    return dictionary_comparison(CONFIG), feature_feasibility(CONFIG), raw_columns(CONFIG)


def construir_cache() -> None:
    """Corre el builder desde la app: la primera vez no hace falta salir a la terminal."""
    with st.status("Construyendo el cache dev-only (una pasada por 1,2 GB)…", expanded=True) as estado:
        st.write("Canonizando los 13 clones, filtrando a dev y agregando por vehículo.")
        proceso = subprocess.run(
            [sys.executable, str(repo_root() / "scripts" / "build_eda_cache.py"),
             "--config", CONFIG, "--force"],
            cwd=str(repo_root()), capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        st.code((proceso.stdout or "")[-3000:] or (proceso.stderr or "")[-3000:])
        if proceso.returncode != 0:
            estado.update(label="Falló la construcción del cache", state="error")
            st.stop()
        estado.update(label="Cache listo", state="complete")
    st.cache_data.clear()
    st.rerun()


# ======================================================================================
# Secciones
# ======================================================================================
def kpis(veh: pd.DataFrame, meta: dict[str, Any]) -> None:
    columnas = st.columns(6)
    columnas[0].metric("Vehículos (dev)", f"{len(veh):,}".replace(",", "."),
                       help=f"{meta['n_test_vehicles_excluidos']} de test y "
                            f"{meta.get('n_fuera_del_universo', 0)} fuera del universo "
                            "quedan afuera del cache.")
    columnas[1].metric("Con evento", f"{int(veh['event_observed'].sum())}",
                       f"tasa {veh['event_observed'].mean():.3f}", delta_color="off")
    columnas[2].metric("Viajes", f"{meta['trips']['n_dev_deduplicadas']:,}".replace(",", "."),
                       help="Ya deduplicados: clones y filas exactamente repetidas.")
    columnas[3].metric("Señales", f"{meta['signals']['n_dev_deduplicadas']:,}".replace(",", "."))
    columnas[4].metric("Odómetro mediano", f"{veh['trip_odo_max'].median():,.0f} km".replace(",", "."))
    columnas[5].metric("Historia mediana", f"{veh['span_days'].median():,.0f} días".replace(",", "."))


def seccion_estaticas(veh: pd.DataFrame) -> None:
    st.subheader("Estáticas")
    st.caption(
        "`Engine` está **excluida del modelo** (`features.static_excluded` de "
        "`configs/data/panel_v1.yaml`): dentro de dev, `ENG_3` es el 20% de los sanos y el 0% de los que "
        "tienen evento. Se muestra para reportar el sesgo, no para usarla. **Pero sacarla no "
        "alcanza:** el cruce con `ModelSeries` es casi diagonal, así que `MODEL_3` (0 eventos de 24) "
        "y `MODEL_4` (2 de 91) llevan la misma información con otro nombre."
    )
    columna = st.segmented_control(
        "Variable", CATEGORICAS, default=CATEGORICAS[0],
        format_func=lambda c: c.replace("static_", ""), key="estatica",
    ) or CATEGORICAS[0]

    tabla = pd.crosstab(veh[columna], veh["event_observed"])
    largo = (tabla / tabla.sum()).reset_index().melt(id_vars=columna, var_name="event_observed",
                                                     value_name="frac")
    largo = con_etiqueta(largo).merge(
        tabla.reset_index().melt(id_vars=columna, var_name="event_observed", value_name="n"),
        on=[columna, "event_observed"],
    )
    izquierda, derecha = st.columns([3, 2])
    with izquierda:
        barras = (
            alt.Chart(largo)
            .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, stroke="white", strokeWidth=2)
            .encode(
                x=alt.X(f"{columna}:N", title=columna.replace("static_", ""), axis=alt.Axis(labelAngle=0)),
                y=alt.Y("frac:Q", title="fracción dentro de la cohorte", axis=alt.Axis(format=".0%")),
                xOffset="cohorte:N",
                color=color_cohorte(),
                tooltip=[alt.Tooltip(f"{columna}:N", title="valor"), "cohorte:N",
                         alt.Tooltip("n:Q", title="vehículos"),
                         alt.Tooltip("frac:Q", title="fracción", format=".1%")],
            )
            .properties(title=f"{columna.replace('static_', '')} por cohorte")
        )
        mostrar(barras, altura=320)
    with derecha:
        resumen = tabla.rename(columns=COHORT_LABELS)
        resumen["tasa de eventos"] = (resumen["con evento"] /
                                      (resumen["con evento"] + resumen["sin evento"])).round(3)
        st.dataframe(resumen, width="stretch")
        if columna == "static_Engine" and int(tabla.get(1, pd.Series(dtype=int)).get("ENG_3", 0)) == 0:
            st.error(
                "**ENG_3: 0 eventos en "
                f"{int(tabla.loc['ENG_3', 0])} vehículos.** Es cómo se muestrearon los datos, "
                "no física del motor. Por eso `Engine` no entra al modelo.",
                icon=":material/warning:",
            )

    st.markdown("**Estáticas numéricas** — las dos que **sí** están en el set base.")
    numericas = ["static_daysUntilSale", "static_ProductionDay"]
    largo = con_etiqueta(veh[numericas + ["event_observed"]]).melt(
        id_vars=["event_observed", "cohorte"], var_name="variable", value_name="valor")
    densidad = (
        alt.Chart(largo)
        .transform_density("valor", groupby=["variable", "cohorte"], as_=["valor", "densidad"], steps=80)
        .mark_area(opacity=0.35, line={"strokeWidth": 2})
        .encode(
            x=alt.X("valor:Q", title="días"),
            y=alt.Y("densidad:Q", title="densidad", stack=None),
            color=color_cohorte(),
            tooltip=[alt.Tooltip("valor:Q", format=".0f"), "cohorte:N"],
        )
        .facet(column=alt.Column("variable:N", title=None,
                                 header=alt.Header(labelColor=paleta()["ink"]["primary"],
                                                   labelFontWeight=600)))
        .resolve_scale(x="independent", y="independent")
    )
    st.altair_chart(estilar(densidad, altura=None), width="stretch", theme=None)
    st.warning(
        "**`ProductionDay` es la variable más correlacionada de todo el EDA** (ρ de Spearman −0,435; "
        "`daysUntilSale` queda en −0,261). No es física: la ventana de observación termina el mismo "
        "día para toda la flota, así que un vehículo producido tarde tuvo menos kilómetros para "
        "llegar a fallar. La tasa de eventos por quintil va de 0,459 a **0,000**, y contra "
        "`span_days` da ρ = −0,933: son la misma variable. Las dos están en el set base de "
        "`panel_v1.yaml` y **`ProductionDay` habría que sacarla**, con el mismo criterio que `Engine`.",
        icon=":material/priority_high:",
    )


def seccion_etiqueta(cache: dict[str, Any], veh: pd.DataFrame) -> None:
    from lifelines import KaplanMeierFitter

    meta = cache["meta"]
    eventos = veh[veh["event_observed"].eq(1)].copy()

    st.subheader("Etiqueta, anclaje y censura")
    columnas = st.columns(4)
    columnas[0].metric("Eventos", f"{len(eventos)}")
    columnas[1].metric("Censurados", f"{len(veh) - len(eventos)}")
    columnas[2].metric("IQR del anclaje", f"{meta['anchor_offset_iqr_days']:.1f} d",
                       help="`primer_viaje − ProductionDay`. F1 midió 0,0 sobre los 1094.")
    columnas[3].metric("Odómetro del evento", f"{meta['event_odo_median_km']:,.0f} km".replace(",", "."),
                       f"mediana · {eventos['event_frac_trips_before'].median():.0%} del historial antes",
                       delta_color="off")

    universo = meta.get("universe") or {}
    st.success(
        f"**El evento se puede ubicar sobre el eje de km para los {len(eventos)} positivos de dev.** "
        f"Mediana de {meta['event_odo_median_km']:,.0f} km de odómetro y "
        f"{eventos['event_frac_trips_before'].median():.0%} del historial de viajes por delante. "
        "Eso es lo que hace que haya ventana W que agregar y gap G que blanquear.\n\n"
        f"Se paga con volumen: de {universo.get('events_input', '?')} eventos del dataset quedan "
        f"{universo.get('events_kept', '?')}, y de {universo.get('n_input', '?')} vehículos, "
        f"{universo.get('n_kept', '?')}. El criterio está abajo.".replace(",", "."),
        icon=":material/check_circle:",
    )

    with st.expander("Por qué el universo son 364 vehículos y no 1081", expanded=False):
        st.markdown(
            "`IdentificationDate` trae **dos convenciones de registro mezcladas**. En 284 de los 365 "
            "positivos vale *exactamente* lo mismo que `daysUntilSale`, y ahí el evento queda pegado "
            "al día de la venta, con el odómetro en ~13 km y sin historial por delante. Que sea "
            "administrativo y no físico lo cierra un conteo: **`IdentificationDate < daysUntilSale` "
            "no pasa nunca, 0 de 365**.\n\n"
            "Y la convención es **del mercado, no del vehículo**: por eso no alcanza con filtrar los "
            "positivos. Si se los tira y se dejan sus vehículos sanos, los mercados sin eventos "
            "observables aportan 432 negativos y 1 positivo — eso es selección sobre el resultado, y "
            "deja a los negativos viniendo de otra población que los positivos."
        )
        mercados = pd.DataFrame(meta.get("market_usability") or [])
        if not mercados.empty:
            mercados = mercados.set_index("static_SalesCountry_cd")
            mercados.columns = ["vehículos", "sanos", "eventos", "fecha real",
                                "fecha = venta", "frac. usable"]
            st.dataframe(
                mercados.style.format({"frac. usable": "{:.1%}"})
                .background_gradient(subset=["frac. usable"], cmap="Blues"),
                width="stretch",
            )
        st.caption(
            "Este cuadro cubre los 1081 a propósito: es el **registro** de una decisión ya congelada "
            "sobre la calidad del dato (bloque `universe` de `test_split.json`), no una medición que "
            "se esté haciendo ahora. Define qué mercado entra, no qué modelo gana. "
            "Detalle: `docs/reproducibilidad.md`."
        )

    izquierda, derecha = st.columns(2)
    with izquierda:
        puntos = (
            alt.Chart(eventos)
            .mark_circle(size=55, opacity=0.6, stroke="white", strokeWidth=0.5)
            .encode(
                x=alt.X("static_daysUntilSale:Q", title="daysUntilSale [días desde producción]"),
                y=alt.Y("event_day_since_production:Q", title="IdentificationDate [días desde producción]"),
                color=alt.value(paleta()["cohort"][1]),
                tooltip=["vehicle_id:N", "static_daysUntilSale:Q", "event_day_since_production:Q",
                         alt.Tooltip("event_odo_km:Q", title="odo del evento [km]", format=".0f")],
            )
            .properties(title="La etiqueta contra la fecha de venta")
        )
        diagonal = (
            alt.Chart(pd.DataFrame({"x": [0, float(eventos["static_daysUntilSale"].max())]}))
            .mark_line(strokeDash=[5, 4], color=paleta()["ink"]["muted"], strokeWidth=1.5)
            .encode(x="x:Q", y="x:Q")
        )
        mostrar(puntos + diagonal, altura=330)
    with derecha:
        perfil = cache["odo_profile_dev"]
        perfil = con_etiqueta(perfil[perfil["day_since_production"].le(500)])
        linea = (
            alt.Chart(perfil)
            .mark_line(strokeWidth=2.5)
            .encode(
                x=alt.X("day_since_production:Q", title="días desde producción"),
                y=alt.Y("median:Q", title="odómetro mediano [km]"),
                color=color_cohorte(),
                tooltip=[alt.Tooltip("day_since_production:Q", title="día"), "cohorte:N",
                         alt.Tooltip("median:Q", title="odómetro", format=".0f")],
            )
            .properties(title="La flota no se mueve hasta que se vende")
        )
        banda = (
            alt.Chart(perfil).mark_area(opacity=0.15)
            .encode(x="day_since_production:Q", y="p25:Q", y2="p75:Q", color=color_cohorte(None))
        )
        venta = (
            alt.Chart(pd.DataFrame({"d": [float(veh["static_daysUntilSale"].median())]}))
            .mark_rule(strokeDash=[5, 4], color=paleta()["ink"]["secondary"], strokeWidth=1.5)
            .encode(x="d:Q", tooltip=[alt.Tooltip("d:Q", title="venta mediana [días]")])
        )
        mostrar(banda + linea + venta, altura=330)

    st.markdown("**Odómetro del evento según cómo se lea `IdentificationDate`**")
    hipotesis = st.radio(
        "Lectura del eje de días", ["A · días desde producción (anexo 7.3)", "B · días desde la venta"],
        horizontal=True, key="hipotesis",
        help="La B no está respaldada por el diccionario: se muestra para comparar, no como decisión.",
    )
    prefijo = "event" if hipotesis.startswith("A") else "event_alt"
    # El corte por `ident_equals_sale` desapareció con el universo: en dev no queda ni un
    # vehículo del grupo "Ident == venta". Lo que sí discrimina ahora es cuánto historial
    # deja el evento por delante, que es la pregunta que decide si la fila se puede etiquetar.
    frac_actual = eventos[f"{prefijo}_frac_trips_before"]
    datos = eventos.assign(
        grupo=np.where(frac_actual < 0.05, "<5% del historial antes", "historial suficiente"),
        odo=eventos[f"{prefijo}_odo_km"],
        frac=frac_actual,
    )
    izquierda, derecha = st.columns([3, 2])
    with izquierda:
        histograma = (
            alt.Chart(datos)
            .mark_bar(opacity=0.75, stroke="white", strokeWidth=1)
            .encode(
                x=alt.X("odo:Q", bin=alt.Bin(maxbins=45), title="odómetro del evento [km]",
                        scale=alt.Scale(type="symlog")),
                y=alt.Y("count():Q", title="vehículos"),
                color=alt.Color("grupo:N",
                                scale=alt.Scale(domain=["<5% del historial antes",
                                                        "historial suficiente"],
                                                range=[paleta()["cohort"][1], paleta()["cohort"][0]]),
                                legend=alt.Legend(title=None, orient="top")),
                tooltip=["grupo:N", alt.Tooltip("count():Q", title="vehículos")],
            )
            .properties(title=f"Odómetro del evento · hipótesis {hipotesis[0]}")
        )
        mostrar(histograma, altura=300)
    with derecha:
        resumen = datos.groupby("grupo").agg(
            vehículos=("vehicle_id", "size"),
            odo_mediano_km=("odo", "median"),
            frac_historial_antes=("frac", "median"),
        ).round(3)
        st.dataframe(resumen, width="stretch")
        st.caption(
            "`frac_historial_antes` es la fracción de los viajes del vehículo anteriores al evento. "
            "Sin historial por delante no hay ventana W que agregar ni gap G que blanquear. "
            "El criterio del universo es necesario pero no suficiente: **2 de los 60** pasan con un "
            "delta de 1 y 2 días y caen igual en 8 y 5 km de odómetro. Los descarta "
            "`sampling.min_trips_in_window` al armar el panel."
        )

    st.markdown("**Supervivencia sobre el eje de odómetro (Kaplan-Meier)**")
    corte = st.selectbox("Cortar por", ["(toda la flota)"] + CATEGORICAS,
                         format_func=lambda c: c.replace("static_", ""), key="km_corte")
    columna_duracion = "duration_km" if prefijo == "event" else "duration_km_alt"
    curvas = []
    grupos = ([("flota dev", veh)] if corte == "(toda la flota)"
              else [(str(k), g) for k, g in veh.groupby(corte, observed=True) if len(g) >= 20])
    for nombre, grupo in grupos:
        kmf = KaplanMeierFitter().fit(grupo[columna_duracion].clip(lower=1), grupo["event_observed"])
        curva = kmf.survival_function_.reset_index()
        curva.columns = ["odo", "supervivencia"]
        curva["grupo"] = f"{nombre} (n={len(grupo)})"
        curvas.append(curva)
    curvas = pd.concat(curvas, ignore_index=True)
    rango = list(paleta()["categorical"][: curvas["grupo"].nunique()])
    km_chart = (
        alt.Chart(curvas)
        .mark_line(interpolate="step-after", strokeWidth=2.5)
        .encode(
            x=alt.X("odo:Q", title="odómetro [km]", scale=alt.Scale(type="symlog", constant=100)),
            y=alt.Y("supervivencia:Q", title="P(sin evento)", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("grupo:N", scale=alt.Scale(range=rango),
                            legend=alt.Legend(title=None, orient="top", columns=3)),
            tooltip=["grupo:N", alt.Tooltip("odo:Q", format=".0f"),
                     alt.Tooltip("supervivencia:Q", format=".3f")],
        )
        .properties(title="Kaplan-Meier · el lado censurado no depende del anclaje")
    )
    mostrar(km_chart, altura=340)
    if corte == "static_Engine":
        st.info("La curva plana de `ENG_3` es el sesgo de muestreo, no un motor mejor: "
                "con 0 eventos la curva no puede bajar.", icon=":material/info:")


def seccion_odometro(veh: pd.DataFrame) -> None:
    st.subheader("Calidad del eje de odómetro")
    nulos = (veh["signal_odo_null"] / veh["n_signals"]).rename("frac_nulo")
    datos = veh.assign(frac_nulo=nulos)

    columnas = st.columns(4)
    columnas[0].metric("Nulos de `OdometerValue`",
                       f"{veh['signal_odo_null'].sum() / veh['n_signals'].sum():.2%}",
                       help="Global sobre dev. El universo completo documenta 11,15%.")
    columnas[1].metric("Mediana por vehículo", f"{nulos.median():.2%}")
    columnas[2].metric("Vehículos con >50% nulo", f"{int((nulos > 0.5).sum())}")
    columnas[3].metric("Vehículos con retroceso", f"{int((veh['n_km_negative'] > 0).sum())}",
                       f"{int(veh['n_km_negative'].sum())} viajes", delta_color="off")

    st.info(
        "**El odómetro nulo de `signals` se fue con el recorte del universo.** Sobre los 1081 era "
        "el 11,15% de las filas, concentrado en cuatro vehículos con entre el 66% y el 97% de sus "
        "señales sin odómetro; esos cuatro estaban en los mercados descartados. Acá el nulo global "
        "es 0,009%, la mediana por vehículo 0,00% y **ninguno pasa del 1%**, así que descartar esas "
        "filas ya no tiene costo y la imputación por viaje deja de hacer falta. El que sí empeoró "
        "es `KilometerPerHour`: 35,4% de nulos contra 32,5% sobre los 1081.",
        icon=":material/info:",
    )

    izquierda, derecha = st.columns(2)
    with izquierda:
        peores = con_etiqueta(datos.nlargest(12, "frac_nulo"))
        barras = (
            alt.Chart(peores)
            .mark_bar(cornerRadiusEnd=4, stroke="white", strokeWidth=1.5)
            .encode(
                y=alt.Y("vehicle_id:N", sort="-x", title=None),
                x=alt.X("frac_nulo:Q", title="fracción de señales sin odómetro",
                        axis=alt.Axis(format=".0%")),
                color=color_cohorte(),
                tooltip=["vehicle_id:N", "cohorte:N",
                         alt.Tooltip("frac_nulo:Q", format=".2%"),
                         alt.Tooltip("n_signals:Q", title="señales", format=",.0f")],
            )
            .properties(title="Los 12 vehículos con más odómetro nulo")
        )
        mostrar(barras, altura=330)
    with derecha:
        desfase = con_etiqueta(
            veh.assign(desfase=(veh["signal_odo_max"] - veh["trip_odo_max"]).clip(-400, 200))
        )
        histograma = (
            alt.Chart(desfase)
            .mark_bar(opacity=0.75, stroke="white", strokeWidth=1)
            .encode(
                x=alt.X("desfase:Q", bin=alt.Bin(maxbins=45),
                        title="odo_max(`signals`) − odo_max(`trips`) [km, recortado]"),
                y=alt.Y("count():Q", title="vehículos", stack=None),
                color=color_cohorte(),
                tooltip=["cohorte:N", alt.Tooltip("count():Q", title="vehículos")],
            )
            .properties(title="Desfase entre los dos odómetros")
        )
        mostrar(histograma, altura=330)


def seccion_uso(cache: dict[str, Any], veh: pd.DataFrame) -> None:
    st.subheader("Uso y contexto")
    st.caption(
        "Arriba, las distribuciones **a nivel viaje** (histograma exacto sobre los 1,9M de viajes "
        "de dev). Abajo, los agregados **por vehículo**, que es el grano que va a tener el panel."
    )
    hist = cache["trip_hist_dev"]
    etiquetas = {
        "trip_km": "Distancia por viaje [km]",
        "trip_duration_min": "Duración del viaje [min]",
        "KilometerPerHour": "Velocidad media [km/h]",
        "EngineTemperatureAvg": "Temperatura media de motor [°C]",
        "EngineTemperatureMax": "Temperatura máxima de motor [°C]",
        "engine_temp_amplitude": "Amplitud térmica del viaje [°C]",
        "CoolantTemperatureEnd": "Refrigerante al final [°C]",
        "AirTemperatureAvg": "Temperatura ambiente media [°C]",
        "EngineOilLifePCStart": "Vida útil del aceite [%]",
        "AirRegenerationEnd": "AirRegenerationEnd [0–95]",
        "air_regen_delta": "Δ AirRegeneration en el viaje",
        "FuelLvlStartPc": "Nivel de combustible al inicio [%]",
    }
    metrica = st.selectbox("Métrica a nivel viaje", [m for m in etiquetas if m in set(hist["metric"])],
                           format_func=lambda m: etiquetas[m], key="metrica_viaje")
    datos = con_etiqueta(hist[hist["metric"].eq(metrica)]).assign(
        centro=lambda d: (d["bin_left"] + d["bin_right"]) / 2)
    area = (
        alt.Chart(datos)
        .mark_area(opacity=0.28, interpolate="step", line={"strokeWidth": 2})
        .encode(
            x=alt.X("centro:Q", title=etiquetas[metrica]),
            y=alt.Y("frac:Q", title="fracción de viajes", stack=None, axis=alt.Axis(format=".1%")),
            color=color_cohorte(),
            tooltip=[alt.Tooltip("centro:Q", title="valor", format=".2f"), "cohorte:N",
                     alt.Tooltip("frac:Q", title="fracción", format=".2%"),
                     alt.Tooltip("n:Q", title="viajes", format=",.0f")],
        )
        .properties(title=f"{etiquetas[metrica]} · histograma exacto sobre dev")
    )
    mostrar(area, altura=300)

    agregados = {
        "short_trip_frac": "Fracción de viajes < 5 km",
        "very_short_trip_frac": "Fracción de viajes < 2 km",
        "urban_trip_frac": "Fracción de viajes < 30 km/h",
        "below_regime_frac": "Sin llegar a régimen (< 70 °C)",
        "km_per_day": "km por día",
        "trips_per_day": "Viajes por día",
        "trip_km_median": "Distancia mediana por viaje [km]",
        "hours_between_trips_median": "Horas entre viajes",
        "engine_temp_avg_median": "Temperatura media de motor [°C]",
        "engine_temp_amplitude_mean": "Amplitud térmica media [°C]",
        "coolant_end_mean": "Refrigerante al final [°C]",
        "air_temp_avg_mean": "Temperatura ambiente media [°C]",
        "speed_mean": "Velocidad media [km/h]",
        "refuel_frac": "Fracción de viajes con recarga",
    }
    elegidas = st.multiselect(
        "Agregados por vehículo", list(agregados), format_func=lambda c: agregados[c],
        default=["short_trip_frac", "below_regime_frac", "km_per_day", "engine_temp_avg_median"],
        key="agregados_uso",
    )
    if elegidas:
        largo = con_etiqueta(veh[elegidas + ["event_observed", "vehicle_id"]]).melt(
            id_vars=["vehicle_id", "event_observed", "cohorte"], var_name="variable", value_name="valor")
        largo["variable"] = largo["variable"].map(agregados)
        cajas = (
            alt.Chart(largo)
            .mark_boxplot(size=34, outliers={"size": 8, "opacity": 0.25}, median={"color": "white"})
            .encode(
                x=alt.X("cohorte:N", title=None, axis=alt.Axis(labelAngle=0)),
                y=alt.Y("valor:Q", title=None),
                color=color_cohorte(),
                tooltip=["cohorte:N", alt.Tooltip("valor:Q", format=".3f")],
            )
            .properties(width=150, height=260)
            .facet(column=alt.Column("variable:N", title=None,
                                     header=alt.Header(labelColor=paleta()["ink"]["primary"],
                                                       labelFontWeight=600, labelLimit=220)))
            .resolve_scale(y="independent")
        )
        st.altair_chart(estilar(cajas, altura=None), width="stretch", theme=None)
        with st.expander("Tabla · mediana por cohorte"):
            tabla = veh.groupby("event_observed")[elegidas].median().T
            tabla.columns = [COHORT_LABELS[c] for c in tabla.columns]
            tabla["ratio"] = (tabla["con evento"] / tabla["sin evento"]).replace([np.inf, -np.inf], np.nan)
            st.dataframe(tabla.round(4), width="stretch")


def seccion_severidad(cache: dict[str, Any], veh: pd.DataFrame) -> None:
    st.subheader("Severidad y postratamiento")
    st.caption("Donde está la señal. Las tasas son **por vehículo**, no el agregado de la cohorte.")

    izquierda, derecha = st.columns([3, 2])
    with izquierda:
        conteos = cache["message_counts_dev"]
        conteos = con_etiqueta(
            conteos[conteos["scope"].eq("full")].merge(veh[["vehicle_id", "event_observed"]],
                                                       on="vehicle_id")
        )
        vistos = (conteos.groupby(["Message", "cohorte"])["vehicle_id"].nunique().reset_index(name="n"))
        total = veh.groupby("event_observed").size().rename(index=COHORT_LABELS)
        vistos["frac"] = vistos["n"] / vistos["cohorte"].map(total)
        barras = (
            alt.Chart(vistos)
            .mark_bar(cornerRadiusEnd=4, stroke="white", strokeWidth=1.5)
            .encode(
                y=alt.Y("Message:N", sort="-x", title=None),
                x=alt.X("frac:Q", title="fracción de la cohorte que lo ve alguna vez",
                        axis=alt.Axis(format=".0%")),
                yOffset="cohorte:N",
                color=color_cohorte(),
                tooltip=["Message:N", "cohorte:N", alt.Tooltip("n:Q", title="vehículos"),
                         alt.Tooltip("frac:Q", format=".1%")],
            )
            .properties(title="Qué fracción de cada cohorte ve cada nivel de `Message`")
        )
        mostrar(barras, altura=340)
    with derecha:
        severidad = {
            "regen_per_1000km": "Regeneraciones por 1.000 km",
            "msg_overloaded_per_1000km": "`Overloaded` por 1.000 km",
            "msg_full_per_1000km": "`Full` por 1.000 km",
            "acumulation_mean": "`Acumulation` medio",
            "acumulation_saturated_frac": "Fracción del tiempo en 95",
            "dist_between_regen_mean": "Distancia entre regeneraciones [km]",
            "air_regen_end_mean": "`AirRegenerationEnd` medio",
            "air_filter_abnormal_frac": "Viajes con `AirFilterEnd` anormal",
        }
        columna = st.selectbox("Variable", list(severidad), format_func=lambda c: severidad[c],
                               key="severidad_var")
        caja = (
            alt.Chart(con_etiqueta(veh))
            .mark_boxplot(size=52, outliers={"size": 10, "opacity": 0.25}, median={"color": "white"})
            .encode(
                x=alt.X("cohorte:N", title=None, axis=alt.Axis(labelAngle=0)),
                y=alt.Y(f"{columna}:Q", title=severidad[columna]),
                color=color_cohorte(None),
                tooltip=["vehicle_id:N", alt.Tooltip(f"{columna}:Q", format=".3f")],
            )
            .properties(title=severidad[columna])
        )
        mostrar(caja, altura=340)
        medianas = veh.groupby("event_observed")[columna].median()
        st.metric(
            "Mediana con evento / sin evento",
            f"{medianas.get(1, np.nan):.3f} vs {medianas.get(0, np.nan):.3f}",
            help="Un punto = un vehículo. OJO: `regen_per_1000km` y `dist_between_regen_mean` salen del "
                 "marcador `Regenerations`, cortado el 25-05-2026: su separación es exposición al "
                 "calendario, no física (corrección del 18-09, ver el aviso de arriba).",
        )

    st.divider()
    st.markdown("#### `AirFilter*` y `AirRegeneration*`: las columnas que el diccionario no declara")
    columnas = st.columns(3)
    columnas[0].metric("Vocabulario compartido con `Message`", "9 de 9 niveles")
    columnas[1].metric("`AirRegenerationEnd` == `Acumulation`", "99,8%",
                       help="Sobre 16.324 pares alineados a menos de 30 minutos (Pearson 0,9995).")
    columnas[2].metric("`AirFilterEnd` == `Message`", "99,98%")
    st.success(
        "**`trips` trae una foto por viaje de las dos variables dinámicas de `signals`.** Eso "
        "desbloquea la familia B del plan §4 a nivel viaje: `feat_dpf_*` sale de "
        "`AirRegenerationStart/End`. *Que además sean el `DieselParticulateFilter*` del anexo 7.1 "
        "es la explicación que mejor encaja —el PDF avisa que las variables vienen renombradas—, "
        "pero no está confirmado por Ford.*", icon=":material/check_circle:",
    )

    sondas = sorted(cache["probe_trips_dev"]["vehicle_id"].unique())
    sonda = st.selectbox("Vehículo sonda (historial completo en el cache)", sondas, key="sonda")
    viajes = cache["probe_trips_dev"]
    viajes = viajes[viajes["vehicle_id"].eq(sonda)].sort_values("OdometerTripEnd").tail(900)
    senales = cache["probe_signals_dev"]
    senales = senales[senales["vehicle_id"].eq(sonda) & senales["Regenerations"]]
    senales = senales[senales["OdometerValue"].between(viajes["OdometerTripEnd"].min(),
                                                       viajes["OdometerTripEnd"].max())]
    nivel = (
        alt.Chart(viajes)
        .mark_line(strokeWidth=1.6, color=paleta()["cohort"][0])
        .encode(
            x=alt.X("OdometerTripEnd:Q", title="odómetro [km]", scale=alt.Scale(zero=False)),
            y=alt.Y("AirRegenerationEnd:Q", title="nivel [0–95]"),
            tooltip=[alt.Tooltip("OdometerTripEnd:Q", title="odómetro", format=".0f"),
                     alt.Tooltip("AirRegenerationEnd:Q", title="nivel"),
                     alt.Tooltip("AirFilterEnd:N", title="AirFilterEnd")],
        )
    )
    marcas = (
        alt.Chart(senales)
        .mark_rule(strokeWidth=1, opacity=0.45, color=paleta()["cohort"][1])
        .encode(x="OdometerValue:Q",
                tooltip=[alt.Tooltip("OdometerValue:Q", title="regeneración en [km]", format=".0f")])
    )
    mostrar((nivel + marcas).properties(
        title=f"{sonda} · el nivel sube con el uso y se descarga en cada regeneración (líneas rojas)"),
        altura=280)


def seccion_correlaciones(cache: dict[str, Any], veh: pd.DataFrame) -> None:
    from scipy.stats import spearmanr

    st.subheader("Correlaciones")
    st.warning(
        "**Regla 6.** Todo lo de esta pestaña se calcula sobre el **historial completo** del "
        "vehículo y contra una etiqueta **a nivel vehículo**, sin gap de blanking. Un ρ alto acá no "
        "es un PR-AUC: es por dónde va a intentar memorizar el modelo si F2 no pone el gap. Las "
        "variables de cobertura (`n_trips`, `span_days`, `trip_odo_max`) no son features: son "
        "cuánto se observó a cada vehículo.", icon=":material/warning:",
    )

    familias = {
        "A · térmica / trayectos cortos": ["short_trip_frac", "very_short_trip_frac", "below_regime_frac",
                                           "engine_temp_avg_median", "engine_temp_max_median",
                                           "engine_temp_amplitude_mean", "coolant_end_mean",
                                           "trip_km_median", "trip_km_p25"],
        "B · ciclo de regeneración": ["acumulation_mean", "acumulation_max", "acumulation_high_frac",
                                      "acumulation_saturated_frac", "acumulation_slope_per_1000km",
                                      "air_regen_end_mean", "air_regen_saturated_frac",
                                      "air_regen_positive_delta_frac", "air_regen_slope_per_1000km",
                                      "air_regen_drops_per_1000km", "regen_per_1000km",
                                      "dist_between_regen_mean", "dist_between_regen_slope_per_1000km"],
        "C · uso y ambiente": ["speed_mean", "urban_trip_frac", "km_per_day", "trips_per_day",
                               "hours_between_trips_median", "air_temp_avg_mean", "air_temp_min_mean",
                               "refuel_frac"],
        "D · severidad": ["oil_life_start_mean", "air_filter_abnormal_frac", "msg_full_per_1000km",
                          "msg_overloaded_per_1000km", "msg_cleaning_auto_per_1000km",
                          "msg_over_limit_per_1000km"],
        "· cobertura (NO es feature)": ["n_trips", "n_signals", "span_days", "trip_odo_max",
                                        "trip_km_sum", "signals_per_1000km"],
        "· estática del set base": ["static_daysUntilSale", "static_ProductionDay"],
    }
    elegidas = st.multiselect("Familias", list(familias), default=list(familias), key="familias_corr")
    columnas = [c for f in elegidas for c in familias[f] if c in veh.columns]
    if not columnas:
        st.info("Elegí al menos una familia.")
        return

    filas = []
    for columna in columnas:
        serie = veh[columna]
        if serie.notna().sum() < 100 or serie.nunique() < 3:
            continue
        resultado = spearmanr(serie, veh["event_observed"], nan_policy="omit")
        familia = next(f for f, cols in familias.items() if columna in cols)
        filas.append({"variable": columna, "familia": familia, "rho": float(resultado.statistic),
                      "p": float(resultado.pvalue)})
    contra = pd.DataFrame(filas).sort_values("rho", key=abs, ascending=False)

    rango = list(paleta()["categorical"][: len(elegidas)])
    barras = (
        alt.Chart(contra.head(30))
        .mark_bar(cornerRadiusEnd=4, stroke="white", strokeWidth=1.2)
        .encode(
            y=alt.Y("variable:N", sort="-x", title=None),
            x=alt.X("rho:Q", title="ρ de Spearman con `event_observed`",
                    scale=alt.Scale(domain=[-0.45, 0.45])),
            color=alt.Color("familia:N", scale=alt.Scale(domain=elegidas, range=rango),
                            legend=alt.Legend(title=None, orient="top", columns=2)),
            tooltip=["variable:N", "familia:N", alt.Tooltip("rho:Q", format=".3f"),
                     alt.Tooltip("p:Q", format=".2e")],
        )
        .properties(title="Correlación con la etiqueta · historial completo, SIN gap")
    )
    mostrar(barras, altura=max(320, 22 * min(len(contra), 30)))

    st.markdown("**Matriz entre agregados** — para detectar features que miden lo mismo.")
    matriz = veh[columnas].corr(method="spearman")
    largo = matriz.stack().rename("rho").reset_index()
    largo.columns = ["a", "b", "rho"]
    mapa = (
        alt.Chart(largo)
        .mark_rect(stroke="white", strokeWidth=1)
        .encode(
            x=alt.X("a:N", title=None, axis=alt.Axis(labelAngle=-45)),
            y=alt.Y("b:N", title=None),
            color=alt.Color("rho:Q",
                            scale=alt.Scale(scheme="redblue", domain=[-1, 1], reverse=True),
                            legend=alt.Legend(title="ρ")),
            tooltip=["a:N", "b:N", alt.Tooltip("rho:Q", format=".3f")],
        )
        .properties(title="Spearman entre agregados por vehículo")
    )
    mostrar(mapa, altura=max(360, 16 * len(columnas)))

    st.markdown("**Diagnóstico de forma** — qué transformación pediría cada feature, y si serviría.")
    st.caption(
        "Acá no se transforma nada: se mide. El punto que hay que tener presente es que **ρ de "
        "Spearman no cambia con una transformación monótona**, así que a un árbol el `log1p` no le "
        "agrega ni le saca nada. Lo que cambia es **Pearson**, o sea lo que ve un modelo lineal — la "
        "regresión logística que es la línea de base."
    )
    candidatas = [c for fam, cols in familias.items() if not fam.startswith("·")
                  for c in cols if c in veh.columns]
    filas = []
    for columna in dict.fromkeys(candidatas):
        x = veh[columna].dropna()
        if len(x) < 100 or x.nunique() < 3:
            continue
        objetivo_y = veh.loc[x.index, "event_observed"]
        mediana = float(x.median())
        pearson = abs(float(np.corrcoef(x, objetivo_y)[0, 1]))
        pearson_log = (abs(float(np.corrcoef(np.log1p(x), objetivo_y)[0, 1]))
                       if x.min() >= 0 else np.nan)
        binaria = (x > 0).astype(int)
        filas.append({
            "variable": columna,
            "asimetría": float(x.skew()),
            "frac. ceros": float((x == 0).mean()),
            "ρ Spearman": float(spearmanr(x, objetivo_y).statistic),
            "|Pearson|": pearson,
            "|Pearson| log1p": pearson_log,
            "ganancia log": pearson_log - pearson,
            "ρ del indicador": (float(spearmanr(binaria, objetivo_y).statistic)
                                if binaria.nunique() > 1 else np.nan),
        })
    forma = pd.DataFrame(filas)

    def _sugerencia(fila: pd.Series) -> str:
        if fila["frac. ceros"] >= 0.60:
            return "indicador binario"
        if fila["asimetría"] > 2 and fila["ganancia log"] > 0.03:
            return "log1p"
        if fila["asimetría"] > 2:
            return "log1p (cosmético)"
        if fila["asimetría"] < -2:
            return "satura arriba: usar frac. en nivel alto"
        return "dejar como está"

    forma["sugerencia"] = forma.apply(_sugerencia, axis=1)
    forma = forma.sort_values("ganancia log", ascending=False)

    izquierda, derecha = st.columns([3, 2])
    with izquierda:
        dispersion = (
            alt.Chart(forma.dropna(subset=["ganancia log"]))
            .mark_circle(size=130, opacity=0.8, stroke="white", strokeWidth=1.2)
            .encode(
                x=alt.X("asimetría:Q", title="asimetría (skew)"),
                y=alt.Y("ganancia log:Q", title="ganancia de |Pearson| con log1p"),
                color=alt.Color("sugerencia:N",
                                scale=alt.Scale(range=list(paleta()["categorical"])),
                                legend=alt.Legend(title=None, orient="top", columns=2)),
                size=alt.Size("frac. ceros:Q", legend=None, scale=alt.Scale(range=[40, 420])),
                tooltip=["variable:N", "sugerencia:N",
                         alt.Tooltip("asimetría:Q", format=".2f"),
                         alt.Tooltip("frac. ceros:Q", format=".1%"),
                         alt.Tooltip("ρ Spearman:Q", format=".3f"),
                         alt.Tooltip("ganancia log:Q", format=".3f")],
            )
            .properties(title="Asimetría alta no implica que el log sirva")
        )
        cero = (alt.Chart(pd.DataFrame({"y": [0.0]}))
                .mark_rule(color=paleta()["ink"]["secondary"], strokeWidth=1).encode(y="y:Q"))
        mostrar(dispersion + cero, altura=340)
    with derecha:
        st.dataframe(
            forma.head(10)[["variable", "ρ Spearman", "ganancia log", "sugerencia"]]
            .style.format({"ρ Spearman": "{:.3f}", "ganancia log": "{:+.3f}"}),
            width="stretch", hide_index=True,
        )
        st.caption(
            "**Ojo con la ganancia sin Spearman detrás.** `acumulation_median` gana +0,064 de "
            "|Pearson| y su ρ de Spearman es −0,003: no hay señal que linealizar, el |Pearson| "
            "subió por azar. Mirar siempre las dos columnas juntas."
        )

    st.markdown("**¿El valor aporta sobre el indicador `pasó / no pasó`?**")
    zero = forma.dropna(subset=["ρ del indicador"]).copy()
    zero["el valor aporta"] = zero["ρ Spearman"].abs() - zero["ρ del indicador"].abs()
    comparacion = (
        alt.Chart(zero)
        .mark_circle(opacity=0.8, stroke="white", strokeWidth=1.2)
        .encode(
            x=alt.X("ρ del indicador:Q", title="|ρ| del indicador · ¿pasó alguna vez?",
                    scale=alt.Scale(zero=False)),
            y=alt.Y("ρ Spearman:Q", title="|ρ| del valor · ¿cuánto?", scale=alt.Scale(zero=False)),
            size=alt.Size("frac. ceros:Q", legend=alt.Legend(title="frac. ceros"),
                          scale=alt.Scale(range=[40, 420])),
            color=alt.Color("el valor aporta:Q",
                            scale=alt.Scale(scheme="redblue", domainMid=0),
                            legend=alt.Legend(title="valor − indicador")),
            tooltip=["variable:N", alt.Tooltip("frac. ceros:Q", format=".1%"),
                     alt.Tooltip("ρ Spearman:Q", format=".3f"),
                     alt.Tooltip("ρ del indicador:Q", format=".3f"),
                     alt.Tooltip("el valor aporta:Q", format="+.3f")],
        )
        .properties(title="Sobre la diagonal, la magnitud aporta; sobre la línea, alcanza el indicador")
    )
    diagonal = (
        alt.Chart(pd.DataFrame({"x": [0.0, float(zero["ρ Spearman"].abs().max())]}))
        .mark_line(color=paleta()["ink"]["grid"], strokeWidth=1.8).encode(x="x:Q", y="x:Q")
    )
    mostrar(comparacion + diagonal, altura=340)
    st.caption(
        "`msg_at_limit_per_1000km` es 94,5% ceros y `msg_over_limit_per_1000km` 87,6%: en los dos, "
        "el ρ del indicador iguala al del valor, así que **toda la señal está en \"pasó alguna "
        "vez\"**. Al revés en `regen_per_1000km` (0,317 contra 0,183): ahí la magnitud es lo que "
        "importa. Y en las de saturación el indicador **supera** al valor. "
        "Todo esto se mide sobre 290 vehículos y 60 eventos: sirve para ordenar candidatas, no "
        "para descartarlas."
    )

    st.divider()
    st.markdown("**Preview del gap G** — los mismos agregados sobre el primer 70% y 80% del recorrido.")
    severidad = cache["severity_scopes_dev"].merge(veh[["vehicle_id", "event_observed"]], on="vehicle_id")
    objetivo = [c for c in severidad.columns
                if c.startswith("msg_") or c in {"acumulation_mean", "acumulation_high_frac",
                                                 "acumulation_saturated_frac", "regen_per_1000km",
                                                 "dist_between_regen_mean", "air_regen_end_mean",
                                                 "air_regen_saturated_frac", "air_filter_abnormal_frac",
                                                 "short_trip_frac", "below_regime_frac"}]
    filas = []
    for alcance, grupo in severidad.groupby("scope"):
        for columna in objetivo:
            if grupo[columna].notna().sum() < 100 or grupo[columna].nunique() < 3:
                continue
            filas.append({"alcance": alcance, "variable": columna,
                          "rho": float(spearmanr(grupo[columna], grupo["event_observed"],
                                                 nan_policy="omit").statistic)})
    truncado = pd.DataFrame(filas)
    etiquetas_alcance = {"full": "historial completo", "p70": "primer 70%", "p80": "primer 80%"}
    truncado["alcance"] = truncado["alcance"].map(etiquetas_alcance)
    orden = (truncado[truncado["alcance"].eq("historial completo")]
             .sort_values("rho", key=abs, ascending=False)["variable"].tolist())
    puntos = (
        alt.Chart(truncado)
        .mark_point(size=110, filled=True, stroke="white", strokeWidth=1.2)
        .encode(
            y=alt.Y("variable:N", sort=orden, title=None),
            x=alt.X("rho:Q", title="ρ de Spearman con `event_observed`"),
            color=alt.Color("alcance:N",
                            scale=alt.Scale(domain=list(etiquetas_alcance.values()),
                                            range=list(paleta()["categorical"][:3])),
                            legend=alt.Legend(title=None, orient="top")),
            tooltip=["variable:N", "alcance:N", alt.Tooltip("rho:Q", format=".4f")],
        )
        .properties(title="Al blanquear la cola: qué anticipa y qué solo describe el presente")
    )
    regla = (
        alt.Chart(truncado).mark_rule(color=paleta()["ink"]["grid"], strokeWidth=2)
        .encode(y=alt.Y("variable:N", sort=orden), x="min(rho):Q", x2="max(rho):Q")
    )
    mostrar(regla + puntos, altura=max(320, 24 * truncado["variable"].nunique()))
    st.error(
        "**Retirado el 2026-09-18.** El truncamiento al 70% del odómetro saca sobre todo km "
        "posteriores a mayo de 2026, que es justo donde el marcador `Regenerations` deja de "
        "registrarse: por eso las tasas de marcadores \"sobreviven\". No es evidencia de anticipación. "
        "La prueba que sí la mide es el perfil alineado al evento de `scripts/eda_gaps.py` "
        "(`docs/reproducibilidad.md`). El texto de abajo se conserva tal cual.",
        icon=":material/report:",
    )
    st.info(
        "**Ahora esta prueba sí discrimina, y es el resultado más útil del EDA.** Con el evento "
        "cayendo en la mediana a 7.987 km y el 39% del historial por delante, truncar al primer 70% "
        "del recorrido se parece de verdad a blanquear el tramo previo.\n\n"
        "**Sobreviven** (retienen ≥86% de su ρ): `regen_per_1000km` 90%, "
        "`msg_over_limit_per_1000km` 101%, `msg_cleaning_auto_per_1000km` 91%, "
        "`msg_at_limit_per_1000km` 86%. Son **frecuencias**: rasgo del régimen de uso, presentes "
        "desde el principio del historial.\n\n"
        "**Se caen**: `air_regen_saturated_frac` 58%, `air_regen_end_mean` 32%, "
        "`acumulation_high_frac` 18%. Son **niveles de saturación**: estado que aparece cerca del "
        "evento. Un modelo construido sobre estos daría un PR-AUC alto y sería detección reactiva, "
        "que es lo que Ford ya tiene (regla 6).",
        icon=":material/insights:",
    )
    st.caption(
        "Dos caveats: la etiqueta de esta prueba es **a nivel vehículo**, no por punto de corte, así "
        "que el truncamiento es un proxy del gap G y no el gap G; y el 70% del recorrido deja dentro "
        "parte del tramo post-evento para varios vehículos, con lo cual la prueba **subestima** "
        "cuánto sobrevive la señal. La definitiva se hace sobre el panel, con y sin G."
    )


def seccion_temporal(cache: dict[str, Any], veh: pd.DataFrame) -> None:
    st.subheader("Análisis temporal")
    diario = con_etiqueta(cache["trips_daily_dev"].assign(date=lambda d: pd.to_datetime(d["date"])))

    metrica = st.segmented_control(
        "Serie", ["n_vehicles", "n_trips", "km"], default="n_vehicles",
        format_func=lambda m: {"n_vehicles": "vehículos activos", "n_trips": "viajes",
                               "km": "km recorridos"}[m], key="serie_temporal",
    ) or "n_vehicles"
    zoom = alt.selection_interval(encodings=["x"])
    linea = (
        alt.Chart(diario)
        .mark_line(strokeWidth=1.8, opacity=0.9)
        .encode(
            x=alt.X("date:T", title="fecha", scale=alt.Scale(domain=zoom)),
            y=alt.Y(f"{metrica}:Q", title=metrica.replace("n_", "")),
            color=color_cohorte(),
            tooltip=[alt.Tooltip("date:T", title="fecha"), "cohorte:N",
                     alt.Tooltip(f"{metrica}:Q", format=",.0f")],
        )
        .properties(title="Actividad diaria de la flota · arrastrá abajo para hacer zoom")
    )
    contexto = (
        alt.Chart(diario).mark_area(opacity=0.4)
        .encode(x=alt.X("date:T", title=None), y=alt.Y(f"sum({metrica}):Q", title=None, axis=None),
                color=alt.value(paleta()["ink"]["muted"]))
        .add_params(zoom).properties(height=48)
    )
    compuesto = alt.vconcat(linea.properties(height=280), contexto).resolve_scale(color="independent")
    st.altair_chart(estilar(compuesto, altura=None), width="stretch", theme=None)

    izquierda, derecha = st.columns(2)
    DIAS = ["lun", "mar", "mié", "jue", "vie", "sáb", "dom"]
    with izquierda:
        semana = diario.groupby(["weekday", "cohorte"], observed=True)["n_trips"].sum().reset_index()
        semana["frac"] = semana["n_trips"] / semana.groupby("cohorte")["n_trips"].transform("sum")
        semana["dia"] = semana["weekday"].map(dict(enumerate(DIAS)))
        barras = (
            alt.Chart(semana)
            .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, stroke="white", strokeWidth=2)
            .encode(
                x=alt.X("dia:N", sort=DIAS, title=None, axis=alt.Axis(labelAngle=0)),
                y=alt.Y("frac:Q", title="fracción de viajes", axis=alt.Axis(format=".0%")),
                xOffset="cohorte:N", color=color_cohorte(),
                tooltip=["dia:N", "cohorte:N", alt.Tooltip("frac:Q", format=".2%")],
            )
            .properties(title="Viajes por día de la semana")
        )
        mostrar(barras, altura=280)
    with derecha:
        mensual = diario.groupby(["month", "cohorte"], observed=True)["n_trips"].sum().reset_index()
        mensual["frac"] = mensual["n_trips"] / mensual.groupby("cohorte")["n_trips"].transform("sum")
        linea_mes = (
            alt.Chart(mensual)
            .mark_line(strokeWidth=2.5, point=alt.OverlayMarkDef(size=45, filled=True))
            .encode(
                x=alt.X("month:T", title=None),
                y=alt.Y("frac:Q", title="fracción de viajes", axis=alt.Axis(format=".0%")),
                color=color_cohorte(),
                tooltip=[alt.Tooltip("month:T", title="mes"), "cohorte:N",
                         alt.Tooltip("frac:Q", format=".2%")],
            )
            .properties(title="Viajes por mes")
        )
        mostrar(linea_mes, altura=280)

    st.markdown("**`daysUntilSale` y la tasa de eventos**")
    bins = np.linspace(0, float(veh["static_daysUntilSale"].max()), 26)
    tasa = (veh.assign(b=pd.cut(veh["static_daysUntilSale"], bins))
            .groupby("b", observed=True)
            .agg(n=("event_observed", "size"), tasa=("event_observed", "mean")).reset_index())
    tasa["dias"] = [float(i.mid) for i in tasa["b"]]
    barras = (
        alt.Chart(tasa)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, stroke="white", strokeWidth=1.5,
                  color=paleta()["cohort"][0])
        .encode(
            x=alt.X("dias:Q", title="días desde producción hasta la venta"),
            y=alt.Y("tasa:Q", title="tasa de eventos", axis=alt.Axis(format=".0%")),
            tooltip=[alt.Tooltip("dias:Q", title="días", format=".0f"),
                     alt.Tooltip("n:Q", title="vehículos"),
                     alt.Tooltip("tasa:Q", title="tasa", format=".1%")],
        )
    )
    base = (
        alt.Chart(pd.DataFrame({"t": [float(veh["event_observed"].mean())]}))
        .mark_rule(strokeDash=[5, 4], strokeWidth=1.5, color=paleta()["cohort"][1])
        .encode(y="t:Q", tooltip=[alt.Tooltip("t:Q", title="tasa global", format=".1%")])
    )
    mostrar((barras + base).properties(title="Tasa de eventos por `daysUntilSale` · línea roja = tasa global"),
            altura=260)


def seccion_factibilidad() -> None:
    diccionario, factibilidad, reales = cargar_diccionario()
    st.subheader("Diccionario de datos y factibilidad de las features")

    columnas = st.columns(4)
    columnas[0].metric("Columnas reales de `trips`", f"{len(reales['trips'])}",
                       f"el PDF declara {len(PDF_COLUMNS['trips'])}", delta_color="off")
    ausentes = diccionario[diccionario["estado"].eq("documentada y AUSENTE")]
    columnas[1].metric("Prometidas y ausentes", f"{len(ausentes)}")
    columnas[2].metric("Presentes sin documentar",
                       f"{int(diccionario['estado'].eq('presente SIN documentar').sum())}")
    columnas[3].metric("Features bloqueadas",
                       f"{int(factibilidad['estado'].eq('bloqueada').sum())} de {len(factibilidad)}")

    st.error(
        "**El `TripSummary` real tiene 25 columnas, no las 40 del anexo 7.1.** No están, bajo ningún "
        "nombre: GPS (4), elevación (2), presión de neumáticos (8), `FuelLvlAutonomyStart` y el "
        "hollín de regeneración manual (2). Eso deja **4 features del plan §4 sin datos** y sin "
        "sustituto posible.", icon=":material/error:",
    )

    izquierda, derecha = st.columns([2, 3])
    with izquierda:
        resumen = pd.DataFrame([
            {"familia": familia, "columnas": len(cols),
             "todas ausentes": set(cols).issubset(set(ausentes["columna"]))}
            for familia, cols in MISSING_FAMILIES.items()
        ])
        st.dataframe(resumen, width="stretch", hide_index=True)
        st.caption("`DieselParticulateFilter*` está en esta lista por su **nombre**: el dato existe "
                   "como `AirRegeneration*` (ver la pestaña **Severidad**).")
    with derecha:
        conteo = factibilidad.groupby(["familia", "estado"]).size().reset_index(name="n")
        orden = ["disponible", "parcial", "degenerada", "bloqueada"]
        colores = [paleta()["categorical"][2], paleta()["categorical"][3],
                   paleta()["categorical"][1], paleta()["cohort"][1]]
        apiladas = (
            alt.Chart(conteo)
            .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, stroke="white", strokeWidth=2)
            .encode(
                x=alt.X("familia:N", title="familia del plan §4", axis=alt.Axis(labelAngle=0)),
                y=alt.Y("n:Q", title="features"),
                color=alt.Color("estado:N", scale=alt.Scale(domain=orden, range=colores),
                                legend=alt.Legend(title=None, orient="top")),
                order=alt.Order("color_estado_sort_index:Q"),
                tooltip=["familia:N", "estado:N", alt.Tooltip("n:Q", title="features")],
            )
            .properties(title="Features del plan §4 por familia y factibilidad")
        )
        mostrar(apiladas, altura=300)

    estado = st.segmented_control("Filtrar", ["todas", *orden], default="todas", key="estado_feat")
    tabla = factibilidad if estado in (None, "todas") else factibilidad[factibilidad["estado"].eq(estado)]
    st.dataframe(
        tabla[["feature", "familia", "columna_requerida", "tabla", "veredicto", "sustituto"]],
        width="stretch", hide_index=True,
    )

    with st.expander("Comparación completa de tres vías (CSV · `raw_sources.yaml` · PDF)"):
        st.dataframe(diccionario, width="stretch", hide_index=True)


# ======================================================================================
# App
# ======================================================================================
def main() -> None:
    st.set_page_config(page_title="Ford FIC · EDA de datos crudos (dev)", layout="wide",
                       page_icon=":material/monitoring:")

    st.markdown(
        f"<h2 style='margin-bottom:0;color:{'#7ba8ff' if modo() == 'dark' else '#1c3f94'}'>"
        "Ford FIC III · EDA de los datos crudos</h2>"
        f"<p style='margin-top:4px;color:{paleta()['ink']['muted']}'>"
        "Predicción temprana de degradación de eficiencia de combustión · exploración previa a F2"
        "</p>", unsafe_allow_html=True,
    )

    if not cache_exists(CONFIG):
        st.error(
            f"No hay cache del EDA en `{cache_dir(CONFIG)}`. Se construye con una pasada por los "
            "CSV crudos (~5 minutos):\n\n"
            f"```\npython scripts/build_eda_cache.py --config {CONFIG}\n```",
            icon=":material/database_off:",
        )
        if st.button("Construirlo ahora", type="primary"):
            construir_cache()
        st.stop()

    cache = cargar()
    veh = cache["veh"]
    meta = cache["meta"]

    st.warning(
        f"**Alcance: dev, {len(veh)} vehículos.** Los **{meta['n_test_vehicles_excluidos']} de test** "
        "del holdout congelado (`data/processed/test_split.json`, semilla "
        f"{meta['test_split_seed']}) **no están en estos datos**, y tampoco los "
        f"**{meta.get('n_fuera_del_universo', 0)} que quedaron fuera del universo del estudio** "
        "(pestaña **Etiqueta**). El holdout se congeló antes de F2 para que ninguna decisión de "
        "diseño se tome mirándolo, y el recorte del universo se aplicó **sobre ese sorteo**, sin "
        "volver a tirar el dado: cada vehículo que sobrevive conserva el lado que le había tocado.",
        icon=":material/lock:",
    )
    st.error(
        "**Corrección del 2026-09-18 (vale para las pestañas Severidad y Correlaciones).** "
        "`regen_per_1000km` y `dist_between_regen_*` salen del marcador `signals.Regenerations`, que "
        "**se corta el 25-05-2026 para toda la flota** (0 marcadores en jun–sep 2026 contra ~2.000 "
        "caídas de nivel por mes en `trips`). Su correlación con la etiqueta (ρ = 0,317) es exposición "
        "al calendario —ρ con `ProductionDay` = −0,515—, no física: contada desde las caídas de "
        "`AirRegeneration` da ρ = 0,072. La conclusión \"las frecuencias sobreviven al truncamiento\" "
        "queda retirada. La versión vigente del análisis es `docs/reproducibilidad.md` "
        "y `scripts/eda_gaps.py`; este dashboard no se regeneró.",
        icon=":material/report:",
    )

    with st.sidebar:
        st.markdown("### Alcance")
        st.metric("Vehículos de dev", len(veh))
        st.metric("De test (intocables)", meta["n_test_vehicles_excluidos"])
        st.metric("Fuera del universo", meta.get("n_fuera_del_universo", 0),
                  help="Positivos sin fecha de evento utilizable, y los mercados donde ningún "
                       "evento es observable. Ver `src/data/usable.py`.")
        st.caption(
            f"Cache construido el {meta['created_at'][:10]} · semilla {meta['seed']}.\n\n"
            f"Filas deduplicadas: {meta['trips']['dup_clones']:,} viajes y "
            f"{meta['signals']['dup_clones'] + meta['signals']['dup_otros']:,} señales "
            "(clones + repetidas exactas).".replace(",", ".")
        )
        st.divider()
        st.markdown("### Fuentes")
        st.caption(
            "Todo sale de `experiments/eda/dev/`, que produce "
            "`scripts/build_eda_cache.py` en una sola pasada por los CSV.\n\n"
            "El recorrido completo, con el detalle y la evidencia, está en "
            "`notebooks/eda-exhaustivo-dev.ipynb`."
        )
        if st.button("Reconstruir el cache", width="stretch"):
            construir_cache()
        st.divider()
        st.caption(f"Repo: `{repo_root()}`")

    kpis(veh, meta)
    st.divider()

    pestanas = st.tabs([
        "Estáticas", "Etiqueta y supervivencia", "Odómetro", "Uso", "Severidad",
        "Correlaciones", "Temporal", "Diccionario y features",
    ])
    with pestanas[0]:
        seccion_estaticas(veh)
    with pestanas[1]:
        seccion_etiqueta(cache, veh)
    with pestanas[2]:
        seccion_odometro(veh)
    with pestanas[3]:
        seccion_uso(cache, veh)
    with pestanas[4]:
        seccion_severidad(cache, veh)
    with pestanas[5]:
        seccion_correlaciones(cache, veh)
    with pestanas[6]:
        seccion_temporal(cache, veh)
    with pestanas[7]:
        seccion_factibilidad()


if __name__ == "__main__":
    main()
