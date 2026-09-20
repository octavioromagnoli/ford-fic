"""Derivadas a nivel viaje: lo que el crudo no trae y toda feature de ventana necesita.

Una sola función, `derive_trip_columns`, que recibe `trips` (ya canonizado y filtrado
al universo) y los umbrales del YAML, y devuelve la misma tabla con columnas nuevas.
Ninguna de estas columnas mira otro viaje que no sea el anterior del mismo vehículo
(para el tiempo de reposo), así que todas se pueden agregar hacia atrás sin riesgo.

Lo que se decide acá y por qué (evidencia en `docs/memoria/f2-eda-revision-y-features.md`):

- **`idle` vs. `moving`.** El 35% de las filas de `trips` tiene 0 km: motor encendido
  sin desplazamiento (mediana 0,8 min). No son viajes cortos, son otra cosa, y son
  la señal que más anticipa el evento. Se cuentan aparte, y toda fracción "de
  viaje" (trayecto corto, régimen, velocidad) se calcula solo entre los `moving`.
- **Velocidad recalculada.** `KilometerPerHour` es nulo exactamente cuando
  `trip_km == 0` y, cuando existe, es idéntica a `trip_km / duración`
  (Pearson 0,99999). Se recalcula, y se recorta a un tope físico.
- **Recorte físico, no por cuantil.** Los cuantiles dependerían del train de cada
  fold; los topes físicos (una temperatura ambiente de −73 °C, 20 días de viaje) no
  dependen de ningún dato y se declaran en el YAML.
- **`FuelLvl*Pc` no se recorta a [0, 100]**: supera 100 en el 10% de las filas, la
  escala no es un porcentaje literal.
- **`EngineOilLifePC` no cambia dentro del viaje** (100% de las filas): la feature es
  la pendiente del nivel dentro de la ventana, no la suma de caídas.
- **Regeneración = caída de `AirRegeneration` dentro del viaje.** El marcador
  `Regenerations` de `signals` se corta el 25-05-2026 para toda la flota y no sirve
  como contador; la caída del nivel al final del viaje respecto del inicio sí se
  registra hasta el final del historial.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ID_COL = "vehicle_id"
NORMAL_MESSAGE = "Air Filter Normal Operation"
CLEANING_AUTO = "Cleaning Automatically Air Filter"
MANUAL_LEVELS = ("Cleanning Manually Air Filter", "Cleaning Manually Air Filter",
                 "Stopped Cleanning Manually Air Filter", "Stopped Cleaning Manually Air Filter")

DEFAULT_THRESHOLDS: dict[str, float] = {
    "short_trip_km": 5.0,        # trayecto corto (entre viajes con desplazamiento)
    "regime_temp_c": 70.0,       # EngineTemperatureMax por debajo => no llegó a régimen
    "urban_speed_kmh": 30.0,     # velocidad media por debajo => uso urbano
    "cold_start_c": 40.0,        # refrigerante al arrancar por debajo => arranque en frío
    "chained_trip_min": 30.0,    # arranca a menos de X min del viaje anterior => motor aún caliente
    "regen_drop_points": 5.0,    # caída de AirRegeneration dentro del viaje que cuenta como regeneración
    "saturation_level": 95.0,    # nivel de AirRegeneration considerado saturado
    "fuel_min_trip_km": 5.0,     # km mínimos para estimar consumo en un viaje
    "aborted_level": 50.0,       # nivel al que todavía se considera "cargado": ciclo que no terminó
    "long_trip_km": 15.0,        # viaje largo: si con esa distancia no llega a régimen, es defecto
    "opportunity_km": 15.0,      # corrida mínima para que quepa un ciclo de regeneración (p25 de las que ocurren)
    "opportunity_temp_c": 90.0,  # y temperatura mínima (p10 de EngineTemperatureMax de las que ocurren)
    "min_warmup_min": 5.0,       # duración mínima para que °C/min signifique algo
    "long_soak_min": 240.0,      # horas parado antes del viaje => el motor arranca a ambiente
}

DEFAULT_CLIP: dict[str, tuple[float, float]] = {
    "speed_kmh": (0.0, 200.0),
    "trip_duration_min": (0.0, 1440.0),
    "AirTemperatureAvg": (-40.0, 60.0),
    "AirTemperatureMin": (-40.0, 60.0),
    "AirTemperatureMax": (-40.0, 60.0),
    "CoolantTemperatureStart": (-40.0, 130.0),
    "CoolantTemperatureEnd": (-40.0, 130.0),
    "EngineTemperatureMin": (-40.0, 130.0),
    "EngineTemperatureMax": (-40.0, 130.0),
    "EngineTemperatureAvg": (-40.0, 130.0),
}

REQUIRED = [
    ID_COL, "OdometerTripStart", "OdometerTripEnd", "TripDatetimeStart", "TripDatetimeEnd",
    "EngineTemperatureMin", "EngineTemperatureMax", "EngineTemperatureAvg",
    "CoolantTemperatureStart", "CoolantTemperatureEnd", "AirTemperatureAvg", "AirTemperatureMin",
    "AirRegenerationStart", "AirRegenerationEnd", "AirFilterStart", "AirFilterEnd",
    "FuelLvlStartPc", "FuelLvlEndPc", "EngineOilLifePCStart",
]


def derive_trip_columns(
    trips: pd.DataFrame,
    *,
    thresholds: dict[str, Any] | None = None,
    clip: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Agrega las derivadas a `trips` y descarta las filas que no se pueden ubicar en el eje de km.

    Devuelve `(trips, contadores)`. Se descartan (y se cuentan) las filas sin
    `OdometerTripEnd` y las de kilometraje negativo: no tienen posición confiable
    sobre el eje del panel. Todo lo demás se conserva; las fechas nulas (0,3%) solo
    dejan en NaN las derivadas temporales.
    """
    missing = [c for c in REQUIRED if c not in trips.columns]
    if missing:
        raise KeyError(f"`trips` no tiene las columnas requeridas: {missing}")
    thr = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    bounds = {**DEFAULT_CLIP, **{k: tuple(v) for k, v in (clip or {}).items()}}

    out = trips.copy()
    n_in = len(out)
    out["trip_km"] = (out["OdometerTripEnd"] - out["OdometerTripStart"]).astype("float64")
    drop_null = out["OdometerTripEnd"].isna()
    drop_negative = out["trip_km"] < 0
    out = out.loc[~(drop_null | drop_negative)].copy()
    counters = {
        "n_in": n_in,
        "dropped_null_odometer": int(drop_null.sum()),
        "dropped_negative_km": int(drop_negative.sum()),
        "n_out": int(len(out)),
    }

    for column, (lo, hi) in bounds.items():
        if column in out.columns:
            out[column] = out[column].astype("float64").clip(lo, hi)

    duration = (out["TripDatetimeEnd"] - out["TripDatetimeStart"]).dt.total_seconds() / 60.0
    out["trip_duration_min"] = duration.clip(*bounds["trip_duration_min"])
    hours = out["trip_duration_min"] / 60.0
    speed = (out["trip_km"] / hours.where(hours > 0)).where(out["trip_km"] > 0)
    out["speed_kmh"] = speed.clip(*bounds["speed_kmh"])

    out["idle"] = out["trip_km"].eq(0)
    out["moving"] = out["trip_km"].gt(0)
    out["short_trip"] = out["moving"] & out["trip_km"].lt(thr["short_trip_km"])
    out["below_regime"] = out["EngineTemperatureMax"].lt(thr["regime_temp_c"])
    out["urban"] = out["moving"] & out["speed_kmh"].lt(thr["urban_speed_kmh"])
    out["cold_start"] = out["CoolantTemperatureStart"].lt(thr["cold_start_c"])
    out["engine_temp_amplitude"] = out["EngineTemperatureMax"] - out["EngineTemperatureMin"]

    # Postratamiento a nivel viaje (AirRegeneration* == signals.Acumulation, misma escala 0-100).
    out["air_regen_delta"] = out["AirRegenerationEnd"] - out["AirRegenerationStart"]
    out["regen_drop"] = out["air_regen_delta"].lt(-thr["regen_drop_points"])
    out["regen_residual"] = out["AirRegenerationEnd"].where(out["regen_drop"])
    out["regen_start_level"] = out["AirRegenerationStart"].where(out["regen_drop"])
    out["dpf_positive_delta"] = out["moving"] & out["air_regen_delta"].gt(0)
    # Magnitud de la carga, no solo si hubo: puntos que SUBE el filtro en el viaje. Es el
    # denominador natural del esfuerzo de regeneración (cuánto hollín entró de verdad,
    # medido por el mismo sensor que después baja).
    out["air_regen_up"] = out["air_regen_delta"].clip(lower=0).where(out["moving"])
    out["dpf_saturated_end"] = out["AirRegenerationEnd"].ge(thr["saturation_level"])
    out["filter_abnormal_end"] = out["AirFilterEnd"].notna() & out["AirFilterEnd"].ne(NORMAL_MESSAGE)
    out["filter_cleaning_auto_end"] = out["AirFilterEnd"].eq(CLEANING_AUTO)
    out["filter_manual"] = out["AirFilterEnd"].isin(MANUAL_LEVELS) | out["AirFilterStart"].isin(MANUAL_LEVELS)

    # F3 · esfuerzo de control, no nivel. El nivel del DPF es una variable controlada
    # (los que fallan terminan los viajes MENOS cargados: P = 0,31), así que lo
    # informativo es cuánto trabajo cuesta mantenerlo bajo. Detalle del mecanismo en
    # `docs/f3-features-candidatas-fisica.md` §1.
    # `regen_efficiency` son los puntos que removió el ciclo; la fracción normaliza por
    # el nivel de arranque, que se correlaciona 0,55 con la caída (regenerar desde 90 y
    # desde 60 no es lo mismo, y sin normalizar la feature mide desde dónde arrancó).
    out["regen_efficiency"] = (-out["air_regen_delta"]).where(out["regen_drop"])
    start = out["AirRegenerationStart"].where(out["AirRegenerationStart"] > 0)
    out["regen_efficiency_frac"] = (out["regen_efficiency"] / start).where(out["regen_drop"])

    # Ciclo que no termina. Dos lecturas del mismo mecanismo ("el viaje corto corta la
    # regeneración"), porque cada una falla distinto:
    #   - `regen_aborted_end`: el viaje TERMINA con el sistema limpiando y el filtro
    #     todavía cargado. Es literal, pero raro (0,28% de los viajes, 161 vehículos).
    #   - `regen_partial`: hubo caída pero el nivel final sigue alto. Más frecuente
    #     (0,4-0,8%) y no depende de que el mensaje esté puesto en el último registro.
    out["regen_aborted_end"] = out["filter_cleaning_auto_end"] & out["AirRegenerationEnd"].gt(thr["aborted_level"])
    out["regen_partial"] = out["regen_drop"] & out["AirRegenerationEnd"].gt(thr["aborted_level"])

    # F3 §3 · dosis de km fríos, no fracción de viajes fríos. El hollín se acumula por
    # kilómetro frío: 40 viajes de 1 km y 3 de 20 km dan fracciones opuestas y dosis
    # parecidas. Agregado con `per_1000km` queda "km fríos por cada 1.000 km".
    out["km_below_regime"] = out["trip_km"].where(out["below_regime"], 0.0)

    # F3 §4 · la temperatura baja significa cosas distintas según el largo del viaje:
    # un viaje largo que no llega a régimen es un defecto; uno corto, uso normal. Las
    # dos columnas son NaN fuera de su subconjunto, así que `mean` es la fracción dentro.
    long_trip = out["moving"] & out["trip_km"].ge(thr["long_trip_km"])
    out["long_trip_below_regime"] = out["below_regime"].astype("float64").where(long_trip)
    out["short_trip_below_regime"] = out["below_regime"].astype("float64").where(out["short_trip"])

    # F3 §5 · oportunidad de regenerar: corrida lo bastante larga y caliente como para
    # que quepa un ciclo. Los umbrales salen de los viajes donde SÍ ocurre una
    # regeneración en dev (p25 de trip_km = 15 km, p10 de EngineTemperatureMax = 91 °C),
    # no de un número inventado.
    out["regen_opportunity"] = (out["moving"]
                                & out["trip_km"].ge(thr["opportunity_km"])
                                & out["EngineTemperatureMax"].ge(thr["opportunity_temp_c"]))
    out["km_in_opportunity"] = out["trip_km"].where(out["regen_opportunity"], 0.0)

    # ----------------------------------------------------------------------------------
    # F3 · familia F: lo que no se ve mirando una columna a la vez.
    # ----------------------------------------------------------------------------------
    # Dosis de TIEMPO, no de viajes ni de km. El motor encendido produce hollín por
    # minuto, no por kilómetro: un idle de 40 minutos y uno de 1 minuto cuentan igual en
    # `idle_frac` y en `idle_per_1000km` (los dos son "un idle"), y no son lo mismo. La
    # duración de los idle no estaba medida en ninguna de las 69 features.
    out["idle_min"] = out["trip_duration_min"].where(out["idle"], 0.0)
    out["below_regime_min"] = out["trip_duration_min"].where(out["below_regime"], 0.0)

    # Ritmo de calentamiento: cuántos grados por minuto gana el motor. Un vehículo que
    # tarda más en llegar a régimen pasa más tiempo en la zona donde se produce hollín,
    # y eso es distinto de "hace viajes cortos" —es el auto, no el conductor—. Solo en
    # viajes con desplazamiento y duración suficiente: en un viaje de 30 segundos el
    # cociente es ruido dividido por cero.
    warm_up = out["moving"] & out["trip_duration_min"].ge(thr["min_warmup_min"])
    out["warmup_rate_c_per_min"] = (out["engine_temp_amplitude"] / out["trip_duration_min"]).where(warm_up)
    out["temp_gain_per_km"] = (out["engine_temp_amplitude"] / out["trip_km"].where(out["trip_km"] > 0)).where(out["moving"])

    # Conjunciones. Cada término ya está en el panel por separado; la combinación no, y
    # un modelo lineal no la puede construir.
    #   - el peor viaje posible (corto, en frío y después de horas parado) necesita
    #     `soak_min`, que se calcula más abajo: se arma ahí.
    #   - oportunidad desperdiciada: el viaje daba para un ciclo completo (largo y
    #     caliente) y el sistema NO regeneró. Es la pregunta del §0 dada vuelta: no
    #     "¿cuánto regenera?" sino "¿regenera cuando puede?".
    out["opportunity_wasted"] = out["regen_opportunity"] & ~out["regen_drop"]

    # ----------------------------------------------------------------------------------
    # F3 · familia G: lo que dice la literatura de DPF y el panel no medía.
    # ----------------------------------------------------------------------------------
    # 1) **Regeneración pasiva vs. activa.** Arriba de ~250 °C de escape el hollín se
    #    quema solo, sin que el sistema tenga que inyectar combustible. Lo que se rompe
    #    primero no es la capacidad de regenerar sino la pasiva: el auto empieza a
    #    depender de ciclos activos. Acá se aproxima con una bajada CHICA del nivel
    #    (menor al umbral que cuenta como ciclo) en un viaje caliente.
    passive = (out["moving"] & out["air_regen_delta"].lt(0)
               & out["air_regen_delta"].ge(-thr["regen_drop_points"])
               & out["EngineTemperatureMax"].ge(thr["opportunity_temp_c"]))
    out["passive_regen"] = passive
    out["passive_regen_points"] = (-out["air_regen_delta"]).where(passive, 0.0)

    # 2) **Cuánto le cuesta cada ciclo.** Distancia y minutos del viaje donde ocurre la
    #    regeneración: un filtro cargado de ceniza necesita corridas más largas para
    #    completar un ciclo. Es la "duración de la regeneración" que la literatura de
    #    telemática monitorea, con la resolución que tenemos (el viaje, no el ciclo).
    out["regen_trip_km"] = out["trip_km"].where(out["regen_drop"])
    out["regen_trip_min"] = out["trip_duration_min"].where(out["regen_drop"])

    # Consumo: caída del nivel de combustible por 100 km, solo en viajes sin recarga y con distancia.
    fuel_drop = out["FuelLvlStartPc"] - out["FuelLvlEndPc"]
    valid = fuel_drop.ge(0) & out["trip_km"].ge(thr["fuel_min_trip_km"])
    out["fuel_pct_per_100km"] = (fuel_drop / out["trip_km"].where(out["trip_km"] > 0) * 100.0).where(valid)
    out["oil_life"] = out["EngineOilLifePCStart"].astype("float64")

    # Reposo desde el viaje anterior, en orden temporal dentro del vehículo.
    by_time = out.sort_values([ID_COL, "TripDatetimeStart"])
    prev_end = by_time.groupby(ID_COL, observed=True)["TripDatetimeEnd"].shift()
    soak = (by_time["TripDatetimeStart"] - prev_end).dt.total_seconds() / 60.0
    out["soak_min"] = soak.reindex(out.index).clip(lower=0)
    out["chained"] = out["soak_min"].le(thr["chained_trip_min"])

    # Familia F, la conjunción que necesitaba `soak_min`: el peor viaje posible es corto,
    # arrancado en frío y después de horas parado —el motor arranca a temperatura
    # ambiente y no llega a ningún lado—. Los tres términos ya están en el panel por
    # separado; la combinación no, y un modelo lineal no la puede construir.
    out["cold_soak_short_trip"] = (out["short_trip"] & out["cold_start"]
                                   & out["soak_min"].ge(thr["long_soak_min"]))

    # Las fracciones "solo entre viajes con desplazamiento" se materializan como
    # columnas con NaN en los idle: así el agregador `mean` las ignora sin filtros ad hoc.
    moving = out["moving"]
    for base in ("short_trip", "below_regime", "urban", "dpf_positive_delta"):
        out[f"{base}_moving"] = out[base].astype("float64").where(moving)
    for base in ("trip_km", "trip_duration_min", "speed_kmh", "EngineTemperatureAvg", "EngineTemperatureMax",
                 "engine_temp_amplitude", "CoolantTemperatureEnd", "AirRegenerationEnd"):
        out[f"{base}_moving"] = out[base].astype("float64").where(moving)

    out = out.sort_values([ID_COL, "OdometerTripEnd", "TripDatetimeStart"], ignore_index=True)
    logger.info("trips: %d filas -> %d (sin odómetro %d, km negativo %d)", n_in, len(out),
                counters["dropped_null_odometer"], counters["dropped_negative_km"])
    return out, counters


def trip_days_covered(trips: pd.DataFrame) -> pd.Series:
    """Días entre el primer y el último viaje (para tasas por día)."""
    span = trips.groupby(ID_COL, observed=True)["TripDatetimeStart"].agg(["min", "max"])
    return ((span["max"] - span["min"]).dt.total_seconds() / 86400.0).clip(lower=1.0)
