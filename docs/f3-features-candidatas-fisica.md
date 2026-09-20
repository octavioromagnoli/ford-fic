# Features candidatas por mecanismo físico · nada implementado

**Fecha:** 2026-09-19 · **Estado: ninguna está implementada.** Es el menú de lo que sigue
después de que la familia E (secuencia) diera negativo
([`memoria/f3-secuencia-zself-cusum.md`](memoria/f3-secuencia-zself-cusum.md)). La
decisión fue **trabajar features útiles antes que modelos**: con ROC ≈ 0,58–0,60 de las
features actuales, un modelo mejor no va a rescatar un panel que no mide el mecanismo.

Cada candidata se mide **primero** con el perfil alineado al evento de
`scripts/eda_gaps.py` (bloque 2: P(evento > sano) por tramo de km previo al evento) y
recién si sobrevive se agrega a `configs/data/features_v1.yaml`. Y todas pasan la
auditoría de [`memoria/f3-posicion-en-la-serie.md`](memoria/f3-posicion-en-la-serie.md):
correlación con la posición del corte, y separación dentro de estratos de posición.

---

## 0 · La contradicción que hay que resolver primero

El EDA midió, en los tramos previos al evento, P(vehículo con evento > sano):

| | 4–8k antes | 2–4k antes | 0–2k antes |
|---|---|---|---|
| `air_regen_end_mean` (nivel del DPF) | 0,35 | 0,35 | **0,31** |
| `regen_drops_per_1000km` (frecuencia de regeneración) | 0,43 | 0,43 | 0,42 |

**Los que fallan terminan los viajes con el filtro menos cargado y regeneran menos.** Es
consistente en los tres tramos, así que no es ruido. Va en contra de la intuición física
(un filtro que se obstruye debería leer *más* carga y pedir *más* regeneraciones), y toda
la familia B está construida sobre esa intuición. Tres explicaciones posibles:

1. **El nivel del DPF es una variable controlada.** El sistema regenera para mantenerlo
   bajo, así que medir el nivel dice qué tan bien está actuando el controlador, no qué tan
   mal está la pieza. Es el error clásico de medir la salida de un lazo cerrado en vez del
   esfuerzo del lazo. **Es la hipótesis más probable y la que organiza todo el §1.**
2. **El sensor se degrada con la pieza** y subestima la carga justo en los peores.
3. **El evento no es obstrucción del DPF.** `IdentificationDate` marca algo que Ford
   identificó y que no sabemos qué es —ya figura como pregunta abierta en
   [`memoria/f2-eda-revision-y-features.md`](memoria/f2-eda-revision-y-features.md) §5—.
   Si fuera un reemplazo de sensor o una intervención de taller, la hipótesis de
   saturación se cae entera.

Distinguir (1) de (2) y (3) es lo que más valor tiene ahora, y el §1 es el instrumento.

---

## 1 · Esfuerzo de control, no nivel (la más importante)

Si el nivel es una variable controlada, lo informativo es **cuánto trabajo le cuesta al
sistema mantener el filtro limpio**. Tres cosas que hoy no están medidas:

### 1.1 · Eficiencia de cada regeneración

`AirRegenerationStart − AirRegenerationEnd` en cada viaje donde hay regeneración (caída
> 5). Hoy existen `regen_start_level_mean` y `regen_residual_mean` **por separado**, nunca
la diferencia. Un filtro que envejece remueve menos por ciclo. Features: media de la
caída, y su tendencia dentro de la ventana (`half_mean_delta`).

### 1.2 · Residuo que no se va

La ceniza es irreversible y se acumula; el hollín se quema. `regen_residual_mean` existe
como **nivel**, pero falta su **pendiente a lo largo de las regeneraciones sucesivas de la
ventana**. Esa pendiente es la firma de envejecimiento real del DPF, es intra-vehículo
—donde el EDA dice que está la señal— y a diferencia del CUSUM **no es monótona en la
posición**, así que no hereda el atajo.

### 1.3 · Distancia entre regeneraciones encogiéndose

`distance_between_regen_trend` ya existe en la familia B pero nunca se midió contra la
etiqueta por separado. Un plant que necesita intervención más seguido es la definición de
degradación en un lazo cerrado. Costo: cero, es mirar una columna que ya está.

---

## 2 · Regeneraciones abortadas, desde `trips` en vez del mensaje

El EDA descartó `Stopped Cleaning Automatically` porque es **1 mensaje de cada 100.000**.
Pero una regeneración interrumpida se puede derivar del viaje: el viaje **termina con
`AirRegeneration` bajando** (o sea, en pleno ciclo) **y todavía por encima de un umbral**.

Es literalmente el mecanismo de "los viajes cortos rompen el DPF": arranca a limpiar, el
conductor llega, el ciclo se corta, el próximo arranca desde peor. Si aparece con
frecuencia medible, es la mejor feature del proyecto: une la hipótesis de uso con la de
pieza y se explica en una frase en el pitch.

**Primero hay que medir si existe**: cuántos viajes terminan en ese estado, en cuántos
vehículos. Si es 1 de cada 100.000 como el mensaje, se descarta y se anota.

---

## 3 · Dosis de km fríos, no fracción de viajes fríos

Hoy **todas** las fracciones térmicas son *fracción de viajes* (`idle_frac`,
`trips_below_regime_temp_frac`, `short_trip_frac_5km`). El hollín se acumula en función de
los **kilómetros fríos**, no de la cantidad de viajes fríos. Un vehículo con 40 viajes de
1 km en frío y otro con 3 viajes de 20 km en frío tienen fracciones opuestas y dosis
parecidas.

La versión correcta del mecanismo es una **dosis ponderada por km**: km recorridos por
debajo de régimen, por cada 1.000 km de ventana. Es la misma columna cruda
(`EngineTemperatureMax` / `below_regime`) con otro agregador, así que el costo es una
línea en `features_v1.yaml` más un agregador `km_weighted_mean` en
`src/features/windows.py`.

---

## 4 · Temperatura condicionada por el largo del viaje

La temperatura alta es ambigua: puede ser "bien, llegó a régimen" o "mal, se
sobrecalienta". La ambigüedad se rompe **condicionando por el viaje**:

- un **viaje largo** (> 15 km) que no llega a régimen es un **defecto** (termostato,
  refrigeración, sensor);
- un **viaje corto** (< 5 km) que no llega es **uso normal**.

Hoy `trips_below_regime_temp_frac` mezcla los dos casos en una columna y se cancelan
parcialmente. Partirla en dos separa "el auto está roto" de "al auto lo usan mal", que son
dos historias distintas —y las dos sirven para el pitch, pero por motivos opuestos—.

---

## 5 · Oportunidad de regenerar

Una regeneración necesita una corrida sostenida, caliente y con carga. Si el vehículo
nunca le da esa ventana al sistema, el hollín se acumula **aunque el resto del uso sea
impecable**. Dos features:

- fracción de km en viajes lo bastante largos y calientes como para permitir un ciclo
  completo (el umbral sale de mirar la duración/temperatura típica de las regeneraciones
  que **sí** se completan en los datos, no de un número inventado);
- `km_since_last_regen_opportunity`: hace cuánto que el auto no tiene una corrida así.
  **Ojo**: es un "hace cuánto que", así que entra en la lista de riesgo de
  [`memoria/f3-posicion-en-la-serie.md`](memoria/f3-posicion-en-la-serie.md) y hay que
  auditarla por estratos antes de creerle.

Es la versión llevada al límite de "los usos de poco tiempo hacen trabajar al auto en
frío y lo rompen", y a diferencia de las fracciones térmicas, apunta a la **consecuencia**
(el sistema no puede limpiarse) y no solo a la causa.

---

## Orden sugerido

| # | candidata | costo | por qué primero |
|---|---|---|---|
| 1 | eficiencia y residuo de regeneración (§1.1, §1.2) | ~1 h | resuelve la contradicción del §0, que condiciona todo lo demás |
| 2 | tendencia de distancia entre regeneraciones (§1.3) | 10 min | ya está construida, solo falta medirla |
| 3 | regeneraciones abortadas (§2) | ~1 h | si existe, es la mejor feature del proyecto; si no, se descarta rápido |
| 4 | dosis de km fríos (§3) | ~1 h | arregla un error de unidad en 17 features de la familia A |
| 5 | temperatura condicionada por largo de viaje (§4) | ~1 h | separa defecto de uso |
| 6 | oportunidad de regenerar (§5) | ~2 h | la más vendible, pero la que más supuestos mete |

1 a 3 son una sola pasada sobre `trips`. Ninguna requiere reconstruir el panel para
**mirarla**: se miden con el perfil alineado al evento y recién después se agregan.
