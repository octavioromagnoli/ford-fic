# F3 · El reloj del evento y la ventana del registro

**22-09-2026 · solo dev** (290 vehículos, 60 con evento; test sin mirar) · rama
`feat/landmark-post-venta` · Fase 1 del cure model.

```bash
python scripts/audit_event_clock.py --config configs/data/event_clock.yaml   # --refresh relee los crudos
```

Deja un CSV por bloque y `summary.json` en `experiments/audit_event_clock/`. Pasa a código del
repo la evidencia que motivó el panel de hitos post-venta, que antes vivía en scripts de
scratch. El script imprime cada número al lado del que había medido el scratch (bloque
`expected` del YAML); las diferencias están en el §9.

**Fechas** (`src/data/anchor.py`):
- producción = origen + `ProductionDay`. El origen congelado es el día 20107, y al re-estimarlo
  sobre dev da lo mismo (IQR 0 d);
- venta = producción + `daysUntilSale`;
- evento = producción + `IdentificationDate`.

Así la edad al evento es exactamente `IdentificationDate`, y los días post-venta al evento son
`IdentificationDate − daysUntilSale`.

## Resumen

1. **El evento se ordena por días, no por km.** sd(log) del momento del evento: 0,24 en edad
   contra 0,97 en odómetro.
2. **El registro de eventos tiene una ventana de calendario**, del 03-09-2025 al 11-03-2026.
   Afuera hay 100–180 autos por mes en la edad típica del evento y ningún evento. **Un sano
   está observado para el evento solo dentro de esa ventana**, no en toda su telemetría.
3. **Dentro de la ventana, el riesgo lo ordenan los días desde la venta.** No lo ordenan ni la
   edad ni el calendario (Poisson sobre autos-día, p = 0,014).
4. **No hay trayectoria previa.** El exceso de idle y de viajes fríos de los fallados es
   máximo lejos del evento y se achica al acercarse.
5. **Hay un rasgo temprano y estable.**
   - Viajes bajo régimen contra la flota: AUC 0,65–0,68 desde los primeros 30 días post-venta;
     nulo estratificado p = 0,0005.
   - Es débil en la cola.
   - A 30 días, la cola son autos quietos: vendidos, pero casi sin km.
6. **El 28% de los horizontes sanos del panel v1 no cae entero dentro de la ventana.**

## 1 · Escala (H1)

Criterio de Kordonsky & Gertsbakh (1993): la mejor escala de tiempo es la que minimiza la
dispersión del momento del evento entre los que fallan.

Muestra: 57 fallados con telemetría desde menos de 300 km y más de 500 km recorridos antes
del evento. Si la telemetría arranca tarde, los acumulados quedan truncados.

| escala | sd(log) | mediana |
|---|---|---|
| edad (días desde producción) | **0,240** | 257 d |
| días desde la venta | 0,438 | 168 d |
| viajes bajo régimen | 0,484 | 212 |
| cantidad de idle | 0,502 | 305 |
| viajes | 0,536 | 767 |
| minutos de idle | 0,547 | 1.151 |
| horas de motor | 0,810 | 280 h |
| odómetro | 0,966 | 9.080 km |
| km bajo régimen | 1,002 | 55 km |

La diferencia odómetro − edad es **+0,72, IC95 [+0,56, +0,86]** (bootstrap por vehículo,
2.000 réplicas). Toda escala de conteo (viajes, idle, viajes fríos) concentra más que los km.

Los días post-venta al evento, sobre los 60 eventos de dev: mediana **163**, IQR 105–204,
máximo 268.

**Ojo con la fila "días desde la venta".** Da peor que la edad, pero sd(log) no es invariante
a correr el origen: con un origen más cercano al evento, la media es más chica y el log
dispersa más. En días absolutos, las dos tienen un desvío de ~60 d (57 contra 68 sobre los
60 eventos). H1 separa días de km, no producción de venta. Eso lo decide el §4.

## 2 · El uso mueve los km al evento, no los días (H2)

Uso en los primeros 2.000 km. Muestra: 47 fallados con el evento después de 2.500 km.

| uso temprano | ρ con km al evento | ρ con días al evento |
|---|---|---|
| km/día | **+0,48** (p = 0,001) | −0,07 (p = 0,66) |
| minutos de idle / 1.000 km | −0,53 (p < 0,001) | −0,15 (p = 0,31) |
| idle / 1.000 km | −0,52 (p < 0,001) | −0,06 (p = 0,69) |
| horas de motor / 1.000 km | −0,40 (p = 0,006) | −0,10 (p = 0,50) |

El "mucho idle ⇒ falla antes en km" es km/día disfrazado. **Toda anticipación medida en km
es, en buena parte, km/día × meses.**

## 3 · La ventana del registro (H3)

"En riesgo" quiere decir que:
- la edad cae dentro del rango de edades al evento de dev (116–336 días) en algún día del mes;
- hay telemetría ese mes o después;
- no hubo evento antes.

| mes | en riesgo | eventos | | mes | en riesgo | eventos |
|---|---|---|---|---|---|---|
| jun-25 | 19 | 0 | | ene-26 | 177 | 12 |
| jul-25 | 61 | 0 | | feb-26 | 186 | 14 |
| ago-25 | 99 | 0 | | mar-26 | 181 | 4 (hasta el 11) |
| sep-25 | 126 | 4 | | abr-26 | 177 | 0 |
| oct-25 | 148 | 6 | | may-26 | 158 | 0 |
| nov-25 | 160 | 11 | | jun-26 | 138 | 0 |
| dic-25 | 165 | 9 | | jul–sep-26 | 115 → 76 | 0 |

Los eventos de dev van del 03-09-2025 al 11-03-2026, así que ninguno cae fuera de la ventana
congelada en el YAML (`event_window`). Ninguno cae antes de la venta, y 2 caen después del
último viaje del vehículo.

De abril a septiembre de 2026 hay 100–180 autos en riesgo por mes y ningún evento. Esa caída
a cero no es física: **es el borde de la extracción**.

**Lo que se sigue:**
- **la censura de un sano es el fin de su exposición dentro de la ventana, no su último
  viaje**;
- un sano sin exposición en la ventana no informa nada sobre el evento;
- esto explica el sesgo de `ProductionDay` del panel v1 (los producidos tarde casi no
  estuvieron expuestos), que `panel_v1.yaml` ya documentaba sin saber de dónde venía.

## 4 · Riesgo dentro de la ventana y origen del reloj (H4, H4b)

**Tasa por 100 autos-mes dentro de la ventana.** Los autos-día cuentan desde la producción y
llegan hasta el último viaje o el evento.

| días post-venta | autos-mes | eventos | tasa | | edad (días) | autos-mes | eventos | tasa |
|---|---|---|---|---|---|---|---|---|
| antes de la venta | 540 | **0** | 0,0 | | 0–120 | 542 | 2 | 0,4 |
| 0–60 | 271 | 3 | 1,1 | | 120–180 | 303 | 9 | 3,0 |
| 60–120 | 247 | 17 | 6,9 | | 180–240 | 301 | 14 | 4,6 |
| 120–180 | 200 | 13 | 6,5 | | 240–300 | 202 | 20 | 9,9 |
| 180–240 | 106 | 17 | 16,1 | | 300–360 | 86 | 13 | 15,1 |
| 240–300 | 43 | 8 | 18,4 | | | | | |

**Que no haya ningún evento antes de la venta es en parte convención del registro, no física.**
El universo ya excluye los eventos con `IdentificationDate == daysUntilSale`, y
`IdentificationDate < daysUntilSale` no pasa nunca (`src/data/usable.py`).

Edad y días post-venta crecen juntos, así que esta tabla no decide el origen. La **regresión
ingenua entre fallados** tampoco lo decide:
- la edad al evento no depende de `daysUntilSale` (pendiente 0,01 ± 0,21);
- los días post-venta sí (ρ = −0,52).

Eso parecería decir "reloj de producción", pero la ventana trunca la fecha del evento: entre
los eventos observados, el que se vendió tarde solo pudo fallar temprano.

Lo que maneja el truncamiento es modelar el riesgo sobre los autos-día. **H4b** ajusta un
Poisson sobre 26.580 autos-día post-venta dentro de la ventana, con 58 eventos y
ρ(edad, post-venta) = 0,80. Edad y post-venta difieren en `daysUntilSale`, así que se pueden
separar:

| términos | modelo base | agrega | LR | p |
|---|---|---|---|---|
| lineal | edad | días post-venta | 6,06 | **0,014** |
| lineal | días post-venta | edad | 0,00 | 0,96 |
| lineal | edad + calendario | días post-venta | 5,63 | 0,018 |
| lineal | días post-venta + calendario | edad | 0,12 | 0,73 |
| cuadrático | edad | días post-venta | 6,72 | 0,035 |
| cuadrático | días post-venta | edad | 2,18 | 0,34 |

El calendario solo no mejora al modelo nulo: logL −413,0 contra −413,4.

**Conclusión:** el reloj del riesgo arranca en la venta. La evidencia es moderada (58 eventos).
El panel de hitos usa días post-venta tanto para los hitos como para la latencia.

## 5 · No hay trayectoria previa (H5)

Muestra: fallados, viajes post-venta anteriores al evento. Se miden como exceso contra los
sanos del mismo mercado × mes × etapa (pre/post venta), promediado por vehículo.

| días antes del evento | vehículos | exceso de idle | exceso de viajes bajo régimen |
|---|---|---|---|
| > 180 | 25 | +0,28 | +0,39 |
| 120–180 | 38 | +0,22 | +0,21 |
| 90–120 | 44 | +0,18 | +0,17 |
| 60–90 | 52 | +0,14 | +0,07 |
| 30–60 | 56 | +0,08 | +0,06 |
| 0–30 | 55 | +0,12 | +0,04 |

El exceso es **máximo lejos del evento**, o sea en los primeros meses post-venta, y se achica
al acercarse. Es lo contrario de una degradación. Coincide con lo que encontró la rama
`feat/zself-cusum`: el desvío contra el propio pasado no separa. **El *cuándo* no se modela
con features.**

## 6 · El rasgo temprano (H6)

**Cómo se mide.** Viajes post-venta hasta el hito, desviados contra los sanos post-venta del
mismo mercado × mes. El rasgo es el promedio por vehículo. Entran los vehículos vivos en el
hito y con ≥ 15 viajes.

**Población comparable:**
- los 60 fallados;
- 119 sanos producidos hasta el día 215 (el del último fallado) y seguidos ≥ 336 días (la
  mayor edad al evento).

| hito | fallados · sanos | AUC viajes bajo régimen [IC95] | AUC idle | AUC de −km (piso de uso) | ρ(bajo régimen, km) |
|---|---|---|---|---|---|
| 30 d | 42 · 102 | **0,678** [0,59, 0,76] | 0,626 | 0,572 | −0,63 |
| 60 d | 55 · 109 | 0,672 [0,58, 0,75] | 0,626 | 0,593 | −0,57 |
| 90 d | 47 · 111 | 0,646 [0,55, 0,73] | 0,620 | 0,565 | −0,39 |
| 120 d | 40 · 112 | 0,654 [0,56, 0,75] | 0,634 | 0,551 | −0,35 |
| 180 d | 27 · 114 | 0,664 [0,55, 0,78] | 0,614 | 0,535 | −0,32 |

- **No mejora con más días.** Es un rasgo del vehículo, no un síntoma que crece.
- **Le gana al piso de uso** (−km hasta el hito), aunque está correlacionado con él.

**No es la fecha de producción.** A 60 d, el combo (z(idle) + z(bajo régimen), pesos
unitarios) da AUC 0,653:
- ρ con `ProductionDay` +0,05;
- ρ con `daysUntilSale` −0,07.

| estrato | fallados · sanos | AUC combo | AUC bajo régimen |
|---|---|---|---|
| tercil de producción 1 (temprano) | 24 · 31 | 0,718 | 0,698 |
| tercil 2 | 21 · 33 | 0,569 | 0,639 |
| tercil 3 (tarde) | 10 · 45 | 0,738 | 0,736 |
| CNTRY_3 | 12 · 49 | 0,718 | 0,760 |
| CNTRY_4 | 43 · 60 | 0,617 | 0,645 |

Nulo por permutación de la etiqueta dentro de tercil de producción × mercado (2.000
permutaciones): 0,509 ± 0,043 contra 0,653, **p = 0,0005**.

**El límite: la cola, contra todos los sanos de dev**, con un umbral que alerta al 5% de los
sanos:

| hito | fallados · sanos | AUC | detectados | anticipación mediana | km hasta el hito de los detectados · fallados · sanos |
|---|---|---|---|---|---|
| 30 d | 42 · 200 | 0,597 | 4 (9,5%) | 102 d · 3.972 km | **1,5** · 709 · 938 |
| 60 d | 55 · 216 | 0,630 | 5 (9,1%) | 119 d · 5.278 km | **2,0** · 1.813 · 2.506 |
| 90 d | 47 · 220 | 0,630 | 2 (4,3%) | 151 d · 6.385 km | **2,5** · 3.925 · 4.199 |

**Los que alertan son autos quietos**: vendidos, pero con 1,5–2,5 km recorridos en sus
primeros 30–90 días. Sus viajes son casi todos idle, así que el exceso de idle los pone
arriba del ranking.

Autos con < 100 km post-venta hasta el hito (contra todos los sanos):

| hito | autos quietos | de ellos, fallados | tasa quietos · resto | AUC combo con todos → sin quietos | sin quietos: bajo régimen · idle |
|---|---|---|---|---|---|
| 30 d | 62 | 16 | 0,26 · 0,14 | 0,597 → 0,547 | **0,641** · 0,515 |
| 60 d | 30 | 8 | 0,27 · 0,20 | 0,630 → 0,638 | **0,683** · 0,602 |
| 90 d | 16 | 2 | 0,13 · 0,18 | 0,630 → 0,656 | **0,668** · 0,608 |

**La lectura:**
- **el rasgo físico que sobrevive es el de los viajes bajo régimen entre los que se mueven**
  (0,64–0,68 sin autos quietos);
- **a 30 días, el idle es casi entero "el auto no se movió"**: sin quietos, cae a 0,52;
- que un auto quieto tenga más riesgo (0,26 contra 0,14 a 30 d) puede ser real, o puede ser
  que `daysUntilSale` no sea la entrega al cliente (pregunta 6 del §10).

## 7 · Consecuencia para el panel v1 (H9)

Se toman las 1.062 filas sanas de dev (118 vehículos). En cada una, el horizonte
`[c+G, c+G+H]` se traduce a fecha con los viajes del vehículo:

| horizonte | filas sanas |
|---|---|
| entero dentro de la ventana | 71,8% |
| sale después del 11-03-2026 | 17,9% |
| entero después del 11-03-2026 | 1,6% |
| empieza antes del 01-09-2025 | 8,7% |

El panel v1 trata como negativas verificadas filas cuyo horizonte el registro no cubría del
todo. El 1,6% no estaba cubierto en ningún día. **Esto no se arregla en esta rama**: queda
anotado en `decisiones.md` (22-09) como aviso para cualquier relectura de `results/`.

## 8 · Factibilidad del panel de hitos

Parámetros: G = 30 d, H = 240 d y ≥ 15 viajes post-venta hasta el hito. La exposición se mide
en la ventana, desde el hito + G.

| hito | en riesgo | eventos (todos en el horizonte) | sanos | exposición ≥ 60 d | ≥ 90 d | ≥ 120 d | 1–119 d | sin exposición |
|---|---|---|---|---|---|---|---|---|
| 30 d | 240 | 43 | 197 | 86 | 64 | **52** | 68 | 77 |
| 60 d | 250 | 46 | 204 | 68 | 56 | **38** | 69 | 97 |
| 90 d | 237 | 38 | 199 | 57 | 38 | **28** | 62 | 109 |

Descartes por hito (30 / 60 / 90 d):
- 6 vehículos sin `daysUntilSale`;
- 6 / 15 / 23 no llegan al hito;
- 3 / 12 / 20 tienen el evento antes del hito o dentro del gap;
- 35 / 7 / 4 tienen pocos viajes.

km/día post-venta: p25 36, p50 57, p75 83. **500 km ≈ 9 días.**

Con `E_min` = 90 días, que es el primario del prompt, quedan **64 / 56 / 38 negativos
resueltos** por hito. Con 40–60 negativos, alertar al 5% son 2 o 3 autos.

## 9 · Diferencias con la exploración de scratch

Ninguna cambia una conclusión.

- **Producción = anclaje, no primer viaje.** Las fechas de un vehículo se mueven a lo sumo
  unos días (IQR 0 d, rango 6 d). Los conteos por mes y la factibilidad cambian en ±1.
- **H4: los autos-día cuentan desde la producción.** El scratch contaba desde el 01-09-2025
  aunque el auto no existiera. Quedan 540 autos-mes pre-venta en vez de 748, siempre con cero
  eventos. Un evento cambia de tramo: 0–60 d pasa de 1,5 a 1,1.
- **H6: confusor y nulo exigen que el vehículo esté vivo en el hito.** El scratch no lo
  exigía. El combo da 0,653 en vez de 0,645, y el tercil de producción del medio baja a 0,57.
- **Detección a 60 d: 5/55 en vez de 3/55.** El umbral del 5% queda en ~11 sanos, así que 2
  autos de diferencia están dentro del ruido.
- **Anticipación en km:** ahora es odómetro del evento − odómetro en el hito. El scratch
  multiplicaba km/día por días, y con autos quietos eso da ~0.
- **Nuevo:** H4b (origen del reloj) y la tabla de autos quietos.

## 10 · Qué implica y qué queda abierto

**Para el preregistro (Fase 2):**
- **Reloj post-venta** para hitos y latencia: respaldado por H4b, no solo por H4.
- **Censura = exposición dentro de `event_window`.**
- **Autos quietos, a declarar antes de mirar D1/D2.**
  - `idle_per_1000km` no está definido con ~0 km. Con `min_cell_km: 100`, esos meses quedan
    en NaN y se imputan por la mediana: el vehículo pierde el valor de esa feature.
  - `speed_kmh_mean` y `coolant_temp_end_mean` (entre los que se mueven) también quedan en NaN.
  - `trips_below_regime_temp_frac` (media sobre todos los viajes) sí lo conserva.
  - Hay que decidir si eso es lo que se quiere, porque cambia quién queda arriba del ranking
    a 30 días.
- **El techo realista de la detección** con este rasgo es de un dígito alertando al 5% de los
  sanos. Un cure model encima no puede discriminar mejor que su incidencia.

**Preguntas para Ford o la mentora** (no bloquean):
1. ¿Qué es `IdentificationDate` en la práctica: service, inspección, campaña?
2. ¿Cuál es la ventana real de extracción del registro de eventos? En dev es del 03-09-2025
   al 11-03-2026.
3. ¿Se corrigió el error de SQL de la query de eventos (rama `feat/f3-features-regeneracion`)?
   Si las fechas cambian, se rehace esta fase antes que nada.
4. ¿Cuál es la prevalencia real en la flota, para calibrar probabilidades absolutas?
5. ¿Hay fechas de visita al taller? Permitirían modelar la identificación con censura por
   intervalo.
6. **(nueva)** ¿`daysUntilSale` es la entrega al cliente o la registración? El 26% de los
   vehículos de dev que llegan al hito de 30 días recorrió menos de 100 km desde la "venta".
