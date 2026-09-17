# Decisiones

Una entrada por decisión, la más nueva arriba. El **porqué** es la parte que
importa: sin él, el que venga la revierte sin enterarse de qué estaba resolviendo.

---

## 2026-09-16 · F2 no define W, G, H ni Δ hasta resolver la fecha del evento — DECISIÓN PENDIENTE

**Registró:** Santino · **Fase:** F2 (previo) · **Estado: abierta.** Esta entrada
no decide nada; registra un bloqueante y quién tiene que decidirlo.

En dev, `IdentificationDate == daysUntilSale` **exactamente** en 231 de los 292
vehículos con evento (79,1%), y nunca es menor. Bajo la lectura literal del anexo
7.3 —días desde producción—, esos 231 eventos caen con una mediana de **13 km** de
odómetro y el **2,3%** de su historial de viajes por delante: sin historial previo
no hay ventana W que agregar ni gap G que blanquear, así que la fila no se puede
etiquetar.

**Por qué queda abierta y no se resuelve acá:** las tres salidas posibles tienen
costos distintos y ninguna es técnica:

1. preguntarle a Ford qué es `IdentificationDate` para esos 231 vehículos;
2. etiquetar solo los 61 con fecha discriminante — quedan ~20% de los positivos, y
   la métrica del pitch pierde potencia;
3. pasar a un objetivo a nivel vehículo, resignando la curva de anticipación.

Elegir la lectura alternativa (`IdentificationDate` contado desde la venta) porque
los números quedan mejor **no** es una salida: el anexo dice "días desde
producción" sin ambigüedad, y es exactamente la clase de decisión contra la que
existe el holdout. Las dos lecturas están materializadas en el cache del EDA
(`event_odo_km` y `event_alt_odo_km`) para que la comparación se haga con datos.

**Segundo pendiente del mismo hallazgo:** `static_daysUntilSale` está en
`features.static_columns` de `configs/data/panel_v1.yaml` y, para el 79% de los
positivos, es numéricamente igual a la columna que define la etiqueta (ρ de
Spearman −0,249 con `event_observed`). Hay que auditarla con el mismo criterio con
el que se sacó `Engine`. Mientras tanto, cualquier PR-AUC alto que salga con esa
columna adentro cae de lleno en la regla 6.

**Lo que este bloqueante NO toca:** el anclaje temporal de F1 sigue en pie (IQR de
0 días re-medido en dev), la etiqueta a nivel vehículo sigue siendo la cohorte, y
el holdout dev/test no cambia.

**Detalle:** [f2-identificationdate-igual-a-venta.md](f2-identificationdate-igual-a-venta.md)

---

## 2026-09-16 · Las features de elevación y presión de neumáticos salen del alcance de F2

**Decidió:** Santino · **Fase:** F2 (previo)

Cuatro de las 31 features reservadas en `scripts/make_dummy.py::FEATURE_SPECS` se
declaran **no construibles** y salen del alcance de F2:
`feat_elevation_mean_m` y `feat_elevation_range_m` (familia C),
`feat_tire_pressure_mean` y `feat_tire_pressure_below_thr_frac` (familia D).

**Por qué:** las columnas no existen. El anexo 7.1 del PDF declara 40 columnas para
`TripSummary` y el CSV entregado trae 25; entre las 19 ausentes están las 8 de
presión de neumáticos, las 2 de elevación y las 4 de GPS. No es un problema de
nombres —el PDF avisa que las variables vienen renombradas y ese caso existe, ver
la decisión de abajo—: acá no hay ninguna columna con ese contenido. Y sin GPS
tampoco se puede inferir altitud, así que no hay sustituto posible.

**Qué se pierde:** el ángulo de altitud, que el plan §4 señalaba como "ligado
directamente al ingreso de oxígeno que menciona el título del desafío". Vale
decirlo en el informe en vez de que se note por omisión.

**Qué NO se hace:** no se edita `configs/data/raw_sources.yaml`. Ese archivo ya
declara el esquema real (F1 lo auditó contra los archivos); el desvío es contra el
diccionario oficial, que no es un config del repo.

**Pendiente:** sacar esos cuatro nombres de `FEATURE_SPECS` cuando se toque el
panel dummy, o dejarlos con un comentario que diga que no se materializan.

**Detalle:** [f2-diccionario-trips-incompleto.md](f2-diccionario-trips-incompleto.md)

---

## 2026-09-16 · La familia B se construye desde `AirRegenerationStart/End`

**Decidió:** Santino · **Fase:** F2 (previo)

Las features de DPF del plan §4 (`feat_dpf_end_slope_per_1000km`,
`feat_dpf_end_mean`, `feat_dpf_end_max`, `feat_dpf_positive_delta_frac`) se
materializan desde `trips.AirRegenerationStart/End`, **no** desde
`signals.Acumulation`.

**Por qué:** son la misma variable. Alineando el fin de cada viaje con la señal más
cercana dentro de 30 minutos, `AirRegenerationEnd == Acumulation` en el **99,80%**
de 16.324 pares (Pearson 0,9995), y `AirFilterEnd == Message` en el 99,98% con el
vocabulario de 9 niveles idéntico. Se elige `trips` porque tiene el odómetro
prácticamente completo, mientras que `OdometerValue` de `signals` es 11% nulo, y
porque el grano de viaje es mejor para cortar ventanas.

Corolario que conviene no olvidar: **no son dos features**. `acumulation_mean` y
`air_regen_end_mean` correlacionan a ρ = 0,996 sobre dev. Una fuente, no las dos.

**Lo que esto corrige:** `f1-datos-reales.md` dice "no hay columna de DPF". Hay
dos, con otro nombre y en las dos tablas. La familia B pasa de parcial a completa.

**Lo que queda como hipótesis, no como decisión:** que esas columnas *sean* el
`DieselParticulateFilterStart/End` del anexo 7.1. Es la explicación que mejor
encaja —ocupan esa posición faltante, el anexo describe `Acumulation` como
"acumulación en el filtro de aire [%]", el PDF avisa que las variables vienen
renombradas, y la serie carga y se descarga como un filtro de partículas— pero no
hay confirmación de Ford, y la escala llega a 95 y no a 100. En el informe se
nombra como "nivel de acumulación del filtro (0–95)", que es siempre correcto, y no
como "% de saturación del DPF", que hay que poder defender.

**Detalle:** [f2-columnas-airfilter-airregeneration.md](f2-columnas-airfilter-airregeneration.md)

---

## 2026-09-16 · Test del 20% congelado antes de F2, no después

**Decidió:** Santino · **Fase:** F2 (previo)

Los 1081 vehículos se parten en **dev (864) y test (217)** antes de construir el
panel. El test se guarda en `data/processed/test_split.json` y no se toca hasta
tener el modelo elegido. La CV de 5 folds de `make_splits()` corre **dentro de
dev**.

El panel de F2 se materializa con **los 1081**, y el recorte es por máscara
(`test_split_masks`), no por un panel más chico: así la evaluación final no obliga
a regenerar el panel con otros vehículos —que es cuando aparecen las diferencias
silenciosas entre lo que se entrenó y lo que se midió— y la protección la da el
código, que es auditable, en vez de la ausencia del archivo, que no se puede
verificar.

**Por qué ahora:** todo lo que viene después de este punto —qué features agregar
en la ventana, qué W, G y H, dónde poner el umbral de operación— son decisiones
que se toman mirando los datos. Si el test se recorta al final, esas decisiones ya
se tomaron con el test adentro, y el número del pitch queda optimista sin que
nadie pueda decir por cuánto. La auditoría de esquema y calidad (nulos, tipos,
duplicados, cobertura del join) sí se hizo sobre el dataset completo: no depende de
qué vehículo caiga de cada lado, así que no contamina nada.

**Por qué 20% y no 10%:** la métrica de portada se estima sobre los vehículos con
evento del test. Con 73 (20%), una tasa de detección de 0,7 tiene un error
estándar de 0,054 —±11 pp de intervalo—; con 36 (10%) pasa a ±15 pp, que es más
ancho que la diferencia entre modelos que esperamos medir. Del otro lado, dev
conserva 292 vehículos con evento: ~58 por fold, suficiente para que ningún fold
quede sin positivos.

**Por qué no estratificar por motor, modelo y país:** la estratificación es solo
por `event_observed`. Se reporta cómo quedaron repartidas las tres estáticas
(peor desvío: `MODEL_3`, 4,2 pp) pero no se fuerza: con 1081 vehículos, cada
restricción extra gasta grados de libertad para arreglar algo que ya quedó
razonable. Y elegir la semilla hasta que el cuadro quede lindo es la misma trampa
que el holdout viene a evitar.

**Alternativa descartada:** vivir solo con la CV. Se reusa decenas de veces
durante la selección de modelo, y para la décima comparación la métrica
out-of-fold ya está sobreajustada al procedimiento. Sirve para elegir; no sirve
para reportar.

**Cómo se aplica:** `configs/data/test_split.yaml` (semilla y `test_size`) +
`src/eval/splits.py::make_test_split` / `test_split_masks`. Se congela una vez con
`python scripts/make_test_split.py`; el script no pisa un holdout ya congelado sin
`--force`.

El recorte a dev lo hace **`scripts/train.py` solo**, en `select_dev()`, antes de
armar los folds: ninguna corrida ve el test sin que alguien lo pida explícitamente.
Y la clave `splits.test_split` es **obligatoria en todo YAML de experimento** —con
el path del holdout, o `null` explícito para el panel dummy—: si falta, `train.py`
levanta un `KeyError` y no entrena.

**Por qué obligatoria y no con un default:** olvidarse del holdout no rompe nada.
No hay excepción, ni warning que alguien vaya a leer en 300 líneas de log: sale un
PR-AUC más alto y la corrida entra a la tabla comparativa como si fuera buena. El
único momento en que ese error se puede detectar es antes de correr, así que el
config tiene que declarar la intención aunque sea para decir "este panel no tiene
holdout".

**Detalle:** [f2-union-y-holdout-dev-test.md](f2-union-y-holdout-dev-test.md)

---

## 2026-09-15 · Los 13 clones se colapsan a 13 vehículos

**Decidió:** Gonzalo · **Fase:** F1

Los 26 `VehicleCode` duplicados se colapsan a 13 `vehicle_id`, quedándose con el
código menor de cada par. La etiqueta se resuelve a `failed`.

**Por qué:** son el mismo auto con dos códigos y sus viajes son idénticos
(Jaccard 1.000). Con los dos códigos vivos, uno puede caer en train y el otro en
validación sin que `src/eval/splits.py` note nada: la regla 2 se cumple en el
código y se viola en los hechos. La etiqueta va a `failed` porque `not_failed` es
ausencia de registro, no evidencia de que el vehículo esté sano.

**Alternativa descartada:** tirar los 26 códigos. Son 13 vehículos con evento de
365 — el 3,6% de la señal positiva. Caro para un problema que se arregla mapeando.

**Cómo se aplica:** `configs/data/vehicle_dedupe.yaml` (lista congelada) +
`src/data/dedupe.py`. Se corre una vez al construir el panel, antes de cualquier split.

**Detalle:** [f1-clones-vehiculos.md](f1-clones-vehiculos.md)

---

## 2026-09-15 · `Engine` queda afuera del set base de features

**Decidió:** Gonzalo · **Fase:** F1

`static_Engine` no entra al modelo base. Queda declarado en
`features.static_excluded` de `configs/data/panel_v1.yaml`.

**Por qué:** `ENG_3` es el 36,1% de los vehículos sanos y el 0% de los que tienen
evento. El modelo puede clasificar 268 vehículos como sanos por su motor y acertar
siempre — dentro de este dataset. Es cómo se muestrearon los datos, no física del
motor, y no sobrevive a la primera pregunta del jurado.

**Qué queda abierto:** medirlo como ablación explícita (otro YAML) si alguien
quiere cuantificar el aporte real. `ModelSeries` y `SalesCountry_cd` sí entran:
están en las dos cohortes con proporciones comparables.

**Detalle:** [f1-sesgo-eng3.md](f1-sesgo-eng3.md)

---

## 2026-09-15 · Las cohortes se declaran como `parts`, no como tablas separadas

**Decidió:** decisión de implementación tomada al reescribir el contrato de
fuentes; queda a revisión en el PR de F1.

Los seis CSV se declaran como tres tablas lógicas con dos `parts` cada una. El
loader concatena y agrega la columna `cohort`.

**Por qué:** la cohorte es la etiqueta, no una dimensión de los datos. Si río abajo
hay seis tablas, cada consumidor tiene que acordarse de unirlas y de que `cohort`
no es una feature. Con `parts`, el contrato de datos sigue teniendo tres tablas y
un solo lugar donde se decide qué significa la cohorte.
