# Decisiones

Una entrada por decisión, la más nueva arriba. El **porqué** es la parte que
importa: sin él, el que venga la revierte sin enterarse de qué estaba resolviendo.

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
