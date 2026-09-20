# Modelos candidatos para F3/F5 · catálogo, con lo ya probado marcado

**Fecha:** 2026-09-18, actualizado el 19-09. Es el menú para elegir, con el porqué de
cada idea anclado en lo que el EDA y el panel v1 mostraron. Cuando una se implementa, va
por `/mlmodel` (builder en `src/models/registry.py` + un YAML de experimento) y su
resultado a `docs/memoria/decisiones.md`.

## Lo que ya está en el registry (19-09)

**Base disponible: `baserate`, `random`, `logistic`, `logistic_l1`, `gbm`, `lgbm`.**
Cada uno con su config; los resultados salen de `python scripts/compare.py` y todos son
**provisorios** (los eventos tienen una falla conocida que la mentora va a corregir).

| modelo | idea | config | panel | PR-AUC (tasa base) | ROC | estado |
|---|---|---|---|---|---|---|
| `baserate` | piso absoluto | `exp_baserate.yaml` | v1 | 0,121 (0,125) | 0,486 | ✅ piso |
| `random` | control de métricas | — (solo en `check_setup`) | — | — | — | ✅ control |
| `logistic` | §1.2, versión cruda | `exp_logistic.yaml` | v1 | 0,159 (0,125) | 0,577 | ✅ medido |
| **`logistic_l1`** | §1.2 + selección de features | `exp_logistic_l1.yaml` | v1 | **0,181 (0,125)** | **0,617** | ✅ **mejor en v1** |
| `gbm` | §1.3 sin restricciones monótonas | `exp_gbm.yaml` | v1 | 0,166 (0,125) | 0,603 | ✅ medido |
| `lgbm` | §1.3 sin restricciones monótonas | `exp_lgbm.yaml` | v1 | 0,155 (0,125) | 0,571 | ✅ medido |
| `logistic` / `lgbm` | los mismos, panel de 118 features | `exp_logistic_v5.yaml`, `exp_lgbm_v5.yaml` | v5 | 0,170 / 0,179 | 0,605 / 0,608 | ✅ las features extra no aportan |
| `logistic`, `logistic_l1`, `gbm` | los mismos, sanos emparejados por posición | `exp_*_pos.yaml` | posmatch | 0,313 / 0,325 / 0,273 (0,228) | 0,607 / 0,587 / 0,590 | ✅ la señal sobrevive |
| `logistic`, `logistic_l1`, `gbm` | los mismos, Δ=250 + posición | `exp_*_d250pos.yaml` | delta250-posmatch | 0,212 / 0,209 / 0,198 (0,128) | 0,669 / 0,670 / 0,636 | ⚠️ **no atribuible**: el calendario quedó abierto |
| `logistic_l1` | **ablación de calendario** (`extra_prefixes: [aux_]`) | `exp_l1_v1_auxcal.yaml`, `exp_l1_d250pos_auxcal.yaml` | v1 / d250pos | 0,222 / **0,617** | 0,719 / **0,922** | 🔍 auditoría, no candidato |

**La lectura que importa:** el mejor modelo honesto da ROC 0,617 y la mejor **columna
sola** daba 0,60. Combinar 54 features suma 0,02, y duplicar el panel a 118 no suma nada:
el panel tiene una sola dimensión de señal
([f3-primer-modelo-y-emparejado-por-posicion.md](memoria/f3-primer-modelo-y-emparejado-por-posicion.md)).
Gana la regularización fuerte; el boosting no le gana a la lineal. Antes de escribir otro
builder, conviene tener presente que **el cuello de botella no es el modelo**.

**Y la alarma:** la ablación de calendario muestra ROC 0,719 en el panel v1 y **0,922** en
el Δ=250 emparejado por posición. Las `aux_` están fuera del modelo justamente por eso, y
ahora está cuantificado: cualquier panel nuevo pasa esa ablación antes de que se le crea
un número.

---


## 0 · Lo que condiciona todo (leer antes de elegir)

Lo que sabemos del panel v1 (`docs/memoria/f2-eda-revision-y-features.md`):

| hecho | consecuencia para el modelo |
|---|---|
| 53 vehículos con positivo en dev, 14 en test; 254 filas positivas | cualquier modelo con más de ~10 parámetros efectivos por feature sobreajusta; **regularización fuerte, pocas comparaciones, intervalos siempre** |
| la señal honesta es débil: ROC ≈ 0,58–0,60 con logística/LGBM de auditoría | el objetivo realista es **0,65–0,70 de ROC y 1,6–2× de lift**, no 0,9. Todo lo que pase de ahí se audita (regla 6) |
| la señal es térmica/de uso y **progresiva** en los últimos ~4.000 km (idle, no llegar a régimen, más lento) | ganan los modelos que ven **el cambio del vehículo respecto de sí mismo** y la **secuencia de cortes**, no solo la foto de una ventana |
| los sanos son censurados ("no falló todavía"), no negativos limpios | la supervivencia usa esa información; la clasificación la tira |
| la etiqueta binaria tiene un borde arbitrario (el corte en E−G−H−Δ es 0, el siguiente es 1, con features casi iguales) | **etiquetas blandas** o regresión del tiempo al evento quitan ese ruido |
| calendario y odómetro son atajos; el emparejado los cierra pero no los borra | cualquier modelo se compara **con y sin** las columnas `aux_` de calendario: si mejora mucho con ellas, aprende el atajo |
| dos mercados con distinto nivel (`msg_full` 5,45×, temperatura +8 °C) | normalizar dentro del mercado, en el `Pipeline` de cada fold |

Y el protocolo, que no cambia con el modelo:

1. Mismos folds (`splits.json`), PR-AUC out-of-fold para elegir, curva de anticipación
   vs. falsas alarmas para el pitch, intervalos por bootstrap sobre folds.
2. **Presupuesto de comparaciones**: con ~12 eventos por fold, más de 6–8 candidatos
   garantiza que "el mejor" sea ruido. Preregistrar la lista y no agregar sobre la marcha.
3. CV repetida (3 semillas de folds) para cualquier diferencia que se quiera declarar.
4. Tres auditorías obligatorias por modelo: (a) permutar `label` dentro de cada
   vehículo → PR-AUC tiene que caer a la tasa base; (b) agregar `aux_air_temp_avg`,
   `aux_regen_marker_per_1000km`, `aux_static_ProductionDay` como `feat_` → el ROC no
   tiene que saltar; (c) importancias/SHAP contra la hipótesis física (§3.2 del doc de F2).

---

## 1 · Pisos que hay que tener antes de nada (F3, costo bajo)

### 1.1 · Regla física: índice de régimen térmico

**Qué es.** Un score sin aprendizaje: suma estandarizada de 3–4 features que el perfil
alineado al evento mostró que anticipan: `feat_idle_per_1000km`,
`feat_trips_below_regime_temp_frac`, `−feat_speed_kmh_mean`, `−feat_coolant_temp_end_mean`.
Estandarización con media/desvío del train del fold (dentro del `Pipeline`).

**Por qué acá.** Es lo que un ingeniero de Ford haría sin ML y es la referencia que
un modelo tiene que superar "claramente" (plan §5). Con ROC ≈ 0,58 del LGBM de
auditoría, no es obvio que lo supere: saberlo temprano cambia el pitch.

**Cómo.** Builder `physical_rule` que implementa `fit` (guarda medias y desvíos) y
`predict_proba` (sigmoide del índice). Pesos ±1 fijos o, variante, pesos aprendidos con
una logística de 4 features. Cero hiperparámetros que barrer.

### 1.2 · Logística regularizada bien hecha — **IMPLEMENTADA PARCIALMENTE (19-09)**

> Están `logistic` (L2, C = 0,1) y `logistic_l1` (L1, C = 0,05), con `class_weight`
> balanceado y el preprocesado del `Pipeline` de `cv.py`. `logistic_l1` es el mejor
> modelo del repo: ROC 0,617. **Falta** lo que esta sección pide de más: elastic net,
> normalización por mercado y `log1p` en las features de cola larga.


**Qué es.** `LogisticRegression` con elastic-net (`saga`, `l1_ratio` ≈ 0,5) y
`class_weight="balanced"`, sobre un preprocesamiento que hoy no existe en `cv.py`:

- `log1p` en las tasas por 1.000 km (candidatas §0: linealiza `msg_*`,
  `regenerations_per_1000km`, `idle_per_1000km`);
- **normalización dentro del mercado**: z-score por `static_SalesCountry_cd` con
  estadísticos del train del fold (candidatas §2). Es un transformer propio en el
  `ColumnTransformer`;
- recorte por cuantil [p1, p99] **del train** para las colas que quedan.

**Por qué acá.** Con 254 positivos la logística compite con el boosting y es lo más
explicable del menú. La normalización por mercado es lo que más protege la
generalización según el EDA.

**Cómo.** Dos piezas: `src/training/transformers.py` (Log1pTransformer,
GroupStandardizer) y un builder `logreg_en`. El `Pipeline` de `cv.py` tiene que poder
recibir la lista de columnas a loguear y la columna de grupo desde el YAML
(`preprocessing:` nueva en el config de experimento). Es el único cambio a Track B
que casi todas las ideas de abajo reusan.

### 1.3 · LightGBM con restricciones monótonas — **SIN LAS RESTRICCIONES (19-09)**

> Están `gbm` (HistGradientBoosting) y `lgbm`, los dos con regularización fuerte y
> **sin** restricciones monótonas: ROC 0,603 y 0,571, por debajo de la logística L1.
> Con 53 vehículos con evento, más capacidad es más varianza. El mapa de monotonías
> sigue sin escribirse, y es lo que podría dar vuelta ese resultado.


**Qué es.** `LGBMClassifier` chico (`num_leaves` 4–7, `min_child_samples` ≥ 40,
`n_estimators` 200–400 con `learning_rate` 0,03, `colsample_bytree` 0,7,
`reg_lambda` ≥ 5) **con `monotone_constraints`** que codifican la física: riesgo no
decreciente en `idle_*`, `trips_below_regime_temp_frac`, `short_trip_frac_5km`,
`trips_below_30kmh_frac`, `msg_*_per_1000km`; no creciente en `speed_kmh_mean`,
`coolant_temp_end_mean`, `engine_temp_*`, `trip_distance_*`. Las demás, libres.

**Por qué acá.** Las restricciones monótonas son una regularización fuerte que no
cuesta datos y hacen que el SHAP cuente la historia del mecanismo en vez de una lista
de columnas. Con 53 eventos, un GBM libre memoriza vehículos (la auditoría mostró
`static_daysUntilSale` con 285 splits antes de sacarla).

**Cómo.** Builder `lgbm_monotone`; el vector de restricciones se arma desde un mapa
`{feature: +1/−1/0}` en el YAML, alineado al orden de columnas que salga del
preprocesador (`get_feature_names_out`). Sweep en wandb solo sobre `num_leaves`,
`min_child_samples`, `n_estimators` (3 dimensiones, ≤ 30 corridas).

---

## 2 · Las apuestas con más chance de dar un salto

### 2.1 · Supervivencia sobre el eje de km con covariables que cambian (el diferencial técnico)

**Qué es.** En vez de "¿el evento cae en [c+G, c+G+H]?", modelar **cuántos km faltan**
usando también a los sanos como censurados. Tres variantes de menor a mayor complejidad:

- **Cox penalizado por fila** (`lifelines.CoxPHFitter(penalizer, l1_ratio)`): cada fila
  del panel es una observación con duración = `time_to_event_km − G` para los eventos y
  `km observados después del corte − G` para los censurados, y evento = `label` a nivel
  vehículo. Riesgo relativo = score.
- **Cox con covariables tiempo-dependientes** (`lifelines.CoxTimeVaryingFitter`): el
  panel ya está en formato *counting process*: para cada vehículo, intervalos
  `(cut_odo, cut_odo + Δ]` con las features de esa ventana y `event=1` solo en el último
  intervalo de los vehículos con evento. Es la forma correcta de usar todos los cortes
  de un vehículo sin tratarlos como independientes.
- **XGBoost con objetivo `survival:aft`** (xgboost 3.4 pinneado): no lineal, con
  distribución `logistic` o `extreme`, `aft_loss_distribution_scale` ~ 1–2. Predice
  directamente km hasta el evento; `survival:cox` es la alternativa si solo se quiere
  ordenar.

**Por qué acá.** (1) Aprovecha la censura: los 144 vehículos sanos aportan "llegó hasta
X km sin fallar", no solo ceros. (2) Elimina H como hiperparámetro arbitrario y el borde
de la etiqueta. (3) Predice literalmente lo que el desafío pide reportar
("tiempo/kilometraje hasta el evento"). (4) Se convierte a lo que ya medimos: riesgo
acumulado a H km, `1 − S(H | x)`, entra a `lead_time_curve` como score.

**Qué hace falta en el repo.** Dos cosas chicas: el panel tiene que emitir, para los
censurados, los km observados después del corte (`aux_km_observed_after_cut`, sale
gratis en `src/data/panel.py`), y `run_cv` necesita un modo `target: survival` que
pase `(duración, evento)` al estimador en vez de `label` y pida `predict_risk_at(H)`.
Una interfaz mínima: un wrapper sklearn-like con `fit(X, y_struct)` y
`predict_proba` = riesgo a H. Métrica extra para el informe: C-index out-of-fold
(`lifelines.utils.concordance_index`).

**Trampas.** El Cox por fila trata cortes del mismo vehículo como independientes:
los errores estándar mienten, pero el ranking sirve; para intervalos usar bootstrap
por vehículo. Con AFT, la predicción para censurados lejanos es extrapolación: no
reportar km absolutos fuera de [0, 20.000].

### 2.2 · Etiquetas blandas y regresión del tiempo al evento (el arreglo barato del borde)

**Qué es.** Reemplazar el 0/1 por un objetivo continuo que decrezca suavemente con la
distancia al evento, p. ej. `y = sigmoid((G + H − tte) / τ)` con τ ≈ 500–1.000 km para los
vehículos con evento y `y = 0` para los sanos; o directamente `y = −log(tte)` y un
regresor. El score es la predicción del regresor.

**Por qué acá.** Hoy dos filas del mismo vehículo separadas 500 km reciben 0 y 1 con
features casi idénticas: eso es ruido de etiqueta puro en un problema con 254
positivos. Suavizarlo suele valer varios puntos de PR-AUC en paneles de ventana como
este, y no cambia la evaluación (la etiqueta dura sigue siendo la de las métricas).

**Cómo.** LightGBM/XGBoost con `objective="regression"` (o `"cross_entropy"` que
acepta targets en [0, 1]) y un `target: {kind: soft, tau_km: 750}` en el YAML; `run_cv`
tiene que construir el target blando desde `time_to_event_km` del train del fold y
evaluar con `label`. Cambio de ~30 líneas en `cv.py`. Barrer τ ∈ {250, 500, 1.000}.

### 2.3 · Ranking dentro de celdas emparejadas (el modelo que no puede aprender el atajo)

> **Más relevante que el 18-09.** El emparejado ya no puede cerrar odómetro, mes y
> posición a la vez (son estructuralmente incompatibles, ver
> [f3-emparejado-posicion-vs-calendario.md](memoria/f3-emparejado-posicion-vs-calendario.md)),
> y la ablación de calendario da ROC 0,92 en el panel emparejado por posición. Un
> ranker que solo compara dentro de la celda no necesita que el emparejado cierre.


**Qué es.** `LGBMRanker` con `objective="lambdarank"` donde cada *query* es una celda
(bin de odómetro × mes) y dentro de la celda hay filas positivas y sanas. El modelo
aprende a **ordenar** la fila que va a fallar por encima de la sana **con el mismo
odómetro y el mismo calendario**: los atajos son constantes dentro de la query y no
aportan nada al gradiente.

**Por qué acá.** Es la versión "por construcción" del emparejado del panel: hoy
emparejamos la distribución; el ranker empareja cada comparación. Es especialmente
robusto a cualquier deriva que no hayamos detectado todavía (firmware, campañas).

**Cómo.** `build_dataset.py` ya calcula la celda de cada fila para el emparejado:
emitirla como `aux_match_cell`. `run_cv` necesita pasar `group` (tamaños de query, con
las filas ordenadas por celda) al `fit` cuando el builder lo declare. Los scores del
ranker no son probabilidades: para la curva de anticipación alcanza (usa umbrales
sobre el score), para el Brier se calibra con Platt sobre out-of-fold.

**Trampas.** Celdas con un solo tipo de fila no enseñan nada (se descartan); con
celdas de 4.000 km × mes hay ~40, así que la señal por query es chica: usar
`lambdarank_truncation_level` alto y pocas hojas.

### 2.4 · Desvío respecto del propio vehículo (cambio, no nivel) — **IMPLEMENTADO (19-09)**

**Qué es.** Para cada corte, además del nivel de cada feature en la ventana, su
**desvío respecto del historial previo del mismo vehículo**: `z_self = (x_t − media(x_{<t}))
/ desvío(x_{<t})` usando solo cortes anteriores (nada posterior al corte: sigue siendo
hacia atrás). Variante acumulativa: **CUSUM/EWMA** por vehículo sobre 3–4 features
clave, con la estadística acumulada como feature o directamente como score (control
estadístico de procesos, sin aprendizaje).

**Por qué acá.** El EDA mostró que la señal es **progresiva** (idle 0,58 → 0,74) y que
las features que anticipan son justamente las que varían dentro del vehículo (ICC
0,3–0,6). Un modelo de nivel confunde "vehículo urbano" con "vehículo que se está
degradando"; el desvío respecto de sí mismo separa las dos cosas. Y es la explicación
más vendible en la demo: "este auto empezó a comportarse distinto a como venía".

**Cómo quedó.** En `src/features/sequence.py`, llamado desde `build_dataset.py` entre
`build_panel()` y `match_healthy_cuts()` —la serie del vehículo tiene que estar completa
cuando se calcula el desvío—. Parámetros en el bloque `sequence:` de
`configs/data/panel_v1.yaml` (`min_history`, `k`, pesos del índice).

El CUSUM **no** terminó siendo un builder: como `cv.py` selecciona por prefijo y le pasa
al estimador solo las columnas de features, un modelo sklearn no ve `vehicle_id` ni el
orden de los cortes, así que no puede acumular nada. Como el acumulador solo usa el
pasado del propio vehículo, es una feature legítima (`feat_degradation_cusum`) y
cualquier modelo del menú la consume sin cambios. Tampoco hace falta fitear `k` por fold:
el estadístico de referencia es el propio vehículo, no la cohorte, así que no hay nada
compartido entre train y validación que pueda filtrar. `h` deja de ser un parámetro del
modelo y pasa a ser el umbral de la curva de anticipación, que es donde se elige.

Chequeos en `scripts/check_setup.py` (bloque 5): que tocar el último corte no mueva
ninguna fila anterior, que el acumulador arranque en cero en cada vehículo, que sin
historial el índice sea NaN y no cero, y que sobre features sin señal una racha de 4
cortes sea rara (0,08% de las filas del panel dummy).

### 2.5 · TabPFN v2 (el que puede sorprender sin esfuerzo)

**Qué es.** Un transformer preentrenado sobre millones de datasets sintéticos que hace
inferencia bayesiana aproximada en contexto: se le pasa el train del fold como contexto
y predice el valid, sin entrenar ni tunear. Está diseñado para n < 10.000 y pocas
features, que es exactamente el panel (2.029 × 54).

**Por qué acá.** En benchmarks tabulares chicos suele ganar a GBM tuneado, y con 254
positivos el "sin tunear" es una ventaja real: no gasta comparaciones de CV. Es
sklearn-compatible, así que entra como builder sin tocar `cv.py`.

**Cómo.** Dependencia nueva (`tabpfn`, pesos locales; CPU alcanza para este tamaño,
~segundos por fold). Builder `tabpfn` con `TabPFNClassifier(n_estimators=8)`. Hay
variante de regresión (para 2.2) y la comunidad tiene extensiones de supervivencia.
Riesgo: licencia/uso del modelo en un pitch corporativo; verificar antes de mostrarlo.

---

## 3 · Ideas de segunda línea (si sobra tiempo o si las de arriba empatan)

### 3.1 · Modelo de estados ocultos por vehículo (HMM de dos estados)

Cada vehículo es una secuencia de cortes; un HMM con estados {normal, degradándose} y
emisiones gaussianas sobre 3–5 features (`idle_per_1000km`, `trips_below_regime_temp_frac`,
`speed_kmh_mean`, `coolant_temp_end_mean`) ajustado sobre los sanos (estado normal) y
las colas pre-evento (estado degradándose). Score = posterior filtrado (solo hacia
atrás) del estado degradándose. Pocos parámetros (~30), interpretable, y da "alerta
sostenida" de forma natural. Requiere `hmmlearn` y un runner por secuencias.

### 3.2 · Detección de novedad entrenada solo con sanos

`IsolationForest` o un GMM ajustado sobre ventanas sanas del train; score = anomalía.
No usa los positivos para entrenar, así que no puede memorizarlos. Suele quedar
por debajo de un supervisado, pero es un chequeo barato de cuánta señal hay "sin
etiqueta" y una alternativa honesta si el supervisado no supera a la regla física.

### 3.3 · Ensamble por rango

Promedio de rangos out-of-fold de logística, LGBM monótono y riesgo a H de
supervivencia. Con modelos débiles y distintos sesgos, el ensamble suele sumar 1–3
puntos de ROC gratis, y el bootstrap dice si es real. Se implementa como builder
`rank_ensemble` que envuelve builders del registry.

### 3.4 · Bagging por vehículo

Cualquier modelo de arriba entrenado 20 veces sobre re-muestreos **de vehículos** (no
de filas) del train del fold, promediando scores. Baja la varianza que domina con 53
eventos y da intervalos por vehículo para la demo. Costo: 20× de cómputo, que acá es
nada.

### 3.5 · Modelo bayesiano jerárquico (logística con efectos por mercado y priors horseshoe)

Logística con intercepto y pendientes por mercado (efectos aleatorios) y prior
horseshoe sobre las 53 features: selección de variables con incertidumbre honesta, y
la normalización por mercado sale del modelo. `pymc` o `numpyro` como dependencia;
5–10 minutos por fold. Vale como pieza del informe más que como ganador de PR-AUC.

---

## 4 · Lo que NO conviene intentar (y por qué)

- **GRU/LSTM/Transformer sobre viajes o cortes.** 53 eventos. Va a perder contra el
  boosting y va a costar la semana que el dashboard necesita (plan §6, stretch goal
  explícitamente condicionado a "más de cien").
- **SMOTE o re-muestreo de positivos.** Fabrica filas sintéticas a partir de cortes
  del mismo vehículo → leakage dentro del fold. Pesos de clase, nunca re-muestreo
  (plan §6, 5a).
- **Barridos grandes de hiperparámetros.** Con ~12 eventos por fold, 200 corridas de
  sweep eligen ruido. Tres dimensiones, ≤ 30 corridas, y CV repetida para el ganador.
- **Volver a meter `ProductionDay`, `daysUntilSale`, `ModelSeries`, temperatura ambiente
  o el marcador `Regenerations`.** Suben el número y no sobreviven a la primera pregunta
  del jurado (están como `aux_` para demostrarlo, no para usarlos).
- **Elegir el modelo mirando el test.** Se elige con PR-AUC out-of-fold en dev; el test
  se toca una vez, al final, con el modelo ya elegido.

---

## 5 · Orden sugerido y qué esperar

| # | idea | costo | cambio en el repo | qué esperar |
|---|---|---|---|---|
| 1 | regla física (1.1) | 1 h | builder | fija el piso real; puede quedar cerca del LGBM |
| 2 | logística EN + normalización por mercado + log1p (1.2) | medio día | transformers + `preprocessing:` en el YAML | explicable; +2–4 pp de ROC sobre la logística cruda |
| 3 | LGBM monótono (1.3) | medio día | builder + mapa de restricciones | igual o mejor que el LGBM libre, con SHAP que cuenta la historia |
| 4 | desvío respecto de sí mismo (2.4) | medio día | columnas `_zself` en `panel.py` | la que más puede mover la anticipación |
| 5 | etiquetas blandas (2.2) | medio día | `target: soft` en `cv.py` | +ruido de etiqueta afuera; barato de probar con 3 y 4 |
| 6 | Cox tiempo-dependiente / XGB-AFT (2.1) | 1–2 días | `target: survival` + km observados post-corte en el panel | el diferencial técnico del informe; km hasta el evento directo |
| 7 | TabPFN (2.5) | 2 h | dependencia + builder | sin tunear; puede ser el mejor número |
| 8 | ranker por celdas (2.3) | 1 día | `aux_match_cell` + `group` en `cv.py` | seguro contra atajos; útil como control |
| 9 | ensamble por rango + bagging por vehículo (3.3, 3.4) | medio día | builder envolvente | 1–3 pp y los intervalos para la demo |

El presupuesto de comparaciones se gasta en 1–7; 8 y 9 se usan como control y cierre.
Cada corrida al mismo wandb, con `group` por idea, para que `scripts/compare.py` arme
la tabla final sin copiar nada a mano.
