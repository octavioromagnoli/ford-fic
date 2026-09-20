# Modelos candidatos para F3/F5 · ideas detalladas, nada implementado

**Fecha:** 2026-09-18 · **Estado: ninguna de estas ideas está implementada.** Es el
menú para elegir, con el porqué de cada una anclado en lo que el EDA y el panel v1
mostraron. Cuando una se implemente, va por `/mlmodel` (builder en
`src/models/registry.py` + un YAML de experimento) y su resultado a
`docs/memoria/decisiones.md`.

## 0 · Lo que condiciona todo (leer antes de elegir)

> **Leer primero la primera entrada de [`docs/memoria/decisiones.md`](memoria/decisiones.md)**
> («Cierre de F3»). Consolida lo que midieron las tres ramas que cerraron la fase y
> corrige dos cosas que este doc decía: la meta de lift de la tabla de abajo y qué
> aprueba la auditoría (a) del punto 4. Lo de acá sigue siendo el menú de modelos.

Lo que sabemos del panel v1 (`docs/memoria/f2-eda-revision-y-features.md`):

| hecho | consecuencia para el modelo |
|---|---|
| 53 vehículos con positivo en dev, 14 en test; 254 filas positivas | cualquier modelo con más de ~10 parámetros efectivos por feature sobreajusta; **regularización fuerte, pocas comparaciones, intervalos siempre** |
| la señal honesta es débil: ROC ≈ 0,58–0,60 con logística/LGBM de auditoría | el objetivo realista es **0,65–0,70 de ROC**, no 0,9. Todo lo que pase de ahí se audita (regla 6). **Ojo con el lift**: la meta de 1,6–2× que figuraba acá está *por debajo* del techo de cohorte (2,10×), así que alcanzarla no demuestra anticipación — ver el punto 5 del protocolo |
| la señal es térmica/de uso y **progresiva** en los últimos ~4.000 km (idle, no llegar a régimen, más lento) | ganan los modelos que ven **el cambio del vehículo respecto de sí mismo** y la **secuencia de cortes**, no solo la foto de una ventana |
| los sanos son censurados ("no falló todavía"), no negativos limpios | la supervivencia usa esa información; la clasificación la tira |
| la etiqueta binaria tiene un borde arbitrario (el corte en E−G−H−Δ es 0, el siguiente es 1, con features casi iguales) | **etiquetas blandas** o regresión del tiempo al evento quitan ese ruido |
| calendario y odómetro son atajos; el emparejado los cierra pero no los borra | cualquier modelo se compara **con y sin** las columnas `aux_` de calendario: si mejora mucho con ellas, aprende el atajo |
| dos mercados con distinto nivel (`msg_full` 5,45×, temperatura +8 °C) | normalizar dentro del mercado, en el `Pipeline` de cada fold |

Y el protocolo, que no cambia con el modelo:

1. Mismos folds (`splits.json`), PR-AUC out-of-fold **para ordenar**, curva de
   anticipación vs. falsas alarmas para el pitch, intervalos por bootstrap sobre folds.
   **El PR-AUC por fila ya no elige el finalista**: por el punto 5, uno por debajo de
   0,2627 no demuestra anticipación, así que ordenar candidatos por él es ordenarlos por
   cuán bien identifican la cohorte de muestreo. El finalista se elige por **(a′) y por
   la estabilidad entre repeticiones** (el único con (a′) positivo confirmado a R=3 es
   survival stacking: +0,0162 ± 0,0033 contra −0,0055 ± 0,0040 del control, y ±22 km de
   anticipación contra ±2.237). El PR-AUC por fila queda como descarte, no como premio.
2. **Presupuesto de comparaciones**: con ~12 eventos por fold, más de 6–8 candidatos
   garantiza que "el mejor" sea ruido. Preregistrar la lista y no agregar sobre la marcha.
3. CV repetida (3 semillas de folds) para cualquier diferencia que se quiera declarar.
4. **Auditorías obligatorias por modelo** (`scripts/audit_model.py` y
   `scripts/audit_ordinal_horizon.py`, que las corren sobre el YAML del experimento).
   Se aprueban dos:

   - **(a0) el null.** Permutar las **features** entre *todas* las filas → el PR-AUC
     tiene que caer a la tasa base. Si no cae, hay leakage y ningún otro número vale.
   - **(a') el aporte del *cuándo*.** Reemplazar el score de cada fila por el promedio
     de su vehículo, sin reentrenar; la caída de PR-AUC es lo que el modelo sabía del
     *cuándo*. Delta ≤ 0 significa que no anticipa, por más PR-AUC que tenga.

   Y dos informativas: **(b)** agregar `aux_air_temp_avg`,
   `aux_regen_marker_per_1000km`, `aux_static_ProductionDay` como `feat_` → el ROC no
   tiene que saltar; **(c)** importancias/SHAP contra la hipótesis física (§3.2 del doc
   de F2).

   **(a) permutar dentro del vehículo ya no es criterio, y nunca fue un null.** La
   versión vieja de este punto pedía permutar dentro de cada vehículo y esperar que el
   PR-AUC cayera a la tasa base. No cae: **sube**, con los dos modelos probados y con
   tres semillas (0,17–0,21 contra 0,161 de referencia). Deja intacto *qué* vehículos
   fallan —de donde sale casi todo el PR-AUC de este panel— y encima le saca a cada auto
   el ruido de qué ventana le tocó, así que entrena un ordenador de vehículos mejor. No
   se marca ni como pass ni como falla; se lee junto a (a'). Cualquier informe que la
   cite como aprobación hay que rehacerlo. Medida como piso —no como null— **ningún
   modelo del repo le gana a su propio nulo intra-vehículo**, tampoco el control
   binario: 0,1928 / 0,1867 / 0,1784 contra 0,1653 / 0,1524 / 0,1567.

   (Detalle y evidencia: [`docs/memoria/decisiones.md`](memoria/decisiones.md) y
   [`docs/memoria/f3-ordinal-horizonte.md`](memoria/f3-ordinal-horizonte.md).)

5. **El piso es el techo de cohorte, no la tasa base.** En dev las 254 filas positivas
   están **todas** dentro de los 967 cortes de vehículos fallados, así que puntuar cada
   fila con "¿este auto falla?" —sin nada del *cuándo*— da **PR-AUC 0,2627 y lift
   2,10×** (`src/eval/metrics.py::cohort_ceiling`, lo reporta toda corrida). De ahí
   salen dos correcciones a la fila de la tabla de arriba: un PR-AUC por fila por debajo
   de 0,2627 **no demuestra timing**, y **el objetivo de 1,6–2× de lift se alcanza sin
   anticipar nunca**, porque el techo ya está por encima. Para el *cuándo* se mira
   `pr_auc_within_failed` (lift sobre 0,2627, no sobre 0,1252) y (a').

6. **Toda métrica que dependa del tamaño de la bolsa o del largo del historial se
   compara contra un nulo que conserve esa magnitud, nunca contra la tasa base.** El
   panel le deja 18,25 cortes por bolsa a un vehículo con evento y 9,00 a un sano (el
   emparejado ralea al sano), y el tamaño de bolsa **solo**, como score, ya da lift
   1,79×. Eso contamina toda agregación que crezca con el tamaño: `noisy_or` dio 1,83×
   y pierde contra su propio nulo. `mean` es la única insensible (corr. de rango con
   `n_cuts` 0,04 contra 0,85), y es la que se reporta. La misma trampa aparece dada
   vuelta en los crudos, donde el evento **corta** el historial del fallado: un AUC de
   0,694 que al igualar la ventana de odómetro cae a 0,564.
   (`scripts/audit_mil_bagsize.py`.)

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

### 1.2 · Logística regularizada bien hecha

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

### 1.3 · LightGBM con restricciones monótonas

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

### 2.4 · Desvío respecto del propio vehículo (cambio, no nivel)

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

**Cómo.** Es medio feature engineering, medio modelo: las columnas `feat_*_zself` se
generan en `src/data/panel.py` después de `build_panel` (agrupando por vehículo,
ordenando por `cut_odo`, con `expanding()` y `shift(1)`), y cualquier modelo de arriba
las consume. La variante CUSUM es un builder `cusum_rule` con dos parámetros
(`k`, `h`) que se fijan con el train del fold. Costo: bajo. Cuidado con el primer corte
de cada vehículo (sin historial: NaN, que el imputador maneja).

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
  explícitamente condicionado a "más de cien"). *Nota 19-09:* la solución de la tutora
  (CNN-LSTM) se implementó igual como baseline, con ~3.000 parámetros y reusando
  `train.py`; quedó a 0,01 de PR-AUC del LightGBM solo con `signals`
  ([f3-cnn-lstm-tutora.md](memoria/f3-cnn-lstm-tutora.md)). La advertencia sigue
  valiendo para arquitecturas más grandes.
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
