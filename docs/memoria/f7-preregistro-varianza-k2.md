# Preregistro: bajar la varianza de K2 (F7) · lista cerrada de tres candidatos

**Fecha:** 2026-09-24 · **Fase:** F7 · **Rama:** `exp/f7-varianza-k2`.
**Estado: BORRADOR para discutir en equipo.** Todavía no se commiteó. Cuando se commitee, ese
commit es la marca de tiempo. Tiene que ser anterior a la implementación, a todo entrenamiento y
a mirar cualquier score nuevo contra la etiqueta.

**Por qué existe.** K2 es el finalista (17,0% ± 3,8 de detección con V), pero el bootstrap por
vehículo no separa su mejora del cero, y el desvío entre repeticiones es del orden de la
mejora. La bibliografía de F5 (§0, §2) y el tamaño muestral apuntan al mismo diagnóstico:
**con 45–53 fallados, el límite no es el learner sino la varianza** (Riley et al. 2019:
con ~50 eventos y C ≈ 0,65 entran 2–3 parámetros; Burk et al. 2026: en baja dimensión nada le
gana significativamente a un Cox). CLAUDE.md dice que el presupuesto de comparaciones está
agotado. Esta lista gasta uno nuevo, **explícito y cerrado**:
- tres candidatos, los tres K2 con **menos varianza** por tres caminos distintos;
- la misma regla de seis condiciones de F6 (§5), ahora contra K2;
- una idea que aparezca en el camino va a "Qué queda abierto" de la ficha, no a esta lista.

## 0 · Lo que ya se vio antes de escribir esto

**Dev, contra la etiqueta:**
- K2 (`f6-ss-hw-r3`), la referencia, está entero en
  [f6-deteccion-vehiculo.md](f6-deteccion-vehiculo.md).
- **El componente embolsado del finalista anterior** (`f3-survival-stacking-r3-bag10`, de E2)
  detectó 18,2% ± 3,2 con D, contra 15,7% ± 0,9 sin embolsar. Su lift bajó de 1,616 a 1,591.
  **Ese número se vio y motiva P1**: P1 no es una hipótesis ciega. Por eso la regla de §5 exige
  ganarle a K2 con V, que es una etiqueta y un conjunto en riesgo distintos de los de esa
  corrida.
- El mapa de signos de P2 **se escribió el 18-09** (`docs/f3-modelos-candidatos.md` §1.3,
  commit b88e1ce), antes del primer modelo del repo (19-09). Es la hipótesis física del plan, no
  un ajuste a lo que vio un modelo. Pero ojo: los signos del idle y el refrigerante salieron del
  EDA sobre dev, y en la fuente externa de F5 §3.2 **fueron al revés** (AUC 0,43 y 0,44). Si P2
  pierde, esa es la primera explicación a revisar.
- Las cinco covariables de P3 son las de la regla física de `f3-modelos-candidatos.md` §1.1
  (18-09) más `km_per_day`: el uso como covariable es lo que la literatura de garantía
  automotriz recomienda (Lawless, Crowder & Lee 2009), y el cure model mostró que −km/L sola
  empata con 4 hábitos (0,58 contra 0,60).

**Test:** nada.

**Sin etiqueta:** nada nuevo.

## 1 · Qué se compara

**Referencia:** `f6-ss-hw-r3` (K2), sin reentrenar. Sus números:

| etiqueta | detección @ ≤ 50 FA/1.000 | detectados | lift por vehículo | anticipación mediana |
|---|---|---|---|---|
| **V (decide)** | **17,0% ± 3,8** | 7 / 10 / 6 de 45 | 1,658 ± 0,074 | 7.438 ± 2.005 km |
| D (informativa) | 15,1% ± 3,1 | 8 / 10 / 6 de 53 | 1,671 ± 0,034 | 8.811 ± 370 km |

**Filas, target y folds:** los de K2 en los tres candidatos:
- el mismo panel (`panel_survival_kmw.parquet`, build `data/rebuild-0921`) y las 2.029 filas de dev;
- el target `window_km_survival`, con entrada tardía y salida en la ventana del registro;
- `max_horizon_km: 55000`, bins de 500 km, score `1 − S(3.000 km | x)`;
- preprocesamiento `standard` y `splits_r3.json`.

## 2 · Los tres candidatos (lista cerrada)

### P1 · K2 embolsado por vehículo

**Qué es.** `vehicle_bagging` (`src/models/bagging.py`) con `base_model: survival_stacking` y
**exactamente** los `params` de K2. `n_bags: 10`, `random_state: 42`, como en E2. Cada bolsa es
un bootstrap de vehículos del train del fold; el score es el promedio de las bolsas. Nada se
re-tunea.

**Por qué podría detectar más.** El desvío de K2 entre repeticiones (±3,8 puntos, 6 a 10 autos)
es varianza de entrenamiento: con 45 fallados, qué autos caen en el train de cada fold mueve
el booster. Promediar bolsas de autos ataca justo eso. En el finalista anterior subió la
detección con D en +2,5 puntos.

**Por qué podría perder.**
- En el finalista anterior **bajó el lift** (1,616 → 1,591) y **subió el desvío** (0,9 → 3,2).
- Un bootstrap de vehículos deja afuera ~37% de los autos en cada bolsa: de los 42–43 fallados
  del train de un fold, cada bolsa aprende con ~27 distintos.

### P2 · K2 con restricciones monótonas

**Qué es.** El mismo `DiscreteSurvivalStacker` de K2, con `monotone_constraints` en el LightGBM
interno. El mapa es el del 18-09 (`f3-modelos-candidatos.md` §1.3), sin agregar ni sacar nada:

| signo | features (`feat_` del panel v1) |
|---|---|
| **+1** (el riesgo no baja) | `idle_frac`, `idle_per_1000km`, `trips_below_regime_temp_frac`, `short_trip_frac_5km`, `trips_below_30kmh_frac`, `msg_full_per_1000km`, `msg_overloaded_per_1000km`, `msg_over_limit_per_1000km`, `msg_cleaning_auto_per_1000km` |
| **−1** (el riesgo no sube) | `speed_kmh_mean`, `coolant_temp_end_mean`, `engine_temp_avg_median`, `engine_temp_max_median`, `engine_temp_amplitude_mean`, `trip_distance_median_km`, `trip_distance_p25_km` |
| 0 | todas las demás, y la columna del bin del hazard |

Única lectura del mapa del 18-09 que no es literal: el patrón `msg_*_per_1000km` también
abarca `msg_cleaning_auto_trend_per_1000km`, que es una **diferencia** entre mitades de la
ventana, no una tasa, y no tiene un signo físico claro. Queda libre (0).

Los hiperparámetros del booster son los defaults de `survival_stacking` (`num_leaves` 7,
`min_child_samples` 60, `reg_lambda` 5, `n_estimators` 300): ya están dentro del rango que
pedía §1.3. `monotone_constraints_method` queda en el default de LightGBM.

**Por qué podría detectar más.** Una restricción de signo es regularización que no cuesta
datos: le prohíbe al booster memorizar un auto con un corte raro en una feature física. Con 45
fallados, eso es varianza que se va.

**Por qué podría perder.**
- Si el efecto real no es monótono (por ejemplo, idle alto por uso en ciudad *y* bajo por uso
  en ruta en autos distintos), la restricción lo aplana.
- Los signos del idle y del refrigerante no se replicaron en la fuente externa (§0).

### P3 · Survival stacking parsimonioso: hazard logístico con cinco covariables

**Qué es.** El mismo apilado de K2 (mismas filas en riesgo, mismos bins, misma entrada
tardía, mismo score), pero el learner del hazard es una **logística con L2**, no un LightGBM.
Entra:
- el bin del hazard, como spline natural de 4 grados de libertad (la forma del hazard base);
- **cinco covariables**, fijadas acá: `feat_idle_per_1000km`,
  `feat_trips_below_regime_temp_frac`, `feat_speed_kmh_mean`, `feat_coolant_temp_end_mean` y
  `feat_km_per_day`;
- `static_SalesCountry_cd` (el nivel del mercado, como en K2).

No entra `feat_cut_odo`: la posición la lleva el bin. `C = 1,0` y `class_weight` en `None`, sin
tunear. Son ~11 parámetros efectivos contra los cientos de hojas de K2.

**Por qué podría detectar más.** Es un Cox en tiempo discreto (Bender et al. 2020) con el
número de covariables que admite el tamaño muestral. Si el panel tiene una sola dimensión de
señal (F3) y el límite es la varianza (F5), un modelo con cinco coeficientes debería ordenar
los autos igual que K2 con mucho menos desvío entre repeticiones.

**Por qué podría perder.**
- La logística no ve interacciones ni umbrales. K2 pone la velocidad y la temperatura de motor
  arriba en importancia; si hay no linealidad ahí, se pierde.
- El cure model, con cuatro hábitos parecidos, fue débil (D1 0,60). Pero ese era otro panel
  (hitos post-venta) y otra regla de alerta.

## 3 · Cómo se mide

Es la cuenta de F6, sin cambios: `scripts/eval_window_label.py`, con V (decide) y D
(informativa), contra K2 sin reentrenarla. Por repetición salen:
- la detección a ≤ 50 falsas alarmas cada 1.000 sanos, con `k_consecutive: 2` y 50 umbrales;
- el lift por vehículo con `mean`;
- la anticipación mediana.

Antes de medir, cada candidato tiene que coincidir con K2 por `(vehicle_id, cut_odo)`, en
`label` y `event_observed`, y en `fold_r{r}`.

**Pisos:** `cut_odo` y `aux_cut_dss`, los mismos de F5 y F6.

## 4 · Auditorías y nulos (antes de leer el veredicto)

- **(a0):** features permutadas entre todas las filas (semilla 0), reentrenado. El PR-AUC por
  fila tiene que quedar a menos de 0,02 de la tasa base. `audit_model.py` en los tres.
- **Nulo de tamaño de bolsa para la detección** (`audit_detection_null.py`, 200 permutaciones,
  semilla 0). Ninguno de los tres transforma el score después de entrenar, así que el nulo es
  el mismo en distribución que el de K2; el exceso se calcula igual.
- **Informativas, no deciden:**
  - (a′) y la (b) de calendario. **La (b) tiene que seguir cerca de la de K2 (+0,020)**; si un
    candidato la sube, va escrito como límite aunque gane;
  - el Brier;
  - la correlación de rango con K2;
  - el C-index;
  - el bootstrap pareado por vehículo (1.000 remuestreos estratificados) de la diferencia de
    detección con V;
  - **el desvío entre repeticiones de la detección**, que es lo que esta lista promete bajar.
    Se reporta para los tres al lado del de K2 (3,8).

## 5 · Qué decide (se aplica en este orden)

Con la etiqueta **V**, un candidato **supera a K2** si cumple las seis condiciones de F6:

1. **Gana en detección:** la diferencia de medias supera el desvío combinado
   (`ensemble_rank.py::compare`).
2. **No pierde en lift por vehículo:** la diferencia negativa no supera el desvío combinado.
3. **Guarda de multiplicidad:** la diferencia de detección pareada es ≥ 0 en las tres
   repeticiones.
4. **Guarda de anticipación:** la anticipación mediana no cae más que el desvío combinado.
5. **Les gana a los dos pisos** en detección y en lift, y **pasa (a0)**.
6. **Su exceso sobre el nulo de tamaño de bolsa es ≥ el de K2** (+9,5 puntos).

**Si más de uno cumple**, se adopta el de mayor detección media con V. Si la diferencia entre
ellos no supera su desvío combinado, se adopta el de **menor desvío entre repeticiones**; si
tampoco se separan, el de menos parámetros: P3, después P2, después P1.

**Si ninguno cumple, K2 sigue siendo el finalista** y se escribe por qué, con los números.

**Un empate también es un resultado:** si P3 empata con K2 en detección y lift con menos
desvío, **no se adopta** (no cumple la condición 1), pero va escrito: diría que K2 no aprende
nada que cinco coeficientes no sepan, y eso cambia qué se le explica al jurado.

## 6 · Fuera de la lista: la capa de decisión (no consume presupuesto)

Esto no compite con K2: es **dónde se opera** su score. No cambia el orden de los autos, así
que no cambia ni la detección a un umbral dado ni el lift (F5 §3.6). Se corre al lado, sobre
las predicciones out-of-fold de K2 y del ganador si lo hay:

- **La curva completa de detección contra falsas alarmas**, no solo el punto de 5%: 2%, 5%,
  10% y 20% de falsas alarmas, con la anticipación en cada punto. Es la respuesta a "¿cuántos
  autos más encontramos si aceptamos más alarmas?", y el punto de operación es una decisión de
  costo que se muestra en el pitch, no un número que se optimiza en dev.
- **El umbral con garantía de Neyman-Pearson** (Tong, Feng & Li 2018): el umbral que asegura
  ≤ α de falsas alarmas con probabilidad alta. Hoy el 5% se fija sobre 95 sanos (~5 autos de
  margen), y en el test (54 sanos) el 5% realizado puede quedar bastante arriba.

## 7 · Considerado y descartado (no entra en la lista)

| idea | por qué no |
|---|---|
| **Grafos** (GNN, propagación de etiquetas entre vehículos) | Con aristas por similitud de features, repite la única dimensión de señal que ya está; con aristas por metadatos (mercado, `ModelSeries`, lote), mete el sesgo de muestreo de las estáticas (+0,16 de ROC en la ablación). Con 45–53 fallados, una GNN no tiene de dónde aprender. En Component X, la GNN perdió contra un GBM tabular (Dimidov et al. 2026) |
| **Más features de ventana** | `feat/f3-features-regeneracion` barrió 7.310 cocientes contra su nulo: nada pasa la barra |
| **Evento ficticio + ventana de riesgo** (misma rama) | Resuelve el atajo de posición para el clasificador binario; survival stacking ya lo resuelve comparando contra el conjunto en riesgo. Mide *qué auto*, como K2 |
| **TabPFN** | Otro learner sobre el mismo panel; F5 §2 dice que eso compra poco. Además, licencia para un pitch corporativo |
| **Sesgar el modelo hacia positivos** (`class_weight`, umbral bajo) | No cambia el orden: es moverse sobre la misma curva. Va en §6 como punto de operación |

## 8 · Lo que no cambia

- **El test no se toca.** Si un candidato se adopta, sigue siendo "el mejor en dev".
- **wandb:** `WANDB_MODE=disabled`, como en F5 y F6. Las corridas van a `results/`.
- **Implementación antes de entrenar**, en un commit propio y posterior a este:
  - `configs/exp_f7_p1_bag.yaml` (P1: solo config);
  - `monotone_constraints` por nombre de feature en `DiscreteSurvivalStacker`, más
    `configs/exp_f7_p2_monotone.yaml` (P2);
  - el learner `logistic` con spline del bin en `DiscreteSurvivalStacker`, una lista explícita
    de columnas para el experimento, y `configs/exp_f7_p3_pem.yaml` (P3);
  - chequeos en `check_setup.py` que fallen con estas mutaciones: signo invertido en el mapa
    monótono, una feature del mapa que no existe en el panel, P3 con una columna fuera de la
    lista, y bolsas que no respetan el vehículo.
