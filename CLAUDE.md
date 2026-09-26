# CLAUDE.md — contrato y reglas del repo

Ford Innovation Challenge III · *Data-Driven Powertrain Intelligence*: predicción
temprana de degradación de eficiencia de combustión en vehículos conectados.
El plan completo está en `plan-implementacion-ford.md`; este archivo es el
resumen operativo que hay que respetar al escribir código.

## Estado

**Entrega v2 de Ford (26-09-2026): el pipeline vigente lee v2, y todo lo de F1–F8 de abajo se
midió con la entrega 1.** Ford mandó estática nueva (con ciudad de venta y país ISO) y los
viajes y señales de fallados nuevos. Los de sanos son byte a byte los de la entrega 1. Lo que
cambia río abajo, con evidencia en `docs/memoria/f9-*.md` y en las entradas del 26-09 de
[`decisiones.md`](docs/memoria/decisiones.md):
- **Cinco trampas de formato**, corregidas en el loader desde `raw_sources.yaml`:
  - la columna del evento renombrada;
  - `ProductionDay` de fallados +538 d;
  - una fila por evento;
  - filas repetidas;
  - la extracción de fallados diez días más larga.
- **Universo v2: 557 vehículos (177 eventos), 446 dev / 111 test**, re-sorteado con la semilla
  42 y un estrato evento × mercado × motor. **36 autos del test estaban en el dev viejo**
  (`test_split.json → previous`): el test final se reporta también sin ellos.
- **No hay ventana del registro**, la **fecha registrada cae ~2 semanas después de la
  intervención** (la referencia del evento es fecha − 21 d) y **ENG_3 ya falla** (46 de sus 47
  eventos en Brasil).
- **Re-medido sobre v2:**
  - survival stacking aprueba (a0) y (a′) como en v1;
  - la separación agrupada sube (ROC por fila 0,59 → 0,71), pero **dentro de mercado × motor el
    uso ordena autos con AUC ~0,58**, como en v1;
  - K2 perdió su corrección (la ventana) y **no hay finalista elegido sobre v2**.

  Ver `docs/memoria/f9-remedicion-v2.md`.

Fase 0 cerrada (infraestructura + panel dummy + harness verde). F1 cerrada del
lado de los datos crudos: `configs/data/raw_sources.yaml` está auditado contra los
archivos reales. El bloqueante de la fecha del evento se cerró el 17-09 recortando
el universo del estudio. **F2 cerrada el 18-09**: `scripts/build_dataset.py`
construye `data/processed/panel.parquet` (W=1000, G=500, H=3000, Δ=500) con 53
features declaradas en `configs/data/features_v1.yaml`; el panel, los splits y el
holdout están publicados como wandb Artifacts (`panel-v1`, `test-split`; se suben con
`scripts/log_panel_artifact.py`) y `configs/exp_baserate.yaml` es el piso contra el
panel real. **F3 cerrada el 20-09**: los candidatos medidos están en `results/`, y la
lectura conjunta —qué mide en realidad el PR-AUC por fila de este panel, y por qué el
finalista es survival stacking— es la primera entrada de
[`docs/memoria/decisiones.md`](docs/memoria/decisiones.md). Hay que leerla antes de
volver a comparar modelos: cambia el criterio. **El 22-09 se midió el cure model por hito
post-venta**, con preregistro y lista cerrada. El rasgo temprano existe, pero no le gana al piso
de producción y empata con un solo número de uso. No se adopta y **el finalista sigue siendo
survival stacking** (`docs/memoria/f3-cure-model.md`). El mismo día corrió el ensamble
preregistrado con el CNN-LSTM (E1/E2): pierde en lift por vehículo
(`docs/memoria/f3-ensamble-e1-e2.md`). También el 22-09, la incidencia aprendida con los
fallados sin fecha (F5 §3.2) paró en su compuerta preregistrada: en CNTRY_1/2/5 el rasgo temprano
no separa fallados de sanos (AUC 0,513), así que nada se aplicó a dev
(`docs/memoria/f5-incidencia-externa.md`). Y survival stacking en días post-venta con la ventana
del registro (F5 §3.3) aprende el *cuándo* pero pierde en lift por vehículo, así que no se adopta
(`docs/memoria/f5-ss-post-venta.md`). Deja dos cosas: la (b) del finalista es exposición al
registro, y con la etiqueta corregida el finalista detecta 11,9%, no 15,7%. Lo que sí cambia todo panel es la ventana
del registro de eventos (abajo).

**F6 (22-09) cambió el finalista.** Hubo una lista preregistrada de tres candidatos
(`docs/memoria/f6-preregistro-deteccion-vehiculo.md`), y el nuevo finalista es **survival stacking
con el conjunto en riesgo de la ventana del registro, en km y con horizonte completo** (K2,
`configs/exp_ss_hw_r3.yaml`, target `window_km_survival`).
- Con la etiqueta corregida detecta 17,0% ± 3,8 contra 11,9% ± 2,8, con el mismo lift y el mismo
  Brier, y la (b) de calendario baja de +0,049 a +0,020.
- La mejora es modesta: +2,3 autos de 45, y el bootstrap por vehículo no la separa del cero.
  Con la etiqueta dura empata (`docs/memoria/f6-deteccion-vehiculo.md`).

**F7 (24-09) no lo cambió.** Tres candidatos preregistrados para bajar la varianza de K2 (embolsado,
monótono, hazard logístico con cinco covariables): ninguno cumple las seis condiciones
(`docs/memoria/f7-varianza-k2.md`). Lo que deja: la versión embolsada de K2 detecta 7 de 45 en las
tres repeticiones, así que **el 17,0% es la lectura optimista y el pitch cita ~15–17%**.

**F8 (24-09, sin presupuesto) tampoco lo cambió.** No se pudo fechar a los fallados sin fecha
desde la telemetría (`docs/memoria/f8-datar-eventos-fase0.md`): sin Ford no hay más eventos. Y la
capa de decisión (`docs/memoria/f8-capa-decision-k2.md`) muestra que el 5% de K2 se sostiene fuera de
muestra (3,5% de falsas alarmas realizadas, 15,6% de detección) y que a 10% detecta ~26%.

**F4 (dashboard contra el panel real, con K2) está en `scripts/dashboard_k2/`**; las ideas que quedaron sin probar
siguen en `docs/f3-modelos-candidatos.md` y en "Qué queda abierto" de la ficha de F6. **El porqué de una alerta
(24-09, no es candidato)** sale de SHAP sobre K2 reconstruido bit a bit: V3 (TreeSHAP del hazard, promedio de 3
repeticiones), y al cliente solo llegan hábitos cuyo efecto coincide con la física del DPF; síntomas y contexto
nunca se nombran (`docs/memoria/f4-explicabilidad-k2.md`).

**El presupuesto de comparaciones está agotado** (§0, punto 2 de ese doc): con ~12
eventos por fold, agregar candidatos sobre la marcha garantiza que "el mejor" sea ruido.
Una corrida nueva se justifica por hacer comparable una fila que ya existe, no por sumar
un modelo. F6 gastó uno nuevo, explícito y cerrado: tres candidatos preregistrados con una
regla más estricta. Otro candidato necesita su propio preregistro.

**Antes de tocar los datos, leer [`docs/memoria/`](docs/memoria/README.md).** Ahí
están los hallazgos de F1/F2/F9 y las decisiones tomadas, con la evidencia y el comando
que las reproduce. Las que cambian cómo se escribe el código (entrega v2; entre
paréntesis, lo que valía con la entrega 1):

- Los datos vienen en **dos cohortes de muestreo** (failed / not_failed) y la
  cohorte *es* la etiqueta: `IdentificationDate` nula ⇔ sin evento. Nunca entra
  como feature. En v2 la estática de fallados trae **una fila por evento**: el dedupe
  se queda con el primero (`n_events_recorded` cuenta el resto).
- **La entrega v2 se lee con correcciones por parte** (`src/data/loader.py`, declaradas en
  `configs/data/raw_sources.yaml`):
  - la columna del evento de la estática de fallados se renombra;
  - `ProductionDay` de fallados se corre −538 d;
  - todo lo posterior al fin de extracción de los sanos (14-09-2026) se descarta.

  La entrega 1 se lee con `raw_sources_v1.yaml` (`data/old_raw/`).
- El **universo del estudio son 557 vehículos** (entrega 1: 364). Cambió la selección:
  - las dos cohortes se muestrearon en períodos de producción distintos, así que se conserva
    el período común, del 20-01-2025 al 31-07-2025 (`production_day_window: [1, 193]`);
  - el recorte por mercado de la entrega 1 ya no hace falta.

  El criterio vive en `src/data/usable.py` y se declara en `universe` de
  `configs/data/test_split.yaml`. **El panel se construye con los 557.**
- Hay **13 vehículos duplicados bajo dos códigos** (en v2 los 7 pares que siguen en
  fallados vuelven a traer los viajes repetidos): se colapsan con `src/data/dedupe.py` antes
  de cualquier split, o la regla 2 se viola en silencio.
- **`Engine` y `ModelSeries` son candidatas, ya no excluidas** (entrega 1: ENG_3 era el 0%
  de los fallados). Aportan la tasa de su celda y no orden entre autos: todo modelo que
  las use se reporta también **dentro de mercado × motor**.
- **Toda comparación fallado/sano se hace dentro del mercado** (y el AUC también dentro de
  mercado × motor). La tasa por mercado está cruzada con cómo Ford armó la lista.
- El **test es el holdout v2** (111 vehículos, `test_split.json`, re-sorteado el 26-09).
  El recorte a dev lo hace `test_split_masks()`, y `dev_mask` es pertenencia explícita
  a `dev_vehicles`, **no** el complemento de test. 36 autos del test estaban en el dev
  viejo (`previous.test_vehicles_in_old_dev`).
- **El marcador `signals.Regenerations` se corta el 25-05-2026** para toda la flota (también
  en v2). Una tasa de marcadores por km mide el calendario. La familia B se cuenta desde las
  caídas de `trips.AirRegeneration*`; el marcador y `DistanceBetweenRegenerations` quedan
  como `aux_`.
- **Los sanos se emparejan por odómetro y por mes** (`sampling` en `panel_v1.yaml` /
  `panel_v2.yaml`). En v2 la exposición previa al evento de los fallados sigue cayendo antes
  que la de los sanos. Cualquier feature con deriva temporal o de cohorte (temperatura
  ambiente, `ProductionDay`, `daysUntilSale`, fecha de venta) va como `aux_`, no como
  `feat_`/`static_`.
- **El 33% de las filas de `trips` son idle de 0 km** (motor encendido sin moverse) y
  son la señal que más anticipa. Toda fracción "de viaje" se calcula entre los que se
  mueven; `KilometerPerHour` es nulo exactamente ahí y se recalcula como km/duración.
- **En v2 no hay ventana del registro** (entrega 1: 01-09-2025 → 11-03-2026,
  `configs/data/event_clock.yaml`, que describe esa entrega).
  - La censura de un sano es el fin de extracción.
  - El riesgo se mide en días desde la venta: el evento se ordena por días, no por km.
- **La fecha registrada v2 cae ~2 semanas después de la intervención.** La referencia del
  evento del panel es `fecha − 21 d` (`label.reference_offset_days`). Sin eso, el gap G deja
  el taller y los síntomas en la última ventana.

## Contrato de datos

Artefacto: `data/processed/panel.parquet` (dummy: `panel_dummy.parquet`).
**Una fila por par `(vehículo, punto de corte)`.**

| Columna | Tipo | Descripción |
|---|---|---|
| `vehicle_id` | str | identificador unificado entre las tres tablas |
| `cut_odo` | float | odómetro en el punto de corte [km] |
| `cut_date` | datetime | fecha del corte (**nullable**: puede no haber anclaje) |
| `window_km` | float | ventana W usada para agregar las features |
| `horizon_km` | float | horizonte de anticipación H usado en esta fila |
| `gap_km` | float | gap de blanking G usado en esta fila |
| `label` | int | 1 si el evento cae en `[corte+G, corte+G+H]` |
| `time_to_event_km` | float | km hasta el evento; NaN si censurado |
| `event_observed` | int | 1 si el vehículo tiene evento registrado |
| `feat_*` | float | todas las features de ventana (53 en v1, declaradas en `configs/data/features_v1.yaml`) |
| `static_*` | mixto | solo `SalesCountry_cd` en el set base v1 |
| `aux_km_observed_after_cut` | float | km observados **después** del corte (`last_odo − c`). En los censurados es la única forma de saber hasta dónde estuvieron en riesgo: un sano no es un cero, es "llegó hasta acá sin fallar" |
| `aux_*` | mixto | **en el panel, fuera del modelo**: `aux_static_{Engine, ModelSeries, ProductionDay, daysUntilSale}` (+ `SalesCity` en v2; motor y modelo pasan a `static_` en `panel_v2_estaticas.yaml`), `aux_air_temp_*`, `aux_regen_marker_per_1000km`, controles de ventana. Para ablaciones y auditorías sin reconstruir |

**El panel de hitos del cure model** (`panel_landmark_ps.parquet`) usa los mismos nombres, pero
tiene una fila por (vehículo, hito post-venta). `horizon_km`/`gap_km` van en NaN, y el riesgo
vive en `aux_dss_entry`, `aux_dss_exit` y `aux_event_in_window` (`docs/memoria/f3-cure-model.md`).

**Regla de prefijos:** toda columna que entra a un modelo se llama `feat_` o
`static_`, y `src/training/cv.py` la selecciona por prefijo. Agregar una feature
no requiere tocar el código de entrenamiento ni coordinar con nadie: es una línea
en `configs/data/features_v1.yaml` (`name`, `source`, `column`, `agg`; `aux: true`
la deja fuera del modelo).

**Con qué se entrena vs. con qué se mide.** Un YAML de experimento puede declarar
`target: {name, params}` para entrenar contra otra cosa que `label` (los modos están
en `src/training/targets.py`; sin la clave, se entrena con `label`). **Lo que se
evalúa no cambia nunca**: PR-AUC out-of-fold sobre la etiqueta dura, con los mismos
folds. Agregar un modo es una función con `@register_target` y cero líneas en `cv.py`.

## Reglas que no se negocian

1. **Gap de blanking.** El modelo nunca ve los km inmediatamente previos al
   evento. Sin G esto es detección reactiva, que es justo lo que Ford ya tiene.
2. **Split agrupado por vehículo.** Un `vehicle_id` nunca cae en train y
   validación a la vez. La lógica está centralizada en `src/eval/splits.py` y no
   se reimplementa en ningún otro lado. Arriba de la CV hay un **holdout dev/test
   congelado** (`data/processed/test_split.json`, re-sorteado sobre el universo v2 el
   26-09 antes de mirar ninguna feature contra la etiqueta), con tres listas:
   `dev_vehicles` (446), `test_vehicles` (111) y `excluded_vehicles` (434, fuera del
   universo). El de la entrega 1 (290/74/717) sigue congelado en `data/processed-v1/`.
   El panel se construye con los **557** del universo, y
   **todo lo que se mira es dev**: la CV, la selección de modelo y cualquier figura
   salen de `dev_mask`. Las filas de test existen y nadie las toca hasta que el
   modelo está elegido. El recorte lo hace `scripts/train.py` (`select_dev()`) antes
   de armar los folds, nunca filtrando `vehicle_id` a mano. Dentro de dev, los folds
   se estratifican por `label` a nivel vehículo —la variable que mide el PR-AUC, no
   `event_observed`—, en v2 cruzado con mercado y motor (`stratify.extra_columns`),
   ningún fold puede quedar con menos de `min_valid_positives`
   filas positivas en validación, y `n_repeats` habilita CV repetida; los tres salen
   del bloque `splits:` del YAML y el archivo congelado declara con cuáles se armó
   (si el YAML dice otra cosa, `train.py` falla). Por eso **todo YAML de
   experimento declara `splits.test_split`** —el path del holdout, o `null` explícito
   si el panel no tiene—: si falta la clave, `train.py` no corre. Si el panel trae un
   vehículo excluido, `test_split_masks()` falla: significa que se construyó con el
   universo viejo.
3. **Features solo hacia atrás.** Ninguna feature usa información posterior al
   punto de corte. Escalado e imputación se ajustan solo con el train de cada
   fold (van dentro del `Pipeline`, nunca sobre el panel entero).
4. **Eje de odómetro por defecto.** `IdentificationDate` está en días desde
   producción y `TripDatetimeStart` es calendario. F1 encontró el anclaje que
   faltaba (`ProductionDay` está en el eje del calendario, IQR de 0 días: ver
   `docs/memoria/f1-anclaje-temporal.md`), así que el evento **sí** se puede
   traducir al eje de km —en v2 cae en una mediana de 12.808 km, desde la referencia
   `fecha registrada − 21 d` (entrega 1: 7.987 km)—. Eso no asciende al eje de días:
   sigue siendo reporte secundario, y el origen se estima una vez y se congela.
5. **Métricas.** PR-AUC out-of-fold **para ordenar**, curva de anticipación vs.
   falsas alarmas para el pitch, accuracy nunca. **No elige el finalista**: por la
   regla 6, por debajo del techo de cohorte un PR-AUC por fila mide *qué auto* y no
   *cuándo*, así que el finalista se elige por **(a')** y por la **estabilidad entre
   repeticiones** (CV repetida). La lectura completa es la primera entrada de
   `docs/memoria/decisiones.md`.
6. **Un PR-AUC se audita antes de celebrarse, y el piso es el techo de cohorte.**
   Dos mitades. (i) Uno **sospechosamente alto**: variables como el nivel del DPF
   son casi la definición del evento; sin gap, el modelo memoriza en vez de
   predecir. (ii) Uno **normal tampoco se celebra solo**: en el dev del panel v2 las
   682 filas positivas están *todas* dentro de los 3.860 cortes de vehículos
   fallados, así que puntuar cada fila con "¿este auto falla?" —sin nada del
   *cuándo*— da **PR-AUC 0,1767 y lift 3,03×** (`src/eval/metrics.py::cohort_ceiling`;
   panel v1: 0,2627 y 2,10×). Ese es el piso, no la tasa base de 0,0583: **por debajo
   del techo un PR-AUC por fila no demuestra anticipación**, y el objetivo de 1,6–2×
   de lift del plan se alcanza sin anticipar nunca. Para el *cuándo* se miran
   `pr_auc_within_failed` (lift sobre el techo) y **(a')**, y las dos las reporta toda
   corrida. En v2, además, un AUC agrupado se lee al lado del AUC dentro de
   mercado × motor: la diferencia es composición de la muestra.

   Y **toda métrica que dependa del tamaño de la bolsa o del largo del historial se
   compara contra un nulo que conserve esa magnitud**, nunca contra la tasa base: el
   panel v1 le dejaba 18,25 cortes por vehículo fallado y 9,00 por sano, y el tamaño
   solo, como score, daba lift 1,79× (`scripts/audit_mil_bagsize.py`). En el panel v2
   las bolsas son casi simétricas (28,8 / 26,9), pero con 27 cortes por sano la
   detección a un presupuesto fijo de falsas alarmas cambia de escala: no se compara
   entre paneles sin ese nulo.

   Las auditorías obligatorias son `scripts/audit_model.py`. Aprueban **(a0)**
   —features permutadas entre todas las filas, el PR-AUC cae a la tasa base o hay
   leakage— y **(a')** —colapsar el score al promedio del vehículo sin
   reentrenar; la caída es lo que el modelo sabía del *cuándo*—. **(a)**, la
   permutación dentro del vehículo, es **informativa y no es un null**: deja
   intacto qué vehículos fallan y sube. No aprueba nada, y lo que la haya citado
   como aprobación hay que rehacerlo (`docs/memoria/decisiones.md`).
7. **Nada se hardcodea.** Paths, semillas e hiperparámetros salen de un YAML de
   `configs/`. Para cambiar un hiperparámetro se escribe otro YAML, no se edita
   el código.
8. **Datos y outputs no se versionan.** `data/`, `experiments/` y `wandb/` están
   en `.gitignore`; lo que se comparte va como wandb Artifact.
9. **Todas las corridas van al mismo lado.** Team `oromagnoli-`, proyecto
   `ford-fic` (`wandb.entity`/`wandb.project` en el YAML del experimento). Se
   pisan con `WANDB_ENTITY`/`WANDB_MODE` para trabajar sin red o en privado,
   nunca editando el config compartido: si cada uno loguea a su cuenta, las
   métricas dejan de ser comparables.

## Mapa del código

```
src/config.py            carga de YAML, resolución de paths, semillas
src/data/loader.py       carga de las tres tablas crudas (esquema en configs/data/raw_sources.yaml); correcciones
                         por parte (`rename`, `offsets`) y por tabla (`truncate_after`)
                         load_table() entera, iter_table() por chunks (13M de filas)
src/data/dedupe.py       colapso de los 13 vehículos duplicados (lista en configs/data/vehicle_dedupe.yaml)
src/data/usable.py       universo del estudio: qué vehículo entra y por qué los demás no (fecha utilizable,
                         mercado y, desde v2, ventana de producción `production_day_window`)
src/data/join.py         unión a nivel vehículo + enriquecimiento de trips/signals con las estáticas
                         (trips y signals NO se mergean entre sí: no hay clave fila a fila)
src/data/subset.py       trips/signals para un conjunto de vehículos: canoniza → filtra → deduplica la fila completa
src/data/anchor.py       origen del calendario (estimado sobre dev, congelado en panel_meta.json) y odómetro del evento;
                         `project_odometer_to_dates`, la inversa exacta (fecha en que el odómetro llega a un valor)
src/data/panel.py        cortes en grilla de Δ, etiqueta con gap y horizonte, censura, QC de ventana, emparejado de sanos
src/data/landmark.py     panel de hitos post-venta (cure model): una fila por (vehículo, hito en días desde la venta),
                         riesgo dentro de la ventana del registro, features por mes para la referencia de flota;
                         opcional: ventana de features fija (`feature_window_days`) y km/día (`usage_feature`)
src/data/window_risk.py  el panel v1 en días (F5 §3.3): tramo en riesgo por fila desde que el odómetro llega a c + G,
                         dentro de la ventana del registro; `feat_cut_dss`; filas evaluables con la etiqueta corregida
src/data/external.py     conjunto externo de la incidencia (F5 §3.2): reproduce el sorteo padre contra su huella y
                         toma los excluidos del lado dev, por mercado; nunca un vehículo de dev ni de test
src/features/trips.py    derivadas a nivel viaje (idle/moving, velocidad recalculada, topes físicos, regen = caída de AirRegeneration)
src/features/signals.py  una booleana por nivel de Message; regen_marker solo como aux
src/features/windows.py  primitiva de ventana (c−W, c] sobre odómetro + agregadores (per_1000km, half_*, gap_*, km_since_last…)
                         + compute_history_deviation(): agg(historia previa) − agg(ventana), NaN sin historia mínima
src/features/sequences.py  la ventana en T bins de km × C canales (entrada de modelos secuenciales), aplanada a feat_seq_*
src/models/registry.py   get_model(name, params); agregar un modelo = registrar un builder
src/models/timesfm_zeroshot.py  series por km + TimesFM 3.0 zero-shot sobre los cortes del panel (no es del registry)
src/models/cnn_lstm.py   baseline de la tutora: Conv1D+LSTM sobre la secuencia + rama estática (torch, opcional)
src/models/survival_stacking.py  supervivencia en tiempo discreto: apila (fila × bin de km), hazard por bin,
                         score = 1 − S(H|x). Backend lightgbm o gpboost (efecto aleatorio por vehículo, opcional).
                         Entrada tardía opcional (`entry_km` en el `y`, target `window_km_survival`)
src/models/bagging.py    bagging por vehículo (`vehicle_bagging`): N bootstraps de autos del train del fold, promedio;
                         el vehículo llega por el `y` (`discrete_survival` o `grouped_label`)
src/models/cure.py       mixture cure model por hito (incidencia Firth + FLIC, pesos unitarios o pesos fijos de afuera,
                         latencia Weibull con entrada tardía, EM que falla si la verosimilitud baja); trae su propio pipeline
src/models/window_stacking.py  survival stacking en días con exposición exacta (PEM: LightGBM Poisson con offset);
                         `window_survival_stacking`, va con el target `window_survival`
src/models/incidence.py  incidencia por vehículo aprendida con la fuente externa (Firth + estrato de mercado) y aplicada
                         congelada: `external_incidence` no aprende nada en `fit`
src/training/cv.py       loop de CV agrupada; selección de features por prefijo; hooks `target:`,
                         `preprocessing: standard|none` y `carry_columns` (eval.carry_columns)
src/training/transformers.py  FleetReferenceNormalizer: desvío contra la mediana de los sanos del train por
                         mercado × mes (se ajusta por fold, nunca con validación)
src/training/targets.py  con qué se entrena (no con qué se mide) y cómo la salida del modelo vuelve a un
                         score comparable: registro por nombre, `discrete_survival`, `ordinal_horizon` y
                         `cure_window` (exposición en la ventana del registro, para el cure model) y
                         `window_survival` (el tramo en riesgo del panel v1 en días, F5 §3.3) y
                         `window_km_survival` (el mismo tramo en km, con entrada tardía: el finalista de F6).
                         Lo que se evalúa sigue siendo `label`; cv.py no sabe qué modos hay
src/eval/splits.py       splits antileakage + serialización a splits.json; estrato compuesto (`composite_strata`,
                         `extra_columns`), holdout sobre el universo (`holdout_on_universe`) y `compare_holdouts`
                         estratificación (columna/nivel), guarda de positivos por fold y CV repetida: todo del YAML;
                         extend_splits() conserva los folds de un split existente y reparte solo los vehículos nuevos
src/eval/metrics.py      PR-AUC/ROC/Brier + lead_time_curve() + false_alarm_rate() + bootstrap
                         (por folds y por vehículo) + C-index out-of-fold +
                         cohort_ceiling()/pr_auc_within_failed()/when_contribution() (el piso real y la
                         descomposición cohorte/cuándo), vehicle_scores()/vehicle_metrics() (la decisión
                         por vehículo, MIL) y cost_ratio_sweep() (C_FN/C_FP: reporte, nunca selección);
                         landmark_metrics(): D1 (C con entrada tardía) y D2 (primera alerta sobre los hitos) del
                         panel de hitos, con bootstrap pareado por vehículo
src/eval/plots.py        figuras compartidas entre dashboard e informe
src/eval/dashboard_data.py  datos del dashboard de K2: carga, alerta por vehículo (misma regla que la curva) y
                         perfil de la flota sana; los números oficiales salen de window_eval.json y decision_layer.json;
                         load_explanations()/vehicle_why() leen la explicabilidad (sin recalcular SHAP)
src/eval/explain.py      explicabilidad de K2, funciones puras: refit_folds() (el loop de run_cv, devuelve los modelos),
                         V1 tree_shap_hazard (TreeSHAP del hazard sobre los 6 tramos de H, aditivo), V2
                         permutation_shap_score, V4 family_sums; explained_cuts() (la alerta de la curva), la mediana
                         sana comparable (FleetReferenceNormalizer, train del fold), borrado, estabilidad, plausibilidad
                         física, message_factors()/render_vehicle_message() y dev_only_guard()
scripts/make_dummy.py    panel dummy con el esquema del contrato
scripts/make_test_split.py  auditoría del join + universo + holdout (`draw.mode: universe` en v2; `all` =
                         sorteo + recorte de la entrega 1, con test_split_v1.yaml). Se corre una vez
scripts/compare_deliveries.py  entrega 1 contra v2: archivos, estática, anclaje, etiquetas, telemetría, corte de
                         producción y qué marca la fecha nueva (sin test) -> experiments/entregas/
scripts/build_city_elevation.py  altura de cada ciudad de venta (Open-Meteo/GeoNames) -> configs/data/city_elevation.yaml
scripts/eda_v2.py        EDA de la entrega v2 sobre dev, con las comparaciones dentro del mercado (riesgo con
                         exposición, ventana, reloj, perfil alineado, rasgo temprano, altura, motor)
scripts/build_dataset.py panel real: universo del holdout → crudos → evento en km → cortes/etiqueta/features →
                         sanos emparejados → panel.parquet + splits.json (folds sobre dev) + panel_meta.json
scripts/eda_gaps.py      complemento del EDA sobre dev: factibilidad de W/G/H, perfil alineado al evento,
                         post-evento, calendario, ICC intra-vehículo (experiments/eda/dev/gaps/)
scripts/log_panel_artifact.py  publica panel.parquet + splits.json + panel_meta.json (`panel-v1`) y
                         test_split.json (`test-split`) como wandb Artifacts
scripts/build_survival_panel.py  panel v1 + `feat_cut_odo` (el odómetro del corte como covariable del hazard base)
scripts/audit_model.py   las auditorías obligatorias de F3 §0.4 sobre cualquier YAML de experimento:
                         null global, permutación intra-vehículo, aporte del `cuándo`, aux_ de calendario, importancias
scripts/build_seq_panel.py  panel secuencial: mismas filas que el panel v1, feat_seq_* en vez de agregados
                         (+ _meta.json con T y C); mismo splits.json
scripts/make_splits.py   rearma splits.json sobre un panel que ya existe (cambiar folds no es reconstruir el panel);
                         con `splits.extend_from` extiende uno congelado, y no pisa un reparto distinto sin --force
scripts/audit_event_clock.py  en qué reloj ocurre el evento y en qué ventana se registra (Fase 1 del cure model)
scripts/build_landmark_panel.py  panel de hitos post-venta + _meta.json (universo de 364, conteos solo de dev)
scripts/audit_cure.py    auditorías del cure model (A0, C1, C2, C3, C6, A3, A5, A6), veredicto y adopción
scripts/build_external_panel.py  panel de la fuente de la incidencia externa (un hito, ventana de 30 d) + embudo
                         por mercado (`--counts-only`)
scripts/fit_external_incidence.py  compuertas G1/G2, ajuste congelado e I1 de la incidencia externa (solo la fuente)
scripts/build_window_survival_panel.py  panel v1 + reloj en días y ventana (`panel_survival_ps.parquet`) + embudo (`--counts-only`)
scripts/eval_window_label.py  una corrida contra la referencia con la etiqueta dura y la corregida por ventana, pisos y
                         veredicto del preregistro (`--diagnose`: diagnóstico posterior, no preregistrado)
scripts/build_km_window_panel.py  panel del finalista + tramo en riesgo de la ventana en km (`panel_survival_kmw.parquet`,
                         F6) + embudo (`--counts-only`)
scripts/smooth_scores.py media acumulada causal del score por vehículo sobre una corrida existente (F6 K1/K3), con (a0)/(b)
scripts/audit_detection_null.py  la detección contra un nulo que conserva el largo de cada historial (regla 6) +
                         bootstrap pareado por vehículo contra la referencia
scripts/audit_event_dating.py  ¿una marca de intervención (aceite, días sin uso, DPF con motor apagado) fecha el
                         evento? Solo dev, criterio en configs/data/event_dating.yaml (F8, no pasó)
scripts/cost_scenarios.py  punto de operación de K2 bajo escenarios de costo con fuente (configs/cost_scenarios_k2.yaml),
                         con el ahorro sobre el azar; reporte, nunca selección
scripts/decision_layer.py  capa de decisión sobre una corrida: curva a varios presupuestos de falsas alarmas con
                         su nulo, y el umbral fijado fuera de muestra (empírico y Neyman-Pearson)
scripts/explain_k2.py    explicabilidad de K2 de punta a punta (preregistro configs/explain_k2.yaml): reentrena y
                         verifica los folds, V1-V4, criterios, elección, mensajes al cliente (configs/explain_texts.yaml)
                         y casos → experiments/explain-k2/; scripts/explain_k2_report.py hace las figuras y cases.md
scripts/eval_timesfm.py  TimesFM zero-shot en los cortes del panel v1 (mide solo dev) + forecasts.parquet
scripts/build_timesfm_panel.py  panel_timesfm.parquet = panel v1 + feat_tfm_* (mismas filas)
scripts/build_history_panel.py  panel_history.parquet = panel_survival + feat_*_hist_delta (mismas filas; verifica
                         que el lado de la ventana reproduzca el panel v1)
scripts/audit_history_univariate.py  ROC en dev y ρ con mes/cut_odo/largo de historia de las feat_*_hist_delta
scripts/train.py         entrypoint único de entrenamiento
scripts/audit_mil_bagsize.py  ¿el lift por vehículo es señal o tamaño de bolsa? (nulo de permutación)
scripts/audit_ordinal_horizon.py  las tres auditorías obligatorias de cualquier corrida: permutación
                         (nulo global y nulo intra-vehículo), aux_ de calendario como feat_, importancias
scripts/compare.py       tabla comparativa de corridas (markdown)
scripts/results.py       registro versionado en results/: métricas + config completa por corrida (log/table/show)
scripts/ensemble_rank.py ensamble por rango de corridas existentes (mismas filas y folds, verificado), medido con
                         `evaluate_predictions` y juzgado con la regla del preregistro; `--audit` agrega (a0) y (b)
scripts/rescore_run.py   re-mide una corrida vieja desde su predictions.parquet con la misma cuenta que train.py
                         (`evaluate_predictions`), sin reentrenar; falla si lo ya medido no se reproduce
scripts/dashboard.py     dashboard genérico de una corrida (streamlit; el del panel dummy)
scripts/dashboard_k2/    dashboard de F4 con K2, solo dev: `streamlit run scripts/dashboard_k2/app.py`. Resumen (dial de
                         falsas alarmas contra el azar, punto fuera de muestra), Vehículo (score, umbral, alerta, por qué
                         con el waterfall y el mensaje al cliente, perfil de uso contra los sanos), Costos (punto de
                         operación y ahorro según costos y prevalencia real, con escenarios precargados) y Modelo
                         (contra la referencia y los pisos, límites); configs/dashboard_k2.yaml
scripts/check_setup.py   smoke test del harness (209 chequeos)
scripts/eda_raw.py       diagnóstico de F1 sobre los crudos; deja CSVs en experiments/eda/
scripts/build_eda_cache.py  cache dev-only del EDA (una pasada por los crudos) + paleta,
                         diccionario de 3 vías y factibilidad de las features del plan §4
scripts/dashboard_eda.py dashboard del EDA de datos crudos, dev-only (streamlit)
docs/memoria/            hallazgos y decisiones, con la evidencia para reproducirlos
```

## Features ya implementadas

Las 53 `feat_*` del panel v1 están en `configs/data/features_v1.yaml`, por familia
del plan §4, y `scripts/make_dummy.py::FEATURE_SPECS` replica los mismos nombres para
que B y C trabajen contra el mismo esquema. Antes de crear una feature nueva,
revisar el YAML para no duplicar con otro nombre. Resumen:

- **A térmica / trayectos cortos (17):** `idle_frac`, `idle_per_1000km`, `idle_frac_trend`,
  `short_trip_frac_5km` (entre viajes con desplazamiento), `trips_below_regime_temp_frac`,
  `below_regime_moving_frac`, `below_regime_frac_trend`, temperaturas de motor y
  refrigerante, `cold_start_frac`, `chained_trip_frac`, distancia y duración por viaje.
- **B regeneración (15), desde `trips.AirRegeneration*`:** `regenerations_per_1000km`
  (caídas > 15 puntos; 5 cuenta ruido), su tendencia, distancia entre regeneraciones y su tendencia,
  `km_since_last_regen`, nivel residual y de arranque de la regeneración, `dpf_end_*`,
  `dpf_positive_delta_frac`, `dpf_saturated_frac`, `filter_*`, `manual_regen_ever`.
- **C uso (6):** velocidad recalculada, fracción urbana, viajes por 1.000 km y por día,
  km por día, reposo mediano entre viajes. Temperatura ambiente: `aux_`.
- **D severidad (13):** tasas de `Message` por 1.000 km + indicadores `_ever` para las
  zero-inflated, `msg_abnormal_frac`, `msgs_per_1000km`, `oil_life_mean` y su pendiente
  (el delta intra-viaje es 0 siempre), consumo por 100 km.
- **Control de ventana (2):** `n_trips_window`, `window_km_covered`.

Medidos como variante del panel (mismas filas, mismos folds) y **fuera** del set base porque
no le suman al LightGBM de control: los resúmenes de TimesFM (`feat_tfm_*`,
docs/memoria/f3-timesfm-zeroshot.md), y el desvío de la ventana respecto de la historia
previa del vehículo (`feat_*_hist_delta`, `configs/data/features_history.yaml`), que no le
suma a survival stacking más allá del sorteo (docs/memoria/f3-desvio-historia.md).

Lo que **no** se construye y por qué: elevación y presión de neumáticos (no hay
columna), `accumulation_*` desde `signals` (es la misma variable que `AirRegeneration`),
marcador `Regenerations` (cortado el 25-05-2026, queda como `aux_`), interrupciones de
regeneración (`Stopped Cleaning Automatically` es 1 mensaje de cada 100.000).

## Flujo de trabajo

- Una rama por feature o experimento (`feat/...`, `exp/...`), `main` siempre
  funcional. El merge no espera revisión de otro: se mergea cuando la rama pasa el
  checklist, y lo que hay que entender queda escrito en `docs/memoria/decisiones.md`,
  no en un hilo de PR.
- Nunca dos personas editan el mismo archivo: Track A datos (`src/data`,
  `src/features`), Track B modelos (`src/models`, `src/training`), Track C
  evaluación (`src/eval`, dashboard).
- Antes de cada merge a `main`, pasar el checklist de trampas técnicas del plan §9 y
  correr `python scripts/check_setup.py`.
- Toda corrida que valga la pena se anota en `results/` (`python scripts/results.py
  log <corrida> --note "..."`) y se commitea con su `configs/exp_*.yaml`: `experiments/`
  y wandb no se versionan. La config del modelo final sale de ahí (`results.py show`).
  Detalle en `results/README.md`.
