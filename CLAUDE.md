# CLAUDE.md — contrato y reglas del repo

Ford Innovation Challenge III · *Data-Driven Powertrain Intelligence*: predicción
temprana de degradación de eficiencia de combustión en vehículos conectados.
El plan completo está en `plan-implementacion-ford.md`; este archivo es el
resumen operativo que hay que respetar al escribir código.

## Estado

Fase 0 cerrada (infraestructura + panel dummy + harness verde). F1 cerrada del
lado de los datos crudos: `configs/data/raw_sources.yaml` está auditado contra los
archivos reales. El bloqueante de la fecha del evento se cerró el 17-09 recortando
el universo del estudio. **F2 cerrada el 18-09**: `scripts/build_dataset.py`
construye `data/processed/panel.parquet` (W=1000, G=500, H=3000, Δ=500) con 53
features declaradas en `configs/data/features_v1.yaml`; el panel, los splits y el
holdout están publicados como wandb Artifacts (`panel-v1`, `test-split`; se suben con
`scripts/log_panel_artifact.py`) y `configs/exp_baserate.yaml` es el piso contra el
panel real. **Hay modelos desde el 19-09**: `logistic`, `logistic_l1`, `gbm` y `lgbm` en el registry,
con sus `configs/exp_*.yaml`. El mejor da PR-AUC 0,181 contra 0,125 de tasa base (1,44×) y
ROC 0,617 — apenas por encima de la mejor columna sola (0,60), que es la confirmación
multivariada de que **el panel tiene una sola dimensión de señal**
(`docs/memoria/f3-primer-modelo-y-emparejado-por-posicion.md`).
**Lo siguiente es F4 (dashboard contra el panel real)**, y dentro de F3 la decisión
del 19-09 es **features antes que modelos**: con ROC ≈ 0,58–0,60, un modelo mejor no
rescata un panel que no mide el mecanismo. El menú de features está en
`docs/f3-features-candidatas-fisica.md` (arranca por la contradicción del catalizador) y
el de modelos en `docs/f3-modelos-candidatos.md`. **Ese menú ya se agotó**: las 16
features de las seis candidatas están construidas (`configs/data/features_v2.yaml`,
`panel_v2.yaml`), medidas y ninguna gana; y la contradicción del catalizador se resolvió
**en contra** de la hipótesis del esfuerzo de control —los que fallan no regeneran más ni
cargan más por km—, así que la pregunta abierta ya no es qué feature falta sino **qué es
el evento** (`docs/memoria/f3-esfuerzo-de-control-y-dosis.md`). La §2.4 de ese menú (desvío contra uno
mismo + CUSUM) **se implementó, se midió y quedó apagada**: no le gana al nivel y heredaba
el atajo de posición.

**Antes de tocar los datos, leer [`docs/memoria/`](docs/memoria/README.md).** Ahí
están los hallazgos de F1/F2 y las decisiones tomadas, con la evidencia y el comando
que las reproduce. Nueve que cambian cómo se escribe el código:

- Los datos vienen en **dos cohortes de muestreo** (failed / not_failed) y la
  cohorte *es* la etiqueta: `IdentificationDate` nula ⇔ sin evento. Nunca entra
  como feature.
- El **universo del estudio son 364 vehículos, no 1081**. `IdentificationDate`
  mezcla dos convenciones de registro y la que no sirve es mayoría; además la
  convención es **del mercado**, así que se descartan los positivos sin fecha
  utilizable *y* los sanos de los mercados donde ningún evento es observable. El
  criterio vive en `src/data/usable.py` y se declara en `universe` de
  `configs/data/test_split.yaml`. **El panel se construye con los 364.**
- **El 77% de las filas positivas cae en el último cuarto de la serie de su vehículo**,
  contra el 30% de las sanas, y **eso no se arregla emparejando**: odómetro, mes y
  posición son estructuralmente incompatibles (un sano no puede estar en el mismo
  odómetro, el mismo mes y el final de su serie). Cerrar posición abre calendario, y el
  **eso ya está resuelto** (19-09): con `pseudo_event.enabled` (a cada sano se le sortea
  un evento ficticio y se le corta la serie ahí) **más** `label.window_only` (de todos
  los vehículos se conservan solo los `H/Δ` cortes previos al evento real o ficticio), la
  posición cae a **P = 0,49** y el modelo sube a **ROC 0,703** contra 0,617 del v1. Las
  dos claves hacen falta: la primera sola deja el atajo en 0,88, porque dentro de un
  vehículo la etiqueta *es* la posición. Vienen apagadas por default; el detalle está en
  `docs/memoria/f3-evento-ficticio-y-ventana-de-riesgo.md`. La posición viaja en el panel
  como `aux_cut_position`, calculada antes de muestrear: las auditorías usan esa columna,
  porque recalcularla sobre el panel ya muestreado la distorsiona.
- **Todo panel nuevo pasa la ablación de `aux_`** antes de que se le crea un número, y
  **se corre por familia, no en bloque**: la gruesa (`extra_prefixes: [aux_]`) detecta que
  hay un atajo, la fina dice cuál. Medido en los tres paneles, el que filtra es
  `aux_static_` (el sesgo de `Engine` de F1): +0,09 a +0,16 de ROC. La temperatura
  ambiente aporta +0,007 y el marcador `Regenerations` 0,000 — salvo donde el mes no se
  empareja, ahí el marcador sí se vuelve un reloj (+0,072). Configs:
  `configs/exp_l1_*_abl_*.yaml`.
- Hay **13 vehículos duplicados bajo dos códigos**: se colapsan con
  `src/data/dedupe.py` antes de cualquier split, o la regla 2 se viola en silencio.
- **`Engine` está excluido** del set base: `ENG_3` es el 36% de los sanos y el 0%
  de los fallados. Ojo: en el universo recortado `ModelSeries` lleva casi la misma
  información (el cruce es diagonal), así que excluir solo `Engine` no alcanza.
- El **test está congelado** desde antes de F2 (74 vehículos, `test_split.json`).
  El recorte a dev lo hace `test_split_masks()`, y `dev_mask` es pertenencia
  explícita a `dev_vehicles`, **no** el complemento de test.
- **El marcador `signals.Regenerations` se corta el 25-05-2026** para toda la flota.
  Una tasa de marcadores por km mide el calendario, y el calendario mide la etiqueta
  (los eventos caen entre sep-2025 y mar-2026; la exposición sana, en 2026). La
  familia B se cuenta desde las caídas de `trips.AirRegeneration*`; el marcador y
  `DistanceBetweenRegenerations` quedan como `aux_`.
- **Los sanos se emparejan por odómetro y por mes** (`sampling` en `panel_v1.yaml`).
  Solo por odómetro, tres columnas de calendario suben el ROC de 0,67 a 0,76; con mes,
  de 0,57 a 0,60. Cualquier feature con deriva temporal (temperatura ambiente,
  `ProductionDay`, `daysUntilSale`) va como `aux_`, no como `feat_`/`static_`.
- **El 35% de las filas de `trips` son idle de 0 km** (motor encendido sin moverse) y
  son la señal que más anticipa. Toda fracción "de viaje" se calcula entre los que se
  mueven; `KilometerPerHour` es nulo exactamente ahí y se recalcula como km/duración.
- **La posición de un corte dentro de la serie de su vehículo separa con P = 0,83**, más
  que la mejor feature del panel. Es construcción, no física: las filas positivas son los
  últimos `H/Δ` cortes de una serie que termina en `E − G`. Por eso **ninguna feature
  puede ser monótona en el índice de corte** (acumuladores, contadores, `km_since_last_*`,
  estadísticos con ventana expandida), y cualquier gradiente "hacia el evento" se compara
  contra sanos **en la misma posición**. Se audita con `scripts/audit_sequence.py`.
- **Los viajes de un vehículo se recorren por `TripNumber`, no por fecha.** El 0,67%
  comparte `TripDatetimeStart` y ordenar por fecha baraja esos empates: el odómetro parece
  retroceder en 231 vehículos de 290 cuando en realidad son 8.

> ⚠️ **19-09: hay una falla en los eventos y la mentora los va a corregir.** Todo número
> medido contra `label` —las tres auditorías de F3 incluidas— es provisorio hasta que
> llegue la corrección, y **no se promueve ninguna feature al set base hasta entonces**.
> La maquinaria (columnas, `derived:`, agregadores, auditorías) es independiente de la
> etiqueta: re-medir es reconstruir el panel y volver a correr `audit_sequence.py`.

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
| `aux_*` | mixto | **en el panel, fuera del modelo**: `aux_static_{Engine, ModelSeries, ProductionDay, daysUntilSale}`, `aux_air_temp_*`, `aux_regen_marker_per_1000km`, controles de ventana. Para ablaciones y auditorías sin reconstruir |

**Regla de prefijos:** toda columna que entra a un modelo se llama `feat_` o
`static_`, y `src/training/cv.py` la selecciona por prefijo. Agregar una feature
no requiere tocar el código de entrenamiento ni coordinar con nadie: es una línea
en `configs/data/features_v1.yaml` (`name`, `source`, `column`, `agg`; `aux: true`
la deja fuera del modelo).

## Reglas que no se negocian

1. **Gap de blanking.** El modelo nunca ve los km inmediatamente previos al
   evento. Sin G esto es detección reactiva, que es justo lo que Ford ya tiene.
2. **Split agrupado por vehículo.** Un `vehicle_id` nunca cae en train y
   validación a la vez. La lógica está centralizada en `src/eval/splits.py` y no
   se reimplementa en ningún otro lado. Arriba de la CV hay un **holdout dev/test
   congelado** (`data/processed/test_split.json`, sorteado antes de F2), con tres
   listas: `dev_vehicles` (290), `test_vehicles` (74) y `excluded_vehicles` (717,
   fuera del universo). El panel se construye con los **364** del universo, y
   **todo lo que se mira es dev**: la CV, la selección de modelo y cualquier figura
   salen de `dev_mask`. Las filas de test existen y nadie las toca hasta que el
   modelo está elegido. El recorte lo hace `scripts/train.py` (`select_dev()`) antes
   de armar los folds, nunca filtrando `vehicle_id` a mano. Dentro de dev, los folds
   se estratifican por `label` a nivel vehículo —la variable que mide el PR-AUC, no
   `event_observed`—, ningún fold puede quedar con menos de `min_valid_positives`
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
   traducir al eje de km —para los vehículos del universo cae en una mediana de
   7.987 km con el 39% del historial por delante—. Eso no asciende al eje de días:
   sigue siendo reporte secundario, y el origen se estima una vez y se congela.
5. **Métricas.** PR-AUC out-of-fold para seleccionar modelo, curva de
   anticipación vs. falsas alarmas para el pitch, accuracy nunca. **PR-AUC y F1
   dependen de la prevalencia**, que en este proyecto cambia según cómo se construya el
   panel (0,125 en el v1, 0,558 en el del evento ficticio): entre paneles distintos se
   comparan `roc_auc` y `pr_auc_norm` = (AP − π)/(1 − π), nunca el PR-AUC crudo. El F1
   va con su `f1_trivial` = 2π/(1+π) al lado, que es lo que saca decir "positivo"
   siempre. Los cinco los devuelve `classification_metrics`.
5b. **Todo número bueno pasa el test de permutación** (`scripts/permutation_test.py`):
   la misma CV con la etiqueta permutada a nivel vehículo. El nulo **no es 0,50 por
   definición** —en el panel v1 es 0,555, porque las positivas son los últimos cortes de
   su vehículo—, y contra ese nulo el 0,617 del v1 no se distingue del ruido (p = 0,20).
   En el panel del evento ficticio el nulo vuelve a 0,501 y el 0,703 da p < 0,001.
6. **Un PR-AUC sospechosamente alto se audita antes de celebrarse.** Variables
   como el nivel del DPF son casi la definición del evento: sin gap, el modelo
   memoriza en vez de predecir.
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
src/data/loader.py       carga de las tres tablas crudas (esquema en configs/data/raw_sources.yaml)
                         load_table() entera, iter_table() por chunks (13M de filas)
src/data/dedupe.py       colapso de los 13 vehículos duplicados (lista en configs/data/vehicle_dedupe.yaml)
src/data/usable.py       universo del estudio: qué vehículo entra y por qué los demás no
src/data/join.py         unión a nivel vehículo + enriquecimiento de trips/signals con las estáticas
                         (trips y signals NO se mergean entre sí: no hay clave fila a fila)
src/data/subset.py       trips/signals para un conjunto de vehículos: canoniza → filtra → deduplica la fila completa
src/data/anchor.py       origen del calendario (estimado sobre dev, congelado en panel_meta.json) y odómetro del evento
src/data/panel.py        cortes en grilla de Δ, etiqueta con gap y horizonte, censura, QC de ventana, emparejado de sanos
src/features/trips.py    derivadas a nivel viaje (idle/moving, velocidad recalculada, topes físicos, regen = caída de AirRegeneration)
src/features/signals.py  una booleana por nivel de Message; regen_marker solo como aux
src/features/windows.py  primitiva de ventana (c−W, c] sobre odómetro + agregadores (per_1000km, half_*, gap_*, km_since_last…)
src/features/sequence.py el vehículo contra su propio pasado: `_zself` (expanding+shift, solo hacia atrás),
                         índice de degradación con signos físicos, CUSUM y racha. Se corre entre build_panel y el emparejado
src/models/registry.py   get_model(name, params); agregar un modelo = registrar un builder
src/training/cv.py       loop de CV agrupada; selección de features por prefijo
src/eval/splits.py       splits antileakage + serialización a splits.json
                         estratificación (columna/nivel), guarda de positivos por fold y CV repetida: todo del YAML
src/eval/metrics.py      PR-AUC/ROC/Brier + lead_time_curve() + false_alarm_rate() + bootstrap
src/eval/plots.py        figuras compartidas entre dashboard e informe
scripts/make_dummy.py    panel dummy con el esquema del contrato
scripts/make_test_split.py  auditoría del join + sorteo dev/test + recorte al universo (se corre una vez)
scripts/build_dataset.py panel real: universo del holdout → crudos → evento en km → cortes/etiqueta/features →
                         sanos emparejados → panel.parquet + splits.json (folds sobre dev) + panel_meta.json
scripts/eda_gaps.py      complemento del EDA sobre dev: factibilidad de W/G/H, perfil alineado al evento,
                         post-evento, calendario, ICC intra-vehículo (experiments/eda/dev/gaps/)
scripts/log_panel_artifact.py  publica panel.parquet + splits.json + panel_meta.json (`panel-v1`) y
                         test_split.json (`test-split`) como wandb Artifacts
scripts/make_splits.py   rearma splits.json sobre un panel que ya existe (cambiar folds no es reconstruir el panel)
scripts/train.py         entrypoint único de entrenamiento
scripts/compare.py       tabla comparativa de corridas (markdown)
scripts/dashboard.py     dashboard de resultados de modelo (streamlit)
scripts/check_setup.py   smoke test del harness (34 chequeos)
scripts/audit_sequence.py  ¿una feature separa, o mide la posición del corte en su serie? (dev-only, no escribe)
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
  (caídas > 5 puntos), su tendencia, distancia entre regeneraciones y su tendencia,
  `km_since_last_regen`, nivel residual y de arranque de la regeneración, `dpf_end_*`,
  `dpf_positive_delta_frac`, `dpf_saturated_frac`, `filter_*`, `manual_regen_ever`.
- **C uso (6):** velocidad recalculada, fracción urbana, viajes por 1.000 km y por día,
  km por día, reposo mediano entre viajes. Temperatura ambiente: `aux_`.
- **D severidad (13):** tasas de `Message` por 1.000 km + indicadores `_ever` para las
  zero-inflated, `msg_abnormal_frac`, `msgs_per_1000km`, `oil_life_mean` y su pendiente
  (el delta intra-viaje es 0 siempre), consumo por 100 km.
- **Control de ventana (2):** `n_trips_window`, `window_km_covered`.
- **E secuencia (7), `src/features/sequence.py` — APAGADA**, `sequence.enabled: false`.
  Se midió y no le gana al nivel; el detalle está en
  `docs/memoria/f3-secuencia-zself-cusum.md`. Reactivarla es cambiar una clave. Produce: un `feat_<x>_zself` por componente del índice —desvío estandarizado del
  corte contra los cortes **anteriores del mismo vehículo**—, más
  `feat_degradation_index` (media de los `zself` con el signo físico de cada uno),
  `feat_degradation_cusum` (evidencia acumulada, `S_t = max(0, S_{t−1} + idx_t − k)`) y
  `feat_degradation_run` (cortes consecutivos por encima de `k`). No se declaran en
  `features_v1.yaml`: no son agregados de una ventana sino de la **serie de cortes**, y
  por eso viven en su propio bloque. Los primeros `min_history` cortes de cada vehículo
  salen NaN a propósito.

- **F3 candidatas físicas (16), `configs/data/features_v2.yaml` — MEDIDAS Y NEGATIVAS.**
  Eficiencia y residuo de la regeneración, ciclos que no terminan, dosis de km fríos,
  temperatura condicionada por el largo del viaje y oportunidad de regenerar. Las
  columnas derivadas están en `src/features/trips.py` y el agregador `slope_per_1000km`
  en `windows.py`; el panel que las lleva es `panel_v2.yaml`, que **no** reemplaza al v1.
  Antes de proponer una de estas otra vez, leer
  `docs/memoria/f3-esfuerzo-de-control-y-dosis.md`.
- **F3 relaciones (familia F) y literatura de DPF (familia G), `features_v4.yaml`.** Una
  feature también puede ser **una relación entre dos features**: el bloque `derived:`
  (`ratio`/`product`/`diff` sobre columnas de la misma fila, `src/features/derived.py`).
  De ahí sale lo único que le gana al panel hasta ahora —`regen_per_idle_min`, P = 0,370
  contra 0,597 de `idle_frac`, que aguanta las tres auditorías—, y de la familia G, nada
  (los cinco mecanismos publicados de degradación de DPF dan planos). Detalle en
  `docs/memoria/f3-relaciones-y-literatura-dpf.md`.
- **F3 rarezas (familia H), `features_v5.yaml` — MEDIDAS Y PLANAS.** El reloj (idle
  nocturno), el viaje anterior (constante de enfriamiento, rearranque en caliente) y la
  forma del reparto (Gini de km, CV térmico), con los agregadores `top_share`/`gini`/`cv`.
  **Antes de proponer una feature nueva, leer `docs/memoria/f3-barrido-de-relaciones.md`:**
  el barrido de los 7.310 cocientes posibles ya se corrió con su nulo y no pasa nada la
  barra del azar. Toda feature nueva se testea por permutación a nivel vehículo, no por
  su P a secas.

Lo que **no** se construye y por qué: elevación y presión de neumáticos (no hay
columna), `accumulation_*` desde `signals` (es la misma variable que `AirRegeneration`),
marcador `Regenerations` (cortado el 25-05-2026, queda como `aux_`), interrupciones de
regeneración (`Stopped Cleaning Automatically` es 1 mensaje de cada 100.000).

## Flujo de trabajo

- Una rama por feature o experimento (`feat/...`, `exp/...`), `main` siempre
  funcional, merge solo por PR con revisión de otro.
- Nunca dos personas editan el mismo archivo: Track A datos (`src/data`,
  `src/features`), Track B modelos (`src/models`, `src/training`), Track C
  evaluación (`src/eval`, dashboard).
- Antes de cada PR, pasar el checklist de trampas técnicas del plan §9 y correr
  `python scripts/check_setup.py`.
