# Plan de implementación — Predicción temprana de degradación de eficiencia de combustión en vehículos conectados

Documento operativo del Ford Innovation Challenge III, desafío *Data-Driven Powertrain Intelligence*. Pensado para tres personas trabajando en paralelo, alineado con las mismas pautas de flujo del TP final (`src/`, configs YAML, `scripts/train.py`, wandb, sweeps, PRs, bitácora en Docs).

---

## 0. Lectura técnica del problema (esto condiciona todo el plan)

Cinco decisiones que, si se hacen mal, invalidan toda comparación posterior:

**1. El dataset de modelado no existe: hay que construirlo.** Esta es la diferencia estructural más grande respecto del TP final. Allá la matriz X venía dada y por eso EDA, baselines y reducción de dimensionalidad podían correr en paralelo desde el día uno. Acá tenemos tres tablas crudas (viajes, señales dinámicas, info del vehículo) y ninguna fila de entrenamiento hasta que alguien defina qué es una observación. Toda la carga conceptual del proyecto está en esa definición, no en la elección del modelo. En consecuencia, el mecanismo de paralelización cambia: lo que desbloquea a los otros dos integrantes no es que los datos existan, sino un **panel dummy con el esquema acordado**, generado en la primera hora.

**2. Es un problema de anticipación, no de detección.** Si el modelo mira datos inmediatamente previos al evento, va a aprender a leer un filtro de partículas ya saturado, que es exactamente el análisis reactivo que Ford dice que ya tiene. La defensa es un **gap de blanking** entre el punto de corte y la ventana de evento: se predice a partir de información que termina G kilómetros antes. El gap es lo que hace que el trabajo responda al desafío; sin él los números van a salir hermosos y mentirosos, igual que un CV aleatorio sobre años mezclados.

**3. El eje temporal no está resuelto y puede no resolverse.** `IdentificationDate` está en días desde producción y `ProductionDay` en días desde el primer vehículo de la lista, mientras que `TripDatetimeStart` y `EventTimestamp` son fechas de calendario. Alinear ambos ejes requiere un anclaje que el enunciado no garantiza. Por eso el plan se construye por defecto sobre el **eje de odómetro**, que es intrínseco al vehículo, no necesita anclaje y coincide con lo que el desafío pide reportar ("tiempo/kilometraje"). Si el anclaje temporal cierra, se agrega el eje de días como reporte secundario.

**4. El n efectivo son los vehículos con evento, no los viajes.** Puede haber millones de filas de telemetría y aun así ochenta vehículos etiquetados. Ese número, que se conoce en la primera hora de F1, define la escalera de modelos completa. Con pocos eventos, la regularización fuerte y el split por grupo no son opcionales, y una red recurrente sobre secuencias de viajes va a sobreajustar y perder contra un gradient boosting. Vale la misma expectativa honesta que tuvimos con deep learning en genómica: el proyecto no es "la red gana", es "¿la estructura temporal aporta algo medible sobre agregados de ventana?". Que la respuesta sea que no es un resultado válido y defendible.

**5. Antileakage en todo el armado del panel.** Ninguna feature puede usar información posterior al punto de corte. El escalado y cualquier imputación se ajustan solo con train. Un mismo vehículo nunca se parte entre train y test (split agrupado por VIN). Y hay una trampa específica de este dataset: variables como el nivel del DPF o la acumulación del filtro de aire son casi la definición del evento, así que si entran sin gap el modelo no predice, memoriza. Un PR-AUC de 0,95 en el primer intento es motivo de sospecha, no de festejo.

---

## 1. Mapa de fases y paralelización

| Fase | Depende de | Corre en paralelo con | ¿Bloqueante? |
|---|---|---|---|
| F0 Infraestructura | — | (interna: datos ∥ harness ∥ eval) | Sí, bloquea todo |
| F1 Auditoría y contrato | F0 (loader) | — | Sí, fija etiqueta, eje y splits |
| F2 Panel y features | F1 | F3, F4 | Habilita los números reales |
| F3 Baselines | F0 + contrato | F2, F4 | Es el piso de referencia |
| F4 Evaluación y visualización | F0 + contrato | F2, F3 | No, pero define la métrica del pitch |
| F5 Modelos avanzados | F2 (panel real) | 5a ∥ 5b | No |
| F6 Consolidación y pitch | todo | — | Cierre |

**Regla de oro de paralelización:** F0 y F1 son cuellos de botella absolutos y hay que atravesarlos rápido. Una vez congelado el contrato de datos al final de F1, F2, F3 y F4 corren en simultáneo: F2 construye el panel real mientras F3 y F4 trabajan contra el panel dummy, que tiene exactamente el mismo esquema. Cuando el panel real aterriza, F3 y F4 solo cambian el path del archivo. La única sincronización dura intermedia es que ningún número se anota como "oficial" hasta que el panel real y los splits estén congelados.

### Asignación a tres personas

Respetando "uno arma el pipeline de datos, el otro los modelos" y "nunca editen el mismo archivo", con una tercera persona dedicada a evaluación y producto (que en este desafío pesa tanto como el modelo: los criterios incluyen explícitamente claridad de visualización, explicabilidad y power pitch).

- **Track A (datos):** F0-datos → F1 auditoría → F2 panel y features → F5a árboles → F6.
- **Track B (modelos):** F0-harness → F3 baselines → F5b supervivencia y variante secuencial → F6.
- **Track C (evaluación y producto):** F0-eval → F4 métricas de anticipación y dashboard → SHAP → F6 pitch. Transversalmente, es quien audita leakage: revisa cada feature nueva preguntando si pudo haberse medido antes del corte.

---

## 2. Fase 0 — Infraestructura y contrato de datos

**Objetivo:** que exista una corrida end-to-end verde (panel dummy → modelo dummy → métrica → wandb) antes de que nadie escriba un modelo real ni mire un dato real. Esta es la fase que estamos por hacer ahora.

### Scaffold del repo

```
ford-hackathon/
├── src/
│   ├── data/          # loaders, join de las tres tablas
│   ├── features/      # agregados de ventana móvil
│   ├── models/        # registry: get_model(name, params)
│   ├── training/      # loop de CV
│   └── eval/          # métricas, splits antileakage
├── configs/           # un YAML por experimento
├── notebooks/         # solo exploración, sin lógica
├── scripts/           # build_dataset.py, train.py, make_dummy.py
├── experiments/       # outputs y checkpoints (gitignored)
├── data/              # datos crudos y procesados (gitignored)
├── CLAUDE.md
├── README.md
└── requirements.txt
```

### Track A — datos

- Scaffold del repo y `.gitignore` (`data/`, `experiments/`, `wandb/`, checkpoints).
- `requirements.txt` con versiones pinneadas; semillas de numpy y sklearn seteadas y guardadas dentro del config.
- `src/data/loader.py`: carga las tres tablas crudas con dtypes explícitos y parseo de fechas. Rutas por config o variable de entorno, nada hardcodeado.
- `scripts/make_dummy.py`: **genera `data/processed/panel_dummy.parquet`** con el esquema exacto del contrato y valores aleatorios plausibles. Este es el entregable que desbloquea a B y a C.
- Redacción del contrato de datos (abajo) en la bitácora.

### Track B — harness

- `scripts/train.py --config configs/exp_xxx.yaml`: entrypoint único. Levanta el config, fija la semilla, instancia el modelo desde el registry, corre la CV agrupada y loguea a wandb.
- `src/models/registry.py` con un `DummyClassifier` (predictor por la tasa base) para probar el loop completo.
- `src/eval/splits.py`: **split agrupado por vehículo**, estratificado por presencia de evento, 5 folds, serializado a `data/processed/splits.json`. Acá vive la lógica antileakage, centralizada, para que ningún modelo se la saltee.
- Proyecto de equipo en wandb con los tres como miembros.

### Track C — evaluación

- `src/eval/metrics.py`: PR-AUC, ROC-AUC, Brier, y sobre todo las dos métricas propias del desafío, `lead_time_curve()` y `false_alarm_rate()` (definidas en §5). Se implementan contra predicciones aleatorias sobre el panel dummy.
- Esqueleto del dashboard (una app mínima que levanta un parquet de predicciones y dibuja las curvas), también contra dummy.
- Apertura de la bitácora en Docs y del board de Issues y Projects.

### Contrato de datos

Se congela al final de F1 y después solo se toca por acuerdo explícito de los tres. Artefacto: `data/processed/panel.parquet`, una fila por par `(vehículo, punto de corte)`.

| Columna | Tipo | Descripción |
|---|---|---|
| `vehicle_id` | str | identificador unificado entre las tres tablas |
| `cut_odo` | float | odómetro en el punto de corte [km] |
| `cut_date` | datetime | fecha del corte (nullable si no hay anclaje) |
| `horizon_km` | float | horizonte de anticipación H usado en esta fila |
| `gap_km` | float | gap de blanking G usado en esta fila |
| `label` | int | 1 si el evento cae en `[corte+G, corte+G+H]` |
| `time_to_event_km` | float | km hasta el evento; NaN si censurado |
| `event_observed` | int | 1 si el vehículo tiene evento registrado |
| `feat_*` | float | todas las features de ventana |
| `static_*` | mixto | `Engine`, `ModelSeries`, `SalesCountryCd`, `daysUntilSale`, `ProductionDay` |

**Regla fija:** toda columna que entra a un modelo se llama `feat_` o `static_`, y `train.py` selecciona por prefijo. Agregar una feature no requiere tocar el código de entrenamiento ni coordinar con nadie.

**Entregable de F0:** corrida dummy logueada en wandb con su config completo, sobre el panel dummy, con las métricas de anticipación ya calculadas (y dando basura, como corresponde a un modelo aleatorio). Branch `feat/infra`, merge por PR.

---

## 3. Fase 1 — Auditoría, EDA y lock del protocolo

Depende solo del loader de F0. Es corta y bloqueante: no se puede construir el panel sin cerrarla.

- Conteos básicos: vehículos únicos por tabla, **solapamiento de identificadores entre `Vin`, `VehicleCode` y `VehCode`** (si no joinean, el plan cambia de raíz), vehículos con `IdentificationDate` no nulo, rango de odómetro y de fechas, porcentaje de nulos por columna, duplicados.
- Composición de la flota por `Engine` y `ModelSeries`. El DPF solo aplica a diésel; si la flota es mixta hay que segmentar, no promediar poblaciones distintas.
- Distribución del kilometraje al evento y de la duración del histórico por vehículo. Esto define qué combinaciones de ventana W, gap G y horizonte H son siquiera factibles: si el histórico mediano es de 3.000 km, un horizonte de 5.000 no existe.
- Perfil de uso comparado: distribución de distancia por viaje, temperatura máxima de motor y `DistanceBetweenRegenerations` en vehículos con y sin evento. Es la evidencia directa de la hipótesis física y la primera figura del pitch.
- **Decisión y lock de tres cosas:** (a) eje temporal (odómetro por defecto), (b) definición de la etiqueta con sus W, G, H iniciales, (c) protocolo de validación (split agrupado por vehículo, 5 folds, estratificado por evento).

**Entregable:** notebook de auditoría + entrada en la bitácora con esas tres decisiones cerradas y el contrato de datos congelado.

---

## 4. Fase 2 — Panel y features

Depende de F1. Corre en paralelo con F3 y F4.

`scripts/build_dataset.py --config configs/data/panel_v1.yaml` materializa el panel: recorre la vida de cada vehículo cada Δ km, calcula agregados sobre la ventana móvil hacia atrás de tamaño W, y arma la etiqueta con el gap G y el horizonte H. Para los vehículos sanos, los cortes se muestrean con la **misma distribución de odómetro** que los de los vehículos con evento, para que el modelo no aprenda "kilometraje alto ⇒ riesgo" en lugar del patrón de uso.

El enunciado prácticamente regala la hipótesis causal: trayectos cortos, conducción urbana intensiva y operación que no permite completar los ciclos de recuperación. Traducido: **el motor no alcanza temperatura de régimen, las regeneraciones no se completan, el hollín se acumula y cae la eficiencia**. Las features tienen que medir ese mecanismo, no ser un volcado de agregados; eso es lo que hace que el SHAP de F5 cuente una historia en vez de listar columnas.

Cuatro familias, en orden de prioridad:

- **A · Régimen térmico y trayectos cortos.** Fracción de viajes bajo 5 km; fracción de viajes donde `EngineTemperatureMax` no supera el umbral de régimen; mediana de `EngineTemperatureAvg`; amplitud térmica por viaje; `CoolantTemperatureEnd` medio; distancia mediana y percentil 25 por viaje.
- **B · Salud del ciclo de regeneración.** Pendiente de `DieselParticulateFilterEnd` contra odómetro; nivel medio y máximo de saturación; fracción de viajes con delta de DPF positivo (carga sin descarga); regeneraciones completadas desde `Regenerations`; **tendencia de `DistanceBetweenRegenerations`**, probablemente la señal más fuerte del dataset; frecuencia de regeneración manual; nivel y pendiente de `Acumulation` normalizada por 1.000 km.
- **C · Contexto de uso y ambiente.** `KilometerPerHour` medio y fracción de viajes bajo 30 km/h (proxy urbano); km por día y viajes por día; tiempo entre viajes; `AirTemperatureAvg` y `AirTemperatureMin` (el frío agrava la familia A); elevación media y rango (altitud, ligada directamente al ingreso de oxígeno que menciona el título del desafío).
- **D · Severidad y proxies baratos.** Caída de `EngineOilLifePC` por cada 1.000 km (el vehículo ya calcula un índice de severidad de uso: conviene aprovecharlo); presión media de neumáticos y fracción de viajes bajo umbral.

Todas se calculan con la misma primitiva en `src/features/windows.py`, que recibe una lista de columnas y una lista de agregadores. Agregar una feature es agregar una línea en un YAML.

**Entregable:** `panel.parquet` v1 + `splits.json` como wandb Artifacts, para que B y C se los bajen en vez de reconstruirlos. Branch `feat/panel`.

---

## 5. Fase 3 — Baselines · Fase 4 — Evaluación

Ambas arrancan contra el panel dummy y se revalidan contra el panel real apenas existe.

### F3 · Baselines (Track B)

De menor a mayor:

- **Tasa base** (piso absoluto; si un modelo no le gana, está roto).
- **Regla física:** umbral sobre la pendiente de saturación del DPF o sobre la tendencia de `DistanceBetweenRegenerations`. Es lo que un ingeniero de Ford haría hoy sin ML. Si el modelo no lo supera claramente no hay proyecto, y conviene saberlo temprano.
- **Regresión logística regularizada** sobre las features de ventana: rápida, interpretable y sorprendentemente competitiva cuando hay pocos eventos.
- **Gradient boosting** (XGBoost o LightGBM) con búsqueda de hiperparámetros vía wandb Sweeps, más un barrido sobre la terna (W, G, H) que es el hiperparámetro *del problema*, no del modelo.

**Entregable:** tabla de baselines en wandb con números congelados. Branch `exp/baselines`.

### F4 · Evaluación y visualización (Track C)

Tres métricas con jerarquía explícita:

1. **PR-AUC out-of-fold** como métrica de selección de modelo. ROC-AUC de apoyo, accuracy nunca (el problema está desbalanceado por construcción).
2. **Curva de anticipación vs. falsas alarmas**, que es la métrica del pitch. Para cada umbral: en el eje x, falsas alarmas por cada 1.000 vehículos sanos; en el eje y, mediana de anticipación en km de la primera alerta sostenida en los vehículos con evento. "Alerta sostenida" = el score supera el umbral en K cortes consecutivos, con K de 2 o 3; sin esa condición un pico de ruido cuenta como acierto y la anticipación reportada queda inflada. Esta curva permite decir una frase concreta y verificable: *a una falsa alarma cada X vehículos, detectamos el 70% de los casos con una mediana de N km de anticipación*.
3. **Significancia:** intervalos por bootstrap sobre los folds. Que una diferencia exista no significa que sea real.

Sobre eso se monta el dashboard: perfil de uso de un vehículo en el tiempo, score de riesgo con su umbral, y la curva de anticipación como pieza central.

**Entregable:** módulo de métricas + dashboard funcional. Branch `feat/eval`.

---

## 6. Fase 5 — Modelos avanzados

Depende del panel real. 5a y 5b se paralelizan entre Track A y Track B. Todo se evalúa con los mismos folds y se compara contra los baselines de F3.

**5a · Árboles y ensambles.** Profundización del gradient boosting: calibración de probabilidades, barrido fino de (W, G, H), manejo del desbalance por pesos en vez de resampleo, e importancia de features contrastada contra la hipótesis física.

**5b · Supervivencia.** Cox regularizado o XGBoost con objetivo AFT. Es la opción que recomiendo como diferencial técnico por tres razones: aprovecha la censura de los vehículos sanos en lugar de descartarla, predice directamente kilometraje hasta el evento (que es literalmente el objetivo declarado del desafío) y elimina la necesidad de fijar un horizonte H arbitrario.

**Stretch goal.** Una GRU sobre la secuencia de viajes, solo si el conteo de eventos de F1 resulta sorprendentemente alto. La decisión la toma ese número, no la agenda: si son menos de cien vehículos etiquetados, el tiempo rinde mucho más en el dashboard y el pitch que en una arquitectura que va a perder contra el boosting.

**Entregable:** resultados por modelo en wandb, todos linkeados. Branches `exp/trees` y `exp/survival`.

---

## 7. Fase 6 — Consolidación, explicabilidad y pitch

- Tabla final comparativa: regla física vs. logística vs. árboles vs. supervivencia, misma métrica, mismos folds.
- **SHAP** global (¿qué familia de features domina?) y por caso (un vehículo concreto con su explicación de por qué se alertó). Lo segundo es lo que convence en una demo.
- Contraste entre la importancia aprendida y la hipótesis física de §4. Si coinciden, es el argumento más fuerte del pitch: el modelo no es una caja negra, mide un mecanismo conocido.
- Narrativa honesta: cuánta anticipación se logra, a qué costo de falsas alarmas, y qué haría falta para llevarlo a producción.
- README del repo e informe con los links de wandb al lado de cada número, nada copiado a mano.

---

## 8. Cronograma orientativo

Dimensionado sobre unas 48 horas de trabajo efectivo. Si la ventana es más corta se recorta F5 y parte del barrido de F3; si es más larga, el excedente va a F5 y F4, nunca a F0 y F1.

| Franja | Track A (datos) | Track B (modelos) | Track C (evaluación) | Sincronización |
|---|---|---|---|---|
| 0–3 h | Scaffold + loader + panel dummy | `train.py` + wandb + splits | Métricas de anticipación | Corrida dummy verde |
| 3–6 h | Auditoría y EDA | Baselines contra dummy | Esqueleto de dashboard | **Contrato congelado** |
| 6–16 h | Panel real + features A y B | Regla física + logística | Dashboard contra dummy | Primer número real |
| 16–28 h | Features C y D | GBM + sweep de (W, G, H) | Curva de anticipación | Tabla de baselines |
| 28–36 h | 5a árboles | 5b supervivencia | SHAP + figuras | Comparación vs. baseline |
| 36–44 h | F6 | F6 | Dashboard final | Cierre de código |
| 44–48 h | Pitch de a tres | | | Ensayo cronometrado |

Check-in de diez minutos cada cuatro horas con tres preguntas fijas: qué cerré, qué me bloquea, qué cambia del contrato.

---

## 9. Checklist de trampas técnicas (revisar antes de cada PR)

- [ ] Ninguna feature usa información posterior al punto de corte.
- [ ] Gap de blanking aplicado; el modelo nunca ve los km inmediatamente previos al evento.
- [ ] Split agrupado por vehículo; ningún VIN aparece en train y test a la vez.
- [ ] Cortes de vehículos sanos muestreados con la misma distribución de odómetro que los de vehículos con evento.
- [ ] Escalado e imputación ajustados solo con el train de cada fold.
- [ ] Segmentación por tipo de motor cuando la feature solo aplica a diésel.
- [ ] PR-AUC como número de selección, curva de anticipación como número de portada, accuracy nunca.
- [ ] Un PR-AUC sospechosamente alto se audita antes de celebrarse.
- [ ] Cada número del informe linkea a su corrida de wandb.

---

## 10. Riesgos y planes de contingencia

| Riesgo | Señal temprana | Plan B |
|---|---|---|
| Las tablas no joinean | F1, primera hora | Modelar sobre la tabla con etiqueta cruzable y declararlo como supuesto en el pitch |
| Muy pocos vehículos con evento | F1 | Bajar la escalera: logística + GBM poco profundo, CV repetida, intervalos en vez de puntos |
| No se puede alinear el eje temporal | F1 | Panel sobre eje de odómetro (recomendado por defecto igual) |
| Histórico por vehículo demasiado corto | F1 | Reducir W y H; si no alcanza, pasar de panel a una sola observación por vehículo |
| Leakage por features post-evento | Métrica anómala en F3 | Auditoría de Track C sobre cada feature nueva |
| El dashboard queda pobre por falta de tiempo | Hora 32 sin nada visual | Track C no se distrae con modelado; un buen modelo mal mostrado pierde contra uno decente bien contado |

---

## 11. Mapeo al flujo de trabajo del repo

- Un YAML por experimento en `configs/`; nunca código editado a mano para cambiar un hiperparámetro.
- Una rama por feature o experimento (`feat/...`, `exp/...`); `main` siempre funcional; merge solo por PR con revisión de otro.
- wandb: `wandb.init()` loguea el config completo; el panel, los splits y los modelos van como Artifacts; para usar lo que generó otro, bajar el artifact en vez de reconstruirlo.
- Sweeps en `configs/sweep_xxx.yaml`, lanzados con `wandb agent <sweep_id>` dentro de `tmux`.
- Bitácora en Google Docs para decisiones y narrativa; README para instalación y cómo lanzar experimentos.
- `CLAUDE.md` con el contrato de datos, la regla del gap, la regla de split por vehículo, la convención de prefijos `feat_`/`static_` y la lista de features ya implementadas, para que Claude Code no duplique.
