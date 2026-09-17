# CLAUDE.md — contrato y reglas del repo

Ford Innovation Challenge III · *Data-Driven Powertrain Intelligence*: predicción
temprana de degradación de eficiencia de combustión en vehículos conectados.
El plan completo está en `plan-implementacion-ford.md`; este archivo es el
resumen operativo que hay que respetar al escribir código.

## Estado

Fase 0 cerrada (infraestructura + panel dummy + harness verde). F1 cerrada del
lado de los datos crudos: `configs/data/raw_sources.yaml` está auditado contra los
archivos reales. Falta F2 (panel real).

**Antes de tocar los datos, leer [`docs/memoria/`](docs/memoria/README.md).** Ahí
están los hallazgos de F1 y las decisiones tomadas, con la evidencia y el comando
que las reproduce. Tres que cambian cómo se escribe el código:

- Los datos vienen en **dos cohortes de muestreo** (failed / not_failed) y la
  cohorte *es* la etiqueta: `IdentificationDate` nula ⇔ sin evento. Nunca entra
  como feature.
- Hay **13 vehículos duplicados bajo dos códigos**: se colapsan con
  `src/data/dedupe.py` antes de cualquier split, o la regla 2 se viola en silencio.
- **`Engine` está excluido** del set base: `ENG_3` es el 36% de los sanos y el 0%
  de los fallados.
- El **test está congelado** desde antes de F2 (217 vehículos, `test_split.json`).
  El panel se construye con los 1081, pero se entrena y se compara **solo sobre
  dev**: el recorte lo hace `test_split_masks()`, no un panel más chico.

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
| `feat_*` | float | todas las features de ventana |
| `static_*` | mixto | `Engine`, `ModelSeries`, `SalesCountryCd`, `daysUntilSale`, `ProductionDay` |

**Regla de prefijos:** toda columna que entra a un modelo se llama `feat_` o
`static_`, y `src/training/cv.py` la selecciona por prefijo. Agregar una feature
no requiere tocar el código de entrenamiento ni coordinar con nadie.

## Reglas que no se negocian

1. **Gap de blanking.** El modelo nunca ve los km inmediatamente previos al
   evento. Sin G esto es detección reactiva, que es justo lo que Ford ya tiene.
2. **Split agrupado por vehículo.** Un `vehicle_id` nunca cae en train y
   validación a la vez. La lógica está centralizada en `src/eval/splits.py` y no
   se reimplementa en ningún otro lado. Arriba de la CV hay un **holdout dev/test
   80/20 congelado** (`data/processed/test_split.json`, generado antes de F2). El
   panel incluye a los 1081 vehículos, pero **todo lo que se mira es dev**: la CV,
   la selección de modelo y cualquier figura salen de `dev_mask`. Las filas de test
   existen y nadie las toca hasta que el modelo está elegido. El recorte lo hace
   `scripts/train.py` (`select_dev()`) antes de armar los folds, nunca filtrando
   `vehicle_id` a mano. Por eso **todo YAML de experimento declara
   `splits.test_split`** —el path del holdout, o `null` explícito si el panel no
   tiene—: si falta la clave, `train.py` no corre.
3. **Features solo hacia atrás.** Ninguna feature usa información posterior al
   punto de corte. Escalado e imputación se ajustan solo con el train de cada
   fold (van dentro del `Pipeline`, nunca sobre el panel entero).
4. **Eje de odómetro por defecto.** `IdentificationDate` está en días desde
   producción y `TripDatetimeStart` es calendario. F1 encontró el anclaje que
   faltaba (`ProductionDay` está en el eje del calendario, IQR de 0 días: ver
   `docs/memoria/f1-anclaje-temporal.md`), así que el evento **sí** se puede
   traducir al eje de km. Eso no asciende al eje de días: sigue siendo reporte
   secundario, y el origen se estima una vez y se congela.
5. **Métricas.** PR-AUC out-of-fold para seleccionar modelo, curva de
   anticipación vs. falsas alarmas para el pitch, accuracy nunca.
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
src/data/join.py         unión a nivel vehículo + enriquecimiento de trips/signals con las estáticas
                         (trips y signals NO se mergean entre sí: no hay clave fila a fila)
src/features/            [F2] windows.py: primitiva de agregación de ventana
src/models/registry.py   get_model(name, params); agregar un modelo = registrar un builder
src/training/cv.py       loop de CV agrupada; selección de features por prefijo
src/eval/splits.py       splits antileakage + serialización a splits.json
src/eval/metrics.py      PR-AUC/ROC/Brier + lead_time_curve() + false_alarm_rate() + bootstrap
src/eval/plots.py        figuras compartidas entre dashboard e informe
scripts/make_dummy.py    panel dummy con el esquema del contrato
scripts/make_test_split.py  auditoría del join + holdout dev/test congelado (se corre una vez)
scripts/build_dataset.py [F2] panel real
scripts/train.py         entrypoint único de entrenamiento
scripts/compare.py       tabla comparativa de corridas (markdown)
scripts/dashboard.py     dashboard de resultados de modelo (streamlit)
scripts/check_setup.py   smoke test del harness (15 chequeos)
scripts/eda_raw.py       diagnóstico de F1 sobre los crudos; deja CSVs en experiments/eda/
scripts/build_eda_cache.py  cache dev-only del EDA (una pasada por los crudos) + paleta,
                         diccionario de 3 vías y factibilidad de las features del plan §4
scripts/dashboard_eda.py dashboard del EDA de datos crudos, dev-only (streamlit)
docs/memoria/            hallazgos y decisiones, con la evidencia para reproducirlos
```

## Features ya implementadas

Ninguna real todavía: F2 las materializa. Los nombres que ya **están reservados**
por el panel dummy (`scripts/make_dummy.py`, `FEATURE_SPECS`) son los de las
cuatro familias del plan §4 — A térmica/trayectos cortos, B ciclo de
regeneración, C uso y ambiente, D severidad. Antes de crear una feature nueva,
revisar esa lista para no duplicar con otro nombre.

## Flujo de trabajo

- Una rama por feature o experimento (`feat/...`, `exp/...`), `main` siempre
  funcional, merge solo por PR con revisión de otro.
- Nunca dos personas editan el mismo archivo: Track A datos (`src/data`,
  `src/features`), Track B modelos (`src/models`, `src/training`), Track C
  evaluación (`src/eval`, dashboard).
- Antes de cada PR, pasar el checklist de trampas técnicas del plan §9 y correr
  `python scripts/check_setup.py`.
