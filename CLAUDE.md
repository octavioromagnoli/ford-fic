# Guía técnica del repositorio

Consultar [README.md](README.md) para instalar y ejecutar el proyecto y
[docs/reproducibilidad.md](docs/reproducibilidad.md) para preparar los insumos.
El [informe](docs/informe/informe.pdf) describe el método y los resultados.

## Modelo y configuración

El modelo finalista es una GRU con atención: 20 tramos de 50 km × 15 canales,
más país, motor y serie. El ensamble promedia rangos de las semillas 42, 1 y 2.
Las configuraciones son `configs/exp_v2all_gru_trips_estaticas{,_s1,_s2}_r3.yaml`
y `configs/exp_v2all_seeds3_gru_trips_estaticas.yaml`.

Los parámetros, semillas y rutas se declaran en YAML. Para cambiar una
configuración de experimento, crear otro YAML. Los modelos alternativos se
conservan en `src/models/`, registrados mediante `registry.py`.

## Contrato de datos

Una fila representa un vehículo y un corte de odómetro: `(vehicle_id, cut_odo)`.
El panel secuencial del finalista es
`data/processed/panel_seq_trips_v2_estaticas.parquet`, acompañado por su `_meta.json`.

| Columnas | Función |
|---|---|
| `vehicle_id`, `cut_odo`, `cut_date` | Identidad y posición del corte |
| `window_km`, `gap_km`, `horizon_km` | Ventana, separación y horizonte |
| `label` | Evento entre corte + gap y corte + gap + horizonte |
| `event_observed`, `time_to_event_km` | Información de evento y censura |
| `feat_seq_*` | Canales secuenciales aplanados |
| `static_*` | País, motor y serie del vehículo |
| `aux_*` | Controles y auditorías; no entran al modelo |

Solo `feat_*` y `static_*` se seleccionan como entradas del modelo. El panel
agregado de los modelos tabulares usa las variables de `configs/data/features_v1.yaml`.

## Validación y tratamiento de datos

- Aplicar las correcciones de la entrega v2 declaradas en `raw_sources.yaml`:
  renombre de la fecha del evento, ajuste de ProductionDay y corte común de extracción.
- Conservar el primer evento y colapsar vehículos duplicados antes del split.
- Usar el universo común de producción de `configs/data/test_split.yaml`.
- Referencia del evento: fecha registrada menos 21 días. Ventana de 1.000 km,
  gap de 500 km, horizonte de 3.000 km y cortes cada 500 km, según los YAML.
- Mantener los splits congelados. `src/eval/splits.py` centraliza su validación;
  un vehículo no puede pertenecer a entrenamiento y validación simultáneamente.
- `scripts/train.py` selecciona desarrollo antes de la CV. Imputación, escalado
  y codificación se ajustan únicamente con el entrenamiento de cada fold.
- Las features solo usan datos anteriores al corte. `cohort`, fechas del evento
  y variables auxiliares no se incorporan como predictores.
- El marcador `signals.Regenerations` tiene un corte de cobertura; las
  regeneraciones se derivan de caídas de acumulación, según los YAML.
- Distinguir viajes idle de viajes en movimiento y conservar el emparejamiento
  de sanos por odómetro y mes.

La métrica del informe es detección por vehículo a presupuestos fijos de falsas
alarmas, contrastada con un nulo que conserva la longitud del historial. PR-AUC
por fila no demuestra por sí sola anticipación. Comparar sobre los mismos datos
y splits, con bootstrap pareado por vehículo y controles dentro de mercado × motor.

El conjunto de prueba ya fue evaluado y no se utiliza para ajustar nuevas variantes.
De sus 111 vehículos, 103 tienen cortes evaluables. Como 36 vehículos habían
pertenecido al desarrollo de la entrega anterior, la evaluación incluye también
el subconjunto sin ellos. La GRU puede variar entre ejecuciones con la misma semilla.

## Desarrollo y comprobación

Usar `.venv/`, crear una rama por cambio y ejecutar `python scripts/check_setup.py`
antes de un PR o merge. No versionar datos, credenciales, entornos ni corridas.
W&B es opcional; `WANDB_MODE=disabled` habilita ejecución local sin cuenta.

- `src/data/`, `src/features/`: carga y construcción de paneles.
- `src/models/`, `src/training/`: modelos y validación cruzada.
- `src/eval/`: splits, métricas y auditorías.
- `scripts/train.py`, `scripts/ensemble_rank.py`: entrenamiento y ensamble en desarrollo.
- `scripts/eval_test.py`: procedimiento de evaluación final.
- `scripts/make_report_figures.py`: figuras a partir de resultados guardados.

## Demo GRU

Esta rama incluye `scripts/demo_app/` y `src/agents/`. Leer
[scripts/demo_app/README.md](scripts/demo_app/README.md) para ejecutar la aplicación.
La demo reproduce únicamente desarrollo desde un bundle precalculado.
La política de acción es determinista; los agentes redactan y organizan mensajes
a partir de hechos verificados. El perfil frente a la flota sana es descriptivo,
no una atribución causal del modelo. Usar `DEMO_LLM_MODE=cache_only` para revisar
la aplicación sin llamadas a la API.
