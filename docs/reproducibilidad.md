# Reproducción de la entrega

Ejecutar desde la raíz del repositorio, con `.venv/` activo, las dependencias
base y `requirements-dl.txt` instaladas y `WANDB_MODE=disabled`.

## Insumos que se entregan por separado

Los datos y las corridas no están incluidos en Git. Quien reproduce necesita
acceso autorizado a los CSV de la entrega v2 de Ford y a los artefactos congelados
del equipo. Se pueden copiar directamente, sin usar W&B:

| Ubicación | Insumo |
|---|---|
| `data/raw/` | Los seis CSV con los nombres declarados en `configs/data/raw_sources.yaml` |
| `data/processed/test_split.json` | Holdout v2 congelado, con pertenencias dev/test y antecedentes del split anterior |
| `data/processed/splits_r3.json` | Folds congelados, cinco folds × tres repeticiones |
| `data/processed/panel_seq_trips_v2_estaticas.parquet` | Panel secuencial del finalista, si se recibe ya construido |
| `data/processed/panel_seq_trips_v2_estaticas_meta.json` | Metadatos de la secuencia, necesarios para construir la red |

`FORD_DATA_DIR` cambia la raíz `data/`, no solamente la carpeta de CSV.
Los archivos de splits deben ser los usados en la entrega: no volver a sortearlos
para reproducir sus resultados. `build_if_missing: false` y `strict: true`
verifican que el panel corresponde al reparto esperado.

No hay una descarga pública automatizada de estos insumos en este checkout.
El paquete de datos y artefactos debe acompañar la entrega por el canal autorizado
por Ford; un clon del repositorio por sí solo permite correr el ejemplo sintético.

## Construir el panel desde los CSV

Con el holdout congelado ya en `data/processed/test_split.json`:

```bash
python scripts/build_dataset.py --config configs/data/panel_v2_estaticas.yaml
python scripts/build_seq_panel.py --config configs/data/panel_seq_trips_v2_estaticas.yaml
```

El primer comando crea el panel agregado con país, motor y serie. El segundo
conserva sus filas y sustituye los agregados por la secuencia de 20 × 15 canales.
Usar el archivo compartido `splits_r3.json` para entrenar; el split de una sola
repetición que produce el primer comando no lo reemplaza.

Los YAML aplican el universo común de producción, el primer evento por vehículo,
el dedupe, el corte común de extracción y la referencia de evento desplazada 21 días.
La ventana es de 1.000 km, el gap de 500 km y el horizonte de 3.000 km, con cortes
cada 500 km. Imputación y escalado se ajustan dentro del entrenamiento de cada fold.

## Entrenamiento y ensamble en desarrollo

```bash
python scripts/train.py --config configs/exp_v2all_gru_trips_estaticas_r3.yaml
python scripts/train.py --config configs/exp_v2all_gru_trips_estaticas_s1_r3.yaml
python scripts/train.py --config configs/exp_v2all_gru_trips_estaticas_s2_r3.yaml
python scripts/ensemble_rank.py --config configs/exp_v2all_seeds3_gru_trips_estaticas.yaml
```

El ensamble promedia rangos de las predicciones OOF alineadas por fila y fold.
Se guarda en `experiments/v2all-seeds3-gru-trips-estaticas/`.
La métrica principal del informe es detección por vehículo a presupuestos fijos
de falsas alarmas, con un nulo que conserva el largo de los historiales.
PR-AUC por fila es diagnóstico y no basta para demostrar anticipación.

## Resultados cerrados y figuras del informe

El test ya fue utilizado. Para reconstruir el documento, leer sus salidas guardadas;
no ejecutar de nuevo una búsqueda ni una selección sobre test. `scripts/eval_test.py`
y los YAML `configs/eval_test_*.yaml` conservan el procedimiento de evaluación.
Los nombres históricos de archivos y claves se mantienen para no romper compatibilidad.

`configs/report_figures.yaml` enumera las fuentes exactas. Se necesitan:

- `experiments/test-modelos-v2/{curve,operating,paired}.csv`.
- `experiments/test-f10-s42/report.json`.
- `experiments/explain-gru-final/units.csv`.

La última salida corresponde a la explicabilidad de la GRU y se debe suministrar
como artefacto. La aplicación de demostración se ejecuta según la
[guía de la demo](../scripts/demo_app/README.md). Las tablas auxiliares están
incorporadas en el YAML.

```bash
python scripts/make_report_figures.py --config configs/report_figures.yaml
```

Las figuras y tablas de la entrega ya están versionadas: no hace falta disponer de
las corridas para compilar el [informe](informe/README.md). La GRU no es determinista
bit a bit; volver a entrenar no garantiza repetir exactamente sus métricas guardadas.

## Empaquetado

Antes de cerrar una versión, ejecutar `python scripts/check_setup.py` desde un
checkout limpio y verificar los artefactos del paquete de datos. Una vez
commiteada y etiquetada la versión aprobada, `git archive` permite generar un ZIP
con los archivos versionados, sin `.env`, `.venv/`, datos ni corridas locales.
