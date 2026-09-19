# Las fechas de `trips` y `signals` mezclan dos formatos: sin `ISO8601`, pandas las anula

**Fecha:** 2026-09-16 (encontrado armando la deuda térmica), medido sobre el panel v1 el
2026-09-19 · **Fase:** F2 · **Arreglo:** `date_format: ISO8601` en `trips` y `signals` de
`configs/data/raw_sources.yaml`. · **Reproduce:**
`python scripts/make_test_split.py` (deja `raw_quality.csv` con los nulos por columna) y
`python scripts/build_dataset.py --config configs/data/panel_v1.yaml` con y sin la clave.

## El hecho

Los timestamps vienen en dos formatos en el mismo archivo: `2026-08-13 23:39:18+00:00` y
`2026-09-11 19:23:57.731000+00:00`. `pd.to_datetime(errors="coerce")` sin formato infiere
el de la **primera fila de cada chunk** y convierte las del otro formato en `NaT`, sin
avisar. El crudo tiene **0 nulos** en las tres columnas de fecha.

| columna | nulos sin `ISO8601` (tablas completas) | con `ISO8601` |
|---|---|---|
| `trips.TripDatetimeStart` | 4.464 | **0** |
| `trips.TripDatetimeEnd` | 3.298 | **0** |
| `signals.eventTimestamp` | 18.124 | **0** |

Los "nulos nuevos" que registró [f2-union-y-holdout-dev-test.md](f2-union-y-holdout-dev-test.md)
y el "0,3% de fechas nulas" que menciona `src/features/trips.py` eran esto.

## Qué le hacía al panel v1

Construido con y sin el arreglo, el panel tiene **las mismas 2.507 filas, las mismas
etiquetas y el mismo emparejado** (el origen del calendario, 20107, y el evento mediano,
6.477 km, no se mueven). Cambian 13 features en hasta 102 filas (4%):

- las temporales de `trips`: duración, velocidad, reposo entre viajes, `chained_trip_frac`,
  `trips_below_30kmh_frac` (una duración `NaT` deja la fila en NaN);
- **las de `signals`, aunque se ubican por odómetro** (`msgs_per_1000km`,
  `msg_abnormal_frac`, `msg_cleaning_auto_*`): el dedupe de `read_table_for_vehicles` es
  por fila completa, y dos mensajes que solo difieren en el timestamp quedan iguales
  cuando los dos son `NaT`. Sin el arreglo se descartan **2.420 mensajes distintos** como
  repetidos en el universo (29.443 "otros" contra 27.088 repetidos reales).

No cambia ninguna conclusión del EDA; sí cambia el panel publicado como `panel-v1` en
esas celdas. Hay que regenerarlo y volver a publicarlo con el arreglo.
