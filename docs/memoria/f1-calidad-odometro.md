# Calidad del eje de odómetro

**Fecha:** 2026-09-15 · **Revisado:** 2026-09-19 (bloque "Los retrocesos son del
orden, no del odómetro") · **Fase:** F1 · **Reproduce:** `python scripts/eda_raw.py`
(sección 3; deja `experiments/eda/per_vehicle.csv`) y
`python scripts/eda_gaps.py` (bloque 8; deja `experiments/eda/dev/gaps/data_checks.json`)

El odómetro es el eje por defecto del contrato (regla 4), así que sus defectos son
los defectos del panel.

## Lo que está bien

- Todos los vehículos arrancan en ~0 km (`odo_min` mediana 0, máximo 7).
- Hay recorrido de sobra: mediana 21.328 km en fallados y 16.914 en sanos, con
  spans de ~470 y ~411 días.
- `trips` tiene el odómetro prácticamente **completo**: 8 filas nulas en
  `OdometerTripStart/End` (4 vehículos, 3 por millón; medido en F2, ver
  [f2-union-y-holdout-dev-test.md](f2-union-y-holdout-dev-test.md)).

## Lo que hay que manejar

| Problema | Magnitud |
|---|---|
| `OdometerValue` nulo en `signals` | 1.183.603 filas (**11,15%**) |
| Viajes con km negativo (retroceso) | 165 viajes en 24 vehículos (137 en 8 vehículos sobre dev) |
| `odo_max(signals) − odo_max(trips)` | mediana −18 km, p25 −77 km, mínimo −7821 km |
| `KilometerPerHour` nulo en `trips` | ~31%, y es exactamente `trip_km == 0` (ver abajo) |

## Los retrocesos son del orden, no del odómetro (medido el 19-09 sobre dev)

La magnitud del "odómetro que va para atrás" **depende de con qué orden se recorren
los viajes**, y la diferencia es de dos órdenes de magnitud:

| orden | retrocesos > 1 km | vehículos | > 500 km | filas que corrige `cummax` | km corregidos (p50) |
|---|---|---|---|---|---|
| por `TripDatetimeStart` | 927 | **231** | 299 | 1.717 | **2.596** |
| por **`TripNumber`** | 245 | **8** | 1 | 659 | 265 |
| por odómetro (control) | 168 | 7 | 1 | 1.389 | 280 |

La causa es que el **0,67% de los viajes comparte `TripDatetimeStart`** con otro del
mismo vehículo (183 vehículos de 290 tienen algún empate): ordenar por fecha deja esos
empates barajados y el "retroceso" que aparece es el desempate, no el odómetro.

Con el orden por `TripNumber` los dos ejes quedan limpios a la vez:

- **0 saltos de tiempo hacia atrás** (de 526.420 pares consecutivos) y **0 viajes
  solapados**: `TripNumber` *es* el orden temporal.
- Velocidad implícita entre viajes consecutivos: p99,9 = 102 km/h, **máximo 134 km/h,
  ninguna por encima de 200**. El odómetro y el reloj son coherentes entre sí.
- Spearman odómetro↔fecha por vehículo: mediana 1,000; solo 5 vehículos bajo 0,95.

**Corrupción real: 8 vehículos de 290 (2,8%)** — `VEH_0070`, `VEH_0328`, `VEH_0441`,
`VEH_0494`, `VEH_0545`, `VEH_0548`, `VEH_0572`, `VEH_0792`. Uno solo es grave:
`VEH_0572` retrocede en 223 de sus 2.610 viajes y tiene 128 viajes de km negativo
(8,5%) — telemetría rota, no un odómetro adulterado. Los otros siete son incidentes
puntuales (1 a 10 filas) que el máximo acumulado absorbe. **No hay un solo vehículo con
retrocesos sistemáticos**: nadie "bajó el kilometraje".

## Qué implica para F2

- **El 11% de señales sin odómetro no se puede tirar sin pensar**: si el nulo
  correlaciona con el tipo de mensaje, descartarlas sesga las features de
  severidad. La salida razonable es imputar el odómetro por el viaje que contiene
  al `eventTimestamp` (ahí `trips` está completo), no descartar la fila.
- **Los retrocesos son 8 vehículos, no un patrón**: conviene forzar monotonía con un
  máximo acumulado por vehículo antes de cortar ventanas —**recorriendo los viajes por
  `TripNumber`, no por fecha**— y dejar registrado a cuántos les tocó. Con el orden por
  fecha la corrección "arregla" 231 vehículos que no tienen nada roto, y les mueve el
  odómetro una mediana de 2.596 km.
- **El desfase signals/trips es chico salvo un caso** (−7821 km). Para armar
  ventanas sobre un eje común, `trips` manda y `signals` se alinea contra él.
- **`KilometerPerHour` con 31% de nulos** es recuperable: `trips` trae
  `OdometerTripStart/End` y `TripDatetimeStart/End`, así que la velocidad media se
  puede recalcular en vez de imputarla. *Medido el 2026-09-18:* el nulo es
  **exactamente** `trip_km == 0` (100% de esas filas, 0,05% del resto), y cuando existe
  es `trip_km / duración` con Pearson 0,99999. No hay nada que imputar: las filas de 0
  km son idle (motor encendido sin moverse, el 35% de `trips`) y se cuentan aparte
  ([f2-eda-revision-y-features.md](f2-eda-revision-y-features.md) §2.3).
