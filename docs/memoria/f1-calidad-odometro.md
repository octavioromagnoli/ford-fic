# Calidad del eje de odómetro

**Fecha:** 2026-09-15 · **Fase:** F1 · **Reproduce:** `python scripts/eda_raw.py`
(sección 3; deja `experiments/eda/per_vehicle.csv`)

El odómetro es el eje por defecto del contrato (regla 4), así que sus defectos son
los defectos del panel.

## Lo que está bien

- Todos los vehículos arrancan en ~0 km (`odo_min` mediana 0, máximo 7).
- Hay recorrido de sobra: mediana 21.328 km en fallados y 16.914 en sanos, con
  spans de ~470 y ~411 días.
- `trips` tiene el odómetro **completo**: ningún nulo en `OdometerTripStart/End`.

## Lo que hay que manejar

| Problema | Magnitud |
|---|---|
| `OdometerValue` nulo en `signals` | 1.183.603 filas (**11,15%**) |
| Viajes con km negativo (retroceso) | 165 viajes en 24 vehículos |
| `odo_max(signals) − odo_max(trips)` | mediana −18 km, p25 −77 km, mínimo −7821 km |
| `KilometerPerHour` nulo en `trips` | ~31% |

## Qué implica para F2

- **El 11% de señales sin odómetro no se puede tirar sin pensar**: si el nulo
  correlaciona con el tipo de mensaje, descartarlas sesga las features de
  severidad. La salida razonable es imputar el odómetro por el viaje que contiene
  al `eventTimestamp` (ahí `trips` está completo), no descartar la fila.
- **Los retrocesos son 24 vehículos, no un patrón**: conviene forzar monotonía con
  un máximo acumulado por vehículo antes de cortar ventanas, y dejar registrado a
  cuántos les tocó.
- **El desfase signals/trips es chico salvo un caso** (−7821 km). Para armar
  ventanas sobre un eje común, `trips` manda y `signals` se alinea contra él.
- **`KilometerPerHour` con 31% de nulos** es recuperable: `trips` trae
  `OdometerTripStart/End` y `TripDatetimeStart/End`, así que la velocidad media se
  puede recalcular en vez de imputarla.
