# Tres defectos de columna que no se ven en la tabla de nulos

**Fecha:** 2026-09-16 · **Fase:** F2 (previo) · **Alcance de la medición:** dev
(864 vehículos, 1.947.866 viajes, 7.534.459 señales) · **Reproduce:**
`notebooks/eda-exhaustivo-dev.ipynb` §4 y §5.3.

Los tres pasan el chequeo de nulos de
[f2-union-y-holdout-dev-test.md](f2-union-y-holdout-dev-test.md) y aun así rompen
features.

## 1 · El odómetro nulo de `signals` está concentrado, no repartido

[f1-calidad-odometro.md](f1-calidad-odometro.md) reporta 11,15% de `OdometerValue`
nulo y concluye: *"si el nulo correlaciona con el tipo de mensaje, descartarlas
sesga las features de severidad"*. La estructura real es otra: **el nulo
correlaciona con el vehículo.**

Sobre los 1081 vehículos (`data/processed/vehicles.parquet`):

| | valor |
|---|---|
| fracción global de filas sin odómetro | 0,1157 |
| **mediana de la fracción por vehículo** | **0,0000** |
| p75 de la fracción por vehículo | 0,0001 |
| vehículos con >50% de sus señales sin odómetro | **9** |
| vehículos con >20% | 10 |
| vehículos con <1% | 1069 de 1081 |

Esos pocos vehículos son además los que más señales tienen (90.000 a 340.000
filas, contra una mediana de flota de ~7.000), así que arrastran el porcentaje
global. En dev la foto es la misma con otro número: 4,71% global, mediana por
vehículo 0,00%, **858 de 864 por debajo del 1%** y 4 vehículos entre el 66% y el
97%.

**Qué implica:** descartar las filas sin odómetro **no** sesga a la flota —le saca
menos del 1% a casi todos— pero deja a ese puñado de vehículos **sin ninguna
feature de `signals`**. La imputación por el viaje que contiene al `eventTimestamp`
que propone F1 sigue siendo la salida correcta; lo que cambia es que hace falta
para 9 vehículos, no para el 11% del dataset. Y hay que decidir explícitamente qué
se hace con ellos: un vehículo sin features de severidad no es lo mismo que un
vehículo con features en cero.

## 2 · `EngineOilLifePC` no cambia **dentro** de un viaje

```
EngineOilLifePCStart == EngineOilLifePCEnd  →  100,00% de los viajes
delta intra-viaje distinto de cero          →  0 de 249.400 viajes con el dato
```

(medido sobre la muestra determinística de 250.000 viajes de dev del cache; 600
filas tienen el dato nulo y no cuentan).

La columna **no está vacía**: el nivel recorre 0–100 con 101 valores distintos y
baja a lo largo de la vida del vehículo. Lo que pasa es que el índice se actualiza
**entre** viajes, no dentro.

**Qué implica:** `feat_oil_life_drop_per_1000km` del plan §4, especificada como
"caída de `EngineOilLifePC` por cada 1.000 km", da **exactamente cero para todos
los vehículos** si se calcula como suma de deltas por viaje. Hay que redefinirla
como **pendiente del nivel contra odómetro dentro de la ventana**. Es la misma
primitiva que `feat_dpf_end_slope_per_1000km`, así que no cuesta nada — pero si
nadie lo mira, la feature entra al panel constante en cero y el modelo la ignora
sin que salte ningún error.

## 3 · Valores fuera de rango físico

Sobre la muestra de 250.000 viajes de dev:

| Columna | mín | p01 | p99 | máx | fuera de rango |
|---|---|---|---|---|---|
| `KilometerPerHour` | 0,002 | 3,3 | 91,3 | **135.600** | 0,03% (> 200 km/h) |
| `AirTemperatureAvg` | **−73,0** | 4,2 | 38,0 | **79,1** | 0,03% |
| `trip_duration_min` | 0 | 0 | 159 | **29.125** (20 días) | 0,01% |
| `trip_km` | **−230** | 0 | 134 | 777 | 0,01% |
| `CoolantTemperatureStart` | −60,0 | 2,0 | 97,0 | 109,0 | 0,00% |
| `FuelLvlStartPc` | **−5,2** | 0,5 | 102,8 | **103,5** | **10,5%** |

Las cinco primeras son colas finitas: menos del 0,05% de los viajes cada una, pero
suficientes para arrastrar una media sin recorte. `f1-calidad-odometro.md` ya
sugiere recalcular la velocidad desde odómetro y tiempo en vez de imputarla; el
135.600 km/h es el número que lo justifica.

**`FuelLvl*Pc` es un caso distinto y hay que no equivocarse**: supera 100 en el
**10,5%** de los viajes (hasta 103,5) y baja a −5,2. Eso no es una cola, es que la
escala no es un porcentaje literal. Recortarla a [0, 100] "porque es un porcentaje"
tira información real de una de cada diez filas. Conviene tratarla como índice
relativo.

## Qué se lleva F2

| Columna | Acción |
|---|---|
| `signals.OdometerValue` | imputar o excluir **por vehículo**, no por fila; 9 vehículos afectados |
| `EngineOilLifePCStart/End` | redefinir la feature como pendiente del nivel, no suma de deltas |
| `KilometerPerHour` | recalcular desde odómetro/tiempo, o recortar a [0, 200] |
| `AirTemperatureAvg`, `CoolantTemperature*`, `trip_duration_min` | recortar por percentil antes de agregar |
| `FuelLvl*Pc` | **no** recortar a [0, 100]: tratarla como índice relativo |
| `trip_km` | forzar monotonía del odómetro por vehículo antes de cortar ventanas (24 vehículos, 165 viajes) |
