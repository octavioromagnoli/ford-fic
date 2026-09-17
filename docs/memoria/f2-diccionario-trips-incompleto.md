# `TripSummary` tiene 25 columnas, no las 40 del diccionario oficial

**Fecha:** 2026-09-16 · **Fase:** F2 (previo) · **Reproduce:**
`notebooks/eda-exhaustivo-dev.ipynb` §1 y §9, o desde Python:

```python
from scripts.build_eda_cache import dictionary_comparison, feature_feasibility, raw_columns
raw_columns()["trips"]                 # 25 columnas
dictionary_comparison().query("estado == 'documentada y AUSENTE'")
feature_feasibility().query("estado == 'bloqueada'")
```

El anexo 7.1 del PDF de Ford
(`data/raw/FIC_III___Data_Driven_Powertrain_Intelligence (2).pdf`) declara **40
columnas** para `TripSummary`. El CSV entregado trae **25**. Los anexos 7.2
(`DynamicInformation`) y 7.3 (`VehicleGeneralInformation`) sí cierran: 7 y 7, con
renombres cosméticos (`Vin`→`VehicleCode`, `EventTimestamp`→`eventTimestamp`,
`SalesCountryCd`→`SalesCountry_cd`).

## Qué falta, bajo cualquier nombre

| Familia | Columnas del anexo | n |
|---|---|---|
| GPS | `VehicleGPSLat/LongDataStart/End` | 4 |
| Elevación | `VehicleElevationRangeStart/End` | 2 |
| Presión de neumáticos | `TirePressure{LF,RF,LR,RR}{Start,End}` | 8 |
| Hollín de regeneración manual | `ManualRegenerationSootStart/End` | 2 |
| Autonomía de combustible | `FuelLvlAutonomyStart` | 1 |
| DPF **por ese nombre** | `DieselParticulateFilterStart/End` | 2 |
| | **total ausente** | **19** |

Y aparecen **4 columnas que el anexo no menciona**: `AirFilterStart/End` y
`AirRegenerationStart/End`. No son un misterio: ver
[f2-columnas-airfilter-airregeneration.md](f2-columnas-airfilter-airregeneration.md).
El DPF de la tabla de arriba está en esa fila **solo por el nombre**: el dato
existe.

## Qué implica para F2

Cruzando las 31 features reservadas en `scripts/make_dummy.py::FEATURE_SPECS`
contra las columnas reales (`feature_feasibility()`):

| Veredicto | n |
|---|---|
| se puede construir | 21 |
| se puede construir con sustituto (`AirRegeneration*`) | 4 |
| parcial con sustituto (`feat_manual_regen_per_1000km`) | 1 |
| **requiere redefinir** (`feat_oil_life_drop_per_1000km`) | 1 |
| **BLOQUEADA · no hay dato** | **4** |

Las cuatro bloqueadas:

- `feat_elevation_mean_m` y `feat_elevation_range_m` (**familia C**);
- `feat_tire_pressure_mean` y `feat_tire_pressure_below_thr_frac` (**familia D**).

No hay sustituto: sin GPS tampoco se puede inferir altitud. Es el único ángulo del
plan §4 que muere por falta de dato — y justo el que el plan señalaba como "ligado
directamente al ingreso de oxígeno que menciona el título del desafío".

`feat_manual_regen_per_1000km` queda parcial: el hollín de regeneración manual no
está, pero los niveles `Cleanning Manually Air Filter` y
`Stopped Cleanning Manually Air Filter` del vocabulario compartido marcan el
evento. Son rarísimos —los ven 4 y 31 vehículos de dev respectivamente—, así que
como tasa por 1.000 km va a ser casi siempre cero.

`feat_oil_life_drop_per_1000km` hay que redefinirla, por otro motivo: ver
[f2-calidad-columnas-dev.md](f2-calidad-columnas-dev.md).

## Ojo con el contrato

`configs/data/raw_sources.yaml` **ya declara el esquema real** (F1 lo auditó contra
los archivos): no hay nada que arreglar ahí. Lo que quedaba sin documentar era la
distancia contra el diccionario oficial, que es lo que cambia el alcance de F2 y no
se ve mirando el YAML.

La decisión está en [decisiones.md](decisiones.md).
