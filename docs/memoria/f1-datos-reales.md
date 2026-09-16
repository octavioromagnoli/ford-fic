# Qué hay realmente en `data/raw/`

**Fecha:** 2026-09-15 · **Fase:** F1 · **Reproduce:** `python scripts/eda_raw.py`

Seis CSV, 1,2 GB, **tres tablas lógicas × dos cohortes de muestreo**
(`*_Failed*` = vehículos con evento registrado, `*_NotFailed*` = sin evento).
El loader las declara como tres tablas con sus `parts` y agrega la columna
`cohort` (`configs/data/raw_sources.yaml`).

| Tabla | Archivo | Filas | Vehículos |
|---|---|---|---|
| `vehicles` | `StaticInformation_{Failed,NotFailed}Vins.csv` | 1120 | 1094 únicos |
| `trips` | `TripSummary_{Failed,NotFailed}_SelectionVins.csv` | 2.531.435 | 1094 |
| `signals` | `DynamicInformation_{Failed,NotFailed}_SelectionVins.csv` | 10.617.817 | 1094 |

## El join cierra

`VehicleCode` es la clave en las seis tablas: 1094 vehículos aparecen en las tres,
y no hay ningún código en `trips`/`signals` que falte en `vehicles` ni al revés.
El riesgo #1 del plan §10 (que las tablas no joineen) está descartado.

## Diferencias contra el esquema provisorio

- **No hay `Vin`.** El único identificador es `VehicleCode`.
- **No hay columna de DPF.** El postratamiento viene como `Message` (10 niveles de
  filtro de aire) y `Acumulation` (0–95). Ver [f1-senal-postratamiento.md](f1-senal-postratamiento.md).
- **El evento no está marcado fila a fila.** Se deriva del vehículo:
  `IdentificationDate` está presente en el 100% de la cohorte failed y nula en el
  100% de la not_failed. *Es la etiqueta*, no una covariable — nunca entra como `static_*`.
- **`Regenerations` no es un conteo**: es el string literal `"Regeneration"` que
  marca la fila del evento (25.701 en la cohorte failed). Declararla `float64` la
  convierte en NaN en silencio.
- **`trips` trae start/end de casi todo** (`OdometerTripStart`/`End`,
  `FuelLvlStartPc`/`EndPc`, `AirFilterStart`/`End`…), no solo el agregado.
- Los seis archivos vienen **con BOM**: sin `encoding="utf-8-sig"` la primera
  columna se llama `﻿VehicleCode`.

## Cobertura por vehículo (mediana)

| | fallados | sanos |
|---|---|---|
| viajes | 2172 | 1811 |
| señales | 8146 | 6842 |
| odómetro final [km] | 21.328 | 16.914 |
| span [días] | 470 | 411 |

Sobra histórico para ventanas de varios miles de km.

## Ojo

Esas 1120 filas son 1081 vehículos reales, no 1094:
ver [f1-clones-vehiculos.md](f1-clones-vehiculos.md).
