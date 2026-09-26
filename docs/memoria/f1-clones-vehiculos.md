# 13 vehículos contados cuatro veces

> **Se sostiene en la entrega v2 (26-09-2026):** los 7 pares que siguen en la cohorte de fallados vienen otra vez con los viajes repetidos bajo los dos códigos, con historia idéntica (Jaccard 1,0); los otros 6 están solo entre los sanos, cuyo archivo no cambió ([f9-entrega-v2.md](f9-entrega-v2.md) §3).

**Fecha:** 2026-09-15 · **Fase:** F1 · **Reproduce:** `python scripts/eda_raw.py`
(sección 1b; deja `experiments/eda/clones.csv`)

`StaticInformation` tiene **1120 filas y 1094 `VehicleCode` únicos**. La diferencia
son 26 códigos duplicados, y están duplicados de dos maneras a la vez.

## a) El mismo código en las dos cohortes

Los 26 códigos aparecen en el archivo `Failed` **y** en el `NotFailed`, con
estáticas idénticas y `IdentificationDate` presente en la fila failed, nula en la
otra. La etiqueta se contradice a sí misma. Sus viajes vienen repetidos fila por
fila: 62.064 viajes idénticos en las dos cohortes (124.128 filas en total, ~5% de
`trips`).

## b) Dos códigos distintos para el mismo auto

Esos 26 códigos son en realidad **13 vehículos**, cada uno bajo dos códigos
consecutivos. Verificado comparando los conjuntos de `(TripDatetimeStart,
OdometerTripEnd)`: **Jaccard 1.000** en los 13 pares.

```
VEH_0240/VEH_0241  VEH_0344/VEH_0345  VEH_0392/VEH_0393  VEH_0444/VEH_0445
VEH_0459/VEH_0460  VEH_0463/VEH_0464  VEH_0488/VEH_0489  VEH_0511/VEH_0512
VEH_0583/VEH_0584  VEH_0590/VEH_0591  VEH_0609/VEH_0610  VEH_0994/VEH_0995
VEH_1010/VEH_1011
```

Los 13 son de la cohorte failed.

(Hay otros 12 grupos de vehículos con estáticas idénticas — mismo `ProductionDay`,
motor, modelo y país — pero con historias de viaje distintas: **esos no son clones**,
comparten configuración y nada más. Por eso el criterio es la huella de viajes, no
las estáticas.)

## Por qué importa

Es la regla 2 evadiéndose sin hacer ruido. Si `VEH_0344` cae en train y `VEH_0345`
en validación, `src/eval/splits.py` no ve ningún `vehicle_id` repetido y da el split
por limpio, mientras el modelo valida contra los mismos viajes con los que entrenó.
No falla ningún chequeo: aparece como un PR-AUC más alto y nadie sabe por qué.

## Qué se hace

Colapsarlos a 13 `vehicle_id` antes de armar el panel, y resolver la etiqueta a
favor de `failed` (`not_failed` es ausencia de registro, no evidencia de que el
vehículo esté sano). Ver [decisiones.md](decisiones.md).

- Lista congelada: `configs/data/vehicle_dedupe.yaml`
- Implementación: `src/data/dedupe.py` (`dedupe_vehicles`, `apply_canonical_ids`)

Resultado verificado sobre los datos reales:

| | antes | después |
|---|---|---|
| filas en static | 1120 | 1081 |
| vehículos | 1094 | **1081** |
| con evento | 378 | **365** |
| sanos | 742 | **716** |
| tasa de eventos | 0,3375 | **0,3377** |
| viajes de los clones | 124.128 | 31.032 |
