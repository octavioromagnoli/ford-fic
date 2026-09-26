# La entrega v2 de Ford: qué cambió, qué trampas trae y qué marca la fecha nueva

**Fecha:** 2026-09-26 · **Fase:** F9 (datos) · **Alcance:** los seis archivos de las dos
entregas, completos (auditoría de archivos y etiquetas, como la de F1); lo que mira
telemetría alrededor del evento excluye a los autos de test. **Reproduce:**

```bash
python scripts/compare_deliveries.py --config configs/data/compare_deliveries.yaml  # §1-§6 -> experiments/entregas/
python scripts/make_test_split.py --config configs/data/test_split.yaml --force     # auditoría del join sobre v2
```

La entrega 1 (15-09) quedó en `data/old_raw/` y se lee con `configs/data/raw_sources_v1.yaml`.
`configs/data/raw_sources.yaml` ahora lee v2, con cada corrección declarada en su parte.

## 1 · Qué archivos cambiaron

| tabla | cohorte | v2 | ¿igual a v1? | filas v1 → v2 |
|---|---|---|---|---|
| estática | fallados | `StaticInformation_FailedVins_v2_20260924_144239.csv` | no | 378 → 298 |
| estática | sanos | `StaticInformation_NotFailedVins_v2_20260924_144209.csv` | no | 742 → 768 |
| viajes | fallados | `TripSummary_Failed_SelectionVins_vehiclecode_v2 (2).csv` | no | 947.477 → 934.463 |
| viajes | sanos | `TripSummary_NotFailed_SelectionVins (1).csv` | **sí, byte a byte** | 1.583.958 |
| señales | fallados | `DynamicInformation_Failed_SelectionVins_v2.csv` | no | 4.202.484 → 3.595.216 |
| señales | sanos | `DynamicInformation_NotFailed_SelectionVins (1).csv` | **sí, byte a byte** | 6.415.333 |

La telemetría de los sanos no se re-extrajo: es la misma de la entrega 1. Todo lo nuevo
está en la estática y en los dos archivos de fallados.

## 2 · Cinco trampas de formato, y cómo las corrige el loader

Cada una rompe algo en silencio si no se corrige. Las cinco están declaradas en
`raw_sources.yaml` (`rename`, `offsets`, `truncate_after`) y probadas en
`scripts/check_setup.py`.

1. **La columna del evento se renombró solo en la estática de fallados:**
   `IdentificationDaysSinceProduction`. En sanos sigue siendo `IdentificationDate`.
2. **El `ProductionDay` de la estática de fallados está contado desde otro origen, +538 días.**
   - Sin corregirlo, `primer viaje − ProductionDay` da el día 19569 para los fallados y el
     20107 para los sanos. Corrigiéndolo, da 20107 para los 1.019 códigos con viajes (IQR 0 d).
   - En los 15 códigos que aparecen en los dos archivos, la diferencia es exactamente 538.
   - `IdentificationDate` es relativa a la producción y no cambia.
3. **La estática de fallados trae una fila por evento, no por auto.** Son 298 filas, 277
   códigos y 270 autos: 19 autos tienen 2 eventos y uno tiene 4, con las filas idénticas salvo
   la fecha. El dedupe se queda con el primero y cuenta los demás en `n_events_recorded`.
   - Entre dos eventos del mismo auto pasan 54 días de mediana (rango 1–192).
4. **La estática de sanos repite 26 filas idénticas**, dos por cada código de los 13 pares de
   clones. El dedupe las colapsa.
5. **Las dos cohortes se extrajeron en fechas distintas.**
   - Los viajes de fallados llegan al 24-09-2026 y los de sanos terminan el 14-09-2026 06:02.
   - Hay 16.911 viajes de fallados posteriores al último de un sano. Un modelo que los viera
     aprendería "si hay datos después del 14-09, es un fallado".
   - `truncate_after` corta todo en el fin de extracción de los sanos, por tabla.

Hay dos cosas menores:
- Los países vienen con su código ISO: CNTRY_1..5 = **ARG, BRA, CHL, COL, PER**, con el mapa
  exacto en los 809 códigos comunes. PRY es nuevo, con 1 auto.
- `VEH_0437` está en los dos archivos (sano en v1, fallado en v2). Sus 2.963 viajes del archivo
  de sanos están idénticos en el de fallados: son los "duplicados que no son clones" que reporta
  el scan, y el dedupe los colapsa.

## 3 · Los clones se sostienen

Como en la entrega 1, los viajes de los 7 pares de clones que siguen en fallados vienen
repetidos bajo los dos códigos (17.557 filas). Los 7 pares tienen historia de viajes idéntica
(Jaccard 1,0 sobre `(TripDatetimeStart, OdometerTripEnd)`, el criterio de F1). Los otros 6
pares solo aparecen en el archivo de sanos, que no cambió. `vehicle_dedupe.yaml` no se toca.
Después de canonizar y deduplicar, los viajes enriquecidos son 2.435.711.

## 4 · Qué pasó con cada código

| código v1 → v2 | n | qué es |
|---|---|---|
| sano → sano | 715 | la misma telemetría y la misma estática (+ ciudad, país ISO) |
| fallado → **ausente** | **285** | 278 tenían la fecha por defecto (`IdentificationDate == daysUntilSale`) y 7 fecha real. No están ni en fallados ni en sanos |
| **nuevo** → fallado | **195** | VEH_2000–2194, con telemetría desde agosto de 2024 |
| fallado → fallado | 67 + 14 | 67 códigos y 7 pares de clones |
| fallado y sano → sano | 12 | 6 pares de clones. 5 tenían la fecha por defecto y VEH_0444/0445 una real |
| sano → fallado | 1 | VEH_0437, evento del 21-04-2026 |

- **Ford no corrigió las fechas por defecto: sacó a esos autos.** Solo VEH_0121 (ARG) pasó de
  la fecha por defecto (16 d) a una real (453 d). En v2 hay 0 fallados con
  `IdentificationDate == daysUntilSale` y 0 con el evento antes de la venta.
- **Nadie sabe qué son los 278 ausentes.** Hay dos lecturas y los datos no las separan:
  - la consulta vieja tenía un error (la memoria de la mentora habla de un bug de SQL) y no
    eran fallas: serían sanos que no volvieron a la muestra;
  - eran fallas reales que no se pudieron fechar, y se descartaron.

  Importa porque se concentran por mercado (§ del universo): en la ventana de producción del
  estudio faltan 88 de ARG, 70 de BRA, 24 de CHL, 8 de PER y 6 de COL. **Es la pregunta #1 para
  Ford.**
- **VEH_0437** (sano en v1, fallado en v2 con fecha anterior a la extracción vieja) muestra
  que la consulta nueva se corrió también sobre la población de sanos.

## 5 · La telemetría de los fallados que siguen no cambió; la fecha sí

- **Viajes:** de los 74 fallados comunes con viajes, los 182.472 viajes de v1 están todos en
  v2. v2 solo agrega 3.988, posteriores al 13-09 (la extracción nueva llega al 24-09). Las
  señales, igual.
- **Fecha del evento:** en los 81 códigos comunes, la fecha v2 cae **siempre después** que
  la v1. La diferencia tiene mediana de 14 días, IQR 9–33 y máximo 437 (VEH_0121, la fecha
  corregida).
- **Los fallados nuevos son una cohorte más vieja:**
  - se produjeron entre el 05-08-2024 y el 28-07-2025 (mediana 28-01-2025);
  - sus viajes empiezan en agosto de 2024;
  - su odómetro final tiene mediana de 35.000 km.

  Los sanos se produjeron del 20-01-2025 al 23-12-2025. Consecuencia en
  [f9-universo-v2.md](f9-universo-v2.md).
- **Nulos:** `OdometerValue` de las señales es 0,5% nulo en fallados v2 (13% en fallados v1).
  Los nulos estaban concentrados en los 285 autos que ya no están.

## 6 · Qué marca la fecha v2: ~2 semanas después de la intervención

Firmas de servicio alrededor de las dos fechas, sin autos de test (203 fallados; 45 con las
dos fechas). Por auto y por día:

| días respecto de la fecha | −21…−14 | −14…−7 | −7…−3 | −3…0 | 0…+3 | +3…+7 |
|---|---|---|---|---|---|---|
| cambios de aceite, fecha **v1** | 0,003 | 0,003 | 0,000 | **0,015** | **0,022** | 0,006 |
| cambios de aceite, fecha **v2** | 0,006 | **0,016** | **0,017** | **0,015** | 0,000 | 0,000 |
| días con Overloaded/Over Limit, fecha **v1** | 0,079 | 0,085 | **0,116** | 0,061 | 0,049 | 0,050 |
| días con Overloaded/Over Limit, fecha **v2** | 0,082 | **0,019** | 0,037 | 0,029 | 0,038 | 0,037 |

- **La fecha v1 es la intervención.** El cambio de aceite se concentra en ±3 d, unas 5× la
  base, y la sobrecarga del filtro llega a su pico justo antes y cae justo después.
- **La fecha v2 es ~2 semanas posterior.** El aceite ya se cambió entre −14 y 0 d, no hay
  ninguno en la semana siguiente, y la sobrecarga desaparece desde −14 d.
- **Entre las dos fechas el auto circula normal:** 66% de los días con viajes, contra 68% en
  el mismo largo antes de v1. No es una estadía en taller: la fecha v2 registra algo
  administrativo posterior (cierre, reclamo).
- El ralentí largo (≥ 15 min a 0 km, regeneración forzada con el auto parado) dice lo mismo
  sobre el dev ([f9-eda-v2.md](f9-eda-v2.md) §H): 2,2× la base en 0…+3 d de v1, y 1,5–1,8× en
  las dos semanas previas a v2.

**Consecuencia para la etiqueta:** con la fecha v2 tal cual y G = 500 km (~9 días a 57 km/día),
la última ventana antes del gap contiene el taller y los síntomas. Es detección reactiva, que
es lo que la regla 1 prohíbe. `configs/data/panel_v2.yaml` usa como referencia **la fecha
registrada − 21 d** (`label.reference_offset_days`), entre la mediana (14 d) y el p75 (27–33 d)
del corrimiento. No hay forma de fechar la intervención auto por auto para los 195 nuevos (lo
intentó F8): el corrimiento es poblacional.

## Lo que queda para preguntarle a Ford

1. **Qué son los 278 fallados con fecha por defecto que sacaron** y por qué no están en sanos.
2. **Qué registra `IdentificationDaysSinceProduction`** respecto del ingreso al taller: la
   fecha v1 coincidía con el cambio de aceite, la v2 cae dos semanas después.
3. **Con qué criterio se armó la lista de fallados.** Ninguno se produjo después de julio de
   2025, y la exposición predice ~50.
4. Por qué los 195 nuevos son casi todos de producción 2024 y de BRA/CHL, y por qué los ENG_3
   fallados son todos nuevos.
