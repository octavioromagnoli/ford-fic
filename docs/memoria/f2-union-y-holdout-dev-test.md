# La unión cierra 1:1 y el holdout dev/test quedó congelado

**Fecha:** 2026-09-16 · **Fase:** F2 (previo) · **Reproduce:**
`python scripts/make_test_split.py --config configs/data/test_split.yaml`
(deja `data/processed/test_split.json`, `vehicles.parquet` y `raw_quality.csv`).
El mismo recorrido, celda por celda, en
[`notebooks/f2-union-auditoria-split.ipynb`](../../notebooks/f2-union-auditoria-split.ipynb).

## El join cierra, y cierra exacto

Con los códigos ya canonizados (dedupe de los 13 clones aplicado **antes** de
cualquier conteo), los **1081 vehículos aparecen en las tres tablas** y no sobra
ninguno de ningún lado:

| | n |
|---|---|
| vehículos en `vehicles` / `trips` / `signals` | 1081 / 1081 / 1081 |
| en las tres tablas | **1081** |
| en `vehicles` y no en `trips` (y al revés) | 0 |
| en `vehicles` y no en `signals` (y al revés) | 0 |

## `trips` y `signals` no se mergean entre sí

Comparten `VehicleCode`, pero no una clave fila a fila: `vehicles` es una fila por
vehículo, `trips` una por viaje y `signals` una por mensaje. Un merge por
`VehicleCode` entre las dos grandes es muchos-a-muchos y devuelve el producto de
los históricos:

| | |
|---|---|
| vehículo mediano | 1971 viajes × 7195 señales ≈ **1,4×10⁷ filas** (un solo vehículo) |
| peor vehículo | 8,0×10⁸ filas |
| los 1081 juntos | **3,2×10¹⁰ filas** |

No es que tarde: no entra. Alinear señales contra viajes se hace por
odómetro/timestamp y es trabajo de F2 (`trips` manda, ver
[f1-calidad-odometro.md](f1-calidad-odometro.md)).

Las dos uniones que el dato sí admite están implementadas en `src/data/join.py`:

**(a) Enriquecer cada tabla grande por separado** (`iter_enriched_table`): left
join 1-a-muchos de las estáticas sobre `trips` y sobre `signals`, cada una por su
lado, con ids ya canonizados. Salen dos tablas separadas, con sus filas y
columnas nuevas pegadas:

| Tabla | filas crudas | esperadas | enriquecidas | huérfanas | columnas |
|---|---|---|---|---|---|
| `trips` | 2.531.435 | 2.438.339 | **2.438.339** | **0** | 26 → 33 |
| `signals` | 10.617.817 | 10.232.950 | **10.232.950** | **0** | 8 → 15 |

`esperadas = crudas − dup_clones`: el merge no cambia el número de filas, lo único
que lo cambia es el colapso de los clones, que es el objetivo. Cero huérfanas es
la validación fila a fila de que `VehicleCode` cierra —la de arriba es por id—.

**(b) Bajar a nivel vehículo** (`build_vehicle_table`): una fila por `vehicle_id`,
que es lo único que hace falta para congelar el holdout.

## Duplicados: hay una capa más que los clones

Contados por hash de la fila completa, ignorando la columna de cohorte:

| Tabla | filas | mismo código | clones | **otros** |
|---|---|---|---|---|
| `vehicles` | 1.120 | 0 | 26 | 0 |
| `trips` | 2.531.435 | 62.064 | 93.096 | **0** |
| `signals` | 10.617.817 | 378.723 | 384.867 | **120.385** |

Los 93.096 de `trips` son exactamente los clones que documentó F1 (124.128 filas
→ 31.032): ahí el dedupe explica el 100% de los duplicados.

En `signals` no. Quedan **120.385 filas exactamente repetidas (1,13%) que no son
clones**: son filas idénticas dentro de un mismo archivo. Afectan a **1061 de los
1068 vehículos no clonados** (mediana 0,99% de sus filas, máximo **26%**), así
que no es un vehículo roto, es cómo se publicó la tabla. Hay además 132.856 filas
(1,31%) con `(vehículo, timestamp)` repetido: la diferencia contra las exactas son
mensajes distintos en el mismo segundo, que sí son legítimos.

**Qué implica para F2:** toda feature de `signals` normalizada por distancia
—tasa de `Full`/`Overloaded` por 1000 km, regeneraciones por 1000 km— viene
inflada ~1% si no se deduplica, y para el vehículo del máximo, un 26%. El sesgo
es parejo entre cohortes, así que no inventa señal, pero conviene tirar las filas
exactas repetidas antes de agregar la ventana y dejarlo escrito.

## Nulos por columna (sobre las tres tablas completas)

Solo las columnas con algún nulo:

| Tabla | Columna | nulos | frac |
|---|---|---|---|
| `signals` | `Regenerations` / `DistanceBetweenRegenerations` | 10.561.841 | 0,9947 |
| `signals` | `OdometerValue` | 1.183.603 | **0,1115** |
| `signals` | `Message` | 4.227 | 0,0004 |
| `trips` | `KilometerPerHour` | 821.929 | 0,3247 |
| `trips` | `EngineOilLifePC*` / `AirFilter*` / `AirRegeneration*` | ~6.750 | 0,0027 |
| `trips` | `OdometerTripStart` / `End`, `FuelLvl*` | 8 | 0,000003 |
| `vehicles` | `IdentificationDate` | 742 | 0,6625 (es la etiqueta) |
| `vehicles` | `daysUntilSale` | 18 | 0,0161 |

### Desvíos contra lo ya documentado

- **`OdometerValue` es 11,15% nulo, no ~5%.** El 11,15% coincide con
  [f1-calidad-odometro.md](f1-calidad-odometro.md); el que estaba mal era el
  comentario de `configs/data/raw_sources.yaml`, ya corregido.
- **`trips` no tiene el odómetro perfecto: hay 8 filas nulas** (4 vehículos, todos
  de la cohorte `not_failed`). En esas filas `OdometerTripStart/End`,
  `KilometerPerHour` y `FuelLvl*` están vacíos y el resto poblado. Son 3 por millón
  —no cambia la decisión de usar `trips` como eje— pero el "ningún nulo" de F1 era
  literal y ya no lo es.
- **Las fechas no tienen nulos.** Esta tabla registraba 4.464 / 3.298 nulos en
  `TripDatetimeStart` / `TripDatetimeEnd` y 18.124 en `eventTimestamp`: eran un error
  de parseo (dos formatos mezclados), no del dato. Con `date_format: ISO8601` son 0.
  Ver [f2-fechas-formato-mixto.md](f2-fechas-formato-mixto.md).
- `KilometerPerHour` da 32,47% contra el "~31%" documentado: misma foto.
- `daysUntilSale`, que **sí es una `static_*` del set base**, tiene 18 nulos. La
  imputación va dentro del `Pipeline` de cada fold (regla 3), no sobre el panel.

## Los tres hallazgos de F1 siguen en pie

Medidos sobre la unión deduplicada (1081 vehículos), no re-descubiertos:

**`ENG_3`** ([f1-sesgo-eng3.md](f1-sesgo-eng3.md)): 268 vehículos sanos, **0 con
evento**. La proporción da 0,374 de los sanos contra el 0,361 del doc: la
diferencia es el dedupe (los 26 sanos que se colapsaron eran códigos de vehículos
fallados), no un cambio en el hecho.

| Engine | sanos | con evento | frac sanos | frac con evento |
|---|---|---|---|---|
| ENG_1 | 125 | 72 | 0,175 | 0,197 |
| ENG_2 | 323 | 293 | 0,451 | 0,803 |
| **ENG_3** | **268** | **0** | **0,374** | **0,000** |

**`IdentificationDate` ⇔ cohorte:** 378 filas `failed`, todas con fecha; 742
`not_failed`, todas nulas. Sin excepciones. El chequeo se hace sobre la tabla
**cruda** a propósito: `dedupe_vehicles()` deriva `cohort` de `IdentificationDate`,
así que post-dedupe la equivalencia se cumple por construcción y no probaría nada.

**Anclaje temporal** ([f1-anclaje-temporal.md](f1-anclaje-temporal.md)):

| | `primer_viaje − ProductionDay` | `… − daysUntilSale` |
|---|---|---|
| std | **0,70 d** | 61,97 d |
| IQR | **0,00 d** | 73,00 d |
| rango | 12 d | 353 d |

Idéntico a lo documentado (0,7 / 0,0 / 12 y 60,6 / 73,8 / 353). El puente al eje
de km sigue disponible para F2.

## El holdout dev/test

> **Actualizado el 2026-09-17.** Lo que sigue describe el **sorteo**, que no cambió
> y sigue siendo bit a bit el mismo. Lo que sí cambió es que encima de ese sorteo se
> aplica un **recorte del universo** a 364 vehículos (290 dev / 74 test), y las
> cifras de esta sección son las de antes del recorte. Ver
> [f2-universo-fecha-usable.md](f2-universo-fecha-usable.md).

80/20 sobre el universo de 1081 vehículos, agrupado por `vehicle_id` y
estratificado por `event_observed`, con `StratifiedGroupKFold(5)` y semilla 42
(`src/eval/splits.py::make_test_split`, parámetros en
`configs/data/test_split.yaml`).

| | vehículos | con evento | tasa | % del universo |
|---|---|---|---|---|
| dev | 864 | 292 | 0,3380 | 0,799 |
| **test** | **217** | **73** | **0,3364** | 0,201 |

La tasa de eventos queda a **0,0016** de diferencia, y ningún vehículo aparece de
los dos lados (`test_split_masks` cubre 1081 de 1081 sin solapamiento).

Reparto de las estáticas —**reportado, no estratificado**— para detectar un
desbalance grosero:

| Variable | Valor | frac dev | frac test | dif |
|---|---|---|---|---|
| `ModelSeries` | MODEL_3 | 0,124 | 0,166 | **+4,2 pp** |
| `SalesCountry_cd` | CNTRY_3 | 0,142 | 0,106 | −3,6 pp |
| `ModelSeries` | MODEL_1 | 0,271 | 0,240 | −3,1 pp |
| `SalesCountry_cd` | CNTRY_4 | 0,232 | 0,263 | +3,1 pp |
| `Engine` | ENG_3 | 0,245 | 0,258 | +1,3 pp |
| resto | | | | < 3 pp |

El peor desvío es 4,2 pp en `MODEL_3` (107 vs 36 vehículos). Ninguna categoría
queda concentrada de un lado, así que no hay motivo para volver a tirar el dado
—y si lo hubiera, elegir la semilla mirando este cuadro es exactamente la clase de
decisión que el holdout existe para evitar.

## Congelado

`data/processed/test_split.json` tiene las dos listas de `vehicle_id` con su
metadata (semilla, `test_size`, fecha, huella sha256 del universo). Se guardan las
**dos** listas, no solo la de test: los folds de CV pueden derivar el train como
"el resto del panel" porque el panel ya existe, pero el holdout se congela antes
de que exista, así que "el resto" no está definido al momento de aplicarlo. Con el
universo escrito, `test_split_masks()` puede fallar cuando el panel de F2 traiga un
vehículo que el holdout no conoce, que es justo el caso en que uno se colaría a dev
sin que nadie lo note.

## Cómo se consume

El panel de F2 se construye con **los 364 del universo** —no con los 1081: los
717 excluidos no son holdout, están fuera del estudio— y el recorte dev/test es por
máscara, no por un panel más chico: así la evaluación final no obliga a regenerar el
panel con otro conjunto de vehículos.

```yaml
# en el YAML del experimento — clave OBLIGATORIA
splits:
  test_split: ${DATA_DIR}/processed/test_split.json   # null explícito solo en el dummy
```

`scripts/train.py` llama a `select_dev()` **antes** de armar los folds, así que la
CV, las métricas out-of-fold y las figuras salen solo de dev. Dos formas de fallar,
las dos ruidosas y probadas:

| Situación | Qué pasa |
|---|---|
| el YAML no declara `splits.test_split` | `KeyError` antes de entrenar |
| el panel trae vehículos **excluidos** del universo | `ValueError: N vehículo(s) del panel están EXCLUIDOS…` |
| el panel trae vehículos que el holdout **no conoce** | `ValueError: N vehículo(s) del panel no están en el holdout congelado` |

Las dos últimas están separadas a propósito: la primera es un panel construido con
el universo viejo (se regenera el panel), la segunda es que el universo cambió abajo
del holdout (se regenera el holdout). Sin distinguirlas, las dos terminan en dev por
descarte.

El segundo caso es el que importa: un panel regenerado con vehículos nuevos los
mandaría a dev por descarte, sin que ninguna métrica lo delate.

El motivo del 80/20, del "ahora y no después" y de por qué la clave es obligatoria
está en [decisiones.md](decisiones.md).
