# La contradicción del catalizador no era un error de medición: el DPF no anticipa

**Fecha:** 2026-09-19 · **Fase:** F3 · **Estado: medido, negativo.** Las 16 features
están construidas y declaradas en `configs/data/features_v2.yaml`; **ninguna se promueve
a v1**. El panel canónico sigue siendo el de 53 features.

Reproduce:

```bash
python scripts/build_dataset.py --config configs/data/panel_v2.yaml
python scripts/audit_sequence.py --panel data/processed/panel_v2.parquet --columns all
```

---

## 1 · Qué se probó

Las cinco candidatas de [`../f3-features-candidatas-fisica.md`](../f3-features-candidatas-fisica.md),
todas de una sola pasada: 16 columnas nuevas sobre el mismo panel (W=1000, G=500,
H=3000, Δ=500, mismo muestreo de sanos, mismos folds). El panel v2 reproduce el v1
**exacto** en las 53 features compartidas (diferencia máxima 0 sobre las 2.029 filas de
dev), así que cualquier diferencia es de las nuevas y de nada más.

| candidata | columnas | P(pos>sano) de la mejor | referencia que ya existía |
|---|---|---|---|
| §1.1 eficiencia de la regeneración | 3 | 0,536 | 0,449 (`regen_start_level_mean`) |
| §1.2 residuo que no se va | 2 | 0,454 | 0,461 (`regen_residual_mean`) |
| §2 ciclos que no terminan | 4 | 0,478 | 0,490 (`filter_cleaning_auto_end_frac`) |
| §3 dosis de km fríos | 2 | 0,549 | **0,573** (`trips_below_regime_temp_frac`) |
| §4 temperatura por largo de viaje | 2 | 0,541 | **0,573** (la columna sin partir) |
| §5 oportunidad de regenerar | 3 | 0,561 | 0,474 (`trip_distance_median_km`) |

El techo del panel sigue siendo `feat_idle_frac` con P = 0,597. La mejor nueva,
`feat_regen_opportunity_per_1000km`, da 0,561 con **IC95% [0,501 – 0,628]** (bootstrap de
300 réplicas agrupado por vehículo, que es la unidad de muestreo). Roza el 0,5. Con 254
filas positivas en dev, cualquier P por debajo de ~0,57 es indistinguible de ruido.

Peor que la magnitud: **los signos no cierran entre sí**. La eficiencia medida en puntos
va para donde predecía la hipótesis (los que fallan remueven menos por ciclo, P = 0,465)
y esa misma eficiencia normalizada por el nivel de arranque va para el otro lado
(remueven una fracción *mayor*, P = 0,536). Los ciclos que quedan a medias son **menos**
frecuentes en los que fallan (0,477–0,478), no más. Y la oportunidad de regenerar es
**mayor** en los que fallan (0,561), que es lo contrario de lo que decía el §5. Cuando el
mismo mecanismo medido de tres maneras da tres signos, lo que falla no es la feature.

Lo único que no es ruido y que además tiene la forma correcta es la **dosis de km
fríos contra sanos en la misma posición de la serie**:

| km hasta el evento | >8k | 4–8k | 2–4k | 0–2k |
|---|---|---|---|---|
| `cold_km_per_1000km`, con evento | 2,94 | 3,98 | 5,01 | **5,97** |
| `cold_km_per_1000km`, sanos comparables | 3,48 | 3,17 | 3,25 | 3,52 |

Se duplica hacia el evento mientras el sano comparable queda plano: no es el atajo de
posición. Pero **la fracción de viajes fríos que ya teníamos separa más** (0,573 contra
0,549), así que la dosis no compra nada; el mecanismo térmico ya estaba medido, solo que
con la unidad "equivocada" que resulta funcionar igual de bien.

## 2 · La contradicción del §0, resuelta: no es (1) ni (2)

La hipótesis que organizaba todo el menú era que **el nivel del DPF es una variable
controlada** y lo informativo es el esfuerzo del lazo. Si fuera cierta, los que fallan
tendrían que regenerar **más seguido**, remover **menos por ciclo**, o dejar **más
residuo**. Medido sobre dev, por vehículo y solo con viajes previos al último corte:

| | sanos | con evento |
|---|---|---|
| regeneraciones por 1.000 km | 3,47 | 3,67 |
| `AirRegenerationEnd` medio | 35,8 | 35,2 |
| viajes donde el nivel sube | 43,3% | 42,4% |
| puntos de carga por km recorrido | 0,235 | 0,222 |
| fracción de viajes con lectura de DPF | 1,00 | 1,00 |
| valores distintos de DPF por vehículo | 20 | 20 |

**No hay esfuerzo de control extra.** Y el canal está igual de vivo en las dos cohortes
—misma cobertura, misma resolución, mismo máximo—, así que tampoco es (2) "el sensor se
muere con la pieza". Queda en pie la tercera explicación, que es la incómoda: **el evento
que marca `IdentificationDate` probablemente no es la obstrucción progresiva del DPF**.
Toda la familia B está construida sobre un mecanismo que los datos no muestran.

La única señal del subsistema del filtro que sí distingue cohortes es el nivel
`Air Filter Over Limit`: lo tienen **9 de 53 vehículos con evento (17%) y 3 de 118 sanos
(2,5%)**. Pero no sirve para anticipar: dentro del panel —que respeta el gap de 500 km—
la feature que ya lo mide (`feat_msg_over_limit_ever`) separa con P = 0,509 y aparece en
el 2% de las filas. Además está concentradísimo: de los 387 viajes con ese nivel en dev,
280 son de un solo vehículo. Es un síntoma que llega cuando ya no hay nada que anticipar
—detección reactiva, que es justo lo que Ford ya tiene—.

## 2b · Hallazgo lateral: el P del panel entero **subestima** el mecanismo térmico

Auditando por estratos de posición del corte (`--columns all`, bloque 3), las features
térmicas separan bastante **mejor** dentro de cada estrato que sobre el panel entero:

| | q1 | q2 | q3 | q4 | panel entero |
|---|---|---|---|---|---|
| `feat_idle_per_1000km` | **0,743** | 0,642 | 0,655 | 0,572 | 0,588 |
| `feat_idle_frac` | 0,694 | 0,590 | 0,701 | 0,559 | 0,597 |
| `feat_trips_below_30kmh_frac` | 0,681 | 0,686 | 0,515 | 0,495 | 0,563 |
| `feat_regen_opportunity_per_1000km` | 0,593 | 0,628 | 0,501 | 0,560 | 0,561 |

(q1 = cortes más tempranos de la serie de su vehículo; 577 filas y 43 positivas. q4
concentra 115 de las 254 positivas.)

Hasta ahora el bloque 3 se usaba en una sola dirección —si una columna separa pooled
pero no dentro del estrato, lo que mide es la posición—. Acá pasa lo contrario: la mezcla
de estratos **diluye** la señal, porque las positivas se apilan en q4, que es justo donde
menos separa. El ROC ≈ 0,58–0,60 del panel no es el techo del mecanismo térmico; es el
techo de medirlo sin condicionar por dónde cae el corte. Y la posición **no** puede
entrar como feature (separa sola con P = 0,836, es la etiqueta disfrazada), así que esto
es una pregunta para Track B: cómo aprovecharlo sin comerse el atajo. La familia térmica,
no la de regeneración, es donde vale la pena gastar la próxima hora.

## 3 · Qué queda en el repo

- **Se queda el código.** `src/features/trips.py` tiene las 11 columnas derivadas nuevas
  y `src/features/windows.py` el agregador `slope_per_1000km` (recta sobre los puntos de
  la ventana: existe porque `half_delta` pide 3 puntos **por mitad** y hay ~4
  regeneraciones por cada 1.000 km. Aun con la recta sobre toda la ventana, las
  pendientes del residuo salen NaN en el 32% de las filas).
- **Se queda el YAML v2 y su panel**, como registro medido y para no re-proponerlas.
  `configs/data/panel_v2.yaml` es el v1 con `features_v2.yaml`; nada más cambia.
- **No se promueve nada a v1.** El panel canónico, los splits y el artifact `panel-v1`
  quedan como estaban.
- `scripts/audit_sequence.py` acepta `--columns all|<lista>`: la auditoría de posición
  ya no es solo para la familia E.

## 4 · Qué NO queda descartado

- El mecanismo **térmico** sigue siendo el que más separa (`idle_frac` 0,597,
  `trips_below_regime_temp_frac` 0,573) y es el que tiene gradiente real hacia el evento
  contra sanos comparables. Lo que se descartó es medirlo por dosis en vez de por
  fracción, no el mecanismo.
- **Qué es el evento.** Es ahora la pregunta que más vale: si `IdentificationDate` marca
  una intervención de taller o un reemplazo de sensor, la familia B entera está apuntando
  al lugar equivocado y el proyecto necesita otra hipótesis, no otra feature. Ya figuraba
  como pregunta abierta en [f2-eda-revision-y-features.md](f2-eda-revision-y-features.md) §5.
- **Lo multivariado.** Todo esto es separación univariada: el registry solo tiene
  `baserate` y `random`, así que ninguna de estas columnas se probó dentro de un modelo.
  Una feature que sola da 0,54 puede aportar en combinación; lo que no puede es rescatar
  un panel por sí misma.
