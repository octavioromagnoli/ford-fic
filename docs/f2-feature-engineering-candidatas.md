# Feature engineering para F2 · candidatas medidas

**Fecha:** 2026-09-17 · **Estado (actualizado el 18-09):** parcialmente aplicado en
`configs/data/features_v1.yaml`; lo adoptado y lo retirado está en
[`docs/memoria/f2-eda-revision-y-features.md`](memoria/f2-eda-revision-y-features.md)
y en `decisiones.md`.

> **Corrección que manda sobre §0, §1 y §3.** `regen_per_1000km` —la "feature más
> correlacionada del EDA" que ancla las tres secciones— se calculaba con el marcador
> `Regenerations` de `signals`, que **se corta el 25-05-2026 para toda la flota**. Su
> ρ = 0,317 es exposición al calendario (ρ con `ProductionDay` −0,515); contada desde las
> caídas de `AirRegeneration` en `trips` da ρ = 0,072. La conclusión de §3 ("las
> frecuencias sobreviven al truncamiento, los niveles no") queda **retirada**: el
> truncamiento al 70% del odómetro quitaba sobre todo km posteriores a mayo de 2026, no
> el tramo previo al evento. Lo que sí sobrevivió a la revisión: §1 (indicador +
> magnitud en severidad, adoptado), §4 (pendientes por mitades, adoptado), §5 (recorte
> físico antes de agregar, adoptado con topes en vez de cuantiles), §6 (`oil_life` como
> pendiente, adoptado) y §7. La normalización por mercado (§2) sigue pendiente en Track B.

**Todo lo que sigue se midió sobre los 290 vehículos de dev del universo recortado**
(60 con evento), a nivel **vehículo** y sobre el **historial completo**. Ese no es el
grano en el que van a vivir las features —van a vivir en una ventana W— así que
sirve para **ordenar candidatas**, no para elegirlas. Reproduce:
`notebooks/eda-exhaustivo-dev.ipynb` §7.5.

> **El caveat que manda sobre todo el documento.** 60 eventos en dev, ~12 por fold.
> Una diferencia de |ρ| menor a **~0,08** no se distingue del ruido. Las features se
> eligen **por razones** —física, o sacar un confounder— y no persiguiendo deltas de
> correlación. Y cada variante que probemos es una comparación más que gasta la CV:
> con 5 folds y esta cantidad de positivos, probar veinte ideas garantiza que la
> mejor sea ruido.

---

## 0 · Lo primero: por qué el log hace menos de lo que parece

La intuición es correcta —hay features con valores chicos concentrados y cola larga—
pero el efecto es más chico y más específico de lo que uno espera, por un motivo
matemático:

> **ρ de Spearman es invariante a cualquier transformación monótona.** El log no
> cambia el orden, y Spearman solo mira el orden.

Como los modelos de árbol (LightGBM, XGBoost) **también** solo miran el orden al
elegir los cortes, **el `log1p` no le cambia nada a un GBM**. Lo único que cambia es
**Pearson**, es decir la relación *lineal*, y por lo tanto lo que ve una **regresión
logística** — que es nuestra línea de base y el modelo que más fácil se explica en
el pitch.

Dicho de otra forma: el log no crea señal, **la linealiza**. Acerca Pearson a
Spearman. Y por eso la ganancia máxima posible de transformar una feature es
`|Spearman| − |Pearson|`: si esa brecha es chica, no hay nada que ganar.

### Lo que gana de verdad

| feature | \|Pearson\| | \|Pearson\| con log1p | ganancia | \|Spearman\| |
|---|---|---|---|---|
| `msg_cleaning_auto_per_1000km` | 0,102 | **0,212** | +0,110 | 0,246 |
| `msg_full_per_1000km` | 0,063 | **0,136** | +0,073 | 0,137 |
| `msg_overloaded_per_1000km` | 0,032 | **0,103** | +0,070 | 0,149 |
| `msg_over_limit_per_1000km` | 0,180 | **0,231** | +0,051 | 0,299 |
| `regen_per_1000km` | 0,265 | **0,304** | +0,039 | 0,317 |

Las cinco son **tasas por 1.000 km** de las familias B y D. Tiene sentido: una tasa
es un cociente, y los cocientes se distribuyen de forma multiplicativa. El log es la
transformación natural de una variable multiplicativa.

### Dos trampas que aparecieron al medir

**Trampa 1 · La asimetría sola no es criterio.**

| feature | skew | ganancia log | qué es |
|---|---|---|---|
| `speed_mean` | 12,4 | +0,024 | outliers de medición (hasta 135.600 km/h) |
| `trip_odo_min` | 10,0 | ~0 | unos pocos vehículos con odómetro inicial raro |
| `msg_cleaning_auto_per_1000km` | 7,6 | **+0,110** | cola multiplicativa real |

Las dos primeras tienen skew altísimo y no ganan nada, porque su cola son **errores
de medición**, no una distribución sesgada. Ahí lo que corresponde es **winsorizar
(recortar por cuantil), no transformar**: el log de un dato imposible sigue siendo un
dato imposible, solo que más chico.

**Trampa 2 · Una ganancia de Pearson sin Spearman detrás es ruido.**

| feature | ganancia log | \|Spearman\| |
|---|---|---|
| `acumulation_median` | **+0,064** | **0,003** |
| `acumulation_mean` | +0,032 | 0,025 |
| `air_regen_end_mean` | +0,041 | 0,016 |

`acumulation_median` aparece cuarta en el ranking de ganancia y su correlación de
rangos con la etiqueta es **−0,003**: no separa cohortes en absoluto. El log le
cambió la forma a una variable sin señal y el |Pearson| subió por azar. **Antes de
celebrar una ganancia, mirar la columna de Spearman.**

---

## 1 · Zero-inflation: separar "pasó" de "cuánto"

Ésta rinde más que el log y es más barata.

Varias features de severidad no son continuas: son **casi siempre cero**. Medido
contra la etiqueta, la pregunta es si el *valor* aporta algo por encima del simple
indicador `> 0`.

| feature | % ceros | ρ del valor | ρ del indicador | veredicto |
|---|---|---|---|---|
| `msg_at_limit_per_1000km` | 94,5% | 0,215 | 0,212 | **empatan** → indicador |
| `msg_over_limit_per_1000km` | 87,6% | 0,299 | 0,298 | **empatan** → indicador |
| `msg_overloaded_per_1000km` | 36,9% | 0,149 | 0,161 | gana el indicador |
| `msg_full_per_1000km` | 10,3% | 0,137 | **0,173** | gana el indicador |
| `air_regen_saturated_frac` | 10,0% | 0,096 | **0,170** | gana el indicador |
| `acumulation_saturated_frac` | 8,3% | 0,104 | **0,153** | gana el indicador |
| `air_filter_abnormal_frac` | 11,4% | 0,093 | **0,156** | gana el indicador |
| `regen_per_1000km` | 11,4% | **0,317** | 0,183 | **gana el valor, por mucho** |

**Dos lecturas, opuestas entre sí:**

1. **En los mensajes de severidad y en las fracciones de saturación, toda la señal
   está en "pasó alguna vez".** Para `msg_at_limit` y `msg_over_limit` el indicador
   empata al valor a tres decimales — con 94,5% y 87,6% de ceros, el "cuántas veces"
   es ruido sobre los pocos que no son cero. Y en las de saturación el indicador
   **supera** al valor: haber saturado alguna vez discrimina más que cuánto tiempo se
   estuvo saturado.
2. **En `regen_per_1000km` es al revés:** 0,317 contra 0,183. Binarizarla tiraría
   casi la mitad de la señal. Tiene sentido físico — lo que importa no es *si*
   regenera (todos regeneran) sino **cada cuántos km**.

**Propuesta concreta:** para cada familia de mensaje, emitir **las dos**:

```
feat_msg_over_limit_ever          # bool: ¿apareció al menos una vez en la ventana?
feat_msg_over_limit_log_per_1000km  # log1p de la tasa
```

Cuesta una columna extra por feature, el pipeline las toma por prefijo sin tocar
código, y deja que el modelo decida. Con árboles el indicador es además un corte que
el modelo puede encontrar solo, así que el costo real es casi nulo.

**Lo que NO hay que hacer:** binarizar `regen_per_1000km` ni las features de
distancia entre regeneraciones.

---

## 2 · Normalizar por mercado (la que más protege la generalización)

Ésta no mejora una correlación: **evita que el modelo aprenda el país en vez del
síntoma.**

El universo quedó con dos mercados y **no miden parejo**:

| feature | CNTRY_3 (n=99) | CNTRY_4 (n=191) | ratio |
|---|---|---|---|
| `msg_full_per_1000km` | 2,78 | 15,11 | **5,45x** |
| `air_temp_avg_mean` | 17,9 °C | 26,0 °C | +8 °C |
| `speed_mean` | 31,9 km/h | 19,4 km/h | 0,61x |
| `urban_trip_frac` | 0,38 | 0,51 | 1,36x |
| `km_per_day` | 42,1 | 34,1 | 0,81x |

La buena noticia es que la **dirección** de la señal coincide: 9 de las 10 variables
del ranking tienen el mismo signo en los dos mercados (la décima, `short_trip_frac`,
es ρ ≈ 0 en ambos, o sea que no discrepan, no dicen nada). Por eso se pueden
entrenar juntos.

Pero el **nivel** difiere tanto que una tasa absoluta lleva adentro la etiqueta del
país. Con 191 vehículos de CNTRY_4 (tasa de eventos 0,236) y 99 de CNTRY_3 (0,152),
un modelo que aprenda "tasa de `Full` alta ⇒ riesgo" puede estar aprendiendo
"CNTRY_4 ⇒ riesgo". Eso funciona en la CV y se cae con el primer mercado nuevo.

**Propuesta:** para las features más sensibles al mercado (todas las
`msg_*_per_1000km`, `air_temp_*`, `speed_*`), emitir además la versión normalizada
**dentro del mercado**:

```
feat_msg_full_rank_in_market      # percentil dentro de su SalesCountry_cd
```

**Restricción no negociable (regla 3):** el percentil o el z-score se calcula con los
estadísticos del **train de cada fold**, dentro del `Pipeline`. Calcularlo sobre el
panel entero es leakage aunque no lo parezca, porque el ranking de un vehículo de
validación depende de dónde caen los de train.

Implementación: un transformer propio en el `ColumnTransformer` de
`src/training/cv.py`, o precalcularlo en el panel **por fold** — lo primero es más
limpio y respeta la arquitectura actual.

---

## 3 · Frecuencias antes que niveles (lo dice el truncamiento)

No es una transformación: es **qué features construir primero**, y es el resultado
mejor respaldado de todo el EDA.

Recalculando cada agregado usando solo el **primer 70% del recorrido** de cada
vehículo —un proxy del gap de blanking— la correlación con la etiqueta se comporta
de dos maneras distintas:

**Sobreviven** (retienen ≥ 86% de su ρ):

| feature | historial completo | primer 70% | retiene |
|---|---|---|---|
| `regen_per_1000km` | 0,317 | 0,286 | **90%** |
| `msg_over_limit_per_1000km` | 0,299 | 0,302 | **101%** |
| `msg_cleaning_auto_per_1000km` | 0,246 | 0,225 | **91%** |
| `msg_at_limit_per_1000km` | 0,215 | 0,184 | 86% |
| `msg_overloaded_per_1000km` | 0,149 | 0,144 | 97% |

**Se caen:**

| feature | historial completo | primer 70% | retiene |
|---|---|---|---|
| `acumulation_saturated_frac` | 0,104 | 0,073 | 70% |
| `below_regime_frac` | 0,101 | 0,072 | 71% |
| `air_regen_saturated_frac` | 0,096 | 0,055 | **58%** |
| `air_regen_end_mean` | 0,016 | 0,005 | **32%** |
| `acumulation_high_frac` | 0,006 | 0,001 | **18%** |

**La interpretación es la regla 6 de `CLAUDE.md` medida con datos.** Las
**frecuencias de regeneración** son un rasgo del régimen de uso: están presentes
desde el principio del historial, así que blanquear la cola no les saca nada — eso
es **anticipar**. Los **niveles de saturación del filtro** son un estado que aparece
cerca del evento: describen lo que ya está pasando — eso es **detectar**, que es lo
que Ford ya tiene.

**Propuesta:** construir la familia B alrededor de frecuencias
(`regenerations_per_1000km`, `distance_between_regen_*`) y sus tendencias, y tratar
`dpf_end_mean` / `dpf_end_max` / `accumulation_mean` como **sospechosas**: se
incluyen, pero cualquier modelo que dependa mucho de ellas cae bajo la regla 6 y hay
que auditarlo antes de festejarlo.

**Dos caveats de esta prueba, para no sobre-venderla:**

1. La etiqueta es **a nivel vehículo**, no por punto de corte. El truncamiento es un
   proxy del gap G, no el gap G. La prueba definitiva es PR-AUC con G y sin G sobre
   el panel.
2. El 70% del recorrido **no** es el 70% del tiempo hasta el evento: con el evento en
   la mediana al 39% del historial, truncar al 70% deja *dentro* parte del tramo
   post-evento para varios vehículos. O sea que la prueba es **conservadora**, y que
   `regen_per_1000km` retenga el 90% es mejor noticia de lo que parece.

---

## 4 · Pendientes dentro de la ventana, no sobre el historial

Hay un bug conceptual en cómo están medidas hoy las pendientes, y se ve en el signo.

`acumulation_slope_per_1000km` da **ρ = −0,148**: los vehículos **sanos** tienen
pendiente más empinada (0,978) que los que fallan (0,568). Al revés de lo que dice
la física.

La explicación no es física, es aritmética: la pendiente está ajustada sobre el
**historial completo**, y los vehículos con evento tienen historiales más largos
(mediana 26.175 km contra 15.508 km). Una recta ajustada sobre un tramo más largo de
una serie que **satura en 95** tiene menos pendiente por construcción. Es el mismo
confounder de exposición que `ProductionDay`, disfrazado de feature física.

**Dentro de una ventana W fija el problema desaparece**, porque todas las pendientes
se ajustan sobre el mismo largo. Es una de las razones principales por las que el
panel existe.

**Además, hace falta un estimador robusto.** `dist_between_regen_slope_per_1000km`
llega a **−900,76** en un vehículo: mínimos cuadrados sobre pocos puntos con un
outlier explota. Opciones, de más simple a más cara:

1. **Diferencia de medianas entre la primera y la segunda mitad de la ventana.**
   Trivial de implementar, imposible de explotar por un outlier, y se explica en una
   frase en el pitch.
2. **Regresión de Theil-Sen** (`sklearn.linear_model.TheilSenRegressor`): mediana de
   las pendientes de todos los pares. Robusta y sigue siendo una pendiente.
3. Mínimos cuadrados sobre datos winsorizados. Lo menos convincente de los tres.

Recomiendo la (1) para la primera versión: `feat_distance_between_regen_trend` ya
está reservada en `FEATURE_SPECS` y la definición por mitades es la más defendible.

---

## 5 · Recortar antes de agregar

`notebooks/eda-exhaustivo-dev.ipynb` §5.3 encontró rangos imposibles:

| columna | valor extremo | qué es |
|---|---|---|
| `KilometerPerHour` | hasta **135.600 km/h** | error de medición |
| `AirTemperatureAvg` | hasta **−73 °C** | error de medición |
| `trip_duration_min` | hasta 20 días | viaje sin cerrar |
| `FuelLvl*Pc` | **>100** en el 10% de los viajes (hasta 103,5) y hasta −5,2 | la escala no es un % literal |

Son colas finitas (menos del 0,05% de los viajes cada una) pero **una media sin
recorte las arrastra**. Winsorizar a [p1, p99] **antes** de agregar la ventana, con
los cuantiles del train de cada fold.

Caso aparte `FuelLvl*Pc`: no es una cola, es que la escala no es un porcentaje
literal. Tratarla como **índice relativo** y no recortarla a [0, 100] sin pensarlo,
porque ese recorte inventa un techo que el dato no tiene.

---

## 6 · Redefinir `feat_oil_life_drop_per_1000km`

Ya documentado, se repite acá porque es feature engineering y no calidad de dato:
`EngineOilLifePCStart == EngineOilLifePCEnd` en el **100%** de los viajes. El delta
intra-viaje es exactamente cero, siempre: el índice se actualiza **entre** viajes, no
dentro.

La columna no está vacía —el nivel recorre 0–100 y baja a lo largo de la vida del
vehículo— así que la feature hay que **redefinirla como pendiente del nivel contra
odómetro dentro de la ventana**, no como suma de caídas por viaje. Con el mismo
estimador robusto de §4.

---

## 7 · Lo que NO hay que hacer

- **No aplicar `log1p` a `acumulation_*` ni a `air_regen_end_*`.** Son asimétricas a
  la **izquierda** (skew −1,5) porque saturan en 95: el log las empeora. Lo que
  funciona ahí es "fracción de tiempo en nivel alto", que ya está en el cache.
- **No binarizar `regen_per_1000km`** (§1).
- **No agregar interacciones ni términos polinómicos.** Con 60 eventos en dev y ~12
  por fold, cualquier expansión del espacio de features es sobreajuste garantizado.
- **No usar las variables de cobertura como features**: `span_days` (ρ = 0,361),
  `n_signals` (0,279), `n_trips` (0,260), `km_observed` (0,226) son exposición, no
  física. Dentro de una ventana W fija dejan de serlo, pero
  `feat_n_trips_window` y `feat_window_km_covered` están reservadas como **control
  de calidad de la ventana**, no como hipótesis: si el modelo se apoya en ellas, es
  una señal de alarma, no un hallazgo.
- **No elegir features mirando el ρ de este documento.** Ver el caveat de arriba.

---

## Orden sugerido de implementación

| # | Qué | Costo | Por qué en ese orden |
|---|---|---|---|
| 1 | Winsorizar antes de agregar (§5) | bajo | Sin esto, todas las medias están contaminadas |
| 2 | Pendientes dentro de W, estimador robusto (§4, §6) | bajo | Arregla un signo que hoy está al revés |
| 3 | Indicador + magnitud en severidad (§1) | bajo | La mejor relación señal/esfuerzo del documento |
| 4 | `log1p` en las 5 tasas de §0 | trivial | Solo mueve la logística; gratis igual |
| 5 | Normalización por mercado (§2) | medio | Necesita un transformer por fold; es lo que más protege |

Los primeros cuatro son features nuevas o corregidas y no requieren tocar
`src/training/cv.py` —entran por prefijo—. El quinto sí toca el pipeline, y por eso
va último: conviene tener una línea de base medible antes de cambiar la mecánica.

**Cada uno de estos pasos es una comparación out-of-fold contra la corrida anterior,
y se registra en `docs/memoria/decisiones.md` si se adopta.** No se adoptan en
bloque.
