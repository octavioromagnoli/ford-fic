# Revisión del EDA y feature engineering de F2: qué se sostiene, qué estaba mal y qué se construyó

**Fecha:** 2026-09-18 · **Fase:** F2 · **Alcance:** dev (290 vehículos, 60 eventos) para
todo lo que se mide; el panel se construye con los 364 del universo. · **Reproduce:**

> **Corrección del 20-09:** las mediciones de regeneraciones de este documento usaron
> caídas de 5 puntos. Ese umbral cuenta ruido y fue reemplazado por 15 puntos; ver
> [f2-umbral-regeneraciones.md](f2-umbral-regeneraciones.md). Las cifras históricas de
> abajo se conservan para no reescribir qué se midió.

```bash
python scripts/eda_gaps.py                                  # deja experiments/eda/dev/gaps/*.csv
python scripts/build_dataset.py --config configs/data/panel_v1.yaml
python scripts/train.py --config configs/exp_baserate.yaml  # piso contra el panel real
```

Este archivo es el **resumen operativo** de la segunda pasada por el EDA. Lo que ya
estaba en `notebooks/eda-exhaustivo-dev.ipynb` y sigue valiendo no se repite; lo que
se corrige se dice con el número que lo corrige.

---

## 1 · Lo que se revisó y se sostiene

Verificado sobre los mismos datos: el universo de 364 (§3.3 del notebook), el anclaje
temporal (IQR 0 días también con la estimación congelada del panel: día 20107 desde
epoch), el alias `AirRegeneration* == Acumulation` y `AirFilter* == Message`, el sesgo
de `ENG_3` y el cruce diagonal con `ModelSeries`, `ProductionDay` como ventana de
observación, `EngineOilLifePC` constante dentro del viaje, los rangos imposibles de
§5.3, la cobertura del join y los duplicados. Ninguno cambió.

## 2 · Tres cosas que el EDA tenía mal o incompletas

### 2.1 · `regen_per_1000km` no era señal: el marcador `Regenerations` se corta el 25-05-2026

La "variable que más separa de todo el EDA" (§6.2 y §7.4 del notebook, ρ = 0,317) se
calculaba con el marcador literal de `signals`. Ese marcador **deja de registrarse el
25-05-2026 para toda la flota**; los viajes siguen hasta el 14-09-2026.

| mes | marcadores `Regenerations` | caídas de `AirRegeneration` en `trips` | vehículos activos |
|---|---|---|---|
| 2026-03 | 1.876 | 1.834 | 274 |
| 2026-04 | 1.539 | 1.845 | 273 |
| 2026-05 | 1.520 | 1.935 | 275 |
| **2026-06** | **0** | 1.910 | 272 |
| 2026-07 | 1 | 2.080 | 272 |
| 2026-08 | 0 | 2.168 | 274 |

Hasta marzo la razón marcadores/caídas es ≈ 1,0; después es 0. Último marcador por
vehículo: p50 21-05-2026 y p90 25-05-2026 en las dos cohortes; 33 sanos no tienen ni
uno. La tasa por km sobre el historial completo mide entonces **qué fracción del
historial cae antes de esa fecha**, y eso lo decide cuándo se produjo el vehículo:

| tasa por 1.000 km | ρ con `ProductionDay` | ρ con la etiqueta |
|---|---|---|
| marcadores `Regenerations` (signals) | **−0,515** | **0,317** |
| caídas > 5 de `AirRegeneration` (trips) | 0,066 | 0,072 |

Es el sesgo de `ProductionDay` disfrazado de feature física. `DistanceBetweenRegenerations`
es la distancia al marcador anterior (corr 0,92, |diferencia| mediana 0 km) y hereda el
corte. **Consecuencia:** la familia B se construye desde las caídas de `AirRegeneration`
en `trips`; el marcador queda en el panel como `aux_regen_marker_per_1000km` solo para
auditar. Las recomendaciones de `docs/f2-feature-engineering-candidatas.md` §3 ("las
frecuencias sobreviven al truncamiento") descansaban sobre este artefacto y quedan
retiradas.

### 2.2 · La escala de `AirRegeneration` / `Acumulation` es 0–100, no 0–95

100 aparece en el 0,58% de los fines de viaje (39 vehículos de dev) y en 11.193 filas
de `signals`. 95 es un escalón con masa (5,8%), no el tope. Corrige
[f2-columnas-airfilter-airregeneration.md](f2-columnas-airfilter-airregeneration.md).
`saturation_level: 95` sigue siendo un umbral razonable.

### 2.3 · El 35% de las filas de `trips` no son viajes: son idle de 0 km

`trip_km == 0` en el 35,3% de las filas de dev: motor encendido sin desplazamiento,
mediana 0,8 min, p90 6,4 min, `EngineTemperatureMax` mediana 76 °C. `KilometerPerHour`
es nulo **exactamente** en esas filas (100%) y en ninguna otra (0,05%); cuando existe
es `trip_km / duración` con Pearson 0,99999. Dos consecuencias:

- toda fracción "de viaje" (trayecto corto, régimen, urbano) se calcula entre los
  viajes con desplazamiento, y los idle son una feature aparte;
- la velocidad se recalcula desde odómetro y tiempo, con tope físico de 200 km/h.

## 3 · Lo que faltaba medir

### 3.1 · Factibilidad de la terna (W, G, H, Δ)

km antes del evento en dev: p10 1.792, p25 3.036, **p50 7.987**, p75 12.814. Con Δ = 500:

| W | G | H | eventos con positivo (de 60) | sanos con corte verificable (de 230) | filas positivas | filas sanas |
|---|---|---|---|---|---|---|
| 500 | 250 | 3000 | 58 | 207 | 348 | 7.103 |
| **1000** | **500** | **3000** | **54** | 201 | 318 | 6.765 |
| 2000 | 500 | 3000 | 48 | 196 | 288 | 6.366 |
| 3000 | 500 | 3000 | 41 | 193 | 261 | 5.977 |

Cada 500 km de W cuesta ~3 eventos. Se toma W = 1000 (barrido en §4.4).

### 3.2 · Perfil alineado al evento: qué anticipa y qué solo detecta

P(vehículo con evento > sano), comparando el agregado de cada vehículo con evento en un
tramo de km previo al evento contra el agregado de los sanos en 2.000–10.000 km de odómetro:

| feature | 4–8k antes | 2–4k antes | 0–2k antes | después |
|---|---|---|---|---|
| `idle_frac` (filas de 0 km) | 0,58 | 0,66 | **0,74** | 0,66 |
| `below_regime_frac` (EngineTemperatureMax < 70) | 0,66 | 0,71 | **0,75** | 0,63 |
| `below_regime_moving_frac` | 0,65 | 0,68 | 0,67 | 0,56 |
| `short_moving_frac` (< 5 km entre los que se mueven) | 0,53 | 0,61 | 0,60 | 0,47 |
| `speed_calc_median` | 0,42 | 0,33 | **0,28** | 0,42 |
| `coolant_end_mean` | 0,44 | 0,38 | 0,32 | 0,40 |
| `msg_cleaning_auto_per_1000km` | 0,48 | 0,54 | 0,63 | 0,59 |
| `regen_drops_per_1000km` | 0,43 | 0,43 | 0,42 | 0,51 |
| `air_regen_end_mean` (nivel DPF) | 0,35 | 0,35 | 0,31 | 0,36 |
| `cold_start_frac`, `fuel_pct_per_100km`, `stopped_auto` | ≈ 0,5 | ≈ 0,5 | ≈ 0,5 | ≈ 0,5 |

Lectura: **la señal que anticipa es térmica y de uso** (más arranques sin moverse, más
viajes que no llegan a régimen, más lento y más frío), y crece de forma progresiva en
los últimos ~4.000 km. Los **niveles** del filtro van al revés de lo esperado (los que
fallan terminan los viajes con el filtro *menos* cargado) y las regeneraciones por km
son levemente *menores*. Las interrupciones de regeneración
(`Stopped Cleaning Automatically`) son 1 mensaje de cada 100.000: no se pueden medir.

### 3.3 · Después del evento el uso cambia

Mismo vehículo, antes vs. después de `IdentificationDate` (57 vehículos con ambos
tramos): `below_regime_frac` baja en el 81% (0,32 → 0,22), la velocidad sube en el 81%,
`air_regen_end_mean` sube en el 84% (35 → 42), los mensajes `Full` por km suben en el
77%. La fecha marca una intervención real, y lo posterior **no** puede entrar a
ninguna ventana ni como negativo.

### 3.4 · El confusor calendario

Los 60 eventos de dev caen entre **2025-09 y 2026-03** (pico nov–feb). Los km de los
sanos se concentran en **2026-03 a 2026-08** (~0,10 de la exposición por mes, contra
0,03–0,07 en los meses de los eventos). Cualquier canal con deriva temporal se vuelve
detector de etiqueta: el marcador de §2.1, la temperatura ambiente (mediana mensual
16,7 → 25,0 °C), `ProductionDay`.

Medido sobre el panel (LGBM, CV agrupada sobre dev): agregar tres columnas de
calendario (`aux_air_temp_avg`, `aux_regen_marker_per_1000km`, `aux_static_ProductionDay`)
sube el ROC-AUC de **0,67 → 0,76** cuando los sanos se emparejan solo por odómetro, y
de **0,57 → 0,60** cuando se emparejan por odómetro **y mes**. El emparejado por mes
cierra el atajo; el emparejado solo por odómetro no.

### 3.5 · `daysUntilSale` es exposición, igual que `ProductionDay`

La fecha de venta (`ProductionDay + daysUntilSale`) da ρ = **−0,48** con la etiqueta y
la tasa de eventos por quintil va de **0,53 a 0,00**; `daysUntilSale` sola: ρ = −0,41 con
el odómetro final, tasa por quintil 0,38 → 0,05. Un vehículo vendido tarde no llegó a
acumular km antes del corte de los datos. Con W/G/H fijos, un modelo que la use aprende
cuándo se vendió el auto. Sale del set base (`aux_static_daysUntilSale`).

### 3.6 · Rasgo o estado: ICC del vehículo en ventanas de 2.000 km (sanos)

`air_temp` 0,77 · `urban_frac` 0,76 · `cold_start_frac` 0,70 · `msg_full` 0,64 ·
`regen_drops` 0,59 · `short_trip_frac` 0,58 · **`below_regime_frac` 0,31** ·
`air_regen_end_mean` 0,27 · `msg_over_limit` / `msg_at_limit` ≈ 0. Las features que
anticipan (§3.2) son justamente las que varían dentro del vehículo: el panel de ventanas
tiene sentido. Las de mensajes raros son ruido puro por ventana.

### 3.7 · Chequeos de dato

- `TripNumber` es único por vehículo y consecutivo (99,4% de saltos = 1) pero no
  monótono en el tiempo en el 80% de los vehículos: hay viajes fuera de orden temporal.
  El panel ordena por odómetro, no por fecha.
- 395 viajes (224 vehículos) empiezan más de 1 km antes del fin del viaje anterior en
  orden temporal; 137 viajes en 8 vehículos tienen km negativo y se descartan.
- `signals` termina 25 km / 1 día antes que `trips` en la mediana; 2 vehículos con más
  de 30 días de diferencia. Con ventanas de 1.000 km no importa.
- `daysUntilSale` tiene 6 nulos en dev; `ProductionDay` ninguno.
- El consumo estimado desde `FuelLvl*Pc` (caída por 100 km en viajes sin recarga) solo
  existe en el 33% de los viajes y no separa cohortes (P ≈ 0,43). Queda como feature
  barata, sin expectativa.

## 4 · Lo que se construyó

### 4.1 · Pipeline

| Módulo | Qué hace |
|---|---|
| `src/data/subset.py` | lee `trips`/`signals` para un conjunto de vehículos: canoniza → filtra → deduplica filas exactas (la fila completa) |
| `src/data/anchor.py` | origen del calendario (estimado sobre dev, congelado en `panel_meta.json`) y odómetro del evento por interpolación |
| `src/features/trips.py` | derivadas a nivel viaje: idle/moving, velocidad recalculada, topes físicos, regen = caída > 15 de `AirRegeneration`, reposo, consumo |
| `src/features/signals.py` | una booleana por nivel de `Message`; `regen_marker` solo como aux |
| `src/features/windows.py` | primitiva de ventana `(c − W, c]` sobre odómetro + agregadores (`per_1000km`, `half_*` para tendencias robustas, `gap_*`, `km_since_last`, …) |
| `src/data/panel.py` | cortes en grilla de Δ, etiqueta, política de censura, QC de ventana, emparejado de sanos |
| `scripts/build_dataset.py` | orquesta todo y deja `panel.parquet`, `splits.json` (folds sobre dev) y `panel_meta.json` |
| `configs/data/features_v1.yaml` | **la lista de features**: una línea por feature; agregar una es agregar una línea |

### 4.2 · Terna y etiqueta (panel v1)

W = 1000, G = 500, H = 3000, Δ = 500 km. Cortes en múltiplos de Δ desde `first_odo + W`.
Con evento en E: cortes hasta E − G; `label = 1` si c ≥ E − G − H. Sanos: solo cortes con
c + G + H dentro de lo observado (`censored_policy: drop`). QC: ≥ 5 viajes y ≥ 50% de W
en km recorridos por ventana (871 filas descartadas de 9.445).

### 4.3 · Emparejado de sanos

Se conserva el mayor subconjunto de filas sanas cuya distribución de **(bin de 4.000 km,
mes del corte)** reproduce la de las filas positivas de dev, tolerando un 25% de déficit.

| regla | filas sanas (de 7.308) | vehículos sanos (de 243) | TV odómetro | TV mes |
|---|---|---|---|---|
| solo odómetro (2.000 km) | 4.916 | 243 | 0,07 | **0,57** |
| 2.000 km × mes, déficit 10% | 682 | 125 | 0,08 | 0,04 |
| **4.000 km × mes, déficit 25%** | **1.241** | **144** | 0,15 | 0,07 |
| solo mes | 1.889 | 153 | 0,19 | 0,07 |

(TV = distancia de variación total entre las marginales de positivos y sanos en dev.)

Panel resultante: **2.507 filas, 211 vehículos**; dev 2.029 filas / 171 vehículos / 53
con evento (todos con algún positivo) / 254 positivas (tasa 0,125); test 478 filas / 40
vehículos. Del test no se registra nada más (ni positivos ni tasa): `build_dataset.py`
solo imprime filas y vehículos de ese lado, y `panel_meta.json` igual. 34 vehículos de
test no aparecen (sanos sin corte verificable o fuera de las celdas emparejadas, o
eventos con menos de 1.500 km de historial previo).

### 4.4 · Qué da, honestamente (dev, CV agrupada, folds congelados)

| panel | modelo | PR-AUC (lift) | ROC-AUC | detección @ ≤200 FA/1000 |
|---|---|---|---|---|
| v1 (W=1000) | tasa base | 0,121 (1,0×) | 0,49 | — |
| v1 (W=1000) | logística L2 balanceada | 0,159 (1,3×) | 0,58 | 0,26 · 9.181 km |
| v1 (W=1000) | LightGBM chico | 0,165 (1,3×) | 0,59 | 0,43 · 8.179 km |
| W=2000 | logística / LGBM | 0,162 / 0,166 | 0,60 / 0,58 | 0,46 / 0,52 · ~7–9k km |
| W=3000 | logística / LGBM | 0,173 / 0,158 | 0,59 / 0,51 | 0,53 / 0,33 |
| v1 sin emparejar | LGBM | 0,073 (2,0×) | 0,67 | 0,43 |
| v1 solo odómetro | LGBM | 0,102 (2,1×) | 0,66 | 0,43 |

Los modelos son de auditoría (no están en el registry): sirven para saber cuánta señal
hay, no para elegir. Dos lecturas:

1. **La señal honesta es débil**: ROC ≈ 0,58–0,60 y lift ≈ 1,3× con 53 eventos. Es el
   techo realista de esta terna con estos datos; un número mucho mejor es sospechoso
   (regla 6), como muestran las dos últimas filas, donde el modelo gana con odómetro y
   calendario y no con uso.
2. **W no cambia la conclusión** (1.000–3.000 km dan lo mismo dentro del ruido); se
   queda W = 1000 porque conserva más eventos.

Las features que más separan a nivel fila (P(positivo > sano)): `speed_kmh_mean` 0,39,
`idle_frac` 0,61, `trip_duration_median_min` 0,60, `trips_below_regime_temp_frac` 0,59,
`trips_below_30kmh_frac` 0,59. Coincide con §3.2: la historia del pitch es térmica y
de uso, no de saturación del filtro.

### 4.5 · Convención nueva: `aux_`

Columnas que están en el panel **y no entran al modelo** (`cv.py` selecciona solo
`feat_`/`static_`): `aux_air_temp_*`, `aux_regen_marker_per_1000km`,
`aux_static_{Engine, ModelSeries, ProductionDay, daysUntilSale}`, y los controles de
ventana. Sirven para ablaciones y para auditar atajos sin reconstruir el panel. La única
`static_` del set base es `SalesCountry_cd`.

## 5 · Lo que queda abierto

- **Preguntarle a Ford** por qué `Regenerations` se corta el 25-05-2026 (¿cambio de
  firmware, de extracción?) y qué es `IdentificationDate` en los 284 descartados.
- Normalización por mercado dentro del `Pipeline` (Track B) y `log1p` para la
  logística como transformer, no como columnas.
- El test tiene **14 eventos con positivo**: la curva de anticipación final va a tener
  intervalos de ±25 pp. Decirlo en el informe.
- El notebook `eda-exhaustivo-dev.ipynb` §6.2, §6.3 y §7.4 siguen mostrando
  `regen_per_1000km` como señal; no se re-ejecutó. Este archivo y
  `scripts/eda_gaps.py` son la versión vigente.
- Barrido de G y H (el de W ya está); H más corto (2.000) daría menos positivos pero
  más cercanos a la ventana donde la señal existe (§3.2).
