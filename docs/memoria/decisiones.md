# Decisiones

Una entrada por decisión, la más nueva arriba. El **porqué** es la parte que
importa: sin él, el que venga la revierte sin enterarse de qué estaba resolviendo.

---

## 2026-09-19 · TimesFM re-medido sobre el panel v1: no entra, ni como score ni como features

**Decidió:** rama `feat/model-timesfm-zeroshot` (Octavio), al traer el EDA de main; queda
a revisión en el PR. **Detalle:** [f3-timesfm-zeroshot.md](f3-timesfm-zeroshot.md).

La primera versión (16-09) armaba sus propios cortes sobre los 1081, contaba
regeneraciones con el marcador `Regenerations` y emparejaba los sanos solo por odómetro:
las tres cosas que el EDA del 18-09 invalidó. Se rehízo sobre los cortes, el holdout y
los folds del panel v1, y los resúmenes del pronóstico se miden como **variante del panel
con las mismas filas** (`src/data/panel.py::attach_columns`) contra el mismo LightGBM
sobre el panel v1:

| | PR-AUC |
|---|---|
| zero-shot, mejor canal (mensajes malos) | 0,128 (tasa base 0,125) |
| LightGBM sobre el panel v1 (control), R=3 | 0,161 ± 0,006 |
| LightGBM + 9 `feat_tfm_*`, R=3 | 0,169 ± 0,007: +0,008, pero 9/15 folds y ROC igual. Ruido |

No entra a `features_v1.yaml`, y tampoco podría: los pesos de la 3.0 son de licencia no
comercial. **Lo que sí queda es una pista:** la feature que más separa de todo lo medido
a nivel fila es `feat_tfm_regen_delta` (ROC 0,614), y su versión sin modelo —media de
la historia previa menos media de la ventana— da 0,602 y no está en el panel. Es la idea
2.4 de `docs/f3-modelos-candidatos.md`; se construye sin TimesFM, auditando calendario antes.

**Cambios chicos en código compartido que salen de acá:** `attach_columns` en
`src/data/panel.py`; builder `lgbm` con los defaults del LightGBM chico del §1.3 del doc
de F3; `configs/exp_lgbm_panel_v1.yaml` como control de cualquier ablación de familia.

---

## 2026-09-19 · `date_format: ISO8601` para `trips` y `signals`

**Decidió:** rama `exp/hazard-xgb-landmark` (Octavio); queda a revisión en el PR.
**Detalle:** [f2-fechas-formato-mixto.md](f2-fechas-formato-mixto.md).

Los timestamps mezclan dos formatos y pandas anulaba en silencio los del formato que no
era el de la primera fila del chunk: el 0,3% de "fechas nulas" era eso (el crudo tiene
0). Con el formato declarado, el panel v1 conserva filas, etiquetas y emparejado, pero
cambian 13 features en hasta 102 filas: las temporales de `trips` y, por el dedupe de
fila completa, las de `signals` (2.420 mensajes distintos se descartaban como
repetidos). **Pendiente:** regenerar y volver a publicar `panel-v1` con el arreglo.

---

## 2026-09-18 · La CV estratifica por `label` a nivel vehículo, con guarda de positivos y CV repetida

**Decidió:** Santino (auditoría de la CV sobre el universo de 364). **Código:**
`src/eval/splits.py`. **Parámetros:** bloque `splits:` de
`configs/data/panel_v1.yaml` y de cada `configs/exp_*.yaml`.

**Qué cambia.** `make_splits()` estratifica por `label` colapsado a nivel vehículo
(*¿tiene al menos un corte positivo?*) en lugar de `event_observed`; falla si algún
fold queda con menos de `min_valid_positives` filas `label=1` en validación
(default 5); y acepta `n_repeats` juegos de folds con semillas distintas
(`seed + r·10000`, default 1). Las tres salen del YAML.

**Por qué la estratificación.** El PR-AUC se calcula sobre `label` —¿el evento cae
en el horizonte de ESTA fila?— y se estratificaba por `event_observed` —¿el vehículo
falló alguna vez?—. Hoy las dos coinciden y por eso **el reparto no cambió**: en los
cuatro paneles que existen, todo vehículo con evento que llega al panel tiene al
menos un corte positivo (panel v1: 53 de 53; w2000: 46; w3000: 40; sin emparejar:
53). No es casualidad: los cortes de un vehículo con evento terminan en `E − G` y
`label = 1 ⟺ corte ≥ E − G − H`, así que **el último corte es positivo por
construcción** (`src/data/panel.py::vehicle_cuts`). La brecha se abre solo si el QC
de ventana (`min_trips_in_window`, `min_km_covered_frac`) descarta los cortes
tardíos de un vehículo, que es más probable cuanto más grande sea W. Derivarla de
`label` cierra esa puerta antes de que W crezca, y el día que se abra, la CV se
entera en vez de equilibrar una variable que la métrica no usa.

**Lo que NO arregla.** Estratificar a nivel vehículo no achica la dispersión de
positivos por fold, porque un vehículo aporta entre 1 y 6 filas positivas. Con la
terna actual el spread es 1,13x (47/53/53/52/49) y sigue igual. Lo único medido que
lo baja es estratificar por `label` **fila a fila** (`stratify.level: row`): 1,06x
con seed 42, y entre 1,04x y 1,08x en cinco semillas, contra 1,08x–1,20x. Quedó
implementado y configurable, pero **no adoptado**: cambia el reparto y obliga a
recorrer de nuevo todo lo comparado. Se decide cuando haya modelos que comparar.

**Por qué la guarda.** `make_splits` reportaba `valid_positive_rate` y no lo
chequeaba: un fold sin positivos daba PR-AUC = NaN y se promediaba en silencio. La
guarda vieja (`n_event_vehicles < n_splits`) miraba la variable equivocada por el
mismo motivo que la estratificación. Ahora falla nombrando repetición, fold y
conteo, y se **re-chequea al entrenar** (`iter_repeats`), porque un `splits.json`
congelado antes de que la guarda existiera no la trae adentro.

**Por qué la CV repetida.** Con 47–53 positivos por fold, la diferencia de PR-AUC
entre dos modelos se confunde con la varianza del sorteo (plan §10, "pocos
eventos"). Default 1 para no mover nada: la repetición 0 usa `seed` tal cual, así
que subir R **suma** pasadas y no invalida lo ya corrido. Medido con el piso de
tasa base y R=5: PR-AUC 0,1221 ± 0,0007 (min 0,1211, max 0,1230).

**Qué se guarda y qué no.** `predictions.parquet` sigue teniendo **una fila por
corte**, con `score` = promedio de las R repeticiones (más `score_std` y
`score_r{i}`/`fold_r{i}` cuando R > 1). El motivo es `lead_time_curve()`: agrupa por
vehículo y busca K cortes seguidos sobre el umbral, así que R filas por corte harían
pasar un pico aislado por alerta sostenida e inflarían la anticipación —justo el
número del pitch—. `_vehicle_sequences()` ahora falla si le llegan cortes
duplicados. El **PR-AUC de selección es el promedio de los PR-AUC de cada
repetición**, no el PR-AUC de los scores promediados: eso último es un ensamble
encubierto y da mejor de lo que el modelo es.

**Cómo se reproduce:**

```bash
python scripts/check_setup.py                                      # 28 chequeos
python scripts/make_splits.py --config configs/data/panel_v1.yaml --dry-run
```

El segundo rearma los folds sobre el panel que ya existe (sin releer los crudos) y
dice si el reparto cambia respecto del `splits.json` congelado. Al adoptar esto
informó `IDÉNTICO`.

---

## 2026-09-18 · Panel v1: W=1000, G=500, H=3000, Δ=500; sanos emparejados por odómetro **y mes**

**Decidió:** Santino (implementación de F2). **Detalle y evidencia:**
[f2-eda-revision-y-features.md](f2-eda-revision-y-features.md).

**Terna.** Con Δ = 500, W = 1000 y G = 500 conservan 54 de los 60 eventos de dev con al
menos un corte positivo; cada 500 km más de W cuesta ~3 eventos, y el barrido
W ∈ {1000, 2000, 3000} da la misma métrica dentro del ruido. H = 3000 pone los positivos
en 500–3.500 km antes del evento, que es donde el perfil alineado muestra que la señal
existe. Los sanos solo aportan cortes **verificables** (`corte + G + H` dentro de lo
observado, `censored_policy: drop`); los cortes de un vehículo con evento llegan hasta
E − G y nada posterior al evento entra a ninguna ventana (después de la fecha el uso
cambia en el 80% de los vehículos).

**Emparejado.** Los 60 eventos caen entre sep-2025 y mar-2026 y la exposición sana se
concentra en 2026: emparejar los cortes de los sanos solo por odómetro deja el
calendario como atajo (agregar tres columnas de calendario sube el ROC de 0,67 a 0,76;
con emparejado por mes, de 0,57 a 0,60). Se conserva el mayor subconjunto de filas sanas
que reproduce la distribución conjunta (4.000 km × mes) de los positivos de dev, con 25%
de déficit: 1.241 filas de 7.308, 144 vehículos de 243. Celdas más finas dejan 682 filas.
El test no se mira ni para esto (la referencia son los positivos de dev).

**Qué se resigna:** volumen de negativos y 34 vehículos de test que quedan sin filas.
**Alternativa disponible sin tocar código:** `configs/data/panel_v1_unmatched.yaml`
(todo), `panel_w2000.yaml`, `panel_w3000.yaml`.

---

## 2026-09-18 · `Regenerations` y `DistanceBetweenRegenerations` fuera del modelo; la familia B se cuenta desde `AirRegeneration`

**Decidió:** Santino. El marcador `Regenerations` de `signals` se corta el 25-05-2026
para toda la flota (0 marcadores en jun–sep 2026 contra ~2.000 caídas de nivel por mes
en `trips`). Una tasa de marcadores por km mide qué fracción del historial cae antes de
esa fecha, y eso lo fija `ProductionDay` (ρ = −0,515). Era la "feature más fuerte del
EDA" (ρ = 0,317 con la etiqueta); contada desde las caídas > 5 puntos de
`AirRegeneration` dentro del viaje —que se registran hasta el final— da ρ = 0,072. Se
queda en el panel como `aux_regen_marker_per_1000km` solo para auditar. Retira la
recomendación §3 de `docs/f2-feature-engineering-candidatas.md`.

---

## 2026-09-18 · Fuera del set base: `ProductionDay` (aplica la pendiente del 17-09), `ModelSeries`, `daysUntilSale`, temperatura ambiente

**Decidió:** Santino. Todas quedan en el panel como `aux_*` (convención nueva: columna
que está y no entra al modelo), así las ablaciones no requieren reconstruir.

- `ProductionDay`: ρ = −0,435, tasa por quintil 0,46 → 0,00 (ventana de observación).
- `ModelSeries`: el cruce con `Engine` es casi diagonal; MODEL_3 0 eventos de 24,
  MODEL_4 2 de 91. Sacar `Engine` y dejarlo renombra el sesgo.
- `daysUntilSale`: la fecha de venta (`ProductionDay + daysUntilSale`) da ρ = −0,48 y
  tasa por quintil 0,53 → 0,00; sola, ρ = −0,41 con el odómetro final. Es exposición.
- `AirTemperature*`: estacional (16,7 → 25,0 °C de mediana mensual) con los eventos
  concentrados en siete meses: mide el calendario.

La única `static_` del set base es `SalesCountry_cd` (dos mercados que miden con
distinto nivel; el modelo necesita verlo).

---

## 2026-09-18 · Idle (0 km) separado de los viajes; velocidad recalculada; topes físicos, no cuantiles

**Decidió:** Santino. El 35% de las filas de `trips` tiene 0 km (motor encendido sin
moverse); `KilometerPerHour` es nulo exactamente ahí y, cuando existe, es
`trip_km / duración` (Pearson 0,99999). Los idle son la señal que más anticipa
(P(evento > sano) 0,58 → 0,74 al acercarse el evento) y contaminaban cualquier
"fracción de viajes cortos". Toda fracción de viaje se calcula entre los que se
mueven; los idle tienen su propia feature. Los recortes son a topes físicos declarados
en `features_v1.yaml` (200 km/h, [−40, 60] °C ambiente, 24 h por viaje) y no a
cuantiles, porque un cuantil dependería del train de cada fold.

---

## 2026-09-17 · El universo del estudio son 364 vehículos: fecha de evento utilizable + mercados que la registran

**Decidió:** Santino · **Fase:** F2 (previo). **Cierra la entrada de abajo**, que
quedaba abierta desde el 16.

De los 1081 vehículos entran al estudio **364** (290 dev / 74 test, 80 eventos). Se
descartan 717 por dos criterios que se componen, declarados en `universe` de
`configs/data/test_split.yaml`:

1. **`require_usable_event_date`** — un positivo entra solo si
   `IdentificationDate > daysUntilSale`. Saca 284.
2. **`keep_markets: [CNTRY_3, CNTRY_4]`** — solo los mercados donde el evento se
   puede observar. Saca 433 más, casi todos sanos.

**Por qué el criterio 1:** para 284 de los 365 positivos `IdentificationDate` es
*exactamente* `daysUntilSale`, y ahí el evento cae con el odómetro en ~13 km y el
2,3% del historial por delante: no hay ventana W que agregar ni gap G que blanquear,
así que la fila no se puede etiquetar. Que sea una convención administrativa y no
física lo cierra un conteo que no admite lectura alternativa: **`IdentificationDate`
nunca es menor que `daysUntilSale`, 0 casos de 365**. Si la fecha marcara la falla
real, alguna caería antes de la venta.

**Por qué el criterio 2, que no estaba en el plan:** la convención resultó ser **del
mercado, no del vehículo**. En CNTRY_1 y CNTRY_5 ningún evento tiene fecha
utilizable; en CNTRY_2, uno de 105. Tirando solo los positivos, esos tres mercados
quedaban aportando **432 negativos y 1 positivo** —el 54% de dev— y eso no es un
desbalance: es **selección sobre el resultado**. Se descarta un vehículo según lo
que le pasó, y los negativos dejan de venir de la misma población que los
positivos. Excluir `SalesCountry_cd` del set de features no lo arregla: el sesgo
está en quién entró a la muestra. La regla que sí cierra es simétrica: **se conserva
el mercado donde el evento se puede observar**. Un sano de CNTRY_4 es un negativo
legítimo porque si hubiera fallado lo habríamos visto; uno de CNTRY_1 no.

**Qué se eligió y qué se resignó.** De las tres salidas que dejaba abierta la
entrada de abajo se tomó la (b), quedarse con los datos confiables, porque el
objetivo es predecir la falla por **uso del cliente** y un evento pegado al día de
la venta no habla de uso. Se resigna volumen: 365 eventos pasan a 80, y el test de
73 a 20. El error estándar de la curva de anticipación se ensancha de ±11 pp a
~±20 pp, que es del orden de la diferencia entre modelos que esperamos medir. **Se
acepta a conciencia:** 73 eventos mal ubicados en el tiempo no permiten medir
anticipación, y la anticipación es el pedido central del desafío. Un intervalo ancho
sobre la métrica que se pidió es mejor que uno angosto sobre otra cosa.

**Cómo se aplicó, que es la parte que puede salir mal.** `make_test_split.py` corre
en **dos etapas**: primero sortea los 1081 con la semilla de siempre, después
recorta el resultado conservando el lado de cada sobreviviente. El recorte **no
re-sortea**. Verificado: la huella del reparto padre (`ba9aa4d290bc6610`) coincide
con la del holdout congelado el 16, ningún vehículo cambió de lado y las dos listas
nuevas son subconjuntos de las viejas.

**Por qué no re-sortear, si el balance queda peor.** Re-sortear el universo
recortado sería un dado nuevo tirado *después* de haber mirado los datos que
motivaron el recorte —exactamente lo que el holdout existe para impedir—. Filtrar es
determinista y no depende de ninguna métrica. El costo es que las tasas quedan
desparejas (0,207 dev contra 0,270 test) porque la estratificación se hizo sobre la
población vieja; el desvío juega a favor del número de portada, porque deja 20
eventos en test en vez de los ~16 que habría dado un sorteo nuevo.

**Alternativa descartada:** quedarse con los 797 (solo el criterio 1). Más
vehículos, pero con el 54% de dev viniendo de mercados sin eventos observables. El
número sería más grande y mediría menos.

**Qué cambia río abajo:**

- **El panel se construye con los 364, no con los 1081.** El contrato de
  `CLAUDE.md` decía lo contrario y quedó actualizado: `test_split_masks()` ahora
  falla si el panel trae un vehículo excluido.
- **`test_split.json` guarda tres listas** (`dev`, `test`, `excluded`) más el
  informe del universo y la metadata del sorteo padre.
- **`dev_mask` dejó de ser el complemento de `test_mask`**: con universo recortado,
  "no es test" incluye a los excluidos.

**Qué queda pendiente y no lo resuelve esta decisión:** preguntarle a Ford qué es
`IdentificationDate` para los 284 descartados. Si la respuesta permite recuperarlos,
el universo se amplía cambiando dos claves del YAML y regenerando: no hay código que
tocar.

**Detalle:** [f2-universo-fecha-usable.md](f2-universo-fecha-usable.md)

---

## 2026-09-17 · `static_ProductionDay` fuera del set base — PENDIENTE DE APLICAR

**Registró:** Santino · **Fase:** F2 (previo) · **Estado: recomendación del EDA,
todavía no aplicada en `panel_v1.yaml`.**

Medido sobre los 290 de dev del universo recortado, `static_ProductionDay` es **la
variable más correlacionada con la etiqueta de todo el EDA**: ρ de Spearman
**−0,435**, por encima de cualquier feature física. La tasa de eventos por quintil
va de 0,459 a **0,000** —el quintil de los producidos más tarde tiene 56 vehículos y
ni un solo evento— y contra `span_days` da **ρ = −0,933**.

**Por qué:** la ventana de observación termina el mismo día para toda la flota, así
que un vehículo producido tarde tuvo menos kilómetros de exposición para llegar a
fallar. Es el mismo caso de `ENG_3`: el modelo puede declarar sanos a 56 vehículos
por su fecha de producción y acertar siempre, dentro de este dataset y en ninguno
futuro.

**Qué se recomienda, con tres niveles distintos de urgencia:**

1. **`static_ProductionDay` a `features.static_excluded`.** Mismo criterio que
   `Engine`.
2. **`ModelSeries` hay que decidirlo explícitamente.** El EDA encontró algo que
   antes no se veía: el cruce `Engine × ModelSeries` es **casi diagonal**
   (ENG_1 con MODEL_4, ENG_2 con MODEL_1 y MODEL_2, ENG_3 con MODEL_3 y MODEL_4), y
   en el universo recortado `MODEL_3` tiene 0 eventos de 24 y `MODEL_4` 2 de 91.
   **Sacar `Engine` y dejar `ModelSeries` no saca el sesgo: lo renombra.** Esto
   contradice la decisión del 15-09, que declaraba a `ModelSeries` limpio; era
   cierto sobre los 1081 y dejó de serlo sobre los 364.
3. **`static_daysUntilSale` se audita, no se saca de oficio.** Ya no es la etiqueta
   disfrazada —ese caso se fue con el universo— y queda en ρ = −0,261. Puede tener
   contenido real: un auto que tarda en venderse pasa meses parado. Ablación
   explícita en otro YAML.

**Por qué queda pendiente y no aplicada:** tocar `features.static_columns` es tocar
el contrato del panel, y el panel todavía no existe. Se aplica en el mismo PR que
construya `build_dataset.py`, para que el cambio y su efecto se vean juntos.

**Detalle:** `notebooks/eda-exhaustivo-dev.ipynb` §2 y §7.3

---

## 2026-09-16 · F2 no define W, G, H ni Δ hasta resolver la fecha del evento — CERRADA el 17-09

**Registró:** Santino · **Fase:** F2 (previo) · **Estado: cerrada** por la entrada
de arriba, que tomó la salida (2) —quedarse con los datos confiables— y le agregó el
criterio de mercado que este registro todavía no había visto. Se deja tal cual
porque el razonamiento de por qué quedó abierta explica por qué se eligió lo que se
eligió.

En dev, `IdentificationDate == daysUntilSale` **exactamente** en 231 de los 292
vehículos con evento (79,1%), y nunca es menor. Bajo la lectura literal del anexo
7.3 —días desde producción—, esos 231 eventos caen con una mediana de **13 km** de
odómetro y el **2,3%** de su historial de viajes por delante: sin historial previo
no hay ventana W que agregar ni gap G que blanquear, así que la fila no se puede
etiquetar.

**Por qué queda abierta y no se resuelve acá:** las tres salidas posibles tienen
costos distintos y ninguna es técnica:

1. preguntarle a Ford qué es `IdentificationDate` para esos 231 vehículos;
2. etiquetar solo los 61 con fecha discriminante — quedan ~20% de los positivos, y
   la métrica del pitch pierde potencia;
3. pasar a un objetivo a nivel vehículo, resignando la curva de anticipación.

Elegir la lectura alternativa (`IdentificationDate` contado desde la venta) porque
los números quedan mejor **no** es una salida: el anexo dice "días desde
producción" sin ambigüedad, y es exactamente la clase de decisión contra la que
existe el holdout. Las dos lecturas están materializadas en el cache del EDA
(`event_odo_km` y `event_alt_odo_km`) para que la comparación se haga con datos.

**Segundo pendiente del mismo hallazgo:** `static_daysUntilSale` está en
`features.static_columns` de `configs/data/panel_v1.yaml` y, para el 79% de los
positivos, es numéricamente igual a la columna que define la etiqueta (ρ de
Spearman −0,249 con `event_observed`). Hay que auditarla con el mismo criterio con
el que se sacó `Engine`. Mientras tanto, cualquier PR-AUC alto que salga con esa
columna adentro cae de lleno en la regla 6.

**Lo que este bloqueante NO toca:** el anclaje temporal de F1 sigue en pie (IQR de
0 días re-medido en dev), la etiqueta a nivel vehículo sigue siendo la cohorte, y
el holdout dev/test no cambia.

**Detalle:** [f2-identificationdate-igual-a-venta.md](f2-identificationdate-igual-a-venta.md)

---

## 2026-09-16 · Las features de elevación y presión de neumáticos salen del alcance de F2

**Decidió:** Santino · **Fase:** F2 (previo)

Cuatro de las 31 features reservadas en `scripts/make_dummy.py::FEATURE_SPECS` se
declaran **no construibles** y salen del alcance de F2:
`feat_elevation_mean_m` y `feat_elevation_range_m` (familia C),
`feat_tire_pressure_mean` y `feat_tire_pressure_below_thr_frac` (familia D).

**Por qué:** las columnas no existen. El anexo 7.1 del PDF declara 40 columnas para
`TripSummary` y el CSV entregado trae 25; entre las 19 ausentes están las 8 de
presión de neumáticos, las 2 de elevación y las 4 de GPS. No es un problema de
nombres —el PDF avisa que las variables vienen renombradas y ese caso existe, ver
la decisión de abajo—: acá no hay ninguna columna con ese contenido. Y sin GPS
tampoco se puede inferir altitud, así que no hay sustituto posible.

**Qué se pierde:** el ángulo de altitud, que el plan §4 señalaba como "ligado
directamente al ingreso de oxígeno que menciona el título del desafío". Vale
decirlo en el informe en vez de que se note por omisión.

**Qué NO se hace:** no se edita `configs/data/raw_sources.yaml`. Ese archivo ya
declara el esquema real (F1 lo auditó contra los archivos); el desvío es contra el
diccionario oficial, que no es un config del repo.

**Pendiente:** sacar esos cuatro nombres de `FEATURE_SPECS` cuando se toque el
panel dummy, o dejarlos con un comentario que diga que no se materializan.

**Detalle:** [f2-diccionario-trips-incompleto.md](f2-diccionario-trips-incompleto.md)

---

## 2026-09-16 · La familia B se construye desde `AirRegenerationStart/End`

**Decidió:** Santino · **Fase:** F2 (previo)

Las features de DPF del plan §4 (`feat_dpf_end_slope_per_1000km`,
`feat_dpf_end_mean`, `feat_dpf_end_max`, `feat_dpf_positive_delta_frac`) se
materializan desde `trips.AirRegenerationStart/End`, **no** desde
`signals.Acumulation`.

**Por qué:** son la misma variable. Alineando el fin de cada viaje con la señal más
cercana dentro de 30 minutos, `AirRegenerationEnd == Acumulation` en el **99,80%**
de 16.324 pares (Pearson 0,9995), y `AirFilterEnd == Message` en el 99,98% con el
vocabulario de 9 niveles idéntico. Se elige `trips` porque tiene el odómetro
prácticamente completo, mientras que `OdometerValue` de `signals` es 11% nulo, y
porque el grano de viaje es mejor para cortar ventanas.

Corolario que conviene no olvidar: **no son dos features**. `acumulation_mean` y
`air_regen_end_mean` correlacionan a ρ = 0,996 sobre dev. Una fuente, no las dos.

**Lo que esto corrige:** `f1-datos-reales.md` dice "no hay columna de DPF". Hay
dos, con otro nombre y en las dos tablas. La familia B pasa de parcial a completa.

**Lo que queda como hipótesis, no como decisión:** que esas columnas *sean* el
`DieselParticulateFilterStart/End` del anexo 7.1. Es la explicación que mejor
encaja —ocupan esa posición faltante, el anexo describe `Acumulation` como
"acumulación en el filtro de aire [%]", el PDF avisa que las variables vienen
renombradas, y la serie carga y se descarga como un filtro de partículas— pero no
hay confirmación de Ford, y la escala llega a 95 y no a 100. En el informe se
nombra como "nivel de acumulación del filtro (0–95)", que es siempre correcto, y no
como "% de saturación del DPF", que hay que poder defender.

**Detalle:** [f2-columnas-airfilter-airregeneration.md](f2-columnas-airfilter-airregeneration.md)

---

## 2026-09-16 · Test del 20% congelado antes de F2, no después

**Decidió:** Santino · **Fase:** F2 (previo)

Los 1081 vehículos se parten en **dev (864) y test (217)** antes de construir el
panel. El test se guarda en `data/processed/test_split.json` y no se toca hasta
tener el modelo elegido. La CV de 5 folds de `make_splits()` corre **dentro de
dev**.

El panel de F2 se materializa con **los 1081**, y el recorte es por máscara
(`test_split_masks`), no por un panel más chico: así la evaluación final no obliga
a regenerar el panel con otros vehículos —que es cuando aparecen las diferencias
silenciosas entre lo que se entrenó y lo que se midió— y la protección la da el
código, que es auditable, en vez de la ausencia del archivo, que no se puede
verificar.

**Por qué ahora:** todo lo que viene después de este punto —qué features agregar
en la ventana, qué W, G y H, dónde poner el umbral de operación— son decisiones
que se toman mirando los datos. Si el test se recorta al final, esas decisiones ya
se tomaron con el test adentro, y el número del pitch queda optimista sin que
nadie pueda decir por cuánto. La auditoría de esquema y calidad (nulos, tipos,
duplicados, cobertura del join) sí se hizo sobre el dataset completo: no depende de
qué vehículo caiga de cada lado, así que no contamina nada.

**Por qué 20% y no 10%:** la métrica de portada se estima sobre los vehículos con
evento del test. Con 73 (20%), una tasa de detección de 0,7 tiene un error
estándar de 0,054 —±11 pp de intervalo—; con 36 (10%) pasa a ±15 pp, que es más
ancho que la diferencia entre modelos que esperamos medir. Del otro lado, dev
conserva 292 vehículos con evento: ~58 por fold, suficiente para que ningún fold
quede sin positivos.

**Por qué no estratificar por motor, modelo y país:** la estratificación es solo
por `event_observed`. Se reporta cómo quedaron repartidas las tres estáticas
(peor desvío: `MODEL_3`, 4,2 pp) pero no se fuerza: con 1081 vehículos, cada
restricción extra gasta grados de libertad para arreglar algo que ya quedó
razonable. Y elegir la semilla hasta que el cuadro quede lindo es la misma trampa
que el holdout viene a evitar.

**Alternativa descartada:** vivir solo con la CV. Se reusa decenas de veces
durante la selección de modelo, y para la décima comparación la métrica
out-of-fold ya está sobreajustada al procedimiento. Sirve para elegir; no sirve
para reportar.

**Cómo se aplica:** `configs/data/test_split.yaml` (semilla y `test_size`) +
`src/eval/splits.py::make_test_split` / `test_split_masks`. Se congela una vez con
`python scripts/make_test_split.py`; el script no pisa un holdout ya congelado sin
`--force`.

El recorte a dev lo hace **`scripts/train.py` solo**, en `select_dev()`, antes de
armar los folds: ninguna corrida ve el test sin que alguien lo pida explícitamente.
Y la clave `splits.test_split` es **obligatoria en todo YAML de experimento** —con
el path del holdout, o `null` explícito para el panel dummy—: si falta, `train.py`
levanta un `KeyError` y no entrena.

**Por qué obligatoria y no con un default:** olvidarse del holdout no rompe nada.
No hay excepción, ni warning que alguien vaya a leer en 300 líneas de log: sale un
PR-AUC más alto y la corrida entra a la tabla comparativa como si fuera buena. El
único momento en que ese error se puede detectar es antes de correr, así que el
config tiene que declarar la intención aunque sea para decir "este panel no tiene
holdout".

**Detalle:** [f2-union-y-holdout-dev-test.md](f2-union-y-holdout-dev-test.md)

---

## 2026-09-15 · Los 13 clones se colapsan a 13 vehículos

**Decidió:** Gonzalo · **Fase:** F1

Los 26 `VehicleCode` duplicados se colapsan a 13 `vehicle_id`, quedándose con el
código menor de cada par. La etiqueta se resuelve a `failed`.

**Por qué:** son el mismo auto con dos códigos y sus viajes son idénticos
(Jaccard 1.000). Con los dos códigos vivos, uno puede caer en train y el otro en
validación sin que `src/eval/splits.py` note nada: la regla 2 se cumple en el
código y se viola en los hechos. La etiqueta va a `failed` porque `not_failed` es
ausencia de registro, no evidencia de que el vehículo esté sano.

**Alternativa descartada:** tirar los 26 códigos. Son 13 vehículos con evento de
365 — el 3,6% de la señal positiva. Caro para un problema que se arregla mapeando.

**Cómo se aplica:** `configs/data/vehicle_dedupe.yaml` (lista congelada) +
`src/data/dedupe.py`. Se corre una vez al construir el panel, antes de cualquier split.

**Detalle:** [f1-clones-vehiculos.md](f1-clones-vehiculos.md)

---

## 2026-09-15 · `Engine` queda afuera del set base de features

**Decidió:** Gonzalo · **Fase:** F1

`static_Engine` no entra al modelo base. Queda declarado en
`features.static_excluded` de `configs/data/panel_v1.yaml`.

**Por qué:** `ENG_3` es el 36,1% de los vehículos sanos y el 0% de los que tienen
evento. El modelo puede clasificar 268 vehículos como sanos por su motor y acertar
siempre — dentro de este dataset. Es cómo se muestrearon los datos, no física del
motor, y no sobrevive a la primera pregunta del jurado.

**Qué queda abierto:** medirlo como ablación explícita (otro YAML) si alguien
quiere cuantificar el aporte real. `ModelSeries` y `SalesCountry_cd` sí entran:
están en las dos cohortes con proporciones comparables.

**Detalle:** [f1-sesgo-eng3.md](f1-sesgo-eng3.md)

---

## 2026-09-15 · Las cohortes se declaran como `parts`, no como tablas separadas

**Decidió:** decisión de implementación tomada al reescribir el contrato de
fuentes; queda a revisión en el PR de F1.

Los seis CSV se declaran como tres tablas lógicas con dos `parts` cada una. El
loader concatena y agrega la columna `cohort`.

**Por qué:** la cohorte es la etiqueta, no una dimensión de los datos. Si río abajo
hay seis tablas, cada consumidor tiene que acordarse de unirlas y de que `cohort`
no es una feature. Con `parts`, el contrato de datos sigue teniendo tres tablas y
un solo lugar donde se decide qué significa la cohorte.
