# Preregistro: incidencia aprendida con los fallados sin fecha (F5 §3.2)

**Fecha:** 2026-09-22 · **Fase:** F5 · **Rama:** `exp/incidencia-externa`.

**Se escribió** después de los conteos de la Fase 1 y antes de todo lo demás:
- antes de calcular ninguna feature del conjunto externo contra su etiqueta;
- antes de ajustar el modelo;
- antes de aplicarlo a dev.

El commit que agrega este archivo es la marca de tiempo. Cualquier cambio posterior va en un
commit propio, anterior a la corrida que afecta, y dice por qué.

**De dónde sale:** [../f5-modelos-candidatos-bibliografia.md](../f5-modelos-candidatos-bibliografia.md)
§3.2 y §5 (lugar 1). Lo que dejó el cure model ([f3-cure-model.md](f3-cure-model.md)) es que
incidencia y latencia se separan, y que la incidencia no necesita la fecha del evento.

**Motivo de la lista cerrada:** el presupuesto de comparaciones está agotado (CLAUDE.md). Esto
no es otro learner sobre el panel v1: cambia **con qué datos** se aprende *qué auto*. Igual va
con lista cerrada. Una idea que aparezca en el camino va a "Qué queda abierto" de la ficha del
resultado, no a una corrida.

## 0 · Lo que ya se vio antes de escribir esto

**Dev:** nada nuevo. Todo lo de dev que se usa acá ya está medido en el cure model (P0, los
pisos C2 y C3) y en las fichas de F3.

**Test:** nada. Tampoco se tocan los 143 excluidos que el sorteo padre puso del lado test:
- no se leyeron sus viajes;
- no entran a ningún conteo.

**Conjunto externo, Fase 1 (solo conteos):**
`python scripts/build_external_panel.py --config configs/data/panel_external_incidence.yaml --counts-only`
- **Sorteo padre reproducido.** `make_test_split` sobre los 1081, con semilla 42, da la huella
  congelada `ba9aa4d290bc6610`. Del lado dev hay 574 excluidos, con 232 fallados.
- **Quedan 541 vehículos**, de CNTRY_1, CNTRY_2 y CNTRY_5. Se van los 33 fallados con fecha por
  defecto de CNTRY_3 y CNTRY_4: sus sanos están en dev, así que dentro de su mercado no tienen
  contra quién compararse (§1).
- **El puente de calendario vale en el conjunto externo:** origen 20107, igual al congelado
  sobre dev, con IQR 0 d sobre 541 vehículos.
- **No hay clones escapados.** Ningún vehículo externo comparte más del 0,1% de sus viajes con
  uno del universo (dev + test).

Embudo al hito de 30 días. **Fila** = llega a venta + 30 con telemetría y tiene ≥ 15 viajes en
(venta, venta + 30]. **Elegible** = su exposición potencial en la ventana del registro es ≥ E
días (§1).

| mercado | cohorte | vehículos | fila | quietos (< 100 km) | elegibles E90 | E60 | E120 |
|---|---|---|---|---|---|---|---|
| CNTRY_1 | fallado | 99 | 90 | 3 | 61 | 67 | 55 |
| CNTRY_1 | sano | 156 | 143 | 2 | 73 | 87 | 60 |
| CNTRY_2 | fallado | 88 | 67 | 17 | 49 | 54 | 37 |
| CNTRY_2 | sano | 156 | 139 | 17 | 79 | 86 | 62 |
| CNTRY_5 | fallado | 12 | 11 | 0 | 4 | 6 | 3 |
| CNTRY_5 | sano | 30 | 29 | 3 | 17 | 18 | 15 |
| **total** | fallado | 199 | 168 | 20 | **114** | 127 | 95 |
| **total** | sano | 342 | 311 | 22 | **169** | 191 | 137 |

**Fecha de venta de los que tienen fila:**
- **Fallados:** ninguno se vendió después del 11-03-2026. El máximo es el 25-02-2026, y el p90
  cae entre dic-2025 y ene-2026.
- **Sanos:** 50 se vendieron después del 11-03-2026.

Es lo que se espera si la ventana del registro de estos mercados termina, como en dev, antes
de marzo de 2026. Un fallado no puede registrarse antes de su venta. **No prueba** que la
ventana sea la misma: es consistente con eso.

**Dos relaciones con la etiqueta que el embudo deja ver**, y que se declaran:
- en CNTRY_2, los fallados tienen más autos quietos (17 de 67) y más vehículos con < 15 viajes
  en 30 días (20 de 87) que los sanos (17 de 139, 14 de 153);
- en CNTRY_1 y CNTRY_5 no se ve esa diferencia.

Es la misma dirección que en dev: menos uso, más riesgo (cure model, C3).

**Lo que cambia respecto del doc F5.** Allá se esperaban "~290 eventos" y "una decena de
parámetros". Con la elegibilidad simétrica (§1) quedan **114 eventos entre 283 vehículos**.
- Es 2,6 veces los 43 eventos de dev al hito de 30 días, no 4–5.
- Con C ≈ 0,65–0,68 y prevalencia 0,40, el criterio (i) de Riley et al. (2019), que pide
  encogimiento ≥ 0,9, admite **~2–3 parámetros**. Simulado: C 0,634 → 1,8; C 0,679 → 3,1.
- La "decena" del doc era para ~290 eventos y un R² fijo. **Por eso el modelo primario tiene
  dos pendientes (§3).**

## 1 · El conjunto externo (la fuente)

- **Quién entra.** Los excluidos del universo que el sorteo padre puso del lado dev, de los
  mercados CNTRY_1, CNTRY_2 y CNTRY_5 (`src/data/external.py`). En esos mercados hay fallados y
  sanos.
- **El mercado es un estrato.** Las proporciones muestreadas de fallados y sanos difieren entre
  mercados, así que el mercado no entra como feature. Se descarta un mercado con solo fallados.
- **Positivo:** `event_observed = 1`. Los 114 elegibles tienen fecha por defecto; el único
  fallado de la fuente con fecha real (CNTRY_2) no es elegible. La fecha no se usa.
- **Negativo:** `event_observed = 0`.
- **Elegibilidad simétrica** (`aux_eligible`).
  - La exposición potencial en la ventana del registro se calcula **desde la fecha de venta**:
    `fin de la ventana − max(venta + 30 + G, inicio de la ventana)`, con G = 30 d. Tiene que
    ser ≥ **E = 90 d**.
  - Es la misma cuenta que el negativo resuelto del cure model, aplicada también a los
    positivos, para que la muestra no dependa del desenlace a través del calendario.
  - Sin eso, los positivos llegarían hasta ventas de enero de 2026 y los negativos resueltos
    se cortarían en octubre de 2025. Con features que tienen estación, eso sería un atajo de
    calendario armado a mano.
  - **Supuesto declarado:** la ventana del registro (01-09-2025 → 11-03-2026) es la misma en
    CNTRY_1/2/5. Es la pregunta 2 para Ford. Lo único que la respalda es la fecha de venta de
    los fallados (§0).
- **La fecha por defecto no se lee como fecha.** `event_dss` queda en NaN, así que ningún
  fallado se descarta por "evento antes del gap".
  - **Trampa 2 del doc F5, aceptada:** en los mercados con fecha, 3 de 60 eventos caen antes
    de 30 + G días. Con la ventana de 30 días, un ~5% de los positivos puede tener telemetría
    del gap o posterior al evento.
- **Clones:** se colapsan con `src/data/dedupe.py` antes de filtrar, y se verificó que no hay
  fugas (§0).

## 2 · Features: fijas, las del cure model más el uso

**Ventana:** los viajes que arrancan después de la venta y terminan a más tardar en venta + 30
d (`feature_window_days: 30`).

- **Las cuatro del índice físico**, con las mismas definiciones, pesos de celda y mínimos que
  el cure model (preregistro del cure, §3):
  - `idle_per_1000km` (+);
  - `trips_below_regime_temp_frac` (+);
  - `speed_kmh_mean` (−);
  - `coolant_temp_end_mean` (−).
- **El uso:** `feat_log1p_km_per_day` = log(1 + km en la ventana / 30).
- **Normalización contra la flota** (`FleetReferenceNormalizer`, ajustado **solo con la
  fuente**). La referencia es la mediana, **por mes**, de los sanos de la fuente con los tres
  mercados juntos (`levels: [[month], []]`): con menos de 10 vehículos se cae a la mediana
  global.
  - **Por qué por mes y no por mercado × mes:** dev (CNTRY_3/4) no está en la fuente. Aplicarle
    una referencia de su mercado exigiría los sanos de dev, que es ajustar sobre dev.
  - Con la referencia por mes, la fuente y dev se construyen igual. El mercado de la fuente lo
    absorbe el estrato del modelo.
  - Lo que se pierde, el desvío dentro del mercado de dev, se mide aparte (§6, I2).
- **Imputación y escala:** mediana y estandarización, ajustadas con las filas elegibles de la
  fuente.
- **Fuera, a propósito:** `ProductionDay`, `daysUntilSale`, fecha de venta, mes, temperatura
  ambiente, `Engine` y `ModelSeries` (trampa 6). El mercado solo entra como estrato.

## 3 · El modelo primario (A-solo)

**Qué es.** Una logística con penalización de Firth, sobre las 283 filas elegibles de la fuente:

    logit P(falla) = α_mercado + a · s + b · z(log1p km/día)

- `s = z(idle) + z(bajo régimen) − z(velocidad) − z(refrigerante)` es el índice de pesos
  unitarios de P0, con los signos físicos fijos.
- Las z son de la fuente.
- **Dos pendientes (a, b)** y el intercepto por mercado (el estrato).
- Es lo que admite Riley (§0), y separa las dos cosas que el cure model no pudo separar: el
  rasgo térmico y la intensidad de uso.

**Por qué Firth:** es el mismo ajuste del cure model (P1), y el sesgo de muestra chica en
logística es el que corrige (Firth 1993; Heinze & Schemper 2002). Sin FLIC, porque el intercepto
no se usa.

**Cómo se aplica a dev, congelado.**
- Todo se ajustó con la fuente: la referencia de flota, la imputación, la escala, a y b.
- En dev no se ajusta **ningún** parámetro. No hay fold ni fuga posible.
- **Score de una fila de dev:** `a·s + b·z(log1p km/día)`, con las features de su ventana
  (venta, venta + 30]. Sin el intercepto del mercado, que dev no tiene.
- Es **el mismo en los tres hitos** de un vehículo: el rasgo de los primeros 30 días.
- Las filas de dev de los hitos 60 y 90 cuyo vehículo tiene < 15 viajes en sus primeros 30 días
  se puntúan igual, con lo que haya. Las celdas bajo el mínimo quedan en NaN y se imputan con
  la mediana de la fuente. Se cuentan.

## 4 · Compuertas en la fuente, antes de tocar dev

- **G0 · conteos:** hechos (§0). La corrida sigue con 114 eventos elegibles.
- **G1 · qué es "fallado" sin fecha** (trampa 1, descriptiva, no para).
  - Por mercado de la fuente, la razón fallados/sanos de los mensajes `Air Filter Full` y
    `Air Filter Overloaded` por 1.000 km, sobre toda la telemetría post-venta.
  - Al lado van los 33 fallados con fecha por defecto de CNTRY_3/4.
  - F1 midió 1,81× y 2,63× con todos los vehículos juntos.
  - Si en un mercado la razón es ≤ 1 en los dos mensajes, se marca en la ficha. No cambia la
    corrida.
- **G2 · el rasgo temprano apunta igual en la fuente** (trampa 1, con etiqueta, solo la
  fuente). **Esta sí para.**
  - AUC de `s` entre fallados y sanos elegibles de la fuente, con IC95 por bootstrap de
    vehículos (2.000).
  - **Si el límite inferior no supera 0,5, se para.** O el rasgo no se replica en tres
    mercados nuevos, o "fallado" ahí es otra cosa. En los dos casos no hay nada que transferir.
    Se documenta el negativo.
  - Se reportan además la AUC de cada feature con su signo, la de `s` por mercado y la de −km/día.

## 5 · Qué decide (A-solo sobre dev)

**Dónde se mide:** las mismas filas de dev del panel de hitos del cure model (727 filas, 268
vehículos), con `landmark_metrics`.
- El panel `panel_landmark_ps_w30.parquet` tiene las mismas filas que `panel_landmark_ps.parquet`,
  con las features de la ventana de 30 días.
- Los folds son `splits_landmark_ps_r3.json`, R = 3.
  - El score congelado es idéntico en las tres repeticiones, así que su desvío es 0, como el de
    los pisos.
  - La CV no ajusta nada. Existe para que las auditorías y el veredicto usen la misma cuenta y
    los mismos vehículos que P0.

**Deciden D1 y D2**, definidos como en el preregistro del cure, §5:
- D1 = C con entrada tardía, promedio de los hitos 30/60/90;
- D2 = detección con primera alerta, a ≤ 5% de negativos resueltos.

**"Le gana a X en M"** exige las dos condiciones del cure:
- la diferencia de medias supera √(σ² + σ_X²);
- el IC95 del bootstrap pareado por vehículo (2.000) excluye el 0.

**Adopción de A-solo.** La misma regla que el cure model aplicó a P0:
1. **A0** · nulo global: se permutan las features dentro de cada hito. D1 tiene que caer a
   0,5 ± 0,05 y D2 al presupuesto ± 0,05. Si no, no se lee nada más.
2. **C1** · nulo estratificado: se permuta el score dentro de hito × tercil de producción ×
   mercado (2.000 veces). Tiene que dar p < 0,05.
3. **C2** · le gana en D1 a **−ProductionDay y a −fecha de venta**.

**A3 no aplica:** no hay coeficientes ajustados en dev.

**Se reportan y no deciden la adopción:**
- **C3:** −km/L, con `beats`. Es el piso de uso;
- **C6:** sin autos quietos;
- **el veredicto contra P0** (`f3-cure-p0-unitweight`, mismas filas): gana / empata / pierde,
  con la regla del cure.

**Lectura declarada ahora:**
- si A-solo se adopta pero empata con C3, el rasgo aprendido afuera **es uso**, y así se
  escribe;
- si pierde contra P0, los pesos aprendidos afuera transfieren peor que los unitarios.

## 6 · Informativas (no deciden, lista cerrada)

- **I1 · la fuente por dentro.** Todo sale de la fuente y nada de dev:
  - a y b con IC por bootstrap de vehículos (1.000);
  - encogimiento heurístico (χ² del modelo − gl) / χ²;
  - AUC con CV interna (5 folds × 3, estratificada por mercado × etiqueta, todo el pipeline
    dentro del fold);
  - AUC de −`ProductionDay` y de −fecha de venta entre los elegibles de la fuente (¿hay atajo
    de calendario en la fuente?);
  - a y b con E = 60 y E = 120, y sin elegibilidad para los positivos.
- **I2 · normalización dentro del mercado de dev.** Los mismos a y b, aplicados con el pipeline
  de P0: referencia por mercado × mes con los sanos del train de cada fold, e imputación y
  escala por hito. Es la trampa 3 medida.
- **I3 · cinco pendientes.** Firth sobre las cinco z sueltas más el estrato: lo que el doc F5
  proponía. Se aplica a dev igual que A-solo, con D1/D2 y `beats` contra A-solo.
- **I4 · ventana de 60 días** (trampa 2).
  - El mismo modelo primario, entrenado con (venta, venta + 60] en la fuente: hito 60, misma
    elegibilidad.
  - Se aplica a dev con las features de 60 días en los hitos 60 y 90.
  - Se compara con A-solo **en esos dos hitos**.

## 7 · Paso 2 (condicional): A dentro de survival stacking

Corre **solo si A-solo se adopta** (§5). Si no, la covariable externa no tiene nada que aportar
(doc F5, §3.4).

- **Panel:** `panel_survival.parquet` del build `data/rebuild-0921` (el de la referencia), más
  `feat_ext_incidence`: el score congelado de A-solo, calculado con la ventana (venta, venta +
  30] del vehículo.
  - Vale NaN en los cortes con `cut_date` < venta + 30 d, y en los vehículos sin fecha de venta
    (regla 3).
  - Son las mismas filas del panel, así que valen los mismos folds.
- **Modelo:** el YAML de `f3-survival-stacking-r3-rebuild` sin cambios, más la columna.
- **Folds:** `splits_r3.json`, R = 3.
- **Decide** la regla de [f3-preregistro-landmark-ensamble.md](f3-preregistro-landmark-ensamble.md):
  - detección a ≤ 50 falsas alarmas / 1.000 sanos, más el lift por vehículo con `mean`;
  - "le gana" = diferencia mayor que el desvío combinado;
  - gana / empata / pierde contra la referencia, con la estabilidad como criterio en el empate;
  - tiene que ganarle también al piso posicional.
- **Auditorías:** (a0), (a′), (b), piso posicional y `audit_mil_bagsize.py`.

## 8 · Orden y paradas

1. Panel externo y panel de dev con la ventana de 30 días.
   - Verificación: el panel de dev tiene exactamente las filas y las `aux_` de
     `panel_landmark_ps.parquet`.
2. Tests de correctitud en verde (`check_setup.py`):
   - la huella del sorteo padre;
   - la ventana fija, que no mueve las filas;
   - el scorer congelado no ajusta nada en `fit`;
   - el normalizador por mes sin mercado.
3. **G1 y G2.** Si G2 no pasa, se para y se documenta.
4. Ajuste en la fuente (I1) y congelado del modelo.
5. **A-solo** sobre dev, con A0, C1, C2, C3, C6 y el veredicto contra P0 → adopción.
6. I2, I3, I4.
7. **Paso 2**, solo si A-solo se adoptó.
8. Veredicto escrito, gane o pierda: la ficha `f5-incidencia-externa.md` y `decisiones.md`.

## Lo que queda fuera a propósito

- La verosimilitud completa: censura por intervalo de los fallados sin fecha, EM de
  autoconsistencia (doc F5, "más adelante").
- Otros mercados, otra ventana, otro E primario u otro mínimo de viajes.
- Referencia por mercado × mes para la fuente en el primario.
- Otras features, otras definiciones de las cuatro, o km/día sin logaritmo.
- Más de dos pendientes en el primario, otras penalizaciones o learners (GBM, TabPFN).
- Ajustar nada en dev: recalibrar a y b, elegir entre A-solo e I2/I3/I4 según el resultado, o
  combinar con P0.
- Tocar los 143 excluidos del lado test del padre.
- Survival stacking en el reloj post-venta (doc F5 §3.3) y su combinación con esto (§3.4): son
  otro preregistro.

Todo eso es un barrido o un candidato nuevo.

## Preguntas para Ford que pueden invalidar esto

- **¿Qué significa `IdentificationDate == daysUntilSale`?** Si "fallado" en CNTRY_1/2/5 es otra
  cosa (una campaña, un reclamo de entrega), la transferencia mide eso. G1 y G2 lo acotan, no lo
  cierran.
- **¿La ventana del registro es la misma en todos los mercados?** De eso depende la
  elegibilidad de los sanos de la fuente.
- **El error de SQL de la query de eventos** (rama `feat/f3-features-regeneracion`). Si se
  corrige y cambia quién es fallado, esto se rehace primero.
