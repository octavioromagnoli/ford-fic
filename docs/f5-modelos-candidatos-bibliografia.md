# Qué podría superar a survival stacking · bibliografía y candidatos (F5)

**Fecha:** 2026-09-22 · **Fase:** F5 (modelos avanzados, plan §6) · **Estado: ideas, nada
implementado ni corrido.** Se escribió leyendo `docs/memoria/` y la bibliografía de abajo. No
hay ningún número nuevo sobre dev y el test no se tocó. Es el sucesor de
[f3-modelos-candidatos.md](f3-modelos-candidatos.md). Cuando una idea se implemente, va por
`/mlmodel`, con un preregistro propio, y su resultado a `docs/memoria/decisiones.md`.

**Objetivo:** detectar la mayor cantidad de autos que van a fallar, con la mayor anticipación
posible. Se mide igual que hasta ahora ([f3-piso-posicional.md](memoria/f3-piso-posicional.md)):
- **detección por vehículo a ≤ 50 falsas alarmas cada 1.000 sanos**, con R = 3;
- **lift por vehículo con `mean`**;
- **anticipación y su estabilidad**.

## 0 · En una línea

**El límite no es el learner.** Hay dos límites:
- **pocos eventos** para aprender *qué auto*: 60 en dev, y de ahí sale casi toda la señal;
- **reloj y conjunto en riesgo equivocados:** el panel v1 mide en km y cuenta como negativos
  horizontes que el registro no cubría.

La bibliografía coincide: en datos de baja dimensión, **ningún método le gana
significativamente a un Cox** (Burk et al., 2026). Cambiar LightGBM por otro aprendiz compra
poco. Lo que puede mover la detección, en orden de costo:

1. **Correr lo que ya está preregistrado y nunca corrió:** el ensamble E1/E2 con el CNN-LSTM
   (§3.1). No gasta presupuesto.
2. **Aprender *qué auto* con 4–5× más eventos** (§3.2). El universo dejó afuera **285 fallados
   cuyo evento no tiene fecha utilizable**; 232 están del lado dev del sorteo original. Lo que
   dejó el cure model es que **incidencia y latencia se separan**, y la incidencia no necesita
   la fecha.
3. **Rehacer survival stacking en el reloj del evento** (días desde la venta), con el conjunto
   en riesgo de la ventana del registro y un offset de exposición (§3.3). Es el mismo modelo
   con la cuenta arreglada, no un candidato nuevo.

Todo lo demás es condicional o capa de decisión (§3.5–3.7). Lo que no conviene está en el §4,
con la evidencia del repo al lado.

---

## 1 · De dónde se parte (lo que la memoria ya midió)

| hecho | número | dónde |
|---|---|---|
| finalista: survival stacking, panel v1, R = 3 | detección **15,7% ± 0,9** (8/9/8 de 53) · lift veh. **1,616×** · Brier 0,112 · anticipación 8.330 ± 22 km | [f3-mejor-modelo-a-la-fecha.md](memoria/f3-mejor-modelo-a-la-fecha.md) |
| piso posicional (odómetro crudo) | 7,55% · 1,018× | [f3-piso-posicional.md](memoria/f3-piso-posicional.md) |
| lo que sabe del *cuándo* | (a′) +0,016 ± 0,003: casi todo es *qué auto* | [decisiones.md](memoria/decisiones.md), 20-09 |
| calendario abierto | (b) **+0,049 de ROC** con R = 3 | decisiones.md, 21-09 |
| el evento sigue la **edad / días post-venta**, no los km | sd(log) 0,24 contra 0,97 · Poisson LR p = 0,014 a favor de días post-venta | [f3-reloj-y-ventana-del-evento.md](memoria/f3-reloj-y-ventana-del-evento.md) |
| **ventana del registro** 01-09-2025 → 11-03-2026 | el 28% de los horizontes sanos del panel v1 cae en parte fuera | ídem, §3 y §7 |
| **no hay trayectoria previa** | el exceso de idle/frío es máximo lejos del evento | ídem, §5 |
| **rasgo temprano** | viajes bajo régimen contra la flota: AUC 0,65–0,68 desde 30 d post-venta | ídem, §6 |
| cure model P0 / P1 | D1 0,599 / 0,584 · D2 4,2% · empata con −km/L (0,581) | [f3-cure-model.md](memoria/f3-cure-model.md) |
| en las filas que comparten, el finalista ordena mejor que el cure | D1 0,592 contra 0,543 (A6) | ídem |
| CNN-LSTM y survival stacking ordenan distinto | ρ de rango 0,24–0,36 por fila, 0,32–0,45 por vehículo | f3-piso-posicional.md |
| E1, E2 y L1 **preregistrados y sin correr** | commit 9b7bfb4 | [f3-preregistro-landmark-ensamble.md](memoria/f3-preregistro-landmark-ensamble.md) |
| **285 fallados fuera del universo** por la fecha, no por la etiqueta | *"No invalida la etiqueta a nivel vehículo"* | [f2-identificationdate-igual-a-venta.md](memoria/f2-identificationdate-igual-a-venta.md), [f2-universo-fecha-usable.md](memoria/f2-universo-fecha-usable.md) |

**Qué implica para buscar modelos:**
- **Anticipación.** Si no hay trayectoria, un modelo más fino del *cuándo* a partir de
  features no tiene de dónde aprender. La anticipación sale del reloj (días post-venta ×
  ventana) y de alertar temprano.
- **Detección.** La detección a falsas alarmas fijas la pone la discriminación **entre
  vehículos**. Hay dos formas de mejorarla: más eventos para aprenderla, o menos ruido de
  etiqueta.

## 2 · Qué dice la bibliografía sobre un problema con esta forma

Los datos de Ford tienen la forma de los **datos de campo o de garantía automotriz**:
- dos escalas de tiempo (edad y uso);
- tasa de uso muy heterogénea;
- eventos que solo se registran dentro de una ventana de calendario;
- una muestra armada por cohortes de resultado (failed / not_failed).

Esa literatura existe desde hace 35 años y trata esas cuatro cosas explícitamente.

| tema | qué aporta acá | fuentes |
|---|---|---|
| **Garantía automotriz, edad y uso** | modelos sobre la escala de edad con la **tasa de uso como covariable**, para poblaciones heterogéneas; motivados por confiabilidad de vehículos. Es el "−km/L empata" dicho como modelo | Lawless, Crowder & Lee 2009; Kalbfleisch, Lawless & Robinson 1991; revisión de Wu 2012 |
| **Elegir la escala de tiempo** | criterios para una escala "buena" cuando hay varias medidas de uso (el H1 del repo ya usa Kordonsky & Gertsbakh); modelos con dos escalas | Duchesne & Lawless 2000; Carollo et al. 2025 (`TwoTimeScales`) |
| **Escala + entrada tardía** | usar la escala del riesgo (edad, acá días post-venta) con **entrada tardía**; controlar calendario/cohorte **estratificando**, no como covariable | Korn, Graubard & Midthune 1997; Thiébaut & Bénichou 2004 |
| **Ventana de calendario** | "muestreo por intervalo": solo se ven los eventos dentro de dos fechas, así que los tiempos quedan doblemente truncados | Zhu & Wang 2012 |
| **Muestreo por cohortes de resultado** | case-cohort y nested case-control; el muestreo del conjunto en riesgo lleva a la logística condicional, que es la verosimilitud parcial de Cox; ajuste de la probabilidad a otra prevalencia | Prentice 1986; Langholz & Borgan 1995; Kvamme, Borgan & Scheel 2019 (Cox-CC); Saerens et al. 2002 |
| **Supervivencia con features que cambian + truncamiento, con ML** | reducir la supervivencia a regresión de Poisson con offset (PEM): admite truncamiento, features que cambian y eventos competitivos con cualquier learner | Bender, Rügamer, Scheipl & Bischl 2020; Bender et al. 2019 (exposición-rezago-respuesta) |
| **Aprendices de hazard que cambian con el tiempo** | boosting no paramétrico del hazard con truncamiento; bosques LTRC; BART en tiempo discreto | Pakbin et al. 2025 (BoXHED2.0); Yao et al. 2022 (LTRCforests); Sparapani et al. 2016 |
| **Benchmarks** | 34 datasets, 19 modelos: RSF oblicuo y boosting por verosimilitud tienen el mejor rango promedio, pero **ninguno le gana significativamente a Cox** | Burk et al. 2026; Jaeger et al. 2024 (aorsf) |
| **Mantenimiento predictivo automotriz** | RSF para baterías de camiones Scania (33.603 vehículos, 5 mercados); compresores de Volvo con registros de taller; desvío contra la flota (COSMO); SCANIA Component X | Frisk et al. 2014; Prytz et al. 2015; Rögnvaldsson et al. 2018; Kharazian et al. 2025 |
| **Lo que ganó en Component X** | GBM tabular sobre la **última observación** le ganó a Bi-LSTM y a GNN; **solo detecta lo inminente**, no los estados intermedios | Dimidov, Jafarnejad & Frank 2026 |
| **Predicción dinámica** | landmarking; landmarking dentro de un mixture cure model; Dynamic-DeepHit | van Houwelingen 2007; Cipriani, Alfò & Signorelli 2025; Lee, Yoon & van der Schaar 2019 |
| **Modelos fundacionales tabulares para supervivencia** | TabPFN-2.5/MITRA con la supervivencia como probabilidad de falla acumulada, y landmarking para las features que cambian; SurvPFN gana a baselines tuneados con n ≤ 100, pero acepta ≤ 1.000 filas, ≤ 10 features y solo estáticas | Kim, Lai & Zhang 2026; Böhm et al. 2026 |
| **Transfer entre poblaciones** | pedir prestados coeficientes y hazard base a una cohorte fuente, admitiendo heterogeneidad; también con censura por intervalo y con mezcla de Cox | Li, Shen & Ning 2023 (TransCox); CoxTL 2025; Xie 2024; Liu 2026 |
| **Eventos con tiempo desconocido** | EM de autoconsistencia para datos agrupados, censurados y truncados; mixture cure con censura por intervalo | Turnbull 1976; Liu, Szabo & Xiang 2026 |
| **Capa de decisión** | umbral con garantía de tasa de falsas alarmas (Neyman-Pearson); selección con control de FDR; optimizar el AUC parcial | Tong, Feng & Li 2018; Jin & Candès 2023; Yuan et al. 2023 (LibAUC) |
| **Tamaño muestral** | con ~40–60 eventos y C ≈ 0,65, entran 2–3 parámetros | Riley et al. 2019 (ya citado en el preregistro del cure) |

**Tres lecturas que atraviesan la tabla:**
1. **Los benchmarks neutrales dicen que el learner importa poco** en baja dimensión (Burk et al.
   2026). Cualquier idea que sea "otro learner sobre el mismo panel" tiene esperanza chica y
   consume presupuesto.
2. **La literatura de garantía automotriz modela explícitamente las cuatro cosas** que el repo
   descubrió a mano en F1–F3: escala de edad, uso como covariable, ventana de observación y
   muestreo por resultado. El panel v1 trata las cuatro de costado.
3. **En mantenimiento predictivo industrial, lo simple gana y lo lejano es difícil.** En
   Component X, el GBM sobre la última observación solo acierta lo inminente (Dimidov et al.
   2026). La anticipación de ~5 meses del finalista ya es rara en la literatura. Viene del rasgo
   de *qué auto*, no de una degradación visible.

---

## 3 · Candidatos, en orden de prioridad

### 3.1 · E1/E2: el ensamble ya preregistrado (costo casi nulo)

> **Medido el 22-09: no suma.** E1 detecta 12,6% ± 3,2 con lift 1,52× y E2 13,8% ± 5,0 con 1,48×,
> contra 15,7% ± 0,9 y 1,62× de survival stacking. Los dos pierden en lift por vehículo: el
> CNN-LSTM ordena autos peor y el promedio 50/50 lo diluye. Detalle en
> [memoria/f3-ensamble-e1-e2.md](memoria/f3-ensamble-e1-e2.md). Lo que sigue quedó como
> estaba escrito antes de correrlo.

**Qué es.** El promedio de rangos 50/50 de survival stacking y el CNN-LSTM, sin reentrenar (E1),
y el mismo con bagging por vehículo (E2). Está tal cual en
[f3-preregistro-landmark-ensamble.md](memoria/f3-preregistro-landmark-ensamble.md).

**Por qué podría ganar:**
- Los dos ordenan distinto: ρ de rango **0,24–0,36** por fila.
- Cada uno detecta autos que el otro no (8/9/8 contra 9/5/5).
- Con modelos débiles y poco correlacionados, el ensamble es la mejora más barata que conoce la
  literatura de supervivencia: los stacked survival models de Wey et al. 2015 equilibran sesgo y
  varianza entre candidatos.

**Qué hay:** las dos corridas R = 3 con sus OOF, `vehicle_bagging` en el registry y
`rescore_run.py`.

**Trampa.** El rango no es una probabilidad. Si gana, el Brier se reporta con Platt fuera de fold,
como dice el preregistro.

### 3.2 · Incidencia con 4–5× más eventos: los fallados sin fecha (la palanca más grande)

**El hecho que la habilita:**
- Fuera del universo quedaron **717 vehículos, 285 con evento**: 284 con la fecha por defecto
  (`IdentificationDate == daysUntilSale`) y 1 de un mercado excluido.
- La memoria dice textualmente que eso **no invalida la etiqueta a nivel vehículo**: son
  fallados, solo que sin fecha utilizable.
- Los sanos de CNTRY_1, CNTRY_2 y CNTRY_5 (432) se descartaron porque en esos mercados **el
  momento** del evento no se observa. Pero **el evento sí**: si hubieran fallado, figurarían como
  fallados con la fecha por defecto.
- Para la pregunta "¿este auto falla?" son negativos legítimos, siempre que hayan tenido
  exposición.

**Del lado dev del sorteo original** (el `parent` de `test_split.json`: 864 vehículos, 292
eventos) hay **574 excluidos con 232 eventos**. Los 143 del lado test original (53 eventos) no se
tocan, para no romper la simetría del sorteo. El test congelado (74) no se ve afectado en ningún
caso.

**Qué es:** un modelo de **incidencia** a nivel vehículo, "¿falla en algún momento?". Se
entrena **solo con esos 574 excluidos**, con features de la ventana temprana post-venta (primeros
30 o 60 días): las del rasgo temprano de la Fase 1, más km/día como control de uso. Se usa de
dos formas:
- **A-solo:** el score externo, aplicado a los 290 de dev **sin ajustar ni un parámetro sobre
  dev**. Es una prueba de transferencia limpia: no hay fold, no hay fuga posible. Se compara
  con el cure P0 (D1 0,599), con −km/L (0,581) y con −ProductionDay, en el mismo panel de hitos
  y con `landmark_metrics`.
- **A dentro de survival stacking:** `feat_ext_incidence` como covariable estática del
  vehículo. Vale NaN en los cortes anteriores al hito, por la regla 3.

**Por qué podría ganar:**
- *Qué auto* es de donde sale casi toda la detección (el techo de cohorte y (a′) lo muestran),
  y hoy se aprende con 60 eventos.
- Con ~290 eventos, el techo de complejidad de Riley et al. 2019 sube del orden de 5–7×: de 2–3
  parámetros a una decena. Es una cuenta aproximada, porque el criterio escala con n para un R²
  fijo.
- Es exactamente la descomposición del cure model (incidencia × latencia), llevada a donde rinde.
  La incidencia no necesita el tiempo; la latencia sí, y para ella quedan los 60 con fecha.
- La transferencia entre poblaciones con heterogeneidad es un problema estudiado: TransCox (Li,
  Shen & Ning 2023) y CoxTL (2025) toman coeficientes de una cohorte fuente y admiten diferencias
  de hazard base y de efectos.
- Hay versiones para las dos particularidades de acá: fuente con **censura por intervalo** (Xie
  2024) y fuente que es una **mezcla de Cox** (Liu 2026).

**Versión de verosimilitud completa (más adelante, no primero):**
- Un fallado sin fecha aporta "el evento cayó en algún momento entre la venta y el fin del
  registro". Es censura por intervalo.
- En un mixture cure, su contribución es π(x)·[1 − S_u(fin)]. En survival stacking, el bin del
  evento se imputa por EM: en el paso E se reparte el evento sobre los bins según el hazard
  actual, y en el paso M se reajusta el clasificador ponderado. Es la autoconsistencia de
  Turnbull 1976 con un GBM adentro.
- Hay mixture cure para censura por intervalo recientes (Liu, Szabo & Xiang 2026) que le dan
  base formal.

**Trampas, a declarar antes de mirar nada:**
1. **Qué significa la fecha por defecto.** Si en CNTRY_1 "fallado" es otra cosa (una campaña,
   un reclamo de entrega), la transferencia mide eso.
   - Chequeo sin etiqueta: comparar la distribución de uso y de mensajes de los fallados por
     mercado.
   - Chequeo con etiqueta, solo sobre excluidos: que el rasgo temprano apunte en la misma
     dirección que en CNTRY_3/4.
   - Es la pregunta 1 para Ford (§6).
2. **Telemetría posterior al evento.** Del fallado sin fecha no se sabe cuándo falló. En los
   mercados con fecha, el evento cae a 105–204 días post-venta (IQR) y solo 3 de 60 caen antes de
   30 + G. Usar **30 días** deja afuera casi toda la contaminación; 60 va como sensibilidad.
3. **Mercados con otro nivel** (CNTRY_4 da `msg_full_per_1000km` 5,45× más alto).
   - Estandarizar dentro del mercado × mes contra sus sanos.
   - El mercado va como **estrato, no como feature**, porque la proporción fallados/sanos
     muestreada difiere entre mercados.
   - Ojo: A5 del cure dice que la normalización contra la flota **no ayudó** dentro de dos
     mercados. Con cinco es probablemente obligatoria, pero es una decisión del preregistro.
4. **Exposición de los sanos excluidos.** No se sabe si la ventana del registro es la misma en
   los otros mercados. Como en la Fase 1 (H6), usar solo sanos producidos temprano y seguidos lo
   suficiente.
5. **Clones:** deduplicar con `src/data/dedupe.py` **antes** de armar el conjunto externo. Hay
   que verificar que ningún clon de un vehículo de test o de dev entre por la otra puerta.
6. **Sesgos de muestreo por estáticas:** `Engine`/`ModelSeries` quedan afuera, como en el set
   base (`ENG_3` es el 36% de los sanos y el 0% de los fallados).
7. **El error de SQL de la query de eventos** (rama `feat/f3-features-regeneracion`, pendiente
   con la mentora) puede cambiar quién es fallado. Si se corrige, esto se rehace primero.

**Qué hay en el repo:**
- `src/data/landmark.py` arma features por mes post-venta, pero `build_landmark_panel.py` está
  atado al universo de 364: hay que extenderlo (Track A).
- `FleetReferenceNormalizer` sirve para la estandarización por mercado × mes.
- `landmark_metrics` y el bootstrap pareado miden A-solo contra los pisos.
- `test_split.json` guarda la huella del sorteo padre (`ba9aa4d290bc6610`). Reproducir la
  primera etapa de `make_test_split.py` (semilla 42) da el lado de cada excluido y se verifica
  contra esa huella.

### 3.3 · Survival stacking en el reloj del evento, con el conjunto en riesgo de la ventana

**Qué es.** El mismo survival stacking, las **mismas filas y las mismas 53 features del panel v1**
y los mismos folds. Cambia solo cómo se arma el objetivo:
- **Reloj:** días desde la venta. `feat_cut_dss` reemplaza a `feat_cut_odo` como covariable del
  hazard base. Respaldo: Fase 1, H1 y H4b; Korn et al. 1997 y Thiébaut & Bénichou 2004 para la
  elección de escala con entrada tardía.
- **Conjunto en riesgo:** cada fila está en riesgo desde `max(fecha en que el odómetro llega a
  c + G, inicio de la ventana)` hasta `min(evento, fin de la ventana)`.
  - La regla 1 se conserva: el gap sigue en km.
  - Un sano sin exposición en la ventana no genera filas apiladas (CLAUDE.md: *"un sano sin
    exposición en la ventana no es un negativo"*).
- **Estimación con offset de exposición (PEM).** Se ajusta una regresión de Poisson con
  `log(días en riesgo del bin)` como offset. En LightGBM es `objective="poisson"` con
  `init_score`, en vez de la binaria por bin (Bender et al. 2020).
  - La ventana corta bins por la mitad, y el apilado binario tendría que tirar esos bins o
    contarlos enteros.
  - El offset los cuenta exacto, y el marco admite truncamiento y features que cambian con
    cualquier learner.
- **Score:** `1 − S(H_d | x)` con un horizonte fijo en días.
  - **Trampa:** no traducir H = 3.000 km a días con los km **futuros** del vehículo, porque eso
    es información posterior al corte.
  - Si hace falta, usar los km/día del pasado del vehículo.

**Por qué podría ganar:**
1. **Saca el ruido de etiqueta que ya está medido.** El 28% de los horizontes sanos cae en parte
   fuera del registro, y el 1,6% entero afuera. Hoy son negativos que el modelo aprende como
   "sano verificado".
2. **Ataca la (b) de raíz.** Si el calendario ayuda porque mide la exposición, definir el
   conjunto en riesgo con la ventana le quita ese papel. La (b) de esta corrida es la prueba.
3. **Mide el *cuándo* en el reloj del evento.** Hoy lo aporta `feat_cut_odo`, pero sd(log) en km
   es 0,97 contra 0,24 en edad.
4. **Conserva lo que el cure no tenía:** las 53 features. A6 muestra que en las filas que
   comparten el finalista ordena mejor que el cure (0,592 contra 0,543).

**Por qué no es "otro modelo" en términos de presupuesto.** Mismas filas, mismas features, mismos
folds, mismo learner. Es survival stacking con la censura que CLAUDE.md ya declaró correcta el
22-09: la regla de *"una corrida nueva se justifica por hacer comparable una fila que ya
existe"*. Además **reemplaza a L1** del preregistro del 21-09, que proponía landmarks en km: la
Fase 1 mostró después que ese era el reloj equivocado.

**Cómo se mide.** Survival stacking y esta variante sobre las mismas filas, con dos etiquetas de
evaluación:
- **la dura del panel v1** (`label`, la de toda la tabla de `results/`);
- **la corregida por ventana:** el horizonte entero dentro del registro, y las filas con el
  horizonte fuera quedan sin etiqueta.

La primera mantiene la comparabilidad; la segunda es la correcta. Decide la que diga el
preregistro, no la que dé mejor.

**Variante B-cal (condicional, si la (b) sigue marcando).** Verosimilitud parcial **estratificada
por mes de calendario**: cada evento se compara solo contra los que estaban en riesgo el mismo día
post-venta y el mismo mes.
- Es el nested case-control o la logística condicional (Langholz & Borgan 1995), la versión
  "por construcción" del ranker por celdas de f3-modelos-candidatos.md §2.3.
- Con un learner, es la aproximación caso-control de la parcial de Cox de Kvamme et al. 2019
  (Cox-CC), o un LightGBM con objetivo de ranking y una query por conjunto en riesgo.

**Qué hay en el repo:**
- el registry de `targets.py` (un modo nuevo);
- `cure_window`, que ya calcula exposición dentro de la ventana;
- `aux_km_observed_after_cut`;
- el anclaje de fechas (`src/data/anchor.py`);
- `event_clock.yaml`.

Falta la fecha en que cada corte llega a c + G. Sale de los viajes del vehículo, igual que
`cut_date`.

### 3.4 · 3.3 + 3.2: el reloj correcto con la incidencia externa

La combinación que la teoría dice que debería ganar:
- **qué auto**, aprendido con ~290 eventos (§3.2);
- **cuándo**, con el reloj y el conjunto en riesgo correctos (§3.3).

Se corre solo si 3.2 (A-solo) le gana a su piso en el panel de hitos. Si no, la covariable
externa no tiene nada que aportar.

### 3.5 · Otro learner dentro de 3.3 (condicional, un solo lugar)

Solo si 3.3 le gana a survival stacking, igual que la regla que dejaba correr P2 en el cure
solo si P1 ganaba. En orden:

| learner | a favor | en contra | fuente |
|---|---|---|---|
| **BART en tiempo discreto** (probit sobre el mismo dataset persona-período) | el prior regulariza con muestras chicas; da **intervalo posterior por vehículo**, útil en el dashboard; admite entrada tardía gratis (las filas en riesgo empiezan tarde) | MCMC más lento; en Python es `pymc-bart` y en R el paquete `BART` (`surv.bart`) | Sparapani et al. 2016 |
| **BoXHED2.0** | boosting no paramétrico del hazard con features que cambian y truncamiento a izquierda, verosimilitud exacta de proceso de conteo | binarios precompilados para **Python 3.8** y nada de pip: fricción con el entorno | Pakbin et al. 2025 |
| **LTRC forests** | bosques para truncados a izquierda con features que cambian | solo R | Yao et al. 2022 |
| **TabPFN-2.5 / MITRA como probabilidad de falla acumulada** | en contexto, sin tunear; el paper reporta ganar en promedio sobre Cox, RSF y DeepHit, y resuelve las features que cambian con landmarking | **licencia:** Prior Labs lista todos los pesos, v2 incluido, como no comerciales; el preregistro del cure asumía v2 "con atribución", hay que resolverlo antes del pitch. SurvPFN acepta ≤ 10 features y solo estáticas | Kim et al. 2026; Böhm et al. 2026; docs de Prior Labs |

**Qué esperar:** poco. Es lo que dice Burk et al. 2026, y este repo ya lo vio: cinco familias de
modelos quedan en 0,15–0,17 de PR-AUC sobre el mismo panel.

### 3.6 · Capa de decisión: el umbral con garantía (no consume presupuesto)

No mejora el orden: hace que el punto de operación sea **estable y defendible**, y la estabilidad
es criterio del proyecto.

Hoy el 5% de falsas alarmas se fija empíricamente sobre ~118 sanos: ~6 vehículos de margen. En el
test (54 sanos), el 5% realizado puede quedar bastante por encima. Dos herramientas:
- **Neyman-Pearson (algoritmo "umbrella").** Elige el umbral para que la tasa de falsas alarmas
  sea ≤ α con probabilidad alta (Tong, Feng & Li 2018; paquete `nproc`).
- **Selección conforme.** Elige qué autos alertar controlando la proporción de alertas falsas
  (FDR) sobre cualquier score (Jin & Candès 2023).

Las dos son monótonas en el score: no cambian ni la detección-a-umbral ni el lift, así que no son
candidatos. La alternativa de **entrenar** para la región de falsas alarmas bajas (AUC parcial,
LibAUC) sí cambia el orden y consume un lugar. Además está pensada para redes, no para GBM, así que
no se recomienda.

### 3.7 · Exposición acumulada con rezagos (tercera línea)

**Qué es.** En vez de la ventana de W = 1.000 km, el riesgo depende de toda la historia de uso
desde la venta, con pesos por rezago que se estiman suavizados (exposición-rezago-respuesta,
Bender et al. 2019; `pammtools`).

**Por qué tiene sentido acá:**
- La auditoría (a) mostró que sacar "el ruido de qué ventana le tocó" **mejora** el ordenador de
  vehículos.
- La Fase 1 mostró que el rasgo es estable desde el primer mes.
- Un promedio acumulado desde la venta, con los pesos estimados, es la versión con modelo de esa
  observación.

**En contra:** es R, y el desvío contra la historia previa ya no sumó
([f3-desvio-historia.md](memoria/f3-desvio-historia.md)). Solo si 3.3 abre el camino.

---

## 4 · Lo que la bibliografía ofrece y acá no conviene

| idea | por qué no, con evidencia del repo |
|---|---|
| supervivencia profunda (DeepHit, Dynamic-DeepHit, Cox-Time) | 60 eventos. GRU y CNN-LSTM no superan al finalista ([f3-gru-secuencial.md](memoria/f3-gru-secuencial.md), f3-piso-posicional.md). Burk et al. 2026 no encuentra ventaja en baja dimensión |
| joint models, HMM, landmarking con trayectoria modelada (Cipriani et al. 2025) | necesitan una trayectoria que anticipe; la Fase 1 (H5) muestra que **no la hay** |
| fragilidad / efecto aleatorio por vehículo | GPBoost: σ² ≈ 9,4–10,3, C-index 0,507 ([f3-survival-stacking.md](memoria/f3-survival-stacking.md)) |
| proceso gamma / umbral de degradación | no hay carga irreversible medible a 8.000 km ([f3-proceso-gamma.md](memoria/f3-proceso-gamma.md)) |
| desvío contra la flota (COSMO) o contra sí mismo (z-self, CUSUM) | A5 del cure: la normalización contra la flota no ayuda; `feat/zself-cusum` P ≈ 0,52–0,54; el desvío contra la historia no suma |
| modelos fundacionales de series (TimesFM, MOMENT, Chronos) | TimesFM medido: no entra ni como score ni como features ([f3-timesfm-zeroshot.md](memoria/f3-timesfm-zeroshot.md)) |
| barridos de hiperparámetros, SMOTE | presupuesto de comparaciones; fuga dentro del fold (f3-modelos-candidatos.md §4) |
| más features de ventana | la rama `feat/f3-features-regeneracion` barrió 7.310 cocientes sin superar su nulo: la señal es una dimensión |

## 5 · Propuesta de lista cerrada para el próximo preregistro

Con las mismas reglas que [f3-preregistro-landmark-ensamble.md](memoria/f3-preregistro-landmark-ensamble.md):
- R = 3, solo dev;
- deciden la detección a ≤ 50 FA/1.000 y el lift por vehículo;
- "le gana" = diferencia mayor que el desvío combinado;
- nada reemplaza a la referencia sin ganarle también al piso.

| lugar | corrida | consume presupuesto | condición |
|---|---|---|---|
| 0 | ~~**E1, E2**~~ **corridas el 22-09: pierden** ([f3-ensamble-e1-e2.md](memoria/f3-ensamble-e1-e2.md)) | no: ya preregistradas en 9b7bfb4 | ninguna |
| 1 | **A-solo** (§3.2): incidencia externa, cero parámetros en dev, panel de hitos | sí | antes, un punto de control **solo con conteos**: excluidos del lado dev con `daysUntilSale`, ≥ 15 viajes en 30 d y exposición suficiente, por mercado |
| 2 | **SS post-venta con ventana** (§3.3), reemplaza a L1 | no en el sentido de CLAUDE.md: hace comparable la fila del finalista | ninguna |
| 3 | **2 + A** (§3.4) | sí | A-solo le gana a su piso |
| cond. | B-cal (§3.3) | sí | la (b) de la corrida 2 marca |
| cond. | BART dentro de 2 (§3.5) | sí | la corrida 2 le gana a survival stacking |

**Cuánto hay que ganar para que se vea.** Con 53 fallados, un vehículo son 1,9 puntos de
detección. El desvío de la referencia es 0,9 y el de un candidato nuevo suele ser 2–4 (CNN-LSTM
3,6, control 3,2). Para "ganarle", la mejora tiene que ser de **≥ 2–3 vehículos** en promedio. Esa
es la vara con la que hay que leer cualquier expectativa de las fuentes.

## 6 · Preguntas para Ford que cambian este orden

Son las del §10 de [f3-reloj-y-ventana-del-evento.md](memoria/f3-reloj-y-ventana-del-evento.md).
Estas dos reordenan esta lista:

1. **¿Qué significa `IdentificationDate == daysUntilSale`?**
   - Si es "evento registrado sin fecha", el §3.2 pasa a ser el camino principal. Y si Ford
     puede dar la fecha real, el universo se amplía de 364 a ~1.000 vehículos cambiando dos
     claves del YAML (f2-universo-fecha-usable.md).
   - Si es otra cosa (un reclamo de entrega, una campaña), el §3.2 muere y queda el §3.3.
2. **¿La ventana de extracción es la misma en todos los mercados?** Define si los sanos de
   CNTRY_1/2/5 son negativos resueltos.

---

## Fuentes

**Garantía automotriz, escalas de tiempo, muestreo**
- Lawless, Crowder & Lee (2009). *Analysis of reliability and warranty claims in products with age and usage scales.* Technometrics 51(1):14–24. [Semantic Scholar](https://www.semanticscholar.org/paper/Analysis-of-Reliability-and-Warranty-Claims-in-With-Lawless-Crowder/f3dd52d70791b0fb7304ab81d5f09d1d8c9df422)
- Kalbfleisch, Lawless & Robinson (1991). *Methods for the analysis and prediction of warranty claims.* Technometrics. [ACM DL](https://dl.acm.org/doi/10.2307/1268780)
- Wu (2012). *Warranty data analysis: a review.* Quality and Reliability Engineering International. [Wiley](https://onlinelibrary.wiley.com/doi/abs/10.1002/qre.1282) · [preprint](https://kar.kent.ac.uk/31005/1/Review01.pdf)
- Duchesne & Lawless (2000). *Alternative time scales and failure time models.* Lifetime Data Analysis 6:157–179. [Springer](https://link.springer.com/article/10.1023/A:1009616111968)
- Carollo, Eilers, Putter & Gampe (2025). *Smooth hazards with multiple time scales.* Statistics in Medicine 44(1–2). [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11701971/) · [TwoTimeScales](https://arxiv.org/html/2603.13009)
- Korn, Graubard & Midthune (1997). *Time-to-event analysis of longitudinal follow-up of a survey: choice of the time-scale.* Am J Epidemiol 145(1):72–80. [OUP](https://academic.oup.com/aje/article/145/1/72/68300)
- Thiébaut & Bénichou (2004). *Choice of time-scale in Cox's model analysis of epidemiologic cohort data: a simulation study.* Statistics in Medicine. [Wiley](https://onlinelibrary.wiley.com/doi/10.1002/sim.2098)
- Zhu & Wang (2012). *Analysing bivariate survival data with interval sampling and application to cancer epidemiology.* Biometrika 99:345–361. [PubMed](https://pubmed.ncbi.nlm.nih.gov/23843662/)
- Langholz & Borgan (1995). *Counter-matching: a stratified nested case-control sampling method.* Biometrika 82(1):69–79. [OUP](https://academic.oup.com/biomet/article-abstract/82/1/69/255344)
- Saerens, Latinne & Decaestecker (2002). *Adjusting the outputs of a classifier to new a priori probabilities.* Neural Computation 14(1):21–41. [MIT Press](https://direct.mit.edu/neco/article/14/1/21/6577/Adjusting-the-Outputs-of-a-Classifier-to-New-a)
- Prentice (1986). *A case-cohort design for epidemiologic cohort studies and disease prevention trials.* Biometrika 73(1):1–11. *(Cita clásica, no reverificada en esta sesión.)*

**Supervivencia con ML, truncamiento y features que cambian**
- Craig, Zhong & Tibshirani (2021). *Survival stacking: casting survival analysis as a classification problem.* [arXiv:2107.13480](https://arxiv.org/abs/2107.13480)
- Bender, Rügamer, Scheipl & Bischl (2020). *A general machine learning framework for survival analysis.* ECML-PKDD. [Springer](https://link.springer.com/chapter/10.1007/978-3-030-67664-3_10)
- Bender, Scheipl, Hartl, Day & Küchenhoff (2019). *Penalized estimation of complex, non-linear exposure-lag-response associations.* Biostatistics 20(2):315–331. [OUP](https://academic.oup.com/biostatistics/article-abstract/20/2/315/4852816)
- Pakbin, Wang, Mortazavi & Lee (2025). *BoXHED2.0: Scalable boosting of dynamic survival analysis.* J Stat Softw 113(3). [JSS](https://www.jstatsoft.org/article/view/v113i03) · [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12314777/) · [GitHub](https://github.com/BoXHED/BoXHED2.0)
- Yao, Frydman, Larocque & Simonoff (2022). *Ensemble methods for survival function estimation with time-varying covariates.* Stat Methods Med Res 31(11):2217–2236. [arXiv:2006.00567](https://arxiv.org/abs/2006.00567)
- Sparapani, Logan, McCulloch & Laud (2016). *Nonparametric survival analysis using BART.* Statistics in Medicine. [Wiley](https://onlinelibrary.wiley.com/doi/10.1002/sim.6893)
- Jaeger et al. (2024). *Accelerated and interpretable oblique random survival forests.* JCGS 33(1):192–207. [arXiv:2208.01129](https://arxiv.org/abs/2208.01129)
- Kvamme, Borgan & Scheel (2019). *Time-to-event prediction with neural networks and Cox regression.* JMLR 20(129). [JMLR](https://jmlr.org/papers/v20/18-424.html)
- Wey, Connett & Rudser (2015). *Combining parametric, semi-parametric, and non-parametric survival models with stacked survival models.* Biostatistics 16(3):537–549. [OUP](https://academic.oup.com/biostatistics/article/16/3/537/269846)
- Burk, Zobolas, Bischl, Bender, Wright & Sonabend (2026). *A large-scale neutral comparison study of survival models on low-dimensional data.* Bioinformatics 42(5). [arXiv:2406.04098](https://arxiv.org/abs/2406.04098) · [OUP](https://academic.oup.com/bioinformatics/article/42/5/btag186/8656833)
- Riley et al. (2019). *Minimum sample size for developing a multivariable prediction model: Part II.* Statistics in Medicine 38(7):1276–1296. [Wiley](https://onlinelibrary.wiley.com/doi/10.1002/sim.7992)

**Predicción dinámica, cure, censura por intervalo, transfer**
- van Houwelingen (2007). *Dynamic prediction by landmarking in event history analysis.* Scand J Stat 34:70–85. [Wiley](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1467-9469.2006.00529.x)
- Cipriani, Alfò & Signorelli (2025). *Dynamic prediction in mixture cure models: a model-based landmarking approach.* [arXiv:2509.18875](https://arxiv.org/abs/2509.18875)
- Lee, Yoon & van der Schaar (2019). *Dynamic-DeepHit.* IEEE TBME. [Semantic Scholar](https://www.semanticscholar.org/paper/Dynamic-DeepHit:-A-Deep-Learning-Approach-for-With-Lee-Yoon/293d91a1d4b3f296cb040a766bf7ab31197b1a8f)
- Turnbull (1976). *The empirical distribution function with arbitrarily grouped, censored and truncated data.* JRSS B 38(3):290–295. [OUP](https://academic.oup.com/jrsssb/article/38/3/290/7027379)
- Liu, Szabo & Xiang (2026). *A double-semiparametric approach for extending mixture cure models with interval-censored data.* Stat Methods Med Res. [doi](https://doi.org/10.1177/09622802261442911)
- Li, Shen & Ning (2023). *Accommodating time-varying heterogeneity in risk estimation under the Cox model: a transfer learning approach (TransCox).* JASA 118. [T&F](https://www.tandfonline.com/doi/full/10.1080/01621459.2023.2210336)
- *Adaptive transfer learning for time-to-event modeling (CoxTL)* (2025). [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12132156/)
- Xie et al. (2024). *Transfer learning under the Cox model with interval-censored data.* Stat Anal Data Min. [Wiley](https://onlinelibrary.wiley.com/doi/abs/10.1002/sam.11680)
- Liu et al. (2026). *Transfer learning analysis of the Cox mixture model.* Stat Anal Data Min. [Wiley](https://onlinelibrary.wiley.com/doi/abs/10.1002/sam.70059)

**Modelos fundacionales tabulares**
- Kim, Lai & Zhang (2026). *Tabular foundation models can do survival analysis.* [arXiv:2601.22259](https://arxiv.org/abs/2601.22259) · [código](https://github.com/dain20000222/TabularFM4Survival)
- Böhm, Purucker, Hutter & Schlosser (2026). *SurvPFN: towards foundation models for survival predictions.* [arXiv:2606.04564](https://arxiv.org/html/2606.04564v2)
- Prior Labs, licencias de los modelos TabPFN. [docs](https://docs.priorlabs.ai/models) · [TabPFN-2.5](https://arxiv.org/pdf/2511.08667)

**Mantenimiento predictivo automotriz**
- Frisk, Krysander & Larsson (2014). *Data-driven lead-acid battery prognostics using random survival forests.* PHM Society. [PHM](https://papers.phmsociety.org/index.php/phmconf/article/view/2370)
- Prytz et al. (2015). *Predicting the need for vehicle compressor repairs using maintenance records and logged vehicle data.* Eng Appl AI. [ACM DL](https://dl.acm.org/doi/abs/10.1016/j.engappai.2015.02.009)
- Rögnvaldsson, Nowaczyk, Byttner, Prytz & Svensson (2018). *Self-monitoring for maintenance of vehicle fleets.* DMKD 32(2):344–384. [Springer](https://link.springer.com/article/10.1007/s10618-017-0538-6)
- Kharazian et al. (2025). *SCANIA Component X dataset.* Scientific Data 12. [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11933314/) · [arXiv:2401.15199](https://arxiv.org/abs/2401.15199)
- Dimidov, Jafarnejad & Frank (2026). *An empirical study on predictive maintenance for Component X in heavy-duty Scania trucks.* [arXiv:2606.12486](https://arxiv.org/html/2606.12486)
- Doikin et al. (2020). *Knowledge-enabled machine learning for predictive diagnostics: a case study for an automotive diesel particulate filter.* ESREL/PSAM. [ResearchGate](https://www.researchgate.net/publication/345518616_Knowledge-Enabled_Machine_Learning_for_Predictive_Diagnostics_A_Case_Study_for_an_Automotive_Diesel_Particulate_Filter)

**Capa de decisión**
- Tong, Feng & Li (2018). *Neyman-Pearson classification algorithms and NP receiver operating characteristics.* Science Advances 4(2):eaao1659. [Science](https://www.science.org/doi/10.1126/sciadv.aao1659)
- Jin & Candès (2023). *Selection by prediction with conformal p-values.* JMLR. [arXiv:2210.01408](https://arxiv.org/abs/2210.01408)
- Yuan et al. (2023). *LibAUC: a deep learning library for X-risk optimization.* KDD. [ACM DL](https://dl.acm.org/doi/10.1145/3580305.3599861)
