# Decisiones

Una entrada por decisión, la más nueva arriba. El **porqué** es la parte que
importa: sin él, el que venga la revierte sin enterarse de qué estaba resolviendo.

---

## 2026-09-24 · F4: el porqué de una alerta de K2 se explica con V3 y solo con hábitos que coinciden con la física del DPF

**Alcance:** explicabilidad de K2, solo dev. **No es un candidato ni gasta presupuesto de
comparaciones**: explica al finalista y no cambia ninguna métrica oficial. Preregistro
`configs/explain_k2.yaml` (3882919), antes de reentrenar y de mirar un SHAP. Evidencia:
[f4-explicabilidad-k2.md](f4-explicabilidad-k2.md).

**Qué se decidió:**
- **La explicación al cliente es V3**: TreeSHAP del hazard de K2, promediado entre las 3
  repeticiones. Las tres variantes elegibles (V1, V2, V3) pasan la fidelidad, y V3 es la más estable
  (Jaccard del top 3 de 0,73, contra 0,57 y 0,48).
  - **Aditividad:** 1e-14.
  - **ρ contra el score:** 0,999.
  - **Borrado:** sacar las 3 accionables de mayor contribución baja el score 0,20, contra 0,04 al
    azar (p = 0,005).
- **Al texto al cliente llegan solo las accionables cuyo efecto global coincide con la física**, que
  son estables entre repeticiones, que suben el riesgo de ese auto en las 3 réplicas y cuyo valor está
  del lado riesgoso de la mediana sana comparable. Hoy son 10 de las 20 accionables.
  - **Afuera quedan cuatro contrarias a la física**: viajes más largos y temperatura máxima más alta
    ⇒ más riesgo.
  - **También quedan afuera** los arranques en frío (débil) y las cinco sin hipótesis, incluida la
    duración del viaje, que es la cuarta más importante.
  - **Si un auto alertado no tiene ningún factor, el mensaje lo dice** (2 de 32 alertas). No se
    rellena.
- **Síntomas y contexto nunca se nombran.** Los síntomas son el estado del filtro, los mensajes, el
  aceite y el consumo; el contexto, odómetro, mercado, ritmo por día y bin. Los síntomas tienen una
  línea aparte ("lo que ve el filtro"), que en las 32 alertas dice que suman riesgo.
- **El horizonte se dice en km (500 a 3.500 desde el corte) y en semanas al ritmo del auto.** No se
  calcula un riesgo a 4 meses: haría falta un horizonte en días evaluado dentro de la ventana del
  registro, y la prevalencia real.

**Por qué:**
- **SHAP explica al modelo, no al auto**, y (a′) = +0,019: K2 sabe *qué auto* se parece a los que
  fallaron, no *qué cambió*. Un texto sin estos filtros le habría dicho "manejás lento" al detectado
  con el score más alto (VEH_0451), que va más rápido que sus pares y cuyo riesgo viene de los
  síntomas.
- **El control de calendario** (desvío contra los sanos del mismo mercado × mes) no da vuelta ningún
  signo. "Motor frío" es el auto, no el invierno, así que la recomendación de uso se sostiene.

---

## 2026-09-24 · F4: dashboard con K2, y el punto de operación se elige con costos, no con una matriz en el entrenamiento

**Alcance:** F4 (dashboard contra el panel real, solo dev) y un reporte de costos. **No cambia el
finalista** ni consume presupuesto. Evidencia: [f8-costos-k2.md](f8-costos-k2.md).

**Qué se decidió:**
- **La matriz de costos no entra al entrenamiento** (pesos de clase, pérdida sesgada hacia los falsos
  negativos). No cambia el orden de los autos: solo mueve el umbral sobre la misma curva, y rompe la
  calibración del hazard (F7 §7). **El costo elige dónde operar**, sobre las predicciones de K2.
- **Escenarios con fuente pública** (diagnóstico, limpieza o reemplazo del DPF, grúa, aviso remoto,
  flota comercial) × efectividad de la prevención × prevalencia real. Resultado:
  - con fallas baratas, lo óptimo es no alertar;
  - en la zona validada (≤ 10% de falsas alarmas), K2 ahorra 0–6%, casi todo por encima del azar;
  - los ahorros grandes piden 15–35% de falsas alarmas, donde nada está validado;
  - si la alerta es un aviso remoto casi gratis, alertar a todos gana y el modelo sobra.
- **Para el pitch no se cita un ahorro único:** se muestra la franja donde el modelo paga. A Ford
  hay que pedirle tres cosas: el costo de una falla contra el de un diagnóstico, el tipo de
  intervención (visita o aviso remoto) y la prevalencia real.
- **El dashboard** (`scripts/dashboard_k2/`) toma sus números oficiales de `window_eval.json` y
  `decision_layer.json`. Lo que recalcula (la alerta por vehículo, el costo esperado) usa las mismas
  funciones que los scripts, y tiene su chequeo (180 en verde).

---

## 2026-09-24 · F8: sin Ford no se recuperan eventos, y el punto de operación de K2 se sostiene

**Alcance:** dos trabajos exploratorios que no consumen presupuesto de comparaciones. **No cambian
el finalista.** Evidencia y comandos: [f8-datar-eventos-fase0.md](f8-datar-eventos-fase0.md) y
[f8-capa-decision-k2.md](f8-capa-decision-k2.md). Solo dev; test sin tocar.

**1 · Fechar a los fallados sin fecha desde la telemetría: no se puede.** La idea era sumar los 33
fallados sin fecha de CNTRY_3/4 (+55% de eventos, misma población) fechándolos con una marca de
intervención. El criterio se commiteó antes de medir.
- Ninguna marca pasa: aceite, días sin uso, DPF con motor apagado, km sin telemetría, mensajes.
- La caída del DPF con motor apagado se concentra cerca del evento (+0,31 sobre el nulo del mismo
  auto), pero es demasiado frecuente para fechar: 14% a ±14 d.
- **Descartado en la misma discusión: SMOTE para "subir a 1.000 autos".** Interpola entre los 45
  fallados, así que no agrega eventos. Hecho antes del split, filtra entre folds, y rompe la
  calibración del hazard. `docs/f3-modelos-candidatos.md` §4 ya lo prohibía.
- **Consecuencia:** las vías para tener más eventos sin Ford están cerradas (esta, y la incidencia
  externa de F5). Lo que queda sin Ford son cambios de modelo, que desde F6 mueven ±2 autos de 45,
  dentro del ruido. **Para mejorar la detección de forma medible hace falta la respuesta de Ford**
  sobre la fecha por defecto (o un registro de taller). Queda abierta una sola idea con datos: usar
  la firma del DPF para probar "es el mismo evento" y sumar a esos autos censurados por intervalo,
  con su preregistro.

**2 · La capa de decisión sobre K2.**
- El umbral del 5%, fijado con los sanos de otros folds, da **3,5% de falsas alarmas realizadas y
  15,6% de detección**: no estaba sobreajustado.
- **A 10% de falsas alarmas detecta 26,7%**, con el mismo exceso sobre el nulo que el 5%. **Al 2% es
  azar.**
- **Neyman-Pearson:** garantizar ≤ 10% con 95% de confianza equivale a ~4% realizado, con 17,0% de
  detección. Garantizar ≤ 5% es demasiado conservador con 95 sanos (5% de detección).
- **Para el pitch:** "~15–17% a ≤ 5% de falsas alarmas, verificado fuera de muestra", la garantía
  de ≤ 10%, y el dial a 10% como decisión de costo de Ford. No prometer alertas con ≤ 2% de falsas
  alarmas.

**Qué se queda:** `scripts/audit_event_dating.py`, `scripts/decision_layer.py` y sus dos configs.

---

## 2026-09-24 · F7 no cambia el finalista: ningún candidato que baja la varianza de K2 cumple la regla

**Alcance:** cierra el preregistro F7 ([f7-preregistro-varianza-k2.md](f7-preregistro-varianza-k2.md),
0b29b2b; implementación 88660e7). **No cambia el finalista.** Evidencia, auditorías y comandos:
[f7-varianza-k2.md](f7-varianza-k2.md). Solo dev, R = 3, contra K2; test sin tocar.

**Qué se decidió, con las seis condiciones de F6** (decide la etiqueta corregida V):
1. **P1, K2 embolsado:** 15,6% ± 0,0 (7 / 7 / 7 de 45) contra 17,0% ± 3,8. Falla las condiciones
   1, 3 y 6 (exceso sobre el nulo +7,7 contra +9,5). No se adopta.
2. **P2, K2 monótono:** 20,7% ± 9,3 (8 / 15 / 5). La diferencia (+0,037) no supera el desvío
   combinado (0,101) y una repetición pierde. No se adopta, aunque es el único con más exceso
   sobre el nulo que K2 (+13,0).
3. **P3, hazard logístico con cinco covariables:** pierde en todo (6,7%, lift 1,33×) y detecta
   menos que un score al azar. No se adopta.

**Por qué importa aunque no gane nada:**
- **El 17,0% de K2 es la lectura optimista.** Su versión embolsada ordena casi igual (ρ 0,90–0,92
  por vehículo) y detecta 7 de 45 en las tres repeticiones; K2 detectó 7, 10 y 6. Con la etiqueta
  dura se invierte (P1 17,0% ± 1,5, K2 15,1% ± 3,1). **Para el pitch se cita el rango, ~15–17%,
  no el número puntual.**
- **Cinco coeficientes no alcanzan.** La señal de K2 no es la regla física lineal del 18-09: usa
  no linealidades o features que esa regla no lleva.
- **Monótono sube la media y el desvío a la vez**, como K1 y K3 en F6.

**Qué se queda:**
- `monotone`, `columns` y `backend: logistic` en `DiscreteSurvivalStacker`, con los defaults
  que no cambian nada, y `wants_feature_names` en `cv.py` / `audit_model.py`;
- los tres configs `exp_f7_*`;
- 12 chequeos (178 en verde).

---

## 2026-09-22 · Nuevo finalista: survival stacking con la ventana del registro en km y horizonte completo (K2)

**Alcance:** cierra el preregistro F6 ([f6-preregistro-deteccion-vehiculo.md](f6-preregistro-deteccion-vehiculo.md),
b12056c). **Cambia el finalista.** Evidencia, auditorías y comandos:
[f6-deteccion-vehiculo.md](f6-deteccion-vehiculo.md). Solo dev, R = 3, con las mismas filas,
features y folds; test sin tocar.

**El pedido:** un modelo que detecte más autos que van a fallar, con antelación. Para eso se gastó un
presupuesto de comparaciones nuevo y cerrado: tres candidatos, con una regla más estricta que la de
F5.

**Qué se decidió, con la regla del preregistro** (decide la etiqueta corregida V):
1. **K2 se adopta.** Es `survival_stacking` con el target `window_km_survival`:
   - entrada tardía en el odómetro del 01-09-2025;
   - un sano sale en `min(último odómetro, odómetro del 11-03-2026)`;
   - se entrena hasta 55.000 km.

   Lo demás no cambia: features, booster y score.
   - **Detección 17,0% ± 3,8 (7 / 10 / 6 de 45) contra 11,9% ± 2,8 (4 / 7 / 5).** Gana en las
     tres repeticiones: +0,067 / +0,067 / +0,022.
   - Lift por vehículo 1,658 contra 1,655, y anticipación mediana 7.438 km contra 9.354.
   - Les gana a los dos pisos y pasa (a0).
   - Su exceso sobre el nulo de tamaño de bolsa es +9,5 puntos, contra +4,1 del finalista anterior.
2. **K1** (media acumulada causal del score del finalista) y **K3** (K2 + esa media) empatan con más
   desvío. No se adoptan.

**Por qué K2 y no otra cosa:** el finalista anterior entrenaba como supervivencia dos cosas que no lo
eran:
- **514 cortes de autos que después fallan**, porque su evento caía más allá de 6.000 km;
- **1,2 millones de km sanos después del cierre del registro.**

Como el rasgo que predice es del auto y está desde el primer mes, eso le enseñaba a llamar "sano" a un
auto que va a fallar. K2 solo corrige el conjunto en riesgo.

**Lo que hay que decir junto con el número:**
- **Es una mejora modesta:** +2,3 autos de 45 en promedio. El bootstrap pareado por vehículo no la
  separa del cero: +5,5 puntos, IC90 [−8,9; +17,8]. Con la etiqueta dura, K2 empata en detección
  (15,1% ± 3,1 contra 15,7% ± 0,9) y ordena mejor los autos (1,67× contra 1,62×).
- **Los autos que suma los detecta más cerca del evento:** la anticipación mediana con V baja unos
  1.900 km. Pasa la guarda preregistrada.
- **La (b) de calendario baja de +0,049 a +0,020.** El límite principal del finalista anterior
  queda casi resuelto: el atajo de calendario era exposición al registro.
- **Suavizar el score hacia atrás no suma.** Le quita al score la ventaja artificial de las bolsas
  largas, pero la alerta llega más tarde o no llega.

**Qué se queda:**
- el target `window_km_survival` y `entry_km` en `DiscreteSurvivalStacker`;
- `scripts/build_km_window_panel.py`;
- `scripts/smooth_scores.py`;
- `scripts/audit_detection_null.py`: el nulo de tamaño de bolsa para la detección, que ninguna
  corrida anterior tenía. Un score al azar detecta ~7,5% con V;
- 8 chequeos (166 en verde).

---

## 2026-09-22 · Survival stacking en días post-venta con la ventana no se adopta: aprende el *cuándo* y pierde el *qué auto*

**Alcance:** cierra el preregistro de F5 §3.3
([f5-preregistro-ss-post-venta.md](f5-preregistro-ss-post-venta.md)). **No cambia el finalista.**
Evidencia, auditorías, diagnóstico y comandos: [f5-ss-post-venta.md](f5-ss-post-venta.md). Solo
dev, R = 3, con las mismas filas, features y folds que el finalista; test sin tocar.

**Qué se decidió, con la regla del preregistro** (decide la etiqueta corregida: solo las filas
con el horizonte entero dentro de la ventana del registro):
1. **Pierde contra el finalista.**
   - En lift por vehículo pierde: 1,376 contra 1,655, una diferencia de −0,28 contra 0,15 de
     desvío combinado.
   - En detección empata: 9,6% ± 2,8 contra 11,9% ± 2,8.
2. Les gana a los dos pisos (`cut_odo` y `feat_cut_dss`) y pasa (a0). No alcanza.
3. Con la etiqueta dura también pierde: 9,4% ± 1,5 y 1,33× contra 15,7% ± 0,9 y 1,62×.
4. **Se cierran** §3.5 (otro learner dentro de 3.3), B-cal (la (b) no marca) y §3.7 del doc F5.

**Lo que esta corrida deja aunque no se adopte:**
- **El atajo de calendario del finalista es exposición al registro.** Con el conjunto en riesgo
  de la ventana, la (b) baja a +0,016, contra +0,049 del finalista. Es un límite del finalista con
  nombre: su (b) no es física.
- **Con la etiqueta corregida, el finalista detecta 11,9% ± 2,8** (4 / 7 / 5 de 45), no 15,7%.
  - Su lift se mantiene (1,66×): la corrección no le saca el orden de autos.
  - Parte de sus aciertos caían en filas que el registro no cubría. **El número honesto para el
    pitch es el corregido**, con su desvío.
- **El reloj en días aprende el *cuándo***: (a′) +0,067, contra +0,016. Pero el score queda
  dominado por los días desde la venta (ρ 0,51 por fila), y al promediar por vehículo eso le resta
  al *qué auto*. Es un diagnóstico posterior al veredicto, no preregistrado.
- **La infraestructura:**
  - el panel v1 en días con el tramo en riesgo por fila (`src/data/window_risk.py`);
  - la inversa de la proyección del evento;
  - el PEM con offset de exposición (`window_survival_stacking`);
  - la evaluación con las dos etiquetas (`scripts/eval_window_label.py`);
  - 10 chequeos (158 en verde).

**Qué no se puede separar con esta corrida:** cambió a la vez el reloj y el conjunto en riesgo.
El diagnóstico apunta al reloj, pero la ablación (km con la ventana) sería un candidato nuevo, y el
presupuesto está agotado.

---

## 2026-09-22 · La incidencia aprendida con los fallados sin fecha no corre: el rasgo temprano no se replica en la fuente

**Alcance:** cierra el preregistro de la incidencia externa
([f5-preregistro-incidencia-externa.md](f5-preregistro-incidencia-externa.md), §3.2 del doc F5)
en su compuerta G2. **No cambia el finalista.** Evidencia, diagnóstico y comandos:
[f5-incidencia-externa.md](f5-incidencia-externa.md). Solo la fuente (541 excluidos del lado dev
del sorteo padre, CNTRY_1/2/5). Dev y test no se tocaron.

**Qué se decidió, con la regla del preregistro:**
1. **G2 para.** El índice de pesos unitarios de P0 da AUC 0,513 [0,445; 0,585] entre los 114
   fallados y los 169 sanos elegibles de la fuente. Por mercado: 0,494 en CNTRY_1 y 0,517 en
   CNTRY_2.
2. Por eso la incidencia no se congela y **A-solo no se aplica a dev**. Survival stacking con la
   covariable externa (el paso 2) tampoco corre.
3. Con esto se cae también el §3.4 del doc F5, que dependía de A-solo.

**Por qué el negativo es creíble** (diagnóstico posterior a la parada, no preregistrado, solo la
fuente):
- Las direcciones no dependen de la normalización: cruda, por mes y por mercado × mes dan lo
  mismo.
- Son las mismas en los dos mercados grandes.
- No hay atajo de calendario en la fuente (−`ProductionDay` 0,455).
- No lo explican los autos quietos (0,490 sin ellos) ni la elegibilidad (0,494 con todos los
  fallados).
- Idle y refrigerante van **al revés** que en dev (0,43 y 0,44), y solo los viajes bajo régimen
  (0,58) y el uso (b = −0,34 [−0,66; −0,02]) apuntan como en dev.

**Qué no se puede separar:**
- **"Fallado" con fecha por defecto puede ser otro evento.** En CNTRY_1 los fallados casi no
  tienen exceso de mensajes de filtro (Full 0,91×, Overloaded 1,19×); en CNTRY_2 sí (1,61×,
  1,85×).
- **El rasgo puede ser de CNTRY_3/4, o en parte ruido.** Tampoco separa en CNTRY_2.
- Para separarlo hace falta la respuesta de Ford sobre `IdentificationDate == daysUntilSale`.

**Qué se queda:**
- el conjunto externo reproducible (`src/data/external.py`, verificado contra la huella del
  sorteo padre);
- la ventana de features fija del panel de hitos;
- el scorer congelado `external_incidence`;
- `fixed_weight` en `cure_mixture`;
- 9 chequeos (148 en verde).

**Lo que corrigió de los números del doc F5:**
- con la elegibilidad simétrica, los eventos de la fuente son 114, no ~290;
- con C ≈ 0,65, Riley admite 2–3 parámetros, no una decena.

---

## 2026-09-22 · El ensamble de finalistas (E1, E2) no se adopta: pierde en lift por vehículo

**Alcance:** corre E1 y E2 del preregistro del 21-09
([f3-preregistro-landmark-ensamble.md](f3-preregistro-landmark-ensamble.md)) sin cambios. **No
cambia el finalista.** Evidencia, auditorías y comandos: [f3-ensamble-e1-e2.md](f3-ensamble-e1-e2.md).
Solo dev, R = 3; test sin tocar.

**Qué se decidió, con la regla del preregistro:**
1. **E1** (rango percentil 50/50 de survival stacking y CNN-LSTM, sin reentrenar) **pierde**.
   - Lift por vehículo 1,515 contra 1,616: diferencia −0,101 contra un desvío combinado de 0,073.
   - En detección empata: 12,6% ± 3,2 contra 15,7% ± 0,9.
   - *"El ensamble no suma y el finalista sigue siendo survival stacking solo."*
2. **E2** (lo mismo con los dos finalistas embolsados por vehículo, 10 bolsas) **pierde**.
   - Lift 1,480: diferencia −0,136 contra 0,078.
   - Contra E1 baja el desvío de anticipación (4.121 contra 4.748 km) y sube el de detección
     (5,0 contra 3,2 puntos).
   - Por la letra de la regla de descarte, el bagging no queda descartado, porque bajó uno de
     los dos desvíos. Pero no hay nada que adoptar: perdió contra la referencia y sus
     componentes no son candidatos.

**Por qué el negativo es creíble:**
- Las dos corridas miembro se reproducen bit a bit.
- Las filas, las etiquetas y los folds se verificaron idénticos.
- (a0) cae a la tasa base.
- El ensamble le gana al piso posicional en las dos métricas que valen.

Pierde por una razón que se ve en los números: **el CNN-LSTM ordena autos bastante peor** (lift
1,33×), y el promedio 50/50 diluye lo que mejor hace survival stacking. La baja correlación
entre los dos (0,24–0,36) no alcanza cuando uno de los dos tiene menos señal.

**Qué se queda:**
- `vehicle_bagging`, `grouped_label` y `scripts/ensemble_rank.py`, que arma cualquier ensamble
  de corridas existentes con la cuenta de `train.py` y aplica el veredicto del preregistro.
- El componente survival stacking embolsado da 18,2% ± 3,2 de detección. No es candidato, y
  contra la referencia empata con más desvío.
- **La (b) sigue marcando** en todo lo que lleva survival stacking (+0,039 en E1, +0,046 en E2).

---

## 2026-09-22 · El cure model por hito post-venta no se adopta: el rasgo temprano no se separa del piso de producción ni del uso

**Alcance:** cierra el preregistro del cure model ([f3-preregistro-cure.md](f3-preregistro-cure.md)).
**No cambia el finalista**, que sigue siendo survival stacking sobre el panel v1, ni el panel v1.
Evidencia, auditorías y comandos: [f3-cure-model.md](f3-cure-model.md). Solo dev, R = 3; test sin tocar.

**Qué se decidió, aplicando la regla del preregistro en orden:**
1. **P0 (pesos unitarios) le gana a su nulo estratificado:** C1 p = 0,0045, así que no se para.
2. **P1 (Firth + FLIC) empata con P0:** D1 −0,015, IC pareado [−0,077; 0,045]. Se queda P0 y P2
   no corre.
3. **P0 no le gana al piso de producción:** +0,110 contra −ProductionDay, IC pareado
   [−0,005; 0,221]. **No se adopta nada.**

**Por qué el negativo es creíble y no mala suerte:**
- **La señal es chica.** D1 = 0,599 y D2 = 4,2%: al 5% de falsas alarmas detecta 2 o 3 autos de 55.
- **Es casi toda uso.** El piso −km/L da D1 = 0,581 y empata con P0 (IC [−0,038; 0,074]).
- **La normalización contra la flota no suma.** Sin normalizar, P1 da 0,611 (A5).
- **No hay fuga** (A0) y **el calendario no está detrás** (A3 baja D1; |ρ| con ProductionDay
  < 0,2).
- **En las filas que comparten, el finalista ordena mejor:** 0,592 contra 0,543 (A6, informativo).

**Qué se queda aunque el modelo no entre:**
- la ventana del registro y el reloj post-venta (la entrada de abajo): valen para cualquier
  panel que modele el *cuándo*;
- la infraestructura, que sirve para medir cualquier candidato futuro sobre hitos:
  - panel de hitos, `cure_mixture` y `FleetReferenceNormalizer`;
  - `landmark_metrics`: D1/D2 con entrada tardía y bootstrap pareado por vehículo;
  - `extend_splits`, que conserva folds para comparar pareado;
  - `audit_cure.py`;
  - los bloques `eval.landmark`, `eval.legacy_blocks` y `eval.carry_columns`, y el hook
    `preprocessing`.

**Para el pitch:** "alertamos desde el primer mes post-venta" no se sostiene con estos datos (4%
al 5% de falsas alarmas). Lo que sí se puede decir es que el riesgo se ordena por días desde la
venta y que el registro solo ve una ventana de seis meses.

---

## 2026-09-22 · El registro de eventos tiene ventana de calendario: la censura de un sano es su exposición dentro de ella, y el reloj arranca en la venta

**Estado: confirmada en el punto de control 1 (22-09).** Es la Fase 1 del cure model. Se
confirmó también tratar los autos quietos sin tocar las features y reportarlos aparte. Lo que
sigue está en [f3-preregistro-cure.md](f3-preregistro-cure.md). No cambia el finalista ni el
panel v1. Evidencia y comando:
[f3-reloj-y-ventana-del-evento.md](f3-reloj-y-ventana-del-evento.md), con
`python scripts/audit_event_clock.py --config configs/data/event_clock.yaml`. Solo dev.

**Qué se encontró:**
- **Los eventos de dev caen todos entre el 03-09-2025 y el 11-03-2026.** Afuera de esa ventana
  hay 100–180 autos por mes en la edad típica del evento y ningún evento. Es el borde de la
  extracción, no física.
- **El evento se ordena por días, no por km.** sd(log) del momento del evento: 0,24 en edad
  contra 0,97 en odómetro.
- **Dentro de la ventana, el riesgo lo explican los días desde la venta.** No lo explican ni la
  edad ni el calendario (Poisson sobre autos-día, LR p = 0,014).
- **No hay trayectoria previa al evento.** La señal es un rasgo temprano del vehículo: viajes
  fríos contra la flota, AUC 0,65–0,68 desde los primeros 30 días post-venta, nulo
  estratificado p = 0,0005. Es débil en la cola, y a 30 días la cola son autos quietos.

**Qué se propone:**
1. **La censura de un sano es el fin de su exposición dentro de `event_window`**
   (`configs/data/event_clock.yaml`), no su último viaje. Un sano sin exposición en la ventana
   no es un negativo.
2. **El reloj del riesgo arranca en la venta.** Todo panel que modele el *cuándo* lo mide en
   días post-venta.
3. Las dos cosas son la base del panel de hitos post-venta del cure model (preregistro en la
   Fase 2).

**Aviso para el panel v1 (no se corrige acá).** De las 1.062 filas sanas de dev, el horizonte
`[c+G, c+G+H]` está:
- entero dentro de la ventana en el 71,8%;
- saliéndose después del 11-03-2026 en el 17,9%;
- entero después en el 1,6%;
- empezando antes del 01-09-2025 en el 8,7%.

El PR-AUC por fila de `results/` contó como negativos verificados horizontes que el registro no
cubría del todo. Eso no invalida las corridas, porque todas comparten el mismo panel y los mismos
folds, pero es otra razón para que el sesgo de `ProductionDay` apareciera: los producidos tarde
casi no estuvieron expuestos.

**Preguntas para Ford o la mentora** (§10 del doc):
- qué es `IdentificationDate`;
- cuál es la ventana real de extracción;
- si se corrigió el error de SQL de la query de eventos;
- cuál es la prevalencia real;
- si hay fechas de taller;
- si `daysUntilSale` es la entrega al cliente: el 26% de los vehículos que llegan al hito de
  30 días recorrió menos de 100 km desde la "venta".

---

## 2026-09-21 · El desvío respecto de la historia del vehículo no entra; la (b) del finalista con R=3 marca +0,049

**Alcance:** cierra la primera mitad del punto 1 de "Qué queda abierto" (20-09) y le pone
número R=3 al punto 4. No cambia el finalista.

**El desvío no suma.** Se agregaron a survival stacking `feat_regenerations_per_1000km_hist_delta`
y `feat_dpf_end_mean_hist_delta` (historia previa hasta 25.600 km menos ventana, sin
TimesFM; ROC univariado 0,598 y 0,579, que reproduce el 0,602 del doc de TimesFM). El
criterio se declaró antes de correr: (a′) ± desvío. Dio +0,0219 ± 0,0047 contra
+0,0162 ± 0,0033. La mejora es de **0,98× el desvío combinado**, pareado +0,0129 / −0,0001 /
+0,0041. La detección sube dentro del ruido (20,1 ± 5,4% contra 15,7 ± 0,9%), anticipando
menos. La anticipación pierde su estabilidad: **±1.987 km contra ±22**, que era uno de los
dos argumentos del §4 de abajo. El lift por vehículo no se mueve (1,64× contra 1,62×). Así
que **no entra al set base**.

**La (b) del finalista, con R=3: +0,0493 de ROC** (0,602 → 0,652 al darle las `aux_` de
calendario), contra el +0,0230 del registro R=1 que cita el §4. El control re-corrido para
esta comparación reproduce el registro dígito a dígito, así que la diferencia es de R, no
de build. **Sigue siendo el pendiente antes de fijar el finalista, y ahora con un número
peor.**

**Por qué importa para lo que sigue:** con el calendario disponible, el modelo con desvío
y el que no lo tiene terminan en el mismo ROC (0,656 contra 0,652). Buena parte de la
señal que suman features de "contexto largo" se superpone con el calendario. Toda feature
futura de este tipo necesita la (b) con R=3 al lado, no con R=1.

Evidencia, comandos y el detalle del rebuild del panel (el local no traía
`aux_km_observed_after_cut`): [f3-desvio-historia.md](f3-desvio-historia.md).

---

## 2026-09-20 · Cierre de F3: el PR-AUC por fila mide *qué auto*, no *cuándo*. El finalista es survival stacking, elegido por (a′) y estabilidad

**Alcance: decisión de proyecto, no de una rama.** Consolida las tres ramas con las que
cerró F3 —`exp/gamma-degradation`, `exp/survival-stacking`, `exp/ordinal-horizon`— y
cambia el criterio con el que se lee todo `results/`, la meta que declara el pitch y qué
se considera una auditoría aprobada. Las fichas por rama siguen siendo la evidencia
detallada; esta entrada es la lectura conjunta, que ninguna de las tres podía hacer sola.

**Los números traen el build del panel** (`2026-09-19` = antes de 1eab4a1, umbral de
regeneraciones de 5 puntos; `2026-09-20` = después, con 15). No se mezclan: el mismo
control da 0,1653 en uno y 0,1612 en el otro, más que lo que separa filas vecinas de la
tabla. Ver `configs/data/panel_builds.yaml` y el preámbulo de `results/README.md`.

### 1 · El techo de cohorte: la meta declarada se alcanza sin anticipar nada

En dev hay 2.029 filas y 254 positivas (tasa base **0,1252**; por vehículo, **0,3099**).
Las 254 están **todas** dentro de los 967 cortes de los 53 vehículos con evento, así que
un identificador de vehículo perfecto —marcar todos los cortes de los fallados, sin
ninguna información del *cuándo*— saca **PR-AUC 0,2627, lift 2,098×**
(`src/eval/metrics.py::cohort_ceiling`, que toda corrida reporta).

La consecuencia es dura y hay que decirla en el pitch: **el objetivo declarado del
proyecto —1,6–2× de lift, `docs/f3-modelos-candidatos.md` §0— cae entero dentro de lo
alcanzable sin anticipar nada**, porque el techo ya está arriba. Ninguna corrida medida
contra esta tasa base lo supera: la mejor es survival stacking con 0,1721. Así que **el PR-AUC por
fila de este panel mide sobre todo *qué auto*, y muy poco *cuándo*.** Sigue siendo el
número que ordena la tabla, pero dejó de ser el número que elige el modelo (§4).

### 2 · La auditoría (a) estaba mal planteada, y por eso la aprobaban todos

El plan pedía permutar `label` **dentro de cada vehículo** y esperar que el PR-AUC
cayera a la tasa base. No cae: **sube**. La razón es de construcción, no de
implementación: permutar adentro del vehículo conserva *qué* vehículos fallan, y como un
vehículo sano no tiene ninguna fila positiva, permutar lo deja con cero positivas — la
etiqueta permutada sigue siendo perfectamente predecible desde la cohorte. Encima le
saca a cada auto el ruido de qué ventana le tocó, así que entrena un ordenador de
vehículos *mejor* que el original.

Medido (panel `2026-09-19`, tres semillas de folds):

| PR-AUC | control binario | ordinal restringido | ordinal extendido |
|---|---:|---:|---:|
| modelo | 0,1653 | 0,1524 | 0,1567 |
| nulo intra-vehículo (cohorte viva) | **0,1928** | **0,1867** | **0,1784** |
| nulo global (cohorte destruida) | 0,1259 | 0,1237 | 0,1250 |
| tasa base | 0,1252 | 0,1252 | 0,1252 |

Los tres nulos intra-vehículo quedan **por encima de su propio modelo**. El nulo global,
en cambio, cae exacto a la tasa base en las tres (+0,0007 / −0,0015 / −0,0002): **el
pipeline está limpio, no hay leakage** — splits agrupados, preprocesamiento por fold y
gap de blanking hacen lo que dicen.

Por eso **(a) pasa a informativa**: no se marca pass ni falla, se lee junto a (a′). Los
criterios que aprueban ahora son **(a0)** —permutar features entre todas las filas, tiene
que caer a la tasa base— y **(a′)**. Y el piso contra el que se mide un PR-AUC por fila
es el techo de cohorte, no la tasa base. Cualquier informe anterior que cite (a) como
aprobación hay que rehacerlo.

### 3 · Tres rutas independientes llegan al mismo lugar

El timing intra-vehículo **no se aprende con features de ventana agregadas y modelos
tipo GBM**. Tres mediciones que no comparten método lo dicen:

- **MIL (eje de decisión).** Colapsar los scores del LightGBM al promedio por vehículo
  *sube* el lift: 1,32 → 1,51 en el panel `2026-09-19`, y 1,29 → 1,54 en el
  `2026-09-20`. Tirar el orden de los cortes mejora la decisión.
- **(a′) (aporte del cuándo).** Para el control con R=3, **−0,0055 ± 0,0040**, negativo
  en las tres pasadas: el orden dentro del vehículo le *resta*.
- **Nulo intra-vehículo.** 0,1928 contra 0,1653 del modelo (§2).

Que tres caminos distintos —cambiar la unidad de decisión, colapsar el score sin
reentrenar, y destruir el timing conservando la cohorte— den la misma respuesta es **la
evidencia más fuerte que tiene el proyecto**. No es un artefacto de una métrica.

**Ojo con el (a′) del control, que cambia de signo con el build del panel.** Corrido
sobre el `2026-09-19` da **+0,0065** (y `scripts/audit_model.py` lo marca APROBADA);
sobre el `2026-09-20`, **−0,0067** con R=1 y **−0,0055 ± 0,0040** con R=3. No es una
contradicción: es la medida de cuán chico es el efecto —±0,007 alrededor de cero,
sensible a 7 columnas de regeneración— y la razón de que **(a′) se declare con CV
repetida y no con una pasada**. El número que vale es el de R=3 sobre el panel
corregido. Vale también como aviso: si alguien corre la auditoría en un checkout cuyo
`panel.parquet` es el viejo, va a ver un veredicto distinto al de esta entrada, y el
build es lo primero que hay que mirar.

Las tres patas tampoco están medidas sobre el mismo build, y conviene saber cuál es
cuál: **MIL aguanta en los dos** (1,32 → 1,51 en el `2026-09-19` y 1,29 → 1,54 en el
`2026-09-20`), el **nulo intra-vehículo está medido solo sobre el `2026-09-19`** y el
**(a′) del control, sobre el `2026-09-20` con R=3**. Las tres apuntan al mismo lado, y
de las tres la única que un cambio de build podría dar vuelta es la del medio.

### 4 · La excepción, y es el finalista: survival stacking

Es el **único modelo del repo cuyo "cuándo" es positivo y sobrevive a R=3** (todo lo de
abajo, panel `2026-09-20`, `f3-survival-stacking-r3` contra `f3-lgbm-panel-v1-r3-regen15`):

| R=3 | survival stacking | control binario |
|---|---:|---:|
| **(a′) aporte del cuándo** | **+0,0162 ± 0,0033** (positivo en las 3) | −0,0055 ± 0,0040 (negativo en las 3) |
| PR-AUC por fila | 0,1721 ± 0,0135 | 0,1595 ± 0,0081 |
| Brier | **0,1123** | 0,1711 |
| lift por vehículo (`mean`) | **1,616** | 1,494 |
| detección @ ≤50 FA/1000 | 15,7% ± 0,9 | 11,9% ± 3,2 |
| anticipación mediana | 8.330 km **± 22** | 9.370 km **± 2.237** |

La diferencia de (a′) es **0,0217, unas 4× el desvío combinado (0,0052)**, y los dos
rangos caen de lados opuestos del cero. **No gana en PR-AUC por fila** —0,1721 ± 0,0135
contra 0,1595 ± 0,0081 se solapan— ni llega al techo de cohorte de 0,2627. Gana en lo
que el producto vende:

- **Calibración.** Brier 0,1123 contra 0,1711, con riesgo medio predicho 0,133 contra
  0,125 real. El hazard se entrena sin `class_weight`, así que `1 − S(H|x)` es una
  probabilidad; el LightGBM con `class_weight="balanced"` predice 0,314 de media. La
  diferencia entre "este auto tiene 13% de riesgo" y un score ordinal es la diferencia
  entre poder priorizar un taller y no.
- **Estabilidad.** ±22 km de anticipación entre repeticiones contra ±2.237 km del
  control. El control cambia 5.400 km de anticipación según qué folds le toquen: su
  "12.322 km" del R=1 era el sorteo.
- **Eje de vehículo.** Lift 1,616 contra 1,494.

**La recomendación que queda escrita: el finalista se elige por (a′) y por estabilidad,
no por PR-AUC por fila.** El motivo está en §1 — un PR-AUC por fila por debajo de 0,2627
no demuestra anticipación, así que ordenar candidatos por él es ordenarlos por cuán bien
identifican la cohorte de muestreo. Pendiente antes de fijarlo: la auditoría (b) le
marca atajo (+0,0230 de ROC al darle las `aux_` de calendario, por encima del umbral de
0,02; el control da +0,0171 y no marca).

Un número que no hay que sobreleer: el ordinal extendido re-corrido contra el panel
`2026-09-20` da (a′) **+0,0118**, el único positivo fuera de survival stacking. Está
medido con **R=1 y nada más**, así que no es comparable con el +0,0162 ± 0,0033
confirmado con R=3: queda como pista sin medir, no como resultado.

### 5 · Dos trampas de exposición, opuestas, con la misma raíz

Las dos ya aparecieron disfrazadas de señal fuerte, en direcciones contrarias, y van a
volver porque la raíz —cuánto historial hay por vehículo— no se arregló:

- **En los crudos, el evento corta el historial del fallado.** Span mediano **7.439 km**
  contra **15.508 km** de los sanos. Produjo un AUC de **0,694** en la pendiente de la
  distancia entre regeneraciones que, al igualar la ventana de odómetro, cae a
  **0,564**: medía cuánto duró el registro. Apuntaba además al revés que la hipótesis.
- **En el panel, el emparejado ralea al sano.** Los fallados conservan todos sus cortes
  (**18,25** por bolsa, span medio 9.302 km) y a los sanos el emparejado por odómetro ×
  mes les deja solo los que llenan cada celda (**9,00** por bolsa, span medio 7.763 km;
  1.241 filas sanas de 7.308). El tamaño de bolsa **solo**, como score, da lift 1,79×.
  Produjo el lift **1,83×** de `noisy_or`, que pierde contra un nulo que conserva los
  tamaños de bolsa.

**La regla que sale de esto: toda métrica que dependa del tamaño de la bolsa o de la
longitud del historial se compara contra un nulo que conserve esa magnitud, nunca contra
la tasa base.** `scripts/audit_mil_bagsize.py` es la implementación para el eje de
vehículo; `mean` es la única agregación insensible al tamaño (corr. de rango con
`n_cuts` = 0,04, contra 0,85 de `noisy_or`), y por eso es la que se reporta.

### 6 · Gamma: descartado con motivo físico, no por no funcionar

No hay **carga irreversible medible en esta flota**, que es la premisa que el proceso
gamma necesita. La monotonía aparente es el asentamiento de los primeros ~1.000 km:
ρ(nivel, odómetro) **+0,457** en 0–6.000 km, pero **−0,036** en 1.000–8.000 y **−0,009**
en 4.000–16.000 (45–47% de vehículos con ρ > 0, que es lo que da una moneda). El proxy
directo de ceniza —el residuo con el que termina la regeneración— no muestra nada
(ρ mediano **+0,013**, p = 0,52) y tiene un problema de resolución: **el 51% de las
14.695 regeneraciones termina en 0 exacto**.

La razón de fondo es de escala de tiempo, y por eso el resultado no depende del modelo:
la ceniza en un DPF es un fenómeno de **>100.000 km**, y acá el evento cae a una
**mediana de 7.987 km**, con **7 vehículos de 290** pasando los 50.000. La rama paró en
el paso de verificación de la premisa, a propósito, y no se escribió el modelo.

**Pregunta abierta, y no la contesta esta entrada: qué es el evento a 8.000 km si no es
ceniza.** Condiciona si falta una familia de features entera. La auditoría
(`scripts/audit_gamma_monotonia.py`) es ejecutable y su criterio de veredicto está
declarado en el YAML, así que una extracción con vehículos más viejos —o con
contrapresión diferencial del filtro, que mide la capacidad perdida directo— vuelve a
contestar la pregunta sin rehacer nada.

### 7 · Costo: el modelo paga a ~7×, y es reporte, nunca criterio

El barrido de `C_FN/C_FP` tiene su pico en **7×** en las tres corridas medidas, y ese 7
no es empírico: es la aritmética. Con `p = 0,125`, alertar siempre le gana a no alertar
nunca en cuanto `C_FN/C_FP > (1−p)/p = 0,875/0,125 = 7`. Por encima de ahí **"revisar
todo" gana sin importar el modelo**, así que la matriz 20–50× de SCANIA es degenerada en
este panel: deja **1,4–1,8%** de ahorro. El ahorro máximo es **~14% en el pico**
(14,6% / 14,1% / 13,7%) y se desarma rápido: 1,7–3,5% a 10×.

**Frase para el pitch:** el modelo paga cuando una degradación no detectada cuesta unas
**7× una inspección innecesaria**. Es reporte, y **nunca** criterio de selección: el
punto de quiebre lo fija la tasa base, no el modelo.

### Qué queda abierto

1. **La formulación intra-vehículo**, que §2 y §3 vuelven urgente: features medidas como
   **desvío respecto de la historia previa del mismo vehículo** —la pista que ya había
   dejado TimesFM y que el panel v1 no tiene— y un target entrenado **solo sobre los
   vehículos que fallan**, que aísla el horizonte de la cohorte en vez de mezclarlos.
2. **Reemparejar el panel a nivel vehículo** para que las bolsas sean simétricas (§5).
   Es Track A.
3. **Qué es el evento a 8.000 km** (§6).
4. **La auditoría (b) de survival stacking** antes de fijarlo como final (§4).

Lo que **no** queda pendiente: afinar bins ordinales (las dos variantes acotan el rango y
ninguna mueve la aguja, en ninguno de los dos builds del panel), el efecto aleatorio por
vehículo (GPBoost queda 0,018 de PR-AUC por debajo del apilado simple) y el proceso gamma.

**Detalle por rama:** [f3-survival-stacking.md](f3-survival-stacking.md) ·
[f3-ordinal-horizonte.md](f3-ordinal-horizonte.md) ·
[f3-proceso-gamma.md](f3-proceso-gamma.md) ·
[f3-mil-agregacion-vehiculo.md](f3-mil-agregacion-vehiculo.md) ·
[f3-timesfm-zeroshot.md](f3-timesfm-zeroshot.md) ·
[f3-cnn-lstm-tutora.md](f3-cnn-lstm-tutora.md).

**Cómo se reproduce:**

```bash
# el techo de cohorte, (a') y el barrido de costo salen de cualquier corrida
python scripts/train.py --config configs/exp_lgbm_panel_v1_mil.yaml
python scripts/audit_model.py --config configs/exp_survival_stacking.yaml   # (a0) y (a')
python scripts/audit_ordinal_horizon.py --config configs/exp_lgbm_panel_v1.yaml  # los dos nulos
python scripts/audit_mil_bagsize.py f3-lgbm-panel-v1-mil                    # el nulo de tamaño de bolsa
python scripts/audit_gamma_monotonia.py --config configs/data/gamma_monotonia.yaml
python scripts/results.py table --out results/README.md                     # la tabla, con el build de cada corrida
```

Las corridas contra el panel `2026-09-20` necesitan
`FORD_DATA_DIR=<dir con el panel reconstruido>`; sin red, `WANDB_MODE=disabled`.

---

## 2026-09-20 · El piso no es la tasa base: es el techo de cohorte (0,2627). PROPUESTA para todo el proyecto

**Código:** `src/eval/metrics.py::cohort_ceiling / pr_auc_within_failed /
when_contribution`, reportadas por `scripts/train.py` en toda corrida y por
`scripts/audit_model.py`. **Alcance: es una decisión de proyecto, no de esta rama.**
Cambia cómo se lee cada número de `results/`, el criterio de las auditorías y la meta
que el pitch declara. Está escrita como propuesta para que el equipo la mire y la
discuta; lo que ya está hecho es medirla y dejarla reportada.

### El hallazgo

En el panel dev hay 2.029 filas y 254 positivas (tasa base 0,1252). **Las 254 están
todas dentro de los 967 cortes de los 53 vehículos que fallan.** Entonces un modelo que
solo sabe "este auto es de los que fallan", sin la menor idea de *cuándo*, puntúa esas
967 filas por encima de las otras 1.062 y saca precisión 254/967 = **PR-AUC 0,2627, lift
2,10×**. Eso es el **techo de cohorte**.

```bash
python -c "
import pandas as pd, sys; sys.path.insert(0,'.')
from src.config import resolve_path
from src.eval.splits import load_test_split, test_split_masks
from src.eval.metrics import cohort_ceiling
panel = pd.read_parquet(resolve_path('data/processed/panel.parquet'))
dev, _ = test_split_masks(panel, load_test_split(resolve_path('data/processed/test_split.json')))
print(cohort_ceiling(panel.loc[dev, 'label'], panel.loc[dev, 'vehicle_id']))"
# pr_auc 0.2627 · lift 2.098 · n_failed_rows 967 · n_positive 254
```

**Consecuencia incómoda: el objetivo del plan (1,6–2× de lift) está por debajo del
techo.** O sea que se alcanza sin anticipar nunca. Los modelos medidos están todos
*abajo* del techo —el control da 0,1612 (1,29×)— así que hoy el problema no es que
alguien haya inflado un número: es que el número que veníamos mirando no contesta la
pregunta del producto.

### La propuesta

1. **Reportar siempre la descomposición**, no un PR-AUC suelto. Tres números, que
   `train.py` ya guarda en `metrics.json` y `compare.py` ya muestra:
   - **techo de cohorte** (cuánto sale de *qué* auto), con `✓`/`✗` según el PR-AUC por
     fila lo supere;
   - **PR-AUC entre fallados** (`pr_auc_within_failed`), medido solo sobre esos 967
     cortes, donde el nivel del vehículo ya no ordena nada. Su azar es 0,2627, **no**
     0,1252: se lee el lift, nunca el número pelado;
   - **(a') el aporte del *cuándo***: colapsar el score al promedio de su vehículo sin
     reentrenar. La caída es lo que el modelo sabía de timing.
2. **(a) pasa a informativa.** La auditoría de
   [f3-modelos-candidatos.md](../f3-modelos-candidatos.md) §0.4 —permutar dentro del
   vehículo esperando que el PR-AUC caiga a la tasa base— **no puede dar nulo**: deja
   intacto *qué* vehículos fallan, que es de donde sale casi todo el PR-AUC, y encima le
   saca a cada auto el ruido de qué ventana le tocó, así que entrena un ordenador de
   vehículos **mejor**. Sube a 0,17–0,21 contra 0,161 de referencia, con tres semillas y
   los dos modelos. `audit_model.py` la imprime y **no la marca ni como pass ni como
   falla**. Aprueban **(a0)** (el null de verdad) y **(a')**.
3. **Un lift dentro de 1,6–2× no demuestra anticipación.** Si el pitch lo usa, tiene que
   ir acompañado del techo, o se está vendiendo "sabemos qué autos fallan" como "sabemos
   cuándo". La curva de anticipación vs. falsas alarmas sigue siendo el número de
   portada, y es el que no tiene este problema.

### Qué cambia en la lectura de lo ya medido

| corrida (panel post-1eab4a1) | PR-AUC fila | vs. techo | (a') |
|---|---|---|---|
| `f3-gpboost-survival` | 0,1431 | ✗ | **+0,0562** |
| `f3-survival-stacking` | 0,1613 | ✗ | +0,0152 |
| `f3-cnn-lstm-tutora-regen15` | 0,1553 | ✗ | +0,0036 |
| `f3-baserate-panel-v1-regen15` | 0,1214 | ✗ | +0,0010 |
| `f3-lgbm-panel-v1-regen15` (control) | 0,1612 | ✗ | **−0,0067** |

Dos lecturas que el PR-AUC por fila escondía:

- **El control ordena los cortes al revés.** Colapsarle el score al promedio del
  vehículo le *sube* el PR-AUC de 0,1612 a 0,1679. Con R=3: −0,0055 ± 0,0040, negativo
  en las tres pasadas. El mejor modelo de la tabla no sabe *cuándo*.
- **`gpboost_survival` tiene el PR-AUC por fila más bajo de los finalistas y es por lejos
  el que más sabe del *cuándo*** ((a') +0,0562, ROC intra-vehículo 0,8582 contra ~0,595
  de los otros dos). El efecto aleatorio por vehículo se come el "qué auto" y deja el
  score midiendo casi solo el "cuándo" — que es justo lo que el PR-AUC por fila penaliza.
  **Si la propuesta se acepta, este modelo hay que volver a mirarlo en serio.**

### Lo que la CV repetida contesta

Con `splits.n_repeats: 3` (`configs/exp_lgbm_panel_v1_r3_regen15.yaml` y
`configs/exp_survival_stacking_r3.yaml`), y la dispersión entre repeticiones:

| | control R=3 | survival stacking R=3 |
|---|---|---|
| PR-AUC por fila | 0,1595 ± 0,0081 | 0,1721 ± 0,0135 |
| **(a')** | **−0,0055 ± 0,0040** | **+0,0162 ± 0,0033** |
| lift entre fallados | 1,093 ± 0,034 | 1,103 ± 0,066 |
| detección | 11,9 % ± 3,2 | 15,7 % ± 0,9 |
| anticipación mediana | 9.370 ± 2.237 km | 8.330 ± 22 km |

**Sí: el +0,015 supera la dispersión.** Los rangos de (a') son disjuntos
(`[−0,0097, −0,0000]` contra `[+0,0128, +0,0207]`) y caen de lados opuestos del cero; la
diferencia de 0,0217 es ~4× el desvío combinado (0,0052). Y es **lo único** que los
separa: el PR-AUC por fila no (rangos solapados) y la detección tampoco del todo (se
tocan en 15,1 %). Lo que sí cambia mucho es la estabilidad: ±22 km de anticipación
contra ±2.237.

### Límites conocidos

- **(a') aprueba a un modelo constante.** `baserate` da +0,0010 y "pasa". (a') dice si el
  *cuándo* que hay es real, no si alcanza; el tamaño se lee contra el techo. Si el equipo
  quiere un criterio con umbral, hay que fijarlo con la dispersión de R=3, no a ojo.
- **(a') y el PR-AUC entre fallados pueden contradecirse.** Control 1,09× contra survival
  stacking 1,05× entre fallados, al revés que (a'). Son dos cortes distintos de la misma
  pregunta: se reportan los dos y no se elige el que conviene.
- **El techo es del panel, no de la flota.** 0,2627 sale de que los sanos están
  emparejados por odómetro y mes (`sampling` en `configs/data/panel_v1.yaml`). Con otra
  prevalencia el techo se mueve; por eso se reporta junto al número, no como constante.
- **`survival_stacking` marca atajo en (b)**: +0,0230 de ROC con las `aux_` de calendario,
  por encima del umbral de 0,02 (el control da +0,0171 y no marca). **Sin resolver.**

**Detalle:** [f3-survival-stacking.md](f3-survival-stacking.md),
[f3-cnn-lstm-tutora.md](f3-cnn-lstm-tutora.md) (sección de corrección).

---

## 2026-09-20 · Con qué se entrena se declara en el YAML (`target:`), separado de con qué se mide

**Código:** `src/training/targets.py`, hook en `src/training/cv.py`.

**Qué.** Un bloque `target: {name, params}` en el YAML del experimento elige el objetivo
de entrenamiento entre los modos registrados en `src/training/targets.py`. Sin la clave
se entrena con `label`, que es lo que hacía siempre. **La evaluación no cambia nunca**:
etiqueta dura y mismos folds, pase lo que pase.

**Por qué.** Media docena de ideas de F3 (supervivencia, etiquetas blandas, horizonte
ordinal, ranking dentro de celdas) quieren entrenar contra algo que no es el 0/1 de la
ventana, y todas empiezan igual: tocando `cv.py`. Con un registro por nombre, agregar un
modo es una función en un archivo y una línea en un YAML; `cv.py` no sabe qué modos
existen. Dos ramas pueden agregar el suyo sin pisarse, que es el motivo concreto: esto
salió junto con `exp/ordinal-horizon`.

**Lo que no se negoció:** que la métrica de selección siga siendo el PR-AUC sobre `label`
con los folds congelados. Si cada modo midiera contra su propio objetivo, la tabla de
`results/` dejaría de significar algo.

---

## 2026-09-20 · El panel emite `aux_km_observed_after_cut`

**Código:** `src/data/panel.py::vehicle_cuts`.

**Qué.** Una columna más, `last_odo − c`: los km que el vehículo se observó después del
corte. `aux_`, porque es información posterior al corte y no entra a ninguna ventana.

**Por qué.** `time_to_event_km` es NaN en los sanos, así que un censurado no se
distinguía de un negativo: el panel decía "no falló" cuando el dato es "llegó hasta acá
sin fallar". Cualquier modelo de supervivencia necesita esa segunda frase. Sale gratis de
lo que `vehicle_cuts` ya tenía.

**Costo.** Hay que regenerar el panel, pero es estrictamente aditivo: mismas filas, mismo
orden, misma huella, mismos folds (verificado). Los `splits.json` congelados siguen
sirviendo.

---

## 2026-09-20 · La permutación dentro del vehículo no es un null: se lee con (a0) y (a')

**Código:** `scripts/audit_model.py`.

**Qué.** La auditoría (a) de [f3-modelos-candidatos.md](../f3-modelos-candidatos.md) §0.4
—permutar dentro de cada vehículo y esperar que el PR-AUC caiga a la tasa base— **sube**
en este panel, con los dos modelos probados y con tres semillas. Se agregan dos
auditorías al lado: **(a0)** permutar entre *todas* las filas, que sí es el null y sí cae
a la tasa base; y **(a')** colapsar el score al promedio de cada vehículo sin reentrenar,
que es la cuenta de cuánto aporta el *cuándo*. Además se permutan las **features** y no
la etiqueta, para que las dos corridas se midan contra exactamente el mismo `label`.

**Por qué.** La permutación intra-vehículo deja intacto *qué* vehículos fallan —de donde
sale casi todo el PR-AUC de este panel— y de paso le saca a cada vehículo el ruido de qué
ventana le tocó. O sea que entrena un ordenador de vehículos **mejor**, no un modelo sin
información. Leída sola, la auditoría dice lo contrario de lo que pasa.

**Alcance.** Vale para todo F3, no solo para la supervivencia. La ficha de
[f3-cnn-lstm-tutora.md](f3-cnn-lstm-tutora.md) lee su 0,143-contra-0,153 con el criterio
viejo; hay que rehacerlo con `scripts/audit_model.py`.

**Detalle:** [f3-survival-stacking.md](f3-survival-stacking.md).

---

## 2026-09-20 · El proceso gamma de degradación no se implementa

**Código:** `scripts/audit_gamma_monotonia.py` + `configs/data/gamma_monotonia.yaml`
(la auditoría queda; el modelo no se escribió). **Rama:** `exp/gamma-degradation`.

**Qué.** Se descarta modelar la degradación como un proceso gamma con covariables
(Lawless & Crowder 2004): daño latente monótono, incrementos gamma, falla = primer
cruce de un umbral. El paso previo —verificar el proxy de carga— dice que no hay
premisa, y la rama se cierra ahí.

**Por qué.** Las dos condiciones que el modelo necesita fallan sobre dev. (1) La
**monotonía** existe solo hasta el km ~1.000: fuera del asentamiento, el 45–47% de los
vehículos tiene ρ(nivel, odómetro) > 0, que es lo que da una moneda. El residuo con el
que termina la regeneración —la ceniza, según la física— no muestra tendencia ni
siquiera sin recortar (ρ mediano +0,013), y la mitad de las 14.695 regeneraciones
termina en 0 exacto: el piso cae debajo de la resolución de la escala. (2) La
**separación** de la tasa no se sostiene: con la exposición igualada ningún proxy pasa
de 0,52 de AUC en la dirección de la hipótesis. El único efecto fuerte —la distancia
entre regeneraciones, AUC 0,694— apunta al revés que la hipótesis y se derrumba a 0,564
al igualar la ventana de odómetro: medía cuánto duró el registro. La causa de fondo es
de escala de tiempo: la ceniza es un fenómeno de >100.000 km y acá el evento cae a una
mediana de 7.987 km, con 7 vehículos de 290 arriba de los 50.000.

**Qué queda abierto:** la auditoría es ejecutable y su criterio de veredicto está
declarado en el YAML, así que una extracción con vehículos más viejos —o con
contrapresión diferencial del filtro, que mide la capacidad perdida directo— vuelve a
contestar la pregunta sin rehacer nada.

**Detalle:** [f3-proceso-gamma.md](f3-proceso-gamma.md)

---

## 2026-09-20 · Ninguna variante ordinal reemplaza al binario, y el costo se reporta como barrido

**Código:** `src/training/targets.py` (registro de targets por nombre),
`src/eval/metrics.py::cost_ratio_sweep`. **Auditoría:** `scripts/audit_ordinal_horizon.py`.
**Configs:** `exp_ordinal_horizon.yaml`, `exp_ordinal_horizon_ext.yaml`.

**Qué se probó: dos variantes de bins, no una.**

1. **Bins restringidos** `[1250, 2000, 2750, 3500]`, cuatro buckets adentro de
   `[G, G+H]`. **Defectuosa por diseño**, y el propio doc lo verificaba sin leerlo:
   `clase > 0 ⇔ label == 1`. El target era un refinamiento estricto adentro de la clase
   positiva, así que no aportaba nada sobre las 1.775 filas negativas (87% del panel) y
   solo pagaba el costo de varianza de partir 254 positivas en cuatro baldes de ~60.
   PR-AUC 0,152 contra 0,165 del control, Brier 0,207 contra 0,173: la firma exacta de
   más varianza sin más señal.
2. **Bins extendidos** `[2000, 3500, 6000, 10000]`, la formulación de SCANIA donde la
   clase 0 es "lejos del fallo" y no "sano". 394 filas que hoy son `label = 0` pero
   vienen de vehículos que fallan pasan a las clases intermedias: ésa es la información
   nueva. PR-AUC 0,157, el **mejor Brier de las tres corridas** (0,152, por debajo del
   control) y clases parejas (mínimo por fold de validación: 21 filas contra 9).

**Resultado: ninguna gana, y se miran los dos ejes.** Extender los bins acerca la tarea
auxiliar a "identificar la cohorte de muestreo", que en este panel *es* la etiqueta, y
ese modo de fallar empeora el PR-AUC por fila mientras mejora el ranking por vehículo.
Por eso se reportan los dos. Por fila: control 0,165 > extendido 0,157 > restringido
0,152. Por vehículo con `mean`: control 1,51× > restringido 1,42× > extendido 1,31×.
**El modo de fallar temido no ocurrió** —el eje de vehículo bajó, no subió—, pero
tampoco hay ganancia: el control les gana en los dos ejes. La selección sigue siendo
PR-AUC OOF contra `label` y sigue eligiendo el binario.

**El hallazgo que sí cambia cosas, y que salió de la auditoría (a).** Permutando la
etiqueta **dentro de cada vehículo** —se conserva la cohorte, se destruye el orden
interno— y reentrenando, el nulo queda muy por encima de la tasa base, y **ni el
ordinal extendido ni el control binario le ganan a su propio nulo**. El nulo global
(cohorte destruida) sí cae exacto a la tasa base, así que el pipeline está limpio: el
problema no es fuga, es que el PR-AUC que tenemos es separación de vehículos y no
anticipación adentro del vehículo. Vale para el control tanto como para el ordinal, así
que **es del panel, no del target**. Detalle y números en el doc.

**El costo deja de ser una matriz.** La matriz 5×5 de SCANIA es degenerada a esta tasa
base: con `p = 0,125`, alertar siempre le gana a no alertar nunca en cuanto
`C_FN/C_FP > (1-p)/p ≈ 7`, así que con sus 20–50× la política óptima es revisar todo
gane quien gane, y el número mide la matriz y no el modelo. La reemplaza
`cost_ratio_sweep()`, que barre el ratio y reporta el punto de quiebre; es
model-agnóstica y sale de `eval.cost_ratios`. Las tres corridas dan el mismo perfil:
**el modelo paga alrededor de `C_FN/C_FP = 7×`** (ahorra 14% sobre revisar todo), por
debajo de 5× conviene no alertar a nadie y por encima de 10× conviene revisar todo,
donde ahorra menos del 3%. Es reporte, no criterio de selección.

**Qué queda pendiente.** (1) La formulación intra-vehículo que la auditoría (a) vuelve
urgente: features medidas como desvío respecto de la historia previa del **mismo**
vehículo —la pista que ya había dejado TimesFM y que el panel v1 no tiene— y, como
variante del target, un ordinal entrenado **solo sobre los vehículos que fallan**, que
aísla el horizonte de la cohorte en vez de mezclarlos. (2) Reemparejar el panel a nivel
vehículo para que las bolsas sean simétricas, que es Track A y ya estaba anotado en
[f3-mil-agregacion-vehiculo.md](f3-mil-agregacion-vehiculo.md). Lo que **no** queda
pendiente es afinar más los bins: las dos variantes de este experimento acotan el
rango y ninguna mueve la aguja.

**Detalle:** [f3-ordinal-horizonte.md](f3-ordinal-horizonte.md).

---

## 2026-09-20 · La decisión se mide por vehículo con `mean`, y siempre como lift

**Código:** `src/eval/metrics.py::vehicle_scores/vehicle_metrics`, bloque
`eval.vehicle_aggregation` en el YAML. **Auditoría:** `scripts/audit_mil_bagsize.py`.

**Qué.** Además del PR-AUC por fila (que sigue siendo el número de selección de modelo),
cada corrida reporta las mismas métricas con el vehículo como unidad de decisión: los
cortes de un vehículo son una bolsa (MIL) y se colapsan a un score único. De las cuatro
agregaciones implementadas, **la que se usa para leer un resultado es `mean`**, y se
reporta el **lift**, nunca el PR-AUC pelado.

**Por qué.** Ford marca autos, no cortes: el PR-AUC por fila contesta una pregunta que
nadie hace. Pero el eje de vehículo tiene dos trampas. (1) La tasa base cambia (0,310
contra 0,125), así que los PR-AUC de los dos ejes no se comparan entre sí. (2) En el
panel v1 la bolsa de los que fallan tiene el doble de cortes que la de los sanos —los
fallados conservan todos sus cortes, los sanos solo los que llenan las celdas del
emparejado—, y el tamaño de la bolsa **solo** ya da lift 1,79×. `max`, `topk` y
`noisy_or` no le ganan a su propio nulo de permutación (p = 0,19 / 0,42 / 0,15); `mean`
sí (1,51× contra 1,02×, p < 0,005), y es la única insensible al tamaño.

**Qué queda abierto:** las bolsas asimétricas son del muestreo del panel, no de los
vehículos. Emparejar a nivel vehículo (misma grilla de cortes para sanos y fallados) en
vez de a nivel fila haría comparables las cuatro agregaciones. Es Track A y cambia el
panel; hasta entonces, cualquier número de `noisy_or` se lee contra la auditoría.

**Detalle:** [f3-mil-agregacion-vehiculo.md](f3-mil-agregacion-vehiculo.md).

---

## 2026-09-20 · Una regeneración exige una caída mínima de 15 puntos

**Qué.** `regen_drop_points` pasa de 5 a 15 en la fuente de verdad
`configs/data/features_v1.yaml`. El panel agregado, el secuencial y los scripts de EDA
leen ese mismo valor; no quedan copias operativas del umbral viejo.

**Por qué.** Con 5 puntos, el saldo de las caídas no seguía la acumulación medida
(correlación 0,005). Con 15, la correlación mediana dentro del vehículo sube a 0,45.
El detector todavía cuenta ~2,5 veces más eventos que el marcador `Regeneration`, así
que 15 queda como umbral operativo y no como ground truth definitivo.

**Costo.** Hay que regenerar `panel-v1`, el panel secuencial y cualquier corrida que use
la familia B. Los resultados anteriores siguen documentando lo que dio el umbral de 5,
pero no se comparan como si fueran el mismo dataset.

**Detalle:** [f2-umbral-regeneraciones.md](f2-umbral-regeneraciones.md).

---

## 2026-09-20 · Los resultados de cada corrida se anotan en `results/`, versionados

**Código:** `scripts/results.py`. **Cómo se usa:** `results/README.md`.

**Qué.** Un `results/<corrida>.yaml` por corrida con las métricas resumen, la config
resuelta **completa** y la procedencia (rama, commit, si el YAML estaba versionado),
más una nota libre. La tabla se genera; los registros no se editan salvo la nota.

**Por qué.** `experiments/` y wandb no se versionan (regla 8), y varios YAML de
experimento vivían solo en otras ramas, en stashes o en `data/v364/cfg/`. Sin un
registro, la config del mejor modelo —la que hay que entregar— se pierde con la
máquina. Un archivo por corrida y no una tabla única: los tres tracks anotan en
paralelo y una tabla compartida se pisaría en cada merge.

**Ojo.** Comparar solo corridas con la misma tasa base (mismas filas): `tfm-full` y
`tfm-window` muestran lift ~3,8× con PR-AUC ~0,09 porque usan otras filas.

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

## 2026-09-19 · Los modelos secuenciales entran por un panel secuencial, no por un entrypoint aparte

**Decidió:** Octavio (pedido: la solución de la tutora como baseline).
**Código:** `src/features/sequences.py`, `scripts/build_seq_panel.py`,
`src/models/cnn_lstm.py`. **Config:** `configs/data/panel_seq_v1.yaml`,
`configs/exp_cnn_lstm.yaml`.

**Qué.** La secuencia de cada corte (ventana en bins de odómetro × canales) se aplana a
columnas `feat_seq_*` de un panel con **las mismas filas** que el panel v1, y el modelo
secuencial es un builder más del registry que reconstruye el tensor con la forma que
deja el builder en `panel_seq_v1_meta.json`. `scripts/train.py` y `src/training/cv.py`
no se tocaron.

**Por qué.** Un `train_seq.py` aparte habría duplicado el recorte a dev, los folds, las
métricas y el logueo, que son justo lo que garantiza que dos números sean comparables.
Así, el CNN-LSTM usa los mismos folds, el mismo holdout y el mismo preprocesado por fold
que cualquier otro modelo, y su número se pone al lado del LightGBM sin asteriscos.

**Costo.** El modelo depende de un orden de columnas (numéricas primero, en orden
`(t, canal)`); el builder verifica que las únicas `feat_*` sean la secuencia, y el
modelo que el ancho alcance. Un panel que mezcle `feat_seq_*` con agregados de ventana
rompería el reshape: si hace falta un híbrido, se cambia el contrato, no se fuerza.
PyTorch queda como dependencia opcional (`requirements-dl.txt`) hasta discutirlo.

**Detalle:** [f3-cnn-lstm-tutora.md](f3-cnn-lstm-tutora.md)

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
