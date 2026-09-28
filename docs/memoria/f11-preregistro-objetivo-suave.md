# Preregistro F11: GRU con etiqueta suave lejos del evento contra la GRU de hoy

**Fecha:** 2026-09-28 · **Fase:** F11 · **Rama:** `exp/gru-objetivo-suave`.

**Se escribió** después de explorar sobre dev con los folds de siempre (`splits_r3.json`) y las
semillas 42, 1 y 2, y **antes** de correr nada con los folds y las semillas de la confirmación.
El commit que agrega este archivo es la marca de tiempo. Lo que cambie después va en un commit
propio, anterior a la corrida que afecta, y dice por qué.

**Por qué existe.** El pedido del 27-09 es un modelo que le gane por bastante a la GRU + trips +
estática ×3 semillas (la mejor de [f9-remedicion-completa-v2.md](f9-remedicion-completa-v2.md) y
[f10-sweep-gru.md](f10-sweep-gru.md)), con significancia estadística y muy por encima del azar.
El presupuesto de comparaciones está agotado (CLAUDE.md): esta ficha gasta **una** comparación
nueva, cerrada, con un solo candidato y un solo criterio primario.

## 0 · Lo que ya se vio antes de escribir esto (exploración, dev, `splits_r3`, semillas 42/1/2)

Todo con la cuenta de `report_v2_models.py` (umbral exacto, `k = 2`) contra la GRU de hoy
×3 semillas. "Primario" es el promedio de la detección al 5, 10, 15 y 20% de falsas alarmas; la
diferencia va con su IC95 del bootstrap pareado por vehículo (400–500 réplicas).

| Variante explorada | Primario (puntos) | Lectura |
|---|---|---|
| Suavizar el score en el tiempo (media acumulada, EWMA, CUSUM, media con prior) | −2 a −7 | la detección vive en los picos cerca del evento |
| Ensambles por rango de las corridas de F9 | ≈ 0 al 5–10% | no suma |
| Estandarizar el score dentro de la celda + logit de la tasa de la celda | +4,1 [−0,1; 6,9] | recalibración; no suma encima de lo de abajo |
| Pérdida por vehículo (MIL, máximo suave del mínimo de 2 cortes) | +4,2 [−1,6; 9,4] | no suma encima de lo de abajo |
| Historia completa del auto como estáticas (`h_*`, 54 o 16) | −6 a 0 | no suma, tampoco con etiqueta suave |
| 11 canales más en la secuencia (temperaturas, aceite, combustible, reposo…) | −8 (−3 sin aceite) | empeora |
| **Etiqueta suave lejos del evento, τ = 0,05 / 0,10 / 0,15 / 0,20 / 0,30** | **+7,3 / +7,4 / +9,5 / +8,5 / +7,2** | meseta, p ≤ 0,02 en todas |
| Etiqueta suave con decaimiento (0,5·e^−d/5.000; 1·e^−d/10.000) | +6,7 / +5,6 | |
| Etiqueta de cohorte (τ = 1) | +3,4 [−5,6; 12,9] | demasiado |
| CNN-LSTM de la tutora con estática, τ = 0,3 | +5,6 [−1,8; 11,9] | |

- Con τ = 0,15 la exploración da **34,6 · 50,6 · 64,7 · 72,1% al 5 · 10 · 15 · 20%**, contra
  27,4 · 42,5 · 53,3 · 60,7% de la GRU de hoy. **Contra la celda mercado × motor** (18 · 33 · 48 ·
  62%) da +15,2 puntos de primario [4,8; 24,5]; la GRU de hoy, +5,7 [−4,0; 16,4].
- **No es calendario:** dentro de celda × mes del corte, el AUC por fila sube lejos del evento
  (0,583 → 0,608 entre 3.500 y 8.000 km; 0,540 → 0,557 a más de 8.000) y queda igual cerca (0,656
  → 0,667). La correlación del score con el mes entre los sanos, dentro de la celda, es 0,000
  (la GRU de hoy: +0,042).
- La implementación del repo (`far_soft_label` + `soft_labels: true`) reproduce la exploración
  con correlación 0,99999 en la semilla 42.
- **Test: nada.** Los folds y semillas de la confirmación: nada.

Sumadas todas, son ~20 variantes miradas contra la etiqueta de dev. Por eso el número de la
exploración no vale como evidencia y hace falta esta confirmación.

## 1 · Candidato (uno solo)

**C — GRU + TripSummary + estática completa con etiqueta suave lejos del evento.**
- Arquitectura, panel e hiperparámetros **idénticos** a la referencia
  (`configs/exp_v2all_gru_trips_estaticas_r3.yaml`), más `soft_labels: true`.
- `target: {name: far_soft_label, params: {far_target: 0.15}}` (`src/training/targets.py`): los
  cortes de autos que fallan con el evento a más de `G + H` del corte entrenan con 0,15 en vez de
  0. Los positivos siguen en 1, los sanos en 0, el `pos_weight` se cuenta con los positivos duros.
- τ = 0,15 es el punto más alto de la meseta 0,05–0,20. Se elige ahora y no se cambia.
- **Lo que se evalúa no cambia:** `label` dura, detección por auto.

**R — referencia:** la GRU de hoy, mismo YAML, entrenada con `label`.

## 2 · Confirmación

- **Datos:** dev v2 (426 autos con cortes, 135 fallados). Test sin tocar.
- **Folds nuevos:** `splits_confirm_r3.json` = `make_splits` con semilla **2026**, R = 3, la misma
  estratificación (`label` por vehículo × mercado × motor, `min_valid_positives: 5`).
- **Semillas nuevas del modelo:** 101, 102 y 103, las mismas para C y para R.
- **Ensamble:** rango promedio de las 3 semillas (`scripts/ensemble_rank.py`), igual para los dos.
- **Criterio primario:** diferencia C − R del promedio de detección al 5, 10, 15 y 20% de falsas
  alarmas (umbral exacto, `k = 2`, promedio de las 3 repeticiones).
- **Prueba:** bootstrap pareado por vehículo, 2.000 réplicas, fallados y sanos remuestreados por
  separado y el umbral re-fijado en cada réplica. **Gana si el límite inferior del IC95 es > 0**
  (p unilateral < 0,025).
- **"Por bastante":** además, la estimación puntual tiene que ser **≥ +5 puntos**.
- **Compuertas** (si falla una, C no se adopta aunque gane el primario):
  1. `audit_model.py` sobre la semilla 101 de C: (a0) aprueba y (b) de calendario no marca.
  2. **Azar:** en cada presupuesto de 5 a 20%, la detección de C supera el p95 del nulo que
     conserva la bolsa (score permutado entre filas, 200 veces).

**Secundarios (se reportan, no deciden):**
- C − R en cada presupuesto (2–30%) con su IC95.
- C y R contra la tasa de la celda mercado × motor (el piso de composición).
- C − R con las semillas 101–103 sobre los folds de la exploración (`splits_r3.json`).
- (a′), AUC dentro de mercado × motor, detección fuera de muestra, anticipación.

## 3 · Qué no se hace

- No se agrega ni se cambia ningún candidato después de este commit (τ, arquitectura,
  ensamble). Lo que aparezca va a "Qué queda abierto" de la ficha de resultados.
- No se toca el test. Usarlo para la cifra final lo decide el equipo.
