# Preregistro: la medición en test de la GRU con etiqueta suave (F11)

**Fecha:** 2026-09-28 · **Fase:** F11 · **Rama:** `exp/gru-objetivo-suave`.

**Se escribió** después de la confirmación sobre dev ([f11-gru-objetivo-suave.md](f11-gru-objetivo-suave.md))
y **antes de leer una sola fila de test contra la etiqueta**. El commit que agrega este archivo es
la marca de tiempo. **Es un solo tiro:** después de mirar el test no se cambia el modelo, las
semillas, el umbral ni la cuenta, y no se re-corre nada para "mejorar" el número.

**Por qué ahora.** El modelo está elegido (preregistro y confirmación de F11), y el equipo pidió
medirlo en test (28-09). Es el uso previsto del holdout (CLAUDE.md, regla 2).

## 1 · Qué se mide

- **C (candidato):** `configs/exp_f11_gru_suave_conf_s{101,102,103}.yaml`.
- **R (referencia):** `configs/exp_f11_gru_ref_conf_s{101,102,103}.yaml`, la GRU de hoy.
- **Modelo final:** cada semilla se entrena con **todo dev** (446 autos, 426 con cortes) y
  puntúa los cortes de test. Las tres semillas se ensamblan por rango, igual que en dev.
- **Piso de composición:** la tasa de fallas de la celda mercado × motor, calculada con los autos
  de dev y usada como score constante en los de test.
- **Test:** los autos de `test_split.json` que tienen cortes en el panel.

## 2 · La cuenta

Es la misma de dev (`report_v2_models.py`): detección por auto, alerta con 2 cortes seguidos ≥ τ,
umbral exacto a cada presupuesto de sanos de test con falsa alarma (2–30%).

- **Primario:** promedio de la detección al 5, 10, 15 y 20%. C − R con bootstrap pareado por
  vehículo (2.000 réplicas, fallados y sanos por separado, τ re-fijado en cada réplica).
- **Cómo se lee.** Con ~33 fallados en test la potencia es baja; la evidencia de superioridad es la
  confirmación de dev. El test dice si se sostiene afuera:
  - **se sostiene** si la diferencia primaria es > 0;
  - **es significativo en test** si además el límite inferior del IC95 es > 0;
  - **no se sostiene** si es ≤ 0. Eso se reporta tal cual, sin buscar variantes.
- **Azar:** la detección de C contra el p95 del nulo que conserva la bolsa (el score permutado
  entre las filas de test, 200 veces), al 5–20%.

## 3 · Secundarios (se reportan, no deciden)

- C y R contra el piso de celda.
- **Punto de operación fijado en dev:** se re-entrenan los 15 modelos de fold de la CV de
  confirmación (3 repeticiones × 5 folds, cada uno con las tres semillas promediadas en
  probabilidad). τ se fija con los niveles de los sanos de dev fuera de fold, al 5, 10 y 20%, y se
  aplica a los cortes de test que puntúa cada modelo de fold. Se reportan la falsa alarma
  realizada en test y la detección, promediadas sobre los 15 modelos.
- **Sin los 36 autos de test que estaban en el dev viejo** (`previous.test_vehicles_in_old_dev`),
  como pide la decisión del 26-09.
- Anticipación mediana de la primera alerta al 10%.

Script: `scripts/eval_test.py`, config `configs/eval_test_f11.yaml`.

**Resultado (01-10-2026):** no se sostiene (−13,3 puntos de primario, IC95 [−25,0; +3,9]). Ver
[f11-test-resultado.md](f11-test-resultado.md). Este protocolo no se editó después del tiro.
