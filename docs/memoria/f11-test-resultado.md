# F11 · El test: la GRU con etiqueta suave no se sostiene, y el finalista es la GRU de F10

**Fecha:** 2026-10-01 · **Fase:** F11 (medición en test) · **Rama:** `docs/finalista-gru-f10` ·
**Preregistro:** [f11-preregistro-test.md](f11-preregistro-test.md) (commit f15f432).

**En corto:**
- La GRU con etiqueta suave (F11) **no se sostiene en test**: primario **31,2 contra 44,5** de la GRU
  con `label`, −13,3 puntos, IC95 [−25,0; +3,9]. La regla del preregistro dice "no se sostiene" y así
  se reporta, sin buscar variantes.
- **El finalista vuelve a ser la GRU de F10**: GRU + TripSummary + estática completa, entrenada con
  `label`, ensamble por rango de las **semillas 42, 1 y 2**. Era la referencia del preregistro, y en test
  repitió lo que daba en dev en tres lecturas independientes (primario 44,5–50,8).
- Los secuenciales (GRU y CNN-LSTM, ~50) le duplican el primario a los tabulares (~21–27). Los tabulares
  quedan en el nivel de la celda mercado × motor (23,4). Ningún modelo le gana con evidencia a la celda
  en test, pero la GRU de F10 queda muy cerca (+27,3 [−0,8; 46,1]).

## El finalista, completo

| | |
|---|---|
| Modelo | `gru_seq`: GRU de 1 capa (24 unidades) con pooling por atención (16) + rama estática (4) + cabeza (16), dropout 0,3 |
| Entrada | `panel_seq_trips_v2_estaticas.parquet`: secuencia de signals + TripSummary en 20 bins de km de la ventana W, más país, motor y modelo como `static_` |
| Entrenamiento | `label` (sin `target:`), 40 épocas fijas, batch 64, Adam lr 0,001, weight decay 1e-4, grad clip 1,0, `class_weight: balanced`, CPU |
| Semillas | **42, 1 y 2** (`random_state` del modelo), ensamble por **rango** de las tres. Las eligió el reporte v2 antes de mirar test, y el sweep de F10 las confirmó |
| Configs | `configs/exp_v2all_gru_trips_estaticas_r3.yaml` (42), `..._s1_r3.yaml` (1), `..._s2_r3.yaml` (2); ensamble `configs/exp_v2all_seeds3_gru_trips_estaticas.yaml` |
| Folds de dev | `splits_r3.json` (5 folds × 3 repeticiones, semilla 42) |
| Modelo final | cada semilla entrenada con **todo dev** (446 autos, 426 con cortes), y ensamble por rango sobre lo que puntúa |
| Regla de alerta | 2 cortes seguidos con score ≥ τ; τ fijado con los sanos de dev al presupuesto de falsas alarmas |

**Dev** (CV, mismas semillas, `f9-remedicion-completa-v2.md`, `f10-sweep-gru.md`): 27,4 · 42,5 · 53,3 ·
60,7% al 5 · 10 · 15 · 20%, primario 46,0; AUC dentro de mercado × motor 0,65.

**Test:**

| GRU F10 (semillas 42/1/2) | 2% | 5% | 10% | 15% | 20% | 25% | 30% | **Primario** |
|---|---|---|---|---|---|---|---|---|
| Detección (%) | 15,6 | 31,2 | 46,9 | 62,5 | 62,5 | 65,6 | 68,8 | **50,8** |
| p95 del nulo | 6,3 | 11,5 | 17,7 | 25,1 | 37,5 | 40,6 | 43,8 | — |
| Piso de celda mercado × motor | 0,0 | 0,0 | 21,9 | 21,9 | 50,0 | 50,0 | 50,0 | 23,4 |

- **Contra la celda: +27,3 puntos de primario, IC95 [−0,8; +46,1]**, p(dif ≤ 0) = 0,028. Queda en el
  límite y no se cita como "le gana".
- **Con el umbral fijado en dev** (15 modelos de fold, `splits_r3`): detecta 19,6 · 34,2 · 57,3% al 5 ·
  10 · 20%, con **falsas alarmas reales de 5,8 · 8,9 · 18,2%**. Es el número para el punto de
  operación: τ se elige sin ver test y cumple su presupuesto.
- AUC por auto 0,794; dentro de mercado × motor 0,675.
- Anticipación mediana de la primera alerta al 10%: ~4.600 km (sin los autos vistos, ~8.800).
- **Sin los 36 autos que estaban en el dev viejo** (71 autos, 22 fallados): 36,4 · 50,0 · 50,0 · 59,1%,
  primario 48,9.

**Qué se cita:** la GRU de F10 se midió tres veces en test con la misma configuración (abajo), así que va
**un rango, no un número**: **~30% al 5%, ~40–50% al 10% y ~50–60% al 20% de falsas alarmas**, y en
operación (umbral de dev) ~20 · 34 · 57%.

## 1 · El tiro preregistrado (F11): no se sostiene

Test: 103 autos con cortes (32 fallados, 71 sanos). Los otros 8 del holdout no tienen cortes en el
panel. Las dos GRU usan las semillas 101–103, cada una entrenada con todo dev.

| | 2% | 5% | 10% | 15% | 20% | 25% | 30% | **Primario** |
|---|---|---|---|---|---|---|---|---|
| **GRU suave (candidato)** | 15,6 | 18,8 | 21,9 | 34,4 | 50,0 | 56,2 | 68,8 | **31,2** |
| GRU con `label` (referencia) | 15,6 | 28,1 | 50,0 | 50,0 | 50,0 | 62,5 | 68,8 | **44,5** |
| Piso de celda | 0,0 | 0,0 | 21,9 | 21,9 | 50,0 | 50,0 | 50,0 | 23,4 |
| p95 del nulo (candidato) | 7,4 | 12,5 | 17,7 | 26,1 | 37,5 | 39,6 | 42,7 | — |

- **Primario: −13,3 puntos, IC95 [−25,0; +3,9]**, p(dif ≤ 0) = 0,92. **No se sostiene** (regla §2).
- Contra la celda: candidato +7,8 [−14,1; 28,1]; referencia +21,1 [−7,0; 41,4].
- Sin los autos vistos: 39,8 contra 42,0 (−2,3 [−17,1; +15,9]).
- Umbral fijado en dev: candidato 23,1 · 34,6 · 58,5% (falsas alarmas reales 6,7 · 11,7 · 19,6%);
  referencia 21,7 · 36,7 · 58,5% (5,5 · 10,6 · 17,7%).
- Anticipación mediana del candidato al 10%: ~6.600 km.

### Por qué (diagnóstico posterior, no cambia el veredicto)

- **En test, la GRU suave se comportó como la tasa de la celda.** Al 10% y al 20% detecta exactamente lo
  mismo que la celda sola (21,9 y 50,0). La correlación de su score medio por auto con la tasa de dev de
  su celda país × motor × serie es 0,89 (la de `label`, 0,83).
- **Lo que no se trasladó es el orden dentro de la celda**, que era su ganancia en dev:

  | AUC dentro de mercado × motor | dev | test |
  |---|---|---|
  | GRU suave | 0,689 | **0,571** |
  | GRU con `label` | 0,645 | 0,668 |

  Las dos celdas más riesgosas (CHL×ENG_2 y COL×ENG_2) tienen 12 de los 71 sanos de test (17%). Al 5–10%
  el umbral cae adentro de esas celdas, y ahí solo cuenta el orden dentro de la celda. Al 10% la
  referencia detecta 16 fallados y la suave 7.
- **La separación promedio sí se sostuvo:** rango percentil medio del score de la suave en test 0,398 ·
  0,660 · 0,680 · 0,721 (sanos · > 10.000 km · 3.500–10.000 · ≤ 3.500), contra 0,406 · 0,663 · 0,703 ·
  0,743 en dev.
- **Mecanismo probable:** el 0,15 en todos los cortes lejanos de un auto que falla premia reconocer a
  *ese auto* en cualquier momento de su historia, y lo más estable de un auto es su país, motor, serie y
  hábitos persistentes. La ficha de F11 ya mostraba más peso en la estática (35% del |Δlogit| contra 22%)
  y que la mitad de la ganancia era composición (+5,1 contra la GRU recalibrada por celda).
- **Por qué no apareció en autos nuevos.** No se puede separar con 32 fallados, y probablemente es una
  suma de tres causas:
  - **ruido:** el AUC dentro de la celda de test tiene un error estándar de ~0,06–0,07;
  - **optimismo de dev:** la confirmación usó folds y semillas nuevos, pero los mismos 446 autos sobre los
    que se exploraron ~20 variantes;
  - **el modelo final:** 40 épocas fijas con 25% más datos que un modelo de fold. La suave entrenada con
    todo dev da 31,2, contra 38,4 ± 3,3 de sus 15 modelos de fold sobre los mismos autos de test; la de
    `label` da 44,5 contra 42,7 ± 5,6.

## 2 · Tres lecturas en test de la GRU de F10

| GRU F10 en test | 2% | 5% | 10% | 15% | 20% | 30% | Primario | AUC celda |
|---|---|---|---|---|---|---|---|---|
| Semillas 42/1/2, 01-10 (`eval_test.py`) | 15,6 | 31,2 | 46,9 | 62,5 | 62,5 | 68,8 | 50,8 | 0,675 |
| Semillas 42/1/2, 27-09 (script previo) | 12,5 | 34,4 | 40,6 | — | 62,5 | 68,8 | — | 0,664 |
| Semillas 101–103, 01-10 (referencia de F11) | 15,6 | 28,1 | 50,0 | 50,0 | 50,0 | 68,8 | 44,5 | 0,668 |

- **42/1/2 contra 101–103: +6,2 puntos de primario, IC95 [−7,8; +18,0].** Es el mismo modelo con ruido
  de semilla.
- **La misma semilla no reproduce el modelo bit a bit.** Los scores del 01-10 correlacionan 0,88–0,91 con
  los del 27-09, con la misma config. El código de la ruta con `label` no cambió: el 28-09 solo se agregó
  `soft_labels`, que con `label` no hace nada. Las causas probables son el número de hilos (el 01-10 corrió
  con `OMP_NUM_THREADS=1`) y que el 27-09 se midió con otro script, descartado después (no está en el
  repo; sus salidas quedaron en `experiments/eval-test-gru/`). Es una razón más para citar rangos.
- **El vistazo del 27-09 fue antes del preregistro de F11** y miró la GRU de F10, no el candidato. No pudo
  inflar el resultado de F11, que además salió peor. Pero el test ya se había mirado una vez, y la
  elección de hoy usa test: **la GRU de F10 queda como finalista porque era la referencia preregistrada
  y el candidato no se sostuvo, no porque se haya elegido entre varias mirando test.**

## 3 · Los demás modelos en test (descriptivo, no elige)

Misma cuenta, cada modelo entrenado con todo dev (semilla 42 salvo los ×3). Se midieron después del tiro
preregistrado: describen, no eligen.

| Modelo | 5% | 10% | 15% | 20% | **Primario** | AUC auto | AUC celda | dev (5 · 10 · 20%) |
|---|---|---|---|---|---|---|---|---|
| **GRU F10 ×3 (42/1/2)** | 31,2 | 46,9 | 62,5 | 62,5 | **50,8** | 0,794 | 0,675 | 27 · 42 · 61 |
| CNN-LSTM + trips + estática ×3 (42/1/2) | 15,6 | 56,2 | 56,2 | 68,8 | **49,2** | 0,790 | 0,638 | 22 · 41 · 62 |
| Survival stacking (F3) | 18,8 | 21,9 | 21,9 | 43,8 | **26,6** | 0,710 | 0,515 | 14 · 24 · 42 |
| LightGBM (control, 53 features) | 15,6 | 25,0 | 28,1 | 34,4 | **25,8** | 0,725 | 0,619 | 18 · 28 · 42 |
| Logística L2 (53 features) | 6,2 | 12,5 | 31,2 | 34,4 | **21,1** | 0,679 | 0,556 | 15 · 23 · 37 |
| SS + efecto aleatorio por auto (GPBoost) | 9,4 | 9,4 | 9,4 | 12,5 | **10,2** | 0,363 | 0,377 | 6 · 9 · 17 |
| Piso de celda (sin modelo) | 0,0 | 21,9 | 21,9 | 50,0 | 23,4 | — | — | 18 · 33 · 62 |

**Con el umbral fijado en dev** (detección % / falsas alarmas reales % al 5 · 10 · 20%):

| Modelo | Detección | Falsas alarmas reales |
|---|---|---|
| GRU F10 | 20 · 34 · 57 | 5,8 · 8,9 · 18,2 |
| CNN-LSTM | 16 · 34 · 55 | 5,6 · 8,4 · 14,8 |
| LightGBM | 17 · 25 · 37 | 7,5 · 10,7 · 16,9 |
| Survival stacking | 13 · 20 · 33 | 4,6 · 8,3 · 15,6 |
| Logística L2 | 12 · 16 · 33 | 8,8 · 11,8 · 18,3 |
| GPBoost | 8 · 11 · 14 | 6,5 · 11,3 · 21,5 |

- **GRU contra survival stacking: −24,2 puntos de primario para SS, IC95 [−43,0; −0,8].** El IC no
  cruza el cero, pero apenas, y no estaba preregistrado.
- Sin los autos vistos (22 fallados), LightGBM y SS suben a 50 de primario y empatan con la GRU
  (SS − GRU +1,1 [−20,5; +27,3]). Con este tamaño, el orden entre familias tampoco es firme.
- La CNN-LSTM es inestable a presupuestos bajos (15,6 al 5%, 56,2 al 10%).
- GPBoost: un auto nuevo entra sin su efecto aleatorio, y su AUC por auto ya estaba invertido en dev
  (0,34).
- La CNN-LSTM va con estática completa: el panel `panel_seq_trips_v2.parquet` (sin estática) no está
  construido en `data/v2`.

## 4 · Lo que cita el pitch del finalista (agregado el 01-10)

El deck (`docs/pitch/deck/index.html`) cita la GRU de F10 en test. Además de §1–§3, usa:

- **Anticipación en test** (umbral exacto al 10%, ensamble 42/1/2): mediana 4.646 km (p25–p75 2.403–9.263), y
  **~99 días** aproximados con los km por día de cada auto (pendiente odómetro ~ fecha de sus cortes, como
  `report_v2_leads.py`). Al 5%: 3.987 km, ~72 días; al 20%: 5.969 km, ~109 días. Los autos detectados de test
  andan menos km por día que los de dev: en km da menos que dev (~7.500), en días parecido.
- **El score sube hacia el evento, en test:** rango percentil medio (promedio de las 3 semillas) 0,44 en sanos,
  0,58 a más de 10.000 km, 0,61 a 3.500–10.000 y 0,72 a menos de 3.500.
- **En qué se apoya** (dev, semilla 42, repetición 0 de `splits_r3`, `configs/explain_gru_final.yaml`): caída de
  la detección media al 5–20% (base 51,9) al permutar cada señal en la validación. País −28,1; duración del viaje
  −24,1; registros −23,4; subida de la acumulación −22,2; motor −15,1; viajes −13,4; nivel medio de acumulación
  −10,2; km por viaje −9,8; máximo de acumulación −9,2; regeneraciones −9,0; serie −8,3; velocidad −5,9; bajo
  régimen −4,8; ralentí −2,2. **Las sumas por familia no se citan:** pasan la detección base (estado del DPF 63,9),
  porque permutar una señal sola no es esconder la familia.
- **La demo** (`demo-bundle-gru-final`, dev, repetición 0) al 5 · 10 · 15 · 20%: 39 · 56 · 69 · 82 de 135 fallas
  avisadas, 14 · 29 · 43 · 58 de 291 sanos con aviso, 53 · 85 · 112 · 140 alertas (48 · 73 · 101 · 125 con hábito),
  7 · 10 · 22 · 36 escalamientos, anticipación mediana 15 · 16 · 16 · 21 semanas. Los agentes escribieron 465 textos,
  el verificador rechazó 103 intentos y no hizo falta ninguna plantilla.

```bash
FORD_DATA_DIR=$PWD/data/v2 WANDB_MODE=disabled python scripts/explain_perm_seq.py --config configs/explain_gru_final.yaml   # -> experiments/explain-gru-final/
```

## Reproduce

Todo con los datos v2 (`FORD_DATA_DIR=$PWD/data/v2`), sin wandb. Cada `--fit` entrena el modelo final y
los 15 de fold (~6 min por GRU en CPU, se pueden correr en paralelo); `--report` mide.

```bash
export FORD_DATA_DIR=$PWD/data/v2 WANDB_MODE=disabled OMP_NUM_THREADS=1

# 1 · el tiro preregistrado (F11) -> experiments/test-f11/
python scripts/eval_test.py --config configs/eval_test_f11.yaml --fit configs/exp_f11_gru_{suave,ref}_conf_s{101,102,103}.yaml   # uno por llamada
python scripts/eval_test.py --config configs/eval_test_f11.yaml --report

# 2 · la GRU de F10 (42/1/2) contra la misma con 101-103 -> experiments/test-f10-s42/
#     (la referencia se re-entrena con --fit, o se copian sus test_/devoof_ de experiments/test-f11/)
python scripts/eval_test.py --config configs/eval_test_f10_s42.yaml --fit configs/exp_v2all_gru_trips_estaticas{,_s1,_s2}_r3.yaml
python scripts/eval_test.py --config configs/eval_test_f10_s42.yaml --report

# 3 · los demás modelos -> experiments/test-modelos-v2/ (la GRU se copia de test-f10-s42/)
python scripts/eval_test.py --config configs/eval_test_modelos_v2.yaml --fit configs/exp_v2_lgbm_r3.yaml   # y el resto del YAML
python scripts/eval_test.py --config configs/eval_test_modelos_v2.yaml --report
```

`eval_test.py` lee mercado y motor del panel del **primer** grupo del YAML, para el piso de celda. Por eso
en `eval_test_modelos_v2.yaml` la GRU va primera: `panel.parquet` trae el motor como `aux_`. La GRU no es
bit a bit entre corridas (§2): una reproducción da números cercanos, no iguales.
