# F11 · GRU con etiqueta suave lejos del evento: le gana a la GRU de hoy con evidencia

**Fecha:** 2026-09-28 · **Fase:** F11 · **Rama:** `exp/gru-objetivo-suave` · **Preregistro:**
[f11-preregistro-objetivo-suave.md](f11-preregistro-objetivo-suave.md) (commit 426066a, anterior a
toda corrida de confirmación). Solo dev; test sin tocar; sin wandb (`WANDB_MODE=disabled`).

> **01-10-2026: en test no se sostuvo.** En el tiro preregistrado dio 31,2 de primario contra 44,5 de la GRU
> con `label` (−13,3, IC95 [−25,0; +3,9]), y el finalista volvió a ser la GRU de F10 (semillas 42, 1 y 2).
> Todo lo de abajo es dev. Ver [f11-test-resultado.md](f11-test-resultado.md).

## Qué es

La misma GRU + TripSummary + estática completa de hoy (arquitectura, panel, hiperparámetros), con
**un solo cambio en con qué se entrena**:
- los cortes de un auto que falla con el evento a más de `G + H` (3.500 km) entrenan con
  **0,15** en vez de 0 (`target: far_soft_label`, `far_target: 0.15`);
- los cortes positivos siguen en 1 y los sanos en 0;
- el `pos_weight` se cuenta con los positivos duros (`gru_seq` con `soft_labels: true`).

**Lo que se evalúa no cambia:** detección por auto con la etiqueta dura, umbral exacto, `k = 2`.

**Por qué funciona.** Con `label` a secas, un corte a 10.000 km del evento de un auto que falla
se entrena igual que un corte de un auto sano. Pero la detección se cuenta por auto, y ese auto ya
se distingue de los sanos lejos del evento (el *qué auto*, [f9-remedicion-completa-v2.md](f9-remedicion-completa-v2.md) §7).
La etiqueta suave deja de castigar esa señal sin convertir el objetivo en la cohorte: con τ = 1
(la cohorte) la ganancia se cae a +3 puntos. Es el objetivo "lineal por tramos" de vida útil
remanente de Heimes (PHM 2008), llevado a clasificación.

## Reproduce

```bash
python scripts/make_splits.py --config configs/data/splits_panel_v2_confirm_r3.yaml   # folds nuevos (semilla 2026)
WANDB_MODE=disabled python scripts/train.py --config configs/exp_f11_gru_{suave,ref}_{conf,expl}_s{101,102,103}.yaml
WANDB_MODE=disabled python scripts/ensemble_rank.py --config configs/exp_f11_seeds3_gru_{suave,ref}_{conf,expl}.yaml
python scripts/report_v2_models.py --config configs/report_f11.yaml        # primario (confirmación)
python scripts/report_v2_models.py --config configs/report_f11_expl.yaml   # secundario (folds de la exploración)
python scripts/report_v2_leads.py --config configs/report_f11_leads.yaml   # anticipación
WANDB_MODE=disabled python scripts/audit_model.py --config configs/exp_f11_gru_{suave,ref}_conf_s101.yaml
```

Las 12 corridas tardan ~20 min en paralelo (2 hilos cada una, 24 núcleos).

## Resultado de la confirmación (preregistrada)

Folds nuevos (semilla 2026), semillas nuevas (101, 102, 103), ensamble por rango de 3 semillas
para los dos. Detección de fallados (%) a cada presupuesto de sanos con falsa alarma:

| | 2% | 5% | 10% | 15% | 20% | 25% | 30% | **Primario** |
|---|---|---|---|---|---|---|---|---|
| **GRU con etiqueta suave ×3** | 16,3 | **34,6** | **48,6** | **60,2** | **70,1** | 73,1 | 78,8 | **53,4** |
| GRU de hoy ×3 | 17,0 | 24,0 | 42,0 | 51,6 | 57,5 | 65,4 | 72,3 | 43,8 |
| Tasa de la celda mercado × motor | 3,7 | 14,6 | 30,4 | 48,1 | 62,0 | 62,0 | 62,0 | 38,8 |
| Nulo (score permutado, bolsa conservada), p95 | 5,2 | 9,6 | 15,8 | 21,8 | 26,9 | 31,9 | 37,0 | — |

**Diferencia contra la GRU de hoy** (bootstrap pareado por vehículo, 2.000 réplicas):

| | 2% | 5% | 10% | 15% | 20% | 25% | 30% | **Primario** |
|---|---|---|---|---|---|---|---|---|
| diferencia (puntos) | −0,7 | +10,6 | +6,7 | +8,6 | +12,6 | +7,7 | +6,4 | **+9,6** |
| IC95 | [−5,4; 9,4] | [−1,2; 17,0] | [0,5; 15,3] | [0,2; 16,8] | [4,2; 17,5] | [2,2; 13,3] | [1,2; 10,9] | **[+3,0; +14,4]** |
| p (dif ≤ 0) | 0,40 | 0,052 | 0,020 | 0,023 | 0,0005 | 0,004 | 0,010 | **0,0015** |

**Veredicto del preregistro:**
- **Primario: gana.** +9,6 puntos, IC95 [+3,0; +14,4], p = 0,0015. Supera también la vara de
  "por bastante" (≥ +5).
- **Compuerta 2, azar: pasa.** Al 5–20% la detección está entre 3,6× y 2,6× el p95 del nulo (34,6
  contra 9,6; 70,1 contra 26,9).
- **Compuerta 1, auditorías: pasa.** (a0) aprueba, (a′) +0,0006, (b) +0,004 (abajo).
- **Se adopta como el mejor modelo sobre dev v2.**

**Secundarios:**
- **Contra la celda mercado × motor:** +14,6 puntos de primario, IC95 [3,7; 22,5], p = 0,003.
  **Es el primer modelo que le gana con evidencia a la composición.** La GRU de hoy, en los mismos
  folds, da +5,0 [−6,4; 14,9].
- **Réplica con los folds de la exploración** (`splits_r3`) y las semillas 101–103: +10,7
  [3,6; 16,0], p = 0,0005 (30,4 · 50,9 · 62,2 · 70,6% contra 24,9 · 40,2 · 47,2 · 58,8%).
- **Por fila y por auto:** PR-AUC 0,164 contra 0,154; ROC por fila 0,758 contra 0,727; AUC por
  auto 0,850 contra 0,812; **AUC dentro de mercado × motor 0,689 contra 0,645**.
- **Fuera de muestra** (umbral fijado con los sanos de los otros folds): 34,3% al 5,0% de falsas
  alarmas realizadas, 49,4% al 10,0%, 69,4% al 19,9%.
- **Por semilla suelta** (la dispersión que el ensamble achica): la suave da 31,4 / 30,6 / 31,1%
  al 5% y la de hoy 25,4 / 26,2 / 24,0%.
- **La exploración ya lo mostraba:** con los folds de siempre y las semillas 42, 1 y 2 dio +9,5
  [3,5; 14,9]. Tres tandas de semillas y dos juegos de folds dan entre +9,5 y +10,7.

## Anticipación y *cuándo* (`report_v2_leads.py`)

| | detectados al 5 / 10 / 20% | km antes (mediana) al 10% | días (aprox.) | alerta en el 1er corte al 10% |
|---|---|---|---|---|
| GRU con etiqueta suave ×3 | 46,7 / 65,7 / 94,7 | 8.300 | 106 | 22% |
| GRU de hoy ×3 | 32,3 / 56,7 / 77,7 | 8.400 | 102 | 15% |

Rango percentil medio del score:

| | sanos | fallados a > 10.000 km | 3.500–10.000 km | ≤ 3.500 km |
|---|---|---|---|---|
| suave | 0,406 | 0,663 | 0,703 | 0,743 |
| de hoy | 0,434 | 0,591 | 0,654 | 0,714 |

- La anticipación mediana no cambia (~8.300 km, ~3 meses). La etiqueta suave suma autos,
  no adelanta ni atrasa la alerta.
- El score sigue subiendo al acercarse el evento: el *cuándo* no se perdió.
- Lo que cambia es la separación a todas las distancias, más grande lejos del evento (+0,07 de
  rango a más de 10.000 km).

## Auditorías (compuerta 1)

`audit_model.py`, semilla 101, folds de confirmación:

| | (a0) PR-AUC permutado | (a′) | (b) calendario (ROC) | veredicto |
|---|---|---|---|---|
| **GRU con etiqueta suave** | 0,0577 vs base 0,0583 | **+0,0006** | +0,0038 | **aprueba; (b) no marca** |
| GRU de hoy | 0,0585 vs base 0,0583 | −0,0093 | +0,0115 | (a0) aprueba; **(a′) falla** |

- **Compuerta 1: pasa.** Sin leakage, y las `aux_` de calendario casi no mueven el ROC.
- El (a′) de la suave es chico: sabe poco del *cuándo*, como todas las de v2. Pero es positivo, y
  la GRU de hoy reentrenada en estos mismos folds y semilla lo tiene negativo (en F9, con la semilla
  42 y los folds de siempre, daba +0,001).
- La (b) de la suave (+0,004) es la mitad de la de hoy en estos folds (+0,012).

**No es calendario** (exploración, dentro de celda × mes del corte): el AUC por fila sube lejos del
evento (0,583 → 0,608 a 3.500–10.000 km; 0,540 → 0,557 a más de 8.000) y queda igual cerca
(0,656 → 0,667). Entre los sanos, la correlación del score con el mes, dentro de la celda, es 0,000
(la de hoy: +0,042).

## Explicabilidad: en qué se apoya

Importancia por permutación de cada canal (sus 20 bins juntos) y de cada estática, medida **sobre la
detección** (repetición 0, semilla 101; promedio de 3 permutaciones). También, la parte de |Δlogit|
que se lleva cada familia.

| familia | caída de detección si se permuta (puntos), suave / de hoy | parte del |Δlogit|, suave / de hoy |
|---|---|---|
| estática (país, motor, modelo) | 48,4 / 42,8 | 35% / 22% |
| estado del DPF (acumulación, subidas, regeneraciones, mensajes) | 37,8 / 33,1 | 26% / 30% |
| hábitos (idle, bajo régimen, velocidad, km y duración del viaje) | 23,0 / 28,6 | 25% / 24% |
| cobertura (registros, viajes, bins observados) | 17,9 / 33,1 | 15% / 24% |

- **La duración de los viajes es el hábito que más pesa** (−23 puntos de detección si se permuta),
  junto con el nivel medio y la subida de la acumulación del DPF (−11 y −9). Es la lectura física
  de siempre: viajes cortos que no dejan terminar la regeneración y el hollín que se acumula.
- **Se apoya menos en la cobertura** (15% del |Δlogit| contra 24%): depende menos de cuántos
  registros tiene la ventana, que es lo más parecido a un artefacto de muestreo.
- **Se apoya más en el país** (19,5% contra 11,4% del |Δlogit|). Una parte de la ganancia es
  composición: contra la GRU de hoy recalibrada por celda (z-score dentro de la celda + logit de la
  tasa de la celda, diagnóstico no preregistrado) la ventaja baja a +5,1 [−0,1; 10,0]. Pero el AUC
  **dentro** de mercado × motor sube 0,645 → 0,689, y le gana a la celda por +14,6.
- El mensaje al cliente por alerta se puede armar igual que para la GRU de hoy
  (`src/eval/explain_seq.py`, Shapley por permutación por canal, en la rama
  `feat/explicabilidad-gru`): la arquitectura es la misma.

Reproducir (reentrena la repetición 0 con la semilla 101 y permuta cada unidad en la validación de
cada fold; ~10 min):

```bash
WANDB_MODE=disabled python scripts/explain_perm_seq.py --config configs/explain_f11.yaml   # -> experiments/explain-f11/
```

## Lo que se exploró antes y no sirvió

Tabla completa en el preregistro §0. En corto:
- suavizar el score en el tiempo (media acumulada, EWMA, CUSUM) le **resta** 2–7 puntos: la
  detección vive en los picos cerca del evento;
- la historia completa del auto como estáticas, y 11 canales más en la secuencia, no suman o
  restan;
- la pérdida por vehículo (MIL) y la recalibración por celda suman ~+4 cada una, pero no encima
  de la etiqueta suave.

## Qué queda abierto

- **El test.** 36 fallados. Es la cifra final para el pitch, y usarlo lo decide el equipo.
- **Más semillas.** En la exploración, 9–12 miembros suben el 2% (14 → 17–19) sin mover el
  primario. Un ensamble de más semillas es una variante nueva: va a otro preregistro.
- **La misma etiqueta en la CNN-LSTM** dio +5,6 en exploración. No se confirmó.
- **Qué parte es composición.** La tasa real por celda de Ford sigue siendo lo que falta para
  separar física de lista (f10-sweep-gru.md).
