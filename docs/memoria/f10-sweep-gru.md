# F10 · Sweep bayesiano de la GRU: el mejor trial no le gana a la configuración de hoy

**27-09-2026.** Pedido del equipo: la GRU + TripSummary + estática completa va a ser el modelo final,
y antes de fijarla se barren sus hiperparámetros. Solo dev, mismos folds (`splits_r3.json`), test sin
tocar. Es exploratorio y no gasta presupuesto de comparaciones: no se adopta nada que no pase la
confirmación con semillas nuevas.

## Qué se corrió

- **Sweep:** Optuna TPE multivariado, 1 h, 6 procesos × 3 hilos, poda de la mediana después de cada
  repetición de la CV. Se usó Optuna y no el sweep de wandb por la poda y el espacio condicional
  (`attention_hidden` solo con `pooling: attention`). Cada trial igual se loguea a wandb, grupo
  `f10-sweep-gru`. Hay 13 hiperparámetros en `configs/sweep_gru.yaml → space`, y el trial 0 es la
  configuración de hoy.
- **Objetivo:** detección por auto con umbral exacto (la cuenta de `report_v2_models.py`), promedio
  al 5 / 10 / 15 / 20% de sanos con falsa alarma y después promedio de las 3 repeticiones, cada una con
  sus propios scores. Todos los trials usan la semilla 42 del modelo.
- **Confirmación:** el mejor trial y la configuración de hoy, re-entrenados con las semillas 3, 4 y 5
  (que el sweep no vio), en ensamble por rango de 3 semillas y con bootstrap pareado por vehículo.

```
python scripts/sweep_gru.py --config configs/sweep_gru.yaml                  # ~1 h
python scripts/sweep_gru.py --config configs/sweep_gru.yaml --summary        # top.yaml, trials.csv
python scripts/sweep_gru.py --config configs/sweep_gru.yaml --write-configs  # configs de confirmación
python scripts/train.py --config configs/exp_sweep_gru_{base,top1}_s{3,4,5}_r3.yaml
python scripts/ensemble_rank.py --config configs/exp_sweep_gru_seeds3_{base,top1}.yaml
python scripts/report_v2_models.py --config configs/report_sweep_gru.yaml    # experiments/report-sweep-gru/
```

## Resultado del sweep

Se corrieron 130 trials: 65 completos y 65 podados. La mediana de los completos dio 0,449.

| Trial | Objetivo | 5% | 10% | 15% | 20% | AUC dentro de celda | Min |
|---|---|---|---|---|---|---|---|
| **t112** | **0,492** | 30,4 | 46,7 | 56,1 | 63,7 | 0,673 | 2,2 |
| t122 | 0,481 | 28,4 | 43,5 | 55,8 | 64,7 | 0,638 | 2,8 |
| t94 | 0,476 | 29,6 | 41,0 | 55,6 | 64,2 | 0,658 | 2,2 |
| t0 (la de hoy) | 0,475 | 28,6 | 44,0 | 54,3 | 63,2 | 0,680 | 7,4 |

t112 es chica y rápida: `hidden` 9, `attention_hidden` 4, `static_hidden` 12, `head_hidden` 36,
dropout 0,32, 30 épocas, batch 256, lr 6,0e-3, weight decay 9,6e-5 y grad clip 1,25. Según la
importancia de Optuna, lo que más pesa es `weight_decay` (0,34), después `learning_rate` (0,15) y
`dropout` (0,11). El pooling y `class_weight` no pesan.

## Confirmación: no hay ganador

Detección por auto con umbral exacto, en ensamble por rango de 3 semillas:

| | 5% | 10% | 15% | 20% | Promedio |
|---|---|---|---|---|---|
| La de hoy, semillas 3, 4, 5 | 28,6 | 41,2 | 54,3 | 64,2 | **0,471** |
| t112, semillas 3, 4, 5 | 27,4 | 44,0 | 53,8 | 62,2 | **0,469** |
| La de hoy, semillas 42, 1, 2 (reporte v2) | 27,4 | 42,5 | 53,3 | 60,7 | 0,460 |
| Tasa de la celda mercado × motor | 18,0 | 32,6 | 48,1 | 62,2 | 0,402 |
| Survival stacking (F3) | 14,1 | 23,7 | 35,8 | 42,2 | 0,290 |

- **t112 − la de hoy:** −1,2 / +2,7 / −0,5 / −2,0 puntos al 5 / 10 / 15 / 20%. En ningún presupuesto
  el IC95 del bootstrap pareado excluye el cero (`paired.csv`).
- Con semillas sueltas pasa lo mismo: 0,449 de promedio la de hoy contra 0,445 t112. Dentro de
  mercado × motor, el AUC es 0,640 contra 0,633.
- **El +1,7 del sweep era selección:** es el máximo de 65 intentos con una misma semilla. Con la
  misma configuración, la semilla sola mueve el objetivo de 0,395 a 0,476.

## Qué se decidió

- **Se queda la configuración de hoy** (`configs/exp_v2all_gru_trips_estaticas_r3.yaml`), con las
  semillas 42, 1 y 2. Son las del reporte v2, las auditorías y la explicabilidad. Elegir las 3, 4 y 5
  porque dieron 0,471 en vez de 0,460 sería volver a seleccionar sobre ruido.
- Para el pitch, la misma configuración dio 0,460 y 0,471 en dos tandas de 3 semillas: **~27–29% al
  5% y ~61–64% al 20%**.
- Ojo con cómo se lee "×3 semillas": es el **ensamble por rango** de las tres (0,460), no el
  promedio de las semillas sueltas (0,439). Ensamblar semillas aporta ~2–3 puntos, más de lo que
  aportó el sweep. Un ensamble de más semillas es una variante nueva y va al preregistro del finalista.

## Por qué la celda es un piso tan alto

El piso de celda le da a cada auto la tasa de fallados de su celda mercado × motor, calculada con los
autos de los otros folds. No usa telemetría: es un score constante por auto. En dev hay 426 autos y
135 fallados:

| Celda | Autos | Fallados | Tasa |
|---|---|---|---|
| CHL × ENG_2 | 48 | 32 | 0,67 |
| COL × ENG_2 | 79 | 50 | 0,63 |
| BRA × ENG_3 | 91 | 35 | 0,38 |
| ARG × ENG_1 | 27 | 5 | 0,19 |
| BRA × ENG_2 | 40 | 7 | 0,18 |
| las otras 9 | 141 | 6 | 0–0,12 |

Tres celdas concentran 117 de los 135 fallados. Por eso el piso llega a 62% al 20% de falsas alarmas
y ahí se clava: alertó celdas enteras y no puede elegir autos dentro de ellas. Una parte puede ser
real (mercado, combustible, motor) y otra es cómo Ford armó las listas de fallados y sanos
(`f1-sesgo-eng3.md`, `f9-universo-v2.md`). La GRU recibe mercado y motor como estáticas. Lo que
aporta por encima es el orden dentro de la celda: AUC ~0,64 contra ~0,5 del piso. Eso le da
~+10 puntos al 5–10% y un empate al 20%, sin separarse con evidencia (bootstrap pareado).

Reproducir la tabla de celdas: tasa de `event_observed` por `static_SalesCountry_cd × static_Engine`
sobre los `dev_vehicles` de `test_split.json`, en `panel_seq_trips_v2_estaticas.parquet`.
