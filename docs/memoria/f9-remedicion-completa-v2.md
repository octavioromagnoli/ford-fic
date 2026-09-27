# Re-medición completa sobre v2: los secuenciales con TripSummary pasan adelante, nadie le gana a la celda

**Fecha:** 2026-09-26 · **Fase:** F9 · **Alcance:** dev v2 (446 autos, 135 fallados con cortes), R = 3,
`splits_r3.json`, test sin tocar, sin wandb (`WANDB_MODE=disabled`). **Es exploratorio y no elige
finalista.** Son 42 modelos o variantes comparables (56 corridas contando semillas sueltas) sobre los
mismos 446 autos. Elegir sobre v2 sigue siendo un preregistro con lista cerrada (CLAUDE.md,
presupuesto de comparaciones), y esta ficha es el insumo para armarla.

Reporte navegable (curva interactiva y todas las tablas):
https://claude.ai/artifact/5QddLUD5j3rtK9SqrgEmx8 (privado: hay que compartirlo desde el menú Share).

## Reproduce

```bash
python scripts/build_seq_panel.py --config configs/data/panel_seq_v2.yaml                 # signals + país (tutora)
python scripts/build_seq_panel.py --config configs/data/panel_seq_trips_v2.yaml           # + TripSummary
python scripts/build_seq_panel.py --config configs/data/panel_seq_v2_estaticas.yaml       # + motor y modelo
python scripts/build_seq_panel.py --config configs/data/panel_seq_trips_v2_estaticas.yaml
python scripts/make_v2all_configs.py                   # configs/exp_v2all_*.yaml desde configs/v2all_runs.yaml
WANDB_MODE=disabled python scripts/train.py --config configs/exp_v2all_<x>.yaml            # una por modelo
WANDB_MODE=disabled python scripts/ensemble_rank.py --config configs/exp_v2all_seeds3_<x>.yaml   # promedio de 3 semillas
WANDB_MODE=disabled python scripts/ensemble_rank.py --config configs/exp_v2all_ens_<x>.yaml
WANDB_MODE=disabled python scripts/audit_model.py --config configs/exp_v2all_<x>.yaml
python scripts/report_v2_models.py --config configs/report_v2_models.yaml    # -> experiments/report-v2-models/
python scripts/report_v2_models.py --config configs/report_v1_models.yaml    # la misma cuenta sobre la entrega 1
python scripts/report_v2_leads.py --config configs/report_v2_leads.yaml      # anticipación y trayectoria del score
```

Las 56 corridas tardan ~25 min con hasta 16 procesos en paralelo (1–3 hilos cada uno, 24 núcleos).

## Cómo se mide

`scripts/report_v2_models.py` lee las predicciones fuera de fold de cada corrida y mide todo con la
misma cuenta:
- **Regla de alerta:** la de siempre, `k = 2` cortes seguidos ≥ τ.
- **Umbral exacto:** para cada presupuesto b (fracción de autos sanos con falsa alarma), τ es el menor
  umbral que deja ≤ b de los sanos alertados. **No se usa la grilla de 50 cuantiles de `train.py`:**
  con ~27 cortes por sano, un paso de la grilla mueve muchos autos juntos (f9-remedicion-v2.md §7).
- **Nulo de tamaño de bolsa** (regla 6): el mismo score permutado entre todas las filas, 200 veces.
- **Fuera de muestra:** τ se fija con los sanos de los otros folds y se aplica al que queda afuera.
- **AUC por auto** (score medio del auto), también dentro de mercado × motor (solo pares de la misma
  celda).
- **Bootstrap pareado por vehículo:** 1.000 réplicas, fallados y sanos remuestreados por separado y τ
  re-fijado en cada réplica. Se compara contra survival stacking, contra la celda y contra la
  CNN-LSTM de la tutora.
- **Piso de composición:** la tasa de fallas de la celda mercado × motor, calculada fuera de fold con
  los autos de los otros folds, usada como score constante por auto. **No es un modelo de uso.**

## Resultados

Detección de fallados (%) a 2 · 5 · 10 · 20 · 30% de autos sanos con falsa alarma, promedio de las 3
repeticiones (a 10%, ± desvío entre repeticiones). "×3 semillas" es el promedio por rango de las
semillas del modelo 42, 1 y 2: el mismo modelo con menos varianza, y el que se usaría.

| Modelo | PR-AUC | (a′) | AUC auto | AUC celda | 2% | 5% | 10% | 20% | 30% |
|---|---|---|---|---|---|---|---|---|---|
| **Secuenciales, ×3 semillas** | | | | | | | | | |
| GRU + TripSummary + estática completa | 0,160 | +0,006 | 0,82 | 0,65 | 14 | 27 | 42 ± 2 | 61 | 75 |
| CNN-LSTM + TripSummary + estática completa | 0,160 | +0,010 | 0,81 | 0,62 | 11 | 22 | 41 ± 6 | 62 | 77 |
| CNN-LSTM tutora con estática completa | 0,163 | +0,019 | 0,82 | 0,62 | 16 | 27 | 40 ± 3 | 68 | 80 |
| CNN-LSTM + TripSummary | 0,149 | +0,018 | 0,75 | 0,62 | 11 | 25 | 36 ± 4 | 56 | 65 |
| GRU + TripSummary | 0,140 | +0,005 | 0,76 | 0,65 | 12 | 23 | 36 ± 4 | 53 | 60 |
| GRU (signals + país) | 0,142 | +0,023 | 0,75 | 0,60 | 15 | 20 | 33 ± 3 | 49 | 61 |
| CNN-LSTM tutora (signals + país) | 0,143 | +0,017 | 0,74 | 0,61 | 13 | 20 | 31 ± 3 | 51 | 63 |
| **Mezclas por rango con la celda** | | | | | | | | | |
| Celda + CNN-LSTM trips ×3 | — | — | 0,82 | 0,60 | 12 | 30 | 43 ± 2 | 66 | 79 |
| Celda + GRU trips ×3 | — | — | 0,82 | 0,63 | 10 | 29 | 40 ± 4 | 67 | 78 |
| Celda + survival stacking | — | — | 0,80 | 0,55 | 8 | 19 | 34 ± 2 | 63 | 77 |
| **Ensambles por rango** | | | | | | | | | |
| SS + CNN-LSTM trips | 0,144 | +0,017 | 0,74 | 0,60 | 11 | 20 | 31 ± 5 | 52 | 63 |
| SS + LightGBM | 0,119 | −0,008 | 0,72 | 0,58 | 11 | 17 | 29 ± 1 | 45 | 58 |
| **Survival stacking y variantes** | | | | | | | | | |
| SS + motor/modelo | 0,141 | +0,025 | 0,77 | 0,57 | 5 | 13 | 28 ± 4 | 55 | 70 |
| Survival stacking (F3) | 0,125 | +0,018 | 0,72 | 0,57 | 7 | 14 | 24 ± 3 | 42 | 59 |
| SS embolsado ×10 | 0,119 | +0,021 | 0,71 | 0,55 | 5 | 11 | 24 ± 1 | 40 | 54 |
| SS horizonte 12.000 km | 0,118 | +0,006 | 0,74 | 0,58 | 8 | 11 | 23 ± 2 | 45 | 58 |
| SS grande (15 hojas, 600 árboles) | 0,131 | +0,020 | 0,73 | 0,58 | 8 | 16 | 23 ± 5 | 43 | 61 |
| SS horizonte completo (K2 sin ventana) | 0,111 | −0,004 | 0,75 | 0,60 | 6 | 12 | 22 ± 2 | 44 | 61 |
| SS + efecto aleatorio por auto (GPBoost) | 0,079 | +0,042 | 0,34 | 0,29 | 2 | 6 | 9 ± 2 | 17 | 21 |
| **Tabulares (53 features)** | | | | | | | | | |
| LightGBM + motor/modelo | 0,130 | −0,018 | 0,77 | 0,58 | 5 | 16 | 30 ± 1 | 53 | 71 |
| LightGBM (control F3) | 0,112 | −0,020 | 0,72 | 0,57 | 4 | 18 | 28 ± 1 | 42 | 55 |
| LightGBM grande | 0,108 | −0,026 | 0,71 | 0,57 | 7 | 16 | 28 ± 3 | 39 | 54 |
| Logística L2 | 0,123 | −0,008 | 0,71 | 0,58 | 9 | 15 | 23 ± 3 | 37 | 53 |
| LightGBM ordinal | 0,108 | −0,022 | 0,72 | 0,57 | 6 | 14 | 23 ± 3 | 38 | 54 |
| LightGBM embolsado ×10 | 0,107 | −0,016 | 0,71 | 0,54 | 6 | 13 | 22 ± 3 | 43 | 54 |
| Logística L1 | 0,126 | +0,002 | 0,71 | 0,58 | 9 | 14 | 21 ± 4 | 36 | 55 |
| **Pisos** | | | | | | | | | |
| Tasa de la celda mercado × motor (sin modelo) | — | — | 0,81 | 0,46 | 4 | 18 | 33 ± 4 | 62 | 62 |
| Piso posicional (solo `cut_odo`) | 0,067 | +0,025 | 0,42 | 0,43 | 0 | 3 | 6 | 15 | 20 |
| Nulo (score permutado, bolsa conservada) | — | — | — | — | 2 | 5 | 11 | 21 | 30 |

Las versiones de una sola semilla, las variantes `grande` y los ensambles de tres miembros están en
`experiments/report-v2-models/summary.csv`. Con el umbral fijado fuera de muestra, todos dan lo mismo
(±1 punto) y la tasa de falsas alarmas realizada queda en el presupuesto.

**Comparaciones pareadas** (diferencia en puntos, IC 95% del bootstrap por vehículo):

| | 5% | 10% | 20% |
|---|---|---|---|
| CNN-LSTM + trips ×3 − SS | +10,6 [−0,5; 18,3] | **+12,3 [1,7; 21,2]** | **+14,1 [3,5; 22,2]** |
| GRU + trips ×3 − SS | **+8,9 [0,2; 19,3]** | **+12,3 [1,7; 20,0]** | **+10,6 [0,2; 17,8]** |
| GRU + trips + estática ×3 − SS | **+13,3 [4,7; 24,2]** | **+18,8 [9,1; 27,9]** | **+18,5 [8,9; 27,4]** |
| GRU + trips + estática ×3 − celda | +9,4 [−5,2; 27,7] | +9,9 [−4,7; 20,7] | −1,5 [−8,2; 10,9] |
| GRU + trips ×3 − celda | +4,9 [−9,1; 23,0] | +3,5 [−11,6; 15,3] | −9,4 [−17,8; 1,5] |
| Celda + GRU trips ×3 − celda | +10,9 [−1,0; 20,2] | +7,9 [−0,7; 16,8] | +4,7 [−1,5; 12,3] |
| CNN-LSTM + trips ×3 − tutora ×3 | +4,4 [−4,7; 11,1] | +4,9 [−5,4; 12,8] | +4,9 [−2,7; 12,1] |
| CNN-LSTM tutora + estática ×3 − tutora ×3 | **+6,7 [0,2; 13,1]** | +8,9 [−0,7; 18,0] | **+16,3 [7,9; 25,4]** |

**Auditorías** (`audit_model.py`, semilla 42 de cada secuencial):

| | (a0) | (a′) | (b) calendario | veredicto |
|---|---|---|---|---|
| CNN-LSTM tutora (signals + país) | 0,056 vs 0,058 | +0,026 | **+0,028: marca** | aprueba |
| CNN-LSTM + trips | 0,060 vs 0,058 | +0,026 | +0,001 | aprueba |
| CNN-LSTM + trips + estática | 0,057 vs 0,058 | +0,009 | +0,018 | aprueba |
| GRU + trips | 0,061 vs 0,058 | +0,003 | **+0,025: marca** | aprueba |
| GRU + trips + estática | 0,061 vs 0,058 | +0,001 | +0,004 | aprueba |
| SS + motor/modelo | 0,056 vs 0,058 | +0,025 | +0,009 | aprueba |

## Lo que dicen

1. **Con más fallados, los secuenciales pasan adelante.**
   - CNN-LSTM y GRU con TripSummary ordenan autos dentro de la celda mercado × motor con AUC
     0,62–0,65, contra 0,57 de survival stacking y de los tabulares.
   - Le ganan a survival stacking con evidencia entre el 10% y el 20% de falsas alarmas.
   - En la entrega 1, con 53 fallados, la CNN-LSTM quedaba abajo de survival stacking (13% contra 19%
     al 5%, con el mismo reporte): los 53 agregados ya estaban saturados y la red necesitaba datos.
2. **El mejor según lo que se permita usar:**
   - **Con motor y modelo:** GRU + trips + estática ×3, que detecta 27 · 42 · 61% al 5 · 10 · 20%.
   - **Solo uso (país incluido):** CNN-LSTM + trips ×3 y GRU + trips ×3 empatan (25/36/56 contra
     23/36/53). **Se prefiere la CNN-LSTM:** la GRU marca (b) +0,025 y su (a′) es casi nula (+0,003).
3. **Ningún modelo le gana con evidencia a la tasa de la celda mercado × motor.**
   - Esa tasa, sin mirar un solo viaje, detecta 18 · 33 · 62%.
   - El uso suma ~5–10 puntos al 5–10%, y al 20% la celda empata o gana.
   - Esa tasa es el confusor de muestreo documentado en f9-entrega-v2 y f9-eda-v2: los fallados se
     sacaron por mercado y ENG_3 falla solo en BRA. Qué parte es física y qué parte es la lista de
     Ford **no se puede separar con estos datos**. Lo que destraba es la tasa real de fallas del DPF
     en la flota por mercado × motor: si coincide con la de la muestra, las estáticas entran sin
     reserva; si no, se reponderan las celdas.
4. **TripSummary le suma a la CNN-LSTM de la tutora ~4–5 puntos, pero no es significativo.**
   - Lo que sí cambia es la auditoría: con solo `signals` la (b) marca un atajo de calendario
     (+0,028), y con trips desaparece.
   - La estática completa (motor y modelo) sí le suma a la tutora con evidencia al 20% (+16 puntos).
5. **Survival stacking no empeoró respecto de la entrega 1: bajó el piso del azar.**

   | con la misma cuenta | detecta 5% | nulo 5% | sobre el nulo | detecta 20% | nulo 20% | sobre el nulo |
   |---|---|---|---|---|---|---|
   | SS, entrega 1 | 18,9 | 10,0 | +8,9 | 43,4 | 35,3 | +8,1 |
   | SS, v2 | 14,1 | 5,4 | +8,7 | 42,2 | 20,8 | **+21,4** |
   | CNN-LSTM tutora, entrega 1 | 13,2 | 10,4 | +2,8 | 42,8 | 35,9 | +6,9 |

   - En la entrega 1 los fallados tenían ~18 cortes y los sanos ~9, y esa asimetría sola "detectaba"
     10% al 5%. En v2 las bolsas están parejas (~29 y ~27).
   - Si a los sanos de v2 se los mira solo sus primeros 9 cortes, survival stacking detecta 33% al 5%
     (nulo 17%). El 15–17% del pitch era K2 con la etiqueta corregida por la ventana, que en v2 no
     existe.
   - La grilla de `train.py` baja SS de 14,1% (exacto) a 9,4% al 5%.
6. **Varianza por semilla.** La CNN-LSTM + trips da 24 · 20 · 19% al 5% con las semillas 42 · 1 · 2,
   y la GRU + trips 28 · 19 · 19%. Además, las redes no son deterministas entre cantidades de hilos:
   reentrenada en la auditoría, la CNN-LSTM + trips dio PR-AUC 0,164 contra 0,146. **Todo número de
   un secuencial se reporta como promedio de semillas.**
7. **Anticipación y *cuándo*** (`report_v2_leads.py`, al 10% de falsas alarmas):

   | | detectados | km antes: p25 · mediana · p75 | días (aprox.) mediana | alerta en el 1er corte |
   |---|---|---|---|---|
   | CNN-LSTM + trips ×3 | 49 | 3.200 · 7.800 · 11.800 | 90 | 22% |
   | GRU + trips ×3 | 49 | 3.700 · 7.500 · 12.300 | 80 | 12% |
   | GRU + trips + estática ×3 | 57 | 3.700 · 7.600 · 13.100 | 86 | 16% |
   | CNN-LSTM tutora ×3 | 42 | 2.800 · 4.600 · 10.900 | 73 | 10% |
   | Survival stacking | 32 | 3.900 · 8.300 · 14.100 | 90 | 9% |

   Rango percentil medio del score en los cortes de fallados según la distancia al evento:

   | | sanos | > 10.000 km | 3.500–10.000 km | ≤ 3.500 km |
   |---|---|---|---|---|
   | CNN-LSTM + trips ×3 | 0,45 | 0,57 | 0,63 | 0,70 |
   | GRU + trips ×3 | 0,45 | 0,57 | 0,62 | 0,69 |
   | Survival stacking | 0,45 | 0,54 | 0,64 | 0,70 |

   - El score sube al acercarse el evento y la primera alerta cae a mitad del historial, no en el
     primer corte: **hay señal del *cuándo***.
   - Pero el fallado ya está sobre los sanos a más de 10.000 km (el *qué auto*), y la anticipación
     tiene una dispersión enorme: p25 de 3.300 km, p75 de 12.000 km.
   - La frase defendible es "lo marcamos con ~7.500 km (2–3 meses) de margen", **no** "predecimos que
     falla en 7.500 km". El piso de 500 km es el gap del panel. Los días salen de km / (km por día del
     auto) y son aproximados.

## Qué no decide

- **El finalista sobre v2.** La lista natural para el preregistro:
  - CNN-LSTM + trips ×3 y GRU + trips ×3, con y sin estática;
  - contra survival stacking y contra la celda;
  - con umbral exacto, fuera de muestra y bootstrap pareado.
- **Si las estáticas entran al set base.** Depende de la tasa real por celda de Ford (punto 3).
- **Lo que no se corrió:**
  - TimesFM (sin ganancia en v1 y caro);
  - cure model y variantes con ventana del registro (en v2 no hay ventana);
  - W = 2.000 / 3.000 (cambian filas y folds);
  - CNN-LSTM embolsada.
