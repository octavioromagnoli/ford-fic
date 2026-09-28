# Preregistro F12 · ¿Se puede subir la detección de la GRU al 5–10% de falsas alarmas?

**Fecha:** 2026-09-28, escrito **antes** de medir. **Pedido:** subir la detección de 42% (al 10%) hacia
60% al 5–10%. **Alcance:** solo dev v2 (446 autos), `splits_r3.json`, test sin tocar.
Se mide con `scripts/report_v2_models.py`, con la misma cuenta que F9/F10: umbral exacto, k = 2 y
promedio de R = 3.

## Lo que hay que saber antes (la aritmética del 60%)

- **La celda ya da 33% al 10%.** Tres celdas (CHL×ENG_2, COL×ENG_2 y BRA×ENG_3) tienen 117 de los 135
  fallados y ~101 sanos. Todo lo que se gana por encima de eso sale de ordenar autos **dentro** de esas
  celdas.
- **Qué pide el 60%:** al 10% se alertan ~29 sanos. Detectar 60% (81 fallados) con ≤ 29 sanos alertados
  pide, dentro de las tres celdas, ~69% de detección con ≤ ~29% de falsas alarmas. Eso es un AUC dentro
  de la celda de ~0,8. Hoy está en 0,62–0,65, en **todas** las arquitecturas medidas. Al 5% pide ~0,85.
- **Lo que ya no movió la aguja:**
  - hiperparámetros (sweep F10);
  - horizonte largo y embolsado (SS h12k, bag10);
  - media acumulada (F6);
  - cure model;
  - desvío contra la historia.
- **El ruido de dev:** con 135 fallados, el error estándar de una detección de ~42% es ~4 puntos. Si
  se prueban 20 ideas y se queda la mejor, se gana ~+6–8 por azar. Por eso la lista es cerrada y
  corta.

**Expectativa declarada: +0 a +5 puntos al 10%. El 60% no se espera con estos datos.**

## Lista cerrada

| id | qué | por qué podría sumar |
|---|---|---|
| P0 | referencia: GRU + TripSummary + estática completa, ×3 semillas (42, 1, 2), reproducida en esta máquina | control; tiene que dar 42 ± ~4 al 10% |
| P1 | la misma GRU, ×10 semillas (42, 1–9), promedio por rango | F10: la semilla mueve más que los hiperparámetros. Promediar más semillas baja la varianza del orden |
| P2 | ensamble heterogéneo por rango: GRU trips-est ×3 + CNN-LSTM trips-est ×3 + CNN-LSTM tutora-est ×3 | tres arquitecturas con AUC por auto parecido pero errores distintos |
| P3 | P2 mezclado por rango con la tasa de la celda (mitad y mitad) | "celda + CNN-LSTM" fue la mejor fila al 5% en F9 (30%) |

No se agrega nada a esta lista después de ver resultados. Una idea nueva va a otro preregistro.

## Regla

- **Métrica primaria:** detección al 10%. Secundaria: al 5%. Las dos con umbral exacto y promedio de 3
  repeticiones.
- **Se adopta un Pk sobre P0 solo si** el bootstrap pareado por vehículo (1.000 réplicas) deja la
  diferencia al 10% con IC 95% por encima de 0, **y** el AUC dentro de mercado × motor no baja.
- Si ninguno pasa, se queda P0 y el techo se reporta como hallazgo: el límite es de información
  (eventos y features), no de modelo.
- **El test no se abre para esto.**

---

## Resultado (28-09, medido después de escribir lo de arriba)

```bash
python scripts/build_dataset.py --config configs/data/panel_v2.yaml             # + panel_v2_estaticas.yaml
python scripts/make_splits.py --config configs/data/splits_panel_v2_r3.yaml
python scripts/build_seq_panel.py --config configs/data/panel_seq_trips_v2_estaticas.yaml   # + panel_seq_v2_estaticas.yaml
WANDB_MODE=disabled python scripts/train.py --config configs/exp_v2all_gru_trips_estaticas{,_s1,...,_s9}_r3.yaml
WANDB_MODE=disabled python scripts/train.py --config configs/exp_v2all_cnn_lstm_{trips,signals}_estaticas{,_s1,_s2}_r3.yaml
WANDB_MODE=disabled python scripts/ensemble_rank.py --config configs/exp_f12_p1_gru_seeds10.yaml   # y exp_f12_p2_hetero9.yaml
python scripts/report_v2_models.py --config configs/report_f12.yaml             # -> experiments/report-f12/
```

- **Dispositivo:** las 10 GRU corrieron en CPU. Las 6 CNN-LSTM corrieron en la GPU (RTX 4050, torch
  2.14.0+cu126), con los mismos YAML y `device: cuda`. En GPU no son determinísticas al bit.
- **P0 reproduce F9** en esta máquina: 28 · 43 · 61% contra 27 · 42 · 61%.

Detección (%) a 2 · 5 · 10 · 20 · 30% de sanos con falsa alarma, promedio de R = 3:

| | AUC auto | AUC celda | 2% | 5% | **10%** | 20% | 30% |
|---|---|---|---|---|---|---|---|
| P0 · GRU ×3 | 0,82 | 0,66 | 14 | 28 | **43** | 61 | 74 |
| P1 · GRU ×10 | 0,82 | 0,66 | 16 | 27 | **43** | 62 | 73 |
| P2 · GRU ×3 + CNN-LSTM trips ×3 + tutora ×3 | 0,83 | 0,64 | 16 | 28 | **42** | 67 | 82 |
| P3 · P2 + celda | 0,83 | 0,62 | 10 | 26 | **45** | 71 | 82 |
| celda (sin modelo) | 0,82 | 0,46 | 4 | 18 | **33** | 62 | 62 |

Bootstrap pareado por vehículo contra P0 (diferencia en puntos, IC 95%):

| | 5% | **10%** | 20% |
|---|---|---|---|
| P1 | −1 [−6; +5] | **0 [−6; +4]** | +1 [−4; +5] |
| P2 | 0 [−7; +7] | **−1 [−7; +8]** | +6 [0; +12] |
| P3 | −2 [−16; +9] | **+3 [−9; +11]** | +10 [0; +17] |

**Veredicto: ninguno pasa. Se queda P0.**
- Al 10%, ningún candidato separa la diferencia del cero.
- Al 5% y al 10%, ninguno se mueve más de ±3 puntos: estamos en el techo que anticipaba la
  aritmética. El AUC dentro de la celda no sube en ninguno (0,62–0,66).
- **Al 20% y 30%, el ensamble heterogéneo sí suma:** P2 +6 y P3 +10, con p ≈ 0,03 y el IC tocando
  el cero. No es la métrica preregistrada, así que **no se adopta por esto**. Queda como hipótesis
  para un preregistro de un punto de operación del 20%.
- **Más semillas no mueven nada** (P1 = P0). Tres semillas ya alcanzan.
