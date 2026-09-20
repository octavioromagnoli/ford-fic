# Resultados por corrida

Generado por `python scripts/results.py table`; no editar a mano (las notas viven en cada `results/<corrida>.yaml`).
PR-AUC out-of-fold. Solo son comparables las corridas con la misma tasa base (mismas filas), splits y presupuesto de falsas alarmas:
un PR-AUC más bajo con otra tasa base puede ser un lift mayor. Las corridas `timesfm3` (zero-shot) no producen estas métricas.

| Corrida | Modelo | Panel | Tasa base | PR-AUC [IC fold] | Lift | Detección | Anticip. mediana | Config | Nota |
|---|---|---|---|---|---|---|---|---|---|
| `f3-lgbm-timesfm` | lgbm | panel_timesfm.parquet | 0.125 | 0.170 [0.168, 0.244] | 1.36× | 13% | 4,802 km | `configs/exp_lgbm_timesfm.yaml` ⚠ |  |
| `f3-lgbm-timesfm-r3` | lgbm | panel_timesfm.parquet | 0.125 | 0.169 [0.175, 0.219] | 1.35× | 9% | 11,906 km | `data/v364/cfg/exp_lgbm_timesfm_r3.yaml` ⚠ |  |
| `f3-lgbm-panel-v1` | lgbm | panel.parquet | 0.125 | 0.165 [0.173, 0.221] | 1.32× | 9% | 10,408 km | `configs/exp_lgbm_panel_v1.yaml` ⚠ |  |
| `f3-survival-stacking` | survival_stacking | panel_survival.parquet | 0.125 | 0.161 [0.164, 0.208] | 1.29× | 15% | 8,333 km | `configs/exp_survival_stacking.yaml` | Survival stacking (Craig/Zhong/Tibshirani 2021) sobre el panel v1 + feat_cut_odo: hazard discreto por bin de 500 km, score = 1-S(H|x). Empata al LightGBM en PR-AUC (0.161 vs 0.161 sobre el MISMO panel reconstruido con el umbral de regen de 15) pero llega distinto: Brier 0.114 vs 0.173, C-index 0.570 vs 0.562, deteccion 15% vs 7.5% a <=50 FA/1000, y es el unico cuyo orden DENTRO del vehiculo suma PR-AUC (+0.015 contra -0.007 del LightGBM). Detalle y auditorias en docs/memoria/f3-survival-stacking.md. |
| `f3-lgbm-panel-v1-regen15` | lgbm | panel.parquet | 0.125 | 0.161 [0.158, 0.213] | 1.29× | 8% | 12,322 km | `configs/exp_lgbm_panel_v1.yaml` | El control de f3-lgbm-panel-v1 recorrido sobre el panel RECONSTRUIDO despues del umbral de regeneraciones de 15 puntos (commit 1eab4a1): el panel con el que se midio el 0.165 historico es anterior al arreglo y tiene otras 7 columnas feat_*regen*. Mismo modelo, mismas filas, mismos folds: 0.1612. Es contra este numero, no contra el 0.165, que se comparan las corridas nuevas. |
| `f3-lgbm-panel-v1-r3` | lgbm | panel.parquet | 0.125 | 0.161 [0.169, 0.200] | 1.29× | 9% | 11,802 km | `data/v364/cfg/exp_lgbm_panel_v1_r3.yaml` ⚠ |  |
| `f3-lgbm-thermal-debt` | lgbm | panel_thermal_debt.parquet | 0.125 | 0.157 [0.158, 0.207] | 1.26× | 9% | 9,908 km | `configs/exp_lgbm_thermal_debt.yaml` ⚠ |  |
| `f3-lgbm-thermal-debt-r3` | lgbm | panel_thermal_debt.parquet | 0.125 | 0.154 [0.157, 0.187] | 1.23× | 11% | 11,855 km | `data/v364/cfg/exp_lgbm_thermal_debt_r3.yaml` ⚠ |  |
| `f3-cnn-lstm-tutora` | cnn_lstm | panel_seq_v1.parquet | 0.125 | 0.153 [0.144, 0.213] | 1.23× | 19% | 7,740 km | `configs/exp_cnn_lstm.yaml` | Baseline de la tutora (Conv1D+LSTM, ~3.000 parámetros) sobre panel_seq_v1: mismas filas y folds que el panel v1. Queda 0,012 de PR-AUC por debajo del LightGBM con la misma tasa base; detecta más (19% vs 9%) pero con menos anticipación. No lo reemplaza: 53 eventos no alcanzan para una red. |
| `f3-gpboost-survival` | gpboost_survival | panel_survival.parquet | 0.125 | 0.143 [0.139, 0.196] | 1.14× | 4% | 19,058 km | `configs/exp_gpboost_survival.yaml` | El mismo apilado con intercept aleatorio por vehiculo (GPBoost). NO mejora: 0.143 de PR-AUC contra 0.161 sin efecto aleatorio. El motivo esta medido: la varianza del intercept se estima en 9.4-10.3 (sd ~3.1 en logit) porque ~100 de los ~137 vehiculos de cada train no tienen ni un hazard positivo y su intercept se va a -inf (cuasi-separacion). El efecto aleatorio se come la senal en vez de regularizarla, y en validacion no sirve porque el vehiculo nunca fue visto (split agrupado). Fijando sigma2 en 0.1-2 recupera hasta 0.154, sigue por debajo. Ver docs/memoria/f3-survival-stacking.md. |
| `f3-baserate-panel-v1` | baserate | panel.parquet | 0.125 | 0.121 [0.120, 0.130] | 0.97× | 0% | — | `configs/exp_baserate.yaml` |  |
| `f3-lgbm-tfm-full` | lgbm | panel_tfm_full.parquet | 0.025 | 0.098 [0.080, 0.161] | 3.89× | 38% | 6,704 km | `configs/exp_lgbm_tfm_full.yaml` ⚠ |  |
| `f3-lgbm-tfm-window` | lgbm | panel_tfm_window.parquet | 0.025 | 0.093 [0.091, 0.167] | 3.69× | 37% | 9,182 km | `configs/exp_lgbm_tfm_window.yaml` ⚠ |  |
| `f3-lgbm-tfm-full-nostatic` | lgbm | panel_tfm_full_nostatic.parquet | 0.025 | 0.034 [0.032, 0.044] | 1.34× | 8% | 14,937 km | `configs/exp_lgbm_tfm_full_nostatic.yaml` ⚠ |  |
| `f3-lgbm-tfm-window-nostatic` | lgbm | panel_tfm_window_nostatic.parquet | 0.025 | 0.030 [0.032, 0.046] | 1.20× | 8% | 8,097 km | `configs/exp_lgbm_tfm_window_nostatic.yaml` ⚠ |  |
| `f0-dummy-baserate` | baserate | panel_dummy.parquet | 0.021 | 0.021 [0.021, 0.021] | 0.99× | 0% | — | `configs/exp_dummy.yaml` |  |
| `f3-timesfm3-zeroshot-accumulation-local` | timesfm3 | — | — | — [—, —] | —× | — | — | `configs/exp_timesfm3_zeroshot_accumulation.yaml` ⚠ |  |
| `f3-timesfm3-zeroshot-accumulation` | timesfm3 | — | — | — [—, —] | —× | — | — | `configs/exp_timesfm3_zeroshot_accumulation.yaml` ⚠ |  |
| `f3-timesfm3-zeroshot-badmsg-local` | timesfm3 | — | — | — [—, —] | —× | — | — | `configs/exp_timesfm3_zeroshot_badmsg.yaml` ⚠ |  |
| `f3-timesfm3-zeroshot-badmsg` | timesfm3 | — | — | — [—, —] | —× | — | — | `configs/exp_timesfm3_zeroshot_badmsg.yaml` ⚠ |  |
| `f3-timesfm3-zeroshot-regen-local` | timesfm3 | — | — | — [—, —] | —× | — | — | `configs/exp_timesfm3_zeroshot.yaml` ⚠ |  |
| `f3-timesfm3-zeroshot-regen` | timesfm3 | — | — | — [—, —] | —× | — | — | `configs/exp_timesfm3_zeroshot.yaml` ⚠ |  |

⚠ = el YAML no estaba commiteado al anotar; la config completa está embebida en `results/<corrida>.yaml`
(`python scripts/results.py show <corrida>` la imprime lista para guardarse como `configs/exp_*.yaml`).

## Cómo se anota una corrida

1. Entrenar: `python scripts/train.py --config configs/exp_<x>.yaml` (deja `experiments/<corrida>/`).
2. Anotar, con la nota de qué se probó y qué se concluyó:
   `python scripts/results.py log <corrida> --note "..."` (`--force` para reescribir; la nota se conserva).
3. Regenerar esta tabla: `python scripts/results.py table --out results/README.md`.
4. Commitear `results/<corrida>.yaml`, la tabla **y el `configs/exp_<x>.yaml`** de la corrida, en la misma rama.

Anotar justo después de entrenar: rama y commit del registro son los del checkout donde se anota.
Para la entrega: elegir la corrida ganadora en la tabla y sacar su config con `show`.
