# Resultados por corrida

Generado por `python scripts/results.py table`; no editar a mano (las notas viven en cada `results/<corrida>.yaml`).
PR-AUC out-of-fold. Solo son comparables las corridas con la misma tasa base (mismas filas), splits y presupuesto de falsas alarmas:
un PR-AUC más bajo con otra tasa base puede ser un lift mayor. Las corridas `timesfm3` (zero-shot) no producen estas métricas.

| Corrida | Modelo | Panel | Tasa base | PR-AUC [IC fold] | Lift | Detección | Anticip. mediana | Config | Nota |
|---|---|---|---|---|---|---|---|---|---|
| `f3-gru-seq-w3000` | gru_seq | panel_seq_w3000.parquet | 0.138 | 0.185 [0.177, 0.242] | 1.34× | 12% | 5,975 km | `configs\exp_gru_seq_w3000.yaml` | Barrido de ventana, W=3000 (T=60 bins de 50 km): la fila donde el barrido separa. 5 semillas: PR-AUC 0,1830 +- 0,0071 (lift 1,328 +- 0,051) / ROC 0,5910 +- 0,0220 contra el LightGBM de 53 agregados en el mismo panel y los mismos folds (lift 1,208 / ROC 0,5152). El GBM se degrada con la ventana larga —colapsa 3.000 km en un numero por feature— y la recurrente se sostiene. La diferencia (+0,08 de ROC, +0,12 de lift) esta muy por encima de la dispersion por semilla. |
| `f3-gru-seq-w2000` | gru_seq | panel_seq_w2000.parquet | 0.131 | 0.176 [0.175, 0.205] | 1.35× | 9% | 16,636 km | `configs\exp_gru_seq_w2000.yaml` | Barrido de ventana, W=2000 (T=40 bins de 50 km). 5 semillas: PR-AUC 0,1706 +- 0,0070 (lift 1,306 +- 0,053) / ROC 0,5888 +- 0,0117. Par: f3-lgbm-w2000 (lift 1,288 / ROC 0,5809). Empatan. |
| `f3-lgbm-timesfm` | lgbm | panel_timesfm.parquet | 0.125 | 0.170 [0.168, 0.244] | 1.36× | 13% | 4,802 km | `configs/exp_lgbm_timesfm.yaml` ⚠ |  |
| `f3-lgbm-timesfm-r3` | lgbm | panel_timesfm.parquet | 0.125 | 0.169 [0.175, 0.219] | 1.35× | 9% | 11,906 km | `data/v364/cfg/exp_lgbm_timesfm_r3.yaml` ⚠ |  |
| `f3-lgbm-w2000` | lgbm | panel_w2000.parquet | 0.131 | 0.168 [0.154, 0.211] | 1.29× | 17% | 7,355 km | `configs\exp_lgbm_w2000.yaml` | Par del barrido en W=2000: LightGBM con los 53 agregados, mismos folds. lift 1,288 / ROC 0,5809. |
| `f3-gru-seq-last` | gru_seq | panel_seq_v1.parquet | 0.125 | 0.168 [0.171, 0.237] | 1.34× | 6% | 15,408 km | `configs\exp_gru_seq_last.yaml` | GRU unidireccional sobre panel_seq_v1 con pooling = ultimo estado oculto: la variante que gana. 5 semillas x R=3: PR-AUC 0,1645 +- 0,0047 (lift 1,31x), ROC 0,5944 +- 0,0110. Le gana al LightGBM de 53 agregados (0,1590 +- 0,0008 / ROC 0,5851) por +0,0055 de PR-AUC, del orden de una sigma de la dispersion por semilla. Test de permutacion (label permutada a nivel vehiculo, 20 permutaciones, mismos folds): nulo ROC 0,517 +- 0,049 contra 0,600 real, p = 0,05. Sin la rama estatica pierde solo 0,0013: el salto NO viene del mercado. |
| `f3-lgbm-w3000` | lgbm | panel_w3000.parquet | 0.138 | 0.166 [0.147, 0.258] | 1.21× | 15% | 3,087 km | `configs\exp_lgbm_w3000.yaml` | Par del barrido en W=3000: LightGBM con los 53 agregados. lift 1,208 / ROC 0,5152, el peor ROC de la tabla entre los modelos que no son el piso: los agregados pierden la ventana larga. |
| `f3-gru-seq-w1000` | gru_seq | panel_seq_trips.parquet | 0.125 | 0.165 [0.160, 0.223] | 1.32× | 9% | 1,331 km | `configs\exp_gru_seq_w1000.yaml` | Barrido de ventana, W=1000: GRU last con 15 canales, folds R=1 del panel v1. 5 semillas: PR-AUC 0,1653 +- 0,0063 (lift 1,320 +- 0,051) / ROC 0,5865 +- 0,0124. Par: f3-lgbm-panel-v1 (lift 1,288 / ROC 0,5862). |
| `f3-gru-seq-mean` | gru_seq | panel_seq_v1.parquet | 0.125 | 0.165 [0.176, 0.204] | 1.32× | 11% | 3,573 km | `configs\exp_gru_seq_mean.yaml` | Ablacion de pooling: promedio de los estados (invariante a la posicion del bin). 5 semillas x R=3: 0,1588 +- 0,0070 / ROC 0,5770. Queda entre attention y last, asi que la posicion dentro de la ventana aporta algo, pero poco: la diferencia last - mean es 0,006 de PR-AUC. |
| `f3-lgbm-panel-v1` | lgbm | panel.parquet | 0.125 | 0.161 [0.158, 0.213] | 1.29× | 8% | 12,322 km | `configs\exp_lgbm_panel_v1.yaml` | Par del barrido en W=1000 (R=1) sobre el panel reconstruido con umbral 15: lift 1,288 / ROC 0,5862. |
| `f3-lgbm-panel-v1-r3` | lgbm | panel.parquet | 0.125 | 0.161 [0.169, 0.200] | 1.29× | 9% | 11,802 km | `data/v364/cfg/exp_lgbm_panel_v1_r3.yaml` ⚠ |  |
| `f3-gru-seq-last-nostatic` | gru_seq | panel_seq_v1.parquet | 0.125 | 0.160 [0.162, 0.209] | 1.28× | 17% | 6,419 km | `configs\exp_gru_seq_last_nostatic.yaml` | Ablacion obligatoria sobre la variante ganadora: GRU + pooling last SIN rama estatica. 5 semillas x R=3: 0,1632 +- 0,0022 / ROC 0,5881 contra 0,1645 +- 0,0047 / ROC 0,5944 con la rama. Apagar el mercado cuesta 0,0013 de PR-AUC y 0,006 de ROC: lo que el modelo aprende sale de la secuencia, no de SalesCountry_cd. Es lo contrario de lo que paso con la CNN-LSTM (donde la rama estatica valia ~0,02 de ROC). |
| `f3-cnn-lstm-r3-regen15` | cnn_lstm | panel_seq_v1.parquet | 0.125 | 0.160 [0.171, 0.218] | 1.28× | 9% | 10,269 km | `configs\exp_cnn_lstm_r3.yaml` | Control: la CNN-LSTM de la tutora re-medida sobre el panel RECONSTRUIDO (umbral de regeneracion 15, commit 1eab4a1) y con R=3. 5 semillas: 0,1580 +- 0,0023 / ROC 0,5699. El numero viejo (0,153 / 0,574, results/f3-cnn-lstm-tutora.yaml) se midio con el umbral de 5 y R=1: no es comparable con las corridas nuevas. |
| `f3-lgbm-panel-v1-r3-regen15` | lgbm | panel.parquet | 0.125 | 0.160 [0.163, 0.192] | 1.27× | 9% | 11,802 km | `configs\exp_lgbm_panel_v1_r3.yaml` | Control: LightGBM chico sobre los 53 agregados del panel v1 reconstruido (umbral 15) con R=3. 5 semillas: 0,1590 +- 0,0008 / ROC 0,5851 (practicamente determinista). Es la referencia contra la que se mide el GRU secuencial. |
| `f3-gru-seq-trips` | gru_seq | panel_seq_trips.parquet | 0.125 | 0.158 [0.156, 0.183] | 1.26× | 9% | 10,737 km | `configs\exp_gru_seq_trips.yaml` | Ablacion de canales: los 9 de signals + 6 de trips (idle, regimen termico, velocidad, distancia y duracion por viaje), mismas filas y folds. 5 semillas x R=3: 0,1541 +- 0,0066 / ROC 0,5658, contra 0,1515 +- 0,0012 / ROC 0,5712 con signals solo. Dentro del ruido: la tabla de viajes no aporta ORDEN que la red pueda usar, aunque sus agregados si le sumen al panel v1. |
| `f3-lgbm-thermal-debt` | lgbm | panel_thermal_debt.parquet | 0.125 | 0.157 [0.158, 0.207] | 1.26× | 9% | 9,908 km | `configs/exp_lgbm_thermal_debt.yaml` ⚠ |  |
| `f3-lgbm-thermal-debt-r3` | lgbm | panel_thermal_debt.parquet | 0.125 | 0.154 [0.157, 0.187] | 1.23× | 11% | 11,855 km | `data/v364/cfg/exp_lgbm_thermal_debt_r3.yaml` ⚠ |  |
| `f3-cnn-lstm-tutora` | cnn_lstm | panel_seq_v1.parquet | 0.125 | 0.153 [0.144, 0.213] | 1.23× | 19% | 7,740 km | `configs/exp_cnn_lstm.yaml` | Baseline de la tutora (Conv1D+LSTM, ~3.000 parámetros) sobre panel_seq_v1: mismas filas y folds que el panel v1. Queda 0,012 de PR-AUC por debajo del LightGBM con la misma tasa base; detecta más (19% vs 9%) pero con menos anticipación. No lo reemplaza: 53 eventos no alcanzan para una red. |
| `f3-gru-seq-signals` | gru_seq | panel_seq_v1.parquet | 0.125 | 0.153 [0.159, 0.206] | 1.22× | 9% | 3,210 km | `configs\exp_gru_seq.yaml` | El mismo GRU con el pooling por atencion (el default del builder). 5 semillas x R=3: PR-AUC 0,1515 +- 0,0012 / ROC 0,5712. Es la PEOR de las tres variantes de pooling: attention < mean (0,1588) < last (0,1645). attention_weights() explica por que: la atencion aprende un peso monotono hacia el corte (1,56x el uniforme en el ultimo bin, 0,63x en el mas viejo, entropia 0,885 del maximo), o sea una version difusa de last, con 416 parametros mas. No hay senal en el medio de la ventana que justifique la atencion. |
| `f3-gru-seq-nostatic` | gru_seq | panel_seq_v1.parquet | 0.125 | 0.151 [0.156, 0.200] | 1.20× | 6% | 15,408 km | `configs\exp_gru_seq_nostatic.yaml` | Ablacion obligatoria sobre la variante de atencion: sin la rama estatica (static_hidden 0). 1 semilla x R=3: 0,1506 / ROC 0,5688 contra 0,1532 / 0,5798 con la rama. La version con 5 semillas de esta ablacion se corrio sobre el pooling que gana (f3-gru-seq-last-nostatic). |
| `f3-baserate-r3-regen15` | baserate | panel.parquet | 0.125 | 0.122 [0.123, 0.128] | 0.97× | 6% | 11,285 km | `configs\exp_baserate_r3.yaml` | Piso sobre el panel reconstruido con R=3: 0,1218 / ROC 0,4874. Tasa base 0,1252. Todo modelo de la tabla le gana. |
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
