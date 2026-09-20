# Resultados por corrida

Generado por `python scripts/results.py table`; no editar a mano (las notas viven en cada `results/<corrida>.yaml`).
PR-AUC out-of-fold. Solo son comparables las corridas con la misma tasa base (mismas filas), splits y presupuesto de falsas alarmas:
un PR-AUC más bajo con otra tasa base puede ser un lift mayor. Las corridas `timesfm3` (zero-shot) no producen estas métricas.

| Corrida | Modelo | Panel | Tasa base | PR-AUC [IC fold] | Lift | Detección | Anticip. mediana | Config | Nota |
|---|---|---|---|---|---|---|---|---|---|
| `f3-lgbm-timesfm` | lgbm | panel_timesfm.parquet | 0.125 | 0.170 [0.168, 0.244] | 1.36× | 13% | 4,802 km | `configs/exp_lgbm_timesfm.yaml` ⚠ |  |
| `f3-lgbm-timesfm-r3` | lgbm | panel_timesfm.parquet | 0.125 | 0.169 [0.175, 0.219] | 1.35× | 9% | 11,906 km | `data/v364/cfg/exp_lgbm_timesfm_r3.yaml` ⚠ |  |
| `f3-lgbm-panel-v1` | lgbm | panel.parquet | 0.125 | 0.165 [0.173, 0.221] | 1.32× | 9% | 10,408 km | `configs/exp_lgbm_panel_v1.yaml` ⚠ |  |
| `f3-lgbm-panel-v1-r3` | lgbm | panel.parquet | 0.125 | 0.161 [0.169, 0.200] | 1.29× | 9% | 11,802 km | `data/v364/cfg/exp_lgbm_panel_v1_r3.yaml` ⚠ |  |
| `f3-lgbm-thermal-debt` | lgbm | panel_thermal_debt.parquet | 0.125 | 0.157 [0.158, 0.207] | 1.26× | 9% | 9,908 km | `configs/exp_lgbm_thermal_debt.yaml` ⚠ |  |
| `f3-lgbm-thermal-debt-r3` | lgbm | panel_thermal_debt.parquet | 0.125 | 0.154 [0.157, 0.187] | 1.23× | 11% | 11,855 km | `data/v364/cfg/exp_lgbm_thermal_debt_r3.yaml` ⚠ |  |
| `f3-cnn-lstm-tutora` | cnn_lstm | panel_seq_v1.parquet | 0.125 | 0.153 [0.144, 0.213] | 1.23× | 19% | 7,740 km | `configs/exp_cnn_lstm.yaml` ⚠ |  |
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
