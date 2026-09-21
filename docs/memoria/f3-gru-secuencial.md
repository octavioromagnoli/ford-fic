# GRU sobre la ventana en orden: el orden en km no aporta nada que los 53 agregados no tengan

**Fecha:** 2026-09-20 · **Fase:** F3 · **Alcance:** dev del panel v1 reconstruido (2.029
filas, 171 vehículos, 254 positivas, tasa base 0,1252), CV agrupada con R = 3 y 5
semillas por modelo. · **Reproduce:**

```bash
uv pip install -r requirements-dl.txt                                         # torch, opcional

# 1 · paneles (el umbral de regeneración cambió: ver "Antes de medir nada")
python scripts/build_dataset.py   --config configs/data/panel_v1.yaml
python scripts/build_seq_panel.py --config configs/data/panel_seq_v1.yaml     # 9 canales de signals
python scripts/build_seq_panel.py --config configs/data/panel_seq_trips.yaml  # + 6 de trips
python scripts/make_splits.py     --config configs/data/splits_panel_v1_r3.yaml   # folds con R=3

# 2 · el modelo y sus controles, todos con los mismos folds
python scripts/train.py --config configs/exp_gru_seq.yaml          # GRU + atención, signals
python scripts/train.py --config configs/exp_gru_seq_trips.yaml    # GRU + atención, signals+trips
python scripts/train.py --config configs/exp_lgbm_panel_v1_r3.yaml # control: 53 agregados
python scripts/train.py --config configs/exp_cnn_lstm_r3.yaml      # control: baseline de la tutora
python scripts/train.py --config configs/exp_baserate_r3.yaml      # piso

# 3 · ablaciones y auditorías
python scripts/train.py --config configs/exp_gru_seq_nostatic.yaml   # sin rama estática
python scripts/train.py --config configs/exp_gru_seq_last.yaml       # pooling = último estado
python scripts/train.py --config configs/exp_gru_seq_mean.yaml       # pooling = promedio
python scripts/seed_spread.py --config configs/exp_gru_seq.yaml --seeds 5
python scripts/permutation_test.py --config configs/exp_gru_seq.yaml --splits data/processed/splits.json --n-perm 20

# 4 · barrido de ventana (W = 2.000 y 3.000, mismo G/H/Δ y mismo emparejado)
python scripts/build_dataset.py   --config configs/data/panel_w2000.yaml
python scripts/build_seq_panel.py --config configs/data/panel_seq_w2000.yaml
python scripts/train.py --config configs/exp_gru_seq_w2000.yaml
python scripts/train.py --config configs/exp_lgbm_w2000.yaml            # ídem con w3000
```

## Qué es

Una GRU unidireccional sobre la ventana `(c − W, c]` partida en bins de km
(`src/models/gru_seq.py`, builder `gru_seq`). Es la tercera lectura de la misma
ventana: el panel v1 la resume en 53 agregados, la CNN-LSTM de la tutora la recorre
con Conv1D + LSTM y se queda con el último estado, y esto la recorre con una
recurrente y la resume con **atención de un head**, que es la única de las tres formas
de pooling que puede darle peso a un bin del medio de la ventana.

La pregunta no es "¿anda una red?" sino **¿el orden en km aporta algo que el agregado
tira?**. Tres cosas la hacen medible:

- `pooling` es un hiperparámetro (`attention` / `last` / `mean`), así que la misma red
  con los mismos pesos iniciales se puede correr ignorando la posición del bin
  (`mean`) o mirando solo el último (`last`).
- `attention_weights()` devuelve el peso por bin: si la atención se va siempre al bin
  pegado al corte, lo aprendido es la posición en la serie, no la física.
- El barrido de W (1.000 / 2.000 / 3.000 km) mide si la recurrente aprovecha más
  historia mejor que los agregados, que es donde debería ganar si el orden importa.

**W, G, H y Δ no se implementan en el modelo.** Son del panel: la etiqueta ya viene con
el gap aplicado y a horizonte. `T`, `C`, `lookback_km` y `bin_km` salen de
`sequence_meta` (el `_meta.json` del panel secuencial), nunca de una constante.
Arquitectura: GRU(C→24) → atención(24→16→1) ‖ Dense(estáticas→4) → Dense(28→16) → 1,
dropout 0,3, **3.429 parámetros** con los 9 canales de `signals` (3.861 con 15). Adam, 40 épocas
fijas, `pos_weight` = negativos/positivos del train de cada fold. **Sin early
stopping**: el modelo no ve `vehicle_id` y una partición interna por filas mezclaría
cortes del mismo vehículo.

## Antes de medir nada: el panel se reconstruyó

El umbral de regeneración pasó de 5 a 15 puntos (commit `1eab4a1`,
[f2-umbral-regeneraciones.md](f2-umbral-regeneraciones.md)) **después** de que se
publicaran el panel secuencial y la corrida de la CNN-LSTM. Antes de comparar nada se
reconstruyeron `panel.parquet`, `panel_seq_v1`, `panel_w2000`, `panel_w3000` y los
paneles secuenciales de los tres, y se volvieron a correr los controles.

Lo que cambió y lo que no:

| | antes (umbral 5) | ahora (umbral 15) |
|---|---|---|
| filas de dev / positivas / tasa base | 2.029 / 254 / 0,1252 | **iguales** |
| folds congelados | `splits.json`, 5 folds | **los mismos** (el panel tiene la misma huella) |
| canal `regen_drops` y features de la familia B | caídas > 5 puntos | caídas > 15 puntos |
| CNN-LSTM, R = 1 (mismos folds) | 0,153 / ROC 0,574 | 0,155 / ROC 0,574 |

Por eso las corridas de este documento llevan el sufijo `-regen15` cuando re-miden algo
que ya estaba anotado: el número viejo no es comparable aunque el panel se llame igual.

Además, las corridas de acá usan **CV repetida R = 3** (`splits_r3.json`, que arma
`configs/data/splits_panel_v1_r3.yaml`). Es un archivo aparte a propósito: `train.py`
falla si el YAML declara un `n_repeats` distinto del que trae el `splits.json`
congelado, y tocar ese archivo obligaría a rehacer lo de todos. La **repetición 0 de
`splits_r3.json` son exactamente los folds de `splits.json`**, así que R = 3 complementa
las corridas de R = 1 en vez de invalidarlas.

## Resultados
