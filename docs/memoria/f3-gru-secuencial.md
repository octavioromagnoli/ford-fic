# GRU sobre la ventana en orden: con el último estado le gana por poco al LightGBM; la atención y la ventana larga no ayudan

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
python scripts/train.py --config configs/exp_lgbm_panel_v1_r3_regen15.yaml # control: 53 agregados
python scripts/train.py --config configs/exp_cnn_lstm_r3.yaml      # control: baseline de la tutora
python scripts/train.py --config configs/exp_baserate_r3.yaml      # piso

# 3 · ablaciones y auditorías
python scripts/train.py --config configs/exp_gru_seq_nostatic.yaml   # sin rama estática
python scripts/train.py --config configs/exp_gru_seq_last.yaml       # pooling = último estado
python scripts/train.py --config configs/exp_gru_seq_mean.yaml       # pooling = promedio
python scripts/train.py --config configs/exp_gru_seq_last_nostatic.yaml  # last sin rama estática
python scripts/seed_spread.py --config configs/exp_gru_seq_last.yaml --seeds 5   # ídem con cada YAML
python scripts/permutation_test.py --config configs/exp_gru_seq_last.yaml --splits data/processed/splits.json --n-perm 20

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

## Resultados (dev, CV agrupada, R = 3, media ± desvío sobre 5 semillas)

Todas las filas: mismo panel de filas, mismos 15 folds (`splits_r3.json`), tasa base 0,1252.
El número de cada semilla es el promedio de las 3 repeticiones (`scripts/seed_spread.py`).

| modelo | entrada | PR-AUC | lift | ROC-AUC |
|---|---|---|---|---|
| tasa base | — | 0,122 | 0,97× | 0,487 |
| LightGBM chico | 53 agregados + mercado | 0,159 ± 0,001 | 1,27× | 0,585 ± 0,002 |
| CNN-LSTM (tutora) | secuencia `signals` + mercado | 0,158 ± 0,002 | 1,26× | 0,570 ± 0,005 |
| GRU · `attention` (default) | secuencia `signals` + mercado | 0,152 ± 0,001 | 1,21× | 0,571 ± 0,005 |
| GRU · `attention` | `signals` + `trips` (15 canales) | 0,154 ± 0,007 | 1,23× | 0,566 ± 0,017 |
| GRU · `mean` | secuencia `signals` + mercado | 0,159 ± 0,007 | 1,27× | 0,577 ± 0,010 |
| **GRU · `last`** | secuencia `signals` + mercado | **0,165 ± 0,005** | **1,31×** | **0,594 ± 0,011** |
| GRU · `last`, sin rama estática | secuencia `signals` | 0,163 ± 0,002 | 1,30× | 0,588 ± 0,007 |

Punto de operación (≤ 50 FA/1000 sanos, semilla 42): LightGBM detecta 9,4% con 11.802 km
de anticipación mediana; GRU `last` 5,7% · 15.408 km; GRU `last` sin estáticas 17,0% ·
6.419 km. Con 53 vehículos, un vehículo es 1,9 puntos de detección: la curva del pitch
no discrimina entre estos modelos.

**Lectura.**

- **La única variante que le gana al LightGBM es `last`**, por +0,005 de PR-AUC y +0,009 de
  ROC. Sus 5 semillas están todas por encima de la media del LightGBM en PR-AUC salvo una
  (rango 0,158–0,170 contra 0,158–0,160). Es una ventaja chica y consistente, no un salto.
- **La hipótesis del default se cae:** la atención es el peor pooling (`last` > `mean` >
  `attention`). La auditoría de pesos explica por qué: la atención aprende a mirar el
  final de la ventana, pero difusa (ver abajo). Lo que anticipa está pegado al corte y la
  atención solo lo diluye con 400 parámetros más.
- **Los canales de `trips` no le suman orden a la red** (0,154 vs 0,152, dentro del ruido),
  aunque sus agregados sí están en el LightGBM. Lo que el EDA encontró en `trips` (idle,
  régimen térmico) se captura igual de bien resumido.
- **La rama estática no es la fuente** (0,165 → 0,163 al apagarla). El sesgo de muestreo de
  F1 por mercado no explica la ventaja de `last`.

## Barrido de ventana (modelo × W)

Mismo G = 500, H = 3.000, Δ = 500 y mismo emparejado; cada W tiene su panel, sus filas y
sus folds congelados (R = 1). GRU con los 15 canales `signals + trips`; LightGBM con los
53 agregados (5 semillas mueven ±0,001). Entre ventanas solo se comparan ROC y lift: la
tasa base cambia.

| W | filas dev · tasa base | LightGBM lift · ROC | GRU `attention` lift · ROC (5 semillas) | GRU `last` lift · ROC (**1 semilla**) |
|---|---|---|---|---|
| 1.000 km | 2.029 · 0,125 | 1,29× · 0,586 ¹ | 1,23× ± 0,05 · 0,566 ± 0,017 ² | 1,32× · 0,587 |
| 2.000 km | 1.807 · 0,131 | 1,29× · 0,581 | 1,17× ± 0,06 · 0,566 ± 0,018 | 1,35× · 0,603 |
| 3.000 km | 1.567 · 0,138 | 1,21× · 0,515 | 1,20× ± 0,07 · 0,553 ± 0,018 | 1,34× · 0,583 |

¹ repetición 0 de la corrida R = 3 (= folds de `splits.json`). ² R = 3.
Los configs `exp_gru_seq_w{1000,2000,3000}.yaml` usan hoy `pooling: last` (commit
`08ba21f`): el barrido tiene que medir la ventana con la variante que eligió la ablación,
no mezclar ventana y pooling. La columna de `attention` es la corrida previa a ese cambio.

**Lectura.**

- **El LightGBM se degrada con la ventana larga** (ROC 0,586 → 0,515 en W = 3.000): los
  agregados sobre 3.000 km diluyen lo que pasa cerca del corte.
- **La GRU `last` se sostiene** en las tres ventanas (lift 1,32–1,35×) y en W = 3.000 le
  saca 0,13× de lift y 0,07 de ROC al LightGBM. Es el único lugar del barrido donde la
  diferencia es mayor que el ruido medido.
- **Pero no mejora con más historia**: 1,32× → 1,35× → 1,34× es plano. La red no está
  encontrando nada en los km viejos; es robusta a que se los agreguen, que es distinto.
  Consistente con la auditoría de atención: lo que anticipa está en los últimos cientos
  de km.
- **Una semilla no es un resultado.** La columna `last` es de una semilla por ventana; en
  W = 3.000 la semilla 42 de `attention` daba ROC 0,581 y la media de 5 fue 0,553. Hasta
  correr las 5 semillas, la columna `last` es una señal a confirmar, no una conclusión.

## Auditorías

| chequeo | resultado | lectura |
|---|---|---|
| test de permutación, GRU `last` (label permutada a nivel vehículo, 20 permutaciones, folds de R = 1) | real ROC 0,600 · nulo 0,517 ± 0,049 (p95 0,600, máx 0,611) → **p = 0,05**; lift real 1,30× · nulo 1,09× ± 0,17 → p = 0,10 | en el borde. Con 20 permutaciones la resolución de p es 0,05: el real iguala el p95 del nulo. Hay señal débil, no una demostración. Este nulo (0,517) es más bajo que el 0,555 del LightGBM con 53 agregados: la red explota menos el atajo de posición |
| dónde mira la atención (5 folds, 2.029 filas de validación) | peso monótono creciente hacia el corte: 0,63× el uniforme en el bin más viejo, 1,56× en el último; entropía 0,885 del máximo; la mitad vieja de la ventana se lleva 38% del peso | aprende "el final", difuso. Nunca concentra en un bin del medio: no hay una señal intermedia que la atención encuentre y `last` pierda |
| rama estática apagada (`static_hidden: 0`), `last`, 5 semillas | 0,163 ± 0,002 / ROC 0,588 (vs 0,165 / 0,594) | el mercado aporta ~0,006 de ROC. La ventaja sobre el LightGBM no viene del sesgo de muestreo |
| ¿sospechosamente alto? (regla 6) | ROC máximo 0,603 (una semilla); media 0,594 | por debajo del umbral de 0,60–0,65 en media. No hay canal casi-definicional: el nivel del filtro (`acum_*`) ya estaba en la CNN-LSTM con el mismo gap |
| dispersión por semilla | con R = 3, ±0,001–0,007 de PR-AUC | R = 3 baja mucho la varianza por semilla que se había medido con R = 1 (±0,01) |
| reconstrucción del panel | umbral 5 → 15: filas, folds y tasa base idénticos; CNN-LSTM R = 1 pasa de 0,153 a 0,155 | los números viejos no se mezclan con los nuevos; los controles se re-midieron (`-regen15`) |

## Qué queda

- **5 semillas del barrido con `pooling: last`**
  (`python scripts/seed_spread.py --config configs/exp_gru_seq_w{1000,2000,3000}.yaml`).
  Es lo que decide si la ventaja en W = 3.000 es real.
- **Más permutaciones** para `last` (100 en vez de 20): con p = 0,05 al borde, la
  resolución de la prueba es el cuello.
- **El default de `gru_seq` sigue siendo `attention`**, documentado como hipótesis
  refutada en el docstring. Si se confirma lo de arriba, cambiarlo a `last` es una línea.
- **Ventana más corta, no más larga:** si lo que anticipa está pegado al corte, lo próximo
  que vale la pena medir es W = 500 o bins de 25 km en los últimos 500 km.
- **Ensamble LightGBM + GRU `last`:** tienen entradas distintas (agregados vs. secuencia) y
  puntos de operación distintos; es barato y no se midió.
- **No se tocó el test** (`test_split.json`): todo lo de acá es dev.
