# El piso posicional: el odómetro solo le gana a los cuatro finalistas

**Fecha:** 2026-09-21 · **Fase:** F3 · **Alcance:** dev del panel v1 (2.029 filas, 171
vehículos, 254 positivas, tasa base 0,1252), mismos folds congelados.
**Reproduce:** `scripts/audit_positional_floor.py`, `configs/exp_positional_baseline.yaml`.

## El hecho

Dentro de un vehículo que falla, la etiqueta es una **función determinista de la posición
del corte**. Fracción positiva por posición desde el final del historial, sobre los 53
fallados de dev:

| posición desde el final | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8+ |
|---|---|---|---|---|---|---|---|---|
| fracción positiva | 1,00 | 1,00 | 0,98 | 0,93 | 0,92 | 0,87 | **0,00** | 0,00 |

No es casualidad: el historial de un vehículo con evento **termina en el evento**, y
`label = 1` es "el evento cae en `[corte+G, corte+G+H]`". "Estar cerca del final" y "estar
etiquetado" son la misma cosa. Un score que sea literalmente el odómetro ordena los cortes
de un fallado casi perfecto **sin saber nada de degradación**: ROC intra-vehículo 1,0000.

## El piso, medido

`score = cut_odo`, sin ajustar ni un parámetro, sobre las mismas filas y los mismos folds:

| métrica | piso posicional |
|---|---|
| PR-AUC por fila | **0,1766** (lift 1,41×) |
| PR-AUC entre fallados | **0,3301** (sobre 0,2627) |
| (a') aporte del cuándo | **+0,0872** |
| lift por vehículo (`mean`) | 1,018× |
| detección @ ≤50 FA/1000 | 7,55% · 7.385 km |

## Contra los cuatro finalistas

| corrida | PR-AUC fila | entre fallados | (a') | veh. `mean` | detección | anticip. |
|---|---|---|---|---|---|---|
| **piso posicional** | **0,1766** | **0,3301** | **+0,0872** | 1,018× | 7,55% | 7.385 km |
| survival stacking | 0,1613 ✗ | 0,2758 ✗ | +0,0152 ✗ | 1,646× ✓ | **15,1%** ✓ | 8.333 ✓ |
| lgbm control (regen15) | 0,1612 ✗ | 0,2871 ✗ | −0,0067 ✗ | 1,538× ✓ | 7,55% **=** | 12.322 ✓ |
| cnn-lstm (regen15) | 0,1553 ✗ | 0,2801 ✗ | +0,0036 ✗ | 1,368× ✓ | **17,0%** ✓ | 10.560 ✓ |
| gpboost survival | 0,1431 ✗ | 0,2977 ✗ | +0,0562 ✗ | 1,010× ✗ | 3,77% ✗ | 19.058 ✓ |

## Qué se concluye

**Tres métricas quedan descalificadas como evidencia.** El piso les gana a las cuatro
corridas en PR-AUC por fila, en PR-AUC entre fallados y en (a'). Un score sin información
de degradación las alcanza, así que **ninguna de las tres demuestra anticipación**. En
particular, **el PR-AUC por fila —la métrica de selección de modelo de todo F3— es
superada por una sola columna cruda**: toda la tabla de `results/` se decidió con un
número que no separa un modelo de un ordenamiento por kilometraje.

Es el mismo error que ya se había corregido en el otro eje. Ahora están los dos pisos:

| eje | piso que se usaba | piso correcto | ¿pasa alguien? |
|---|---|---|---|
| qué auto | tasa base 0,1252 | techo de cohorte **0,2627** | no |
| cuándo | cero | piso posicional **0,3301 / +0,0872** | no |

**Dos métricas sobreviven, y son las que hay que usar.** El piso posicional **pierde** en
el punto de operación y en el eje vehículo con agregación `mean`, y pierde por un motivo
que no se puede esquivar: un score posicional también alerta los últimos cortes de los
**sanos**, que no tienen ningún evento detrás. Esas falsas alarmas son el precio que un
atajo posicional no puede pagar, y por eso ahí la comparación significa algo.

- **Punto de operación** (detección a presupuesto fijo de falsas alarmas): CNN-LSTM 17,0%
  y survival stacking 15,1% contra 7,55% del piso. Le ganan 2:1. *(Con R=1. Con R=3:
  15,7% ± 0,9 y 11,9% ± 3,6; ver el punto 2 de abajo.)*
- **Eje vehículo con `mean`**: 1,65× y 1,54× contra 1,018× del piso.

## Tres consecuencias que no son obvias

**1. El control no le gana al piso — con R=1.** `lgbm-panel-v1` empata exactamente en
detección (7,55%, los mismos 4 vehículos de 53) y pierde en PR-AUC por fila. El modelo
contra el que se midieron todas las ablaciones de familias de features no supera al
odómetro. Cualquier diferencia de ±0,005 medida contra él hay que releerla.
*Corrección del 21-09 (ver la sección R=3 de abajo):* esos 4/53 son la repetición 0; con
R=3 el control detecta 4 / 8 / 7 vehículos, **11,9% ± 3,2**, y sí le gana al piso por el
mismo criterio que el CNN-LSTM. El R=1 era el sorteo, igual que en el punto 2.

**2. El CNN-LSTM estaba mal juzgado — pero no era el mejor detector.** Se descartó por
quedar 0,012 de PR-AUC por debajo del LightGBM, en la métrica degenerada, y eso sigue
siendo cierto. Lo que esta sección decía después no lo era: el 17,0% con R=1 **no
sobrevive a R=3**. Con los tres juegos de folds de `splits_r3.json` detecta 9 / 5 / 5
vehículos de 53, **11,9% ± 3,6**, con 10.366 ± 137 km de anticipación.

El porqué es de construcción: la repetición 0 de `splits_r3.json` son exactamente los
folds de `splits.json`, así que el 17,0% del R=1 era *una* de las tres repeticiones, la
que mejor le salió. Con 53 vehículos un vehículo son 1,9 puntos, y la diferencia entre 9 y
5 es de cuatro autos que cambian de fold. El 9,4% (5/53) que figura en
`results/f3-cnn-lstm-r3-regen15.yaml` tampoco es el número R=3: sale del score promediado
entre las tres repeticiones, que es un ensamble; el número R=3 es la media de las tres
detecciones por separado (`scripts/train.py::detection_by_repeat`).

Contra el piso, con el criterio preregistrado (le gana en detección por más de un desvío
entre repeticiones **y** en lift por vehículo con `mean`): 11,9 − 7,55 = **4,4 puntos contra
un desvío de 3,6** (1,2 desvíos) y **1,333× contra 1,018×**. Sigue siendo finalista, por
poco: en dos de las tres repeticiones detecta 5/53 contra 4/53 del piso, un vehículo. Y no
le gana a nadie más: empata con el control en detección (11,9% los dos) y pierde contra él
en lift (1,333× contra 1,494×) y en Brier; survival stacking le gana en las dos honestas.

### Las cuatro filas con R=3 (build `2026-09-20`, `splits_r3.json`, mismas 2.029 filas)

Media ± desvío entre las 3 repeticiones (desvío poblacional, `metrics.py::dispersion`). El
piso no tiene repeticiones: es el odómetro crudo, sin ajustar nada.

| | detección @ ≤50 FA/1000 | vehículos por repetición | anticipación mediana | lift veh. (`mean`) | (a′) | Brier |
|---|---|---|---|---|---|---|
| **piso posicional** | 7,55% | 4 (fijo) | 7.385 km | 1,018× | +0,0872 | — |
| **survival stacking** (referencia) | **15,7% ± 0,9** | 8 / 9 / 8 | 8.330 ± 22 km | **1,616× ± 0,061** | +0,0162 ± 0,0033 | **0,1123** |
| CNN-LSTM | 11,9% ± 3,6 | 9 / 5 / 5 | 10.366 ± 137 km | 1,333× ± 0,054 | +0,0080 ± 0,0051 | 0,2276 |
| control LGBM | 11,9% ± 3,2 | 4 / 8 / 7 | 9.370 ± 2.237 km | 1,494× ± 0,049 | −0,0055 ± 0,0040 | 0,1711 |

Para completar: PR-AUC entre fallados 0,2896 / 0,2857 / 0,2871 contra 0,3301 del piso y
C-index 0,582 / 0,531 / 0,559 (survival stacking / CNN-LSTM / control). Auditorías del
CNN-LSTM con R=3 (`experiments/f3-cnn-lstm-r3-regen15/audit.json`): (a0) 0,1208 contra
0,1252, PASS; (a′) +0,0114 sobre el score promediado, PASS; **(b) +0,0216 de ROC, marca**
(ver [f3-cnn-lstm-tutora.md](f3-cnn-lstm-tutora.md): el +0,0101 que tenía registrado se
midió con la secuencia corrida).

Un dato que no es una comparación contra la etiqueta pero pesa para armar un ensamble: la
correlación de rango entre los scores OOF de survival stacking y el CNN-LSTM es **0,24–0,36**
por fila (0,32–0,45 por vehículo), y con el control **0,77** (0,85–0,87). El control y
survival stacking leen los mismos 53 agregados y ordenan casi igual; el CNN-LSTM lee la
secuencia y ordena distinto.

**3. El piso re-confirma, por un camino independiente, que `mean` es la única agregación
honesta.** Sobre el panel posicional las agregaciones dependientes del tamaño de bolsa se
inflan igual que en [f3-mil-agregacion-vehiculo.md](f3-mil-agregacion-vehiculo.md):
`noisy_or` 1,72×, `topk` 1,41×, `max` 1,33× — y `mean` 1,00×, que es lo correcto para un
score sin señal. Dos hallazgos independientes, la misma conclusión.

## Por qué la corrida registrada da otro número

`results/f3-positional-baseline.yaml` registra **0,1426**, no 0,1766, porque pasa por
`train.py`: el `Pipeline` de `cv.py` z-scorea las numéricas con el train de cada fold, así
que un LightGBM sobre `feat_cut_odo` es un modelo *ajustado* sobre la posición y paga
varianza. El piso fuerte es el odómetro crudo con ranking global, que no ajusta nada y por
eso no puede sobreajustar; se mide con `scripts/audit_positional_floor.py`. La corrida
registrada queda como la versión "dentro del protocolo"; el número que hay que citar como
piso es el del auditor.

## Cómo se reproduce

```bash
source .venv/bin/activate
FORD_DATA_DIR=data/v364 python scripts/build_positional_panel.py \
  --config configs/data/panel_positional.yaml
FORD_DATA_DIR=data/v364 WANDB_MODE=disabled python scripts/train.py \
  --config configs/exp_positional_baseline.yaml

# Y la comparación contra cualquier corrida ya entrenada:
FORD_DATA_DIR=data/v364 python scripts/audit_positional_floor.py f3-survival-stacking

# La tabla R=3 (21-09). El CNN-LSTM y el control se re-miden desde sus predicciones, sin
# reentrenar: `rescore_run.py` usa la misma función que `train.py` y falla si lo que el
# metrics.json ya traía no se reproduce. `--carry-from` trae `aux_km_observed_after_cut`
# (C-index) del panel reconstruido, que es idéntico al que usaron en todas las demás columnas.
python scripts/rescore_run.py f3-cnn-lstm-r3-regen15 \
  --carry-from data/rebuild-0921/processed/panel_seq_v1.parquet
python scripts/rescore_run.py f3-lgbm-panel-v1-r3-regen15 \
  --carry-from data/rebuild-0921/processed/panel.parquet
FORD_DATA_DIR=data/rebuild-0921 WANDB_MODE=disabled python scripts/audit_model.py \
  --config configs/exp_cnn_lstm_r3.yaml --out experiments/f3-cnn-lstm-r3-regen15/audit.json
python scripts/audit_positional_floor.py f3-cnn-lstm-r3-regen15
```

`audit_positional_floor.py` mide sobre el score **promediado** entre repeticiones (lo que
guarda `predictions.parquet` en `score`): para el CNN-LSTM da 9,4% de detección, que es
el del ensamble de las tres pasadas. La tabla R=3 de arriba usa la media de las tres
detecciones por separado, que es la regla de `train.py`.

## Qué queda abierto

- ~~**R=3 del CNN-LSTM**~~ — hecho el 21-09 (punto 2): 11,9% ± 3,6, sin re-entrenar,
  desde sus predicciones (`scripts/rescore_run.py`). No le discute el lugar a survival
  stacking; sigue como finalista del par para el ensamble.
- **Re-leer la tabla de `results/`** completa contra los dos pisos: es probable que varias
  diferencias históricas no sobrevivan.
- El atajo de calendario que survival stacking marca en la auditoría (b) (+0,0230 de ROC,
  por encima del umbral de 0,02) sigue sin resolver.
