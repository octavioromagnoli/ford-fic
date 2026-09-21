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
  y survival stacking 15,1% contra 7,55% del piso. Le ganan 2:1.
- **Eje vehículo con `mean`**: 1,65× y 1,54× contra 1,018× del piso.

## Tres consecuencias que no son obvias

**1. El control no le gana al piso.** `lgbm-panel-v1` empata exactamente en detección
(7,55%, los mismos 4 vehículos de 53) y pierde en PR-AUC por fila. El modelo contra el que
se midieron todas las ablaciones de familias de features no supera al odómetro. Cualquier
diferencia de ±0,005 medida contra él hay que releerla.

**2. El CNN-LSTM estaba mal juzgado.** Se descartó por quedar 0,012 de PR-AUC por debajo
del LightGBM — en la métrica degenerada. En la métrica honesta es **el mejor detector del
proyecto**: 17,0% con 10.560 km de anticipación. Le falta R=3 para poder declararlo (17,0%
son 9 vehículos de 53 y 15,1% son 8: la diferencia con survival stacking es un vehículo y
no significa nada todavía). Lo que sí se puede declarar es que los dos le ganan al piso.

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
```

## Qué queda abierto

- **R=3 del CNN-LSTM**, que es lo único que falta para poder elegir finalista entre él y
  survival stacking sobre la métrica honesta.
- **Re-leer la tabla de `results/`** completa contra los dos pisos: es probable que varias
  diferencias históricas no sobrevivan.
- El atajo de calendario que survival stacking marca en la auditoría (b) (+0,0230 de ROC,
  por encima del umbral de 0,02) sigue sin resolver.
