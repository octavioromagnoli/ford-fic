# Baseline de la tutora: CNN-LSTM con rama dinámica y rama estática

**Fecha:** 2026-09-19 · **Fase:** F3 · **Alcance:** dev del panel v1 (2.029 filas, 171
vehículos, 254 positivas, tasa base 0,125), mismos folds congelados. · **Reproduce:**

> Esta corrida histórica usó el umbral anterior de 5 puntos para `regen_drops`.
> El contrato vigente usa 15; hay que reconstruir el panel secuencial y reentrenar
> antes de comparar el número con corridas nuevas.

```bash
uv pip install -r requirements-dl.txt                                        # torch, opcional
python scripts/build_seq_panel.py --config configs/data/panel_seq_v1.yaml    # ~15 s
python scripts/train.py --config configs/exp_cnn_lstm.yaml                   # ~20 s en CPU
```

## Qué sabíamos de su solución y cómo se tradujo

Tres frases (11-09): *"usó una red convolucional long short memory"*, *"pero usó solo
dos datasets"*, *"una capa dinámica que procesaba los datos y después una capa estática
que no le llegaban todos los datos"*.

| frase | lectura | en el repo |
|---|---|---|
| red convolucional + LSTM | CNN-LSTM: Conv1D que extrae patrones locales y LSTM que resume el orden. No ConvLSTM (Shi 2015): sin eje espacial no aplica | `src/models/cnn_lstm.py` |
| solo dos datasets | `DynamicInformation` (`signals`) + `StaticInformation`. `TripSummary` no entra | canales solo de `signals` en `configs/data/panel_seq_v1.yaml` |
| capa dinámica que procesa los datos | la rama Conv1D → LSTM recibe la secuencia de la ventana | 20 bins de 50 km × 9 canales sobre `(c − 1.000, c]` |
| capa estática a la que no le llegan todos los datos | una densa que recibe **solo** las `static_*` y se concatena con el último estado de la LSTM (fusión tardía) | `static_hidden` en el YAML; `0` la apaga |

Dos cosas que ella probablemente usó y acá **no** entran, por lo que encontramos en
F1/F2: el marcador `Regenerations` y `DistanceBetweenRegenerations` (se cortan el
25-05-2026 y miden el calendario, [f2-eda-revision-y-features.md](f2-eda-revision-y-features.md)
§2.1; el builder los rechaza), y las estáticas `Engine`, `ModelSeries`, `ProductionDay` y
`daysUntilSale` (sesgo de muestreo o exposición). De la tabla estática entra solo
`SalesCountry_cd`.

## Cómo entra al pipeline sin tocar `train.py` ni `cv.py`

La secuencia viaja aplanada en un **panel secuencial**: las mismas 2.507 filas del panel
v1 (misma huella, así que el `splits.json` sirve tal cual), sin las 53 `feat_*` de
ventana y con 180 columnas `feat_seq_t{t}_c{k}_*` (`scripts/build_seq_panel.py`,
`src/features/sequences.py`). El preprocesador de `cv.py` las imputa y estandariza con el
train de cada fold y las deja primeras en la matriz, en orden `(t, canal)`; el modelo
reconstruye el tensor con `T` y `C` de `panel_seq_v1_meta.json` y toma el resto como
estáticas. El builder verifica que las únicas `feat_*` sean la secuencia, en ese orden.

Canales por bin (todos de `signals`): densidad de registros (`log1p`; `signals` emite cada
~4 min con el motor encendido, así que registros por km ≈ tiempo de motor por km), bin
observado, nivel medio y máximo de `Acumulation`, puntos de carga, regeneraciones (caídas
> 5 puntos, en orden temporal), y fracción de `Full`, `Overloaded`/`Over Limit` y
`Cleaning Automatically`. El 14,6% de los bins no tiene registros: conteos a 0, niveles
arrastrados dentro de la misma secuencia; ninguna fila queda vacía.

Red: Conv1D(9→16, k=3) → LSTM(16) ‖ Dense(estáticas→4) → Dense(20→16) → 1, dropout 0,3,
~3.000 parámetros. Adam, 40 épocas fijas, `pos_weight` = negativos/positivos del train del
fold. **Sin early stopping:** el modelo no ve `vehicle_id`, y una validación interna por
filas mezclaría cortes del mismo vehículo.

## Resultados (dev, CV agrupada, R = 1)

| modelo | datos | PR-AUC (lift) | ROC-AUC | Brier | detección · anticipación @ ≤50 FA/1000 | @ ≤200 FA/1000 |
|---|---|---|---|---|---|---|
| tasa base | — | 0,121 (0,97×) | 0,486 | 0,110 | — | — |
| LightGBM chico¹ | 53 agregados de `trips` + `signals` + mercado | **0,165** (1,32×) | **0,584** | 0,173 | 0,09 · 10.408 km | **0,43** · 7.865 km |
| **CNN-LSTM (tutora)** | secuencia de `signals` + mercado | 0,153 (1,23×) | 0,574 | 0,233 | **0,19** · 7.740 km | 0,32 · 10.269 km |

¹ `experiments/f3-lgbm-panel-v1`, corrido desde la rama `feat/model-timesfm-zeroshot` (el
builder `lgbm` no está en main).

PR-AUC por fold: 0,13 · 0,24 · 0,15 · 0,15 · 0,20. El Brier alto es el `pos_weight`, que
empuja las probabilidades hacia 0,5; no entra en la selección.

**Lectura.** Con una sola tabla dinámica (la que en el EDA menos anticipaba) y sin
agregados a mano, la red queda cerca del LightGBM de 53 features. Los canales, de a uno,
no separan (AUC por canal 0,45–0,54 sobre el promedio de la ventana): lo que aprende sale
de la combinación y del orden. No es sospechosamente alto (regla 6): está en el techo que
se midió para este panel.

## Auditorías

| chequeo | resultado | lectura |
|---|---|---|
| 5 semillas de la red (mismos folds) | PR-AUC 0,153–0,164 (media 0,159), ROC 0,562–0,588 (media 0,576) | la inicialización mueve ±0,01; comparar modelos con una sola semilla es ruido |
| sin rama estática (`static_hidden: 0`) | 0,153 / **0,558** | la rama estática aporta ~0,02 de ROC |
| `SalesCountry_cd` sola | ROC 0,434 (**0,566** invertida) | el mercado solo ya es buena parte de la señal |
| ~~`label` permutada dentro de cada vehículo (`default_rng(0)`, mismos folds) | 0,143 / 0,538 | la distancia hasta 0,574 es lo que aporta el cuándo~~ | **MAL LEÍDA · ver abajo** |
| ρ de cada canal (promedio de la ventana) con el mes del corte | \|ρ\| ≤ 0,11 | a nivel flota la densidad de registros baja de 404 a 320 por 1.000 km en el año y el nivel sube de 34 a 46; dentro del panel, el emparejado por odómetro × mes lo neutraliza |

La auditoría (b) de [f3-modelos-candidatos.md](../f3-modelos-candidatos.md) (sumar las
`aux_` de calendario) no aplica: este modelo no recibe `aux_`.

### Corrección (2026-09-20): la permutación intra-vehículo no aprobaba nada

**La fila tachada de arriba está mal leída, y la conclusión que sacaba no se sostiene.**
Decía que el 0,143-contra-0,153 medía el aporte del *cuándo*: que al destruir el *cuándo*
el PR-AUC bajaba, y que la distancia era lo que el modelo sabía de timing. Dos problemas:

1. **Esa permutación no es un null.** Conserva *qué* vehículos fallan —de donde sale casi
   todo el PR-AUC de este panel— y encima le saca a cada auto el ruido de qué ventana le
   tocó, así que entrena un ordenador de vehículos **mejor**. No hay valor esperado bajo
   "el modelo no aprendió nada" contra el cual leer el resultado. Re-medida ahora con
   `scripts/audit_model.py`, da **0,1663 contra 0,1553: sube**, que es lo que hace
   siempre en este panel (0,17–0,21 con tres semillas y los dos modelos).
2. **Permutaba `label` y no las features**, así que la referencia y la auditoría quedaban
   medidas contra etiquetas distintas y los dos números no eran comparables.

Re-auditado con los criterios vigentes (`python scripts/audit_model.py --config
configs/exp_cnn_lstm.yaml`), contra el panel secuencial reconstruido después de 1eab4a1:

| auditoría | resultado | veredicto |
|---|---|---|
| **(a0)** features permutadas entre *todas* las filas | 0,1182 vs tasa base 0,1252 (−0,0070) | **PASS**, no hay leakage |
| **(a')** score colapsado al promedio de su vehículo | 0,1553 → 0,1517 (**+0,0036**) | **PASS**, el orden dentro del vehículo suma |
| (a) features permutadas dentro del vehículo | 0,1553 → 0,1663 (+0,0110) | informativa, ni pass ni falla |
| (b) `aux_` de calendario | ROC +0,0101 | no hay salto |

**El modelo queda APROBADO**, pero por (a0) y (a'), no por lo que decía la fila vieja. Y
el aporte del *cuándo* es +0,0036, mucho más chico que lo que sugería el 0,143-contra-0,153.

Dos números que esa lectura no tenía y que cambian la conclusión de la sección anterior:

- **El techo de cohorte es 0,2627 (lift 2,10×)**: es lo que saca un modelo que solo sabe
  qué autos fallan. El 0,1553 de este modelo **está por debajo**, así que su PR-AUC por
  fila no demuestra anticipación. Tampoco el del LightGBM (0,1612).
- **PR-AUC entre fallados: 0,2801**, lift 1,07× sobre el azar de 0,2627.

Nota de comparabilidad: el 0,153 de las tablas de arriba es del panel secuencial
**anterior** a 1eab4a1. Reconstruido (el canal `regen_drops` usa el umbral que ese commit
cambió de 5 a 15 puntos), el mismo modelo da **0,1553** (`f3-cnn-lstm-tutora-regen15`).
Las tablas viejas se dejan como estaban; no se comparan con las corridas nuevas.

## Qué queda

- **Es el baseline, no el ganador.** Pierde por 0,01 de PR-AUC contra el LightGBM y la
  diferencia es del orden de la varianza por semilla. Para declarar algo: CV repetida
  (`n_repeats: 3`) y promedio de varias semillas de red. Ya hay R=3 del control y de
  survival stacking (`configs/exp_*_r3*.yaml`); falta la de este modelo, y es la que
  diría si su (a') de +0,0036 se distingue de cero — el control da −0,0055 ± 0,0040 y
  survival stacking +0,0162 ± 0,0033, así que +0,0036 cae justo en la zona dudosa.
- **Primera extensión obvia: sumar `TripSummary`.** Lo que anticipa según el EDA (idle,
  régimen térmico, velocidad) vive en `trips`. El builder ya acepta `source: trips` con
  las derivadas de `src/features/trips.py` (probado: 12 canales, mismas filas); es un YAML
  de panel nuevo y otro de experimento. Deja de ser "la solución de la tutora".
- **Ventana más larga que W.** Es donde una LSTM debería ganarle a los agregados, pero con
  L > W los bins previos al primer registro codifican el odómetro del corte; hay que
  resolver eso (máscara que no se pueda contar, o cortes solo con historia completa)
  antes de medir.
- El número de la rama estática depende casi todo de `SalesCountry_cd`; si se normaliza
  por mercado (idea 1.2 de F3), ese aporte se mueve.
