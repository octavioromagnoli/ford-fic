# Baseline de la tutora: CNN-LSTM con rama dinámica y rama estática

**Fecha:** 2026-09-19 · **Fase:** F3 · **Alcance:** dev del panel v1 (2.029 filas, 171
vehículos, 254 positivas, tasa base 0,125), mismos folds congelados. · **Reproduce:**

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
| `label` permutada dentro de cada vehículo (`default_rng(0)`, mismos folds) | 0,143 / 0,538 | no cae a 0,5, y no tiene por qué: la permutación conserva *qué* vehículos fallan (el mercado lo sabe) y destruye *cuándo*. La distancia hasta 0,574 es lo que aporta el cuándo |
| ρ de cada canal (promedio de la ventana) con el mes del corte | \|ρ\| ≤ 0,11 | a nivel flota la densidad de registros baja de 404 a 320 por 1.000 km en el año y el nivel sube de 34 a 46; dentro del panel, el emparejado por odómetro × mes lo neutraliza |

La auditoría (b) de [f3-modelos-candidatos.md](../f3-modelos-candidatos.md) (sumar las
`aux_` de calendario) no aplica: este modelo no recibe `aux_`.

## Qué queda

- **Es el baseline, no el ganador.** Pierde por 0,01 de PR-AUC contra el LightGBM y la
  diferencia es del orden de la varianza por semilla. Para declarar algo: CV repetida
  (`n_repeats: 3`) y promedio de varias semillas de red.
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
