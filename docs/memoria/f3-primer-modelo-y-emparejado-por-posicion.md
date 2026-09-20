# El primer modelo real, y qué pasa cuando se empareja también por posición

**Fecha:** 2026-09-19 · **Fase:** F3

> ⚠️ Provisorio: los eventos tienen una falla conocida (error de SQL) y la mentora los va
> a corregir. Todo lo de acá se re-mide después; el código no cambia.

Reproduce:

```bash
python scripts/train.py --config configs/exp_logistic_l1.yaml      # y exp_{logistic,gbm,lgbm}.yaml
python scripts/train.py --config configs/exp_logistic_l1_pos.yaml  # el mismo, contra el panel emparejado por posición
python scripts/compare.py
```

---

## 1 · Hasta ahora no había ningún modelo

El registry tenía `baserate` y `random`. Todo lo que sabíamos del panel era separación
**univariada** —la mejor columna sola—, que no dice nada sobre lo que hacen 50 columnas
juntas. Ahora hay cuatro peldaños (`logistic`, `logistic_l1`, `gbm`, `lgbm`), en ese orden
a propósito: con 254 filas positivas en 53 vehículos, la pregunta no es cuál gana sino
**cuánta capacidad tolera el panel antes de sobreajustar**.

| modelo | panel | PR-AUC OOF | lift sobre tasa base | ROC-AUC |
|---|---|---|---|---|
| `baserate` | v1 (54 columnas) | 0,121 | 0,97× | 0,486 |
| `logistic` | v1 | 0,159 | 1,27× | 0,577 |
| **`logistic_l1`** | v1 | **0,181** | **1,44×** | **0,617** |
| `gbm` | v1 | 0,166 | 1,33× | 0,603 |
| `lgbm` | v1 | 0,155 | 1,24× | 0,571 |
| `logistic` | v5 (118 columnas) | 0,170 | 1,36× | 0,605 |
| `lgbm` | v5 | 0,179 | 1,43× | 0,608 |

Tres lecturas, y las tres importan:

1. **Los modelos le ganan al piso**: PR-AUC 1,24–1,44× la tasa base, con el `baserate` en
   0,97× como control. El harness funciona y el panel tiene *algo*.
2. **Pero el multivariado no le gana al univariado.** El mejor ROC-AUC de un modelo con
   54 columnas es 0,617, y la mejor **columna sola** del panel daba 0,60
   (`feat_regen_per_idle_min`, P = 0,370 leído como 0,630). Combinar 54 features suma
   ~0,02 de ROC. Es la confirmación multivariada de lo que venían diciendo las cuatro
   vueltas de feature engineering: **el panel tiene una sola dimensión de señal**.
3. **Las 64 features extra del panel v5 no aportan** (0,605–0,608 contra 0,577–0,617).
   Duplicar el ancho del panel no mueve el número, que es exactamente lo que predecía
   [f3-barrido-de-relaciones.md](f3-barrido-de-relaciones.md).

La regularización fuerte es la que gana (`logistic_l1`, C = 0,05), y el boosting no le
gana a la lineal: con este tamaño de muestra, más capacidad es más varianza. El intervalo
bootstrap del PR-AUC por fold de `logistic_l1` es [0,170 – 0,302]: ancho, como corresponde
a 5 folds con ~50 positivas cada uno.

## 2 · Emparejar los sanos por posición: la señal sobrevive, pero el atajo no se cierra

El [§2b de f3-esfuerzo-de-control-y-dosis.md](f3-esfuerzo-de-control-y-dosis.md) había
dejado una sospecha: la señal térmica separa con 0,74 en los cortes tempranos y con 0,56
en los últimos, y las positivas se apilan al final. Medido ahora sobre el pool completo
de 7.308 filas sanas:

> **El 77% de las filas positivas cae en el último cuarto de la serie de su vehículo,
> contra el 30% de las sanas.**

`match_healthy_cuts` acepta `cut_position` como tercera dimensión. El resultado es una
mala noticia con un consuelo:

- **El pool no alcanza.** Con la grilla del v1 (odómetro de 4.000 km × mes × cuartos de
  serie) el emparejado deja **0 filas sanas**. Hubo que aflojar a 8.000 km × mes ×
  **mitades** y tolerar 50% de déficit para conservar **175 filas sanas de 72
  vehículos** — contra 1.241 filas de 144 vehículos del emparejado actual.
- **Y ni así se cierra el atajo**: sobre el panel emparejado, la posición sigue separando
  sola con P = 0,915. Con dos bins, "la mitad de arriba" mete en la misma celda a una
  positiva que está en el 95% de su serie y a una sana que está en el 55%.
- **Pero la señal sobrevive**, que es la pregunta que importaba: ROC-AUC 0,587–0,607
  contra 0,577–0,617 del panel v1, con la tasa base subiendo de 0,125 a 0,228. **Lo que
  miden los modelos no es el artefacto de posición.** Si lo fuera, balancear la posición
  —aunque sea a medias— lo habría hecho caer, y no cae.

| modelo | panel posmatch · PR-AUC | lift | ROC-AUC |
|---|---|---|---|
| `baserate` | 0,222 | 0,97× | 0,486 |
| `logistic` | 0,313 | 1,37× | 0,607 |
| `logistic_l1` | 0,325 | 1,42× | 0,587 |
| `gbm` | 0,273 | 1,19× | 0,590 |

(El PR-AUC **no** se compara entre paneles: las tasas base son distintas. Lo comparable es
el lift y el ROC. `scripts/compare.py` avisa solo cuando dos corridas no comparten panel.)

**Por dónde se cierra de verdad:** el problema es de resolución, no de método. Con Δ = 500
un vehículo tiene una mediana de 7 cortes, así que "último cuarto de la serie" son dos
filas y el pool sano se agota. Con **Δ = 250** la serie se duplica (mediana 14) y las
celdas de posición se pueblan. `configs/data/panel_delta250.yaml` ya existe: ese es el
experimento que falta, y ahora hay con qué medirlo.

## 3 · Lo que quedó en el repo, además de los modelos

- **`aux_cut_position`**: la posición del corte en la serie **completa**, calculada antes
  de muestrear y guardada en el panel. Arregla un defecto de las auditorías:
  `audit_sequence.py` la recalculaba sobre el panel ya muestreado, donde está
  distorsionada (de un vehículo con evento se conservan todos los cortes; de un sano,
  un subconjunto, así que su rango se comprime). Ahora la usa si está.
- **El knob de ablación** que faltaba: `features.exclude_prefixes` y
  `features.extra_prefixes` en el YAML del experimento, resueltos en
  `cv.py::select_feature_columns`. Una ablación es otro config, no otra rama de código.
  Sirve para lo que pedía el protocolo de F3: meter las `aux_` de calendario y ver si el
  número salta (si salta, confirma que eran un atajo).
- **`scripts/compare.py` andaba roto en Windows**: moría con `UnicodeEncodeError` al
  imprimir la tabla (la consola es cp1252 y la tabla lleva "≤"). Una línea.
- `configs/data/panel_nomatch.yaml`: el panel con el pool sano completo (7.308 filas), para
  simular variantes de emparejado sin releer 1,2 GB. No se entrena con él: sin emparejar,
  el odómetro y el calendario quedan abiertos como atajos.
