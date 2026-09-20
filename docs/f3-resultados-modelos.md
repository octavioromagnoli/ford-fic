# Resultados de modelo · piso contra los dos mejores

**Fecha:** 2026-09-19 · Se regenera con `python scripts/compare.py` (lee
`experiments/*/metrics.json`).

> ⚠️ **Provisorio.** Los eventos tienen una falla conocida —un error de SQL en la query— y
> la mentora los va a corregir. Todo lo que se mide contra `label` se re-mide después; el
> código y el diseño no cambian.

---

## La tabla

Sobre `panel_pseudo` (evento ficticio + ventana de riesgo), que es el único panel donde la
posición del corte **no** es la etiqueta: P = 0,489 contra 0,836 del v1.

| | modelo | PR-AUC | PR-AUC norm | F1 | F1 trivial | ROC-AUC | detección @≤50 FA/1000 | anticipación |
|---|---|---|---|---|---|---|---|---|
| **piso** | `baserate` | 0,557 | −0,004 | 0,717 | 0,717 | 0,496 | 0% | — |
| **1º** | **`logistic`** | **0,732** | **0,392** | 0,758 | 0,717 | **0,703** | 19% | 2.079 km |
| **2º** | `gbm` | 0,721 | 0,369 | 0,755 | 0,717 | 0,692 | **30%** | 2.673 km |

Tasa base 0,558. Intervalo bootstrap del PR-AUC por fold: `logistic` [0,671 – 0,801],
`gbm` [0,689 – 0,780], `baserate` [0,554 – 0,560].

**Las columnas de normalización no son decoración.** `PR-AUC norm` es
`(AP − π) / (1 − π)`: 0 = un modelo al azar, 1 = perfecto, y **no depende de la
prevalencia**. `F1 trivial` es `2π/(1+π)`, el F1 del que dice "positivo" siempre: con
π = 0,558 vale **0,717**, así que un F1 de 0,758 es una mejora de 4 puntos sobre no tener
modelo, no el 76% de acierto que parece. El F1 crudo de esta tabla es el **mejor posible**
sobre todos los umbrales —una cota optimista, porque el umbral se elige mirando las mismas
filas que se evalúan—; el umbral de operación real sale del presupuesto de falsas alarmas.

**Cómo leerla.**

- El **piso hace su trabajo**: PR-AUC = tasa base y ROC = 0,50. Cualquier modelo que no le
  gane está roto (regla 5 de `CLAUDE.md`).
- **`logistic` gana por ROC y `gbm` por detección.** No es contradictorio: el ROC mide el
  ordenamiento completo y la detección mide solo el extremo del ranking, que es lo que
  importa para operar. Si hay que elegir uno **para el pitch, es `gbm`**: detecta el 30% de
  los vehículos con evento con 2.673 km de anticipación mediana y menos de 50 falsas
  alarmas cada 1.000 sanos. Si hay que elegir uno **para reportar la calidad del panel, es
  `logistic`**.
- **La tasa base es 0,558**, no 0,125: el panel del evento ficticio conserva de cada
  vehículo solo su ventana de riesgo, así que las positivas pesan mucho más. **El PR-AUC de
  esta tabla no se compara con el de los otros paneles**; lo comparable es el lift y el ROC.

## Por qué el PR-AUC subió tanto (y qué parte es real)

De 0,181 en el v1 a 0,732 acá: ×4. **Casi todo es la prevalencia, no el modelo.**

El PR-AUC de un clasificador que no ordena nada es la tasa base. El panel del evento
ficticio conserva de cada vehículo solo su ventana de riesgo, así que las positivas pasan
de ser el 12,5% de las filas a ser el 55,8%: el piso del PR-AUC se movió de 0,125 a 0,558
**antes de entrenar nada**. Se ve en la propia tabla: el `baserate` saca 0,557 sin mirar
una sola feature.

| | panel v1 | panel evento ficticio |
|---|---|---|
| tasa base (= piso del PR-AUC) | 0,125 | 0,558 |
| PR-AUC del mejor modelo | 0,181 | 0,732 |
| **PR-AUC normalizado** | **0,063** | **0,392** |
| **ROC-AUC** | **0,617** | **0,703** |
| F1 / F1 trivial | 0,266 / 0,223 | 0,758 / 0,717 |

Las dos métricas que no dependen de la mezcla —ROC y PR-AUC normalizado— dicen que **sí
hay una mejora real**, pero mucho más chica que el ×4: el ROC sube 0,086 puntos.

Y hay una tercera lectura que es la que más importa, del test de permutación:

| panel | nulo (etiqueta permutada) | real | p |
|---|---|---|---|
| v1 canónico | ROC **0,555 ± 0,059** | 0,617 | **0,20** |
| evento ficticio | ROC **0,501 ± 0,057** | 0,703 | **< 0,001** |

**El nulo del panel v1 no es 0,50: es 0,555.** Con la etiqueta rota, el modelo todavía
saca 0,555 de ese panel, porque las filas positivas son los últimos cortes de su vehículo
y varias features siguen la posición. Contra ese nulo, el 0,617 del v1 **no es
distinguible del ruido**. El panel del evento ficticio devuelve el nulo a 0,50 y ahí el
0,703 sí significa algo.

O sea: el evento ficticio no subió el número, **hizo que el número quiera decir algo**.

## Contra el panel anterior

| panel | mejor modelo | ROC | P(posición) | ¿el número se apoya en el atajo? |
|---|---|---|---|---|
| v1 canónico | `logistic_l1` | 0,617 | 0,836 | **sí**: el nulo permutado ya da 0,555 (p = 0,20) |
| Δ=250 + emparejado por posición | `logistic_l1` | 0,670 | — | **sí**: la ablación lo lleva a 0,922 |
| **evento ficticio + ventana de riesgo** | `logistic` | **0,703** | **0,489** | no |

El salto de 0,617 a 0,703 es el único de la fase que sobrevive a la auditoría, porque es el
único medido sobre un panel donde la posición del corte no predice la etiqueta.

## La letra chica, que importa

1. **El panel es chico**: 455 filas, 140 vehículos, 53 con evento, 246 negativos. Los
   intervalos son anchos y la diferencia entre `logistic` (0,703) y `gbm` (0,692) **no es
   significativa**. Son el mismo peldaño.

   Qué tan chico, medido: permutando la etiqueta a nivel vehículo y corriendo la CV
   entera (`scripts/permutation_test.py`), el nulo de este panel es **ROC 0,50 ± 0,06**.
   O sea que el ruido de estimación es de ±0,06 y **cualquier ROC por debajo de ~0,60 es
   indistinguible de nada**. El 0,703 sí lo supera: p < 0,001 sobre 30 permutaciones.
2. **El multivariado casi no le gana al univariado.** La mejor columna sola del panel
   separa con 0,60; el modelo con 54 columnas llega a 0,70 acá y a 0,62 en el v1. Hay
   señal, y es una sola dimensión: tiempo de motor encendido improductivo y frío.
3. **La ablación por familia**, que todo panel tiene que pasar: la temperatura ambiente
   aporta +0,007 y el marcador `Regenerations` 0,000 —el emparejado por mes funciona—, pero
   las estáticas (`Engine` y compañía, el sesgo de muestreo de F1) aportan **+0,164**.
   Están fuera del set base por eso, y ahora está cuantificado.
4. **k controles por sano.** `draws_per_vehicle` recupera negativos a costa del balance de
   posición, y el ROC no mejora, así que queda en 1:

   | k | negativos | vehículos | P(posición) | ROC |
   |---|---|---|---|---|
   | **1** | **246** | **107** | **0,489** | **0,703** |
   | 2 | 408 | 155 | 0,577 | 0,691 |
   | 5 | 985 | 143 | 0,654 | 0,657 |

## Reproducir

```bash
python scripts/build_dataset.py --config configs/data/panel_pseudo.yaml
python scripts/train.py --config configs/exp_baserate_pseudo.yaml
python scripts/train.py --config configs/exp_logistic_pseudo.yaml
python scripts/train.py --config configs/exp_gbm_pseudo.yaml
python scripts/compare.py
python scripts/permutation_test.py --config configs/exp_logistic_pseudo.yaml   # el nulo del panel
```

El catálogo completo de modelos probados, con su config y su panel, está en
[`f3-modelos-candidatos.md`](f3-modelos-candidatos.md).
