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

| | modelo | PR-AUC OOF | tasa base | lift | ROC-AUC | detección @ ≤50 FA/1000 | anticipación mediana |
|---|---|---|---|---|---|---|---|
| **piso** | `baserate` | 0,557 | 0,558 | 1,00× | 0,496 | 0% | — |
| **1º** | **`logistic`** | **0,732** | 0,558 | **1,31×** | **0,703** | 19% | 2.079 km |
| **2º** | `gbm` | 0,721 | 0,558 | 1,29× | 0,692 | **30%** | 2.673 km |

Intervalo bootstrap del PR-AUC por fold: `logistic` [0,671 – 0,801], `gbm` [0,689 – 0,780],
`baserate` [0,554 – 0,560].

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

## Contra el panel anterior

| panel | mejor modelo | ROC | P(posición) | ¿el número se apoya en el atajo? |
|---|---|---|---|---|
| v1 canónico | `logistic_l1` | 0,617 | 0,836 | sí, en parte |
| Δ=250 + emparejado por posición | `logistic_l1` | 0,670 | — | **sí**: la ablación lo lleva a 0,922 |
| **evento ficticio + ventana de riesgo** | `logistic` | **0,703** | **0,489** | no |

El salto de 0,617 a 0,703 es el único de la fase que sobrevive a la auditoría, porque es el
único medido sobre un panel donde la posición del corte no predice la etiqueta.

## La letra chica, que importa

1. **El panel es chico**: 455 filas, 140 vehículos, 53 con evento, 246 negativos. Los
   intervalos son anchos y la diferencia entre `logistic` (0,703) y `gbm` (0,692) **no es
   significativa**. Son el mismo peldaño.
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
```

El catálogo completo de modelos probados, con su config y su panel, está en
[`f3-modelos-candidatos.md`](f3-modelos-candidatos.md).
