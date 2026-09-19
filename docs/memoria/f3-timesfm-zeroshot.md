# TimesFM zero-shot: en el panel v1 no le gana a la tasa base, pero señaló el desvío respecto del propio vehículo

**Fecha:** 2026-09-19 (primera versión: 16-09, con cortes propios sobre el universo de
1081) · **Fase:** F3 · **Alcance:** dev del panel v1 (2.029 filas, 171 vehículos, 254
positivas, tasa base 0,125). · **Reproduce:**

```bash
python scripts/eval_timesfm.py --config configs/exp_timesfm3_zeroshot.yaml               # canal regen
python scripts/eval_timesfm.py --config configs/exp_timesfm3_zeroshot_accumulation.yaml  # canal nivel
python scripts/eval_timesfm.py --config configs/exp_timesfm3_zeroshot_badmsg.yaml        # canal mensajes
python scripts/build_timesfm_panel.py --config configs/data/panel_timesfm_v1.yaml
python scripts/train.py --config configs/exp_lgbm_panel_v1.yaml    # control
python scripts/train.py --config configs/exp_lgbm_timesfm.yaml     # + feat_tfm_*
```

Dependencia opcional: `requirements-timesfm.txt` (MLX en Apple silicon). **Los pesos de
TimesFM 3.0 son de licencia no comercial**: sirven para medir, no para el producto.

## Qué es

TimesFM pronostica, para cada corte del panel, las series por km del vehículo (bins de
100 km) sobre `[G, G+H]`, con contexto estrictamente anterior al corte. Tres canales:
nivel de acumulación (`AirRegenerationEnd`), regeneraciones (caídas del nivel) y
fracción de mensajes Full/Overloaded/Over Limit. El score es un cuantil del pronóstico;
los resúmenes (`delta`, `spread`, `q90_max`) entran como `feat_tfm_*` a una variante
del panel para ver si le suman a un GBM.

## La primera versión medía cosas que el EDA de F2 invalidó

| antes (16-09) | problema | ahora |
|---|---|---|
| cortes y etiquetas propios sobre los 1081 | el 60% de los sanos (432 de 716) son de mercados donde un evento es invisible, y los positivos con fecha = venta no daban cortes ([f2-universo-fecha-usable.md](f2-universo-fecha-usable.md)) | los cortes del panel v1 (364 del universo) |
| sanos emparejados solo por odómetro | deja el calendario como atajo (ROC 0,67 → 0,76 con 3 columnas de calendario, §3.4 de [f2-eda-revision-y-features.md](f2-eda-revision-y-features.md)) | el emparejado del panel: odómetro × mes |
| canal de regeneraciones = marcador `signals.Regenerations` | se corta el 25-05-2026: mide el calendario (§2.1) | caídas de `AirRegeneration` (`regen_drop`, igual que el panel) |
| canal de nivel = `signals.Acumulation` interpolada en el tiempo sobre los viajes | misma variable que `trips.AirRegenerationEnd`, que ya tiene el odómetro | `trips.AirRegenerationEnd` por bin de odómetro |
| panel de ablación con `static_{ModelSeries, ProductionDay, daysUntilSale}` | las tres son exposición/calendario y salieron del set base el 18-09 | el panel v1 (solo `SalesCountry_cd`) |
| series desde el bin 0 con ceros inventados antes de la primera señal | historia que no existe | cada serie empieza en su primer bin con viajes |

Lo que había dado la versión vieja confirma el diagnóstico de main desde otro lado. El
mismo LightGBM, mismos cortes:

| panel (versión vieja) | PR-AUC | lift | ROC-AUC |
|---|---|---|---|
| ventana + estáticas | 0,093 | 3,7× | 0,823 |
| ventana + estáticas + `feat_tfm_*` | 0,098 | 3,9× | 0,822 |
| ventana sin estáticas | 0,030 | 1,2× | 0,567 |
| ventana sin estáticas + `feat_tfm_*` | 0,034 | 1,3× | 0,616 |

Toda la "señal" estaba en las estáticas de calendario. No lo interpretamos así en su
momento; el EDA de main lo explica ([decisiones.md](decisiones.md), 18-09).

## Resultados sobre el panel v1

**Zero-shot** (el pronóstico como score directo; PR-AUC / ROC-AUC sobre dev):

| canal | TimesFM | naive (persistencia de W) | control `−cut_odo` |
|---|---|---|---|
| regeneraciones (cuantil 0,9, suma) | 0,118 / 0,467 | 0,111 / 0,430 | 0,113 / 0,446 |
| nivel (cuantil 0,9, máximo) | 0,112 / 0,445 | 0,122 / 0,445 | 0,113 / 0,446 |
| mensajes malos (cuantil 0,9, media) | 0,128 / 0,518 | 0,130 / 0,487 | 0,113 / 0,446 |

Ningún canal le gana a la tasa base (0,125). Regeneraciones y nivel quedan **por
debajo** de 0,5 de ROC, que es la dirección que midió el perfil alineado al evento de
main: los que fallan regeneran levemente menos y terminan los viajes con el filtro menos
cargado (§3.2). Un pronóstico de "más carga ⇒ más riesgo" apunta al revés.

**Ablación con LightGBM** (mismas filas y folds que el control):

| panel | PR-AUC (R=1) | ROC-AUC | PR-AUC (R=3) |
|---|---|---|---|
| v1 (control) | 0,165 | 0,584 | 0,161 ± 0,006 |
| v1 + 9 `feat_tfm_*` | 0,170 | 0,582 | 0,169 ± 0,007 |

Pareado: +0,005 · +0,012 · +0,008 en las tres repeticiones, pero solo 9 de 15 folds
positivos y el ROC no se mueve (−0,002 · +0,020 · −0,006). Es del orden del ruido del
sorteo.

## Lo que sí dejó: el desvío respecto de la propia historia

Una feature sola separa más que cualquiera del panel v1 a nivel fila:

| columna | ROC univariado (dev) | ρ con mes del corte | ρ con `cut_odo` |
|---|---|---|---|
| `feat_tfm_regen_delta` (pronóstico − ventana reciente) | **0,614** | −0,11 | −0,14 |
| historia (hasta 25.600 km) − ventana reciente, **sin modelo** | 0,602 | | |
| `feat_idle_frac` (la mejor del panel v1) | 0,597 | | |
| `feat_regenerations_trend_per_1000km` (mitad 2 − mitad 1 de W) | 0,490 | | |

`delta` es casi reversión a la media: si en los últimos 1.000 km el vehículo regeneró
menos que en su historia, el pronóstico sube. La versión sin modelo da casi lo mismo
(ρ 0,52 con la de TimesFM) y **no está en el panel**: la tendencia dentro de la ventana
no la ve (ρ −0,002). Es la idea 2.4 de [`docs/f3-modelos-candidatos.md`](../f3-modelos-candidatos.md)
("desvío respecto del propio vehículo"), medida por otro camino. Para el nivel del filtro
la misma resta da 0,560.

## Qué queda

- **TimesFM no entra.** No le gana a la tasa base como score, no mueve el LightGBM más
  allá del ruido y su licencia no permite usarlo en el producto.
- **El desvío respecto de la historia del vehículo sí merece su feature**, sin
  TimesFM: `(media de la historia previa − media de la ventana)` para regeneraciones y
  nivel, calculada hacia atrás desde el corte. Antes de celebrarla, la auditoría (b) del
  doc de F3: el contexto largo abarca meses distintos para vehículos cortados el mismo
  mes, y el calendario es el atajo conocido de este dataset.
- Una sola mirada al número: es dev con 53 vehículos con evento. Con 14 en test, la
  confirmación va a tener intervalos anchos.
