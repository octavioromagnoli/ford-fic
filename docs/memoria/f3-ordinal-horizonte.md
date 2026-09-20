# Target ordinal multi-horizonte: dos variantes de bins y el costo como barrido

**Fecha:** 2026-09-20 · **Fase:** F3 · **Alcance:** dev del panel v1 (2.029
filas, 171 vehículos, 254 positivas, tasa base 0,125), mismos folds congelados que
`f3-lgbm-panel-v1`. · **Configs:** `configs/exp_ordinal_horizon.yaml` (bins
restringidos) y `configs/exp_ordinal_horizon_ext.yaml` (bins extendidos).

El documento tiene dos mitades. La primera es la variante de **bins restringidos**,
medida primero: los cuatro buckets caen adentro de `[G, G+H]`, que es exactamente el
intervalo que ya era `label = 1`. La segunda, [más abajo](#bins-extendidos-la-clase-0-deja-de-significar-sano),
corrige ese defecto de diseño y mide la variante de **bins extendidos**, además de
reemplazar la matriz de costos única por un barrido y de correr las tres auditorías
obligatorias que a la primera le faltaban. La primera mitad se conserva porque su
resultado sigue siendo evidencia válida de qué pasa cuando el target no aporta
información nueva.

---

# Bins restringidos: la primera versión, conservada como evidencia

**Config:** `configs/exp_ordinal_horizon.yaml` · **Corrida:** `f3-ordinal-horizon`.

## Decisión

Se entrenó un LightGBM multiclase sobre cinco estados ordenados: clase 0 = sano/no
inminente y cuatro buckets dentro de la ventana visible `[G, G + H]`. La salida
comparable es la acumulada `P(clase >= 1)`, que reconstruye `P(label = 1)` y entra a
`lead_time_curve()` sin cambiarla. La selección sigue siendo PR-AUC OOF contra la
`label` binaria original; el costo ordinal es reporte.

La referencia es [SCANIA Component X](https://arxiv.org/abs/2401.15199) (Kharazian
et al., 2025): cinco clases temporales y costos definidos por clase real/predicha,
con falsos positivos de 7–10 y falsos negativos de hasta 500. Acá no se cambia de
familia de modelo: el experimento aísla la reformulación del target.

## Por qué estos bins

El intervalo positivo real mide 3.000 km: desde el fin del gap (`G = 500`) hasta
`G + H = 3.500`. Se partió en cuatro buckets de 750 km. Es el máximo de cinco clases
totales advertido por el tamaño de dev, pero no deja celdas testimoniales:

| clase | distancia al evento | filas dev | vehículos | filas por fold de validación |
|---:|---|---:|---:|---:|
| 0 | sano, dentro del gap o > 3.500 km | 1.775 | 157 | 353–359 |
| 1 | (2.750, 3.500] km | 55 | 38 | 9–12 |
| 2 | (2.000, 2.750] km | 63 | 43 | 9–14 |
| 3 | (1.250, 2.000] km | 68 | 46 | 12–15 |
| 4 | [500, 1.250] km | 68 | 47 | 12–16 |

Las cuatro clases positivas quedan entre 55 y 68 filas y entre 38 y 47 vehículos;
ningún fold baja de 9 filas en un bucket. Agregar otro corte ya no aporta una escala
operativa clara y acerca el régimen de 3 positivos por bin que se quería evitar.

La evidencia se reproduce sobre el panel dev congelado:

```bash
FORD_DATA_DIR=data/v364 python - <<'PY'
import numpy as np
import pandas as pd
from scripts.train import select_dev
from src.config import load_config, resolve_path
from src.training.targets import build_ordinal_target

cfg = load_config("configs/exp_ordinal_horizon.yaml")
panel = pd.read_parquet(resolve_path(cfg["data"]["panel"]))
dev = select_dev(panel, cfg)
y = build_ordinal_target(
    dev, np.ones(len(dev), dtype=bool), **cfg["target"]["params"]
).y
print(np.bincount(y).tolist())
print([dev.loc[y == c, "vehicle_id"].nunique() for c in range(5)])
PY
```

## Implementación y antileakage

- `src/training/targets.py` registra por nombre el constructor, el decoder y el
  reporte. `cv.py` no sabe qué es ordinal: pide el target del train del fold y
  decodifica por el mismo registro. El archivo queda listo para sumar el target de
  supervivencia de la rama paralela como otra función.
- `build_ordinal_target()` usa solo `panel.loc[train_mask]`. Clase positiva exige
  `gap_km <= time_to_event_km <= 3.500`; en dev se verifica exactamente
  `clase > 0 ⇔ label == 1`.
- Imputación, escalado y one-hot siguen dentro del `Pipeline` de cada fold. Los
  splits continúan agrupados por vehículo y estratificados por `label`.
- `lgbm_ordinal` replica la capacidad y regularización del control; todos los
  hiperparámetros efectivos están explícitos en el YAML.

La matriz del YAML (filas reales, columnas predichas) es:

| real \ pred. | 0 | 1 | 2 | 3 | 4 |
|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 7 | 8 | 9 | 10 |
| 1 | 200 | 0 | 7 | 8 | 9 |
| 2 | 300 | 200 | 0 | 7 | 8 |
| 3 | 400 | 300 | 200 | 0 | 7 |
| 4 | 500 | 400 | 300 | 200 | 0 |

Para reportarla, cada fila se asigna a la clase que minimiza el costo esperado bajo
sus probabilidades OOF. `cost_matrix_score()` suma `matrix[real, predicha]`; la matriz
nunca vive en Python.

## Resultado

| modelo | PR-AUC (lift) | ROC-AUC | Brier | detección · anticipación @ ≤50 FA/1000 |
|---|---:|---:|---:|---:|
| LightGBM binario (control) | **0,165** (1,32×) | **0,584** | **0,173** | 0,09 · 10.408 km |
| LightGBM ordinal | 0,152 (1,22×) | 0,578 | 0,207 | 0,09 · 10.408 km |
| mejor registrado (`lgbm-timesfm`) | **0,170** (1,36×) | 0,600 | 0,171 | 0,13 · 4.802 km |

IC bootstrap por fold: ordinal `[0,151, 0,204]`, control `[0,173, 0,221]`; se
solapan. La reformulación no demuestra una diferencia y su media queda 0,013 por
debajo del control. Tampoco acerca el objetivo realista de ROC 0,65–0,70 y lift
1,6–2×. **No reemplaza al baseline binario.**

Costo OOF: **18.872** total, 9,30 por fila (9.301 por 1.000). Es mucho menor que
predecir siempre clase 0 (91.100), pero apenas mejora el extremo de alertar siempre
con la clase más cercana (19.225; ahorro 353). La matriz hace visible el trade-off:
con falsos negativos 20–50 veces más caros, la decisión óptima es deliberadamente
agresiva (1.941 de 2.029 filas van a clase 4). Por eso el costo sirve como traducción
operativa y no como criterio para elegir el modelo.

## Reproducción

```bash
source .venv/bin/activate
python scripts/check_setup.py
FORD_DATA_DIR=data/v364 WANDB_MODE=disabled \
  python scripts/train.py --config configs/exp_ordinal_horizon.yaml
python scripts/results.py log f3-ordinal-horizon \
  --note "Target ordinal 5 clases; PR-AUC 0.152 < control 0.165, sin mejora."
python scripts/results.py table --out results/README.md
```

La corrida versionada está en `results/f3-ordinal-horizon.yaml`. Se intentó el log
online, pero el entorno de ejecución bloqueó la salida a W&B; el resultado registrado
es la corrida local determinista con `WANDB_MODE=disabled`.

---

# Bins extendidos: la clase 0 deja de significar "sano"

**Config:** `configs/exp_ordinal_horizon_ext.yaml` · **Corrida:** `f3-ordinal-horizon-ext`
· **Auditorías:** `scripts/audit_ordinal_horizon.py`.

## El defecto de la primera versión

Los cuatro buckets de arriba parten `[G, G+H] = [500, 3500]`, que es **exactamente** el
intervalo que ya era `label = 1`. Por eso la propia verificación de la primera mitad
—`clase > 0 ⇔ label == 1`— no era una garantía: era el síntoma. El target resultó un
refinamiento estricto adentro de la clase positiva, así que:

- no aporta **nada** sobre las 1.775 filas negativas, el 87% del panel;
- en particular, no distingue "vehículo sano" de "vehículo que va a fallar dentro de
  8.000 km", que es justo la distinción que un target multi-horizonte tendría que traer;
- lo único que cambia respecto del binario es partir 254 filas positivas en cuatro
  baldes de ~60 —y son ~50 vehículos **los mismos** en las cuatro clases—.

Eso mide costo de varianza sin beneficio de información, y el resultado tiene esa firma
exacta: PR-AUC 0,152 contra 0,165, ROC casi igual (0,578 contra 0,584) y Brier bastante
peor (0,207 contra 0,173).

La formulación de [SCANIA Component X](https://arxiv.org/abs/2401.15199) no hace eso.
Sus clases son `>48, 48–24, 24–12, 12–6, 6–0`, donde **la clase 0 significa "lejos del
fallo", no "sano"**. Las clases intermedias se llenan con instancias que un binario a
horizonte corto etiquetaría como negativas, y ahí está la información nueva.

## Los bins nuevos, elegidos contra el histograma

En dev hay 967 filas de vehículos con evento y solo 254 tienen `label = 1`. Las **713
restantes son negativas hoy pero vienen de un vehículo que falla**: son las candidatas a
las clases intermedias. La distribución de `time_to_event_km` sobre esas 967 filas:

| tramo [km] | 0,5–1k | 1–1,5k | 1,5–2k | 2–2,5k | 2,5–3k | 3–3,5k | 3,5–4k | 4–5k | 5–6k | 6–8k | 8–10k | 10–12k | 12–15k | >15k |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| filas | 47 | 44 | 45 | 43 | 38 | 37 | 37 | 72 | 61 | 118 | 106 | 74 | 64 | 181 |

La mediana está en 7.060 km y el percentil 75 en 12.180. Los cortes elegidos son
`[2000, 3500, 6000, 10000]`:

| clase | distancia al evento | filas dev | de ellas `label=1` | vehículos | mín. por fold de validación |
|---:|---|---:|---:|---:|---:|
| 0 | censurado **o** evento a > 10.000 km | 1.381 | 0 | 139 | 274 |
| 1 | (6.000, 10.000] km | 224 | 0 | 30 | 44 |
| 2 | (3.500, 6.000] km | 170 | 0 | 39 | 31 |
| 3 | (2.000, 3.500] km | 118 | 118 | 43 | 21 |
| 4 | [500, 2.000] km | 136 | 136 | 52 | 24 |

Tres razones para estos cortes y no otros:

1. **3.500 tiene que ser un corte.** `score_max_tte_km: 3500` exige que `G + H` coincida
   con un borde, porque el score comparable es la acumulada `P(clase >= 3)` y tiene que
   reconstruir `P(label = 1)` exactamente. Las clases 3 y 4 suman 254 filas, que son
   exactamente las positivas: sin eso, la corrida deja de ser comparable con el control.
2. **Las clases quedan parejas y los folds dejan de ser testimoniales.** 136/118/170/224
   contra los 55/63/68/68 de la versión restringida, y el bucket más chico de un fold de
   validación pasa de 9 filas a 21. El costo de varianza que pagaba la primera versión
   se paga mucho menos acá.
3. **La clase 0 conserva 319 filas de vehículos con evento** (23% de la clase). Ese
   número es la medida directa del riesgo que abre extender los bins, y por eso el
   reporte lo emite en `target_report.class_0`: si llegara a cero, "clase 0" y "vehículo
   sano" serían la misma variable y el target auxiliar sería la cohorte de muestreo
   —que en este panel **es** la etiqueta (CLAUDE.md)— disfrazada de horizonte.

Variantes descartadas: cerrar en 8.000 deja la clase 1 en 118 filas y desperdicia 106
filas informativas; cerrar en 12.000 la infla a 298 y vacía la clase 0 de vehículos con
evento (baja de 319 a 245), que es justo lo que no conviene. Un quinto corte
(`[2000, 3500, 5000, 8000, 12000]`) reparte 180/179/109/118/136 pero vuelve a bajar el
mínimo por fold a 20 sin agregar una escala operativa distinguible.

```bash
FORD_DATA_DIR=data/v364 python - <<'PY'
import numpy as np, pandas as pd
from scripts.train import select_dev
from src.config import load_config, resolve_path
from src.training.targets import build_ordinal_target

cfg = load_config("configs/exp_ordinal_horizon_ext.yaml")
dev = select_dev(pd.read_parquet(resolve_path(cfg["data"]["panel"])), cfg)
spec = build_ordinal_target(dev, np.ones(len(dev), dtype=bool), **cfg["target"]["params"])
print(spec.info["class_counts"], spec.info["class_0"])
PY
```

## Resultado: no gana en ninguno de los dos ejes

Mismo panel, mismos folds congelados, mismo modelo salvo el objetivo. Se reportan los
**dos ejes** a propósito: extender los bins acerca la tarea auxiliar a "identificar la
cohorte", y ese modo de fallar empeora el PR-AUC por fila mientras mejora el ranking por
vehículo. Con un solo eje no se distingue.

| | fila: PR-AUC (lift) | ROC | Brier | vehículo (`mean`): lift | ROC | detección · anticip. @ ≤50 FA/1000 |
|---|---:|---:|---:|---:|---:|---:|
| LightGBM binario (control) | **0,165** (1,32×) | **0,584** | 0,173 | **1,51×** | **0,666** | 9,4% · 10.408 km |
| ordinal, bins restringidos | 0,152 (1,22×) | 0,578 | 0,207 | 1,42× | 0,653 | 9,4% · 10.408 km |
| ordinal, bins extendidos | 0,157 (1,25×) | 0,579 | **0,152** | 1,31× | 0,620 | 7,5% · 12.322 km |

El eje de vehículo usa `mean`, la única agregación que no está confundida con el tamaño
de la bolsa en el panel v1 ([f3-mil-agregacion-vehiculo.md](f3-mil-agregacion-vehiculo.md));
su tasa base es 0,310 y no se compara contra la de filas, solo el lift.

Tres lecturas:

- **Los bins extendidos recuperan parte de lo que perdieron los restringidos por fila**
  (0,157 contra 0,152) y **mejoran mucho la calibración**: Brier 0,152, el mejor de las
  tres corridas, por debajo incluso del control binario. Tiene explicación mecánica: con
  `class_weight="balanced"` y cuatro clases positivas adentro de la ventana, la
  acumulada `P(clase >= 1)` de la primera versión salía inflada; acá las clases 1 y 2
  absorben peso y **quedan fuera** del score, que es `P(clase >= 3)`.
- **El modo de fallar que se temía no ocurrió.** Si el modelo hubiera aprendido cohorte,
  el ranking por vehículo habría mejorado. Bajó: 1,31× contra 1,42× y 1,51×. Extender los
  bins no lo convirtió en un detector de cohorte; le sacó resolución en los dos ejes.
- **Ninguna de las dos variantes reemplaza al binario.** El control les gana por fila y
  por vehículo. La selección sigue siendo PR-AUC OOF contra `label` y sigue eligiendo el
  binario.

## La matriz de costos única era degenerada; la reemplaza un barrido

La matriz 5×5 de SCANIA que usa la primera mitad de este documento **no puede
discriminar modelos a esta tasa base**, y el número que produjo (18.872, ahorro de 353
sobre alertar siempre) lo estaba diciendo sin que lo leyéramos.

El argumento es de una línea. Con prevalencia `p` y costos `C_FN`, `C_FP`, alertar
siempre cuesta `n·(1-p)·C_FP` y no alertar nunca cuesta `n·p·C_FN`; la primera le gana a
la segunda en cuanto `C_FN / C_FP > (1-p)/p`. Con `p = 0,125` eso es **7**. La matriz de
SCANIA está en 20–50×, muy por encima: la política óptima es "revisar todo" **sin
importar el modelo**, y lo que mide el costo resultante es la matriz, no el
clasificador. Encima esa matriz es para una rotura en ruta de un camión pesado, que
inmoviliza el vehículo; acá el evento es degradación de eficiencia de combustión y no
inmoviliza a nadie, así que ni siquiera hay razón para adoptar su escala.

`src/eval/metrics.py::cost_ratio_sweep()` reemplaza la matriz por un barrido del ratio
`C_FN / C_FP`, y para cada ratio compara el mejor umbral sobre el score contra las dos
políticas triviales. Es model-agnóstico —sale de `eval.cost_ratios`, corre para
cualquier corrida y cualquier target— y es **reporte, nunca criterio de selección**
(regla 5). El umbral se elige sobre las mismas predicciones que se evalúan, así que el
ahorro es una **cota superior optimista**: se usa así a propósito, porque si ni con el
umbral oráculo el modelo le gana a alertar todo, la conclusión es firme.

`C_FP = 1`, 2.029 filas de dev, alertar siempre cuesta 1.775:

| `C_FN/C_FP` | control binario | ordinal restringido | ordinal extendido | mejor trivial |
|---:|---|---|---|---|
| 2 | — | — | — | nunca (508) |
| 3 | 0,7% | — | — | nunca (762) |
| 5 | 4,4% | 2,0% | 4,6% | nunca (1.270) |
| **7** | **14,6%** | **14,1%** | **13,7%** | siempre (1.775) |
| 10 | 1,7% | 3,5% | 2,5% | siempre (1.775) |
| 20 | 1,4% | 1,6% | 1,8% | siempre (1.775) |
| 50 | 1,4% | 1,6% | 1,8% | siempre (1.775) |

(ahorro sobre la mejor política trivial; "—" = ninguna ganancia)

**El punto de quiebre está en 7× y el pico es estrecho.** Las tres corridas dan
prácticamente el mismo perfil, lo que confirma que la forma de la curva es del problema
y no del modelo: el valor de cualquier clasificador se concentra alrededor del cruce de
las dos políticas triviales. Por debajo de 5× conviene no alertar a nadie; por encima de
10× conviene revisar todo y el modelo ahorra menos del 3%. Con los 20–50× de SCANIA
ahorra 1,4–1,8%, que es la confirmación numérica de que esa matriz era degenerada acá.

Para el pitch: **el modelo paga cuando una degradación no detectada cuesta alrededor de
7 veces una inspección innecesaria** —y ahorra ahí un 14% sobre revisar todo—. Por
debajo de 5× no conviene alertar a nadie y por encima de 10× conviene revisar todo: en
los dos extremos la decisión no necesita modelo. Es una frase honesta y acotada, no la
promesa abierta que sugería la matriz única.

```bash
FORD_DATA_DIR=data/v364 python - <<'PY'
import pandas as pd
from src.eval.metrics import cost_ratio_sweep, cost_ratio_breakeven
pred = pd.read_parquet("experiments/f3-ordinal-horizon-ext/predictions.parquet")
sweep = cost_ratio_sweep(pred.label.to_numpy(int), pred.score.to_numpy(float),
                         [2, 3, 5, 7, 10, 20, 50])
print(cost_ratio_breakeven(sweep))
PY
```

## Las tres auditorías obligatorias

`scripts/audit_ordinal_horizon.py` las corre sobre cualquier YAML de experimento. Se
corrieron sobre las tres corridas —**incluido el control binario**—, porque sin ese
contraste no hay forma de saber si lo que encuentra la auditoría (a) es del target
ordinal o del panel.

```bash
FORD_DATA_DIR=data/v364 python scripts/audit_ordinal_horizon.py \
    --config configs/exp_ordinal_horizon_ext.yaml --n-permutations 10 --seed 20260920
```

### (a) Permutación de la etiqueta — la auditoría que cambió la lectura del proyecto

Es la crítica cuando los bins se extienden, así que se corre con **dos nulos**, 10
permutaciones cada uno, reentrenando la CV entera en cada una. En los dos se mueve el
desenlace completo (`label`, `time_to_event_km`, `event_observed`) en bloque: reasignar
solo `label` dejaría el panel incoherente y el nulo mediría esa incoherencia.

- **Nulo global**, entre todas las filas: destruye la cohorte *y* el orden interno. Es
  el control del pipeline y tiene que caer a la tasa base.
- **Nulo intra-vehículo**, solo entre los cortes de un mismo vehículo: conserva
  `event_observed` y el multiconjunto de distancias al evento de cada vehículo, y
  rompe únicamente la relación entre las features de un corte y *cuándo* cae el evento
  respecto de ese corte.

| PR-AUC | control binario | ordinal restringido | ordinal extendido |
|---|---:|---:|---:|
| modelo | 0,1653 | 0,1524 | 0,1567 |
| nulo global (cohorte destruida) | 0,1259 ± 0,0060 | 0,1237 ± 0,0054 | 0,1250 ± 0,0056 |
| nulo intra-vehículo (cohorte viva) | **0,1928** ± 0,0169 | **0,1867** ± 0,0138 | **0,1784** ± 0,0098 |
| tasa base | 0,1252 | 0,1252 | 0,1252 |
| modelo − nulo intra-vehículo | −0,0275 (−1,6 σ) | −0,0343 (−2,5 σ) | −0,0217 (−2,2 σ) |

**El nulo global cae exacto a la tasa base en las tres corridas** (+0,0007 / −0,0015 /
−0,0002). El pipeline no filtra: splits agrupados, preprocesamiento por fold y gap de
blanking hacen lo que dicen.

**El nulo intra-vehículo no cae a la tasa base, y no puede caer.** Un vehículo sano no
tiene ninguna fila positiva, así que permutar adentro lo deja con cero positivas: la
etiqueta permutada sigue siendo perfectamente predecible desde la cohorte. Por eso **la
expectativa escrita en el plan (§0 de `docs/f3-modelos-candidatos.md`: "el PR-AUC tiene
que caer a la tasa base") no aplica a esta permutación en este panel**. El piso correcto
para leer este nulo es el nulo mismo, y la pregunta es si el modelo le gana.

**No le gana. Ninguno de los tres.** Las tres corridas quedan entre 1,6 y 2,5 desvíos
**por debajo** de su propio nulo de cohorte. Dicho sin rodeos: el PR-AUC de 0,165 del
control binario no es anticipación, es separación de vehículos que fallan de vehículos
sanos — y esa separación, en este panel, es en buena medida la cohorte de muestreo, que
*es* la etiqueta (CLAUDE.md). La tarea real —¿en cuál de los cortes de *este* vehículo
hay que alertar?— hoy no la resuelve ningún modelo del repo mejor que el azar.

Dos cosas importan de este resultado:

1. **No es del target ordinal: es del panel.** El control binario muestra el mismo
   patrón, y con el déficit más grande en términos absolutos. Extender los bins no
   creó el problema ni lo agravó; lo hizo visible porque obligó a correr la auditoría.
   Si la auditoría (a) se hubiera corrido en F3 desde la primera corrida, esto se
   sabría desde antes.
2. **Explica el resultado por vehículo del MIL.** El lift de 1,51× con `mean`
   ([f3-mil-agregacion-vehiculo.md](f3-mil-agregacion-vehiculo.md)) es consistente:
   el modelo es mejor eje de vehículo que eje de corte porque lo que sabe hacer es
   rankear vehículos.

Ojo con lo que **no** dice: el nulo intra-vehículo por encima de la tasa base no es una
fuga ni invalida las corridas. Dice que el número que venimos usando para seleccionar
mide menos de lo que su nombre sugiere, y que la mejora tiene que venir de features que
comparen al vehículo consigo mismo. Está anotado como pendiente en
[decisiones.md](decisiones.md).

### (b) Los `aux_` de calendario promovidos a `feat_`

`aux_air_temp_avg`, `aux_regen_marker_per_1000km` y `aux_static_ProductionDay` entran al
modelo con prefijo `feat_`. Los eventos caen entre sep-2025 y mar-2026 y la exposición
sana en 2026: si el ROC salta, el modelo encontró el calendario.

| | PR-AUC | ROC-AUC | Brier |
|---|---:|---:|---:|
| bins extendidos, set base | 0,157 | 0,579 | 0,152 |
| bins extendidos, + `aux_` | 0,162 | **0,577** | 0,150 |
| control binario, set base | 0,165 | 0,584 | 0,173 |
| control binario, + `aux_` | 0,169 | **0,594** | 0,170 |

**Δ ROC = −0,001 con bins extendidos y +0,010 con el control.** Ninguno salta (el
criterio es 0,03), así que el emparejado por odómetro × mes aguanta también con este
target. Vale notar que el extendido es el *menos* sensible de los dos al calendario, que
es lo contrario de lo que se temía al correr los bins hacia horizontes largos.

### (c) Importancias contra la hipótesis física

Ganancia media sobre los 5 folds del modelo del `Pipeline`, con los nombres que
devuelve el preprocesador, agrupada por las familias del plan §4:

| familia | bins extendidos | control binario |
|---|---:|---:|
| A · térmica / trayectos cortos | **40,4%** | **41,5%** |
| B · regeneración / DPF | 22,9% | 20,0% |
| C · uso | 10,6% | 10,1% |
| D · severidad / mensajes | 16,2% | 17,4% |
| sin familia física (estáticas, control de ventana) | 9,9% | 10,9% |

Las diez primeras del extendido: `oil_life_mean` (5,7%), `fuel_pct_per_100km_median`,
`speed_kmh_mean`, `engine_temp_avg_median`, `hours_between_trips_median`,
`trip_duration_median_min`, `idle_frac`, `regen_start_level_mean`,
`short_trip_frac_5km`, `cold_start_frac`. La hipótesis física del EDA —señal térmica y
de uso, progresiva: idle, no llegar a régimen, más lento— es lo que domina, con el 90%
de la ganancia en las cuatro familias y ninguna feature llevándose más del 6%. Cambiar
el target no reordenó la física: el ranking del extendido y el del control son casi el
mismo. No hay ninguna variable que sea casi la definición del evento arriba de la tabla
(regla 6), consistente con que el PR-AUC esté en 0,16 y no en 0,9.

## Reproducción de la variante extendida

```bash
source .venv/bin/activate
python scripts/check_setup.py                      # 50 chequeos
export FORD_DATA_DIR=data/v364 WANDB_MODE=disabled

python scripts/train.py --config configs/exp_ordinal_horizon_ext.yaml
python scripts/audit_ordinal_horizon.py --config configs/exp_ordinal_horizon_ext.yaml \
    --n-permutations 10 --seed 20260920
# el contraste con el control es parte de la auditoría, no un extra:
python scripts/audit_ordinal_horizon.py --config configs/exp_lgbm_panel_v1.yaml \
    --n-permutations 10 --seed 20260920

python scripts/results.py log f3-ordinal-horizon-ext --note "..."
python scripts/results.py table --out results/README.md
```

Las corridas versionadas están en `results/f3-ordinal-horizon-ext.yaml` y
`results/f3-ordinal-horizon.yaml`. Las dos son locales con `WANDB_MODE=disabled`
—el entorno de ejecución no tiene salida a W&B—, así que **no están en el proyecto
compartido `oromagnoli-/ford-fic`**; los YAML declaran `mode: online` y suben solas
desde una máquina con red, sin editar el config (CLAUDE.md, regla 9).
