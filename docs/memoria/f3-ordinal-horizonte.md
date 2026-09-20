# Target ordinal multi-horizonte y matriz de costos

**Fecha:** 2026-09-20 · **Fase:** F3 · **Alcance:** dev del panel v1 (2.029
filas, 171 vehículos, 254 positivas, tasa base 0,125), mismos folds congelados que
`f3-lgbm-panel-v1`. · **Config:** `configs/exp_ordinal_horizon.yaml`.

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
