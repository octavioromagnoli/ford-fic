---
name: train
description: Lanzar un entrenamiento en este repo (Ford FIC) de punta a punta — entorno, panel, splits congelados, config YAML, corrida y lectura de métricas en wandb. Usar cuando se pida entrenar, correr un experimento, reproducir una corrida, comparar modelos o interpretar el resultado de scripts/train.py.
---

# Correr un entrenamiento

Un experimento = un YAML. `scripts/train.py` es el **único** entrypoint: hace
siempre lo mismo, en el mismo orden, y deja todo en wandb y en
`experiments/<run_name>/`. Nunca se edita para cambiar un hiperparámetro.

## 0. Entorno (una sola vez)

```bash
source .venv/bin/activate      # el repo NO corre sobre conda base
wandb login                    # API key de wandb.ai/authorize
```

Hace falta estar invitado al team `oromagnoli-` antes de la primera corrida; si
no, wandb escribe el run en tu cuenta personal, sin error visible, y no lo ve
nadie más. Verificalo con `wandb.Api().default_entity`.

## 1. Panel

```bash
python scripts/make_dummy.py --config configs/data/dummy_v1.yaml   # dummy (F0)
python scripts/build_dataset.py --config configs/data/panel_v1.yaml  # real (F2)
```

El panel real todavía no existe: `build_dataset.py` levanta `NotImplementedError`
hasta que F2 lo materialice. Contra el dummy, los números **tienen** que dar
basura: es el test del harness, no un resultado.

## 2. Splits

Son **dos niveles**, y el de arriba es obligatorio declararlo.

**Holdout dev/test** (`data/processed/test_split.json`, congelado antes de F2):
`train.py` recorta el panel a dev antes de armar los folds, así que ninguna
corrida ve el test. El YAML **tiene** que declarar `splits.test_split`: el path del
holdout contra el panel real, o `null` explícito contra el dummy (que no tiene).
Si la clave falta, `train.py` corta con `KeyError` y no entrena. Es a propósito:
olvidarse del holdout no rompe nada, solo da un PR-AUC mejor del que corresponde.
El test se mide **una vez**, con el modelo ya elegido, y esa corrida se acuerda
entre los tres.

**Folds de CV** (dentro de dev): se generan **una vez** y se congelan; viven en
`data/processed/splits*.json` y se comparten como wandb Artifact. `train.py` los
regenera solo si el config dice
`splits.build_if_missing: true` (cierto en el dummy, **false** contra el panel
real). `splits.strict: true` verifica que los splits sean de este panel: si
falla, el panel cambió de vehículos y hay que regenerarlos **y avisarle a los
otros dos**, porque sus números dejan de ser comparables con los tuyos.

## 3. Config

Copiá `configs/exp_dummy.yaml` y ajustá `name` (es el nombre del run y del
directorio de salida), `data.panel`, `model.name`/`model.params`, `eval` y
`wandb.group`/`tags`. Modelos disponibles:
`python -c "from src.models.registry import available_models; print(available_models())"`.
Para agregar uno, ver la skill `mlmodel`.

## 4. Corrida

```bash
python scripts/check_setup.py                                       # 15 chequeos antes de nada
WANDB_MODE=disabled python scripts/train.py --config configs/exp_x.yaml   # ensayo, no ensucia el proyecto
python scripts/train.py --config configs/exp_x.yaml                 # la de verdad, va a wandb
```

Overrides útiles sin tocar el config compartido: `--panel <path>` (pasar de dummy
a real), `--run-name <nombre>`, `WANDB_MODE=offline` + `wandb sync wandb/offline-run-*`
si estás sin red, `WANDB_ENTITY` para apuntar a otra cuenta.

## 5. Leer el resultado

`train.py` loguea tres cosas y hay que mirarlas en este orden:

1. **`oof/pr_auc` contra `oof/base_rate`.** Es el número de selección de modelo.
   PR-AUC ≈ tasa base significa sin señal. PR-AUC alto de entrada **se audita**:
   revisá el gap de blanking y las features casi-definicionales del evento
   (nivel de DPF) antes de festejar nada.
2. **Punto de operación** (`pitch/*`): detección y anticipación mediana bajo el
   presupuesto de falsas alarmas de `eval.max_false_alarms_per_1000`. Es el
   número de portada del pitch. Si ningún umbral entra en el presupuesto,
   `train.py` lo dice explícitamente y no hay punto de operación.
3. **`cv/*_mean` con su intervalo bootstrap.** Una diferencia entre modelos que
   cabe dentro de los intervalos no es una diferencia.

Accuracy no se mira nunca. Outputs locales en `experiments/<run_name>/`
(`predictions.parquet` out-of-fold, `lead_time_curve.csv`, `metrics.json`,
`config.yaml`), gitignoreados: lo que se comparte va por wandb.

## 6. Ver y comparar

```bash
streamlit run scripts/dashboard.py -- --run experiments/<run_name>
```

Para comparar modelos, misma `wandb.group` y mismos splits; si cambiaste los
splits, no estás comparando modelos.

## 7. Antes del PR

`python scripts/check_setup.py` en verde, el checklist de trampas técnicas del
plan §9, y cada número del informe linkeado a su corrida de wandb (nada copiado
a mano).

## Si algo falla

- `FileNotFoundError` del panel → paso 1.
- `El config no declara splits.test_split` → agregale la clave al YAML: el path
  del holdout congelado, o `null` explícito si el panel no tiene (solo el dummy).
- `N vehículo(s) del panel no están en el holdout congelado` → el panel trae
  vehículos que el holdout no conoce; sin esa verificación se irían a dev por
  descarte. Regenerá el holdout **y avisá al equipo**: cambia el test de todos.
- `No existen los splits ... y build_if_missing es false` → los splits son un
  artefacto compartido: bajalos del wandb Artifact, no los generes en silencio.
- `Los splits no corresponden a este panel` → el panel cambió; regenerá splits y
  avisá al equipo.
- `El panel no tiene ninguna columna con prefijo feat_/static_` → el panel no
  cumple el contrato de `CLAUDE.md`.
- `N filas quedaron sin predicción out-of-fold` → los folds no cubren el panel.
