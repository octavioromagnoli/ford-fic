# Ford FIC · Predicción temprana de degradación de eficiencia de combustión

Ford Innovation Challenge III — desafío *Data-Driven Powertrain Intelligence*.
Objetivo: anticipar, a partir de telemetría de viajes, qué vehículos van a
degradar su eficiencia de combustión, **con kilómetros de anticipación** y a un
costo de falsas alarmas explícito.

Plan completo: [`plan-implementacion-ford.md`](plan-implementacion-ford.md).
Reglas y contrato de datos: [`CLAUDE.md`](CLAUDE.md).

## Instalación

```bash
python3.12 -m venv .venv          # o: uv venv --python 3.12 .venv
source .venv/bin/activate         # .\.venv\Scripts\Activate.ps1 en Windows
pip install -r requirements.txt   # o: uv pip install -r requirements.txt
```

Los datos crudos no se versionan. Copiarlos a `data/raw/` (o exportar
`FORD_DATA_DIR=/ruta/a/los/datos`) y ajustar `configs/data/raw_sources.yaml`.

## Corrida end-to-end (F0)

Sin ningún dato real, el repo ya corre de punta a punta contra un panel dummy que
tiene el esquema exacto del contrato:

```bash
python scripts/make_dummy.py --config configs/data/dummy_v1.yaml   # panel + splits
python scripts/train.py --config configs/exp_dummy.yaml            # CV + métricas + wandb
python scripts/check_setup.py                                      # smoke test del harness
streamlit run scripts/dashboard.py -- --run experiments/f0-dummy-baserate
```

La corrida dummy usa un predictor por tasa base sobre features aleatorias: tiene
que dar PR-AUC ≈ tasa base y ROC-AUC ≈ 0,5. Si diera mejor, hay un bug.

El dashboard del finalista (K2) corre contra el panel real, solo con dev:

```bash
export FORD_DATA_DIR=$PWD/data/rebuild-0921 WANDB_MODE=disabled
python scripts/train.py --config configs/exp_ss_hw_r3.yaml              # K2
python scripts/eval_window_label.py --config configs/exp_ss_hw_r3.yaml  # etiqueta V y comparación
python scripts/decision_layer.py --config configs/exp_decision_k2.yaml  # curva y punto fuera de muestra
streamlit run scripts/dashboard_k2/app.py
```

## Cómo se lanza un experimento

Un experimento es un YAML en `configs/`. Nunca se edita código para cambiar un
hiperparámetro:

```bash
python scripts/train.py --config configs/exp_mi_experimento.yaml
python scripts/train.py --config configs/exp_mi_experimento.yaml --panel data/processed/panel.parquet
```

Cada corrida deja en `experiments/<run_name>/`: `predictions.parquet` (out-of-fold),
`lead_time_curve.csv`, `metrics.json` y el `config.yaml` completo, y loguea lo
mismo a wandb.

## wandb

Todas las corridas van al team **`oromagnoli-`**, proyecto **`ford-fic`**:
<https://wandb.ai/oromagnoli-/ford-fic>. Setup por persona, una sola vez:

```bash
wandb login                           # pega tu API key de wandb.ai/authorize
python scripts/train.py --config configs/exp_dummy.yaml
```

Hace falta estar invitado al team antes del primer `train.py`; si no, wandb
escribe la corrida en tu cuenta personal y no la ve nadie más.

El YAML manda (`wandb.entity`, `wandb.mode`), pero dos env vars lo pisan sin
tocar el config compartido:

```bash
WANDB_MODE=offline python scripts/train.py --config configs/exp_dummy.yaml
wandb sync wandb/offline-run-*        # subirla después, cuando haya red
WANDB_MODE=disabled ...               # iterar sin ensuciar el proyecto
```

## Flujos para agentes

Tres tareas recurrentes están escritas paso a paso, con las reglas antileakage
incluidas:

| Tarea | Claude Code | Cualquier otro agente |
|---|---|---|
| Implementar un modelo nuevo | `/mlmodel` | `.claude/skills/mlmodel/SKILL.md` |
| Lanzar un entrenamiento | `/train` | `.claude/skills/train/SKILL.md` |
| Comparar corridas | `/compare` | `.claude/skills/compare/SKILL.md` |

`AGENTS.md` apunta ahí para los agentes que no leen skills (Codex y compañía).
Es el mismo archivo en los dos casos: si cambia el flujo, se edita una vez.

## Estructura

```
src/          código (data, features, models, training, eval)
configs/      un YAML por experimento
scripts/      build_dataset.py, train.py, make_dummy.py, dashboard.py, check_setup.py
notebooks/    solo exploración, sin lógica
experiments/  outputs y checkpoints (gitignored)
data/         crudos y procesados (gitignored)
```
