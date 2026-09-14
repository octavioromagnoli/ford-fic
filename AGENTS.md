# AGENTS.md

Contexto y reglas del repo: **`CLAUDE.md`** (contrato de datos, reglas
antileakage, mapa del código, flujo de trabajo). Leelo antes de escribir código.

Plan completo del proyecto: `plan-implementacion-ford.md`.

## Flujos de trabajo

Tres tareas recurrentes están escritas paso a paso. Leé el archivo completo antes
de empezar la tarea correspondiente:

| Tarea | Archivo |
|---|---|
| Implementar un modelo nuevo | `.claude/skills/mlmodel/SKILL.md` |
| Lanzar un entrenamiento / leer sus métricas | `.claude/skills/train/SKILL.md` |
| Comparar corridas entre sí | `.claude/skills/compare/SKILL.md` |

En Claude Code son las skills `/mlmodel`, `/train` y `/compare`. Desde cualquier otro agente
son documentos comunes: mismo contenido, misma fuente de verdad.

## Reglas operativas mínimas

- El repo corre sobre `.venv/`, no sobre conda: `source .venv/bin/activate`.
- Nada de paths, semillas ni hiperparámetros hardcodeados: todo sale de un YAML
  de `configs/`. Para cambiar un hiperparámetro se escribe otro YAML.
- `python scripts/check_setup.py` (15 chequeos) tiene que dar verde antes de cada PR.
- Una rama por feature o experimento; `main` siempre funcional.
- `data/`, `experiments/` y `wandb/` no se versionan.
