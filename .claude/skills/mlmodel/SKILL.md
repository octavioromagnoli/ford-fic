---
name: mlmodel
description: Implementar un modelo nuevo en este repo (Ford FIC) siguiendo el registry, el contrato de features por prefijo y las reglas antileakage. Usar cuando se pida agregar, registrar o probar un modelo (logística, GBM, LightGBM, XGBoost, supervivencia, regla física, ensamble) o cuando haya que escribir un builder en src/models/registry.py.
---

# Implementar un modelo nuevo

Agregar un modelo en este repo son **dos archivos**: un builder en
`src/models/registry.py` y un YAML en `configs/`. Si estás por tocar
`scripts/train.py`, `src/training/cv.py` o `src/eval/splits.py`, parate: casi
seguro el modelo se está saltando una garantía del repo.

## Antes de escribir código

1. Leé `CLAUDE.md` (contrato de datos y las 9 reglas no negociables).
2. Rama propia: `git checkout -b feat/model-<nombre>`. Los modelos son **Track B**
   (`src/models`, `src/training`); no edites `src/data`, `src/features` ni
   `src/eval` en el mismo PR — es de otra persona.
3. `python -c "from src.models.registry import available_models; print(available_models())"`
   para no registrar un nombre que ya existe (`register` falla si se duplica).

## Qué recibe tu modelo (y qué no)

`src/training/cv.py` lo envuelve en un `Pipeline` y le pasa **solo** las columnas
`feat_*` y `static_*` del panel, ya imputadas, escaladas y one-hot-eadas, fiteado
todo con el train de cada fold. Tu builder recibe un `dict` de params del YAML y
devuelve un estimador. Concretamente:

- **No** hagas imputación, escalado ni encoding dentro del modelo: ya está hecho.
- **No** mires el panel, `vehicle_id`, `cut_odo` ni `time_to_event_km`: al modelo
  solo le llegan features y `y = label` binaria.
- **No** hagas resampling (SMOTE, undersampling) fuera del pipeline: es leakage.
  Para desbalance usá `class_weight="balanced"` o `scale_pos_weight`, que se
  ajustan dentro del fold.
- Sí exponé `predict_proba` (preferido) o `decision_function`; `cv.py` toma la
  columna de la clase `1` vía `classes_`, y cae a `decision_function` si no hay
  probabilidad. Un modelo sin ninguno de los dos rompe la corrida.
- La semilla va por params desde el YAML (`random_state`), nunca hardcodeada.

## El builder

```python
@register("lgbm")
def _build_lgbm(params: dict[str, Any]) -> BaseEstimator:
    """Una línea de por qué este modelo está en la escalera."""
    from lightgbm import LGBMClassifier   # import adentro: no encarece el import del registry

    params.setdefault("n_estimators", 300)
    params.setdefault("learning_rate", 0.05)
    params.setdefault("num_leaves", 15)        # flota chica: regularización fuerte
    params.setdefault("class_weight", "balanced")
    params.setdefault("random_state", 42)
    return LGBMClassifier(**params)
```

Los `setdefault` son el default razonable del modelo; **todo** lo que alguien
quiera variar se sobreescribe desde el YAML, nunca editando esta función.

## El config

Copiá `configs/exp_dummy.yaml` a `configs/exp_<nombre>.yaml` y cambiá `name`,
`model.name`, `model.params` y `wandb.group`/`tags`. Contra el panel real,
`splits.build_if_missing` va en **false**. Para barrer un hiperparámetro se
escribe otro YAML, no se edita Python.

## Verificación (obligatoria antes del PR)

```bash
python scripts/check_setup.py                                        # 15 chequeos, todos verdes
WANDB_MODE=disabled python scripts/train.py --config configs/exp_<nombre>.yaml
```

Leé el resultado contra estos tres criterios:

- **Piso**: si no le gana a `baserate` (PR-AUC ≈ tasa base), el modelo está roto
  o el problema no tiene señal; averiguá cuál de las dos antes de seguir.
- **Techo sospechoso**: PR-AUC alto de entrada se **audita**, no se celebra.
  Revisá que el gap de blanking esté aplicado y que ninguna feature sea casi la
  definición del evento (nivel de DPF y similares). Regla 6 de `CLAUDE.md`.
- **Métrica**: PR-AUC out-of-fold para elegir, curva de anticipación para el
  pitch. Accuracy nunca.

Después, corrida online (`python scripts/train.py --config ...`) para que quede
en wandb, y el checklist de trampas técnicas del plan §9 en el PR.

## Límite conocido: modelos de supervivencia (F5b)

`run_cv` entrega `y` binaria y descarta `time_to_event_km` antes de llegar al
modelo, así que un Cox/AFT de `lifelines` **no entra solo registrando un
builder**: necesita o un adapter que exponga la API sklearn y convierta el riesgo
a P(evento en el horizonte), o una extensión de `run_cv` para pasar el target de
supervivencia. Eso es un cambio de contrato: se discute en un issue antes de
escribirlo, no se resuelve dentro del builder.

## Errores que ya sabemos que van a tentar

- Editar `build_preprocessor` en `cv.py` porque "LightGBM no necesita escalado":
  ese archivo es compartido y cambiarlo mueve los números de **todos** los
  modelos ya corridos. Si hace falta preprocesado por modelo, se propone en PR.
- Regenerar los splits para que "dé mejor". Los splits se congelan una vez y se
  comparten; `strict: true` está justamente para que esto falle ruidosamente.
- Hiperparámetros en el código. Van al YAML, siempre.
