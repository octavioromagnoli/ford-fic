# Memoria del proyecto

Lo que cualquiera —persona o agente— necesita saber sobre **los datos y las
decisiones** antes de escribir código, y que no se deduce leyendo el repo.

`CLAUDE.md` dice cómo se trabaja. Acá está qué encontramos y qué decidimos.

## Reglas

- **Un archivo = un hecho.** Kebab-case, prefijo de la fase que lo produjo
  (`f1-...`). Si un hecho cambia, se edita ese archivo; no se apila un post-scriptum.
- **Cada hecho trae su evidencia**: el número, y el comando que lo reproduce. Un
  hallazgo sin forma de re-verificarlo es un rumor.
- **Las decisiones van en [`decisiones.md`](decisiones.md)**, con fecha y motivo.
  El motivo importa más que la decisión: sin él, el que venga la revierte sin saber.
- **Esto se versiona.** `.claude/` y `.agents/` están en `.gitignore` (son
  artefactos locales de cada uno); la memoria compartida vive acá adentro a propósito.
- Los datos crudos NO se citan textualmente acá: van agregados o como estadística.

## Índice

| Archivo | Qué contesta |
|---|---|
| [f1-datos-reales.md](f1-datos-reales.md) | Qué hay en `data/raw/`, con qué esquema y cómo joinea |
| [f1-clones-vehiculos.md](f1-clones-vehiculos.md) | 13 vehículos contados cuatro veces, y por qué rompen el split |
| [f1-sesgo-eng3.md](f1-sesgo-eng3.md) | `ENG_3` no aparece entre los fallados: sesgo de muestreo |
| [f1-anclaje-temporal.md](f1-anclaje-temporal.md) | El eje de días sí se puede anclar al de km |
| [f1-senal-postratamiento.md](f1-senal-postratamiento.md) | Cuánto separan `Message`, `Acumulation` y las regeneraciones |
| [f1-calidad-odometro.md](f1-calidad-odometro.md) | Nulos, retrocesos y desfases del eje de odómetro |
| [f2-union-y-holdout-dev-test.md](f2-union-y-holdout-dev-test.md) | Cómo une (y cómo NO une) el join, y el test 80/20 congelado |
| [f2-diccionario-trips-incompleto.md](f2-diccionario-trips-incompleto.md) | `TripSummary` trae 25 de las 40 columnas del anexo: qué features mueren |
| [f2-columnas-airfilter-airregeneration.md](f2-columnas-airfilter-airregeneration.md) | El DPF estaba en `trips` con otro nombre: la familia B se puede construir |
| [f2-identificationdate-igual-a-venta.md](f2-identificationdate-igual-a-venta.md) | El 78% de los eventos no tiene fecha utilizable (resuelto por el de abajo) |
| [f2-universo-fecha-usable.md](f2-universo-fecha-usable.md) | Por qué el estudio son 364 vehículos y no 1081, y cómo se recortó sin re-sortear |
| [f2-calidad-columnas-dev.md](f2-calidad-columnas-dev.md) | Tres defectos de columna que la tabla de nulos no muestra |
| [f2-eda-revision-y-features.md](f2-eda-revision-y-features.md) | **Revisión del EDA (18-09) y panel v1**: el marcador `Regenerations` cortado, el confusor calendario, los idle de 0 km, la terna, el emparejado y cuánta señal hay de verdad |
| [f3-cnn-lstm-tutora.md](f3-cnn-lstm-tutora.md) | La solución de la tutora (CNN-LSTM, rama dinámica + estática) como baseline: cómo se interpretó, cuánto da y qué auditorías pasó |
| [f2-fechas-formato-mixto.md](f2-fechas-formato-mixto.md) | Las fechas no tienen nulos: el 0,3% era parseo de dos formatos mezclados, y también rompía el dedupe de `signals` |
| [f2-umbral-regeneraciones.md](f2-umbral-regeneraciones.md) | Por qué las caídas de 5 puntos son ruido y el detector pasa a exigir 15 |
| [f3-mil-agregacion-vehiculo.md](f3-mil-agregacion-vehiculo.md) | Medir por vehículo y no por corte (MIL): cuánto sube el lift de verdad y cuánto es el tamaño de la bolsa |
| [f3-timesfm-zeroshot.md](f3-timesfm-zeroshot.md) | TimesFM zero-shot en el panel v1: no le gana a la tasa base, pero señaló el desvío respecto de la historia del vehículo |
| [f3-survival-stacking.md](f3-survival-stacking.md) | Supervivencia en tiempo discreto sobre el panel: empata en PR-AUC pero calibrado y con el doble de detección; por qué el efecto aleatorio por vehículo no paga; y por qué la auditoría (a) de §0.4 no es un null en este panel |
| [f3-piso-posicional.md](f3-piso-posicional.md) | El odómetro solo le gana a los cuatro finalistas en PR-AUC por fila y en las tres métricas del eje "cuándo": qué métricas quedan descalificadas y cuáles dos sobreviven |
| [decisiones.md](decisiones.md) | Qué se decidió, cuándo y por qué |
| [../f2-feature-engineering-candidatas.md](../f2-feature-engineering-candidatas.md) | Candidatas de feature engineering medidas el 17-09. Lo que se adoptó y lo que se retiró está en el archivo de arriba |

## Cómo se reproduce todo esto

```bash
python scripts/eda_raw.py          # deja los CSV en experiments/eda/
python scripts/make_test_split.py  # auditoría del join + holdout dev/test congelado
python scripts/build_eda_cache.py  # cache dev-only del EDA (experiments/eda/dev/)
python scripts/eda_gaps.py         # complemento del EDA: factibilidad, perfil alineado al evento, calendario (experiments/eda/dev/gaps/)
python scripts/build_dataset.py --config configs/data/panel_v1.yaml   # panel real + splits sobre dev + panel_meta.json
python scripts/log_panel_artifact.py --config configs/data/panel_v1.yaml  # publica panel-v1 y test-split como wandb Artifacts
python scripts/train.py --config configs/exp_baserate.yaml            # piso contra el panel real
python scripts/train.py --config configs/exp_lgbm_panel_v1_mil.yaml    # el mismo modelo, medido por vehículo
python scripts/audit_mil_bagsize.py f3-lgbm-panel-v1-mil              # ¿el lift por vehículo es señal o tamaño de bolsa?
python scripts/audit_model.py --config configs/exp_<x>.yaml           # las auditorías obligatorias de F3 §0.4
python scripts/build_positional_panel.py --config configs/data/panel_positional.yaml  # panel de una sola feature: el odómetro
python scripts/audit_positional_floor.py f3-survival-stacking         # ¿le gana al odómetro pelado? (piso del eje "cuándo")
python scripts/check_setup.py      # chequeos del harness
```

> Ojo: `notebooks/eda-exhaustivo-dev.ipynb` §6.2, §6.3 y §7.4 muestran `regen_per_1000km`
> como la señal más fuerte. Está medido con el marcador `Regenerations`, que se corta el
> 25-05-2026: es exposición al calendario, no física. La versión vigente es
> [f2-eda-revision-y-features.md](f2-eda-revision-y-features.md) §2.1.

Los hallazgos `f2-*` de arriba salen del EDA exhaustivo sobre dev:
[`notebooks/eda-exhaustivo-dev.ipynb`](../../notebooks/eda-exhaustivo-dev.ipynb)
(ejecutado, con sus figuras), y su versión navegable en
`streamlit run scripts/dashboard_eda.py`. Los dos leen el mismo cache, así que el
filtro a dev se decide en un solo lugar.
