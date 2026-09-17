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
| [f2-identificationdate-igual-a-venta.md](f2-identificationdate-igual-a-venta.md) | **Bloqueante:** el 79% de los eventos no tiene fecha utilizable |
| [f2-calidad-columnas-dev.md](f2-calidad-columnas-dev.md) | Tres defectos de columna que la tabla de nulos no muestra |
| [decisiones.md](decisiones.md) | Qué se decidió, cuándo y por qué |

## Cómo se reproduce todo esto

```bash
python scripts/eda_raw.py          # deja los CSV en experiments/eda/
python scripts/make_test_split.py  # auditoría del join + holdout dev/test congelado
python scripts/build_eda_cache.py  # cache dev-only del EDA (experiments/eda/dev/)
python scripts/check_setup.py      # 15 chequeos del harness
```

Los cuatro hallazgos `f2-*` de arriba salen del EDA exhaustivo sobre dev:
[`notebooks/eda-exhaustivo-dev.ipynb`](../../notebooks/eda-exhaustivo-dev.ipynb)
(ejecutado, con sus figuras), y su versión navegable en
`streamlit run scripts/dashboard_eda.py`. Los dos leen el mismo cache, así que el
filtro a dev se decide en un solo lugar.
