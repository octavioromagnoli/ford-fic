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
| [f3-cadencia-y-ventana.md](f3-cadencia-y-ventana.md) | Por qué la ventana es de km y la cadencia puede ser quincenal (no son lo mismo) |
| [f3-posicion-en-la-serie.md](f3-posicion-en-la-serie.md) | **La posición del corte en su serie separa con P = 0,83**: el atajo que hereda cualquier feature acumulativa |
| [f3-secuencia-zself-cusum.md](f3-secuencia-zself-cusum.md) | `_zself` + CUSUM: se construyó, se midió y no da. Qué quedó y qué no queda descartado |
| [f3-evento-ficticio-y-ventana-de-riesgo.md](f3-evento-ficticio-y-ventana-de-riesgo.md) | **El atajo de posición, cerrado**: evento ficticio + ventana de riesgo bajan P de 0,84 a 0,49 y el modelo sube a ROC 0,703. Y la ablación fina: el leakage eran las estáticas, no el calendario |
| [f3-emparejado-posicion-vs-calendario.md](f3-emparejado-posicion-vs-calendario.md) | **Posición y calendario no se pueden emparejar a la vez** (es estructural, no de resolución), y la ablación muestra que el calendario da ROC 0,92 donde no se lo empareja. La solución es el evento ficticio |
| [f3-primer-modelo-y-emparejado-por-posicion.md](f3-primer-modelo-y-emparejado-por-posicion.md) | **El primer modelo real (PR-AUC 1,44× la tasa base, ROC 0,62) y por qué el multivariado no le gana al univariado.** Y qué pasa al emparejar los sanos también por posición: la señal sobrevive, el atajo no se cierra |
| [f3-barrido-de-relaciones.md](f3-barrido-de-relaciones.md) | **El barrido de los 7.310 cocientes y su nulo: nada pasa la barra del azar.** Las rarezas (reloj, viaje anterior, forma del reparto) tampoco. Por qué la ingeniería de features tocó techo |
| [f3-relaciones-y-literatura-dpf.md](f3-relaciones-y-literatura-dpf.md) | **Las relaciones entre features (`derived:`): el idle improductivo separa más que todo lo anterior.** Y los cinco mecanismos de la literatura de DPF, medidos y planos |
| [f3-esfuerzo-de-control-y-dosis.md](f3-esfuerzo-de-control-y-dosis.md) | **Las 16 candidatas físicas del menú, medidas: ninguna gana.** Y la contradicción del catalizador resuelta: el DPF no anticipa, y probablemente el evento no es obstrucción |
| [decisiones.md](decisiones.md) | Qué se decidió, cuándo y por qué |
| [../f2-feature-engineering-candidatas.md](../f2-feature-engineering-candidatas.md) | Candidatas de feature engineering medidas el 17-09. Lo que se adoptó y lo que se retiró está en el archivo de arriba |
| [../f3-features-candidatas-fisica.md](../f3-features-candidatas-fisica.md) | **Lo que sigue**: candidatas por mecanismo físico, empezando por la contradicción del catalizador |

## Cómo se reproduce todo esto

```bash
python scripts/eda_raw.py          # deja los CSV en experiments/eda/
python scripts/make_test_split.py  # auditoría del join + holdout dev/test congelado
python scripts/build_eda_cache.py  # cache dev-only del EDA (experiments/eda/dev/)
python scripts/eda_gaps.py         # complemento del EDA: factibilidad, perfil alineado al evento, calendario, monotonía del odómetro y cadencia (experiments/eda/dev/gaps/)
python scripts/build_dataset.py --config configs/data/panel_v1.yaml   # panel real + splits sobre dev + panel_meta.json
python scripts/log_panel_artifact.py --config configs/data/panel_v1.yaml  # publica panel-v1 y test-split como wandb Artifacts
python scripts/train.py --config configs/exp_baserate.yaml            # piso contra el panel real
python scripts/train.py --config configs/exp_logistic_l1.yaml         # el mejor modelo de F3 (y exp_{logistic,gbm,lgbm}.yaml)
python scripts/build_dataset.py --config configs/data/panel_posmatch.yaml  # sanos emparejados TAMBIÉN por posición en la serie
python scripts/build_dataset.py --config configs/data/panel_pseudo.yaml   # evento ficticio + ventana de riesgo: el panel sin atajo de posición
python scripts/train.py --config configs/exp_l1_v1_abl_static.yaml   # ablación POR FAMILIA (la gruesa detecta, la fina dice cuál)
python scripts/compare.py                                             # tabla comparativa de corridas
python scripts/build_dataset.py --config configs/data/panel_v2.yaml   # el mismo panel + las 16 candidatas físicas de F3 (medidas, negativas)
python scripts/audit_sequence.py --panel data/processed/panel.parquet  # atajo de posición + features de secuencia
python scripts/build_dataset.py --config configs/data/panel_v4.yaml   # + relaciones (`derived:`) y literatura de DPF
python scripts/build_dataset.py --config configs/data/panel_v5.yaml   # + rarezas: reloj, viaje anterior, forma del reparto
python scripts/audit_sequence.py --panel data/processed/panel_v4.parquet --columns all  # la misma auditoría, sobre cualquier feature
python scripts/check_setup.py      # 41 chequeos del harness
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
