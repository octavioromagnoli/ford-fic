# Demo de producto con la GRU con etiqueta suave (F11) y una perilla de falsas alarmas

**Fecha:** 2026-09-28 · **Fase:** F9 (producto, para el pitch del 02-10) · **Rama:** `feat/demo-gru` (con `main`
mergeado)
**Alcance:** entrega v2, solo dev (426 autos: 135 fallan, 291 sanos), ensamble por rango de la GRU con etiqueta
suave, semillas 101, 102 y 103, folds de confirmación; etiqueta dura, repetición 1 y umbral exacto en cuatro
puntos de operación. **Test sin tocar**: el bundle falla si trae un vehículo de test.

**No es un candidato ni cambia ninguna métrica.** El 28-09 el equipo adoptó la GRU con etiqueta suave como el
mejor modelo sobre dev v2 ([f11-gru-objetivo-suave.md](f11-gru-objetivo-suave.md)), y la demo pasa a usarla tal
cual: el score, el umbral y la alerta son los del ensamble de la confirmación. El mismo día, a pedido del usuario,
la demo suma una **perilla** para elegir el presupuesto de falsas alarmas entre 5, 10, 15 y 20% (hasta acá estaba
fijo en 5%). Lo que no cambia (el porqué descriptivo, los agentes, el verificador, el diseño) está en
[f9-demo-gru.md](f9-demo-gru.md).

```bash
export FORD_DATA_DIR=$PWD/data/v2 WANDB_MODE=disabled   # los paneles v2 de f9-demo-gru.md
python scripts/make_splits.py --config configs/data/splits_panel_v2_confirm_r3.yaml   # folds de confirmación (semilla 2026)
python scripts/train.py --config configs/exp_f11_gru_suave_conf_s101.yaml             # y s102, s103 (~15 min en paralelo)
python scripts/ensemble_rank.py --config configs/exp_f11_seeds3_gru_suave_conf.yaml
# la demo
python scripts/build_demo_bundle.py --config configs/demo.yaml          # experiments/demo-bundle-gru-suave/
python scripts/warm_demo_cache.py --config configs/demo.yaml --budgets 50    # un proceso por punto (50, 100, 150, 200)
python scripts/warm_demo_cache.py --config configs/demo.yaml --mode cache_only --prune   # después: verifica y poda
python scripts/build_demo_bundle.py --config configs/demo.yaml --publish-only   # wandb Artifact demo-bundle-gru-suave
streamlit run scripts/demo_app/app.py
```

## En una línea

**La demo reproduce la flota con el mejor modelo de v2, y la perilla deja ver qué pasa al tolerar más falsas
alarmas: al 5% se anticipa un tercio de las fallas; al 20%, siete de cada diez, con cuatro veces más alertas de
más.** Cada punto trae su propio replay, verificado contra el reporte, y su triage precalentado.

## El modelo, reconstruido

Las corridas de F11 son del equipo (Windows) y no estaban en disco: se reentrenaron las tres semillas con el
código de `main`.

| | esta Mac | la tabla de F11 |
|---|---|---|
| PR-AUC por fila del ensamble | 0,1642 | 0,1640 |
| detección al 5 / 10 / 15 / 20% | 34,8 / 48,6 / 60,5 / 70,4 | 34,6 / 48,6 / 60,2 / 70,1 |
| celda mercado × motor, sin modelo | 14,6 / 30,4 / 48,1 / 62,0 | 14,6 / 30,4 / 48,1 / 62,0: **idéntica** |
| nulo de bolsa, p95 | 9,1 / 15,8 / 22,0 / 26,9 | 9,6 / 15,8 / 21,8 / 26,9 |

- **La GRU no es bit a bit entre plataformas** (lo mismo en [f9-demo-gru.md](f9-demo-gru.md)). La diferencia es
  de a lo sumo un auto por repetición (un auto son 0,25 puntos del promedio). El pitch cita la tabla del equipo.
- `train.py` y `ensemble_rank.py` terminan con un `ValueError` si `experiments/` es un symlink fuera del repo (el
  log final hace `relative_to` del repo). Las salidas ya están escritas cuando falla: solo se pierde el log a
  wandb, que estaba apagado.

## La perilla

**Un bundle con cuatro replays.** `configs/demo.yaml` declara `operating_points: [50, 100, 150, 200]` (falsas
alarmas cada 1.000 sanos) y `default_budget_per_1000: 50`. El builder calcula una sola vez lo que no depende del
umbral (los cortes, los números oficiales, la referencia de los sanos, las fechas de los eventos) y, por cada
punto:
- el umbral exacto de la repetición 1;
- la alerta y cuándo se confirma;
- los cortes que la dispararon;
- los hábitos que se apartan de los sanos, el mensaje y las señales del filtro.

`vehicles.parquet` y `deviations.parquet` llevan la columna `budget_per_1000`, y la meta, un `operating_points`
con el umbral y los conteos de cada uno. **El replay de cada punto tiene que reproducir su repetición del
reporte**, como antes el del 5%, o el builder falla. Un punto que no está en `official.budgets` tampoco se ofrece.

**Se carga un punto a la vez.** `load_bundle(root, budget)` devuelve el bundle de ese punto, con la meta del
punto arriba (`threshold`, `budget_per_1000`, `counts`). El resto del código (política, hechos, triage, páginas)
no sabe que hay otros. Un bundle viejo, de un solo punto, falla con un mensaje claro. El triage se guarda en
`triage/faNNN/<lunes>.json`, uno por punto. La caché del LLM es una sola: los pedidos iguales (un auto que alerta
igual en dos puntos) se sirven una vez.

**En la app:**
- **La perilla va arriba de las tres pantallas**, en una franja propia debajo de la barra superior. También cambia
  «Qué pasó después». Es un `st.segmented_control` con `required=True`, y al lado dice lo que se anticipa en ese
  punto con el número oficial: «Anticipa el 34,8% de las fallas, medido en desarrollo. Hasta el 5% de los autos
  sanos recibe una alerta de más». Tiene un paso en «Cómo usar».
- **El punto vive en `_budget`**, una clave que no es de un widget, como la semana (`_week`): sobrevive a «Ver
  ficha» y al cambio de página.
- **La semana elegida se queda al mover la perilla:** así se ve qué cambia en la misma semana.
- **Las decisiones de la bandeja** (aprobar, descartar) se guardan por punto.
- **La tabla oficial** de «Qué pasó después» muestra los cuatro puntos, con el elegido marcado. En el celular se
  apila: cada medida con sus cuatro valores.

**Una sola semana de apertura:** la del 25-08-2025, la primera con al menos 3 alertas nuevas al 5%. Son 4 avisos
al conductor: tres de autos que fallaron después (VEH_0072, VEH_0451, VEH_0452) y uno de un auto sano (VEH_0427).
Al 20%, esa semana trae 5.

## Los números por punto (dev v2)

| | 5% | 10% | 15% | 20% |
|---|---|---|---|---|
| **detección oficial (3 repeticiones)** | **34,8% ± 4,8** | **48,6% ± 3,9** | **60,5% ± 2,1** | **70,4% ± 1,8** |
| fuera de muestra: detección / falsas alarmas | 34,6 / 5,2% | 49,4 / 10,1% | 59,3 / 15,1% | 69,6 / 20,4% |
| celda mercado × motor, sin modelo | 14,6% | 30,4% | 48,1% | 62,0% |
| nulo de bolsa (media / p95) | 5,5 / 9,1% | 10,8 / 15,8% | 15,7 / 22,0% | 20,8 / 26,9% |
| umbral exacto (R1) | 0,936 | 0,895 | 0,852 | 0,808 |
| replay: fallas anticipadas, de 135 | 44 | 61 | 78 | 95 |
| replay: sanos con alerta de más, de 291 | 14 | 29 | 43 | 58 |
| replay: alertas nuevas / escalamientos | 58 / 9 | 90 / 13 | 121 / 30 | 153 / 43 |
| replay: alertas con hábitos que nombrar (aviso) | 56 de 58 | 82 de 90 | 107 de 121 | 134 de 153 |
| replay: anticipación mediana (alerta → falla) | 16 sem · 7.200 km | 17 sem · 7.800 km | 19 sem · 8.500 km | 21 sem · 8.700 km |

AUC por auto: 0,85 agrupado, 0,78 dentro del mercado y 0,69 dentro de mercado × motor (la GRU anterior: 0,82,
0,76 y 0,66).

- **Contra la GRU anterior, al 5%:** 44 de 135 en el replay contra 39, y 34,8% oficial contra 28,1%. La semana de
  apertura pasa del 29-09 al 25-08.
- **Con más tolerancia, la celda se acerca:** al 5% da 14,6% contra 34,8% de la GRU; al 20%, 62,0% contra 70,4%.
  Lo que la GRU suma sobre la composición se ve más donde se tolera poco. La perilla muestra el costo: con cuatro
  veces más falsas alarmas se detecta el doble.
- **La anticipación crece con la tolerancia:** un umbral más bajo alerta antes al mismo auto.

## Los agentes

Sin cambios en los prompts, la política ni el verificador. El triage de las 211 semanas con eventos (sumando los
cuatro puntos) está precalentado con `gpt-5.4-mini-2026-03-17`: 1.870 respuestas en la caché, unos 3,0 M tokens
de entrada y 0,24 M de salida.

| | semanas | eventos | textos del agente | resúmenes del agente | intentos rechazados |
|---|---|---|---|---|---|
| 5% | 42 | 67 | 67 | 42 | 11 |
| 10% | 53 | 103 | 103 | 53 | 20 |
| 15% | 55 | 151 | 150 | 55 | 28 |
| 20% | 61 | 196 | 195 | 61 | 47 |

- **Quedan dos plantillas, las dos del mismo auto:** VEH_0001, en la semana del 04-08-2025 al 15% y al 20%. No
  tiene hábitos que nombrar, así que va al concesionario, y el resumen del taller insistió tres veces con «no hay
  hábitos para comparar», que el verificador rechaza desde el 28-09. Queda la plantilla, y la tarjeta muestra los
  tres intentos rechazados. Es el sistema haciendo lo que tiene que hacer: no se forzó.
- **La regla que más rechaza es esa misma** («para comparar»), seguida de la atribución («lo marcó por…») y del
  lenguaje causal.
- **Precalentar los puntos en paralelo tiene una carrera.** La caché es una sola. Si dos procesos piden lo mismo a
  la vez (un auto con los mismos hechos en dos puntos), el segundo pisa la respuesta del primero. El resto del
  recorrido del primero queda huérfano, y al reproducirlo falta. Pasó en 5 semanas. Se rehicieron en un solo
  proceso y una pasada final en `cache_only` confirmó que las 211 semanas salen enteras de la caché. **La pasada
  final con `--mode cache_only --prune` es obligatoria** después de correr en paralelo. La primera vez, esa pasada
  guardó las semanas faltantes como plantilla. Ahora `warm_demo_cache.py` no guarda una semana en la que el agente
  no corrió entero: deja el triage que había, la lista para rehacerla y no poda.

## Decisiones

1. **El ensamble de la confirmación (folds 2026, semillas 101 / 102 / 103), no el de la exploración.** Es el que
   dio la tabla de F11 y el que el equipo adoptó.
2. **Cuatro puntos, no un deslizador continuo.** Cada punto necesita su replay verificado y su triage
   precalentado: los 4 que pidió el usuario son los del criterio primario de F11 (5–20%).
3. **La lectura de la perilla usa el número oficial,** no el del replay: es el que no depende de la repetición.
   La ficha, la bandeja y la temporada siguen mostrando la repetición 1, como antes.
4. **Un bundle nuevo, `demo-bundle-gru-suave`.** El formato cambió, y el de la GRU anterior
   (`demo-bundle-gru:v1`) sigue sirviendo a su código: no se pisa.

## Límites

- Los de [f9-demo-gru.md](f9-demo-gru.md) siguen: el porqué es una comparación con la flota, parte de lo que se
  detecta es composición, la muestra está enriquecida en fallas y la efectividad de los avisos no está medida.
- **(a′) = +0,0006:** el modelo sabe sobre todo *qué auto* y poco *cuándo*, como todos los de v2.
- **Se apoya más en el país que la GRU anterior** (19,5% del |Δlogit| contra 11,4%, F11). El AUC dentro de
  mercado × motor igual sube (0,645 → 0,689).
- **El test no está medido para este modelo:** tiene su propio preregistro ([f11-preregistro-test.md](f11-preregistro-test.md)).

## Código

- `scripts/build_demo_bundle.py`: `operating_points` y `build_point` (un replay por punto, verificado contra el
  reporte); la meta trae `operating_points` y `default_budget_per_1000`.
- `src/agents/bundle.py`: `load_bundle(root, budget)`, `Bundle.budget`, `budgets` y `triage_dir`; `check_meta`
  rechaza un bundle de un solo punto.
- `src/agents/triage.py`: el triage se guarda por punto y lleva su `budget_per_1000`.
- `scripts/warm_demo_cache.py`: recorre todos los puntos (`--budgets` para correrlos en paralelo); `--prune` solo
  poda con todos y sin semanas faltantes, y una semana sin el agente entero no se guarda.
- `scripts/demo_app/`: la perilla (`common.budget_knob`, `wording.operating_line`), el estado por punto, la
  tabla oficial con los cuatro puntos, el paso de «Cómo usar» y los estilos (`theme.css`).
- `scripts/check_setup.py` (218 chequeos): el bundle de prueba se escribe con el formato nuevo, y un chequeo nuevo cubre la
  perilla (carga por punto, triage aparte, puntos inválidos, lo que lee la app, la validación del builder).
