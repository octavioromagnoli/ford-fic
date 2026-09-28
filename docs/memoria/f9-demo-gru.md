# Demo de producto con la GRU final (v2): el porqué pasa a ser una comparación con la flota

> **Reemplazada el 28-09** por [f9-demo-gru-suave.md](f9-demo-gru-suave.md): la demo usa la GRU con etiqueta suave
> de F11 y tiene una perilla de falsas alarmas. Lo de acá (el porqué descriptivo, los agentes, la revisión visual)
> sigue valiendo; los números y la semana de apertura son de la GRU anterior.

**Fecha:** 2026-09-27 (revisión visual y bundle v1: 28-09) · **Fase:** F9 (producto, para el pitch del 02-10) ·
**Rama:** `feat/demo-gru`
**Alcance:** entrega v2, solo dev (426 autos: 135 fallan, 291 sanos), ensamble por rango de la GRU +
TripSummary + estática completa con las semillas 42, 1 y 2, etiqueta dura, repetición 1, 5% de falsas
alarmas con umbral exacto. **Test sin tocar**: el bundle falla si trae un vehículo de test.

**No es un candidato ni cambia ninguna métrica.** La demo deja de usar K2 (pedido del equipo, 27-09). El
modelo es la GRU que el sweep dejó como configuración final ([f10-sweep-gru.md](f10-sweep-gru.md)), sin tocar:
el score, el umbral y la alerta son los del ensamble tal cual. `demo-bundle` (la demo con K2) queda intacto en
wandb; esta sale como `demo-bundle-gru`.

```bash
export FORD_DATA_DIR=$PWD/data/v2 WANDB_MODE=disabled   # data/v2/raw: la entrega v2 (los sanos, enlazados a data/raw)
python scripts/make_test_split.py --config configs/data/test_split.yaml --force   # 446 dev / 111 test / 434 fuera
python scripts/build_dataset.py --config configs/data/panel_v2.yaml
python scripts/build_dataset.py --config configs/data/panel_v2_estaticas.yaml
python scripts/make_splits.py --config configs/data/splits_panel_v2_r3.yaml
python scripts/build_seq_panel.py --config configs/data/panel_seq_trips_v2_estaticas.yaml
python scripts/build_survival_panel.py --config configs/data/panel_survival_v2.yaml
python scripts/train.py --config configs/exp_v2_ss_r3.yaml          # referencias que pide el ensamble
python scripts/train.py --config configs/exp_v2_lgbm_r3.yaml
python scripts/train.py --config configs/exp_v2all_gru_trips_estaticas_r3.yaml     # y _s1_r3, _s2_r3
python scripts/ensemble_rank.py --config configs/exp_v2all_seeds3_gru_trips_estaticas.yaml
# la demo
python scripts/build_demo_bundle.py --config configs/demo.yaml          # experiments/demo-bundle-gru/
python scripts/warm_demo_cache.py --config configs/demo.yaml --prune    # agentes en todas las semanas (.env con OPENAI_API_KEY)
python scripts/build_demo_bundle.py --config configs/demo.yaml --publish-only   # wandb Artifact demo-bundle-gru
streamlit run scripts/demo_app/app.py
```

## En una línea

**La flota de dev v2 se reproduce semana a semana y cada alerta de la GRU llega a la bandeja con sus textos,
pero el "por qué" ya no sale del modelo: es en qué hábitos se aparta el auto de los autos sanos de su mercado.**
- **Números oficiales (umbral exacto, 3 repeticiones):** 28,1% ± 2,8 al 5% de falsas alarmas (27,9% fuera de
  muestra) y 42,5% al 10%. El azar con el mismo historial da 5,4% y 10,7%. La celda mercado × motor sola, sin
  mirar un viaje, da 18,0% y 32,6%.
- **El replay (repetición 1, 5%)** da 53 alertas y 7 escalamientos, entre las semanas del 30-06-2025 y del
  27-07-2026. De 135
  autos que fallan, 39 reciben la alerta antes, con una mediana de 15 semanas (7.000 km) hasta la falla
  registrada. 14 de 291 sanos reciben una alerta de más.
- **El porqué no es lo que usó el modelo.** La GRU no tiene atribución, y 7 de sus 15 canales son el estado del
  propio filtro (nivel, regeneraciones, avisos). El texto dice "comparado con autos sanos de tu mercado, tu auto…", y el verificador
  rechaza el marco viejo ("se parece a autos que fallaron") y la atribución ("lo marcó por…").

## El modelo, reconstruido

La entrega v2 no estaba en disco. Se reconstruyó de punta a punta con el código de `main` y se verificó en cada
paso contra lo que dejó el equipo:

| paso | lo que se comparó | resultado |
|---|---|---|
| holdout v2 | tamaños y eventos ([f9-universo-v2.md](f9-universo-v2.md)) | 446 / 111 / 434 · 177 eventos (173 interpolados) |
| panel v2 (dev) | [f9-remedicion-v2.md](f9-remedicion-v2.md) | 11.703 filas · 426 autos · 682 positivas |
| survival stacking y LightGBM v2 | `results/f9-v2-ss-r3.yaml`, `results/f9-v2-lgbm-r3.yaml` | **idénticos bit a bit** (PR-AUC y ROC) |
| GRU, semillas 42 / 1 / 2 | `results/v2all-gru-trips-estaticas-*.yaml` | PR-AUC 0,1597 / 0,1426 / 0,1452 contra 0,1596 / 0,1428 / 0,1454 |
| ensamble, detección al 5 / 10 / 15 / 20% | la tabla de [f10-sweep-gru.md](f10-sweep-gru.md) | 28,1 / 42,5 / 53,1 / 61,0 contra 27,4 / 42,5 / 53,3 / 60,7 |
| piso de la celda mercado × motor | ídem | 18,0 / 32,6 / 48,1 / 62,2: **idéntico** |

- **La GRU no es bit a bit.** En esta Mac, 1, 2, 3 y 4 hilos de torch dan el mismo número exacto, así que la
  diferencia con las corridas del equipo (Windows) es de plataforma, no de hilos. Las referencias de LightGBM sí
  son idénticas.
- **La diferencia en la detección es de a lo sumo un auto por presupuesto** (un auto son 0,74 puntos). La demo
  usa la reconstrucción; la tabla del equipo sigue siendo la referencia del pitch ("~27–29% al 5%").

## Los números al 5% (dev v2, mismas filas y folds)

| | detección al 5% | al 10% | fuente |
|---|---|---|---|
| **GRU ×3 semillas (esta demo)** | **28,1% ± 2,8** (39 / 42 / 33 de 135) | **42,5% ± 1,9** | `meta.json` del bundle |
| fuera de muestra (τ con los sanos de los otros folds) | 27,9% (4,9% de falsas alarmas) | 42,5% (10,2%) | ídem |
| survival stacking (F3 sobre v2) | 14,1% | 23,7% | [f10-sweep-gru.md](f10-sweep-gru.md) |
| K2 sin su ventana (horizonte completo) | 12% | 22% | [f9-remedicion-completa-v2.md](f9-remedicion-completa-v2.md) |
| celda mercado × motor, sin modelo | 18,0% | 32,6% | `meta.json` |
| azar con el mismo historial | 5,4% | 10,7% | `meta.json` |

K2 con la etiqueta corregida (17,0% ± 3,8 al 5%) es de la entrega 1 y no es comparable: v2 no tiene ventana
del registro. AUC por auto: 0,82 agrupado, 0,76 dentro del mercado y 0,66 dentro de mercado × motor.

## El porqué, sin atribución

`src/eval/fleet_profile.py`. Para cada auto, el promedio de los cortes que dispararon la alerta se compara con
la mediana de los autos sanos de dev de su mercado (una mediana por auto). Es la misma referencia que las señales
del filtro que ve el taller. Tres reglas, las del mensaje de F4:
- **Solo hábitos de la lista cerrada:** accionables con signo físico ≠ 0 en la clasificación preregistrada
  (`configs/explain_k2.yaml`: es física del DPF, no depende del modelo) y con frase y recomendación en
  `configs/explain_texts.yaml`. Síntomas y contexto no se nombran nunca: siguen yendo solo al taller.
- **Solo del lado riesgoso:** más idle es peor, más velocidad no. Un hábito del lado de los sanos no se nombra ni
  se dibuja.
- **"Se aparta" es superar al 75% de los sanos del mercado** hacia ese lado, y se nombran hasta 3. El umbral se
  fijó en `configs/demo.yaml` antes de construir el bundle.

48 de las 53 alertas tienen algún hábito que nombrar (aviso al conductor). Las otras 5 van directo al
concesionario, porque no hay nada que pedirle al conductor. El waterfall de la ficha se reemplazó por un gráfico
de desvíos contra la flota sana, con el título "Comparado con los autos sanos de su mercado".

## Los agentes

Los prompts, los motivos de la política y las plantillas pasaron al marco de la flota. El verificador **solo
sumó** reglas:
- rechaza `se parec…` y "marcó / alertó / detectó … por su/tu/el…";
- exige "sanos" en el cuerpo de un aviso;
- prohíbe hablar de revisión, concesionario o turno en el asunto de un aviso.

Ninguna regla anterior se sacó (`check_setup.py` lo controla). Revisando a mano la primera semana (4 tarjetas)
aparecieron tres cosas, y se corrigieron antes de precalentar:
1. el resumen decía que "la bandeja quedó aprobada";
2. un aviso llevaba de asunto "Revisión preventiva del DPF";
3. el aviso final decía "esto compara tu uso" en mensajes sin hábitos.

El precalentado de la v0 cubre 36 semanas con 60 eventos (53 alertas y 7 escalamientos), con 298 respuestas en la
caché (`gpt-5.4-mini-2026-03-17`; la v1 tiene 302, ver «Revisión visual»). Los 60 textos y los 36 resúmenes son del agente: ninguna plantilla y ninguna
acción rechazada. El verificador atajó 10 intentos: 6 por lenguaje causal ("porque"), 3 por atribución y 1 por
un asunto de revisión en un aviso. Al precalentar aparecieron dos cosas más, y se corrigieron:
- en la semana de apertura, el triage pedía la acción y redactaba en el mismo paso: ahora usa una herramienta
  por paso (`llm.parallel_tool_calls: false`);
- dos resúmenes contaban el trabajo de las herramientas: el prompt pide describir la semana, y el verificador
  rechaza "quedó aprobado" y "plantilla".

## Revisión visual y bundle v1 (28-09)

La extensión de Chrome no estaba conectada, así que la revisión se hizo con un Chrome headless manejado por
Playwright, que espera a que Streamlit termine de dibujar por websocket. Se miró a 1440×900 y a 390×844 (táctil),
con capturas, consola, requests y desborde horizontal. Lo roto se corrigió en `scripts/demo_app/`, sin tocar datos:
- **El gráfico de desvíos de la ficha.** Con `autosize: fit`, las filas quedaban en 21 px y cada barra caía en el
  renglón del hábito siguiente. El eje iba de 0 a 100, y en el celular el título del eje achicaba el gráfico a
  ~165 px. Ahora las filas miden 60 px (`fit-x`), con el hábito y sus valores arriba de la barra. El eje va de 50 a
  100 y la leyenda no se trunca. El pie aclara que se nombran hasta 3: VEH_0563 tiene un cuarto hábito sobre el
  75% que no se nombra.
- **El calendario de 82 semanas.** En el celular se pisaban 17 de las 18 etiquetas de mes y cada semana medía
  2 px. Ahora los meses que no entran se ocultan (medido en el navegador), el año va debajo y las semanas se tocan.
- **«Qué pasó después».**
  - Los meses están en castellano, y el eje y la leyenda se repiten arriba (son 53 filas).
  - La tabla oficial ya no desborda en el celular, y el azar lleva un decimal (5,4% y 10,7%).
  - El dock ya no muestra la semana.
- **Consola:** no hay errores. Quedan avisos de Vega, «Infinite extent», porque Streamlit crea la vista antes de
  insertar los datos. También quedan los del iframe de los contadores.

La revisión a mano de la semana de apertura (las 4 tarjetas, sus hechos y el resumen) encontró un texto
impreciso. El resumen del taller de VEH_0451 decía «No hay hábitos del conductor que se puedan comparar con autos
sanos», pero la comparación se hizo y ningún hábito se aparta. El verificador **sumó tres reglas**, sin sacar
ninguna:
- «se puedan comparar»;
- un hábito como motivo de la acción («VEH_0563 va con aviso por temperatura media del motor de 58 °C…»), que
  apareció al re-precalentar;
- los ids internos («turno_concesionario»), que ya estaban en dos resúmenes de la v0.

Entre los 276 textos del bundle, las reglas atrapan exactamente cuatro, y se regeneraron con 7 llamadas a la API:
los textos de VEH_0451 y los resúmenes del 29-09-2025, el 10-11-2025 y el 02-03-2026. Los datos (`cuts`,
`vehicles`, `deviations`) son idénticos a los de la v0.

**`demo-bundle-gru:v1`** está fijado en `configs/demo.yaml`, y la v0 queda en wandb. Tiene 302 respuestas en la
caché y 14 intentos rechazados en total, 4 de ellos por las reglas nuevas. El contenedor se probó en local: baja
la v1, pide la clave y sirve las tres páginas.

## Decisiones

1. **El ensamble 42 / 1 / 2, no 3 / 4 / 5.** Es lo que decidió F10: elegir por 0,471 contra 0,460 sería
   seleccionar sobre ruido.
2. **Umbral exacto, no la grilla.** Es la cuenta del reporte v2 y del sweep. Con la grilla de `train.py`, la
   detección al 5% cae a 14,6%. El replay tiene que reproducir su repetición del reporte, o el builder falla.
3. **Etiqueta dura y replay del 03-03-2025 al 24-09-2026:** de la primera revisión de dev al fin de la
   extracción de los fallados. La telemetría se corta el 14-09 para todos, pero hay eventos registrados hasta
   el 21-09.
4. **La demo abre en la semana del 29-09-2025**, la primera con al menos 3 alertas nuevas. Son 4, las 4 de autos
   que fallaron: dos avisos (VEH_0563, VEH_0566) y dos contactos del concesionario (VEH_0451, VEH_0583). La
   otra semana con 3 es la del 22-12-2025.
5. **El puntaje del gráfico es un rango medio**, no una probabilidad, y la ficha lo dice
   (`model.score_note`).

## Un primer intento que no quedó: la GRU `last` de F3 sobre la entrega 1

Antes de pasar a v2 se armó la demo con la GRU `last` de F3 y la etiqueta V
(`configs/exp_gru_seq_last_kmw.yaml` y `scripts/join_window_columns.py`, que verifica que el panel secuencial y el de
K2 tienen las mismas filas). Reproduce la corrida registrada al último decimal (PR-AUC 0,1681). Pero **al 5% no le
gana al azar**: 7,4% ± 2,1 (2 / 4 / 4 de 45) contra 7,7% del nulo, y el replay tenía 6 alertas en toda la
temporada. Se deja el código y la corrida (`experiments/decision-gru-f3/`) como registro.

## Límites

- **El porqué es una comparación con la flota, no lo que el modelo usó.** Nada acá dice por qué la GRU alertó.
  Un hábito nombrado puede no tener nada que ver con la alerta.
- **Parte de lo que detecta es composición.** La celda mercado × motor sola llega a 18% al 5% (62% al 20%, donde
  empata). La GRU lee mercado y motor; lo que suma es el orden dentro de la celda (AUC 0,66). Qué parte de la
  tasa de la celda es física y qué parte es cómo Ford armó las listas no se puede separar con estos datos
  ([f10-sweep-gru.md](f10-sweep-gru.md)).
- **Sabe sobre todo *qué auto*, poco *cuándo*:** (a′) = +0,006 en el ensamble (+0,004 ± 0,004 por repetición;
  [f9-remedicion-completa-v2.md](f9-remedicion-completa-v2.md)). La anticipación es la distancia de la primera
  alerta a la falla, no una predicción de fecha.
- **La muestra está enriquecida en fallas** (135 de 426). Los conteos de la bandeja no se trasladan a una flota
  real.
- **El replay es una repetición;** los números oficiales son el promedio de las 3.
- **La referencia de los sanos es descriptiva:** mediana de dev del mercado en todo el período, no una
  referencia de producción. PER tiene 20 autos sanos de referencia y ningún fallado.
- **La efectividad de los avisos no está medida.**

## Código nuevo

- `src/eval/fleet_profile.py` (el porqué descriptivo) y `scripts/demo_app/wording.py` (textos que dependen del
  bundle, sin Streamlit).
- `scripts/build_demo_bundle.py`: la corrida sale de `model`, con umbral exacto y números del reporte v2;
  `src/eval/dashboard_data.py::load_run` genérico (ensambles, corridas sin ventana, un panel aparte para el
  perfil). `load_k2` queda para el dashboard de F4.
- `src/agents/bundle.py`: el bundle trae `deviations.parquet` y declara modelo y tipo de porqué; uno viejo falla
  con un mensaje claro.
- `scripts/join_window_columns.py` y los YAML de la GRU de F3 (el primer intento).
- `scripts/check_setup.py`: 6 chequeos de la demo con la GRU (217 en total).
