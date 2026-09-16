# `ENG_3` no aparece entre los fallados

**Fecha:** 2026-09-15 · **Fase:** F1 · **Reproduce:** `python scripts/eda_raw.py` (sección 2)

Proporción dentro de cada cohorte:

| Engine | fallados | sanos | n fallados |
|---|---|---|---|
| ENG_1 | 0,196 | 0,174 | 74 |
| ENG_2 | 0,804 | 0,465 | 304 |
| **ENG_3** | **0,000** | **0,361** | **0** |

268 vehículos sanos son `ENG_3`. Vehículos con evento y motor `ENG_3`: ninguno.

## Por qué no es una buena noticia

Un motor con cero fallas en 268 unidades no es un motor perfecto: es una muestra
armada así. El enunciado no dice que `ENG_3` esté excluido del fenómeno; dice que
estos son los vehículos que nos dieron. Si `static_Engine` entra al modelo, la
regla más rentable que puede aprender es `Engine == ENG_3 ⇒ sano`, y acierta el
100% de las veces **dentro de este dataset** y ninguna afuera.

El síntoma va a ser un PR-AUC alto que no se sostiene en el pitch cuando alguien
pregunte qué pasa con un `ENG_3` real que empieza a degradarse. Es el caso de la
regla 6: un número sospechosamente bueno que hay que auditar antes de festejarlo.

`ModelSeries` y `SalesCountry_cd` no tienen este problema — están presentes en las
dos cohortes con proporciones parecidas (país casi idéntico: 0,336/0,271 el más
desbalanceado).

## Qué se hace

`Engine` queda afuera del set base (`configs/data/panel_v1.yaml`, clave
`features.static_excluded`). Si alguien quiere medir cuánto aporta de verdad, se
corre una ablación con otro YAML y se compara — que es justamente el flujo que la
regla 7 habilita.
