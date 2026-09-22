# Decidir por vehículo (MIL): el lift sube, pero casi todo lo que sube es el tamaño de la bolsa

**Fecha:** 2026-09-20 · **Fase:** F3 · **Alcance:** dev del panel v1 (2.029 filas, 171
vehículos con cortes, 53 con evento). **Ni el modelo ni los folds cambian**: es solo el
eje de evaluación. · **Reproduce:**

```bash
FORD_DATA_DIR="$PWD/data/v364" WANDB_MODE=offline \
  python scripts/train.py --config configs/exp_lgbm_panel_v1_mil.yaml
python scripts/audit_mil_bagsize.py f3-lgbm-panel-v1-mil
python scripts/check_setup.py     # 42 chequeos, 9 del eje de vehículo
```

(`FORD_DATA_DIR` apunta al directorio donde está el panel de los 364; `WANDB_MODE`
evita la red sin editar el config compartido, CLAUDE.md regla 9.)

## Qué es

El panel tiene una fila por `(vehículo, corte)` y el PR-AUC de selección se mide ahí,
pero **la decisión de negocio es por vehículo**: Ford marca autos, no cortes. Es el
marco de *Multiple-Instance Learning* (Ilse, Tomczak & Welling, ICML 2018): cada
vehículo es una **bolsa**, cada corte una **instancia**, y la etiqueta de la bolsa es
`event_observed` —el vehículo tuvo evento— y no `label`, que es "el evento cae en el
horizonte de *este* corte". En dev son ~53 bolsas positivas de ~11,9 cortes de media.

`src/eval/metrics.py` agrega la bolsa con cuatro pooling: `max`, `mean`, `topk` (media
de los k mayores, k = 3) y `noisy_or` (1 − Π(1 − pᵢ)). **No se entrena nada**: es una
capa de decisión sobre los scores out-of-fold que ya existen. Attention-MIL entrenado
end-to-end queda descartado a propósito: con 53 bolsas sobreajusta, y este repo ya sabe
qué pasa cuando un modelo con más parámetros que eventos parece mejorar.

Como el split es agrupado por vehículo (regla 2), todas las filas de una bolsa salieron
del mismo fold de validación: el score agregado sigue siendo estrictamente out-of-fold.
`vehicle_scores()` lo verifica y falla si algún vehículo tiene cortes en más de un fold.

## El PR-AUC por vehículo no se compara contra el de filas

Dos tasas base distintas, y el PR-AUC arranca en la tasa base:

| eje | unidades | positivas | tasa base |
|---|---|---|---|
| fila (corte) | 2.029 | 254 | **0,125** |
| vehículo (bolsa) | 171 | 53 | **0,310** |

Un PR-AUC por vehículo de 0,47 **no es** "tres veces mejor" que el 0,165 por fila: es
1,51× la tasa base contra 1,32×. **Lo único comparable entre los dos ejes es el lift**,
y es lo que hay que reportar — en el log de `train.py`, en `metrics.json`
(`vehicle.comparabilidad`) y acá.

Y la tasa base de 0,310 tampoco es la de la flota: los sanos del panel están emparejados
por odómetro × mes, no muestreados de la población. El lift sobrevive al cambio de
prevalencia; el PR-AUC pelado, no.

## La trampa: la bolsa de los que fallan es el doble de larga

| | vehículos | cortes por bolsa (media) | mediana | máx |
|---|---|---|---|---|
| con evento | 53 | **18,2** | 17 | 68 |
| sanos | 118 | **9,0** | 5 | 36 |

No es física, es cómo se construye el panel: los vehículos con evento conservan **todos**
sus cortes, mientras que a los sanos el emparejado les deja solo los que hacen falta para
llenar cada celda de odómetro × mes (1.241 filas sanas de 7.308, repartidas entre 144
vehículos; `panel_meta.json`, `sampling`). El tamaño de la bolsa es un artefacto del
muestreo, y sin embargo **solo, como score, da PR-AUC 0,555 — lift 1,79×, ROC 0,672**.

Eso contamina toda agregación que crezca con el tamaño: `noisy_or` por construcción
(corr. de rango con `n_cuts` = 0,85), `max` y `topk` porque el máximo de más muestras es
más alto (0,43 y 0,46). `mean` es la única insensible al tamaño (0,04).

## La auditoría (regla 6): cada agregación contra su propio nulo

`scripts/audit_mil_bagsize.py` baraja los scores entre todas las filas —destruye
cualquier relación score↔etiqueta, **conserva exactamente el tamaño de cada bolsa**— y
vuelve a medir, 200 veces. Lo que sale es cuánto lift saca cada agregación del puro
tamaño. Una agregación solo aporta si supera a *su* nulo; el 1,0 de la tasa base no es
el piso acá.

| agregación | PR-AUC | lift | lift del nulo | p | ROC-AUC | ROC nulo | detección | det. nula |
|---|---|---|---|---|---|---|---|---|
| `max` | 0,466 | 1,50× | 1,37 ± 0,15 | 0,185 | 0,706 | 0,617 | 9% | 10% |
| **`mean`** | 0,469 | **1,51×** | **1,02 ± 0,10** | **0,000** | 0,666 | 0,507 | **15%** | 3% |
| `topk` (k=3) | 0,489 | 1,58× | 1,55 ± 0,13 | 0,415 | 0,724 | 0,654 | 11% | 14% |
| `noisy_or` | 0,567 | 1,83× | 1,78 ± 0,05 | 0,150 | 0,729 | 0,670 | 19% | 21% |

Detección = vehículos con evento marcados aceptando como mucho 50 falsas alarmas cada
1.000 sanos (5%), el mismo presupuesto del punto de operación por cortes.

Leído en orden de arriba abajo, el ranking "natural" (`noisy_or` 1,83× es el mejor
número de todo el proyecto) se da vuelta entero:

- **`noisy_or`, `topk` y `max` no le ganan a su nulo.** El 19% de detección de
  `noisy_or` es *peor* que el 21% que consigue un score permutado: lo que está marcando
  son los vehículos con más historia en el panel, que resultan ser los que fallan porque
  así se armó el emparejado. Contra la flota real, donde la bolsa de un auto sano no es
  más corta por construcción, ese 19% no existe.
- **`mean` es la única con señal propia**: 1,51× contra un nulo de 1,02×, ROC 0,666
  contra 0,507, 15% de detección contra 3%. Es también la única cuyo nulo está en 1,0,
  que es lo que uno espera de una métrica que no mide el tamaño de la bolsa.

Con CV repetida (3 juegos de folds, `splits_r3.json`) el orden se mantiene y la
dispersión es chica: `mean` 1,48 ± 0,04, `max` 1,52 ± 0,05, `topk` 1,62 ± 0,05,
`noisy_or` 1,82 ± 0,03. La conclusión no depende del sorteo de folds.

## Qué usar

**`mean`**, y reportando lift. Contra el mismo modelo medido por fila:

| | PR-AUC | tasa base | lift | detección (≤ 5% de sanos) |
|---|---|---|---|---|
| por fila (corte) | 0,165 | 0,125 | 1,32× | 9,4% * |
| **por vehículo, `mean`** | 0,469 | 0,310 | **1,51×** | **15%** |

\* el 9,4% del punto de operación por cortes ya es una tasa por vehículo
(`first_alert_lead_times()` agrupa por vehículo y exige `k_consecutive` cortes
seguidos); lo que faltaba era la **métrica de ranking** en ese eje, no la detección.

O sea: agregar por vehículo mejora, pero mejora **poco** —1,32× → 1,51× de lift, 9%
→ 15% de detección—, no el salto que sugería la diferencia entre "19% de las filas" y
"la mitad de los autos". La señal del panel v1 sigue siendo débil (§0 de
`docs/f3-modelos-candidatos.md`); el eje de vehículo la presenta mejor, no la crea.

Las otras tres quedan implementadas y se siguen midiendo en cada corrida, porque el
número comparable es el conjunto: una agregación nueva que le gane a `mean` **y** a su
propio nulo sería una mejora de verdad. Mientras el panel tenga bolsas asimétricas,
cualquier número de `noisy_or` hay que leerlo contra `audit_mil_bagsize.py`.

**Lo que no hay que hacer:** "arreglar" `noisy_or` normalizando por el tamaño de la
bolsa. El tamaño desbalanceado es del muestreo del panel, no de los vehículos; la
normalización que corresponde es construir el panel con bolsas comparables (misma grilla
de cortes para sanos y fallados, emparejando a nivel vehículo y no a nivel fila). Eso es
Track A y cambia el panel: queda anotado, no se hizo acá.

## Detalle de implementación

- `src/eval/metrics.py`: `vehicle_scores()` (bolsa → score, con las cuatro
  agregaciones) y `vehicle_metrics()` (las métricas de `classification_metrics()` sobre
  las bolsas). `noisy_or` **se niega a correr** si los scores no están en [0, 1]: un
  modelo que devuelve un margen o un riesgo relativo hay que calibrarlo aparte y a
  propósito, no adentro de una métrica de selección.
- `scripts/train.py`: el bloque `eval.vehicle_aggregation` del YAML elige las
  agregaciones (las cuatro si la clave no está, `[]` o `null` para apagar el eje), y
  `eval.vehicle_topk` la k. Las métricas van a `metrics.json` bajo `vehicle` y a wandb
  con prefijo `vehicle/`. Con CV repetida se agrega **cada repetición por separado** y
  se promedia, por el mismo motivo que las métricas por fila: agregar el score promedio
  de R pasadas es un ensamble encubierto.
- `configs/exp_lgbm_panel_v1_mil.yaml`: copia exacta de `exp_lgbm_panel_v1.yaml` salvo
  el nombre y ese bloque. Mismo modelo, mismo panel, mismos folds — el PR-AUC por fila
  de las dos corridas es idéntico (0,16533), que es la prueba de que lo único que se
  movió es el eje de evaluación.
- `scripts/check_setup.py`: 9 chequeos nuevos, incluidos los dos que documentan las
  trampas (que `noisy_or` satura con el tamaño de la bolsa aun con un oráculo, y que una
  bolsa repartida entre dos folds hace fallar la agregación).

**Referencia:** M. Ilse, J. M. Tomczak, M. Welling, *Attention-based Deep Multiple
Instance Learning*, ICML 2018 — de ahí salen el marco bolsa/instancia y los pooling.
El attention pooling entrenado del paper es justamente lo que **no** se implementó.
