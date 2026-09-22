# Survival stacking en días post-venta, con la ventana del registro: aprende el *cuándo* y pierde el *qué auto*

**Fecha:** 2026-09-22 · **Fase:** F5 (§3.3 del doc de candidatos) · **Rama:** `exp/ss-post-venta`
**Alcance:** solo dev, R = 3, con las mismas 2.029 filas, 171 vehículos y folds que el finalista
(build `data/rebuild-0921`, `splits_r3.json`). **Test sin tocar.**
**Preregistro:** [f5-preregistro-ss-post-venta.md](f5-preregistro-ss-post-venta.md) (958e520),
con una enmienda antes de entrenar. La implementación y los chequeos se commitearon antes de
entrenar (edf4a6d).

## En una línea

Reescribir el objetivo de survival stacking en días desde la venta, con cada fila en riesgo solo
dentro de la ventana del registro, **no le gana al finalista**.
- Con la etiqueta corregida, que es la que decide, **empata en detección** (9,6% ± 2,8 contra
  11,9% ± 2,8) y **pierde en lift por vehículo** (1,38× contra 1,66×).
- **El finalista sigue siendo survival stacking sobre el panel v1.**
- Lo que sí cambia:
  - **la (b) de calendario baja de +0,049 a +0,016**: el atajo de calendario del finalista era
    exposición a la ventana del registro;
  - **con la etiqueta corregida, el finalista detecta 11,9%, no 15,7%.**

## Qué se hizo

- **El panel** (`scripts/build_window_survival_panel.py`, `src/data/window_risk.py`). Son las
  filas del panel v1 tal cual, con `feat_cut_dss` (días desde la venta en el corte) en vez de
  `feat_cut_odo`. Cada fila lleva además su tramo en riesgo en días desde `t0`:
  - `t0` es la fecha en que el odómetro llega a `c + G`, así que el gap sigue en km;
  - la fila entra en `max(t0, 01-09-2025)` y sale en `min(evento, 11-03-2026)`. Un sano sale en
    el fin de la ventana, no en su último viaje.
  - `t0` sale de la inversa exacta de la proyección que ubica el evento en km
    (`src/data/anchor.py::project_odometer_to_dates`): en ninguna fila con evento cae después
    del evento.
- **El modelo** (`src/models/window_stacking.py`, `window_survival_stacking`): exponencial por
  tramos.
  - Bins de 5 d, con la exposición exacta de cada bin como offset de una regresión de Poisson
    (LightGBM).
  - El booster es el del finalista.
  - Entrena hasta 60 d y el score es `1 − S(30 d)`. Es el espejo de 500 / 3.000 / 6.000 km a la
    mediana de las filas: el horizonte de 3.000 km dura 31 d.
- **La evaluación** (`scripts/eval_window_label.py`), con dos etiquetas:
  - **dura (D):** las 2.029 filas;
  - **corregida (V):** las 1.435 filas con el horizonte entero dentro de la ventana, 45 fallados
    y 95 sanos. La regla vale igual para positivas y negativas. **Decide V.**

  Referencia: `f3-survival-stacking-r3-rebuild`, sin reentrenar.

## Los números

**Etiqueta corregida (V), la que decide** (1.435 filas, 182 positivas, 45 fallados, 95 sanos):

| | detección @ ≤ 50 FA/1.000 | detectados | lift por vehículo (`mean`) | anticipación [km] |
|---|---|---|---|---|
| **candidato** | **9,6% ± 2,8** | 4 / 3 / 6 | **1,376 ± 0,049** | 7.742 ± 3.516 |
| referencia | 11,9% ± 2,8 | 4 / 7 / 5 | 1,655 ± 0,142 | 9.354 ± 2.448 |
| piso `cut_odo` | 4,4% | 2 | 1,156 | 8.308 |
| piso `feat_cut_dss` | 0,0% | 0 | 1,210 | — |

- **Detección:** diferencia −0,022 contra un desvío combinado de 0,039, así que **empata**. Por
  repetición (pareado): 0,0 / −0,089 / +0,022.
- **Lift:** diferencia −0,279 contra un desvío combinado de 0,150, así que **pierde**. Por
  repetición: −0,19 / −0,53 / −0,12.
- **Veredicto: pierde.** Les gana a los dos pisos en las dos métricas y pasa (a0), pero con
  perder en una alcanza.

**Etiqueta dura (D), informativa** (2.029 filas, 254 positivas, 53 fallados, 118 sanos):

| | detección | detectados | lift por vehículo | anticipación [km] | PR-AUC por fila | Brier |
|---|---|---|---|---|---|---|
| **candidato** | 9,4% ± 1,5 | 4 / 5 / 6 | 1,328 ± 0,021 | 9.759 ± 3.990 | 0,179 | 0,116 |
| referencia | 15,7% ± 0,9 | 8 / 9 / 8 | 1,616 ± 0,061 | 8.330 ± 22 | 0,172 | 0,112 |
| piso `cut_odo` | 7,5% | 4 | 1,020 | 7.385 | | |

Con D pierde en las dos. El Brier no se compara: el score del candidato es P(evento en 30 d desde
`c + G`), no en 3.000 km.

**Auditorías** (`audit_model.py`, R = 3, con D):

| | valor | referencia |
|---|---|---|
| (a0) features permutadas | 0,1215 contra una tasa base de 0,1252: **pasa** | pasa |
| (a′) aporte del *cuándo* | **+0,067 ± 0,012** | +0,016 ± 0,003 |
| ROC dentro del vehículo | 0,726 (39 vehículos) | |
| **(b) `aux_` de calendario** | **+0,016 de ROC** | +0,049 |

- **Importancias:** `feat_cut_dss` va primera (11%), seguida de la velocidad, la duración de
  viaje y el borde del bin.
- **Piso posicional** (`audit_positional_floor.py`): le gana en lift y en detección, y pierde en
  (a′) y en PR-AUC entre fallados. Es el mismo patrón que el finalista.
- **Nulo de tamaño de bolsa** (`audit_mil_bagsize.py`): lift `mean` de 1,34× contra un nulo de
  1,03 ± 0,10, p = 0,005.
- **Correlación de rango con la referencia:** 0,60–0,66 por fila y 0,71–0,74 por vehículo.

**Entrenamiento:** por fold, 14.195–14.908 filas apiladas y 331–348 eventos, en 69.000–72.000
días-auto. Entre 230 y 304 filas entran tarde a la ventana.

## Por qué ordena autos peor

Es un diagnóstico agregado **después** del veredicto. **No estaba preregistrado y no decide
nada** (`eval_window_label.py --diagnose`).

| | ρ(score, días desde la venta) por fila · por vehículo | AUC por vehículo, todas las filas | AUC por vehículo, filas ≤ 120 d |
|---|---|---|---|
| candidato, D | **+0,51** · +0,56 | 0,624 | 0,654 |
| referencia, D | +0,23 · +0,29 | 0,689 | 0,679 |
| candidato, V | **+0,51** · +0,60 | 0,633 | 0,672 |
| referencia, V | +0,25 · +0,39 | 0,681 | 0,707 |

- **El score del candidato es, sobre todo, la posición en días.** Es el *cuándo* que aprendió:
  el hazard sube fuerte con los días desde la venta (f3-reloj-y-ventana-del-evento.md §4). Al
  promediar por vehículo, esa posición es ruido: un sano seguido mucho tiempo acumula filas en
  días altos.
- **Con autos en posiciones parecidas** (filas ≤ 120 d), el candidato mejora y la brecha se
  achica, pero no se cierra: 0,672 contra 0,707 con V.
- **No es el uso.** Un auto que anda poco tarda más días en recorrer 3.000 km, y el finalista
  podría ganar por eso. Pero su score por vehículo casi no se correlaciona con los km/día
  (ρ −0,16), y −km/día solo da un lift de 1,22× con D y 1,15× con V, muy por debajo de los dos
  modelos.

## Qué significa

1. **La ventana del registro explica el atajo de calendario del finalista.** Con el conjunto en
   riesgo de la ventana, darle al modelo `ProductionDay`, la temperatura ambiente y el marcador
   de regeneración mueve el ROC +0,016, contra +0,049 del finalista. Es lo que el preregistro
   esperaba (§5): el calendario ayudaba porque medía la exposición al registro.
   - Para el finalista es un límite con nombre: su (b) es exposición, no física.
2. **La corrección de la etiqueta no le saca al finalista su orden de autos.** Con V, su lift se
   mantiene (1,62× → 1,66×). **Lo que sí baja es su detección: 15,7% → 11,9%** (4 / 7 / 5 de 45).
   - Parte de las alertas que contaba como aciertos caían en filas que el registro no cubría.
   - Para el pitch, el número honesto es el de V, con el desvío al lado.
3. **El reloj en días no mejora el *qué auto*; mejora el *cuándo*.**
   - (a′) se cuadruplica y la anticipación deja de ser estable (±3.500–4.000 km, contra ±22 del
     finalista).
   - Con la decisión por vehículo y el promedio de filas, el *cuándo* compite con el *qué auto*
     en vez de sumarle.
4. **Se cierran las ramas condicionales del doc F5:**
   - §3.5 (otro learner dentro de 3.3) necesitaba que 3.3 ganara;
   - B-cal necesitaba que la (b) siguiera marcando, y bajó a +0,016;
   - §3.7 decía "solo si 3.3 abre el camino".

## Desviaciones

- **La enmienda antes de entrenar** (preregistro): el offset lleva también la tasa global del
  train. Con `init_score`, LightGBM no arranca desde el promedio, y con la tasa de aprendizaje
  del finalista no alcanzaba a bajar el intercepto. Se encontró con datos simulados.
- **El reloj 5 / 30 / 60 d** reemplazó a 9 / 54 / 108 d antes del preregistro, al ver en el
  embudo que el horizonte de 3.000 km dura 31 d en la fila mediana. Ninguna etiqueta estaba a la
  vista.
- **El diagnóstico** es posterior al veredicto. El umbral de 120 d se eligió al escribirlo.
- **wandb:** todo corrió con `WANDB_MODE=disabled`. La corrida está en `results/`, y la
  evaluación D/V en `experiments/f5-ss-post-venta-r3/window_eval.json`, que no se versiona.

## Qué queda abierto (ideas, no corridas)

- **Cuál de los dos cambios costó el *qué auto*.** Esta corrida cambió a la vez el reloj (km →
  días) y el conjunto en riesgo (toda la telemetría → la ventana).
  - El diagnóstico apunta al reloj: el score quedó dominado por la posición en días.
  - La ablación que lo separa es la misma corrida en km con la ventana. Sería un candidato nuevo,
    y el presupuesto está agotado.
- **Una decisión por vehículo que no mezcle posiciones**, como el score en una edad fija o el
  promedio de las filas de los primeros 120 d. También sería un candidato nuevo, y la idea viene
  de un diagnóstico posterior.
- **El horizonte por fila con el km/día pasado** (dejado fuera en el preregistro): el diagnóstico
  de uso no lo hace prometedor.
- **La capa de decisión (§3.6 del doc F5)** sigue abierta y no consume presupuesto.

## Cómo se reproduce

```bash
export FORD_DATA_DIR=$PWD/data/rebuild-0921 WANDB_MODE=disabled
# el cache de viajes del panel de hitos, copiado al build de la referencia (no se versiona):
mkdir -p data/rebuild-0921/interim && cp data/interim/landmark_trips.parquet data/rebuild-0921/interim/
python scripts/build_window_survival_panel.py --config configs/data/panel_survival_ps.yaml --counts-only  # Fase 1
python scripts/build_window_survival_panel.py --config configs/data/panel_survival_ps.yaml
python scripts/train.py --config configs/exp_ss_post_venta_r3.yaml                  # ~1 min
python scripts/audit_model.py --config configs/exp_ss_post_venta_r3.yaml            # (a0), (a′), (b), (c)
python scripts/eval_window_label.py --config configs/exp_ss_post_venta_r3.yaml --diagnose   # D, V, veredicto
python scripts/audit_positional_floor.py f5-ss-post-venta-r3
python scripts/audit_mil_bagsize.py f5-ss-post-venta-r3
python scripts/check_setup.py                                                        # 158 chequeos
```

La referencia (`f3-survival-stacking-r3-rebuild`) tiene que estar entrenada en el mismo checkout
(ver [f3-ensamble-e1-e2.md](f3-ensamble-e1-e2.md)).

**Código nuevo:**
- `src/data/window_risk.py` y `project_odometer_to_dates` en `src/data/anchor.py`;
- el target `window_survival` y el modelo `window_survival_stacking`
  (`src/models/window_stacking.py`);
- `scripts/build_window_survival_panel.py` y `scripts/eval_window_label.py`;
- 10 chequeos en `check_setup.py`. Diez mutaciones los hacen fallar: sin entrada tardía,
  evaluable sin el inicio de la ventana, sano que sale en su horizonte, meseta del odómetro,
  bins enteros, sin tasa base, sin censura a 60 d, un solo piso, NaN arriba y evento sin tramo.
