# Preregistro: más autos detectados con anticipación (F6) · lista cerrada de tres candidatos

**Fecha:** 2026-09-22 · **Fase:** F6 · **Rama:** `exp/deteccion-vehiculo`.

**Se escribió** después del embudo de K2 (solo conteos del objetivo) y antes de todo lo demás:
- antes de implementar el target y el modelo;
- antes de entrenar;
- antes de mirar ningún score nuevo contra ninguna etiqueta.

El commit que agrega este archivo es la marca de tiempo. Cualquier cambio posterior va en un
commit propio, anterior a la corrida que afecta, y dice por qué.

**Por qué existe.** El pedido del 22-09 es un modelo que supere al finalista y que detecte la
mayor cantidad posible de autos que van a fallar, **con antelación**. CLAUDE.md dice que el
presupuesto de comparaciones está agotado. Esta lista gasta uno nuevo, **explícito y cerrado**:
- tres candidatos;
- una regla más estricta que la de las corridas anteriores (§5);
- una idea que aparezca en el camino va a "Qué queda abierto" de la ficha, no a esta lista.

## 0 · Lo que ya se vio antes de escribir esto

**Dev, contra la etiqueta:** nada nuevo. Lo que se usa del finalista ya está medido
([f3-mejor-modelo-a-la-fecha.md](f3-mejor-modelo-a-la-fecha.md),
[f5-ss-post-venta.md](f5-ss-post-venta.md)).

**Test:** nada.

**Sin etiqueta, sobre el score del finalista** (`f3-survival-stacking-r3-rebuild`, repetición 0):
- el 59% de la varianza del score está **dentro** del vehículo (ICC 0,41);
- ρ mediana de Spearman del score con el odómetro, dentro del vehículo: +0,22 (116 vehículos con
  ≥ 4 cortes).

**Embudo de K2, solo conteos del objetivo** (ninguna feature contra la etiqueta):

```bash
FORD_DATA_DIR=$PWD/data/rebuild-0921 python scripts/build_km_window_panel.py \
  --config configs/data/panel_survival_kmw.yaml --counts-only
```

- **Las filas son las del finalista:** 2.029 de dev, 171 vehículos, 53 fallados. Tienen la misma
  huella, así que `splits_r3.json` sirve sin tocarlo. El odómetro del evento recalculado coincide
  con el del panel (0,000 km).
- **Odómetro al inicio de la ventana:**
  - interpolado en 181 vehículos de dev;
  - en 108 la telemetría arranca después del 01-09-2025, así que la fila entra en `c + G`;
  - 1 vehículo termina su telemetría antes.
- **Odómetro al fin de la ventana:** interpolado en 283; en 7 la telemetría termina antes, y se
  censura en el último odómetro.
- **En riesgo:**
  - las 967 filas de fallados. Ninguna queda afuera;
  - 1.048 de 1.062 sanas. Hay 14 sin exposición en la ventana.
  - **Entran tarde** 242 filas de fallados y 93 sanas.
- **Lo que la referencia entrena distinto:**
  - **514 filas de fallados cuyo evento cae más allá de 6.000 km.** La referencia las apila como
    sobrevivientes en sus 12 bins.
  - **Las 1.062 filas sanas siguen observadas después del 11-03-2026:** 1,20 millones de km de
    "supervivencia" que el registro no cubría.
- **Salida del tramo en riesgo** (km desde `c + G`): mediana 7.161, p90 19.836, **máximo 52.210**.
  El horizonte de entrenamiento de K2 es el menor múltiplo de 5.000 km que no censura ningún
  tramo de dev: **55.000 km**. Con 40.000 quedaban afuera 25 eventos.
- **Apiladas con bins de 500 km:** 35.545 filas con horizonte de 40.000 km (942 eventos). Con
  55.000 son unas 36.000, con los 967 eventos.
- **Por fold de train:** 773–774 filas con evento en riesgo, de 42–43 fallados.

**La etiqueta corregida (V)** es la de F5 §3.3, sin cambios: 1.435 filas evaluables, 182
positivas, 45 fallados y 95 sanos con alguna fila ([f5-preregistro-ss-post-venta.md](f5-preregistro-ss-post-venta.md) §0).

## 1 · Qué se compara

**Referencia:** `f3-survival-stacking-r3-rebuild` (survival stacking, panel v1 + `feat_cut_odo`,
R = 3). Sus números, que ya están medidos:

| etiqueta | detección @ ≤ 50 FA/1.000 | detectados | lift por vehículo | anticipación mediana |
|---|---|---|---|---|
| **V (decide)** | **11,9% ± 2,8** | 4 / 7 / 5 de 45 | 1,655 ± 0,142 | 9.354 ± 2.448 km |
| D (informativa) | 15,7% ± 0,9 | 8 / 9 / 8 de 53 | 1,616 ± 0,061 | 8.330 ± 22 km |

**Filas, features y folds:** los de la referencia en los tres candidatos:
- las 2.029 filas de dev;
- las 53 `feat_`, `static_SalesCountry_cd` y `feat_cut_odo`;
- preprocesamiento `standard` y `splits_r3.json`.

Nada del booster se toca: son los hiperparámetros por defecto de `survival_stacking`, con
`random_state: 42`. Nada se re-tunea.

## 2 · Los tres candidatos (lista cerrada)

### K1 · Media acumulada causal del score del finalista

**Qué es.** Sin reentrenar. Para cada repetición `r` y cada vehículo, las filas de dev se ordenan
por `cut_odo`. El score de la fila `i` pasa a ser el promedio de `score_r` en las filas `1..i` del
mismo vehículo, **la propia incluida**. Solo usa filas anteriores, y todas son out-of-fold del
mismo fold, porque el split es por vehículo. `score` es el promedio de las tres repeticiones, como
en toda corrida.
- Se calcula sobre **todas** las filas de dev y después se enmascara V. Una fila no evaluable
  igual es pasado del vehículo: la evaluabilidad es de la etiqueta, no de las features.
- Es la versión desplegable de la agregación `mean` por vehículo, la única que le gana a su nulo
  de tamaño de bolsa ([f3-mil-agregacion-vehiculo.md](f3-mil-agregacion-vehiculo.md)). Un score
  al que colapsar por vehículo le suma, pero con el colapso hacia atrás.

**Por qué podría detectar más.** La decisión es por vehículo, con una alerta sostenida de 2
cortes. El 59% de la varianza del score del finalista está dentro del vehículo. Las falsas
alarmas de un sano salen de sus picos, y promediar hacia atrás los achica.

**Por qué podría perder.**
- La media arrastra los primeros cortes. En el finalista esos son más bajos (ρ +0,22 con el
  odómetro), así que la alerta puede llegar más tarde, o no llegar.
- Además, los fallados tienen el doble de cortes que los sanos, y con más cortes hay más
  oportunidades de que un pico dispare la alerta. Suavizar le quita al score esa ventaja
  artificial de las bolsas largas. El §5 la mide con un nulo.

### K2 · Survival stacking en km con horizonte completo y el conjunto en riesgo de la ventana

**Qué es.** Es el mismo `DiscreteSurvivalStacker`, con las mismas features y el mismo score
(`1 − S(3.000 km | x)`), los mismos bins de 500 km y el mismo booster. Cambia el tramo en riesgo de
cada fila, en km desde `c + G` (`scripts/build_km_window_panel.py`):
- **entra** en `max(0, odómetro al 01-09-2025 − (c + G))`. Se apilan solo los bins desde el que
  contiene la entrada; los anteriores no son supervivencia verificada;
- un **fallado sale en su evento**. Los 967 caen dentro de la ventana;
- un **sano sale** en `min(último odómetro, odómetro al 11-03-2026) − (c + G)`;
- se entrena hasta **55.000 km**, así que ningún tramo de dev se censura administrativamente.

Es el target `window_km_survival`. `feat_cut_odo` sigue siendo la posición de la fila, así que el
reloj sigue en km. Es la ablación que F5 §3.3 dejó abierta ("la misma corrida en km con la
ventana"), más el horizonte completo.

**Por qué podría detectar más.**
- **No hay trayectoria previa al evento.** La señal es un rasgo del auto presente desde el primer
  mes ([f3-reloj-y-ventana-del-evento.md](f3-reloj-y-ventana-del-evento.md) §5–6).
  - La referencia apila como sobrevivientes 514 filas de autos que después fallan, porque su
    evento cae más allá de 6.000 km. Así le enseña al modelo que ese rasgo, lejos del evento, es
    de un auto sano.
  - Con el horizonte completo esas filas son eventos más tarde. Es la misma lógica por la que
    survival stacking le ganó a la etiqueta binaria (`max_horizon_km > H`), llevada hasta el
    final. También es lo que sugirió la auditoría (a): quitarle al auto el ruido de qué corte le
    tocó mejora el orden de autos.
- **La supervivencia sana después del 11-03-2026 no está verificada** (1,2 millones de km). Sacarla
  alinea el entrenamiento con V, que es la etiqueta que decide. En F5 §3.3 eso bajó la (b) de
  calendario de +0,049 a +0,016.

**Por qué podría perder.**
- Los bins 0–5, que arman el score, se aprenden con menos filas: las que entran tarde no están en
  riesgo ahí.
- Con el horizonte largo, el bin (km desde `c + G`) puede interactuar con las features.
- En F5 §3.3 el conjunto en riesgo de la ventana vino junto con otro reloj, y perdió en lift.

### K3 · K2 + la media acumulada causal de K1

Es la misma post-transformación de K1, aplicada a las predicciones de K2. Existe porque K1 y K2
atacan fuentes distintas: el ruido dentro del vehículo y el entrenamiento del *qué auto*.

## 3 · Cómo se mide

Es la cuenta de F5 §3.3, sin cambios: `scripts/eval_window_label.py`, con la etiqueta corregida V
(decide) y la dura D (informativa), contra la referencia sin reentrenarla. Por repetición salen:
- la detección a ≤ 50 falsas alarmas cada 1.000 sanos, con `k_consecutive: 2` y 50 umbrales;
- el lift por vehículo con `mean`;
- la anticipación mediana.

Antes de medir, cada candidato tiene que coincidir con la referencia por `(vehicle_id, cut_odo)`,
en `label` y `event_observed`, y en `fold_r{r}`.

**Pisos** (score = esa columna, sin ajustar nada): `cut_odo` y `aux_cut_dss`, los días desde la
venta en el corte. Son los mismos dos de F5 §3.3.

## 4 · Auditorías y nulos (antes de leer el veredicto)

- **(a0):** features permutadas entre todas las filas (semilla 0), reentrenado. El PR-AUC por
  fila, con `score`, tiene que quedar a menos de 0,02 de la tasa base.
  - K2: `audit_model.py`, que da también (a′) y (b).
  - K1 y K3: se reentrena el miembro (la referencia o K2) con las features permutadas, se aplica
    la media acumulada y se mide.
- **Nulo de tamaño de bolsa para la detección** (CLAUDE.md, regla 6: la alerta sostenida depende
  del largo del historial).
  - Por repetición, `score_r` se permuta entre todas las filas de dev (200 permutaciones,
    semilla 0), se aplica la misma post-transformación del candidato y se mide la detección con V.
  - El **exceso** es la detección menos la media del nulo.
  - K2 y la referencia no transforman nada, y la detección solo depende del orden, así que
    comparten nulo.
- **Informativas, no deciden:**
  - (a′) y la (b) de calendario;
  - el Brier (en K1 y K3 el score es un promedio de riesgos, no el riesgo a H);
  - la correlación de rango con la referencia;
  - el C-index;
  - el intervalo bootstrap pareado por vehículo (1.000 remuestreos estratificados por
    fallado/sano) de la diferencia de detección con V, sobre `score`.

## 5 · Qué decide (se aplica en este orden)

Con la etiqueta **V**, un candidato **supera al finalista** si cumple las seis:

1. **Gana en detección:** la diferencia de medias supera el desvío combinado
   (`ensemble_rank.py::compare`). El pedido es detectar más autos, así que ganar solo en lift
   no alcanza.
2. **No pierde en lift por vehículo:** la diferencia negativa no supera el desvío combinado.
3. **Guarda de multiplicidad** (son tres candidatos): la diferencia de detección pareada es
   ≥ 0 en las tres repeticiones.
4. **Guarda de anticipación:** la anticipación mediana no cae más que el desvío combinado
   respecto de la referencia. Detectar más, pero tarde, no es lo pedido.
5. **Les gana a los dos pisos** (`cut_odo`, `aux_cut_dss`) en detección y en lift, y **pasa
   (a0)**.
6. **Su exceso sobre el nulo de tamaño de bolsa es ≥ el de la referencia.** Así se descarta que
   gane por explotar mejor las bolsas largas de los fallados.

**Si más de uno cumple**, se adopta el de mayor detección media con V. Si la diferencia entre
ellos no supera su desvío combinado, se adopta el más simple: K1, después K2, después K3.

**Si ninguno cumple**, el finalista no cambia y se escribe por qué, con los números. D se reporta
al lado en toda tabla. No decide, pero si un candidato gana con V y pierde con D, eso va escrito
como límite.

## 6 · Lo que no cambia

- **El test no se toca.** Si un candidato se adopta, sigue siendo "el mejor en dev". El test se
  mide una sola vez, con el modelo elegido, y eso lo decide el equipo.
- **wandb:** `WANDB_MODE=disabled`, como en F5. Las corridas van a `results/`.
- **Implementación antes de entrenar**, en un commit propio:
  - `window_km_survival` (target) y la entrada tardía en `DiscreteSurvivalStacker`;
  - `scripts/smooth_scores.py` (K1/K3 y su (a0));
  - `scripts/audit_detection_null.py`;
  - chequeos en `check_setup.py` que fallen con mutaciones de la entrada tardía, de la censura en
    la ventana y de la causalidad de la media.
