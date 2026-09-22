# Preregistro: survival stacking en el reloj del evento, con el conjunto en riesgo de la ventana (F5 §3.3)

**Fecha:** 2026-09-22 · **Fase:** F5 · **Rama:** `exp/ss-post-venta`.

**Se escribió** después de los conteos de la Fase 1 (5bc5d63) y antes de todo lo demás:
- antes de implementar el modelo y el target;
- antes de entrenar;
- antes de mirar ningún score contra ninguna etiqueta.

El commit que agrega este archivo es la marca de tiempo. Cualquier cambio posterior va en un
commit propio, anterior a la corrida que afecta, y dice por qué.

**De dónde sale:** [../f5-modelos-candidatos-bibliografia.md](../f5-modelos-candidatos-bibliografia.md)
§3.3 y §5 (lugar 2). Lo que lo habilita está en
[f3-reloj-y-ventana-del-evento.md](f3-reloj-y-ventana-del-evento.md):
- el evento se ordena por días desde la venta, no por km;
- el registro solo anota eventos entre el 01-09-2025 y el 11-03-2026;
- el 28% de los horizontes sanos del panel v1 cae en parte fuera de esa ventana, y hoy el
  finalista los aprende como "sano verificado".

**Presupuesto.** No es otro learner: mismas filas, mismas features, mismos folds y el mismo
LightGBM. Cambia cómo se arma el objetivo. Es la corrida que CLAUDE.md admite, *"hacer
comparable una fila que ya existe"*, y **ocupa el lugar de L1** del preregistro del 21-09
([f3-preregistro-landmark-ensamble.md](f3-preregistro-landmark-ensamble.md)). L1 nunca corrió, y
proponía hitos en km: la Fase 1 mostró después que ese era el reloj equivocado. Igual va con
lista cerrada: una idea que aparezca en el camino va a "Qué queda abierto" de la ficha.

## 0 · Lo que ya se vio antes de escribir esto

**Dev:** nada nuevo contra la etiqueta. Lo que se usa del finalista ya está medido
([f3-piso-posicional.md](f3-piso-posicional.md), [f3-ensamble-e1-e2.md](f3-ensamble-e1-e2.md)).

**Test:** nada. El panel nuevo trae las filas de test (el panel v1 las trae), pero los conteos
son solo de dev.

**Fase 1, solo conteos del objetivo** (ninguna feature contra la etiqueta):

```bash
FORD_DATA_DIR=$PWD/data/rebuild-0921 python scripts/build_window_survival_panel.py \
  --config configs/data/panel_survival_ps.yaml --counts-only
```

- **Las filas son las del finalista.** 2.029 filas de dev, 171 vehículos, 53 fallados, 254
  positivas. La misma huella de vehículos, así que `splits_r3.json` sirve sin tocarlo.
- **El reloj reproduce lo que ya se sabía:**
  - el odómetro del evento recalculado con los viajes coincide con el del panel (0,000 km);
  - el 72,0% de las filas sanas tiene el horizonte entero dentro de la ventana y el 8,8% lo
    empieza antes; H9 había medido 71,8% y 8,7%.
- **Origen del riesgo** (la fecha en que el odómetro llega a `c + G`): conocido en las 2.029
  filas.
  - Empieza antes de la ventana en 93 filas sanas y 242 de fallados. Esas entran tarde.
  - Empieza después de la ventana en 14 filas sanas. Esas no están en riesgo.
- **En riesgo:** 1.048 de 1.062 filas sanas y las 967 de fallados. Ninguna fila con evento
  queda fuera de riesgo, y en ninguna el origen cae después del evento.
- **Apiladas**, con bins de 5 d hasta 60 d: 18.116 filas, con 423 eventos dentro del horizonte
  de entrenamiento.
- **El horizonte de 3.000 km en días**, en las filas de dev: mediana 31 d (p25 20, p75 47).
  - De ahí sale el reloj del §2. El km/día mediano por vehículo (57) daría 54 d, pero hay un
    corte cada 500 km: las filas pesan más a los autos que andan más.
  - El YAML del panel decía 9 / 54 / 108 d antes de ver este conteo. Se cambió en el mismo
    commit de la Fase 1, sin ninguna etiqueta a la vista.
- **Etiqueta corregida** (§4):
  - 1.435 filas evaluables, 182 positivas;
  - 45 fallados y 95 sanos con al menos una fila;
  - se pierden 72 filas positivas.
  - Por fold de validación quedan 7–11 fallados evaluables.
- **`feat_cut_dss` sin fecha de venta:** 5 filas de 2 vehículos.

## 1 · Filas, features y folds: los del finalista

- **Panel:** `panel_survival_ps.parquet` = el panel v1 del build de la referencia
  (`data/rebuild-0921`), más las columnas del reloj
  (`scripts/build_window_survival_panel.py`, `src/data/window_risk.py`).
- **Features:** las 53 `feat_` del panel v1, `static_SalesCountry_cd` y `feat_cut_dss`: los
  días desde la venta en la fecha del corte.
  - `feat_cut_dss` **reemplaza** a `feat_cut_odo` como posición de la fila. Es del pasado
    (`cut_date` es el último viaje de la ventana).
  - Preprocesamiento `standard`, igual que la referencia.
- **Folds:** `splits_r3.json`, R = 3, sobre dev (`select_dev()`).
- **Referencia:** `f3-survival-stacking-r3-rebuild`: 15,7% ± 0,9 de detección y 1,616× de lift
  por vehículo con la etiqueta dura.

## 2 · El objetivo y el modelo

**Target `window_survival`.** Por fila de train, en días desde `t0`, la fecha en que el
odómetro llega a `c + G`:
- **Entrada** `max(0, inicio de la ventana − t0)`.
- **Salida** `min(evento, fin de la ventana) − t0`.
  - Un sano sale en el fin de la ventana, aunque su telemetría termine antes.
  - Un sano sin exposición en la ventana no genera filas apiladas.
- **Evento:** el del vehículo, si cae en el tramo.
- **El gap sigue en km** (regla 1): el riesgo arranca en `c + G`.
- `t0` sale de la inversa exacta de la proyección que ubica el evento en km (`src/data/anchor.py`).

**Modelo `window_survival_stacking`: exponencial por tramos (PEM).**
- Los días desde `t0` se parten en bins de **5 d**. Cada fila se apila en los bins en que
  estuvo en riesgo, con la **exposición exacta** del bin: la ventana y la entrada tardía cortan
  bins por la mitad, y el offset los cuenta sin tirarlos ni contarlos enteros.
- LightGBM con `objective: poisson` y `init_score = log(exposición)` (Bender et al. 2020).
  - La covariable del hazard base es el borde del bin en días desde `t0`.
  - Hiperparámetros del booster: los de la referencia (`_booster_defaults` de
    `src/models/survival_stacking.py`), con `random_state` 42.
- **Se entrena hasta 60 d** desde `t0`; después se censura.
- **Score:** `1 − S(30 d | x) = 1 − exp(−Σ λ_k · 5)` sobre los 6 bins de `[0, 30]`.
  - 30 d fijos para todas las filas. No se traducen 3.000 km a días con los km futuros del
    vehículo, ni con los pasados.
  - 5 / 30 / 60 d es el espejo de 500 / 3.000 / 6.000 km a la mediana de las filas (§0).

## 3 · Evaluación: las dos etiquetas, y cuál decide

La detección y el lift por vehículo no usan la etiqueta por fila: usan `event_observed` y qué
filas entran. Por eso "cambiar de etiqueta" es cambiar qué filas se evalúan.

- **Etiqueta dura (D):** las 2.029 filas de dev, igual que toda la tabla de `results/`.
  **Informativa:** mantiene la comparabilidad.
- **Etiqueta corregida (V): decide.** Solo las filas cuyo horizonte `[t0, t_H]` cae **entero**
  dentro de la ventana (`aux_eval_in_window`). `t_H` es la fecha en que el odómetro llega a
  `c + G + H`.
  - La regla vale igual para positivas y negativas. Si valiera solo para las negativas, en los
    bordes de la ventana quedarían solo positivas, y eso sería un atajo de calendario.
  - Una negativa cuenta como verificada solo si el registro cubría su horizonte.
  - Un vehículo sin filas evaluables sale de la cuenta.

**Por qué decide V:** es la única de las dos en la que todo negativo está verificado. D premia
aprender los negativos que el registro no cubría, que es justo lo que esta corrida corrige.

**Qué se calcula**, para el candidato y la referencia sobre las mismas filas y los mismos folds
(se verifica), con `evaluate_predictions` de `train.py`:
1. **detección @ ≤ 50 falsas alarmas cada 1.000 sanos**, media ± desvío entre las 3
   repeticiones, con alerta sostenida `k = 2`;
2. **lift por vehículo con `mean`**, media ± desvío entre repeticiones.

## 4 · Qué decide

Son las reglas del preregistro del 21-09, aplicadas con la etiqueta V
(`scripts/ensemble_rank.py::compare`):
- **"Le gana a X en M"** = la diferencia de medias supera el desvío combinado `√(σ² + σ_X²)`.
- **Contra la referencia:**
  - **gana** si le gana en una de las dos y no pierde en la otra;
  - **pierde** si pierde en cualquiera;
  - si no, **empata**. Un empate con más desvío de detección o de anticipación cuenta como peor.
- **Pisos, también con V:** tiene que ganarle, en las dos métricas, a:
  - el piso posicional, `score = cut_odo`;
  - el mismo piso en el reloj nuevo, `score = feat_cut_dss`. Sin fecha de venta, la fila va
    por debajo de todas.
- **(a0):** las features permutadas entre todas las filas tienen que caer a la tasa base
  (|Δ PR-AUC| < 0,02, con D). Si no caen, la corrida no se lee.

**Resultado:**
- **Se adopta** (reemplaza a survival stacking sobre el panel v1 como finalista) si gana contra
  la referencia, les gana a los dos pisos y pasa (a0).
- **Si empata o pierde**, el finalista sigue siendo el de hoy. Queda escrito cuánto de su
  ventaja con D se sostiene con V.

## 5 · Informativas (no deciden, lista cerrada)

- Las dos métricas con la etiqueta D, y el veredicto que daría D.
- PR-AUC por fila, con D y con V.
- (a′), el aporte del *cuándo*, con D.
- Anticipación mediana en km y su desvío entre repeticiones, con D y con V.
- **(b)**, las `aux_` de calendario como `feat_` (`scripts/audit_model.py`, con D, R = 3). La
  referencia marca +0,049 de ROC.
  - **Hipótesis:** si el calendario ayudaba porque medía la exposición al registro, definir el
    conjunto en riesgo con la ventana le quita ese papel, y la (b) baja de 0,02.
  - Si sigue marcando, B-cal (§3.3 del doc F5) queda elegible para un preregistro propio. No
    corre en esta ronda.
- Brier con D y con V. El score es P(evento en 30 d desde `c + G`), no en 3.000 km: no se
  compara con el de la referencia.
- Correlación de rango con la referencia, por fila y por vehículo.
- Piso posicional con D (`audit_positional_floor.py`) y nulo de tamaño de bolsa
  (`audit_mil_bagsize.py`).
- Importancias (c).

## 6 · Orden y paradas

1. Implementación y chequeos (`check_setup.py`), commiteados antes de entrenar.
2. Construir `panel_survival_ps.parquet` y entrenar el candidato (`train.py`, R = 3).
3. Auditorías: (a0) y (b) con `audit_model.py`. **Si (a0) falla, se para.**
4. Evaluación D y V contra la referencia y los pisos (`scripts/eval_window_label.py`), y el
   veredicto.
5. Ficha, `decisiones.md` y `results/`.

## Lo que queda fuera a propósito

- **Otro horizonte o ancho de bin** (54 / 9 d, 60 / 10 d, …): es un barrido.
- **Un horizonte por fila con el km/día pasado.** El doc F5 lo deja como alternativa "si hace
  falta". No hace falta: las dos métricas que deciden no dependen del horizonte de la etiqueta.
- **B-cal**, la parcial estratificada por mes: condicional a la (b), con preregistro propio.
- **Otro learner** (§3.5): solo si esta corrida gana.
- **Reentrenar la referencia con las filas de V**, o darle `feat_cut_dss`: cambiaría la fila
  que se quiere comparar.
- **`feat_cut_odo` junto con `feat_cut_dss`:** el doc dice "reemplaza".
- **Mover el fin de la ventana un día** para incluir el último día entero: es la convención del
  panel de hitos, y se mantiene.
- **El test:** no se toca hasta que haya un modelo elegido.

## Preguntas para Ford que pueden invalidar esto

- **¿Cuál es la ventana real de extracción del registro?** Si es más ancha que la de dev, los
  sanos tienen más exposición de la que se cuenta acá, y la etiqueta V descarta de más.
- **El error de SQL de la query de eventos** (rama `feat/f3-features-regeneracion`): si cambian
  las fechas, esto se rehace después de la Fase 1 del cure model.
