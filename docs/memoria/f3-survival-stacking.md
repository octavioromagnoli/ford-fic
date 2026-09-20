# Survival stacking: el panel ya era un dataset de supervivencia

**Fecha:** 2026-09-20 · **Fase:** F3 · **Alcance:** dev del panel v1 (2.029 filas, 171
vehículos, 254 positivas, tasa base 0,1252), mismos folds congelados. · **Reproduce:**

```bash
python scripts/build_dataset.py --config configs/data/panel_v1.yaml          # ~3 min (emite aux_km_observed_after_cut)
python scripts/build_survival_panel.py --config configs/data/panel_survival.yaml   # ~2 s
python scripts/train.py --config configs/exp_survival_stacking.yaml          # ~25 s
python scripts/audit_model.py --config configs/exp_survival_stacking.yaml    # ~2 min (las tres auditorías)

uv pip install -r requirements-gpboost.txt                                   # opcional
python scripts/train.py --config configs/exp_gpboost_survival.yaml           # ~3 min
```

> **Ojo con el número contra el que se compara.** El `panel.parquet` con el que se midió
> el 0,165 de `f3-lgbm-panel-v1` es del 19-09 y es **anterior** al arreglo del umbral de
> regeneraciones (commit `1eab4a1`, 20-09): siete columnas `feat_*regen*` cambiaron. El
> control honesto es `f3-lgbm-panel-v1-regen15`, el mismo YAML sobre el panel
> reconstruido: **0,1612**. Todo lo de acá se compara contra ese.

## La idea

El panel es una fila por `(vehículo, corte)`, o sea un dataset persona-período. Craig,
Zhong & Tibshirani ([arXiv:2107.13480](https://arxiv.org/abs/2107.13480)) muestran que
apilar un dataset así con etiqueta binaria y correr un clasificador estándar es
aproximadamente equivalente a un Cox. Nosotros teníamos el dataset y estábamos tirando
tres cosas:

| lo que se tiraba | qué cambia |
|---|---|
| `cut_odo` no entra como covariable (`cv.py` selecciona por prefijo) | el modelo no puede aprender el hazard base sobre el odómetro |
| los 144 sanos entran como ceros limpios | son censurados: "llegó hasta X km sin fallar", no "no falla" |
| los ~5 cortes de un vehículo se tratan como independientes | es la trampa que [f3-modelos-candidatos.md](../f3-modelos-candidatos.md) §2.1 anota para el Cox por fila |

A eso se suma lo que la etiqueta binaria tampoco puede decir: **dónde** dentro del
horizonte cae el evento (un corte a 600 km y otro a 2.900 km valen lo mismo) y **qué
pasa después de H** (un vehículo que falla a 5.000 km es un 0 indistinguible de un sano).

## Cómo quedó

**La ventana `[0, max_horizon_km)` se parte en bins de `bin_km`.** Cada fila del panel se
expande a una fila apilada por bin **en el que estuvo en riesgo**, con el borde izquierdo
del bin como covariable extra (`surv_bin_start_km`: el hazard base, que el modelo aprende
en vez de que se lo impongamos). El objetivo de la fila apilada es el hazard discreto: 1
solo en el bin donde cae el evento. Con W/G/H/Δ del panel v1 y `max_horizon_km: 6000`,
las ~1.623 filas de train de cada fold se vuelven **~17.250 apiladas en 12 bins**, con un
hazard de 0,021.

Predicción: los hazards de los 6 bins que cubren `[0, H]` se componen en
`S(H|x) = Π (1 − h_k)` y el score es **`1 − S(H|x)`**, el riesgo acumulado a H. Es la
probabilidad de exactamente el evento que define `label`, así que el PR-AUC sigue
midiendo lo mismo que el resto de la tabla, con los mismos folds.

**El gap de blanking (regla 1) se aplica corriendo el origen de la duración a `c + G`**,
no en el modelo: `src/training/targets.py` entrega `duration = time_to_event_km − G` para
los que fallan y `aux_km_observed_after_cut − G` para los censurados. De ahí sale la
identidad que hace comparable todo lo demás, y que `check_setup.py` verifica fila a fila:

> `label = 1` ⇔ la fila está en riesgo, tiene evento y su duración cae en `[0, H]`.

No es una reparametrización libre: es la misma definición escrita sobre el eje correcto.
Una fila con duración negativa (el evento cayó dentro del gap) queda marcada
`at_risk = False` y no genera ninguna fila apilada — usarla sería mirar los km que la
regla 1 prohíbe. El panel v1 no tiene ninguna; el dummy sí (20), y el chequeo las usa.

### Lo que hubo que agregar

1. **`aux_km_observed_after_cut`** (`src/data/panel.py`): `last_odo − c`, los km que el
   vehículo se observó después del corte. Sale gratis de lo que `vehicle_cuts` ya tiene.
   Es `aux_` porque es información posterior al corte y no entra a ninguna ventana. Con
   `censored_policy: drop` toda fila sana tiene ≥ G + H, así que **dentro del horizonte
   no hay censura**: recién aparece si el objetivo mira más allá de H, que es justo lo que
   `max_horizon_km` habilita. El panel regenerado tiene **las mismas 2.507 filas en el
   mismo orden, la misma huella (`753cfc698ba41099`) y los mismos folds**; la única
   columna nueva es esa.
2. **`feat_cut_odo`** (`scripts/build_survival_panel.py`, `panel_survival.parquet`):
   `cut_odo` copiado con prefijo de feature. Estaba en el panel desde F2 pero como
   identificador, y la regla de prefijos dice que lo que entra a un modelo se llama
   `feat_`/`static_`; meterlo por la puerta de atrás hubiera sido saltearse el contrato.
   Mismas filas ⇒ misma huella ⇒ el `splits.json` congelado sirve tal cual.
3. **Un hook `target:` genérico en `cv.py`**: nombra un modo de
   `src/training/targets.py`, que devuelve el `y` de train del fold. `cv.py` no sabe qué
   modos existen ni qué devuelven. **Lo que se evalúa no cambia nunca**: la etiqueta
   dura, con los mismos folds. `exp/ordinal-horizon` agrega su función al mismo archivo
   sin tocar nada de esto.
4. **C-index out-of-fold y bootstrap por vehículo** (`src/eval/metrics.py`). El C-index
   se recalcula desde las predicciones y no desde el target, así que el de un LightGBM
   sobre `label` y el de un modelo de supervivencia miden lo mismo.

## Resultados (dev, CV agrupada, R = 1)

| modelo | PR-AUC (lift) | IC por vehículo | ROC | Brier | C-index | detección · anticipación @ ≤50 FA/1000 |
|---|---|---|---|---|---|---|
| tasa base | 0,121 (0,97×) | — | 0,486 | 0,110 | — | — |
| **LightGBM control** (panel reconstruido) | **0,1612** (1,29×) | — | **0,5862** | 0,1727 | 0,5616 | 0,075 · 12.322 km |
| **survival stacking** | **0,1613** (1,29×) | [0,128 – 0,213] | 0,5776 | **0,1144** | **0,5697** | **0,151** · 8.333 km |
| survival + efecto aleatorio (GPBoost) | 0,1431 (1,14×) | [0,107 – 0,194] | 0,5524 | 0,1120 | 0,5069 | 0,038 · 19.058 km |

PR-AUC por fold del apilado: 0,154 · 0,221 · 0,164 · 0,207 · 0,184.

**Empata en PR-AUC, y llega distinto.** +0,0001 sobre el control no es nada, y decir otra
cosa sería mentir. Lo que sí cambia:

- **El Brier baja de 0,173 a 0,114** (la tasa base es 0,110). El hazard se entrena sin
  `class_weight`, así que `1 − S(H|x)` es una probabilidad de verdad: media 0,133 contra
  una tasa real de 0,125. El LightGBM con `class_weight="balanced"` predice 0,314 de
  media. Para el pitch, la diferencia entre "este auto tiene 30% de riesgo" y un score
  ordinal es la diferencia entre poder priorizar un taller y no.
- **Detección del 15,1% contra 7,5%** con el mismo presupuesto de falsas alarmas, a
  cambio de 4.000 km menos de anticipación (8.333 vs 12.322 km).
- **C-index 0,570 contra 0,562.**

### El único cuyo "cuándo" suma

La auditoría (a′) —reemplazar el score de cada fila por el promedio de su vehículo, sin
reentrenar— es la cuenta exacta de cuánto del PR-AUC sale de ordenar *cortes* y no
*vehículos*:

| modelo | PR-AUC | con el score colapsado por vehículo | aporte del *cuándo* | ROC dentro del vehículo |
|---|---|---|---|---|
| LightGBM control | 0,1612 | 0,1679 | **−0,0067** | 0,5950 |
| survival stacking | 0,1613 | 0,1460 | **+0,0152** | 0,5956 |

Los dos ordenan igual de bien *dentro* de un vehículo (ROC ≈ 0,595). La diferencia es que
el apilado está calibrado: sus scores viven en la misma escala de probabilidad entre
vehículos, así que la variación intra-vehículo se compone bien con la inter-vehículo en
el ranking global. Los del LightGBM no, y su variación intra-vehículo **resta** PR-AUC.
Ese es el argumento real de esta rama, y no el empate en PR-AUC.

### Ablaciones (mismas filas, mismos folds)

| panel | `max_horizon_km` | PR-AUC | ROC | Brier | C-index | aporte del *cuándo* | detección · anticipación |
|---|---|---|---|---|---|---|---|
| con `feat_cut_odo` | 3.000 (= H) | 0,1618 | 0,5991 | 0,1189 | 0,5591 | +0,0184 | 0,151 · 7.006 km |
| con `feat_cut_odo` | **6.000** | 0,1613 | 0,5776 | 0,1144 | 0,5697 | +0,0152 | 0,151 · 8.333 km |
| con `feat_cut_odo` | 9.000 | 0,1637 | 0,5837 | 0,1131 | 0,5782 | +0,0193 | 0,170 · 9.865 km |
| sin `feat_cut_odo` | 3.000 | 0,1606 | 0,5874 | 0,1184 | 0,5583 | +0,0033 | 0,189 · 4.384 km |
| sin `feat_cut_odo` | 6.000 | 0,1581 | 0,5804 | 0,1141 | 0,5761 | +0,0080 | 0,189 · 6.881 km |
| sin `feat_cut_odo` | 9.000 | 0,1538 | 0,5757 | 0,1136 | 0,5787 | +0,0022 | 0,170 · 9.908 km |

Los seis PR-AUC entran en 0,154–0,164: **ninguna de las dos perillas mueve la aguja**, y
el intervalo por vehículo de la corrida elegida ([0,128 – 0,213]) los contiene a todos.
Se deja `max_horizon_km: 6000` porque es lo que estaba preregistrado; elegir 9.000 a
posteriori por 0,002 es exactamente el ruido que §0.2 pide no minar.

Lo que sí se ve con claridad es que **`feat_cut_odo` es de dónde sale el "cuándo"**: sin
él, el aporte intra-vehículo cae de +0,015 a +0,008. Tiene sentido — dentro de un
vehículo, el odómetro del corte *es* el cuándo. Y no es un atajo: el emparejado de sanos
por (bin de odómetro × mes) le da a positivos y sanos la misma distribución de odómetro
por construcción, y la auditoría (a0) lo confirma.

## El efecto aleatorio no ayuda, y se sabe por qué

GPBoost queda 0,018 de PR-AUC por debajo y su C-index es 0,507 — ordena como una moneda.
El motivo está medido, no supuesto: **la varianza del intercept por vehículo se estima
en 9,4–10,3** (sd ≈ 3,1 en la escala logit) en los cinco folds.

Es cuasi-separación. En cada train hay ~137 vehículos y el hazard es 0,021; unos 100 de
esos vehículos no tienen **ni un** bin con hazard positivo, así que su intercept se va a
−∞ y la varianza explota. El efecto aleatorio no regulariza: se come la señal, y los
árboles se quedan con menos. Y en validación no compensa, porque el vehículo nunca estuvo
en train (split agrupado, regla 2) y su efecto es la media a priori.

Fijando σ² en vez de estimarla (`gp_params.optim_params`) se recupera parte:

| σ² | PR-AUC | ROC | Brier | C-index |
|---|---|---|---|---|
| 0,1 | 0,1543 | 0,5478 | 0,1094 | 0,5165 |
| 0,5 | 0,1512 | 0,5790 | 0,1105 | 0,5131 |
| 1,0 | 0,1543 | 0,5861 | 0,1118 | 0,5164 |
| 2,0 | 0,1511 | 0,5833 | 0,1143 | 0,5163 |
| estimada (9,5) | 0,1431 | 0,5524 | 0,1120 | 0,5069 |

Ninguna llega a los 0,1613 del apilado sin efecto aleatorio. **La conclusión es que con
splits agrupados y 53 eventos, un efecto aleatorio por vehículo no tiene de dónde pagar
lo que cuesta.** La corrida queda anotada igual: es un negativo con explicación, que es
lo que evita que alguien lo reintente dentro de dos semanas.

Detalle de implementación que cambia el número reportado (pero no el ranking): la
predicción se hace en la escala **latente** y el link se aplica a mano, en vez de pedir
`response_mean`. Los dos ordenan idéntico —el link es monótono, el PR-AUC no se mueve—,
pero `response_mean` integra sobre esa varianza de 9,5 y devuelve el riesgo *promedio de
la flota*: 0,51 de media, con un Brier de 0,26. Con el efecto aleatorio en su media, el
número es el riesgo del **vehículo mediano** con esas features, que es lo que el resto
del pipeline trata como probabilidad.

## Auditorías

`python scripts/audit_model.py --config configs/exp_survival_stacking.yaml`

| chequeo | survival stacking | LightGBM control | lectura |
|---|---|---|---|
| **(a0)** features permutadas entre **todas** las filas | 0,1227 (base 0,1252) | 0,1251 | cae a la tasa base: no hay leakage |
| **(a)** features permutadas **dentro** del vehículo | 0,2004 | 0,1884 | **sube** en los dos modelos, ver abajo |
| **(a′)** score colapsado al promedio del vehículo | 0,1460 | 0,1679 | el *cuándo* suma +0,015 acá y resta −0,007 en el control |
| **(b)** +`aux_air_temp_avg`, `aux_regen_marker_per_1000km`, `aux_static_ProductionDay` como `feat_` | ROC 0,5776 → 0,6006 (+0,023) | 0,5862 → 0,6033 (+0,017) | el mismo salto en los dos: es del panel, no del modelo |

### (a) no es un null en este panel, y hay que decirlo

§0.4 pide permutar `label` dentro de cada vehículo y esperar que el PR-AUC caiga a la
tasa base. Acá **sube**, con los dos modelos y con tres semillas
(survival 0,200 / 0,208 / 0,184; LightGBM 0,188 / 0,211 / 0,174, contra 0,161 de
referencia). No es un bug:

- La permutación deja intacto **qué** vehículos fallan, que es de donde sale casi todo el
  PR-AUC de este panel (mirar (a′): colapsar al promedio del vehículo todavía da 0,146).
- Y de paso le saca a cada vehículo el ruido de qué ventana le tocó, así que lo que se
  entrena es un **ordenador de vehículos más limpio**, con 5× de filas efectivas.

O sea que (a) compara la referencia contra un modelo que resuelve un problema más fácil.
Dos cosas que se cambiaron para poder leerla:

1. Se permutan las **features**, no la etiqueta. Reordenar los pares `(X, y)` de un
   vehículo es la misma reasignación por cualquiera de los dos lados, pero así la
   etiqueta contra la que se mide no se mueve y los dos PR-AUC son el mismo número sobre
   el mismo conjunto. (Permutando `label`, además, el modo de supervivencia ni se
   enteraría: su objetivo sale de `time_to_event_km`.)
2. Se agregaron **(a0)**, que es el null de verdad, y **(a′)**, que es la cuenta del
   aporte del *cuándo* con todo lo demás fijo.

**Esto aplica a todo modelo de F3, no solo a este**, y conviene arrastrarlo al resto: el
`f3-cnn-lstm-tutora` reporta 0,143 contra 0,153 con la permutación dentro del vehículo y
lo lee como "la distancia es lo que aporta el cuándo". Con lo de acá, esa lectura no se
sostiene sin (a′).

### (b): el calendario sigue medio abierto

+0,023 de ROC con las tres `aux_` de calendario, y +0,017 en el LightGBM. Es el mismo
orden en los dos modelos, así que es una propiedad del panel: el emparejado por (odómetro
× mes) cierra la mayor parte del atajo pero no lo borra, que es exactamente lo que
[f2-eda-revision-y-features.md](f2-eda-revision-y-features.md) §2.1 anticipa. No hay nada
que corregir en este modelo; sí conviene que ninguna corrida futura promueva esas
columnas.

### (c): las importancias son las que el mecanismo predice

| # | feature | peso | ¿la física la espera? |
|---|---|---|---|
| 1 | `feat_cut_odo` | 7,4% | es el hazard base sobre el odómetro, para eso se agregó |
| 2 | `feat_engine_temp_avg_median` | 6,9% | sí — familia A, régimen térmico |
| 3 | `feat_speed_kmh_mean` | 6,0% | sí — uso urbano |
| 4 | `feat_trip_duration_median_min` | 6,0% | sí — trayectos cortos |
| 5 | `feat_hours_between_trips_median` | 5,1% | sí — enfriamiento entre viajes |
| 6 | `feat_trips_per_day` | 4,0% | sí — familia C |
| 10 | `surv_bin_start_km` | 2,8% | el hazard base dentro del horizonte |

Abajo siguen `feat_dpf_end_max`, `feat_trips_below_regime_temp_frac`,
`feat_idle_per_1000km` y `feat_idle_frac_trend`. **Nada aparece arriba que no tenga por
qué anticipar una degradación de combustión**, y en particular ninguna `aux_` ni ninguna
estática fuera de `SalesCountry_cd`. El régimen térmico y el uso urbano en los cinco
primeros es la hipótesis del plan §4 tal cual.

Regla 6 no se dispara: no hay salto que auditar. El apilado empata al control.

## Qué queda

- **El techo de este panel es el problema, no el modelo.** Cinco familias de modelos
  (LightGBM, CNN-LSTM, TimesFM, deuda térmica, apilado de supervivencia) caen todas entre
  0,15 y 0,17 de PR-AUC con la misma tasa base. Lo que (a) y (a′) muestran es por qué:
  casi todo lo que hay es señal **entre** vehículos, y el *cuándo* vale ±0,015. Cualquier
  cosa que se intente después debería atacar eso de frente.
- **Lo que vende esta corrida no es el PR-AUC**: es la probabilidad calibrada (Brier
  0,114) y el doble de detección al mismo presupuesto de falsas alarmas. Para el punto de
  operación del pitch, eso pesa más que 0,0001 de PR-AUC.
- **Falta CV repetida.** Todo esto es R = 1. Para declarar cualquiera de estas
  diferencias hay que correr `n_repeats: 3`, como ya existe en `splits_r3.json`.
- **La combinación obvia que no se probó**: el apilado sobre el panel de la deuda térmica
  o con los `feat_tfm_*`. Son un YAML cada una y entran en el presupuesto de
  comparaciones de §0.2 si se preregistran.
- **`scripts/audit_model.py` es genérico**: corre las cinco auditorías sobre el YAML de
  cualquier experimento. Conviene pasarle los modelos ya anotados y actualizar sus fichas.
