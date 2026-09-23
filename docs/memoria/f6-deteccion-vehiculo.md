# Survival stacking con la ventana del registro, en km y con horizonte completo: detecta más autos y es el nuevo finalista

**Fecha:** 2026-09-22 · **Fase:** F6 · **Rama:** `exp/deteccion-vehiculo`
**Alcance:** solo dev, R = 3, con las mismas 2.029 filas, 171 vehículos, features y folds que el
finalista anterior (build `data/rebuild-0921`, `splits_r3.json`). **Test sin tocar.**
**Preregistro:** [f6-preregistro-deteccion-vehiculo.md](f6-preregistro-deteccion-vehiculo.md)
(b12056c). La implementación y los chequeos se commitearon antes de entrenar (de55335).

## En una línea

**K2 le gana al finalista con la etiqueta que decide y pasa las seis condiciones del preregistro.**
K2 es survival stacking con el conjunto en riesgo de la ventana del registro, en km y con horizonte
completo.
- Con 5% de falsas alarmas detecta **17,0% ± 3,8** de los autos que van a fallar (7 / 10 / 6 de
  45), contra **11,9% ± 2,8** (4 / 7 / 5).
- El lift por vehículo es el mismo (1,66×), la calibración también (Brier 0,112), y alerta con una
  mediana de **~7.400 km de anticipación** (unos 4 meses a 57 km/día).
- **Es el nuevo finalista.**

La mejora es modesta: **+2,3 autos de 45 en promedio**.
- El bootstrap pareado por vehículo no la separa del cero: P(diferencia > 0) = 0,74.
- Con la etiqueta dura, empata en detección.

La media acumulada causal del score (K1, y K3 sobre K2) no suma: detecta menos o igual, con mucho
más desvío.

## Qué se probó (lista cerrada)

| | qué es | reentrena |
|---|---|---|
| **K1** | el score del finalista, promediado hacia atrás sobre los cortes del mismo auto | no |
| **K2** | el mismo survival stacking, pero cada fila solo cuenta como "en riesgo" dentro de la ventana del registro (01-09-2025 → 11-03-2026), en km, y se entrena hasta 55.000 km en vez de 6.000 | sí |
| **K3** | K2 + la media acumulada de K1 | no |

**La idea de K2.** El finalista anterior le enseñaba al modelo dos cosas falsas:
1. **514 cortes de autos que después fallan eran "sobrevivientes"**, porque el evento caía más
   allá de 6.000 km. Como no hay trayectoria previa al evento (el rasgo del auto está desde el
   primer mes), eso le decía al modelo que ese rasgo, lejos del evento, era de un auto sano.
2. **Los 1.062 cortes sanos tenían 1,2 millones de km de "supervivencia" después del 11-03-2026.**
   El registro ya no anotaba eventos en ese período.

K2 no toca ni las features ni el booster ni el score (`1 − S(3.000 km | x)`); solo corrige quién
está en riesgo y hasta dónde:
- entrada tardía en el odómetro del 01-09-2025;
- un sano sale en `min(último odómetro, odómetro del 11-03-2026)`;
- horizonte de entrenamiento hasta 55.000 km, que es el menor múltiplo de 5.000 km que no censura
  ningún tramo de dev. Así, todo corte de un fallado es un evento.

## Los números

**Etiqueta corregida (V), la que decide.** Son 1.435 filas con el horizonte dentro de la ventana,
182 positivas, 45 fallados y 95 sanos:

| | detección @ ≤ 50 FA/1.000 | detectados | lift por vehículo | anticipación mediana [km] |
|---|---|---|---|---|
| **K2** | **17,0% ± 3,8** | 7 / 10 / 6 | 1,658 ± 0,074 | 7.438 ± 2.005 |
| K1 | 14,8% ± 6,9 | 5 / 11 / 4 | 1,660 ± 0,137 | 7.317 ± 1.409 |
| K3 | 11,1% ± 7,9 | 0 / 8 / 7 | 1,687 ± 0,092 | — (una repetición no detecta) |
| referencia (finalista anterior) | 11,9% ± 2,8 | 4 / 7 / 5 | 1,655 ± 0,142 | 9.354 ± 2.448 |
| piso `cut_odo` | 4,4% | 2 | 1,156 | 8.308 |
| piso `aux_cut_dss` | 0,0% | 0 | 1,210 | — |

**Etiqueta dura (D), informativa.** Son 2.029 filas, 254 positivas y 53 fallados:

| | detección | detectados | lift por vehículo | anticipación [km] | PR-AUC por fila | Brier |
|---|---|---|---|---|---|---|
| **K2** | 15,1% ± 3,1 | 8 / 10 / 6 | **1,671 ± 0,034** | 8.811 ± 370 | 0,1755 | 0,1124 |
| K1 | 10,7% ± 7,9 | 7 / 10 / 0 | 1,534 ± 0,066 | 7.747 ± 3.159 | 0,1864 | 0,1082 |
| K3 | 10,7% ± 3,2 | 4 / 8 / 5 | 1,555 ± 0,055 | 5.696 ± 961 | 0,1779 | 0,1084 |
| referencia | 15,7% ± 0,9 | 8 / 9 / 8 | 1,616 ± 0,061 | 8.330 ± 22 | 0,1721 | 0,1123 |

(El Brier de K1 y K3 no se compara: su score es un promedio de riesgos, no el riesgo a H.)

## Las seis condiciones del preregistro (§5), con V

| | condición | K2 | K1 | K3 |
|---|---|---|---|---|
| 1 | gana en detección (diferencia > desvío combinado) | **+0,052 > 0,047 ✓** | +0,030 < 0,074 ✗ | −0,007 ✗ |
| 2 | no pierde en lift | +0,002 ✓ | +0,005 ✓ | +0,032 ✓ |
| 3 | diferencia pareada ≥ 0 en las 3 repeticiones | +0,067 / +0,067 / +0,022 ✓ | +0,022 / +0,089 / −0,022 ✗ | −0,089 / +0,022 / +0,044 ✗ |
| 4 | la anticipación no cae más que el desvío combinado | −1.916 contra 3.164 km ✓ | −2.037 contra 2.825 ✓ | sin valor (una repetición no detecta) |
| 5 | les gana a los dos pisos y pasa (a0) | ✓ · (a0) +0,0011 | ✓ · (a0) +0,0116 | pierde con `cut_odo` · (a0) +0,0176 |
| 6 | exceso sobre el nulo de tamaño de bolsa ≥ el de la referencia | +9,5 contra +4,1 pts ✓ | +12,6 ✓ | +8,9 ✓ |
| | **veredicto** | **se adopta** | empata con más desvío | empata con más desvío |

Solo K2 cumple las seis, así que la regla de desempate no hace falta.

## Nulos y auditorías

**Nulo de tamaño de bolsa para la detección** (`scripts/audit_detection_null.py`: 200
permutaciones del score crudo por repetición, con la misma post-transformación):
- **Sin transformar, un score al azar ya detecta ~7,5% con V** (p95 15,6%), porque los fallados
  tienen el doble de cortes que los sanos. Es la versión, para la detección, de lo que
  [f3-mil-agregacion-vehiculo.md](f3-mil-agregacion-vehiculo.md) midió para el lift.
- El finalista anterior le sacaba +4,1 puntos a ese nulo; **K2 le saca +9,5**.
- Con D, los dos le sacan lo mismo: +7,1 y +7,6.
- La media acumulada baja el nulo a ~2%. Por eso el exceso de K1 y K3 es grande aunque detecten
  poco.

**Bootstrap pareado por vehículo** (informativo, 1.000 remuestreos estratificados, sobre `score`):
- con V, K2 − referencia = **+5,5 puntos, IC90 [−8,9; +17,8]**, P(dif > 0) = 0,74;
- con D, −0,5 puntos, IC90 [−13,2; +9,4].

Con 45 fallados, una diferencia de 2–3 autos no sale del ruido en un bootstrap de vehículos.

**Auditorías de K2** (`audit_model.py`, R = 3, con D):

| | K2 | finalista anterior |
|---|---|---|
| (a0) features permutadas | 0,1263 contra una tasa base de 0,1252: **pasa** | pasa |
| (a′) aporte del *cuándo* | +0,019 ± 0,006 | +0,016 ± 0,003 |
| **(b) `aux_` de calendario** | **+0,020 de ROC** | +0,049 |
| C-index fuera de fold | 0,603 | 0,582 |
| lift por vehículo contra el nulo de tamaño de bolsa | 1,70× contra 1,03 ± 0,10 (p < 0,005) | 1,62× |
| contra el piso posicional | gana en detección, lift, anticipación y PR-AUC por fila | gana en las tres primeras |

- **Importancias:** el bin del hazard va primero (13%), seguido de la velocidad, la temperatura de
  motor, `feat_cut_odo`, la vida de aceite, la duración de viaje y el idle.
- **Correlación de rango con el finalista anterior:** 0,81–0,84 por fila y 0,84–0,88 por vehículo.
  Es el mismo modelo, con otro conjunto en riesgo.

## Qué significa

1. **Contar bien quién está en riesgo mejora el *qué auto*.** Tanto el horizonte completo como la
   ventana del registro apuntan a lo mismo: no pedirle al modelo que llame "sano" a un auto que
   después falla, ni a km que nadie verificaba.
   - Con V, la etiqueta honesta, eso vale 2–3 autos más de 45 al mismo presupuesto de falsas
     alarmas.
   - Con D, K2 empata en detección y ordena mejor los autos: 1,67× contra 1,62×.
2. **La (b) de calendario casi desaparece:** +0,020 contra +0,049. Es lo que F5 §3.3 había
   anticipado. El atajo del finalista anterior era exposición al registro, y K2 la saca sin
   perder el *qué auto*.
   - Esto apoya el diagnóstico de F5 §3.3: lo que ahí costaba el orden de autos era el reloj en
     días, no el conjunto en riesgo.
   - No es la ablación limpia, porque K2 también cambia el horizonte.
3. **Suavizar el score hacia atrás no ayuda.**
   - K1 sube la detección con V en una sola repetición (11 de 45) y en otra la baja a 4. Con D,
     en una repetición no detecta a nadie.
   - La media arrastra los primeros cortes. Así la alerta llega más tarde, o no llega: con K3, la
     anticipación con D cae a 5.700 km.
   - Lo que la media sí baja es el nulo. Quitarle al score la ventaja artificial de las bolsas
     largas es honesto, pero no convierte eso en más autos detectados.
4. **La anticipación mediana con V baja de 9.354 a 7.438 km.** Pasa la guarda, pero va dicho: los
   2–3 autos que K2 suma se detectan más cerca de su evento. Con D sube: de 8.330 a 8.811.

## Límites que hay que decir

- **Es "el mejor en dev" con la regla preregistrada, no una diferencia estadísticamente firme.**
  El bootstrap por vehículo cruza el cero, y con D empata en detección. El número del pitch es el
  de V, con su desvío: 17,0% ± 3,8.
- **El PR-AUC por fila (0,176) sigue por debajo del techo de cohorte (0,263).** Lo que el modelo
  sabe sigue siendo, sobre todo, *qué auto*.
- **El test no se tocó.** Medirlo es una decisión del equipo, y se hace una sola vez.
- **La ventana del registro sale de dev** (primer y último evento). Si Ford confirma otra ventana,
  se reconstruye el panel con `configs/data/event_clock.yaml` y K2 se re-corre con el mismo YAML.

## Desviaciones respecto del preregistro

- **Chequeo agregado antes de entrenar:** el stacker modificado reproduce el finalista anterior bit
  a bit (diferencia 0,0 en las tres repeticiones), así que la entrada tardía no toca la corrida
  sin `entry_km`.
- **"K2 y la referencia comparten nulo"** vale en distribución, no en el número: son dos
  simulaciones de Monte Carlo de la misma distribución (7,5% y 7,7%).
- **Informativas agregadas:** `audit_mil_bagsize.py` y `audit_positional_floor.py` sobre K2, como
  en toda ficha anterior. No deciden nada.
- **wandb:** todo corrió con `WANDB_MODE=disabled`, como estaba declarado. Las corridas están en
  `results/`.

## Qué queda abierto (ideas, no corridas)

- **K2 con bagging por vehículo.** El componente embolsado del finalista anterior daba 18,2% con
  D. Sería un candidato nuevo.
- **Una capa de decisión con garantía** (Neyman-Pearson, selección conforme; F5 §3.6) sobre el
  score de K2. No cambia el orden, así que no gasta presupuesto.
- **Preguntas para Ford:**
  - la ventana real del registro, que define el conjunto en riesgo de K2;
  - qué es `IdentificationDate`.

## Cómo se reproduce

```bash
export FORD_DATA_DIR=$PWD/data/rebuild-0921 WANDB_MODE=disabled
# el cache de viajes del panel de hitos, en el build de la referencia (no se versiona):
mkdir -p data/rebuild-0921/interim && cp data/interim/landmark_trips.parquet data/rebuild-0921/interim/
python scripts/build_km_window_panel.py --config configs/data/panel_survival_kmw.yaml --counts-only   # embudo
python scripts/build_km_window_panel.py --config configs/data/panel_survival_kmw.yaml
python scripts/train.py --config configs/exp_ss_hw_r3.yaml                          # K2, ~10 s
python scripts/audit_model.py --config configs/exp_ss_hw_r3.yaml                    # (a0), (a′), (b), (c)
python scripts/smooth_scores.py --config configs/exp_ss_cummean_r3.yaml --audit     # K1 + (a0), (b)
python scripts/smooth_scores.py --config configs/exp_ss_hw_cummean_r3.yaml --audit  # K3 + (a0), (b)
for c in exp_ss_hw_r3 exp_ss_cummean_r3 exp_ss_hw_cummean_r3; do
  python scripts/eval_window_label.py --config configs/$c.yaml        # D, V, veredicto contra la referencia
  python scripts/audit_detection_null.py --config configs/$c.yaml     # nulo de tamaño de bolsa + bootstrap
done
python scripts/audit_mil_bagsize.py f6-ss-hw-r3
python scripts/audit_positional_floor.py f6-ss-hw-r3
python scripts/check_setup.py                                                        # 166 chequeos
```

La referencia (`f3-survival-stacking-r3-rebuild`) tiene que estar entrenada en el mismo checkout
([f3-ensamble-e1-e2.md](f3-ensamble-e1-e2.md)).

**Código nuevo:**
- `scripts/build_km_window_panel.py` y `configs/data/panel_survival_kmw.yaml`;
- el target `window_km_survival` (`src/training/targets.py`) y `entry_km` en
  `DiscreteSurvivalStacker`;
- `scripts/smooth_scores.py` y `scripts/audit_detection_null.py`;
- 8 chequeos en `check_setup.py`. Cuatro mutaciones los hacen fallar: sin entrada tardía, sano
  censurado en su último odómetro, media no causal y target sin entrada.
