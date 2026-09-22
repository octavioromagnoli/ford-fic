# Ensamble de finalistas (E1, E2): no le gana a survival stacking

**Fecha:** 2026-09-22 · **Fase:** F3/F5 · **Rama:** `exp/ensamble-e1-e2` · **Alcance:** dev del
panel v1 (2.029 filas, 171 vehículos, 53 con evento), build `data/rebuild-0921`, folds
`splits_r3.json`, R = 3. **Test sin tocar.**
**Preregistro:** [f3-preregistro-landmark-ensamble.md](f3-preregistro-landmark-ensamble.md) (9b7bfb4),
pasos E1 y E2, aplicados sin cambios. L1 no corrió: el §3.3 de
[../f5-modelos-candidatos-bibliografia.md](../f5-modelos-candidatos-bibliografia.md) propone
reemplazarlo.

## En una línea

**Los dos ensambles pierden contra la referencia en lift por vehículo y no le ganan en
detección.** Promediar el rango de survival stacking con el del CNN-LSTM diluye lo que mejor hace
survival stacking, que es ordenar autos. El bagging por vehículo no lo arregla. **El finalista
sigue siendo survival stacking.**

## Qué se corrió

- **E1:** rango percentil de `score_r{r}` de cada finalista sobre las 2.029 filas, promedio
  50/50, por repetición. No se reentrena nada: son las OOF de `f3-survival-stacking-r3-rebuild`
  y `f3-cnn-lstm-r3-regen15`.
- **E2:** lo mismo, sobre los dos finalistas reentrenados con `vehicle_bagging`: 10 bootstraps de
  vehículos del train de cada fold, semilla 42 y los hiperparámetros internos sin cambios.
  - Los dos embolsados son **componentes**, no candidatos.
- **Antes de medir se verificó:**
  - que las dos corridas de referencia se reproducen **bit a bit** en este entorno (mismas OOF,
    mismos folds, en las 3 repeticiones);
  - que las filas coinciden por `(vehicle_id, cut_odo)` con la misma `label`;
  - que `fold_r{r}` es idéntico en los dos miembros.
- **Métricas:** `scripts/train.py::evaluate_predictions`, la misma cuenta de toda corrida.
- **Veredicto:** con la regla del preregistro. "Le gana" = diferencia de medias mayor que el
  desvío combinado `√(σ² + σ_X²)`.

## Los números

Media ± desvío entre las 3 repeticiones. La detección es a ≤ 50 falsas alarmas cada 1.000 sanos,
sobre 53 fallados. La última columna es el veredicto del candidato de esa fila contra la
referencia.

| corrida | detección | autos por repetición | anticipación mediana | lift veh. (`mean`) | (a′) | Brier | contra la referencia |
|---|---|---|---|---|---|---|---|
| **survival stacking** (referencia) | **15,7% ± 0,9** | 8 / 9 / 8 | **8.330 ± 22 km** | **1,616 ± 0,061** | +0,016 ± 0,003 | 0,112 | — |
| **E1** · SS + CNN-LSTM | 12,6% ± 3,2 | 5 / 9 / 6 | 9.972 ± 4.748 km | 1,515 ± 0,040 | +0,017 ± 0,006 | 0,108 (Platt OOF) | **pierde** (lift) |
| **E2** · E1 embolsado | 13,8% ± 5,0 | 5 / 11 / 6 | 9.323 ± 4.121 km | 1,480 ± 0,048 | +0,029 ± 0,010 | 0,108 (Platt OOF) | **pierde** (lift) |
| *componente:* SS embolsado | 18,2% ± 3,2 | 9 / 8 / 12 | 7.443 ± 712 km | 1,591 ± 0,034 | +0,030 ± 0,008 | 0,110 | empata, con más desvío (no es candidato) |
| *componente:* CNN-LSTM embolsado | 15,7% ± 3,6 | 7 / 11 / 7 | 9.502 ± 3.162 km | 1,321 ± 0,067 | +0,013 ± 0,002 | 0,189 | pierde (no es candidato) |
| CNN-LSTM | 11,9% ± 3,6 | 9 / 5 / 5 | 10.366 ± 137 km | 1,333 ± 0,054 | +0,008 ± 0,005 | 0,228 | pierde |
| control LightGBM | 11,9% ± 3,2 | 4 / 8 / 7 | 9.370 ± 2.237 km | 1,494 ± 0,049 | −0,006 ± 0,004 | 0,171 | pierde |
| SS + desvío contra la historia | 20,1% ± 5,4 | 7 / 14 / 11 | 6.027 ± 1.987 km | 1,642 ± 0,072 | +0,022 ± 0,005 | — | empata, con más desvío |
| piso posicional (odómetro) | 7,5% | 4 | 7.385 km | 1,018 | — | — | pierde |

**Contra la referencia, pareado por repetición (mismos folds):**

| | detección | lift por vehículo |
|---|---|---|
| E1 | −5,7 / 0,0 / −3,8 pts (diferencia −3,1, desvío combinado 3,3: empata) | −0,187 / −0,134 / +0,019 (diferencia −0,101, desvío combinado 0,073: **pierde**) |
| E2 | −5,7 / +3,8 / −3,8 pts (diferencia −1,9, desvío combinado 5,0: empata) | −0,105 / −0,197 / −0,105 (diferencia −0,136, desvío combinado 0,078: **pierde**) |

**Estabilidad de E2 contra E1** (lo único que el bagging promete, según el preregistro):
- el desvío de detección **sube**: 5,0 contra 3,2 puntos;
- el de anticipación **baja**: 4.121 contra 4.748 km.

## Auditorías

| | E1 | E2 | lectura |
|---|---|---|---|
| **(a0)** features permutadas entre todas las filas, los dos miembros reentrenados | PR-AUC 0,1271 contra tasa base 0,1252 (+0,0019) | 0,1281 contra 0,1252 (+0,0029) | sin fuga |
| **(a′)** aporte del cuándo | +0,017 ± 0,006 | +0,029 ± 0,010 | positivo en las tres repeticiones |
| **(b)** `aux_` de calendario como `feat_`, los dos miembros reentrenados | ROC 0,615 → 0,654 (**+0,039, marca**) | 0,616 → 0,663 (**+0,046, marca**) | el atajo de calendario del finalista (+0,049) sigue ahí |
| **piso posicional** (`audit_positional_floor.py`, sobre el score promediado) | le gana en detección (13,2% contra 7,5%) y en lift (1,52× contra 1,02×) | 17,0% contra 7,5% y 1,48× contra 1,02× | los dos le ganan en las dos que valen |
| **tamaño de bolsa** (`audit_mil_bagsize.py`, 200 permutaciones) | `mean`: lift 1,52× contra nulo 1,00 ± 0,10, p < 0,005 | `mean`: 1,48× contra 1,00 ± 0,10, p < 0,005 | la agregación `mean` supera su nulo |

Sobre el score promediado entre repeticiones, que es un ensamble de las tres pasadas, el piso
posicional le da a E2 17,0% de detección, pero con **3.060 km** de anticipación mediana. Ese
número no decide: la regla es la media de las tres repeticiones por separado
(`train.py::detection_by_repeat`).

## Por qué no suma

1. **El CNN-LSTM ordena autos peor.**
   - Su lift por vehículo es 1,33× contra 1,62× de survival stacking.
   - El promedio de rangos 50/50 le da el mismo peso a un ordenador de vehículos más débil, y
     el lift del ensamble queda en el medio (1,52×).
   - La correlación de rango baja entre los dos (0,24–0,36 por fila, 0,32–0,45 por vehículo)
     hacía esperar complementariedad. Pero complementarse sirve cuando los dos modelos tienen
     señal comparable, y acá no la tienen.
2. **La anticipación del ensamble no es estable.**
   - Por repetición: 15.908 / 4.285 / 9.724 km en E1.
   - El punto de operación admite 3–5 falsas alarmas sobre 118 sanos y detecta 5–9 autos: la
     mediana de anticipación se calcula sobre muy pocos autos y salta con cada sorteo.
   - survival stacking detecta casi los mismos 8–9 autos en las tres repeticiones, y por eso su
     ±22 km.
3. **El bagging no baja la varianza que importa.**
   - En los componentes sube la detección media: SS 18,2% contra 15,7%, CNN-LSTM 15,7% contra
     11,9%.
   - Pero también sube su desvío entre repeticiones: SS 3,2 contra 0,9 puntos. Con 53 fallados,
     un auto son 1,9 puntos, y las diferencias son de 1–3 autos que cambian de fold.

## Qué dice el preregistro, al pie de la letra

- **E1 pierde** contra la referencia (en lift) → *"el ensamble no suma y el finalista sigue
  siendo survival stacking solo"*.
- **E2 pierde** contra la referencia (en lift).
  - Contra E1, **baja el desvío de anticipación** (4.121 contra 4.748 km) y **sube el de
    detección** (5,0 contra 3,2).
  - La regla de descarte pide que no baje **ninguno** de los dos. Por la letra, entonces, el
    bagging **no queda descartado**.
  - Pero tampoco hay nada que adoptar: perdió contra la referencia, sus componentes no pueden
    ser finalistas por su cuenta, y la baja de anticipación es del 13% sobre un desvío que
    sigue siendo ~190× el de la referencia.
- El Brier de los rangos no se declara. El de la Platt fuera de fold es 0,108 en los dos
  ensambles, contra 0,112 de la referencia; es informativo, porque E1 no ganó en detección y el
  preregistro solo lo pedía en ese caso.

## Lo que queda

- **El componente survival stacking embolsado** detecta 18,2% ± 3,2 con lift 1,59×. Por el
  preregistro no es candidato. Contra la referencia empata en las dos métricas, con más
  desvío, lo que la regla cuenta como peor. **No se lo promueve sin un preregistro nuevo**, y
  aun así la regla lo dejaría en empate.
- **La (b) sigue marcando** en todo lo que contiene a survival stacking. Es el pendiente del
  finalista, y el §3.3 del doc de candidatos F5 propone atacarlo con el conjunto en riesgo de
  la ventana.
- **Presupuesto:** E1 y E2 eran dos de los cuatro lugares del preregistro del 21-09. L1 y
  L1-cal no se corrieron.

## Cómo se reproduce

```bash
export FORD_DATA_DIR=$PWD/data/rebuild-0921 WANDB_MODE=disabled
python scripts/train.py --config configs/exp_survival_stacking_r3.yaml --run-name f3-survival-stacking-r3-rebuild   # reproduce bit a bit
python scripts/train.py --config configs/exp_cnn_lstm_r3.yaml                                                     # f3-cnn-lstm-r3-regen15, ídem
python scripts/ensemble_rank.py --config configs/exp_ens_e1.yaml --audit       # E1 + (a0) y (b) del ensamble
python scripts/train.py --config configs/exp_survival_stacking_r3_bag10.yaml   # componentes de E2 (~30 s y ~7 min)
python scripts/train.py --config configs/exp_cnn_lstm_r3_bag10.yaml
python scripts/ensemble_rank.py --config configs/exp_ens_e2.yaml --audit       # E2 + (a0) y (b); ~15 min
python scripts/audit_positional_floor.py f3-ens-e1-ss-cnnlstm                  # y f3-ens-e2-ss-cnnlstm-bag10
python scripts/audit_mil_bagsize.py f3-ens-e1-ss-cnnlstm
python scripts/check_setup.py                                                  # 139 chequeos
```

Código nuevo:
- `src/models/bagging.py` y el builder `vehicle_bagging`;
- el target `grouped_label` (solo lleva el vehículo hasta el envoltorio);
- `scripts/ensemble_rank.py`;
- 10 chequeos en `check_setup.py`. Cuatro mutaciones los hacen fallar: bootstrap de filas,
  scores crudos en vez de rangos, no verificar folds y veredicto sin desvío combinado.
