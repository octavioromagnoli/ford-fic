# Preregistro: landmarking y ensamble de finalistas (pasos 2 y 3)

**Fecha:** 2026-09-21 · **Fase:** F3 · **Escrito:** antes de construir el panel de landmarks
y antes de entrenar o combinar ninguna corrida de estos dos pasos. El commit que agrega este
archivo es la marca de tiempo; cualquier cambio posterior va en un commit propio, anterior a
la corrida que afecta, y dice por qué.

Motivo: el presupuesto de comparaciones está agotado (CLAUDE.md; `docs/f3-modelos-candidatos.md`
§0, punto 2). Con ~53 vehículos con evento, un vehículo son 1,9 puntos de detección: sumar
candidatos sobre la marcha garantiza que "el mejor" sea ruido. Acá está la lista cerrada. Si
aparece una idea en el camino, va a "Qué queda abierto" de la ficha del paso, no a una corrida.

## Reglas comunes a todas las corridas

- **Referencia fija:** `f3-survival-stacking-r3-rebuild` (build `2026-09-20`,
  `splits_r3.json`). **Piso:** `scripts/audit_positional_floor.py` (score = `cut_odo`).
  Solo dev (`select_dev()`); `test_split.json` no se toca.
- **R = 3** con los folds de `splits_r3.json`. En el paso 2, la misma asignación
  vehículo → fold en cada repetición.
- **Deciden dos métricas** (`docs/memoria/f3-piso-posicional.md`):
  1. **detección @ ≤ 50 falsas alarmas / 1.000 sanos**, media ± desvío entre las 3
     repeticiones (`train.py::detection_by_repeat`, desvío poblacional);
  2. **lift por vehículo con agregación `mean`**, media entre repeticiones.

  Números de la referencia: **15,7% ± 0,9** (8 / 9 / 8 de 53) y **1,616× ± 0,061**. Del piso:
  7,55% (4/53) y 1,018×.
- **"Le gana a X en M"** = la diferencia de medias supera el desvío combinado
  `√(σ² + σ_X²)`. Se reporta además la diferencia pareada por repetición (mismos folds).
- **Veredicto contra la referencia:**
  - **gana** si le gana en al menos una de las dos y no pierde en la otra (perder = quedar
    por debajo por más del desvío combinado);
  - **pierde** si pierde en cualquiera de las dos;
  - **empata** en cualquier otro caso. Un empate con más desvío de detección o de
    anticipación que la referencia cuenta como peor: la estabilidad es criterio.

  Y ninguna corrida puede reemplazar a la referencia si no le gana **también al piso** en
  las dos (el mismo criterio del punto de control 1).
- **Informativas, no deciden:** PR-AUC por fila, PR-AUC entre fallados, (a′), Brier,
  C-index, anticipación mediana (salvo como estabilidad).
- **Auditorías de cada corrida** (no son corridas y no consumen presupuesto): (a0), (a′),
  (b), piso posicional y `audit_mil_bagsize.py`. Si (a0) no cae a la tasa base, la corrida
  no se lee.
- **Sin barridos.** Hiperparámetros del modelo = los de la referencia. Nada se elige mirando
  un resultado.

## Las corridas (cuatro lugares, uno condicional)

### L1 · survival stacking sobre el panel de landmarks (paso 2)

- **Panel:** `configs/data/panel_landmark.yaml`. Landmarks iniciales 2.000–12.000 km cada
  2.000; se ajustan en el punto de control 2a **solo con conteos** (vehículos en riesgo,
  positivos, positivos por fold), antes de entrenar. W, G, H del panel v1; features de
  `configs/data/features_v1.yaml` vía `src/features/windows.py`; `feat_cut_odo = L`; el mes
  del corte como `aux_`.
- **Universo:** los vehículos de dev que están en `splits_r3.json` (los 171 del panel v1),
  para que "mismo vehículo, mismo fold" se cumpla sin sortear nada. Los sanos censurados
  antes de `L + G + H` se descartan en ese landmark, como `censored_policy: drop` del panel v1.
- **Modelo:** el YAML de la referencia sin cambios (`survival_stacking`, `horizon_km 3000`,
  `bin_km 500`, `max_horizon_km 6000`, `random_state 42`, target `discrete_survival`).
- **Alerta:** un vehículo alerta en el **primer landmark** con score ≥ umbral
  (`k_consecutive: 1`); el umbral es el que respeta ≤ 50 FA/1.000 sanos.
- **Comparación:** contra la referencia **sobre los vehículos que están en los dos**, con el
  punto de operación de cada uno recalculado sobre ese conjunto. Se reporta además cada uno
  sobre su conjunto completo.
- **Paradas antes de leer nada:** si el piso posicional *dentro* de un landmark no da la tasa
  base de ese landmark, o si el agrupado se aparta de la tasa base por algo que no sea el
  hazard base sobre el odómetro, se para y se reporta. Si (b) marca (> 0,02 de ROC), no se
  interpreta L1 hasta proponer la corrección en un punto de control.
- **Resultado que la descarta:** empata o pierde contra la referencia en las dos métricas
  → landmarking limpia las métricas (PR-AUC, piso, bolsas) pero no cambia el eje vehículo ni
  la detección; queda como herramienta de auditoría y no como modelo.

### L1-cal · L1 con la corrección de calendario (condicional)

Solo si la (b) de L1 marca **y** la corrección se aprueba en un punto de control (por
ejemplo, estratificar por mes dentro del landmark). Mismo modelo, mismas reglas, mismo
criterio de descarte que L1. Si no se usa, el lugar queda vacío: no se reasigna.

### E1 · ensamble por rango de los dos finalistas, sin reentrenar (paso 3)

- **Pareja:** la que salga del punto de control 1 (por defecto survival stacking +
  CNN-LSTM, `f3-cnn-lstm-r3-regen15`).
- **Score:** para cada repetición `r`, rango percentil (empates promediados) de `score_r{r}`
  de cada modelo sobre las 2.029 filas de dev, y promedio **50/50**. Pesos fijos, no se
  ajustan. Antes, verificar que las filas coincidan por `(vehicle_id, cut_odo)` y que
  `fold_r{r}` sea idéntico en las dos corridas.
- **Calibración:** el rango no es una probabilidad, así que su Brier no se declara. Si E1
  gana en detección, se reporta el Brier de una **calibración Platt fuera de fold** (para
  cada fold, la logística se ajusta con las filas de los otros folds). Platt es monótona:
  no cambia detección ni lift, así que **no es un candidato aparte y no consume lugar**; es
  la forma de reportar el Brier de E1.
- **Se compara contra:** la referencia, cada finalista solo y el piso.
- **Resultado que lo descarta:** empata o pierde contra la referencia → el ensamble no suma
  y el finalista sigue siendo survival stacking solo.

### E2 · el mismo ensamble con bagging por vehículo (paso 3)

- Cada finalista se reentrena envuelto en `vehicle_bagging` (`src/models/registry.py`):
  **`n_bags: 10`** remuestreos bootstrap **de vehículos** del train de cada fold (con
  reposición), promedio de los scores de las bolsas, semilla 42 en el YAML. Mismos
  hiperparámetros internos que las corridas originales, R = 3 y mismos folds.
- El ensamble es exactamente el de E1 (rango percentil 50/50 por repetición) sobre las OOF de
  los dos modelos embolsados. Misma regla de calibración que E1.
- Los dos modelos embolsados se reportan **como componentes**, no como candidatos: ninguno
  puede quedar como finalista por su cuenta.
- **Qué decide:** el veredicto contra la referencia, y contra E1 la **estabilidad** (desvío
  de detección y de anticipación entre repeticiones). Es lo único que el bagging promete.
- **Resultado que lo descarta:** no gana contra la referencia **y** no baja el desvío de
  detección ni el de anticipación respecto de E1 → el bagging no se adopta.
- E2 corre aunque E1 no gane: E1 mide si la combinación suma; E2, si la varianza del
  entrenamiento era lo que lo impedía.

## Lo que queda fuera a propósito

Pesos del ensamble distintos de 50/50, otros `n_bags`, otra pareja, landmarks más finos
después de ver resultados, un segundo modelo sobre landmarks, stacking con meta-modelo,
survival stacking embolsado solo como candidato. Todo eso es un barrido o un candidato nuevo.
