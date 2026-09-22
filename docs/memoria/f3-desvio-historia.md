# El desvío respecto de la historia del vehículo no le suma al finalista

**Fecha:** 2026-09-21 · **Fase:** F3 · **Rama:** `exp/history-deviation` · **Alcance:** dev
del panel v1 (2.029 filas, 171 vehículos, 254 positivas, tasa base 0,1252), folds
congelados `splits_r3.json`, build `2026-09-20` (ver "El build", abajo). · **Reproduce:**

```bash
export FORD_DATA_DIR=<dir con el panel reconstruido>          # ver "El build"
python scripts/build_survival_panel.py --config configs/data/panel_survival.yaml
python scripts/build_history_panel.py  --config configs/data/panel_history_v1.yaml
python scripts/audit_history_univariate.py --config configs/exp_survival_stacking_history_r3.yaml
python scripts/train.py --config configs/exp_survival_stacking_r3.yaml --run-name f3-survival-stacking-r3-rebuild
python scripts/train.py --config configs/exp_survival_stacking_history_r3.yaml
python scripts/audit_model.py --config configs/exp_survival_stacking_history_r3.yaml
python scripts/audit_model.py --config configs/exp_survival_stacking_r3.yaml --out experiments/f3-survival-stacking-r3-rebuild/audit.json
python scripts/audit_positional_floor.py f3-survival-stacking-history-r3
python scripts/audit_positional_floor.py f3-survival-stacking-r3-rebuild
```

## Qué se midió

La pista que dejó TimesFM ([f3-timesfm-zeroshot.md](f3-timesfm-zeroshot.md), "Lo que sí
dejó") y el punto 1 de "Qué queda abierto" en la entrada del 20-09 de
[decisiones.md](decisiones.md): medir cada ventana **contra la historia previa del mismo
vehículo**, sin TimesFM, sobre el finalista.

`feat_<x>_hist_delta = agg(historia previa) − agg(ventana)`, con la historia previa en
`(c − 25.600, c − W]` y la ventana en `(c − W, c]`
(`src/features/windows.py::compute_history_deviation`, spec en
`configs/data/features_history.yaml`). Dos columnas, cada una con el mismo agregador
que su columna del panel v1:

- `feat_regenerations_per_1000km_hist_delta`: caídas de `AirRegeneration` de 15 puntos o
  más, por cada 1.000 km (el lado de la ventana es `feat_regenerations_per_1000km`);
- `feat_dpf_end_mean_hist_delta`: media de `AirRegenerationEnd` (la ventana es
  `feat_dpf_end_mean`).

Con menos de 1.000 km recorridos antes de `c − W` el desvío es **NaN**, no cero: le pasa al
15% de las filas (cobertura 84,6%). El builder verifica que el lado de la ventana
reproduzca las dos columnas del panel v1 en las 2.507 filas, y que `vehicle_id`, `cut_odo`
y `label` coincidan fila a fila con `panel.parquet`.

## Chequeo univariado (antes de entrenar)

Sobre dev (`audit_history_univariate.py`), en las filas con historia suficiente:

| columna | ROC | ρ con el mes del corte | ρ con `cut_odo` | ρ con km de historia |
|---|---:|---:|---:|---:|
| `feat_regenerations_per_1000km_hist_delta` | **0,598** | +0,11 | +0,29 | +0,29 |
| `feat_dpf_end_mean_hist_delta` | **0,579** | +0,13 | +0,45 | +0,43 |
| `aux_hist_km_covered` (solo el largo de la historia) | 0,548 | +0,37 | +0,97 | — |
| `feat_idle_frac` (la mejor del panel v1) | 0,597 | 0,00 | −0,18 | |

Reproduce el 0,602 del doc de TimesFM (y el 0,560 del nivel), aunque ese se midió sobre el
build `2026-09-19` con bins de 100 km. ρ con la tendencia dentro de W: +0,02, así que el
panel v1 no tenía esta información. La ρ con `cut_odo` es mayor que la de
`feat_tfm_regen_delta` (−0,14) y sale del lado de la historia (ρ +0,53 para el nivel): es
el asentamiento del filtro en los primeros km (§6 de decisiones, 20-09). El modelo ya tiene
`feat_cut_odo`, y el emparejado de sanos iguala la distribución del odómetro.

## Resultado: no suma

El criterio se declaró antes de correr, en el YAML del experimento: comparar contra el
finalista en el mismo build por (a′) ± desvío entre repeticiones, detección @ ≤50 FA/1000,
lift por vehículo con `mean` y el piso posicional. El PR-AUC por fila se reporta pero no
decide. Si la mejora de (a′) no supera el desvío combinado, el resultado es "no suma".

| R=3, mismo build, mismos folds | finalista (control) | + desvío de la historia | diferencia |
|---|---:|---:|---:|
| **(a′) aporte del cuándo** | +0,0162 ± 0,0033 | +0,0219 ± 0,0047 | **+0,0056** (desvío combinado 0,0057: **0,98×**) |
| (a′) pareado por repetición | | | +0,0129 · −0,0001 · +0,0041 |
| detección @ ≤50 FA/1000 | 15,7% ± 0,9 | 20,1% ± 5,4 | +4,4 pp (combinado 5,5) |
| anticipación mediana | 8.330 km **± 22** | 6.027 km **± 1.987** | −2.303 |
| lift por vehículo (`mean`) | 1,62× | 1,64× | +0,02 |
| PR-AUC entre fallados (lift) | 1,10 ± 0,07 | 1,15 ± 0,06 | +0,05 |
| PR-AUC por fila *(no decide)* | 0,1721 ± 0,0135 | 0,1820 ± 0,0133 | +0,0099 |
| Brier | 0,1123 | 0,1112 | |

**La mejora de (a′) es 0,98 veces el desvío combinado: no lo supera, así que el resultado
es "no suma".** Las demás métricas dicen lo mismo:

- **La detección sube, pero dentro del ruido, y a costa de anticipar menos.** Por
  repetición da 13,2% / **26,4%** / 20,8%, con anticipación de 8.365 / **3.507** / 6.210 km.
  El control da 15,1 / 17,0 / 15,1% con 8.333 / 8.302 / 8.355 km. La repetición que más
  detecta alerta a la mitad de distancia del evento: detecta más porque avisa más tarde.
- **Pierde justo la propiedad por la que se eligió al finalista.** La estabilidad era uno
  de sus dos argumentos (decisiones, 20-09, §4): ±22 km de anticipación entre repeticiones.
  Con las dos columnas nuevas pasa a ±1.987 km, casi lo mismo que el control binario
  (±2.237).
- **El lift por vehículo no se mueve** (1,64× contra 1,62×; `audit_positional_floor.py`,
  que lo calcula sobre el score promediado, da 1,61× contra 1,62×).

## Auditorías

| | control | + desvío |
|---|---:|---:|
| (a0) features permutadas entre todas las filas | 0,1215 (−0,0037 de la base) PASS | 0,1243 (−0,0009) PASS |
| (a′) sin retrain, score promediado | +0,0200 PASS | +0,0286 PASS |
| ROC dentro del vehículo (39 vehículos) | 0,604 | 0,628 |
| **(b)** ROC al sumar las `aux_` de calendario | 0,602 → 0,652 (**+0,0493**) | 0,626 → 0,656 (**+0,0301**) |

La (b) importaba acá: el contexto largo abarca meses distintos para cortes del mismo mes,
y el calendario es el atajo conocido de este dataset. **Las columnas nuevas no abren un
atajo nuevo.** La (b) marca en los dos modelos, y menos en el que tiene el desvío. Pero la
lectura honesta va en la otra dirección: con el calendario disponible, los dos terminan
en el mismo ROC (0,652 contra 0,656). O sea que **la mayor parte del ROC que suman las
columnas nuevas (+0,023) es información que las `aux_` de calendario también traen**. Es
coherente con su ρ de +0,11/+0,13 con el mes.

**Piso posicional** (`score = cut_odo`): los dos pierden contra el piso en (a′) (0,0286 y
0,0200 contra 0,0872) y en PR-AUC entre fallados (0,305 y 0,290 contra 0,330). Los dos le
ganan en detección (20,8% y 15,1% contra 7,5%) y en lift por vehículo (1,61× y 1,62× contra
1,02×). El desvío hace que el PR-AUC por fila pase a ganarle al piso (0,1857 contra 0,1766;
el control perdía con 0,1744), pero el PR-AUC por fila no decide. Anticipación en el punto
pooled: 7.353 km, que queda por debajo del piso (7.385).

## El build

El `data/processed/panel.parquet` local (20-09, 20:57) se construyó desde un checkout sin
4ff381e y no trae `aux_km_observed_after_cut`, que survival stacking necesita. Se
reconstruyó con `main` en un `FORD_DATA_DIR` aparte (`data/rebuild-0921/`, con `raw/` como
junction a `data/raw/` y copias de `test_split.json` y `splits_r3.json`), sin tocar
`data/processed/`:

- el panel reconstruido es **idéntico al del 20-09 en todas sus columnas** (comparadas una
  por una) y además trae `aux_km_observed_after_cut`;
- el `splits.json` que escribe `build_dataset.py` coincide con el congelado (misma huella,
  mismos folds), y se reemplazó igual por la copia congelada;
- el control re-corrido **reproduce el registro `f3-survival-stacking-r3` dígito a
  dígito**.

La columna `Panel` de `results/` dice `2026-09-22` porque es el `created_at` en UTC del
rebuild. El contenido es el del build `2026-09-20`.

## Qué queda

- **El desvío respecto de la historia no entra al set base.** Ninguna de las cuatro
  métricas que deciden supera el ruido del sorteo, y la estabilidad de la anticipación se
  pierde.
- **La (b) del finalista, medida con R=3, marca más de lo que decía el registro R=1**
  (+0,0493 contra +0,0230). Era el punto 4 pendiente de decisiones (20-09) y sigue abierto,
  ahora con un número peor: con el calendario disponible, el modelo sube 0,05 de ROC.
- Queda sin probar la otra mitad del punto 1 de decisiones (20-09): entrenar el target
  **solo sobre los vehículos que fallan**. Este resultado no la descarta. La información
  del desvío existe (ROC univariado 0,598), pero el hazard ya la cubre casi toda con las 53
  features más el odómetro.
