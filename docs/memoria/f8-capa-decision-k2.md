# Capa de decisión sobre K2: dónde se opera y si el 5% se sostiene fuera de muestra

**Fecha:** 2026-09-24 · **Fase:** F8 (sin presupuesto: §6 del preregistro F7, F5 §3.6)
**Alcance:** las predicciones fuera de fold de K2 (`f6-ss-hw-r3`), solo dev, R = 3, sin reentrenar.
**Test sin tocar.** La configuración (`configs/exp_decision_k2.yaml`) se commiteó antes de medir
(8b35e33).

```bash
export FORD_DATA_DIR=$PWD/data/rebuild-0921 WANDB_MODE=disabled
python scripts/decision_layer.py --config configs/exp_decision_k2.yaml   # K2 tiene que estar entrenado
```

No es un candidato: todo lo de acá es monótono en el score de K2, así que no cambia el orden de los
autos ni el finalista.

## En una línea

**El punto del 5% se sostiene fuera de muestra, y el mejor punto de operación está entre 5% y 10%
de falsas alarmas.**
- Con el umbral fijado sin mirar a los sanos que se miden, K2 da **3,5% de falsas alarmas
  realizadas** (máximo 4,2%) y **detecta 15,6%** (7 / 9 / 5 de 45). Es el rango ~15–17% que ya
  cita el pitch.
- A **10% de falsas alarmas detecta 26,7%** (10 / 12 / 14), con ~6.900 km de anticipación. Le
  saca al nulo lo mismo que el 5% (+10,5 contra +9,5 puntos): son ~4 autos más.
- **Por debajo del 5%, K2 no funciona:** al 2% detecta lo mismo que un score al azar.

## 1 · La curva (etiqueta V, que decide)

Son 45 fallados y 95 sanos. La media de 3 repeticiones sale de la misma cuenta que `train.py`. El
"nulo" es la detección de un score permutado entre filas, que conserva el largo de cada historial
(regla 6).

| falsas alarmas (presupuesto) | detección | detectados | FA realizadas | anticipación mediana | nulo (p95) | exceso |
|---|---|---|---|---|---|---|
| 2% | 2,2% | 0 / 1 / 2 | 0,7% | — | 2,7% (8,9%) | **−0,5** |
| **5%** | **17,0%** | 7 / 10 / 6 | 3,2% | 7.438 km | 7,5% (16,3%) | **+9,5** |
| **10%** | **26,7%** | 10 / 12 / 14 | 9,5% | 6.938 km | 16,2% (28,1%) | **+10,5** |
| 20% | 39,3% | 16 / 20 / 17 | 18,9% | 5.764 km | 32,2% (44,5%) | +7,1 |

Con la etiqueta dura (D, informativa), el 2% sí detecta: 13,2% (8 / 10 / 3), con un exceso de
+9,6. Pero el desvío es alto, y con V no se sostiene.

**Lectura:** la señal de K2 está en la zona de 5–10% de falsas alarmas. Más arriba, el nulo crece
más rápido que la detección. Más abajo, K2 no sabe separar a los autos más riesgosos de los sanos
más raros.

## 2 · ¿El 5% se sostiene fuera de muestra?

Hoy el umbral del 5% se elige con los mismos 95 sanos con los que se mide, así que la tasa realizada
es ≤ 5% por construcción. Acá, en cada fold de cada repetición, el umbral se fija con los sanos de
los **otros** folds y se aplica al fold que queda afuera:

| objetivo | método | FA realizadas afuera (máx.) | detección afuera | detectados | anticipación |
|---|---|---|---|---|---|
| ≤ 5% | empírico (lo de hoy) | **3,5% (4,2%)** | **15,6% ± 3,6** | 7 / 9 / 5 | 7.455 km |
| ≤ 5% | Neyman-Pearson (δ = 5%) | 1,8% (2,1%) | 5,2% ± 2,8 | 1 / 2 / 4 | 12.149 km |
| ≤ 10% | empírico | 9,5% (9,5%) | 25,9% ± 4,2 | 9 / 13 / 13 | 7.038 km |
| ≤ 10% | Neyman-Pearson (δ = 5%) | **4,2% (4,2%)** | **17,0% ± 3,8** | 7 / 10 / 6 | 7.522 km |

- **El umbral empírico no está sobreajustado.** El 5% da 3,5% afuera y el 10% da 9,5%.
- **El IC95 de Clopper-Pearson de la tasa dentro de muestra llega a 7,9%** (con 3 de 95 sanos). Es
  lo que hay que decir junto al "5%": con 95 sanos, la tasa real puede estar bastante más arriba sin
  que dev lo vea.
- **Neyman-Pearson con garantía de ≤ 5% es demasiado conservador con 95 sanos.** Obliga a no
  alertar a ningún sano del ajuste, y la detección cae a 5%.
- **Garantizar ≤ 10% con 95% de confianza equivale a operar a ~4% realizado, y detecta 17,0%.** Es
  la frase defendible para el pitch.

## Qué significa para el pitch

- **El número:** "con ≤ 5% de falsas alarmas detectamos ~15–17% de los autos que van a fallar, con
  ~7.400 km de anticipación". Ahora está verificado también fuera de muestra (15,6% con 3,5% de
  falsas alarmas).
- **La garantía:** "con 95% de confianza, la tasa de falsas alarmas no supera el 10%, y a ese umbral
  detectamos 17%" (Neyman-Pearson).
- **El dial:** si Ford acepta 10% de falsas alarmas, se detecta ~26% (~12 de 45), con ~6.900 km de
  anticipación. Elegir el punto es una decisión de costo de Ford (`cost_ratio_sweep`), no algo que
  se optimice en dev.
- **Lo que no hay que prometer:** alertas "casi sin falsas alarmas" (≤ 2%). Ahí K2 es azar.

## Límites

- Son 45 fallados y 95 sanos: un auto es 2,2 puntos de detección, y un sano 1,05 puntos de falsas
  alarmas.
- El ajuste cruzado usa scores de modelos distintos por fold. En producción hay un solo modelo, así
  que esto es conservador para el umbral empírico.
- La garantía de Neyman-Pearson supone que los sanos nuevos son intercambiables con los de dev. Con
  otro mercado o con otra ventana del registro, no vale.
