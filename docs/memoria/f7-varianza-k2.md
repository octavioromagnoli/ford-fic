# Bajar la varianza de K2: ninguno de los tres candidatos cumple la regla, y K2 sigue

**Fecha:** 2026-09-24 · **Fase:** F7 · **Rama:** `exp/f7-varianza-k2`
**Alcance:** solo dev, R = 3, con las mismas 2.029 filas, 171 vehículos, target y folds que K2
(build `data/rebuild-0921`, `splits_r3.json`). **Test sin tocar.**
**Preregistro:** [f7-preregistro-varianza-k2.md](f7-preregistro-varianza-k2.md) (0b29b2b). La
implementación y los chequeos se commitearon antes de entrenar (88660e7).

## En una línea

**Ninguno de los tres candidatos cumple las seis condiciones con la etiqueta que decide (V):
el finalista sigue siendo K2.**
- **P1 (K2 embolsado)** cumplió lo que prometía, bajar la varianza: detecta **7 / 7 / 7** de 45
  en las tres repeticiones, contra 7 / 10 / 6 de K2. Pero la media queda abajo (15,6% contra
  17,0%) y le saca menos al nulo (+7,7 contra +9,5 puntos).
- **P2 (K2 monótono)** sube la media (20,7%) pero con más del doble de desvío (8 / 15 / 5 de
  45, ± 9,3 contra ± 3,8), y alerta ~2.100 km más cerca del evento.
- **P3 (hazard logístico con cinco covariables)** pierde en todo: 6,7% contra 17,0%, lift 1,33×
  contra 1,66×, y detecta **menos que un score al azar**.

Lo más útil que deja no es un modelo nuevo sino una lectura del número del pitch: **el 17,0% de K2
tiene una repetición con suerte**. El mismo modelo embolsado, que es la versión de menor varianza
de K2, da 15,6% en las tres.

## Qué se probó (lista cerrada)

| | qué es | config |
|---|---|---|
| **P1** | K2 envuelto en `vehicle_bagging`: 10 bootstraps de autos del train de cada fold, mismos params | `exp_f7_p1_bag.yaml` |
| **P2** | K2 con `monotone_constraints` del mapa del 18-09 (9 features con +1, 7 con −1) | `exp_f7_p2_monotone.yaml` |
| **P3** | el apilado de K2 con un hazard logístico (L2, C = 1) sobre 5 covariables + mercado + spline natural del bin | `exp_f7_p3_pem.yaml` |

## Los números

**Etiqueta corregida (V), la que decide.** Son 1.435 filas, 182 positivas, 45 fallados y 95 sanos:

| | detección @ ≤ 50 FA/1.000 | detectados | lift por vehículo | anticipación mediana [km] | PR-AUC por fila | Brier |
|---|---|---|---|---|---|---|
| **K2 (referencia)** | **17,0% ± 3,8** | 7 / 10 / 6 | 1,658 ± 0,074 | 7.438 ± 2.005 | 0,1574 | 0,1167 |
| P1 embolsado | 15,6% ± 0,0 | 7 / 7 / 7 | 1,680 ± 0,061 | 7.247 ± 2.629 | 0,1670 | 0,1143 |
| P2 monótono | 20,7% ± 9,3 | 8 / 15 / 5 | 1,687 ± 0,076 | 5.306 ± 1.430 | 0,1630 | 0,1156 |
| P3 logístico | 6,7% ± 5,4 | 6 / 3 / 0 | 1,334 ± 0,050 | 5.003 ± 682 | 0,1496 | 0,1124 |
| piso `cut_odo` | 4,4% | 2 | 1,156 | 8.308 | | |
| piso `aux_cut_dss` | 0,0% | 0 | 1,210 | — | | |

**Etiqueta dura (D), informativa.** Son 2.029 filas, 254 positivas y 53 fallados:

| | detección | detectados | lift por vehículo | anticipación [km] | PR-AUC por fila | Brier |
|---|---|---|---|---|---|---|
| **K2** | 15,1% ± 3,1 | 8 / 10 / 6 | **1,671 ± 0,028** | 8.811 ± 370 | 0,1755 | 0,1124 |
| P1 | **17,0% ± 1,5** | 8 / 9 / 10 | 1,630 ± 0,037 | 7.252 ± 2.532 | **0,1837** | **0,1105** |
| P2 | 15,7% ± 5,0 | 7 / 12 / 6 | 1,642 ± 0,039 | 9.547 ± 483 | 0,1738 | 0,1121 |
| P3 | 6,9% ± 0,9 | 4 / 4 / 3 | 1,468 ± 0,055 | 7.631 ± 3.177 | 0,1586 | 0,1103 |

## Las seis condiciones del preregistro (§5), con V

"Desvío combinado" es `√(σ²_candidato + σ²_K2)` entre repeticiones (`ensemble_rank.py::compare`).

| | condición | P1 | P2 | P3 |
|---|---|---|---|---|
| 1 | gana en detección (diferencia > desvío combinado) | −0,015 contra 0,038 ✗ | +0,037 contra 0,101 ✗ | −0,104 contra 0,066 ✗ (pierde) |
| 2 | no pierde en lift | +0,022 ✓ | +0,030 ✓ | −0,324 contra 0,090 ✗ (pierde) |
| 3 | diferencia pareada ≥ 0 en las 3 repeticiones | 0 / −0,067 / +0,022 ✗ | +0,022 / +0,111 / −0,022 ✗ | −0,022 / −0,156 / −0,133 ✗ |
| 4 | la anticipación no cae más que el desvío combinado | −191 contra 3.306 km ✓ | −2.132 contra 2.463 km ✓ | −2.435 contra 2.118 km ✗ |
| 5 | les gana a los dos pisos y pasa (a0) | ✓ · (a0) +0,0045 | ✓ · (a0) +0,0012 | pierde con `cut_odo` · (a0) +0,0051 |
| 6 | exceso sobre el nulo de tamaño de bolsa ≥ el de K2 (+9,5) | +7,7 ✗ | **+13,0 ✓** | −1,6 ✗ |
| | **veredicto** | **no se adopta** | **no se adopta** | **no se adopta** |

Ninguno cumple la condición 1, que es la que expresa el pedido (detectar más autos). La regla de
desempate no hace falta.

## Nulos y auditorías

**Nulo de tamaño de bolsa** (`audit_detection_null.py`, 200 permutaciones por repetición, con V):

| | detección | media del nulo (p95) | exceso |
|---|---|---|---|
| K2 | 17,0% | 7,5% (15,6%) | +9,5 |
| P1 | 15,6% | 7,9% (17,8%) | +7,7 |
| P2 | 20,7% | 7,8% (17,8%) | **+13,0** |
| P3 | 6,7% | 8,3% (17,8%) | **−1,6** |

**Bootstrap pareado por vehículo** contra K2 (informativo, 1.000 remuestreos, sobre `score`):
- P1: −0,3 puntos, IC90 [−8,9; +8,9], P(dif > 0) = 0,36;
- P2: −0,5 puntos, IC90 [−11,1; +8,9], P(dif > 0) = 0,35;
- P3: −10,8 puntos, IC90 [−24,4; +6,7], P(dif > 0) = 0,11.

**Auditorías** (`audit_model.py`, R = 3, con D):

| | K2 | P1 | P2 | P3 |
|---|---|---|---|---|
| (a0) features permutadas | pasa (+0,0011) | pasa (+0,0045) | pasa (+0,0012) | pasa (+0,0051) |
| (a′) aporte del *cuándo* | +0,019 ± 0,006 | +0,028 ± 0,003 | +0,011 ± 0,005 | **−0,044 ± 0,003 (falla)** |
| (b) `aux_` de calendario | +0,020 de ROC | +0,032 | +0,023 | +0,000 |
| C-index fuera de fold | 0,603 | 0,594 | 0,604 | 0,556 |
| ρ de rango con K2, por vehículo (V) | — | 0,90–0,92 | 0,92–0,94 | 0,45–0,54 |

## Qué significa

1. **El 17,0% de K2 es la versión optimista del número.** P1 es K2 promediado sobre 10 bolsas de
   autos: ordena casi igual (ρ 0,90–0,92 por vehículo) y detecta 7 de 45 en las tres
   repeticiones. K2 detectó 7, 10 y 6. Con D pasa lo mismo al revés: K2 15,1% ± 3,1, P1
   17,0% ± 1,5. Las dos lecturas juntas dicen que la detección de esta familia de modelos está
   en **~15–17%, con ~7 autos de 45 como valor típico**, y que el 17,0% ± 3,8 lo empuja una
   repetición de 10. No cambia el finalista (lo decide la regla), pero **para el pitch conviene
   citar el rango y no el número puntual**.
2. **Las restricciones monótonas no bajan la varianza: la suben.** P2 es el candidato con la
   media más alta (20,7%) y el único con más exceso sobre el nulo que K2 (+13,0), pero una
   repetición detecta 15 y otra 5. Además, los autos que suma los alerta más cerca del evento
   (5.306 km contra 7.438). Es la misma forma que tuvieron K1 y K3 en F6: más detección media,
   con un desvío que no deja separarla del ruido.
3. **Cinco coeficientes no saben lo que sabe K2.** P3 detecta menos que un score al azar (exceso
   −1,6), ordena los autos peor (lift 1,33×) y su (a′) es negativo. El "una sola dimensión de
   señal" de F3 no se sostiene en su forma lineal más simple: K2 usa no linealidades o features
   que las cinco preregistradas no llevan. Entre las más importantes de K2 (ficha de F6) están
   `feat_cut_odo`, la vida de aceite y la duración de viaje; en P2 aparecen además los viajes
   por día y las horas entre viajes. Ninguna entró a P3.
   *Diagnóstico posterior al veredicto, no preregistrado:* el 73% del |coeficiente| de P3 es el
   mercado (CNTRY_3 y CNTRY_4), y `km_per_day` entra con signo **positivo** (+0,14), al revés de
   lo que sugería el cure model ("el auto que se usa poco falla más").
4. **Ningún candidato abre un atajo de calendario nuevo.** La (b) de P1 (+0,032) y P2 (+0,023)
   queda en el orden de la de K2 (+0,020). P3, sin `feat_cut_odo` ni calendario, da +0,000.

## Límites que hay que decir

- **Es un resultado negativo con tres candidatos y 45 fallados.** Un vehículo son 2,2 puntos de
  detección con V; ninguna diferencia de la tabla, salvo la de P3, sale del ruido del bootstrap.
- **La lectura del punto 1 no es una condición del preregistro.** Es lo que dicen los números
  de P1 sobre la estabilidad de K2, y va escrito como tal.
- **El test no se tocó.**

## Desviaciones respecto del preregistro

- **`max_iter: 5000`** en la logística de P3 (default del builder, no declarado en el
  preregistro). No cambia el modelo, solo asegura que el optimizador converja.
- **Cómo llegan los nombres al modelo.** El preregistro decía "`monotone_constraints` por nombre";
  la implementación lo resuelve con `wants_feature_names`: `cv.py` le pasa un DataFrame solo al
  modelo que lo pide. K2 reentrenado con ese código reproduce sus 19 métricas exactas.
- **La base del spline**: 5 nodos en los cuantiles 5–95% de los bins apilados del train de cada
  fold, que es la forma estándar de fijar "4 grados de libertad" (Harrell). El preregistro no
  decía dónde van los nodos.
- **wandb:** todo corrió con `WANDB_MODE=disabled`, como estaba declarado. Las corridas están en
  `results/`.

## Qué queda abierto (ideas, no corridas)

- **La capa de decisión del §6 del preregistro** (curva de detección a 2 / 5 / 10 / 20% de falsas
  alarmas y umbral de Neyman-Pearson) sobre K2. No consume presupuesto y todavía no corrió.
- **P1 + P2 (monótono embolsado).** Si el bagging baja la varianza y las restricciones suben la
  media, la combinación es la pregunta obvia. Sería un candidato nuevo, con su preregistro.
- **Por qué P3 pierde tanto.** Qué features de K2 cargan la señal que las cinco preregistradas no
  tienen. Es análisis, no candidato.

## Cómo se reproduce

```bash
export FORD_DATA_DIR=$PWD/data/rebuild-0921 WANDB_MODE=disabled
# el build: ver f3-desvio-historia.md ("El build") y la memoria de F6; K2 tiene que estar entrenado
python scripts/train.py --config configs/exp_ss_hw_r3.yaml                 # K2, la referencia
for c in exp_f7_p1_bag exp_f7_p2_monotone exp_f7_p3_pem; do
  python scripts/train.py --config configs/$c.yaml
  python scripts/audit_model.py --config configs/$c.yaml                   # (a0), (a′), (b), (c)
  python scripts/audit_detection_null.py --config configs/$c.yaml          # nulo de tamaño de bolsa + bootstrap
  python scripts/eval_window_label.py --config configs/$c.yaml             # D, V, veredicto contra K2
done
python scripts/check_setup.py                                               # 178 chequeos
```
