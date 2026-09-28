# Preregistro F13 · MiniRocket sobre la secuencia de la ventana

**Fecha:** 2026-09-28, escrito **antes** de medir. **Alcance:** solo dev v2 (446 autos),
`splits_r3.json`, test sin tocar. Es un solo brazo y gasta un lugar del presupuesto de comparaciones.

## Por qué este candidato

- F12 mostró que GRU, CNN-LSTM y sus ensambles chocan con el mismo techo al 5–10% de falsas alarmas.
  El AUC dentro de mercado × motor queda en 0,62–0,66
  ([f12-preregistro-deteccion.md](f12-preregistro-deteccion.md)).
- **MiniRocket no aprende sus convoluciones:** usa ~10.000 kernels fijos y deja todo el aprendizaje
  en un modelo lineal regularizado. Tiene pocos parámetros efectivos y es estable con pocos datos.
  Si le gana a la GRU, el techo era en parte del modelo. Si empata, es otra evidencia de que el
  límite es la información.
- **Lo que no se prueba** (y por qué), del mismo texto que lo propuso:
  - dT/dt dentro del viaje, ciclos del termostato y correlación con la carga: TripSummary tiene una
    fila por viaje, sin curva interna ni carga/RPM;
  - DTW sobre curvas de viaje: por lo mismo;
  - autoencoder entrenado con sanos: ya se probaron el desvío contra la flota sana y contra la
    historia del auto (`FleetReferenceNormalizer`, `feat_*_hist_delta`).

## El candidato (M1)

- **Entrada:** la misma que la GRU de referencia. Es `panel_seq_trips_v2_estaticas.parquet`: 20
  tramos de 50 km × 15 canales, más país, motor y modelo. El `Pipeline` de `cv.py` imputa y
  estandariza con el train del fold, igual que para la GRU.
- **Modelo:** `minirocket` (`src/models/minirocket.py`), una implementación en numpy de MiniRocket
  multivariado (Dempster, Schmidt y Webb, KDD 2021).
  - 84 kernels de largo 9 con pesos {−1, 2}.
  - Dilataciones de 1 a ⌊(T−1)/8⌋, que con T = 20 dan {1, 2}.
  - Padding alternado y un subconjunto aleatorio de canales por kernel.
  - Sesgos tomados de cuantiles de la convolución sobre filas del train.
  - Como feature, la proporción de valores positivos (PPV): ~10.000 features.
  - Las estáticas se concatenan al final. `RidgeClassifierCV` con α ∈ logspace(−3, 3, 10),
    `class_weight: balanced`, y el score es la función de decisión.
- **No se usa `aeon` ni `sktime`:** bajarían numpy y scipy, que están fijados en el entorno.
- **Semillas:** 42, 1 y 2, en promedio por rango. Es la misma cuenta que la referencia.
- **Hiperparámetros:** los de referencia del paper (10.000 features, 84 kernels, rango de α). No
  se barre nada.

## Regla

- **Referencia:** P0 de F12, la GRU + TripSummary + estática ×3 semillas (43% al 10%).
- **Se mide con** `scripts/report_v2_models.py`: umbral exacto, k = 2, promedio de R = 3 y bootstrap
  pareado por vehículo con 1.000 réplicas.
- **Métrica primaria:** detección al 10%; secundaria, al 5%.
- **Se adopta solo si** la diferencia contra P0 al 10% tiene el IC 95% por encima de 0 **y** el AUC
  dentro de mercado × motor no baja.
- **Expectativa declarada:** empate, entre −5 y +3 puntos al 10%.
- **El test no se abre para esto.**

---

## Resultado (28-09, medido después de escribir lo de arriba)

```bash
WANDB_MODE=disabled python scripts/train.py --config configs/exp_f13_minirocket{,_s1,_s2}_r3.yaml   # ~2,5 h en CPU, ~3 GB de RAM cada una
WANDB_MODE=disabled python scripts/ensemble_rank.py --config configs/exp_f13_minirocket_seeds3.yaml
python scripts/report_v2_models.py --config configs/report_f13.yaml                    # -> experiments/report-f13/
```

Detección (%) a 2 · 5 · 10 · 20 · 30% de sanos con falsa alarma, promedio de R = 3:

| | PR-AUC | (a′) | AUC auto | AUC celda | 2% | 5% | **10%** | 20% | 30% |
|---|---|---|---|---|---|---|---|---|---|
| P0 · GRU ×3 (referencia) | 0,160 | +0,006 | 0,82 | 0,66 | 14 | 28 | **43** | 61 | 74 |
| **M1 · MiniRocket ×3** | 0,087 | −0,046 | 0,71 | 0,63 | 7 | 16 | **27** | 39 | 53 |
| celda (sin modelo) | — | — | 0,82 | 0,46 | 4 | 18 | **33** | 62 | 62 |

Bootstrap pareado contra P0: −13 puntos al 5% [−22; −3], **−16 al 10% [−26; −8]** y −22 al 20% [−32; −13].

**Veredicto: no pasa, y pierde con evidencia.**
- Queda por debajo de la GRU en todos los presupuestos y por debajo del piso de la celda al 20%.
- Su (a′) es negativo: el orden de los cortes dentro de cada auto *empeora* el promedio del auto.
- **Es inestable entre semillas:** la correlación de rango entre semillas es 0,19–0,21 por fila y
  ~0,6 por vehículo (la GRU da 0,64–0,74 y 0,80–0,89). Los kernels al azar pesan más que la señal.

**Qué se lee.** Con 20 tramos de 15 canales ya estandarizados, 10.000 features de PPV sobre ~12.000
filas de train (~700 positivas) dejan al lineal sin suficiente señal. Las convoluciones aprendidas
de la GRU/CNN-LSTM, con ~3.000 parámetros, usan mejor lo poco que hay. El techo de F12 no era por
aprender las convoluciones.

**No se prueban variantes** (menos features, sin estandarizar por columna, otro α): cada una sería
un brazo nuevo buscado después de ver el resultado. Si se quiere, va a otro preregistro.
