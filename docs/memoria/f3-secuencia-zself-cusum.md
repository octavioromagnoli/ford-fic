# El vehículo contra sí mismo (`_zself` + CUSUM): se construyó, se midió y no da

**Fecha:** 2026-09-19 · **Fase:** F3 · **Alcance:** dev · **Estado: implementado y
apagado** (`sequence.enabled: false` en los dos configs de panel) · **Reproduce:**

```bash
python scripts/build_dataset.py --config configs/data/panel_delta250.yaml  # con sequence.enabled: true
python scripts/audit_sequence.py --panel data/processed/panel_delta250.parquet
```

Es la idea §2.4 de [`../f3-modelos-candidatos.md`](../f3-modelos-candidatos.md), que era
la mejor apuesta del menú según el EDA. La hipótesis sigue en pie; lo que no alcanza es
la serie de cortes que el panel deja por vehículo. Queda escrito para que nadie la
vuelva a implementar creyendo que es territorio sin explorar.

## Qué se construyó

`src/features/sequence.py`, 7 columnas por panel: un `feat_<x>_zself` por componente
—desvío del corte contra los cortes **anteriores del mismo vehículo**, `expanding()` +
`shift(1)`—, más `feat_degradation_index` (media de los `zself` con el signo físico de
cada uno), `feat_degradation_cusum` (`S_t = max(0, S_{t−1} + idx_t − k)`) y
`feat_degradation_run` (racha por encima de `k`). Componentes del índice:
`idle_per_1000km` y `trips_below_regime_temp_frac` con signo +1, `speed_kmh_mean` y
`coolant_temp_end_mean` con −1, que son las cuatro del perfil alineado al evento
([f2-eda-revision-y-features.md](f2-eda-revision-y-features.md) §3.2).

## Por qué parecía buena idea

El EDA mostró que la señal es **progresiva** (`idle_frac` va de P = 0,58 a 0,74 acercándose
al evento) y que las features que anticipan son las de **ICC bajo** —las que varían dentro
del vehículo, no entre vehículos—. Un modelo de nivel no distingue "vehículo urbano" de
"vehículo que se está degradando"; el desvío contra el propio historial debería separarlos.

## Lo que salió

### 1 · La serie por vehículo es mucho más corta de lo que parecía

Cortes por vehículo **en el panel dev** (no en los viajes crudos, que es donde me había
fijado primero y da 25–35):

| Δ | p10 | p25 | **p50** | p75 | filas con el índice en NaN |
|---|---|---|---|---|---|
| 500 (panel v1) | 2 | 3 | **7** | 17 | 10,5% |
| 250 (panel_delta250) | 4 | 7 | **14** | 35 | 5,4% |

Con `min_history: 3` y mediana 7, la mitad de los vehículos no tiene sobre qué acumular.
**Bajar Δ a 250 duplica la serie** sin inventar información (son ventanas más solapadas
del mismo historial) y mejora los números, pero no alcanza — ver el punto 3.

### 2 · La z clásica está sesgada: hay que usar mediana e IQR

Las features son asimétricas: `idle_per_1000km` tiene **skew 3,47** y el **72% de sus
valores cae por debajo de su propia media**. Entonces `(x − media)/desvío` contra el
propio pasado tiene mediana **−0,34** en vez de 0, y un CUSUM con `k = 0,5` no acumula
nunca salvo por un outlier (solo el 18% de las filas llegaba a `S > 0`).

| variante | z mediana (idle) | índice P(pos>sano) | CUSUM P (k=0) |
|---|---|---|---|
| media/desvío | −0,334 | 0,501 | 0,539 |
| **mediana/IQR (robusta)** | **−0,071** | **0,520** | **0,554** |
| log1p + media/desvío | −0,145 | 0,502 | 0,552 |
| log1p + robusta | −0,087 | 0,513 | 0,555 |

`self_deviation(robust=True)` quedó como **default** del módulo: el sesgo es real y
aplica a cualquier feature futura que se estandarice contra su propio pasado.

### 3 · Con todo arreglado, sigue sin ganarle al nivel

Panel Δ=250, dev (4.098 filas, 175 vehículos, 53 con evento, 507 positivas):

| columna | P(pos > sano) |
|---|---|
| `feat_idle_per_1000km` (**nivel**) | **0,589** |
| `feat_trips_below_regime_temp_frac` (nivel) | 0,577 |
| `feat_speed_kmh_mean` (nivel) | 0,424 |
| `feat_degradation_cusum` | 0,536 |
| `feat_idle_per_1000km_zself` | 0,535 |
| `feat_degradation_index` | 0,526 |
| `feat_degradation_run` | 0,517 |

Ninguna columna de secuencia le gana a la feature de nivel de la que sale, y el `zself`
de idle correlaciona **0,53** con su propia madre: buena parte de lo poco que aporta ya
estaba en el panel.

### 4 · Y lo que mejoraba era el atajo de posición

El CUSUM mostraba un gradiente limpio hacia el evento (mediana 0,21 → 0,31 → 0,56 → 0,75
a medida que el corte se acerca). Es un artefacto: el acumulador crece con la posición en
la serie, y la posición relativa separa sola con **P = 0,829**
([f3-posicion-en-la-serie.md](f3-posicion-en-la-serie.md)). Contra sanos **en el mismo
rango de posición**:

| tramo hasta el evento | CUSUM del que falla | CUSUM del sano comparable |
|---|---|---|
| 0–2k km | 0,746 | **1,676** |
| 2–4k km | 0,558 | **1,903** |
| 4–8k km | 0,313 | **1,881** |
| > 8k km | 0,211 | **1,449** |

Va para el otro lado: contra una referencia comparable, los que fallan acumulan *menos*.
Y dentro de cuartiles de posición la separación del CUSUM oscila entre 0,42 y 0,57 sin
dirección, mientras el nivel de idle se sostiene entre 0,56 y 0,75 en los cuatro.

## Decisión

**Apagada** (`sequence.enabled: false` en `panel_v1.yaml` y `panel_delta250.yaml`). Son 7
columnas con P ≈ 0,52–0,54 compitiendo por 507 filas positivas, una de ellas con un atajo
de construcción adentro. El módulo, el bloque del YAML y los 5 chequeos de
`scripts/check_setup.py` quedan en el repo: reactivarla es cambiar una clave.

## Qué quedó, que vale más que la feature

1. **`scripts/audit_sequence.py`** y el protocolo de auditoría por estratos de posición,
   que ahora se le aplica a cualquier feature acumulativa.
2. **`self_deviation(robust=True)`**: si alguna vez se estandariza contra el propio
   pasado, se hace con mediana e IQR.
3. **`configs/data/panel_delta250.yaml`**: mismo universo, mismo holdout, misma terna
   salvo Δ; dev 4.098 filas / 175 vehículos / 53 eventos / 507 positivas, tasa 0,1237
   (contra 2.029 / 171 / 53 / 254, tasa 0,1252 del v1). **No agrega eventos**, que es lo
   que escasea, así que no es obviamente mejor: queda como panel alternativo para que lo
   decida una corrida de modelo, no una intuición.

## Lo que NO invalida

La hipótesis del EDA sigue viva: la señal *es* progresiva y *es* intra-vehículo. Lo que
falla es medirla con el desvío contra el propio pasado en una serie de 7 a 14 puntos. Una
implementación distinta de la misma idea —por ejemplo, la pendiente dentro de la ventana,
que ya existe como `*_trend` y no depende de la serie de cortes— no queda descartada por
este resultado.
