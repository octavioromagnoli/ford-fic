# El barrido de relaciones, su nulo, y por qué la ingeniería de features tocó techo

**Fecha:** 2026-09-19 · **Fase:** F3

> ⚠️ Provisorio: los eventos tienen una falla conocida (error de SQL en la query) y la
> mentora los va a corregir. Todo lo que se compare contra `label` se re-mide después.

Reproduce:

```bash
python scripts/build_dataset.py --config configs/data/panel_v5.yaml
python scripts/audit_sequence.py --panel data/processed/panel_v5.parquet --columns all
```

---

## 1 · Primero el nulo, porque sin él cualquier búsqueda "encuentra" algo

Se probaron **los 7.310 cocientes ordenados** entre las 86 features del panel v4, y
después se repitió todo con la etiqueta **permutada a nivel vehículo** (40 veces). El
segundo número es el que importa:

| | mejor \|0,5 − P\| |
|---|---|
| mejor cociente **real** | 0,125 |
| mejor cociente con la etiqueta **permutada** (mediana de 40 corridas) | **0,213** |
| ídem, p95 | **0,278** |

**Cero cocientes de 7.310 superan la barra.** Con 53 vehículos con evento, una búsqueda
de este tamaño produce por azar separaciones bastante mejores que la mejor separación
real. Cualquier feature elegida barriendo combinaciones de lo que ya está en el panel es,
hasta prueba en contrario, ruido con buena presentación.

Los 30 mejores cocientes son además **todos la misma cosa**: `idle` dividido por algo, o
algo dividido por `idle`. No son 30 hallazgos, es uno.

## 2 · Corrección sobre `regen_per_idle_min`

En la vuelta anterior se reportó como "la primera feature que le gana al panel". Con el
nulo en la mano, hay que precisar dos cosas:

1. **Como hipótesis previa, se sostiene.** No salió de barrer: se construyó desde el
   mecanismo (el idle como denominador del trabajo del postratamiento) *antes* del
   barrido, así que el test correcto es el de esa feature sola. Permutación a nivel
   vehículo, 2.000 réplicas: **p = 0,0005** (`dpf_load_per_idle_min`, ídem). Para
   comparar, `feat_idle_frac` da p = 0,022.
2. **Pero no es un mecanismo nuevo.** Correlaciona **ρ = −0,88** con
   `feat_idle_min_per_1000km`. Es la variable de idle afilada, no una dimensión
   distinta. Su |0,5 − P| = 0,130 contra 0,097 del idle solo: la mejora es real y es chica.

La diferencia entre las dos lecturas es la que separa "lo predije y lo testeé" de "lo
encontré buscando". La primera vale; la segunda, con esta cantidad de positivos, no.

## 3 · Familia H: las rarezas tampoco

Si combinar lo que ya está, está agotado, lo que falta es información que **nunca entró**
al panel. Se construyeron 17 features de cosas que ninguna columna miraba —el reloj, el
viaje anterior, la forma de la distribución— y se testeó cada una por permutación:

| rareza | qué mide | P | p (permutación) |
|---|---|---|---|
| `temp_over_ambient_median` | cuánto sube el motor **sobre el ambiente** (controla la estación sin meter calendario) | 0,582 | 0,16 |
| `night_idle_min_per_1000km` | minutos de motor encendido sin moverse **entre 22 y 5 h** | 0,567 | 0,12 |
| `km_gini` | desigualdad del reparto de km: ¿un viaje largo o cien saltos? | 0,565 | 0,22 |
| `engine_temp_max_cv` | dispersión del régimen térmico entre viajes | 0,559 | 0,32 |
| `cooling_rate_median` | °C/min que pierde el motor parado: la constante térmica del **vehículo**, no del conductor | 0,482 | 0,64 |
| `hot_restart_frac` | rearranques con el motor todavía caliente (uso de reparto) | 0,478 | 0,62 |
| `regen_overdue_ratio` | ¿está atrasada la próxima regeneración **respecto de su propio ritmo**? | 0,498 | 0,91 |
| `km_top_trip_share` | qué fracción de los km se la lleva el viaje más largo | 0,495 | 0,90 |

**Ninguna pasa**, y con 17 tests simultáneos un p de 0,12 no es nada. Las dos que más
prometían conceptualmente —la constante de enfriamiento, que mide la pieza y no el uso, y
el "atraso" de la regeneración contra el ritmo propio del vehículo— dan exactamente 0,5.

## 4 · La conclusión, que es la útil

Sumando las cuatro vueltas de F3 —secuencia (E), físicas del menú (candidatas 1–6),
relaciones y literatura de DPF (F y G), y rarezas (H)— se construyeron y midieron **más
de 60 features nuevas** por seis mecanismos distintos, más un barrido exhaustivo de 7.310
combinaciones. **Todo lo que separa es una sola dimensión: tiempo de motor encendido
improductivo y frío.** El resto está plano.

Eso no es un fracaso de la ingeniería de features, es un resultado sobre los datos: con
la etiqueta actual, el panel tiene **una** dimensión de señal, y ya está medida. Seguir
agregando features es comprar más comparaciones múltiples con los mismos 53 vehículos.

**Qué hacer en cambio:**

1. **Esperar los eventos corregidos.** Es lo único que puede cambiar el diagnóstico, y la
   maquinaria para re-medir todo está lista: reconstruir el panel y volver a correr las
   auditorías son dos comandos.
2. **Ojo con el universo.** Los 364 vehículos y el holdout congelado salieron de
   descartar los que no tenían `IdentificationDate` utilizable
   ([f2-universo-fecha-usable.md](f2-universo-fecha-usable.md)). Si la corrección del SQL
   rehabilita positivos, el universo crece y **el test congelado hay que re-sortearlo**.
   Conviene preguntarlo antes de reconstruir nada.
3. **Track B, no Track A.** Lo que queda sin explotar no son features sino el hallazgo de
   [f3-esfuerzo-de-control-y-dosis.md](f3-esfuerzo-de-control-y-dosis.md) §2b: dentro de
   estratos de posición del corte, la señal térmica separa con 0,74 y no con 0,59. La
   dilución es del diseño de la medición, no de las features.
