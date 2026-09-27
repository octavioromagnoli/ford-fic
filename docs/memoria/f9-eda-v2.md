# EDA de la entrega v2: qué se sostiene, qué cambió y qué es nuevo

**Fecha:** 2026-09-26 · **Fase:** F9 · **Alcance:** dev de la entrega v2 (446 vehículos, 141
eventos; 139 hasta el fin de extracción). Test sin tocar. **Reproduce:**

```bash
python scripts/build_eda_cache.py --config configs/data/eda_cache_v2.yaml   # cache dev-only
python scripts/eda_gaps.py --config configs/data/eda_cache_v2.yaml          # la EDA del 18-09, mismo código
python scripts/eda_v2.py --config configs/eda_v2.yaml                       # lo de abajo -> experiments/eda_v2/dev/v2/
```

La versión de la entrega 1 es [f2-eda-revision-y-features.md](f2-eda-revision-y-features.md)
(290 vehículos, 60 eventos), y su cache sigue en `experiments/eda/dev/`
(`configs/data/eda_cache.yaml`, que ahora lee los datos congelados de v1).

## 0 · Lo que cambia en cómo se lee cualquier comparación

**En v2 una comparación fallado/sano agrupada mezcla mercado con señal.** La entrega 1 eran dos
mercados parecidos. Ahora son cinco, con tasas de falla por auto de 0% (PER) a 48% (COL), y esas
tasas están en el orden inverso de cuántos fallados sin fecha sacó Ford de cada mercado
([f9-universo-v2.md](f9-universo-v2.md) §3). Todo lo de abajo compara contra sanos **del mismo
mercado**, y en el perfil alineado también del mismo tramo de odómetro (o del mismo mes). La
versión agrupada va al lado para ver cuánto era mercado.

El caso extremo es el orden de autos que aprende un modelo sobre el panel v2
([f9-remedicion-v2.md](f9-remedicion-v2.md)): AUC por vehículo 0,75 agrupado, 0,64 dentro del
mercado y **0,60 dentro de mercado × motor**.

## 1 · La tabla corta

| hallazgo de la entrega 1 | en v2 | |
|---|---|---|
| El marcador `Regenerations` se corta el 25-05-2026 | último marcador p50 19/20-05-2026 en las dos cohortes | **se sostiene** |
| 35% de los viajes son idle de 0 km; `KilometerPerHour` es nulo exactamente ahí | 33,4%; nulo en el 100% de los 0 km y en el 0,02% del resto | **se sostiene** |
| Escala de `AirRegeneration` 0–100, en pasos de 5 | igual | **se sostiene** |
| El evento se ordena por días, no por km | sd(log) edad 0,38 contra odómetro 0,91 (dif. 0,53 [0,43; 0,64]) | **se sostiene**, menos nítido (v1: 0,24 contra 0,97) |
| Después del evento el uso cambia | bajo régimen baja en el 72% de los autos (v1: 81%) y la velocidad sube en el 79% | **se sostiene**, más débil |
| Confusor calendario: la exposición previa al evento cae antes que la de los sanos | km de fallados previos al evento: pico en sep-2025–ene-2026; los de sanos, parejos en 2026 | **se sostiene**: el emparejado por mes sigue haciendo falta |
| **Ventana del registro** 01-09-2025 → 11-03-2026 | **no existe**: el mes calendario no agrega nada (LR p = 0,16, §C) | **ya no pasa** |
| El riesgo crece con los días desde la venta: 1,5 → 6,5 → 16–18 por 100 autos-mes | sube hasta los ~4 meses y después es **plano**, ~3,3 por 100 autos-mes (§B) | **cambió**: el crecimiento era la ventana |
| `ProductionDay`: ρ = −0,435, exposición | ρ = −0,16; dentro del mercado AUC 0,37 (0,63 invertido); tasa por quintil 0,36 → 0,16 | **cambió de lectura**: era en parte selección (universo §2), y lo que queda sigue siendo un atajo → `aux_` |
| La señal que anticipa es térmica y de uso: idle, bajo régimen, lento, frío (P 0,58 → 0,74) | idle y bajo régimen sí (§E); **velocidad y bajo régimen entre los que se mueven, no** | **cambió en parte** |
| "Viajes bajo régimen" es el rasgo más robusto (0,64–0,68, también en la fuente externa) | entre los que se mueven, 0,51–0,53 dentro del mercado cerca del evento; 0,57 en los primeros 60 días | **ya no pasa** cerca del evento |
| La velocidad baja al acercarse el evento (0,42 → 0,28) | 0,47–0,51 dentro del mercado | **ya no pasa**: en v1 era COL (lento, falla) contra CHL |
| Los niveles del DPF van al revés (0,31–0,35) | 0,40–0,45 dentro del mercado: levemente al revés | **se sostiene**, más débil |
| **No hay trayectoria previa al evento**: el exceso es mayor lejos y se achica cerca | el exceso **crece** hacia el evento: idle 0,60 → 0,64 → 0,76, también con la referencia −21 d | **cambió**: hay trayectoria (§E) |
| Rasgo temprano post-venta: AUC 0,62–0,68 a 30–60 días | 0,60–0,64 dentro del mercado a 60 y 90 días, con 125 fallados (v1: ~42) | **se sostiene**, con 3× los eventos |
| Engine fuera: ENG_3 es 0% de los fallados | ENG_3: 35 eventos en dev (34 en BRA); dentro de BRA falla 2,5× más que ENG_2 | **ya no pasa** |
| No hay columna de altura | la ciudad de venta da una altura: tendencia en COL, no significativa dentro del motor (§I) | **nuevo** |
| La fecha del evento es la intervención | la fecha v2 cae ~2 semanas después ([f9-entrega-v2.md](f9-entrega-v2.md) §6) | **cambió**: la referencia pasa a fecha − 21 d |

## A · Composición del dev

| mercado | autos | eventos | | motor en el mercado (eventos/autos) |
|---|---|---|---|---|
| ARG | 93 | 8 (9%) | | ENG_1 5/29 · ENG_2 2/33 · ENG_3 1/31 |
| BRA | 136 | 41 (30%) | | ENG_2 7/41 · **ENG_3 34/95** |
| CHL | 83 | 36 (43%) | | ENG_1 3/24 · **ENG_2 33/51** · ENG_3 0/8 |
| COL | 113 | 54 (48%) | | ENG_1 0/16 · **ENG_2 54/85** · ENG_3 0/12 |
| PER | 21 | 0 | | — |

Modelo × motor: MODEL_2 es solo ENG_2 (62/135 fallan). MODEL_1 es ENG_2 (34/89) o ENG_3
(1/23). MODEL_3 y MODEL_4 son ENG_1 o ENG_3. El cruce ya no es diagonal pero sigue siendo
estrecho: modelo y motor cuentan casi la misma historia. 11 de 141 fallados tienen un segundo
evento registrado.

## B · Riesgo por días desde la venta (Poisson con exposición)

Un registro por auto-día desde la venta hasta el evento o el fin de extracción (14-09-2026).

| días desde la venta | 0–60 | 60–120 | 120–180 | 180–240 | 240–300 | 300–360 | 360–450 | 450–720 |
|---|---|---|---|---|---|---|---|---|
| eventos / 100 autos-mes | 0,34 | 2,11 | 3,37 | 3,22 | 3,91 | 3,11 | 3,10 | 5,91 |

Ajustando por días desde la venta, mercado y motor (RR contra COL y ENG_2): ARG 0,12
[0,06; 0,26], BRA 0,43 [0,27; 0,68], CHL 0,75 [0,49; 1,14], ENG_1 0,25 [0,12; 0,52], ENG_3
0,64 [0,42; 0,99]. Dentro de BRA, ENG_3 da 3,15 eventos por 100 autos-mes contra 1,24 de ENG_2.

## C · Ya no hay ventana del registro

El mismo Poisson con el mes calendario como factor, sobre días desde la venta, mercado y motor:
**LR = 20,3, 15 gl, p = 0,16**. No hay meses sin registro: los eventos de dev van de mayo de
2025 a septiembre de 2026, con 6–20 por mes desde septiembre de 2025. La ventana de la
entrega 1 (01-09-2025 → 11-03-2026) era de la extracción vieja, no del registro.

- **Consecuencia:** la regla "la censura de un sano es el fin de su exposición dentro de la
  ventana" no aplica a v2. La censura es el fin de extracción.
- **Lo que sí marca es la cohorte de venta.** El trimestre de venta agrega (LR = 16,6, 5 gl,
  p = 0,005): los vendidos en 2025Q1 fallan 2,1× [1,15; 3,78] más que los de 2025Q2 a igual
  edad, y los de 2025Q4 0,47× [0,23; 0,97]. Es el mismo gradiente de producción del universo.
  Las estáticas de fecha (`ProductionDay`, `daysUntilSale`) siguen siendo `aux_`.

## D · El reloj del evento

135 fallados con historial desde menos de 300 km y el evento dentro de la telemetría:

| escala | sd(log) | mediana |
|---|---|---|
| edad (días desde producción) | **0,38** | 314 d |
| días desde la venta | 0,56 | 231 d (IQR 152–310) |
| odómetro | 0,91 | 14.099 km |

El odómetro dispersa más que la edad, con una diferencia de 0,53 [0,43; 0,64] (bootstrap). El
evento sigue ordenándose por el tiempo y no por los km. ρ(km/día, km al evento) = 0,89: quien
maneja más llega al evento con más km. ρ(km/día, días desde la venta al evento) = 0,30, contra
≈ 0 en la entrega 1.

## E · Perfil alineado al evento, dentro del mercado

P(fallado > sano) de cada agregado por tramo de km antes de la referencia, contra sanos **del
mismo mercado y del mismo tramo de odómetro** (bins de 2.000 km). Entre corchetes, IC90 por
bootstrap sobre los fallados (103–132 por tramo).

**Referencia = fecha registrada:**

| feature | 4–8k antes | 2–4k | 0–2k | últimos 1k | después (0–3k) |
|---|---|---|---|---|---|
| idle (fracción de filas de 0 km) | 0,60 [0,55; 0,65] | 0,64 | **0,76** [0,73; 0,79] | **0,78** | 0,62 |
| idle por 1.000 km | 0,54 | 0,59 | 0,73 | 0,75 | 0,63 |
| idle largo (≥ 15 min) por 1.000 km | 0,58 | 0,57 | 0,68 | 0,66 | 0,58 |
| bajo régimen (todas las filas) | 0,55 | 0,62 | **0,73** [0,69; 0,76] | 0,73 | 0,62 |
| bajo régimen entre los que se mueven | 0,51 | 0,52 | 0,53 | 0,51 | 0,58 |
| refrigerante al terminar | 0,47 | 0,39 | **0,29** [0,26; 0,33] | 0,27 | 0,38 |
| velocidad (entre los que se mueven) | 0,51 | 0,47 | 0,49 | 0,50 | 0,46 |
| viajes cortos (entre los que se mueven) | 0,46 | 0,47 | 0,47 | 0,47 | 0,50 |
| nivel del DPF al terminar | 0,45 | 0,43 | 0,40 | 0,39 | 0,43 |
| regeneraciones por 1.000 km | 0,45 | 0,40 | 0,41 | 0,41 | 0,50 |
| arranques en frío | 0,40 | 0,44 | 0,42 | 0,41 | 0,44 |
| temperatura ambiente (control) | 0,45 | 0,42 | 0,38 | 0,37 | 0,36 |

**Con referencia = fecha − 21 d**, el pico se corre a "después" (idle 0,73, bajo régimen 0,73,
refrigerante 0,30) y el tramo 0–2k antes queda en 0,70 / 0,71 / 0,34. Lejos del evento (4–8k,
~3–5 meses antes) la señal es débil: idle 0,60, bajo régimen 0,52.

Cuatro lecturas:
1. **Hay trayectoria.** Idle y bajo régimen crecen en los últimos ~4.000 km (~2 meses), y con la
   referencia corrida 3 semanas siguen creciendo. En la entrega 1 la conclusión era la opuesta,
   medida con 42–55 fallados de dos mercados.
2. **Lo que crece es estar parado con el motor en marcha y no llegar a temperatura**, no manejar
   distinto. Velocidad, viajes cortos y bajo régimen entre los que se mueven quedan en ~0,5
   dentro del mercado. La velocidad, que en v1 era la señal más fuerte, era el contraste COL
   (lento, falla) contra CHL.
3. **La temperatura ambiente de los fallados es más baja que la de sanos del mismo mercado y
   tramo** (0,37–0,45). Pasa también contra el mismo mes (0,42–0,45), así que no es estación sino
   lugar, y en un mismo mercado lugar es altura (Bogotá contra la costa; §I). Una parte del
   "bajo régimen" puede ser clima de la ciudad.
4. **La versión agrupada infla lo que es mercado** (tramo 0–2k, dentro del mercado → agrupado):
   filtro anormal 0,52 → 0,58, DPF saturado 0,52 → 0,59, duración del viaje 0,56 → 0,61,
   temperatura máxima 0,52 → 0,60.

## F · Rasgo temprano post-venta

AUC por vehículo con las features de los primeros 60 días después de la venta (≥ 15 viajes).
Fallados con el evento después de los días 90; sanos seguidos al menos 90 días (125 fallados,
298 sanos). Se mide dentro del mercado y dentro de mercado × trimestre de venta:

| feature | agrupado | dentro del mercado | mercado × trim. venta |
|---|---|---|---|
| idle | 0,657 | **0,636** | 0,635 |
| idle por 1.000 km | 0,588 | 0,607 | 0,592 |
| idle largo por 1.000 km | 0,639 | 0,599 | 0,602 |
| bajo régimen | 0,585 | 0,594 | 0,579 |
| encadenados (< 30 min del anterior) | 0,535 | 0,580 | 0,585 |
| regeneraciones por 1.000 km | 0,417 | 0,382 (0,62 invertido) | 0,373 |
| consumo | 0,429 | 0,396 (0,60 invertido) | 0,378 |

A los 90 días da lo mismo (idle 0,632; idle largo 0,637). **El rasgo temprano de la entrega 1
se replica con 3× los eventos y en cinco mercados**, en el mismo rango de AUC (~0,60–0,64). Es
señal de *qué auto*, disponible desde el primer o segundo mes.

## G · Estáticas de fecha dentro del universo

Dentro de la ventana de producción, ρ(evento) es −0,16 para `ProductionDay`, −0,09 para
`daysUntilSale` y −0,18 para la fecha de venta. La tasa por quintil de `ProductionDay` es 0,36 /
0,43 / 0,37 / 0,26 / 0,16. Dentro del mercado, `ProductionDay` sola ordena con AUC 0,63
(invertida). Es menos que en v1, y es un atajo de cohorte: sigue como `aux_`.

## H · Alrededor de la fecha registrada

Idle largo (≥ 15 min a 0 km) por auto-día, contra la base de cada fecha (días −120 a −60):

| respecto de la fecha | −21…−14 | −14…−7 | −7…−3 | −3…0 | 0…+3 | +3…+7 |
|---|---|---|---|---|---|---|
| fecha **v1** (58 autos de dev) | 1,57× | 1,09× | 1,18× | 1,50× | **2,24×** | 1,74× |
| fecha **v2** (139 autos) | 1,07× | **1,49×** | **1,50×** | **1,79×** | 0,95× | 0,95× |

El auto parado con el motor en marcha es la regeneración forzada del taller. Aparece justo
después de la fecha v1 y en las dos semanas previas a la v2. El día de la fecha v2 el auto se usa
más que nunca (81% de los días con viajes entre −3 y 0, contra 69% de base).

## I · Altura de la ciudad de venta

`configs/data/city_elevation.yaml` (GeoNames vía Open-Meteo, `scripts/build_city_elevation.py`):
151 de 153 ciudades con altura. CAPITAL (ARG, 14 autos) es ambigua y Punta Arenas trae 9999 de
GeoNames, así que quedan sin dato. Hay 22 autos sin ciudad. La variación útil está en COL: Bogotá
y Chía a ~2.580 m y la costa a nivel del mar. CHL es casi todo Santiago (556 m).

| Colombia, altura de la ciudad | autos-mes | eventos | por 100 autos-mes |
|---|---|---|---|
| < 1.000 m | 178 | 3 | 1,7 |
| 1.000–2.000 m | 221 | 11 | 5,0 |
| ≥ 2.000 m | 631 | 40 | 6,3 |

El gradiente crudo es grande, pero en COL solo falla ENG_2 y la mezcla de motores cambia con la
ciudad. Ajustando:

| modelo de Poisson | RR por cada 1.000 m | |
|---|---|---|
| todos los mercados, con días desde la venta + mercado + motor | 1,30 [0,93; 1,81] | |
| solo COL y solo ENG_2 | 1,21 [0,82; 1,78] | |

**La dirección es la que predice la física, pero con estos autos no se separa del cero.** La
altura entre países no se puede usar: la tasa por mercado está cruzada con el muestreo. Si
entra, es como `aux_static_elevation_m` con ablación, y leída dentro del mercado.

## J · La misma EDA sin la ventana de producción (variante, no vigente)

El 26-09 se probó sacar la ventana (universo de 990, dev de 788; se volvió a la ventana, ver
[f9-universo-v2.md](f9-universo-v2.md) §6). La misma EDA sobre ese dev:

```bash
python scripts/build_eda_cache.py --config configs/data/eda_cache_v2_sinventana.yaml
python scripts/eda_v2.py --config configs/eda_v2_sinventana.yaml   # -> experiments/eda_v2_sinventana/dev/v2/
```

| | con la ventana (dev 446) | sin la ventana (dev 788) |
|---|---|---|
| mes calendario, LR | p = 0,16 | **p = 0,0003** |
| trimestre de venta, LR | p = 0,005 | **p = 5·10⁻¹⁰** |
| RR del tramo 450–720 d desde la venta | 24 | **51** |
| AUC de −`ProductionDay` sola, dentro del mercado | 0,63 | **0,87** |
| RR de ENG_1 | 0,25 | 0,71 |
| idle a 60 d, AUC dentro del mercado | 0,636 | **0,563** |
| ídem, dentro de mercado × trimestre de venta | 0,635 | 0,620 |

Por período de producción, en eventos por 100 autos-mes: 7,55 antes de la ventana (71 fallados,
3 sanos con el evento después del fin de extracción), 2,71 adentro y **0,00 después** (268 sanos).

**Todo lo que aparece de nuevo es la selección de la lista, no física.** Vuelve un "efecto
calendario" que dentro de la ventana no existe, porque los autos viejos son todos fallados y los
nuevos todos sanos. El riesgo tardío se duplica por la misma razón. El rasgo temprano se diluye
hasta que se compara dentro del trimestre de venta, o sea, hasta que se vuelve a poner la ventana
a mano.

## Lo que queda abierto

- **El "idle cerca del evento" es anticipación o síntoma.** Con la referencia −21 d sigue
  creciendo, pero la distinción depende de dónde está la intervención real de cada auto, que en
  v2 no se sabe. La curva de anticipación se tiene que reportar con el corrimiento.
- **Si la trayectoria (§E.1) se sostiene dentro de mercado × motor.** Es la pregunta que
  reabre el *cuándo*.
- **La altura**, con los autos de BRA/ARG/PER donde la ciudad varía, si Ford aclara el muestreo
  por mercado.
