# Posición y calendario no se pueden emparejar a la vez, y el calendario es explotable

**Fecha:** 2026-09-19 · **Fase:** F3

> ⚠️ Provisorio: los eventos tienen una falla conocida (error de SQL) y la mentora los va
> a corregir.

Reproduce:

```bash
python scripts/build_dataset.py --config configs/data/panel_delta250_posmatch.yaml
python scripts/train.py --config configs/exp_logistic_l1_d250pos.yaml
python scripts/train.py --config configs/exp_l1_d250pos_auxcal.yaml   # la ablación de calendario
python scripts/train.py --config configs/exp_l1_v1_auxcal.yaml        # el control sobre el panel v1
```

---

## 1 · Δ = 250 no resuelve el problema de posición, porque el problema no es de resolución

La hipótesis era: a Δ = 500 un vehículo tiene 7 cortes de mediana, así que "último cuarto
de la serie" son dos filas y el pool sano se agota; con Δ = 250 la serie se duplica y las
celdas se pueblan. **Falso.** A Δ = 250, emparejando por (odómetro × mes × posición), el
resultado sigue siendo **0 filas sanas** en todas las grillas probadas, de 2.000 a 8.000
km de celda y de 4 a 8 bins de posición.

El diagnóstico, sobre el pool completo de 14.536 filas sanas:

- las positivas ocupan **70 celdas**, y **solo 36 tienen algún sano**;
- apenas el **56%** de la masa de positivas cae en una celda con al menos un sano;
- las celdas más pesadas tienen 0, 1 o 3 sanos disponibles contra el 9% de las positivas.

La razón es estructural, no de granularidad: **un sano no puede estar a la vez en el mismo
odómetro, el mismo mes y el final de su serie.** La serie de un vehículo con evento termina
en `E − G` (temprano, y a pocos km); la de un sano termina en `last_odo − G − H`, o sea
tarde en el calendario y con más kilómetros. "Estar al final de la propia serie" significa
cosas distintas en cada grupo, y las tres dimensiones quedan acopladas.

## 2 · El trade-off que sí existe: o posición o calendario

Midiendo la distancia de variación total entre positivas y sanas en cada dimensión
(Δ = 250, dev):

| emparejado | filas sanas | vehículos | TV posición | TV mes |
|---|---|---|---|---|
| `cut_odo + cut_date` (el actual) | 2.583 | 150 | **0,697** | **0,176** |
| `cut_odo + cut_position` | 2.472 | 241 | **0,278** | **0,615** |
| los tres (8.000 km, trimestre, mitades) | 724 | 127 | 0,640 | 0,354 |

Cerrar una dimensión abre la otra, y quedarse en el medio no cierra ninguna. El panel
`configs/data/panel_delta250_posmatch.yaml` elige cerrar posición, **a propósito y para
auditarlo**, no como candidato a panel canónico.

## 3 · La ablación de calendario, que es el resultado importante

Con el knob nuevo (`features.extra_prefixes: [aux_]`) se le dan al modelo las columnas que
están fuera del set base justamente porque miden calendario: temperatura ambiente,
`ProductionDay`, `daysUntilSale`, el marcador `Regenerations` cortado en 2026.

| panel | solo `feat_` + `static_` | + `aux_` | salto |
|---|---|---|---|
| v1 (empareja odómetro y mes) | ROC 0,617 | ROC **0,719** | +0,10 |
| Δ=250 + posición (no empareja mes) | ROC 0,670 | ROC **0,922** | **+0,25** |

Dos conclusiones, y la segunda es incómoda:

1. **El 0,670 del panel Δ=250 no se puede atribuir al emparejado por posición.** Se ve
   mejor que el 0,617 del v1, pero en ese panel el calendario quedó abierto de par en par
   —ROC 0,92 cuando se lo dejan usar—, y las `feat_` con deriva temporal arrastran parte de
   eso. Es exactamente el caso de la regla 6 de `CLAUDE.md`: un número que mejora se audita
   antes de celebrarse. Este no pasa la auditoría.
2. **El panel v1 también se deja ganar 0,10 con las `aux_`** (0,719 contra 0,617).

> **Corrección (más tarde, el 19-09):** la ablación fina por familia muestra que ese salto
> **no es calendario**. La temperatura ambiente aporta +0,007 en el v1 y el marcador
> `Regenerations`, 0,000: el emparejado por mes hace su trabajo. Lo que filtra son las
> **estáticas** (`Engine` y compañía, el sesgo de muestreo de F1), +0,093. En el panel
> Δ=250+posición sí hay calendario, pero entra por el **marcador cortado en 2026** (+0,072),
> no por la temperatura. Detalle en
> [f3-evento-ficticio-y-ventana-de-riesgo.md](f3-evento-ficticio-y-ventana-de-riesgo.md) §4.

## 4 · Por dónde sí se cierra

El emparejado por celdas no puede arreglar esto porque intenta corregir a posteriori una
asimetría que se genera **al construir las series**. La forma correcta es de diseño:
**asignarle a cada vehículo sano un evento ficticio** (un `pseudo_event_odo` sorteado de la
distribución de los positivos) y **cortarle la serie ahí**, igual que a un positivo. Con
eso, "el final de la serie" pasa a significar lo mismo en los dos grupos por construcción,
la posición deja de separar, y el emparejado por odómetro y mes puede seguir haciendo su
trabajo sin competir con una tercera dimensión. Es la solución estándar del sesgo de
tiempo inmortal (asignación de fecha índice / *risk-set sampling*), y se implementa en
`src/data/panel.py` sin tocar features ni modelos.

Es la única de las tres cosas que probamos (Δ más fino, celdas más gruesas, tercera
dimensión de emparejado) que ataca la causa en vez del síntoma.

> **Implementado el 19-09, y funcionó** —con una pieza más de la que preveía esta sección:
> el evento ficticio solo no alcanza, porque dentro de un vehículo la etiqueta *es* la
> posición. Hace falta además quedarse con la ventana de riesgo de cada vehículo
> (`label.window_only`). Con las dos, la posición cae de P = 0,84 a 0,49 y los tres ejes
> cierran a la vez. Ver
> [f3-evento-ficticio-y-ventana-de-riesgo.md](f3-evento-ficticio-y-ventana-de-riesgo.md).

## 5 · Qué quedó

- `configs/data/panel_delta250_nomatch.yaml` y `panel_delta250_posmatch.yaml`, más
  `panel_nomatch.yaml` a Δ=500: los pools sin emparejar sirven para simular grillas sin
  releer 1,2 GB.
- `configs/exp_l1_v1_auxcal.yaml` y `exp_l1_d250pos_auxcal.yaml`: **la ablación de
  calendario es ahora un chequeo estándar.** Cualquier panel nuevo la pasa antes de que se
  le crea un número.
- El panel canónico **no cambia**: sigue siendo el v1, emparejado por odómetro y mes.
