# `AirRegeneration*` y `AirFilter*`: el DPF estaba en `trips` con otro nombre

**Fecha:** 2026-09-16 · **Fase:** F2 (previo) · **Alcance de la medición:** dev
(864 vehículos) · **Reproduce:** `notebooks/eda-exhaustivo-dev.ipynb` §6.4.

Son las 4 columnas que `trips` trae y el diccionario oficial no menciona
([f2-diccionario-trips-incompleto.md](f2-diccionario-trips-incompleto.md)). Nadie
las había mirado: `scripts/eda_raw.py` no las toca y ningún doc de F1 las nombra.

## Lo que está medido

**(a) `AirFilterStart/End` y `signals.Message` comparten el vocabulario completo.**
Los mismos **9 niveles**, ni uno de más ni de menos en ninguno de los dos lados,
incluida la misma errata:

```
Air Filter Normal Operation · Air Filter Full · Air Filter At Limit
Air Filter Over Limit · Air Filter Overloaded · Cleaning Automatically Air Filter
Cleanning Manually Air Filter · Stopped Cleaning Automatically Air Filter
Stopped Cleanning Manually Air Filter
```

**(b) `AirRegenerationEnd` y `signals.Acumulation` son la misma variable.**
Alineando el fin de cada viaje con la señal más cercana dentro de 30 minutos
(`merge_asof`, `direction="nearest"`) sobre los 8 vehículos sonda del cache:

| | |
|---|---|
| pares alineados | 16.324 |
| `AirRegenerationEnd == Acumulation` | **99,80%** |
| correlación de Pearson | **0,9995** |
| `AirFilterEnd == Message` | **99,98%** |

Las dos comparten además la escala: enteros de 0 a 95 en pasos de 5.

**(c) Se comporta como saturación de hollín.** El nivel sube con el uso y se
descarga de golpe: las caídas de más de 5 puntos entre el inicio y el fin de un
viaje dan 3,70 por 1.000 km (mediana en dev) contra 2,10 regeneraciones por 1.000
km registradas en `signals`, y las dos series correlacionan a ρ = 0,69 por
vehículo.

**Conclusión (a)+(b)+(c): `trips` trae una foto al inicio y al final de cada viaje
de las dos variables dinámicas que `signals` trae por mensaje.** Eso es un hecho
medido, no una interpretación.

**(d) Bonus: `signals.Regenerations` está incompleto.** En el vehículo sonda
`VEH_0162` hay 30 filas marcadas, ninguna por encima de los ~7.500 km, mientras el
nivel sigue descargándose hasta el final de sus 10.800 km: 46 caídas de más de 5
puntos contra 30 marcas. No es un vehículo roto: **781 de los 864 vehículos de dev
tienen más caídas que regeneraciones registradas**, y en el agregado dan 3,70
caídas por 1.000 km contra 2,10 regeneraciones marcadas. Como contador de ciclos,
`AirRegeneration*` ve más que `Regenerations`.

## Lo que es inferencia, no hecho

Que esas columnas **sean** el `DieselParticulateFilterStart/End` del anexo 7.1 es
la explicación que mejor encaja, y hay cuatro indicios:

1. ocupan exactamente la posición que el anexo asigna al DPF en `trips`, y es la
   única de las 19 columnas faltantes cuyo contenido aparece con otro nombre;
2. el anexo 7.2 describe `Acumulation` como "acumulación en el filtro de aire [%]",
   que es el mismo concepto que el anexo 7.1 llama "nivel de saturación / carga de
   hollín del DPF [%]";
3. el PDF avisa en §4.2 que las variables vienen "renombradas y, de corresponder,
   anonimizadas";
4. la dinámica es la de un filtro que carga y se regenera.

**Pero no hay confirmación de Ford**, y hay un detalle que no cierra del todo: la
escala llega a **95**, no a 100, lo que sugiere un tope de reporte y no un
porcentaje literal. Tratarlo como "% de saturación del DPF" en el informe es una
afirmación que hay que poder defender; tratarlo como "nivel de acumulación del
filtro, escala 0–95" siempre es correcto.

## Qué desbloquea

`docs/memoria/f1-datos-reales.md` dice "**No hay columna de DPF**". Hay dos, en
las dos tablas. Eso habilita la **familia B** completa del plan §4:

| Feature del plan | Sale de |
|---|---|
| `feat_dpf_end_slope_per_1000km` | `AirRegenerationEnd` vs `OdometerTripEnd` |
| `feat_dpf_end_mean` / `feat_dpf_end_max` | `AirRegenerationEnd` |
| `feat_dpf_positive_delta_frac` | `AirRegenerationEnd − AirRegenerationStart > 0` |
| `feat_regenerations_per_1000km` | caídas de `AirRegeneration` > umbral — mejor cobertura que `Regenerations`, ver (d) |

Y lo hace **a nivel viaje**, que es mejor grano que `signals` para cortar ventanas:
`trips` tiene el odómetro prácticamente completo, mientras que `OdometerValue` de
`signals` es 11% nulo sobre el universo
([f1-calidad-odometro.md](f1-calidad-odometro.md)).

Cuidado con lo obvio: como `AirRegeneration*` y `Acumulation` son la misma
variable, **no son dos features**. Medido sobre dev, `acumulation_mean` y
`air_regen_end_mean` correlacionan a ρ = 0,996. Elegir una fuente —`trips`, por el
odómetro— y no las dos.

Y sigue valiendo la regla 6: es exactamente la variable que el plan señala como
"casi la definición del evento". Sin gap de blanking, memorizarla es la vía más
corta a un PR-AUC alto que no anticipa nada.

La decisión está en [decisiones.md](decisiones.md).
