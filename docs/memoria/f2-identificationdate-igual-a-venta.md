# `IdentificationDate` es la fecha de venta en el 79% de los positivos

**Fecha:** 2026-09-16 · **Fase:** F2 (previo) · **Alcance de la medición:** dev
(864 vehículos, 292 con evento) · **Reproduce:**
`notebooks/eda-exhaustivo-dev.ipynb` §3.3 y §3.4.

> **Esto bloquea F2.** No es un detalle de calidad: sin la posición del evento
> sobre el eje de km no hay `time_to_event_km`, no hay etiqueta por punto de corte
> y no tiene sentido elegir W, G, H ni Δ.

## El hecho

Sobre los 292 vehículos con evento de dev:

| | n | % |
|---|---|---|
| `IdentificationDate == daysUntilSale` (exacto) | **231** | **79,1%** |
| `IdentificationDate > daysUntilSale` | 61 | 20,9% |
| `IdentificationDate < daysUntilSale` | **0** | 0% |

Que dos columnas coincidan **exactamente** en 231 filas no es física. Y el orden
nunca se invierte: no hay un solo evento anterior a la venta.

## Por qué importa: la flota no se mueve hasta que se vende

El odómetro mediano de la flota contra días desde producción (dev):

| día desde producción | 20 | 40 | 60 | 80 | 100 | 120 | 200 | 300 |
|---|---|---|---|---|---|---|---|---|
| odómetro mediano [km] | 4 | 7 | 10 | 15 | 20 | 300+ | ~5.900 | ~12.000 |

Hasta el día ~80-100 la mediana no pasa de 20 km: el vehículo está en el
concesionario (`daysUntilSale` tiene mediana 86 días). Recién después despega.

## Qué pasa al proyectar el evento al eje de km

El anclaje de F1 ([f1-anclaje-temporal.md](f1-anclaje-temporal.md)) **se sostiene
en dev**: `primer_viaje − ProductionDay` da std 0,76 d, **IQR 0,00 d**, rango 12 d
(F1 midió 0,70 / 0,00 / 12 sobre los 1094). El puente funciona. El problema es lo
que se manda por él.

Aplicando la fórmula literal del anexo 7.3 —`IdentificationDate` son días desde
**producción**—, e interpolando esa fecha sobre los viajes del vehículo:

| grupo | n | odómetro del evento (mediana) | % del historial de viajes ANTES del evento |
|---|---|---|---|
| `Ident == daysUntilSale` | 231 | **13 km** | **2,3%** |
| `Ident > daysUntilSale` | 61 | 7.388 km | 39,0% |

**Para el 79% de los positivos el evento queda con el odómetro casi en cero y sin
historial por delante.** No hay ventana W que agregar ni gap G que blanquear: la
fila no se puede etiquetar. Los 61 restantes sí dan un evento utilizable.

## La lectura alternativa (hipótesis, no hallazgo)

Si `IdentificationDate` se contara desde la **venta** en vez de desde producción
(`fecha = origen + ProductionDay + daysUntilSale + IdentificationDate`), los
números dejan de ser degenerados:

| grupo | odómetro del evento (mediana) | % del historial antes |
|---|---|---|
| `Ident == daysUntilSale` | 2.886 km | 19,6% |
| `Ident > daysUntilSale` | 12.811 km | 56,7% |

**Esto es una hipótesis abierta, no una conclusión.** El anexo 7.3 dice "días desde
producción" sin ambigüedad, y elegir la lectura porque los números quedan más
lindos es exactamente la clase de decisión contra la que existe el holdout. Las dos
lecturas están materializadas en el cache (`event_odo_km` y `event_alt_odo_km`) para
que la comparación se haga con datos y no de memoria.

## El otro problema: `daysUntilSale` está en el set base

`features.static_columns` de `configs/data/panel_v1.yaml` incluye `daysUntilSale` y
`ProductionDay`. Medido sobre dev, su ρ de Spearman con `event_observed` es
**−0,249** y **−0,237** — entre las cinco variables más correlacionadas de todo el
EDA. La tasa de eventos por quintil de `ProductionDay` va de 0,49 a **0,10**.

Con el hecho de arriba, `daysUntilSale` deja de ser una covariable sospechosa:
**para el 79% de los positivos es numéricamente igual a la columna que define la
etiqueta.** Es el mismo caso de `ENG_3` ([f1-sesgo-eng3.md](f1-sesgo-eng3.md)) pero
peor, porque no es una correlación de muestreo sino el dato de la etiqueta con otro
nombre.

`ProductionDay` arrastra el sesgo por otra vía, y esa sí es de muestreo: la ventana
de observación termina el mismo día para toda la flota, así que un vehículo
producido tarde tuvo menos tiempo de registrar un evento.

## Qué queda pendiente de decidir

1. **La posición del evento.** Tres salidas, ninguna gratis: (a) preguntarle a Ford
   qué es `IdentificationDate` para esos 231 vehículos; (b) restringir las filas
   etiquetadas a los 61 con fecha discriminante —quedan ~20% de los positivos, con
   lo que eso implica para la potencia de la métrica—; (c) pasar a un objetivo a
   nivel vehículo en vez de por punto de corte, resignando la curva de anticipación
   que es la portada del pitch.
2. **`static_daysUntilSale` en el set base.** Auditarla con el mismo criterio con el
   que se sacó `Engine`, y en todo caso medir el aporte real como ablación.

Ninguna de las dos se decide en un notebook. Hasta que se decidan, **F2 no tiene
etiqueta por punto de corte**.

## Lo que este hallazgo NO dice

- **No invalida el anclaje.** El puente de F1 sigue midiendo lo que decía medir.
- **No invalida la etiqueta a nivel vehículo.** La cohorte sigue siendo la etiqueta:
  `IdentificationDate` nula ⇔ sin evento, sin excepciones
  ([f2-union-y-holdout-dev-test.md](f2-union-y-holdout-dev-test.md)). Lo que no se
  puede es ubicar el evento *en el tiempo* para el 79% de los casos.
- **No invalida el holdout.** El split es por vehículo y estratificado por
  `event_observed`, que no cambia.
