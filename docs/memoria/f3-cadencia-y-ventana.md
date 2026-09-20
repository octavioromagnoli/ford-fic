# Cadencia de predicción y ventana de agregación son dos cosas distintas

**Fecha:** 2026-09-19 · **Fase:** F3 · **Alcance:** dev (290 vehículos, 526.420 viajes) ·
**Reproduce:**

```bash
python scripts/eda_gaps.py    # bloque 10; deja experiments/eda/dev/gaps/cadence_*.csv
```

Surgió de una pregunta razonable: *"la predicción no tiene que ser diaria; capaz
juntamos 15 días de datos y predecimos"*. La respuesta corta es **sí para la cadencia,
no para la ventana**, y conviene tener los números escritos porque la confusión entre
las dos cosas reaparece cada vez que alguien mira el panel.

- **Ventana de agregación (W)**: cuánto historial mira cada feature. Hoy son 1.000 km
  hacia atrás desde el corte, sobre el eje de odómetro (regla 4).
- **Cadencia de emisión**: cada cuánto se produce un score para un vehículo. Hoy está
  atada a la grilla de Δ = 500 km porque el panel se construye así, pero **no tiene por
  qué estarlo**: son decisiones independientes.

## 1 · Qué junta realmente una ventana de calendario

| ventana | km p10 | km **p50** | km p90 | p90/p10 | ventanas/vehículo (p50) | sin 5 viajes con desplazamiento | bajo 500 km | cobertura p50 |
|---|---|---|---|---|---|---|---|---|
| 7 días | 82 | **290** | 611 | 7,5× | 48 | 25,9% | 65,8% | 0,83 |
| **15 días** | 175 | **621** | 1.309 | 7,5× | 25 | **28,2%** | 44,5% | 0,94 |
| 30 días | 350 | **1.242** | 2.618 | 7,5× | 14 | 26,1% | 37,5% | 1,00 |
| grilla 500 km (panel v1) | — | — | — | — | 35 | — | — | — |

(*cobertura* = fracción de las ventanas de calendario de la vida del vehículo que tienen
al menos un viaje. El QC del panel v1 pide ≥ 5 viajes y ≥ 50% de W en km recorridos.)

## 2 · Por qué la ventana sigue siendo de km

1. **Una quincena son 621 km de mediana, menos que la W actual**, y con una dispersión
   de **7,5× entre p10 y p90**. Un vehículo de reparto y uno de fin de semana producirían
   ventanas que no se pueden comparar entre sí: la misma columna mediría cosas distintas
   según cuánto se usa el auto. La ventana de km normaliza eso por construcción.
2. **El 28% de las quincenas no pasaría el QC** (menos de 5 viajes con desplazamiento), y
   pasar a 30 días casi no ayuda (26,1%): el problema no es el largo de la ventana sino
   que hay vehículos que apenas se mueven. Con ventana de km, un vehículo poco usado
   simplemente genera menos cortes, en vez de generar cortes vacíos.
3. **Una grilla de calendario reimporta el confusor de
   [f2-eda-revision-y-features.md](f2-eda-revision-y-features.md) §3.4.** Los eventos de
   dev caen entre sep-2025 y mar-2026 y la exposición sana está en 2026: si el corte se
   define por fecha, cada fila lleva el calendario pegado y el emparejado por mes
   (`sampling.match_on` del panel) pasa de cerrar el atajo a pelear contra el propio eje.

## 3 · Lo que sí se puede hacer, y es lo que se pidió

**Mover solo la cadencia de emisión**: cada 15 días se toman los últimos 1.000 km del
vehículo y se emite un score. La ventana sigue siendo de km (física, deconfundida), el
panel no cambia y el emparejado tampoco. Operativamente es exactamente "juntamos 15 días
y predecimos".

Los números lo permiten: 25 quincenas por vehículo de mediana y **cobertura 0,94** —el
94% de las quincenas de la vida de un vehículo tiene al menos un viaje—. El resto son
huecos de registro, no vehículos detenidos: el hueco máximo por vehículo tiene mediana de
32 días, 279 de 290 vehículos tienen algún hueco de más de 15 días y 150 alguno de más de
30. Una cadencia quincenal tiene que tolerar "esta quincena no hay dato nuevo",
manteniendo el último score en vez de emitir uno construido con media ventana.

## 4 · Consecuencia para el modelo

La cadencia quincenal es la unidad natural de la **alarma con persistencia** (CUSUM sobre
el índice de degradación): "tres quincenas seguidas subiendo" es una frase que un
operador entiende y que se puede sostener con estos datos. Lo que **no** se puede hacer es
la misma lógica a nivel viaje: con el 35% de las filas siendo idle de 0 km y la mediana
de viaje en pocos km, una secuencia de 10 viajes es ruido. La unidad mínima es la ventana
de 1.000 km (≈ 25 días de mediana), no el viaje.
