# El anclaje temporal existe (la regla 4 se puede levantar)

> **Entrega v2 (26-09-2026):** el anclaje se sostiene (IQR 0 d sobre 1.012 autos) solo después de corregir en −538 d el `ProductionDay` de la estática de fallados, que viene contado desde otro origen ([f9-entrega-v2.md](f9-entrega-v2.md) §2).

**Fecha:** 2026-09-15 · **Fase:** F1 · **Reproduce:** `python scripts/eda_raw.py`
(sección 4; deja `experiments/eda/anchor.csv`)

## El problema

`IdentificationDate` —el día del evento— está en **días desde producción**.
`TripDatetimeStart` y `eventTimestamp` están en **calendario** (2025–2026). Sin un
puente entre los dos ejes, el evento no se puede ubicar sobre el histórico del
vehículo, y sin eso no hay `time_to_event_km` ni etiqueta por punto de corte.

Por eso la regla 4 declaró el eje de km como el único confiable.

## El puente

Si `ProductionDay` cuenta días desde un origen común a toda la flota, entonces
`primer_viaje − ProductionDay` tiene que ser aproximadamente constante entre
vehículos. Medido sobre los 1094:

| Hipótesis | std | IQR | rango |
|---|---|---|---|
| **`primer_viaje − ProductionDay`** | **0,7 d** | **0,0 d** | 12 d |
| `primer_viaje − (ProductionDay + daysUntilSale)` | 60,6 d | 73,8 d | 353 d |
| `primer_viaje` sin offset | 94,3 d | 163,2 d | 337 d |

Contra un rango de `ProductionDay` de 338 días, un IQR de **cero** no es ruido
afortunado: `ProductionDay` está en el mismo eje que el calendario, con origen
común, y el primer viaje ocurre el mismo día relativo a producción en más de la
mitad de la flota (12 días de rango total, extremos incluidos).

## Qué habilita

```
fecha_evento = origen + ProductionDay + IdentificationDate
odo_evento   = interpolar esa fecha sobre los viajes del vehículo
```

Con eso el evento queda sobre el eje de km y `time_to_event_km` sale del contrato
sin inventar un anclaje.

## Antes de usarlo

- El origen se estima de los datos, no está declarado: conviene fijarlo una vez,
  guardarlo y no recalcularlo por corrida.
- Los 12 días de rango son vehículos cuyo primer viaje registrado no es el primero
  real. Para ellos el `odo_evento` interpolado tiene esa incertidumbre; vale
  reportar cuántos eventos caen fuera de la ventana de viajes observada.
- Esto no reemplaza al eje de km como eje principal. Lo que hace es permitir
  traducir la etiqueta al eje de km, que es otra cosa.
