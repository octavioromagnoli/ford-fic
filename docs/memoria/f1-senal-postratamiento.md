# Cuánta señal hay en el postratamiento

**Fecha:** 2026-09-15 · **Fase:** F1 · **Reproduce:** `python scripts/eda_raw.py`
(sección 5; deja `experiments/eda/messages.csv`)

No hay columna de DPF. El estado del postratamiento llega por tres vías, y las tres
separan las cohortes sin definirlas.

## `Message` (10 niveles, 10,6M filas)

Normalizado por distancia recorrida, eventos por 1000 km:

| Nivel | fallados | sanos | ratio |
|---|---|---|---|
| Air Filter Normal Operation | 418,46 | 438,01 | 0,96 |
| **Air Filter Full** | **27,85** | **15,37** | **1,81** |
| **Air Filter Overloaded** | **3,78** | **1,44** | **2,63** |
| Cleaning Automatically Air Filter | 2,80 | 1,24 | 2,25 |
| Air Filter Over Limit | 0,59 | 0,33 | 1,82 |
| Air Filter At Limit | 0,012 | 0,024 | 0,50 |

Lo importante: los niveles "malos" aparecen **en las dos cohortes**. No son el
evento, son severidad. Separan de a poco y con solapamiento, que es el perfil de
una feature honesta.

Y son exactamente lo que la regla 6 advierte: sin gap de blanking, un modelo que
mire los últimos km antes del evento va a memorizar `Overloaded` y reportar un
PR-AUC alto haciendo detección reactiva, que es lo que Ford ya tiene.

## Regeneraciones

`Regenerations` es el string `"Regeneration"` marcando la fila del evento (no un
conteo). `DistanceBetweenRegenerations` acompaña: media 251 km.

| regeneraciones / 1000 km | fallados | sanos |
|---|---|---|
| mediana | 2,63 | 1,79 |
| p25–p75 | 1,92–3,49 | 0,87–2,69 |

1,5× en la mediana, con solapamiento. Es el respaldo empírico de la familia B del
plan §4: regenerar más seguido por kilómetro es el síntoma de trayectos que no
completan el ciclo.

## `Acumulation`

Escala 0–95. El máximo por vehículo llega a 95 en el 80% de los fallados (303/378) y
en el 50% de los sanos (375/742). Como máximo histórico satura; sirve más como
fracción de tiempo en niveles altos dentro de la ventana que como máximo.

## Para F2

- Tasa de `Full` / `Overloaded` por 1000 km en la ventana → familia D (severidad).
- Regeneraciones por 1000 km y distancia media entre regeneraciones → familia B.
- Fracción de la ventana con `Acumulation` alto → familia D.
- Ninguna de estas features tiene sentido sin el gap: son las primeras candidatas
  a auditar si el PR-AUC sale sospechosamente alto.
