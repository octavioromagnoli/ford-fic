# El mejor modelo al 22-09: survival stacking con la ventana del registro (K2)

> **Entrega v2 (26-09-2026):** todo lo de abajo es de la entrega 1. Sobre v2 no hay finalista elegido: K2 perdió su corrección (la ventana) y su variante sin ventana no aprueba (a′). Survival stacking sí la aprueba, y sobre dev v2 detecta ~11–14% al 5% de falsas alarmas; el 15–17% de abajo es de la entrega 1 ([f9-remedicion-v2.md](f9-remedicion-v2.md) § K2 sobre v2).

**Fecha:** 2026-09-22 · **Fase:** F6 · **Alcance:** solo dev, R = 3; **test sin tocar.**
Es el resumen para el equipo y para el pitch. La evidencia vive en
[decisiones.md](decisiones.md) (cierre de F3, 20-09; cure model y F6, 22-09),
[f6-deteccion-vehiculo.md](f6-deteccion-vehiculo.md), [f3-piso-posicional.md](f3-piso-posicional.md)
y [f3-cure-model.md](f3-cure-model.md). Si cambia el finalista, se edita este archivo.

## La respuesta corta

**El mejor modelo es survival stacking con el conjunto en riesgo de la ventana del registro, en km
y con horizonte completo** (K2, `configs/exp_ss_hw_r3.yaml`).
- Con un 5% de falsas alarmas encuentra a **1 de cada 6 autos** que van a fallar (17%), con unos 4
  meses de anticipación.
- El finalista anterior encontraba a 1 de cada 8 (12%) con la misma cuenta.
- Es el mismo modelo con una sola corrección: ya no aprende como "sanos" los cortes de autos que
  después fallan, ni los km que el registro de eventos no cubría.
- Sigue dando probabilidades calibradas.

## La tabla

Se usan las dos métricas que sobrevivieron a las auditorías del proyecto (ver
[f3-piso-posicional.md](f3-piso-posicional.md)): la detección con 5% de falsas alarmas y el
lift por vehículo. El PR-AUC por fila **no** elige: lo supera un score que es solo el
odómetro.

**Con la etiqueta corregida** (solo los cortes cuyo horizonte cubrió el registro de eventos; es el
número honesto, y el que decide desde el 22-09):

| modelo | detecta (5% de falsas alarmas) | anticipación mediana | lift por vehículo | calibración (Brier) |
|---|---|---|---|---|
| **K2: survival stacking + ventana del registro** | **17,0% ± 3,8** (6–10 de 45) | ~7.400 km (≈ 4 meses a 57 km/día) | **1,66×** | **0,112** |
| survival stacking (finalista anterior) | 11,9% ± 2,8 (4–7 de 45) | ~9.400 km | 1,66× | 0,112 |
| survival stacking en días post-venta (F5 §3.3) | 9,6% ± 2,8 | ~7.700 km | 1,38× | — |
| piso: solo el odómetro | 4,4% | 8.300 km | 1,16× | — |

**Con la etiqueta dura** (todas las filas, como en las fichas anteriores al 22-09):

| modelo | detecta | anticipación | lift por vehículo | calibración (Brier) |
|---|---|---|---|---|
| **K2** | 15,1% ± 3,1 (6–10 de 53) | ~8.800 km | **1,67×** | **0,112** |
| survival stacking (finalista anterior) | 15,7% ± 0,9 (8–9 de 53) | ~8.300 km | 1,62× | 0,112 |
| LightGBM de control | 11,9% ± 3,2 | 9.400 km, muy inestable (± 2.200 km según los folds) | 1,49× | 0,171 |
| CNN-LSTM (la de la tutora) | 11,9% ± 3,6 | 10.400 km | 1,33× | 0,228 |
| piso: solo el odómetro | 7,6% | 7.400 km | 1,02× | — |
| cure model P0 (panel de hitos) | 4,2% (2–3 de 55) | decide a los 30–90 días de la venta | — | no se puede leer |

La última fila no se compara uno a uno con las demás. Usa otro panel, otros negativos (los
resueltos dentro de la ventana del registro) y otra regla de alerta (primer hito en vez de alerta
sostenida). Pero la diferencia es grande.

**Lo que no le ganó a survival stacking** (fichas en el índice):
- GRU, TimesFM, ordinal y GPBoost;
- el ensamble con el CNN-LSTM, E1 y E2 ([f3-ensamble-e1-e2.md](f3-ensamble-e1-e2.md));
- la incidencia aprendida con los fallados sin fecha ([f5-incidencia-externa.md](f5-incidencia-externa.md));
- el reloj en días post-venta ([f5-ss-post-venta.md](f5-ss-post-venta.md));
- promediar el score hacia atrás, K1 y K3 ([f6-deteccion-vehiculo.md](f6-deteccion-vehiculo.md)).

## Por qué K2

- **Cumple las seis condiciones del preregistro F6** con la etiqueta que decide:
  - gana en detección en las tres repeticiones y no pierde en lift;
  - la anticipación no cae más que su desvío;
  - les gana a los dos pisos y pasa (a0);
  - le saca a un score al azar el doble de ventaja que el finalista anterior: +9,5 contra +4,1
    puntos.
- **Arregla el límite principal del finalista anterior.** Darle las columnas de calendario mueve
  el ROC +0,020, contra +0,049. Ese atajo era exposición al registro, y K2 lo saca.
- **Sigue calibrado** (Brier 0,112): "este auto tiene 13% de riesgo" sirve para priorizar un
  taller.
- **Ordena mejor en el tiempo:** C-index 0,603 contra 0,582, y (a′) +0,019 contra +0,016.

## El cure model, en criollo

- **Qué se probó:** decidir en los primeros 30, 60 o 90 días después de la venta qué autos van a
  tener el problema. Se usaron cuatro hábitos de manejo (idle, viajes en frío, velocidad y
  temperatura del refrigerante) comparados contra autos sanos parecidos, separando *si* el auto
  va a fallar de *cuándo*.
- **La señal existe.** No es azar: p = 0,005 contra un sorteo que conserva mercado y fecha de
  producción.
- **Pero es débil.** Frente a un par (uno que falla, uno sano), pone más arriba al que falla el
  60% de las veces (una moneda da 50%). Con 5% de falsas alarmas encuentra 2 o 3 de 55.
- **No se adopta, porque dos reglas triviales lo empatan:**
  - "el auto que se usa poco falla más": solo los km por día dan 0,58 contra 0,60 del modelo;
  - ordenar por fecha de producción: el modelo le gana por 0,11, pero el intervalo de confianza
    roza el cero.
- **Lo que queda:** el registro de eventos solo cubre sep-2025 → mar-2026, y el riesgo se mide
  en días desde la venta. K2 usa lo primero.

## Límites que hay que decir en el pitch

1. **Es una mejora modesta en números absolutos:** +2,3 autos de 45 en promedio. Un bootstrap por
   vehículo no la separa del cero (P = 0,74). Con la etiqueta dura, K2 empata en detección y gana
   en lift. Lo que la sostiene es la regla preregistrada, cumplida en las tres repeticiones.
2. **Los autos que suma los detecta más cerca de su evento.** La anticipación mediana con la
   etiqueta corregida baja de ~9.400 a ~7.400 km.
3. **La señal sigue siendo modesta.** El PR-AUC por fila (0,176) queda por debajo del techo de
   cohorte (0,263): lo que el modelo sabe es sobre todo *qué auto*, y poco *cuándo*.
4. **La ventana del registro se estimó con los eventos de dev.** Si Ford da otra, se reconstruye
   el panel y K2 se re-corre con el mismo YAML.
5. **Nada se midió sobre el test** (74 vehículos congelados). Se mide una sola vez, con el modelo
   elegido.

## Lo que sigue

- **F4:** el dashboard contra el panel real, con K2.
- **Test:** decidir en equipo cuándo se mide K2 sobre el holdout congelado.
- **Preguntas para Ford:** la ventana real del registro, qué es `IdentificationDate` y la
  prevalencia real.
