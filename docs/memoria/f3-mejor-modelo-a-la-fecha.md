# El mejor modelo al 22-09: survival stacking, y por qué el cure model no lo reemplaza

**Fecha:** 2026-09-22 · **Fase:** fin de F3 · **Alcance:** solo dev, R = 3; **test sin tocar.**
Es el resumen para el equipo y para el pitch. La evidencia vive en
[decisiones.md](decisiones.md) (cierre de F3, 20-09, y el cure model, 22-09),
[f3-piso-posicional.md](f3-piso-posicional.md) y [f3-cure-model.md](f3-cure-model.md). Si
cambia el finalista, se edita este archivo.

## La respuesta corta

**El mejor modelo sigue siendo survival stacking sobre el panel v1.** Con un 5% de falsas
alarmas encuentra a 1 de cada 6 autos que van a fallar, con unos 5 meses de anticipación. Es
el único con probabilidades calibradas. El cure model nuevo avisa mucho antes (a los 30–90
días de la venta), pero casi nunca acierta, y no se separa de dos reglas triviales.

## La tabla

Se usan las dos métricas que sobrevivieron a las auditorías del proyecto (ver
[f3-piso-posicional.md](f3-piso-posicional.md)): la detección con 5% de falsas alarmas y el
lift por vehículo. El PR-AUC por fila **no** elige: lo supera un score que es solo el
odómetro.

| modelo | detecta (5% de falsas alarmas) | anticipación | lift por vehículo | calibración (Brier) |
|---|---|---|---|---|
| **survival stacking** | **15,7% ± 0,9** (8–9 de 53) | ~8.300 km (≈ 5 meses a 57 km/día) | **1,62×** | **0,112** |
| LightGBM de control | 11,9% ± 3,2 | 9.400 km, muy inestable (± 2.200 km según los folds) | 1,49× | 0,171 |
| CNN-LSTM (la de la tutora) | 11,9% ± 3,6 | 10.400 km | 1,33× | 0,228 |
| piso: solo el odómetro | 7,6% | 7.400 km | 1,02× | — |
| cure model P0 (panel de hitos) | 4,2% (2–3 de 55) | decide a los 30–90 días de la venta | — | no se puede leer |

La última fila no se compara uno a uno con las demás: otro panel, otros negativos (los
resueltos dentro de la ventana del registro) y otra regla de alerta (primer hito en vez de
alerta sostenida). Pero la diferencia es grande. Además, en las filas que comparten, el
finalista ordena mejor: D1 0,592 contra 0,543 (A6 de [f3-cure-model.md](f3-cure-model.md)).

GRU, TimesFM, ordinal y GPBoost no superan a estos en las métricas que valen (sus fichas están
en el índice).

## Por qué survival stacking

- Es el **único** cuyo aporte del "cuándo" (a′) da positivo en las tres repeticiones:
  +0,016 ± 0,003.
- Detecta el **doble** que el piso del odómetro, y es **estable** entre sorteos de folds: la
  anticipación varía ±22 km, contra ±2.237 del control.
- Es el único que da **probabilidades calibradas**, porque el hazard se entrena sin reponderar.
  "Este auto tiene 13% de riesgo" sirve para priorizar un taller; un puntaje ordinal no.

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
- **La versión ajustada (Firth, P1) no mejora a la suma simple (P0).**
- **Lo que queda del trabajo:**
  - el registro de eventos solo cubre sep-2025 → mar-2026;
  - el riesgo se mide en días desde la venta, no en km;
  - la señal temprana es, sobre todo, intensidad de uso.

  Las tres cosas valen para cualquier modelo futuro. También quedó la infraestructura para medir
  cualquier candidato sobre hitos post-venta.

## Límites que hay que decir en el pitch

1. **La auditoría de calendario (b) de survival stacking marca +0,049 de ROC con R = 3.** Está
   pendiente antes de fijarlo como final ([decisiones.md](decisiones.md), 21-09).
2. **Sus etiquetas vienen del panel v1**, donde el 28% de los horizontes sanos cae en parte fuera
   de la ventana del registro de eventos ([decisiones.md](decisiones.md), 22-09).
3. **En números absolutos la señal es modesta.** El PR-AUC por fila (0,172) queda por debajo del
   techo de cohorte (0,263): lo que el modelo sabe es sobre todo *qué auto*, y poco *cuándo*.
4. **Nada se midió sobre el test** (74 vehículos congelados). Se mide una sola vez, con el modelo
   elegido.

## Lo que sigue

- **F4:** el dashboard contra el panel real, con survival stacking.
- **Antes de fijar el finalista:** resolver la (b) de calendario.
- **Preguntas para Ford:** la ventana real del registro, qué es `IdentificationDate` y la
  prevalencia real.
