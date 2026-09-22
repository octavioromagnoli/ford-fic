# Preregistro: cure model por vehículo sobre hitos post-venta

**Fecha:** 2026-09-22 · **Fase:** F3 · **Ramas:** `feat/landmark-post-venta` (panel) y
`exp/cure-model` (modelo).

**Se escribió** antes de construir el panel de hitos, antes de implementar el modelo y
antes de calcular D1 o D2 para cualquier score. El commit que agrega este archivo es la
marca de tiempo. Cualquier cambio posterior va en un commit propio, anterior a la corrida
que afecta, y dice por qué.

**De dónde sale:** de la Fase 1, [f3-reloj-y-ventana-del-evento.md](f3-reloj-y-ventana-del-evento.md),
confirmada en su punto de control el 22-09.

**Motivo:** el presupuesto de comparaciones está agotado (CLAUDE.md;
`docs/f3-modelos-candidatos.md` §0, punto 2). Esto no es otro modelo sobre el panel v1:
cambian la unidad de decisión (vehículo × hito post-venta), el objetivo (qué auto, con el
*cuándo* dado por un reloj y una ventana) y la censura. Igual lleva lista cerrada. Si
aparece una idea en el camino, va a "Qué queda abierto" de `f3-cure-model.md`, no a una
corrida.

## 0 · Lo que ya se vio de dev antes de escribir esto

Se declara para que el lector sepa cuánto de dev está gastado:

- **Fase 1 (y la exploración de scratch que reproduce).** AUC por vehículo de dos definiciones
  del desvío contra la flota: fracción de idle, y viajes bajo régimen entre los que se mueven.
  Se midieron en hitos de 30–180 días, junto con el nulo estratificado y la detección al 5% de
  los sanos.
  - La exploración también miró el desvío de velocidad y de viajes cortos, con otra referencia
    (mediana por viaje).
  - **Ninguna de las cuatro features de abajo se midió en su forma preregistrada** (agregador
    del panel v1, por mes, contra la mediana de vehículos sanos).
- **D1 y D2 no se calcularon nunca**, para ningún score.
- **Test:** nada. `test_split.json` no se toca en estas ramas.

Por eso las features se fijan desde el índice físico que ya estaba escrito antes de la
exploración (`docs/f3-modelos-candidatos.md` §1.1, 20-09), y no desde lo que mejor dio en dev.

## 1 · Reloj, ventana y censura

- **Reloj:** días desde la venta (*dss*).
  - venta = origen + `ProductionDay` + `daysUntilSale`;
  - evento = origen + `ProductionDay` + `IdentificationDate`;
  - origen congelado en el día 20107 (`src/data/anchor.py`).
  - Respaldo: Fase 1, §4 (H4b). El riesgo dentro de la ventana lo explican los días post-venta
    y no la edad (LR p = 0,014); el calendario no agrega.
- **Ventana del registro:** `event_window` = 2025-09-01 → 2026-03-11 (`configs/data/event_clock.yaml`).
  Fuera de ella no hay riesgo observable.
- **Entrada** de la fila (vehículo, hito L): `e = max(L + G, dss del inicio de la ventana)`.
- **Salida:** `x = min(dss del evento, dss del fin de la ventana)`.
  - **No se acota por el último viaje.** Dos eventos de dev caen 54 y 176 días después del
    último viaje: el registro los anota sin telemetría.
  - Acotar por el último viaje les pondría δ = 0 a esos dos fallados.
  - Para los sanos cambia 3 vehículos de 230 (su telemetría termina antes del 11-03-2026).
- **δ** = 1 si el evento cae en `(e, x]`. Es el caso de todo fallado con fila, porque todos los
  eventos de dev están dentro de la ventana.
- **Una fila con `x ≤ e` no informa sobre el evento:** no entra a la verosimilitud ni a D1.
  Sí entra a la referencia de flota y se puntúa.

## 2 · Panel de hitos (Fase 3)

`configs/data/panel_landmark_ps.yaml` tiene que reproducir estos valores. Si difieren, se
commitea el cambio acá antes de construir.

| parámetro | valor |
|---|---|
| hitos L | 30, 60, 90 días post-venta |
| gap G | 30 días (≈ 1.700 km al km/día mediano; G = 500 km del panel v1 son ≈ 9 días) |
| horizonte H | 240 días (cubre la latencia observada: máximo 268 d post-venta) |
| ventana de features | viajes post-venta en `(venta, venta + L]` |
| viajes mínimos en la ventana | 15 |
| `label` | evento en `(L + G, L + G + H]`; con este H, igual a `event_observed` en toda fila |
| negativo resuelto | sano con exposición en la ventana ≥ **`E_min` = 90 d**, medida desde `max(venta + 30 + G, inicio de la ventana)`; sensibilidad: 60 y 120 |
| terciles de producción | de `ProductionDay` sobre los vehículos de dev |

**Una fila por (vehículo, hito)** si se cumple todo esto:
- tiene `daysUntilSale`;
- `venta + L` ≤ su último viaje;
- no tiene evento en `≤ L + G` (regla 1);
- tiene ≥ 15 viajes post-venta hasta el hito.

Cada descarte se cuenta.

## 3 · Features: cuatro, fijas, con signo físico

Definiciones de `configs/data/features_v1.yaml`, calculadas por mes calendario sobre los
viajes post-venta de la ventana:

| feature | columna | agregador | peso de la celda (mínimo) | signo |
|---|---|---|---|---|
| `idle_per_1000km` | `idle` | `per_1000km` | km del mes (≥ 100) | + |
| `trips_below_regime_temp_frac` | `below_regime` | `mean` | viajes del mes (≥ 5) | + |
| `speed_kmh_mean` | `speed_kmh_moving` | `mean` | viajes en movimiento del mes (≥ 5) | − |
| `coolant_temp_end_mean` | `CoolantTemperatureEnd_moving` | `mean` | viajes en movimiento del mes (≥ 5) | − |

- **El mínimo se aplica al peso de cada feature**, no al mes entero. Un auto quieto, con
  muchos viajes y casi sin km:
  - queda en NaN en `idle_per_1000km` y en las dos de viajes en movimiento;
  - **conserva `trips_below_regime_temp_frac`**.
- **Normalización contra la flota** (`FleetReferenceNormalizer`, ajustada con el train de cada
  fold, nunca con validación):
  - **referencia:** para cada feature, la mediana entre vehículos sanos del train con dato en
    la celda mercado × mes. Se deduplica por (vehículo, mes) quedándose con la fila de más
    peso;
  - **si la celda tiene menos de 10 vehículos:** se usa el mismo mes con los dos mercados
    juntos, y si tampoco alcanza, la mediana global de la feature. Se loguea cuántas celdas
    usó cada nivel;
  - **salida por fila:** el promedio de (valor − referencia) sobre los meses con dato,
    ponderado por el peso;
  - el modelo no ve el mes.
- **Después del normalizador:** imputación por la mediana y estandarización, **por hito**,
  con las filas de train de ese hito.
- **Autos quietos:** decisión del punto de control 1, sin cambios de features ni exclusiones.
  Se tratan con la regla de arriba (NaN → mediana del hito) y se reportan aparte (C6).
- **Fuera de las features, a propósito:**
  - `ProductionDay`, `daysUntilSale`, fecha de venta, mes y temperatura ambiente (solo
    entran en la auditoría A3);
  - `static_SalesCountry_cd`, que solo sirve para armar la celda;
  - km/día (solo entra como piso, en C3).

## 4 · Las corridas (lista cerrada)

Reglas comunes:
- solo dev (`select_dev()`);
- CV de 5 folds × **R = 3** con `splits_landmark_ps_r3.json` (§6);
- `random_state` 42;
- sin barridos.

### P0 · pesos unitarios (piso y respaldo)

- **Score:** `s = z(idle) + z(bajo régimen) − z(velocidad) − z(refrigerante)`, sobre las
  cuatro features normalizadas e imputadas (Dawes 1979).
- **Cero coeficientes ajustados.** Solo se ajustan la referencia de flota, la mediana y la
  escala.
- **D1 y D2 se miden con `s`.**
- El cure model con incidencia `sigmoid(a + b·s)` se ajusta **solo para reportar
  calibración**: es monótono en `s` dentro de cada hito y no cambia D1.
- **P0 no compite por un lugar.** Es el piso que P1 tiene que superar, y el respaldo si nadie
  lo supera.

### P1 · cure model, incidencia Firth + FLIC

Modelo `cure_mixture`, target `cure_window`, un ajuste por hito (`per_landmark: true`):

- **incidencia:** logística con penalización de Firth, ponderada por los pesos del EM, sobre
  las cuatro features, con reestimación FLIC del intercepto (Firth 1993; Heinze & Schemper
  2002; Puhr et al. 2017);
- **latencia:** Weibull en dss con entrada tardía, sin covariables;
- **EM** (Sy & Taylor 2000; Peng & Dear 2000): `tol` 1e-7, `max_iter` 1000, y la verosimilitud
  penalizada no puede bajar en ninguna iteración;
- **score:** `π_L(x) · [1 − S_u(L+G+H) / S_u(L+G)]`.

Son 5 parámetros de incidencia por hito, con ~40 eventos: por encima del techo de
Riley et al. (2019), que da 1,6–2,8. Firth encoge, pero **por eso P1 tiene que ganarle a
P0 para quedar**.

### P2 · P1 con TabPFN v2 en la incidencia (condicional)

Corre **solo si P1 le gana a P0** (§5). Va en dos pasos, declarados como aproximación:
1. clasificador sobre los vehículos resueltos del train del hito (eventos contra negativos
   resueltos, las mismas cuatro features);
2. latencia por máxima verosimilitud con π fijo.

Licencia: **TabPFN v2** (con atribución). **No 2.5**, que prohíbe el uso comercial y de
producción. Se compara contra P1 con la regla del §5. Si no corre, el lugar queda vacío y no
se reasigna.

## 5 · Qué decide

**Deciden dos métricas**, con media ± desvío poblacional entre las 3 repeticiones:

- **D1 · C-index con entrada tardía**, por hito y promedio simple de los tres hitos.
  - Reloj en dss.
  - Un par (i, j) es comparable si δ_i = 1, `e_j < x_i` y además `x_j > x_i`, o `x_j = x_i`
    con δ_j = 0.
  - Es concordante si `score_i > score_j`; los empates de score valen ½.
  - Las filas con `x ≤ e` nunca son comparables.
  - Es el C de Harrell con conjuntos de riesgo que respetan la entrada tardía (Harrell 2015,
    cap. 20).
- **D2 · detección a presupuesto de falsas alarmas**, primera alerta sobre los hitos
  {30, 60, 90} con k = 1.
  - Positivos: vehículos con δ = 1 en alguna fila.
  - Negativos: los resueltos (`E_min` = 90 d).
  - Sea m = ⌊0,05 · N_neg⌋. El umbral τ es el (m+1)-ésimo mayor valor, entre los negativos, de
    su score máximo sobre sus hitos.
  - Un vehículo alerta en el primer hito con score > τ. Así alertan a lo sumo m negativos.
  - Se reportan la detección, la anticipación (evento − hito de la alerta) en días y en km
    (mediana e IQR), y el mismo cálculo al 10%.
  - Con 40–60 negativos resueltos, m es 2 o 3 autos: se dice.

**"Le gana a X en M"** exige las dos condiciones:

1. la diferencia de medias entre repeticiones supera `√(σ² + σ_X²)`, que es la convención del
   repo;
2. el **IC95 del bootstrap pareado por vehículo** de la diferencia excluye el 0. Son 2.000
   réplicas: se remuestrean vehículos de dev, las dos métricas se recalculan con las mismas
   predicciones OOF en cada repetición, y se promedian las repeticiones.

**Por qué la condición 2 es nueva:**
- con P0 (sin coeficientes) y P1 (Firth), el desvío entre repeticiones solo mide el sorteo de
  folds, que en estos modelos es casi cero, y deja afuera la incertidumbre de ~40 eventos por
  hito;
- contra un piso sin ajuste (σ = 0), la condición 1 sola se cumple con cualquier diferencia;
- el error estándar de un C con 40 eventos y ~150 controles es ≈ 0,05 (Hanley & McNeil 1982).

"Pierde" es lo simétrico. Se reporta además la diferencia pareada por repetición.

**Veredicto de P1 contra P0 (y de P2 contra P1):**
- **gana** si gana en una métrica y no pierde en la otra;
- **pierde** si pierde en cualquiera de las dos;
- **empata** en otro caso. Si empata, se queda el más simple (P0 antes que P1, P1 antes que P2).

**Adopción:** un score solo se adopta si además:
- le gana a su **nulo estratificado** en D1 (C1, p < 0,05);
- le gana al **piso de producción** en D1 (C2, regla de arriba);
- **no marca en calendario** (A3).

Si nada cumple, el resultado es negativo y se documenta así.

**Informativas, no deciden:**
- AUC y PR-AUC con lift por hito, entre positivos y negativos resueltos;
- calibración: observado contra esperado por tercil de riesgo, con lo esperado =
  `Σ π_L · (1 − S_u(x)/S_u(e))` sobre la exposición real. Más la pendiente de calibración y el
  Brier en los resueltos. **No se declara calibración global** sin la prevalencia real de la
  flota (Prentice & Pyke 1979);
- fracción susceptible y β por hito, con bootstrap por vehículo. Un rasgo estable da β
  parecidos entre hitos;
- (k, λ) de la latencia contra la tabla empírica de la Fase 1, §4;
- seguimiento suficiente (Maller & Zhou 1996);
- D2 con `E_min` 60 y 120;
- D2 sobre los 60 eventos de dev, contando como perdidos los que no tienen fila;
- la curva detección / falsas alarmas.

**No aplican en este panel y no deciden:**
- PR-AUC por fila juntando hitos;
- techo de cohorte y PR-AUC entre fallados (con este H, `label` = `event_observed`);
- (a′);
- `k_consecutive: 2`;
- agregación por vehículo sobre hitos.

## 6 · Splits

`splits_landmark_ps_r3.json` se congela **antes** de entrenar:
- `n_splits` 5, `n_repeats` 3, `seed` 42;
- extiende `${DATA_DIR}/processed/splits_r3.json`: los 171 vehículos del panel v1 conservan
  su fold en cada repetición;
- los vehículos nuevos de dev se asignan estratificando por `label` a nivel vehículo;
- cada fold necesita **≥ 5 vehículos positivos en validación por hito**.

Si extender no es viable, `make_splits` normal, y la comparación con el finalista (A6) se
declara **no pareada**.

## 7 · Controles y auditorías

No son candidatos y no consumen lugar. Los corre `scripts/audit_cure.py` sobre P0 y P1 (y
P2, si corre):

- **A0 · nulo global.** Se permutan las filas de features dentro de cada hito y se reentrena.
  D1 tiene que caer a ~0,5 y D2 al presupuesto; si no, hay fuga y **no se lee ningún otro
  número**.
- **C1 · nulo estratificado** (el que decide la adopción).
  - Se permuta el **score** entre las filas del mismo hito × tercil de producción × mercado,
    2.000 veces, con las mismas predicciones y sin reentrenar. Se recalcula el D1 promedio de
    los tres hitos, en cada repetición, y se promedian las repeticiones.
  - `p = (1 + #{nulo ≥ observado}) / 2.001`.
  - Permutar el score equivale a permutar el desenlace completo (δ, entrada y salida). Permutar
    solo δ dejaría eventos en fechas que no existen.
- **C2 · piso de producción.** Score = −`ProductionDay` y score = −fecha de venta, con D1 y D2.
  Además, el ρ del score con `ProductionDay` (tiene que ser |ρ| < 0,2) y la AUC dentro de cada
  tercil.
- **C3 · piso de uso.** Score = −(km en `(venta, venta + L]` / L): menos uso ⇒ más riesgo. Es la
  orientación que dio la Fase 1, §6.
- **A3 · calendario (b).** P1 con `aux_landmark_month`, `aux_air_temp_window_mean` y
  `aux_static_ProductionDay` como covariables de incidencia. Si ΔD1 o ΔAUC > 0,02, marca, y P1
  no se interpreta hasta proponer la corrección en un punto de control.
- **A5 · ablación de la normalización.** P1 con los cuatro agregados crudos de la ventana
  (`aux_raw_*`) en vez de los normalizados.
- **A6 · referencia: el finalista** (`f3-survival-stacking-r3-rebuild`).
  - En cada hito se toma su score del último corte con `cut_date` ≤ fecha del hito.
  - D1 y AUC entre resueltos sobre los vehículos que están en los dos.
  - Es informativo: sus sanos están submuestreados por el emparejado.
- **C6 · autos quietos.** D1 y D2 de P0 y P1 con todos los vehículos y sin los que recorrieron
  < 100 km post-venta hasta el hito (`immobile_km` de `event_clock.yaml`). Es informativo: no
  decide y no cambia ninguna feature.

## 8 · Orden y paradas

1. Panel → punto de control 3a (**solo conteos**) → splits congelados.
   - Si un fold queda con < 5 positivos en un hito, se ajusta mirando solo conteos y se anota
     acá en un commit propio.
2. Tests de correctitud en verde: EM sobre datos simulados con ventana, Firth contra una
   referencia, normalizador sin fuga, hook `preprocessing` byte a byte.
3. **P0** + A0, C1, C2, C3 y C6.
   - **Si P0 no le gana a C1 en D1 (p ≥ 0,05):** el rasgo temprano era ruido. Se documenta el
     negativo y **se para**: P1 y P2 no corren.
4. **P1** + A0, A3, A5, C6 y A6 → veredicto P1 contra P0.
5. **P2** solo si P1 ganó → veredicto P2 contra P1.
6. Se escribe el veredicto, gane o pierda (`f3-cure-model.md` y `decisiones.md`).

## Lo que queda fuera a propósito

- otras features u otras definiciones de las cuatro;
- otros hitos, gaps, horizontes o `E_min` primario;
- excluir o reponderar a los autos quietos;
- latencia con covariables;
- otras penalizaciones o pesos de Firth;
- otros umbrales de celda o de vehículos de referencia;
- percentiles en vez de diferencias contra la referencia;
- estandarización agrupada entre hitos;
- un meta-modelo o un ensamble con el finalista;
- TabPFN 2.5 o cualquier otro aprendiz en la incidencia.

Todo eso es un barrido o un candidato nuevo.
