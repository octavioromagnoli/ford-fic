# Cure model por hito post-venta: el rasgo temprano existe, pero no se separa de dos pisos triviales

**Fecha:** 2026-09-22 · **Fase:** F3 · **Rama:** `feat/landmark-post-venta` (Track A y B juntos,
sin mergear) · **Alcance:** solo dev (268 vehículos, 727 filas), CV 5 × R = 3 con
`splits_landmark_ps_r3.json`. **Test sin tocar.**
**Preregistro:** [f3-preregistro-cure.md](f3-preregistro-cure.md) (6f53ca7, más tres enmiendas
anteriores a las corridas). **De dónde sale:** [f3-reloj-y-ventana-del-evento.md](f3-reloj-y-ventana-del-evento.md).

## En una línea

El score de pesos unitarios (P0) le gana a su nulo estratificado (**p = 0,0045**), pero **no le gana
al piso de producción** con la regla del preregistro. Además **empata con un solo número de uso**:
los km por día hasta el hito. Firth (P1) empata con P0. **No se adopta nada: el resultado es
negativo.** El finalista sigue siendo survival stacking.

## Qué se midió

- **Unidad:** una fila por vehículo × hito post-venta (30, 60, 90 días). Gap de 30 días,
  horizonte de 240.
- **Reloj:** días desde la venta. El riesgo existe solo dentro de la ventana del registro
  (01-09-2025 → 11-03-2026), así que la censura de un sano es su exposición en esa ventana y
  no su último viaje.
- **Features:** las cuatro del índice físico, fijas y con su signo. Se miden como desvío
  contra la mediana de los sanos del train del mismo mercado × mes.
- **Candidatos:**
  - **P0:** suma de las cuatro z con su signo, cero coeficientes. El cure solo lo calibra.
  - **P1:** mixture cure model con incidencia Firth + FLIC y latencia Weibull con entrada tardía.
- **Deciden:** D1 (C-index con entrada tardía, promedio de los tres hitos) y D2 (detección con
  primera alerta sobre los hitos, alertando a lo sumo el 5% de los negativos resueltos).

## Los números

Media ± desvío entre las 3 repeticiones; entre corchetes, IC95 por bootstrap de vehículos (2.000).

| | P0 · pesos unitarios | P1 · Firth + FLIC |
|---|---|---|
| **D1** | **0,599 ± 0,004** [0,520; 0,675] | 0,584 ± 0,016 [0,518; 0,646] |
| D1 por hito 30 / 60 / 90 | 0,568 / 0,621 / 0,607 | 0,590 / 0,603 / 0,559 |
| **D2** (≤ 3 de 69 negativos alertan, 55 positivos) | **4,2% ± 0,9** (2, 2 y 3 autos) | 3,6% ± 1,5 (2, 1 y 3 autos) |
| anticipación mediana de los detectados | 115 d · 4.069 km | 153 d · 3.262 km |
| D2 al 10% (≤ 6 alertan) | 13,3% | 9,7% |
| D2 con E_min 60 / 120 | 5,5% / 2,4% | 4,8% / 3,0% |
| D2 sobre los 60 fallados de dev (5 sin fila) | 3,9% | 3,3% |
| AUC entre resueltos, por hito | 0,589 / 0,650 / 0,627 | 0,624 / 0,626 / 0,606 |
| D1 solo con filas que entran en L + G | 0,583 | 0,575 |

## Auditorías

| | P0 | P1 | lectura |
|---|---|---|---|
| **A0** nulo global (5 permutaciones, reentrena) | D1 0,483 · D2 3,0% | D1 0,497 · D2 1,7% | sin fuga (tolerancia ±0,05, declarada en 3165340) |
| **C1** nulo estratificado (2.000, hito × tercil × mercado) | nulo 0,515 ± 0,028 · **p = 0,0045** | nulo 0,525 ± 0,025 · p = 0,012 | los dos le ganan |
| **C2** contra −ProductionDay (piso D1 0,489) | +0,110, IC pareado **[−0,005; 0,221]** | +0,095 [−0,021; 0,206] | **ninguno le gana** |
| **C2** contra −fecha de venta (piso D1 0,490) | +0,109 [0,002; 0,212] | +0,094 [−0,011; 0,194] | solo P0 |
| C2 · ρ del score con ProductionDay | 0,03 / −0,06 / −0,06 | −0,04 / −0,18 / −0,18 | \|ρ\| < 0,2 en los dos |
| **C3** piso de uso −km/L (D1 **0,581**) | +0,018 [−0,038; 0,074] | — | P0 **empata** |
| **C6** sin autos quietos (< 100 km) | D1 0,587 · D2 12,4% | D1 0,577 · D2 5,2% | informativo |
| **A3** calendario como covariable | no aplica (sin coeficientes) | D1 −0,044 · AUC −0,076 | no marca |
| **A5** agregados crudos, sin normalizar | — | D1 **0,611** (+0,027) | la normalización no ayuda |
| **A6** el finalista en las filas que comparten (322, 146 vehículos) | — | finalista D1 0,592 · AUC 0,615 contra P1 0,543 · 0,574 | informativo |

**Veredicto P1 contra P0: empata.**
- D1: −0,015, con umbral √(σ² + σ²) = 0,017 e IC pareado [−0,077; 0,045].
- D2: −0,6 puntos, IC [−14; 9].
- Por la regla, se queda P0 y **P2 no corre**.

**Adopción de P0:** le gana a C1 ✓, sin fuga ✓, **no le gana a C2** ✗.
- La lectura que quedó en el código antes de correr (3165340) exige ganarles a los dos pisos
  de C2.
- P0 no le gana justamente al que da nombre al piso, −ProductionDay, así que la conclusión no
  depende de esa lectura.
- A3 no aplica a P0.

## Lo que se aprendió

1. **La señal es real contra su nulo, pero chica.** D1 ≈ 0,60. Al 5% de falsas alarmas detecta
   2 o 3 autos de 55. Es lo que ya anticipaba la Fase 1 (AUC 0,61–0,64 contra todos los sanos,
   4–10% de detección).
2. **Casi todo es intensidad de uso.** −km/L sola da D1 = 0,581. Un auto que se usa poco hace
   más idle, más viajes fríos, va más despacio y termina con el refrigerante más frío: las
   cuatro features miden, sobre todo, cuánto se mueve el auto. No se puede separar de esta
   señal con estos datos.
3. **La normalización contra la flota no suma.** Con los agregados crudos, P1 da D1 = 0,611, más
   que normalizado. El desvío por mercado × mes saca variación que también predecía.
4. **La fracción curada no está identificada.** Solo el 0–1%, 18–21% y 26% de los sanos
   (hitos 30, 60 y 90) tiene exposición hasta el p90 de la latencia. π̄ sale entre 0,5 y 0,7.
   La pendiente de calibración da ~0,4. El orden dentro de un hito no depende de esto, pero las
   probabilidades absolutas no se pueden leer.
5. **La aproximación de la entrada tardía no mueve el resultado.** D1 con solo las filas que
   entran en L + G es casi igual (0,583 y 0,575).
6. **El calendario no está detrás.** A3 baja D1 y la AUC, y el ρ con ProductionDay es chico.
   Lo que falla contra C2 es la potencia: con ~40 eventos por hito, el IC pareado de una
   diferencia de 0,11 toca el cero.

## Desviaciones y enmiendas

- **Enmiendas del preregistro, todas anteriores a las corridas:**
  - 563e8d4: la calibración (a, b) de P0 va por Firth + FLIC, porque por MV degeneraba (π → 1)
    en 2 de 11 submuestras;
  - f17ff63: D1 con las filas que entran en L + G, como informativa.
- **El normalizador descarta el mercado.** El prompt decía dejarlo pasar; el preregistro §3 dice
  que solo arma la celda.
- **P2 (TabPFN v2) no está implementado.** Corría solo si P1 le ganaba a P0, y no le ganó.
- **Tolerancias de A0 (±0,05 en D1 y en D2).** El preregistro no las fijaba en números; se
  declararon en los YAML antes de correr.
- **Dos informativas no se calcularon:** β por hito con bootstrap (hay que reajustar el modelo
  por remuestreo) y (k, λ) contra la tabla empírica de la Fase 1. (k, λ) por fold sí están en
  `metrics.json`: k ≈ 2 / 3,5 / 5 y λ ≈ 210–250 d.

## Qué queda abierto (ideas, no corridas)

- **Preguntas para Ford, las mismas de la Fase 1:** qué es `IdentificationDate`, la ventana
  real de extracción y la prevalencia real. Con una ventana más larga, la fracción curada se
  podría identificar.
- **Uso como variable de negocio.** Si el rasgo temprano es "auto poco usado", la pregunta pasa
  a ser si Ford quiere alertar por uso. Eso es una decisión de producto, no un modelo.
- **Una verosimilitud que descuente π por S(e)/S(L+G)** en las filas que entran tarde. No
  cambiaría D1 (punto 5), pero sí la calibración.
- **Nada de esto se corre sin un preregistro nuevo:** la lista de este quedó cerrada.

## Cómo se reproduce

```bash
python scripts/audit_event_clock.py --config configs/data/event_clock.yaml           # Fase 1
python scripts/build_landmark_panel.py --config configs/data/panel_landmark_ps.yaml  # panel de hitos
python scripts/make_splits.py --config configs/data/panel_landmark_ps.yaml           # folds (ya congelados)
python scripts/train.py --config configs/exp_cure_p0_unitweight.yaml
python scripts/audit_cure.py --config configs/exp_cure_p0_unitweight.yaml
python scripts/train.py --config configs/exp_cure_p1_firth.yaml
python scripts/audit_cure.py --config configs/exp_cure_p1_firth.yaml                  # + veredicto contra P0
python scripts/check_setup.py                                                          # 129 chequeos
```

El panel y los folds están en wandb como Artifact `panel-landmark-ps`. Las corridas están en el
grupo `f3-cure`; las auditorías, en `audit_cure.json` y en las corridas `audit-*`. Sin red:
`WANDB_MODE=disabled`.
