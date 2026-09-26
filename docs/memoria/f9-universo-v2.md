# El universo v2 son 557 vehículos: la selección pasó del mercado a la fecha de producción

**Fecha:** 2026-09-26 · **Fase:** F9 · **Reproduce:**

```bash
python scripts/make_test_split.py --config configs/data/test_split.yaml --force     # universo + holdout v2
python scripts/compare_deliveries.py --config configs/data/compare_deliveries.yaml  # §5: el corte de producción
```

Reemplaza, para la entrega v2, a [f2-universo-fecha-usable.md](f2-universo-fecha-usable.md)
(364 vehículos) y a la sección del holdout de
[f2-union-y-holdout-dev-test.md](f2-union-y-holdout-dev-test.md). El holdout viejo sigue
congelado en `data/processed-v1/test_split.json` y se reproduce bit a bit con
`configs/data/test_split_v1.yaml` (verificado: huella `ba9aa4d290bc6610`, 290/74/717).

## 1 · El criterio de la entrega 1 ya no descarta a nadie

En v2 ningún fallado tiene la fecha por defecto (0 de 298 filas) y los cinco mercados tienen
eventos fechados con 100% de fecha utilizable (ARG 18, BRA 67, CHL 85, COL 97, PER 2).
`require_usable_event_date` saca a 0 vehículos. `keep_markets` deja de ser un recorte: se
declaran los cinco mercados con sanos. PRY tiene 1 fallado y 0 sanos, y ese auto además se
produjo en 2024.

## 2 · El mismo problema, en otro eje: las cohortes se muestrearon en períodos de producción distintos

| producción | fallados | sanos |
|---|---|---|
| 05-08-2024 → 19-01-2025 | **93** | **0** |
| 20-01-2025 → 31-07-2025 | 177 | 380 |
| 01-08-2025 → 23-12-2025 | **0** | **341** |

- **Antes del 20-01-2025 solo hay fallados.** El primer sano es del 20-01-2025 (su
  `ProductionDay` 1 es el origen de la lista de sanos). Los 93 fallados de 2024 no tienen ni un
  sano de su época: son positivos de una población sin negativos.
- **Después de julio de 2025 solo hay sanos**, y no es falta de exposición. El hazard por días
  desde la venta de los producidos entre el 20-01 y el 30-06-2025, aplicado a la exposición de
  los producidos después, da:

  | producción | autos | eventos observados | esperados |
  |---|---|---|---|
  | referencia (20-01 → 30-06-2025) | 447 | 155 | 155,0 |
  | julio 2025 | 107 | 19 | 27,9 |
  | agosto–septiembre 2025 | 127 | **0** | **28,4** |
  | octubre–diciembre 2025 | 199 | **0** | **22,0** |

  Cero contra ~50 no es azar: las fallas de esos autos no están en la muestra. Puede ser por
  cómo se armó la lista o porque el defecto no los afecta. Un sano de octubre de 2025 no es un
  negativo verificable, igual que no lo era uno de CNTRY_1 en la entrega 1.

**La salida es la misma regla simétrica de la entrega 1, aplicada a la producción.** Se
conserva el período donde las dos cohortes se muestrearon: `production_day_window: [1, 193]`,
es decir del 20-01-2025 al 31-07-2025, el mes del último fallado (28-07). Queda declarado en
`configs/data/test_split.yaml` y aplicado en `src/data/usable.py`.

**Lo que esto dice de la entrega 1.** El universo de 364 tenía el mismo corte: sus fallados se
produjeron hasta el 03-09-2025 (5 de 80 después de julio) y 104 sanos del cuarto trimestre de
2025 no tenían ningún fallado. El "sesgo de `ProductionDay`" de v1 (ρ = −0,435, tasa por
quintil 0,46 → 0,00) se leyó como exposición: los producidos tarde no llegaron a fallar. Con
el registro abierto hasta septiembre de 2026, la exposición ya no alcanza para explicarlo:
**era, al menos en parte, selección.**

## 3 · Cómo queda

| | vehículos | eventos | tasa | mercados |
|---|---|---|---|---|
| entrega 1, universo | 364 | 80 | 0,220 | CHL, COL |
| **entrega v2, universo** | **557** | **177** | **0,318** | ARG, BRA, CHL, COL, PER |
| v2 dev | **446** | **141** | 0,316 | |
| v2 test | **111** | **36** | 0,324 | |

Descartados, 434 de 991 vehículos con etiqueta en v2 (y 285 códigos de v1 que ya no vienen):

| motivo | fallados | sanos |
|---|---|---|
| producidos antes del 20-01-2025 | 92 | 0 |
| producidos después del 31-07-2025 | 0 | 341 |
| mercado sin sanos (PRY) | 1 | 0 |

Por mercado y motor (universo; entre paréntesis, los eventos):

| | ENG_1 | ENG_2 | ENG_3 | total |
|---|---|---|---|---|
| ARG | 37 (7) | 41 (2) | 38 (1) | 116 (10) |
| BRA | — | 51 (9) | 120 (46) | 171 (55) |
| CHL | 30 (4) | 64 (41) | 11 (0) | 105 (45) |
| COL | 20 (0) | 105 (67) | 14 (0) | 139 (67) |
| PER | 2 (0) | 18 (0) | 6 (0) | 26 (0) |
| **total** | 89 (11) | 279 (119) | 189 (47) | **557 (177)** |

- **Eventos por mercado:** en la entrega 1 eran CHL 19 y COL 61. Ahora el estudio tiene 2,2×
  los eventos y tres mercados más.
- **ENG_3:** pasa de 0 a 47 eventos, pero 46 son de Brasil. Fuera de Brasil, ENG_3 tiene 1
  evento en 69 autos.
- **Tasa de eventos:** la tasa por auto cambia de mercado a mercado (ARG 9%, BRA 32%, CHL 43%,
  COL 48%, PER 0%). Esa diferencia está cruzada con cuántos fallados con fecha por defecto
  sacó Ford de cada uno dentro de la ventana de producción (ARG 88, BRA 70, CHL 24, COL 5,
  PER 8): es exactamente el orden inverso. **No se puede leer como física del mercado.**

## 4 · El holdout se volvió a sortear, con la misma semilla

Pedido del equipo el 26-09. En v2 cambiaron el universo, las etiquetas y los estratos, así que
el recorte del holdout viejo (`restrict_test_split`) dejaba un reparto armado sobre una
población que ya no existe.

**Procedimiento** (`holdout_on_universe`, `draw.mode: universe`):
1. El universo se fija con criterios de calidad y de muestreo, sin mirar ninguna feature contra
   la etiqueta.
2. Se sortea **una vez**: `StratifiedGroupKFold(5)`, semilla 42, fold 0 = test.
3. El estrato es **evento × mercado × motor**, y un estrato de menos de 5 autos se funde con el
   de nivel superior (`composite_strata`).

Con 27 celdas y riesgos 8× distintos entre mercados, estratificar solo por evento podía dejar
de un lado, por ejemplo, a casi todos los ENG_3 fallados.

| balance dev vs test | peor diferencia |
|---|---|
| tasa de eventos | 0,316 contra 0,324 |
| motor | 0,7 pp (ENG_2) |
| mercado | 1,9 pp (COL) |
| modelo (reportado, no estratificado) | 4,9 pp (MODEL_4) |

Cada estrato manda entre el 14% y el 29% al test. Los estratos con menos de 10 autos quedan en
los extremos, porque un auto es un 10–20% del estrato.

**Lo que costó re-sortear** (`previous` en `test_split.json`):

| lado viejo → nuevo | autos |
|---|---|
| dev → dev | 132 |
| **dev → test** | **36** (13 con evento) |
| test → dev | 30 |
| test → test | 6 |
| excluido o nuevo → dev / test | 284 / 69 |
| dev o test viejo → excluido | 153 (casi todos sanos producidos después de julio de 2025) |

**36 de los 111 autos del test nuevo estaban en el dev viejo**: se miraron durante F2–F8 (EDA,
features, elección de K2). La lista queda en `previous.test_vehicles_in_old_dev`. **La
evaluación final tiene que reportar el test también sin esos 36.** Si los dos números
difieren, lo aprendido mirando la entrega 1 está inflando el test.

La alternativa, que no se tomó por pedido explícito, era extender el holdout viejo (el dev
viejo queda en dev, el test viejo en test, y solo se sortean los nuevos). Eso garantizaba un
test sin autos vistos, a costa de estratos desbalanceados.

## 5 · Los folds de CV también estratifican por mercado y motor

`make_splits` acepta `splits.stratify.extra_columns` (constantes dentro del vehículo) y
`min_stratum_size`. `panel_v2.yaml` estratifica los folds por `label × static_SalesCountry_cd ×
aux_static_Engine`. Sin `extra_columns` el reparto es el de siempre, bit a bit (chequeado).
Con R = 3, cada fold queda con 25–29 autos positivos y 127–144 filas positivas.
