# Incidencia aprendida con los fallados sin fecha: el rasgo temprano no se replica en la fuente

**Fecha:** 2026-09-22 · **Fase:** F5 (§3.2 del doc de candidatos) · **Rama:** `exp/incidencia-externa`
**Alcance:** solo la fuente, que son los excluidos del universo que el sorteo padre puso del
lado dev, en CNTRY_1, CNTRY_2 y CNTRY_5:
- 541 vehículos;
- 283 elegibles, con 114 fallados.

**Dev no se tocó:** la compuerta paró antes de aplicar nada. **Test tampoco**, ni los 143
excluidos del lado test del padre.

**Preregistro:** [f5-preregistro-incidencia-externa.md](f5-preregistro-incidencia-externa.md)
(3abe73d). La implementación y los chequeos se commitearon antes de correr las compuertas
(ea24f65).

## En una línea

El índice del rasgo temprano (las cuatro features de P0 con sus signos) **no separa a los
fallados sin fecha de sus sanos** en los tres mercados nuevos: **AUC 0,513 [0,445; 0,585]**.
- La compuerta G2 del preregistro para. La incidencia no se congela, no se aplica a dev, y
  survival stacking con la covariable externa (el paso 2) no corre.
- **El finalista sigue siendo survival stacking.**

## Qué se hizo

- **La fuente** (`src/data/external.py`, `scripts/build_external_panel.py`).
  - Se reprodujo el sorteo de los 1081 contra la huella congelada (`ba9aa4d290bc6610`).
  - Se tomaron los excluidos del lado dev, en los mercados con fallados y sanos.
  - El puente de calendario vale ahí (origen 20107, IQR 0 d), y ningún vehículo comparte
    viajes con dev o con test.
- **Elegibilidad simétrica.** La exposición potencial en la ventana del registro, calculada
  desde la fecha de venta, es ≥ 90 d. Vale igual para fallados y sanos.
  - Deja 114 fallados y 169 sanos, no los ~290 eventos que suponía el doc F5.
  - Por Riley, con eso entran 2–3 parámetros. Por eso el primario tenía dos pendientes: el
    índice y km/día.
- **Features:** las de los primeros 30 días post-venta. Son las cuatro del índice físico,
  desviadas contra la mediana por mes de los sanos de la fuente, más log(1 + km/día).
- **Compuertas en la fuente, antes de tocar dev:**
  - G1 (qué es "fallado" sin fecha): descriptiva;
  - G2 (el índice apunta igual): para si el IC95 de su AUC toca 0,5.

## Los números

**G2 · el índice en la fuente** (283 elegibles, bootstrap de 2.000):

| | AUC | fallados · sanos |
|---|---|---|
| **índice s (P0)** | **0,513 [0,445; 0,585]** | 114 · 169 |
| CNTRY_1 | 0,494 | 61 · 73 |
| CNTRY_2 | 0,517 | 49 · 79 |
| CNTRY_5 | 0,382 | 4 · 17 |
| −km/día | 0,525 | |

**Cada feature con el signo de P0** (> 0,5 = en la dirección de dev):

| feature (signo) | cruda | desvío por mes (primario) | desvío por mercado × mes | CNTRY_1 | CNTRY_2 |
|---|---|---|---|---|---|
| idle por 1.000 km (+) | 0,428 | 0,431 | 0,423 | 0,426 | 0,464 |
| viajes bajo régimen (+) | **0,584** | **0,576** | **0,554** | 0,550 | 0,570 |
| velocidad (−) | 0,509 | 0,505 | 0,509 | 0,523 | 0,523 |
| refrigerante al final (−) | 0,442 | 0,437 | 0,411 | 0,455 | 0,371 |

Medianas crudas entre los elegibles, de sanos contra fallados:
- idle: 36,7 contra 30,1 por 1.000 km;
- bajo régimen: 0,27 contra 0,34;
- velocidad: 26,2 contra 26,1 km/h;
- refrigerante: 83,8 contra 85,3 °C.

Dos features van al revés que en dev, una no se mueve y una va a favor: sumadas, se cancelan.

**G1 · mensajes de filtro sobre toda la telemetría post-venta** (por 1.000 km del odómetro de
`signals`):

| grupo | fallados · sanos | Full: fallados / sanos | Overloaded: fallados / sanos |
|---|---|---|---|
| CNTRY_1 | 99 · 151 | 3,08 / 3,38 = **0,91×** | 0,48 / 0,41 = **1,19×** |
| CNTRY_2 | 88 · 154 | 26,0 / 16,2 = **1,61×** | 3,11 / 1,68 = **1,85×** |
| CNTRY_5 | 12 · 30 | 0 / 1,60 = 0× | 0 / 0 (indefinida) |
| fuente | 199 · 335 | 1,57× | 1,84× |
| CNTRY_3/4 con fecha por defecto (sin sanos propios) | 33 · — | 5,03 | 0,14 |

F1 midió 1,81× y 2,63× con todos los vehículos juntos
([f1-senal-postratamiento.md](f1-senal-postratamiento.md)).
- **En CNTRY_2 los fallados sin fecha tienen la firma del filtro; en CNTRY_1 casi no.**
- Por la letra del preregistro no se marca ningún mercado:
  - en CNTRY_1 la razón de Overloaded es 1,19;
  - en CNTRY_5 la de Overloaded es 0/0.
- Pero CNTRY_1 queda sin la firma de Full.

## Por qué el negativo es creíble

Diagnóstico agregado **después** de que G2 parara, para descartar un error de cuenta. **No
estaba preregistrado, no decide nada y solo usa la fuente** (`post_stop_diagnostics` en
`scripts/fit_external_incidence.py`):

- **No es la normalización.** Las features crudas, el desvío por mes y el desvío por mercado ×
  mes dan las mismas direcciones (tabla de arriba).
- **No es un mercado.** El patrón es el mismo en los dos mercados grandes. CNTRY_5 tiene 4
  fallados elegibles.
- **No hay atajo de calendario en la fuente.** −`ProductionDay` da 0,455 y −fecha de venta
  0,511: la elegibilidad simétrica hizo su trabajo.
- **No son los autos quietos ni la elegibilidad.**
  - Sin los quietos (< 100 km en 30 días; 16 de 114 fallados y 12 de 169 sanos), el índice da
    0,490.
  - Con todos los fallados, elegibles o no, da 0,494.
- **Lo que la fuente aprende es uso.** La logística del primario, ajustada igual pero **no
  congelada**, da:
  - índice: a = −0,09 [−0,22; 0,04];
  - km/día: b = −0,34 [−0,66; −0,02].

  El peso del índice es cero o negativo; el de km/día va en la dirección de dev (menos uso,
  más riesgo) y su IC apenas excluye el 0.

## Qué significa

1. **El rasgo temprano de dev, como índice, no se transfiere a CNTRY_1/2/5.** Hay dos
   lecturas, y estos datos no las separan:
   - **"Fallado" con fecha por defecto es otra cosa**, al menos en CNTRY_1, donde los fallados
     no tienen exceso de mensajes de filtro. Es la trampa 1 del doc F5, y la pregunta 1 para
     Ford.
   - **El rasgo es de CNTRY_3/4, o en parte ruido.** En dev, P0 le ganó a su nulo pero no al
     piso de producción ([f3-cure-model.md](f3-cure-model.md)). Y en CNTRY_2, donde la firma
     del filtro sí está, el índice tampoco separa (0,517).
2. **Lo único que apunta como en dev** son los viajes bajo régimen (0,58) y el uso (−km/día
   0,525, b < 0).
   - Son justo las dos cosas que la Fase 1 dijo que sobreviven
     ([f3-reloj-y-ventana-del-evento.md](f3-reloj-y-ventana-del-evento.md) §6).
   - Pero es una lectura por feature, hecha después de mirar. No se puede afirmar sin un
     preregistro nuevo, y ese preregistro ya cargaría con esta mirada.
3. **La palanca más grande del doc F5 queda cerrada con estos datos:** aprender *qué auto* con
   más eventos. Con ella muere el §3.4 (survival stacking post-venta + incidencia externa),
   que dependía de que A-solo pasara.
   - Se reabre solo si Ford explica qué es `IdentificationDate == daysUntilSale`, y ese
     "fallado" resulta ser el mismo evento que el de CNTRY_3/4.
4. **Para el pitch:** el rasgo de los primeros 30 días no generaliza a otros mercados. Es otra
   razón para no vender "alertamos desde el primer mes post-venta".

## Desviaciones y lo que no corrió

- **El diagnóstico posterior a la parada** no estaba preregistrado (sección de arriba). No
  cambia el veredicto: solo descarta un error de cuenta.
- **Nada corrió sobre dev:**
  - A-solo, sus auditorías y el veredicto contra P0;
  - I1 (salvo las pendientes, que quedan como diagnóstico), I2, I3 e I4;
  - el paso 2.

  `configs/exp_ext_incidence_a30.yaml` trae los bloques de `train.py` y `audit_cure.py` tal
  como se preregistraron, sin correr. Por eso no hay entrada en `results/`: no hay predicciones
  sobre dev.
- **wandb:** todo corrió con `WANDB_MODE=disabled`. El informe de la fuente está en
  `experiments/f5-ext-incidence-a30/source_fit.json`, que no se versiona.
- **Código que queda y sirve para otras corridas:**
  - `src/data/external.py` y la ventana de features fija de `src/data/landmark.py`
    (`feature_window_days`, `usage_feature`, `aux_potential_exposure_days`). El panel del cure
    model se reconstruye idéntico, salvo dos `aux_` nuevas;
  - `src/models/incidence.py` y `external_incidence` en el registry;
  - `incidence: fixed_weight` en `cure_mixture`;
  - `scripts/build_external_panel.py` y `scripts/fit_external_incidence.py`;
  - 9 chequeos nuevos en `check_setup.py` (148 en verde). De 7 mutaciones se detectan 6: la
    guarda de fuga de `select_external` es redundante con el filtro de excluidos.

## Qué queda abierto (ideas, no corridas)

- **Preguntas para Ford**, que ahora pesan más:
  - qué es la fecha por defecto;
  - si la ventana del registro es la misma en todos los mercados;
  - el error de SQL de la query de eventos.
- **Una incidencia con viajes bajo régimen y km/día solos** sería un candidato nuevo, con su
  preregistro. Esta ficha ya miró esas dos features en la fuente, así que la fuente no serviría
  para decidirlo: habría que decidirlo sobre dev con su propia regla.
- **La verosimilitud completa** (censura por intervalo, EM de autoconsistencia) comparte la
  premisa que falló acá. No tiene sentido sin la respuesta de Ford.
- **La palanca que sigue abierta** es el §3.3 del doc F5: survival stacking en el reloj
  post-venta, con el conjunto en riesgo de la ventana.

## Cómo se reproduce

```bash
python scripts/build_external_panel.py --config configs/data/panel_external_incidence.yaml --counts-only  # Fase 1
python scripts/build_external_panel.py --config configs/data/panel_external_incidence.yaml                # panel de la fuente
python scripts/build_landmark_panel.py --config configs/data/panel_landmark_ps_w30.yaml                   # dev, ventana fija (no se usó)
WANDB_MODE=disabled python scripts/fit_external_incidence.py --config configs/exp_ext_incidence_a30.yaml  # G1, G2 (para) y diagnóstico
python scripts/check_setup.py                                                                              # 148 chequeos
```

G1 lee `signals` una vez y deja el cache en `data/interim/external_g1_messages.parquet`.
