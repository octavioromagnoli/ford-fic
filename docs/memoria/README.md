# Memoria del proyecto

Lo que cualquiera —persona o agente— necesita saber sobre **los datos y las
decisiones** antes de escribir código, y que no se deduce leyendo el repo.

`CLAUDE.md` dice cómo se trabaja. Acá está qué encontramos y qué decidimos.

## Reglas

- **Un archivo = un hecho.** Kebab-case, prefijo de la fase que lo produjo
  (`f1-...`). Si un hecho cambia, se edita ese archivo; no se apila un post-scriptum.
- **Cada hecho trae su evidencia**: el número, y el comando que lo reproduce. Un
  hallazgo sin forma de re-verificarlo es un rumor.
- **Las decisiones van en [`decisiones.md`](decisiones.md)**, con fecha y motivo.
  El motivo importa más que la decisión: sin él, el que venga la revierte sin saber.
- **Esto se versiona.** `.claude/` y `.agents/` están en `.gitignore` (son
  artefactos locales de cada uno); la memoria compartida vive acá adentro a propósito.
- Los datos crudos NO se citan textualmente acá: van agregados o como estadística.

## Índice

| Archivo | Qué contesta |
|---|---|
| [f9-entrega-v2.md](f9-entrega-v2.md) | **La entrega v2 de Ford (26-09):** qué archivos cambiaron (los de sanos son byte a byte los de la entrega 1), las cinco trampas de formato (columna renombrada, `ProductionDay` +538 d, una fila por evento, filas repetidas, extracción asimétrica), qué pasó con cada código y por qué la fecha nueva cae ~2 semanas después de la intervención |
| [f9-universo-v2.md](f9-universo-v2.md) | **Por qué el universo v2 son 557 vehículos (446 dev / 111 test):** la selección pasó del mercado a la fecha de producción (93 fallados de 2024 sin sanos; 0 fallados producidos desde agosto de 2025 contra ~50 esperados), cómo se re-sorteó el holdout (semilla 42, evento × mercado × motor), los 36 autos del test que estaban en el dev viejo, y por qué se probó sacar la ventana (990) y se volvió |
| [f9-eda-v2.md](f9-eda-v2.md) | **La EDA sobre el dev v2, dentro del mercado:** qué hallazgos de la entrega 1 se sostienen (reloj en días, idle, marcador cortado, confusor calendario), cuáles ya no pasan (ventana del registro, velocidad, bajo régimen entre los que se mueven, "sin trayectoria") y qué es nuevo (ENG_3 en Brasil, altura, cohorte de venta) |
| [f9-remedicion-v2.md](f9-remedicion-v2.md) | **Los modelos de F3 re-medidos sobre v2:** survival stacking aprueba (a0) y (a′) igual que en v1; la separación sube (ROC por fila 0,59 → 0,71) pero dentro de mercado × motor el uso ordena autos con AUC ~0,58, como en v1; motor y modelo aportan la tasa de su celda. **K2 sobre v2:** sin su ventana falla (a′); survival stacking detecta ~11–14% al 5% de falsas alarmas (no el 15–17% de la entrega 1) |
| [f9-remedicion-completa-v2.md](f9-remedicion-completa-v2.md) | **Los 42 modelos re-corridos sobre v2 (26-09)**, con la detección por auto de 2% a 30% de falsas alarmas: los secuenciales con TripSummary (CNN-LSTM, GRU) le ganan a survival stacking, pero ninguno le gana con evidencia a la tasa de la celda mercado × motor; por qué SS no empeoró respecto de la entrega 1 (bajó el nulo); varianza por semilla; anticipación (~7.500 km) y cuánto saben del *cuándo* |
| [f10-sweep-gru.md](f10-sweep-gru.md) | **El sweep bayesiano de la GRU (27-09):** 130 trials de Optuna; el mejor (t112, 0,492) re-medido con semillas nuevas empata con la configuración de hoy (0,469 contra 0,471), así que se queda la de hoy. La semilla mueve más que los hiperparámetros, y por qué la celda mercado × motor es un piso tan alto (3 celdas, 117 de 135 fallados) |
| [f11-gru-objetivo-suave.md](f11-gru-objetivo-suave.md) | **El mejor modelo sobre dev v2 (28-09):** la GRU de hoy entrenada con etiqueta suave (0,15) en los cortes lejos del evento de los autos que fallan. Confirmación preregistrada con folds y semillas nuevas: **+9,6 puntos de detección media al 5–20% de falsas alarmas, IC95 [3,0; 14,4], p = 0,0015** (34,6 · 48,6 · 60,2 · 70,1%); primer modelo que le gana con evidencia a la celda mercado × motor; aprueba (a0), (a′) y (b); en qué se apoya |
| [f11-preregistro-objetivo-suave.md](f11-preregistro-objetivo-suave.md) | El preregistro de F11: las ~20 variantes exploradas antes (suavizar el score, MIL, recalibrar por celda, historia completa, más canales, etiqueta suave), el candidato único y el criterio primario |
| [f1-datos-reales.md](f1-datos-reales.md) | Qué hay en `data/raw/`, con qué esquema y cómo joinea |
| [f1-clones-vehiculos.md](f1-clones-vehiculos.md) | 13 vehículos contados cuatro veces, y por qué rompen el split |
| [f1-sesgo-eng3.md](f1-sesgo-eng3.md) | `ENG_3` no aparece entre los fallados: sesgo de muestreo |
| [f1-anclaje-temporal.md](f1-anclaje-temporal.md) | El eje de días sí se puede anclar al de km |
| [f1-senal-postratamiento.md](f1-senal-postratamiento.md) | Cuánto separan `Message`, `Acumulation` y las regeneraciones |
| [f1-calidad-odometro.md](f1-calidad-odometro.md) | Nulos, retrocesos y desfases del eje de odómetro |
| [f2-union-y-holdout-dev-test.md](f2-union-y-holdout-dev-test.md) | Cómo une (y cómo NO une) el join, y el test 80/20 congelado |
| [f2-diccionario-trips-incompleto.md](f2-diccionario-trips-incompleto.md) | `TripSummary` trae 25 de las 40 columnas del anexo: qué features mueren |
| [f2-columnas-airfilter-airregeneration.md](f2-columnas-airfilter-airregeneration.md) | El DPF estaba en `trips` con otro nombre: la familia B se puede construir |
| [f2-identificationdate-igual-a-venta.md](f2-identificationdate-igual-a-venta.md) | El 78% de los eventos no tiene fecha utilizable (resuelto por el de abajo) |
| [f2-universo-fecha-usable.md](f2-universo-fecha-usable.md) | Por qué el estudio son 364 vehículos y no 1081, y cómo se recortó sin re-sortear |
| [f2-calidad-columnas-dev.md](f2-calidad-columnas-dev.md) | Tres defectos de columna que la tabla de nulos no muestra |
| [f2-eda-revision-y-features.md](f2-eda-revision-y-features.md) | **Revisión del EDA (18-09) y panel v1**: el marcador `Regenerations` cortado, el confusor calendario, los idle de 0 km, la terna, el emparejado y cuánta señal hay de verdad |
| [f3-cnn-lstm-tutora.md](f3-cnn-lstm-tutora.md) | La solución de la tutora (CNN-LSTM, rama dinámica + estática) como baseline: cómo se interpretó, cuánto da y qué auditorías pasó |
| [f2-fechas-formato-mixto.md](f2-fechas-formato-mixto.md) | Las fechas no tienen nulos: el 0,3% era parseo de dos formatos mezclados, y también rompía el dedupe de `signals` |
| [f2-umbral-regeneraciones.md](f2-umbral-regeneraciones.md) | Por qué las caídas de 5 puntos son ruido y el detector pasa a exigir 15 |
| [f3-mil-agregacion-vehiculo.md](f3-mil-agregacion-vehiculo.md) | Medir por vehículo y no por corte (MIL): cuánto sube el lift de verdad y cuánto es el tamaño de la bolsa |
| [f3-timesfm-zeroshot.md](f3-timesfm-zeroshot.md) | TimesFM zero-shot en el panel v1: no le gana a la tasa base, pero señaló el desvío respecto de la historia del vehículo |
| [f3-survival-stacking.md](f3-survival-stacking.md) | Supervivencia en tiempo discreto sobre el panel: empata en PR-AUC pero calibrado y con el doble de detección; por qué el efecto aleatorio por vehículo no paga; y por qué la auditoría (a) de §0.4 no es un null en este panel |
| [f3-proceso-gamma.md](f3-proceso-gamma.md) | Proceso gamma de degradación: por qué no se implementó — no hay carga irreversible medible a estos kilometrajes |
| [f3-ordinal-horizonte.md](f3-ordinal-horizonte.md) | Target ordinal: dos variantes de bins (la restringida no aporta información), el costo como barrido de C_FN/C_FP y la permutación intra-vehículo que ningún modelo del repo supera |
| [f3-piso-posicional.md](f3-piso-posicional.md) | El odómetro solo le gana a los cuatro finalistas en PR-AUC por fila y en las tres métricas del eje "cuándo": qué métricas quedan descalificadas y cuáles dos sobreviven. Con R=3 (21-09): el 17,0% del CNN-LSTM era la repetición 0 — da 11,9% ± 3,6, igual que el control |
| [f3-preregistro-landmark-ensamble.md](f3-preregistro-landmark-ensamble.md) | Lista cerrada de las corridas de landmarking y del ensamble de finalistas (máximo 4), con qué métrica decide cada una y qué resultado la descarta, escrita antes de correrlas |
| [f3-desvio-historia.md](f3-desvio-historia.md) | El desvío respecto de la historia previa del vehículo (la pista de TimesFM, sin TimesFM) sobre survival stacking: ROC univariado 0,598, pero (a′) mejora 0,98× el desvío combinado y la anticipación pierde su estabilidad. No entra |
| [f3-reloj-y-ventana-del-evento.md](f3-reloj-y-ventana-del-evento.md) | **En qué reloj ocurre el evento y en qué ventana se registra** (22-09, Fase 1 del cure model): días y no km; los eventos solo existen entre el 03-09-2025 y el 11-03-2026, así que la censura de un sano es su exposición dentro de esa ventana; el riesgo arranca en la venta; sin trayectoria previa, con un rasgo temprano débil en la cola |
| [f3-preregistro-cure.md](f3-preregistro-cure.md) | Lista cerrada del cure model sobre hitos post-venta, escrita antes de construir el panel y de medir D1/D2: reloj, ventana y censura; las cuatro features fijas; P0 (pesos unitarios), P1 (Firth + FLIC), P2 (TabPFN v2, condicional); qué decide, los nulos y las paradas |
| [f3-mejor-modelo-a-la-fecha.md](f3-mejor-modelo-a-la-fecha.md) | **Cuál es el mejor modelo hoy y por qué** (22-09, F6): survival stacking con la ventana del registro en km (K2), con la tabla de las dos métricas que valen en las dos etiquetas, el cure model explicado en criollo y los límites para el pitch |
| [f3-cure-model.md](f3-cure-model.md) | **Resultado del cure model (22-09):** P0 le gana a su nulo (p = 0,0045) pero no al piso de producción, y empata con un solo número de uso (−km/L); P1 empata con P0. No se adopta nada y el finalista no cambia |
| [f3-ensamble-e1-e2.md](f3-ensamble-e1-e2.md) | **E1 y E2 del preregistro (22-09):** el ensamble por rango de survival stacking y CNN-LSTM, con y sin bagging por vehículo, pierde contra survival stacking en lift por vehículo y no le gana en detección. El finalista no cambia |
| [f5-preregistro-incidencia-externa.md](f5-preregistro-incidencia-externa.md) | Lista cerrada de la incidencia aprendida con los fallados sin fecha (F5 §3.2), escrita después de los conteos y antes de mirar una feature contra la etiqueta: quién entra a la fuente, elegibilidad simétrica, el modelo de dos pendientes (Riley), las compuertas y qué decide sobre dev |
| [f5-incidencia-externa.md](f5-incidencia-externa.md) | **Resultado de la incidencia externa (22-09):** el índice del rasgo temprano no separa a los fallados sin fecha de sus sanos en CNTRY_1/2/5 (AUC 0,513 [0,445; 0,585]); la compuerta G2 para y nada se aplica a dev. Solo viajes bajo régimen y uso apuntan como en dev; en CNTRY_1 los fallados no tienen la firma del filtro |
| [f5-preregistro-ss-post-venta.md](f5-preregistro-ss-post-venta.md) | Lista cerrada de survival stacking en días post-venta con la ventana del registro (F5 §3.3), escrita después de los conteos y antes de implementar: las mismas filas con el tramo en riesgo por fila, el PEM con offset de exposición, la etiqueta corregida que decide y los dos pisos |
| [f5-ss-post-venta.md](f5-ss-post-venta.md) | **Resultado de survival stacking post-venta (22-09):** con la etiqueta corregida empata en detección y pierde en lift por vehículo (1,38× contra 1,66×); no se adopta. Pero la (b) de calendario baja de +0,049 a +0,016 (el atajo del finalista era exposición al registro), y con la etiqueta corregida el finalista detecta 11,9%, no 15,7% |
| [f6-preregistro-deteccion-vehiculo.md](f6-preregistro-deteccion-vehiculo.md) | Lista cerrada de F6 (más autos detectados con anticipación), escrita después del embudo y antes de implementar: K1 (media acumulada causal del score), K2 (survival stacking con la ventana del registro en km y horizonte completo), K3 (los dos), y las seis condiciones que deciden con la etiqueta corregida, incluido el nulo de tamaño de bolsa para la detección |
| [f7-preregistro-varianza-k2.md](f7-preregistro-varianza-k2.md) | Lista cerrada de F7 (bajar la varianza de K2): P1 embolsado, P2 monótono con el mapa del 18-09, P3 hazard logístico con cinco covariables; la capa de decisión fuera de la lista, y grafos/features/TabPFN descartados con su motivo |
| [f7-varianza-k2.md](f7-varianza-k2.md) | **Resultado de F7 (24-09): K2 sigue.** Ninguno cumple las seis condiciones. El embolsado detecta 7/7/7 de 45 (15,6%) contra 7/10/6 de K2: el 17,0% es la lectura optimista y el pitch cita ~15–17%. El monótono sube media y desvío; el logístico de cinco covariables pierde contra el azar |
| [f6-deteccion-vehiculo.md](f6-deteccion-vehiculo.md) | **Resultado de F6 (22-09): K2 es el nuevo finalista.** Con la etiqueta corregida detecta 17,0% ± 3,8 contra 11,9% ± 2,8, con el mismo lift y la misma calibración, y la (b) de calendario baja de +0,049 a +0,020. Mejora modesta (+2,3 autos de 45; el bootstrap por vehículo no la separa del cero) y con la etiqueta dura empata. Suavizar el score hacia atrás (K1, K3) no suma |
| [f8-datar-eventos-fase0.md](f8-datar-eventos-fase0.md) | **Fechar a los fallados sin fecha desde la telemetría (24-09, exploratoria): no se puede.** Ninguna marca de intervención (aceite, días sin uso, DPF con motor apagado, mensajes) fecha el evento a ±14 d. La caída del DPF se concentra cerca del evento pero es demasiado frecuente. Como firma de "fallado" le queda algo por encima del largo de la ventana, que solo ya da AUC 0,76 |
| [f8-capa-decision-k2.md](f8-capa-decision-k2.md) | **Capa de decisión sobre K2 (24-09, sin presupuesto):** el 5% se sostiene fuera de muestra (3,5% de falsas alarmas realizadas, 15,6% de detección); a 10% detecta 26,7% con el mismo exceso sobre el nulo; al 2% es azar. Con garantía de Neyman-Pearson de ≤ 10% (95% de confianza) detecta 17,0%. La curva y las frases para el pitch |
| [f4-explicabilidad-k2.md](f4-explicabilidad-k2.md) | **Por qué un auto tiene riesgo alto (24-09, F4, no es candidato):** SHAP sobre K2 reconstruido bit a bit; gana V3 (TreeSHAP del hazard promedio de 3 repeticiones) por estabilidad. 10 de 15 accionables con hipótesis coinciden con la física del DPF y el control de calendario no da vuelta ningún signo; 4 van al revés y quedan fuera del texto. El mensaje al cliente, los casos y la sección "Por qué" del dashboard |
| [f8-costos-k2.md](f8-costos-k2.md) | **Cuánto vale operar K2 (24-09, reporte):** escenarios de costo con fuente pública (diagnóstico, limpieza o reemplazo del DPF, grúa, aviso remoto, flota comercial) × efectividad de la prevención × prevalencia real. Con fallas baratas, no alertar; en la zona validada ahorra 0–6%; los ahorros grandes piden 15–35% de falsas alarmas (sin validar); si la alerta es casi gratis, alertar a todos gana |
| [f3-gru-secuencial.md](f3-gru-secuencial.md) | GRU sobre la ventana en bins de km: `last` le gana por poco al LightGBM (p = 0,05), la atención pierde y la ventana larga no ayuda |
| [decisiones.md](decisiones.md) | Qué se decidió, cuándo y por qué. **Su primera entrada cierra F3**: el techo de cohorte, el piso posicional, por qué el PR-AUC por fila mide *qué auto* y no *cuándo*, y por qué el finalista se elige por el punto de operación |
| [../f2-feature-engineering-candidatas.md](../f2-feature-engineering-candidatas.md) | Candidatas de feature engineering medidas el 17-09. Lo que se adoptó y lo que se retiró está en el archivo de arriba |

## Cómo se reproduce todo esto

```bash
# --- entrega v2 (26-09-2026): lo vigente ---------------------------------------------
python scripts/compare_deliveries.py --config configs/data/compare_deliveries.yaml  # entrega 1 contra v2 (f9-entrega-v2.md)
python scripts/make_test_split.py --config configs/data/test_split.yaml --force     # universo + holdout v2 (f9-universo-v2.md)
python scripts/build_city_elevation.py --config configs/data/city_geocode.yaml      # altura de la ciudad de venta (red)
python scripts/build_eda_cache.py --config configs/data/eda_cache_v2.yaml && python scripts/eda_gaps.py --config configs/data/eda_cache_v2.yaml
python scripts/eda_v2.py --config configs/eda_v2.yaml                               # EDA dentro del mercado (f9-eda-v2.md)
python scripts/build_dataset.py --config configs/data/panel_v2.yaml                 # panel v2 (diseño v1, referencia − 21 d)
WANDB_MODE=disabled python scripts/train.py --config configs/exp_v2_ss_r3.yaml     # re-medición (f9-remedicion-v2.md)
python scripts/decision_layer.py --config configs/exp_decision_v2_ss.yaml           # detección al 5/10% con su nulo (idem _k2)
python scripts/audit_vehicle_strata.py --config configs/audit_vehicle_strata_v2.yaml # AUC por auto dentro de mercado × motor
# variante SIN la ventana de producción (no vigente, f9-universo-v2.md §6): los mismos comandos con *_sinventana*
python scripts/make_test_split.py --config configs/data/test_split_sinventana.yaml --force

# --- entrega 1 (15-09-2026): lo de abajo mide la entrega 1; sus artefactos están en data/processed-v1/ ---
python scripts/eda_raw.py          # deja los CSV en experiments/eda/
python scripts/make_test_split.py  # auditoría del join + holdout dev/test congelado
python scripts/build_eda_cache.py  # cache dev-only del EDA (experiments/eda/dev/)
python scripts/eda_gaps.py         # complemento del EDA: factibilidad, perfil alineado al evento, calendario (experiments/eda/dev/gaps/)
python scripts/build_dataset.py --config configs/data/panel_v1.yaml   # panel real + splits sobre dev + panel_meta.json
python scripts/log_panel_artifact.py --config configs/data/panel_v1.yaml  # publica panel-v1 y test-split como wandb Artifacts
python scripts/train.py --config configs/exp_baserate.yaml            # piso contra el panel real
python scripts/train.py --config configs/exp_lgbm_panel_v1_mil.yaml    # el mismo modelo, medido por vehículo
python scripts/audit_mil_bagsize.py f3-lgbm-panel-v1-mil              # ¿el lift por vehículo es señal o tamaño de bolsa?
python scripts/audit_model.py --config configs/exp_<x>.yaml           # las auditorías obligatorias de F3 §0.4
python scripts/audit_ordinal_horizon.py --config configs/exp_<x>.yaml  # los dos nulos: global e intra-vehículo
python scripts/audit_gamma_monotonia.py --config configs/data/gamma_monotonia.yaml  # ¿hay carga irreversible? (paso 1 del proceso gamma)
python scripts/build_positional_panel.py --config configs/data/panel_positional.yaml  # panel de una sola feature: el odómetro
python scripts/audit_positional_floor.py f3-survival-stacking         # ¿le gana al odómetro pelado? (piso del eje "cuándo")
python scripts/rescore_run.py f3-cnn-lstm-r3-regen15                # completa una corrida vieja desde sus predicciones, sin reentrenar
python scripts/audit_event_clock.py --config configs/data/event_clock.yaml  # reloj y ventana del evento (Fase 1 del cure model)
python scripts/build_landmark_panel.py --config configs/data/panel_landmark_ps.yaml  # panel de hitos post-venta (Fase 3 del cure model)
python scripts/make_splits.py --config configs/data/panel_landmark_ps.yaml          # folds del cure model: extienden splits_r3.json (Fase 7)
python scripts/train.py --config configs/exp_cure_p0_unitweight.yaml               # cure model P0 (y exp_cure_p1_firth.yaml)
python scripts/audit_cure.py --config configs/exp_cure_p0_unitweight.yaml          # sus auditorías y el veredicto (f3-cure-model.md)
python scripts/ensemble_rank.py --config configs/exp_ens_e1.yaml --audit       # ensamble E1 (y exp_ens_e2.yaml), f3-ensamble-e1-e2.md
python scripts/build_external_panel.py --config configs/data/panel_external_incidence.yaml  # fuente de la incidencia externa (--counts-only: Fase 1)
python scripts/fit_external_incidence.py --config configs/exp_ext_incidence_a30.yaml       # G1, G2 (paró) y diagnóstico, f5-incidencia-externa.md
python scripts/build_window_survival_panel.py --config configs/data/panel_survival_ps.yaml  # panel v1 en días con la ventana (--counts-only: Fase 1)
python scripts/eval_window_label.py --config configs/exp_ss_post_venta_r3.yaml --diagnose   # etiqueta dura y corregida, veredicto, f5-ss-post-venta.md
python scripts/explain_k2.py --config configs/explain_k2.yaml       # explicabilidad de K2 (SHAP, mensaje al cliente), f4-explicabilidad-k2.md
python scripts/check_setup.py      # chequeos del harness
```

> Ojo: `notebooks/eda-exhaustivo-dev.ipynb` §6.2, §6.3 y §7.4 muestran `regen_per_1000km`
> como la señal más fuerte. Está medido con el marcador `Regenerations`, que se corta el
> 25-05-2026: es exposición al calendario, no física. La versión vigente es
> [f2-eda-revision-y-features.md](f2-eda-revision-y-features.md) §2.1.

Los hallazgos `f2-*` de arriba salen del EDA exhaustivo sobre dev:
[`notebooks/eda-exhaustivo-dev.ipynb`](../../notebooks/eda-exhaustivo-dev.ipynb)
(ejecutado, con sus figuras), y su versión navegable en
`streamlit run scripts/dashboard_eda.py`. Los dos leen el mismo cache, así que el
filtro a dev se decide en un solo lugar.
