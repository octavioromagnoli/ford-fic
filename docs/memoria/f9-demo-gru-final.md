# Demo de producto con el modelo final: la GRU de F10 con la perilla de falsas alarmas

**Fecha:** 2026-10-01 · **Fase:** F9 (producto, para el pitch del 02-10) · **Rama:** `feat/demo-gru` (con `main`
mergeado)
**Alcance:** entrega v2, solo dev (426 autos: 135 fallan, 291 sanos), ensamble por rango de la GRU + TripSummary +
estática completa con `label`, semillas 42, 1 y 2, folds `splits_r3`; etiqueta dura, repetición 1 y umbral exacto en
cuatro puntos de operación. **El test no se muestra**: el bundle falla si trae un vehículo de test.

**No es un candidato ni cambia ninguna métrica.** El 01-10 la GRU con etiqueta suave no se sostuvo en test y el
finalista pasó a ser la GRU de F10 ([f11-test-resultado.md](f11-test-resultado.md)). La demo usa ese modelo tal cual:
el score, el umbral y la alerta son los del ensamble del reporte v2. Es el mismo modelo que la demo usó el 27/28-09
([f9-demo-gru.md](f9-demo-gru.md), `demo-bundle-gru:v1`), ahora con la perilla de 5, 10, 15 y 20%
([f9-demo-gru-suave.md](f9-demo-gru-suave.md) explica cómo funciona).

```bash
export FORD_DATA_DIR=$PWD/data/v2 WANDB_MODE=disabled
python scripts/train.py --config configs/exp_v2all_gru_trips_estaticas_r3.yaml          # y _s1_r3, _s2_r3
python scripts/ensemble_rank.py --config configs/exp_v2all_seeds3_gru_trips_estaticas.yaml
# la demo
python scripts/build_demo_bundle.py --config configs/demo.yaml          # experiments/demo-bundle-gru-final/
python scripts/warm_demo_cache.py --config configs/demo.yaml --budgets 50    # un proceso por punto (50, 100, 150, 200)
python scripts/warm_demo_cache.py --config configs/demo.yaml --mode cache_only --prune   # después: verifica y poda
python scripts/build_demo_bundle.py --config configs/demo.yaml --publish-only   # wandb Artifact demo-bundle-gru-final
streamlit run scripts/demo_app/app.py
```

Las corridas del ensamble (`experiments/v2all-gru-trips-estaticas{,-s1,-s2}-r3/` y
`experiments/v2all-seeds3-gru-trips-estaticas/`) son las del 27-09 en esta Mac, y no se reentrenaron. El replay al 5%
da lo mismo que `demo-bundle-gru:v1`: 39 de 135 fallas anticipadas, 14 sanos con alerta de más.

## Los números por punto (dev v2)

| | 5% | 10% | 15% | 20% |
|---|---|---|---|---|
| **detección oficial (3 repeticiones)** | **28,1% ± 2,8** | **42,5% ± 1,9** | **53,1% ± 1,5** | **61,0% ± 0,3** |
| fuera de muestra: detección / falsas alarmas | 27,9 / 4,9% | 42,5 / 10,2% | 52,3 / 15,1% | 60,7 / 19,9% |
| celda mercado × motor, sin modelo | 18,0% | 32,6% | 48,1% | 62,2% |
| nulo de bolsa (media / p95) | 5,4 / 9,4% | 10,7 / 16,1% | 15,7 / 21,5% | 20,7 / 27,2% |
| umbral exacto (R1) | 0,938 | 0,902 | 0,877 | 0,844 |
| replay: fallas anticipadas, de 135 | 39 | 56 | 69 | 82 |
| replay: sanos con alerta de más, de 291 | 14 | 29 | 43 | 58 |
| replay: alertas nuevas / escalamientos | 53 / 7 | 85 / 10 | 112 / 22 | 140 / 36 |
| replay: alertas con hábitos que nombrar (aviso) | 48 de 53 | 73 de 85 | 101 de 112 | 125 de 140 |
| replay: anticipación mediana (alerta → falla) | 15 sem · 7.000 km | 16 sem · 7.200 km | 16 sem · 7.500 km | 21 sem · 8.100 km |

AUC por auto: 0,82 agrupado, 0,76 dentro del mercado y 0,66 dentro de mercado × motor.

**En test** (no se muestra en la app; [f11-test-resultado.md](f11-test-resultado.md)): este modelo detecta 31 ·
47 · 63 · 63% al 5 · 10 · 15 · 20% de falsas alarmas (tres lecturas: ~30 · 40–50 · 50–60% al 5 · 10 · 20%).
Con el umbral fijado en dev detecta 20 · 34 · 57%, con 5,8 · 8,9 · 18,2% de falsas alarmas reales. **El pitch cita
el test.** La app muestra los oficiales de dev, rotulados «medido en desarrollo», porque el replay es la flota de dev.

**La celda está cerca.** Al 20%, la celda sola da 62,2% contra 61,0% de la GRU: lo que la GRU suma sobre la
composición se ve donde se tolera poco (al 5%, 28,1% contra 18,0%).

## La semana de apertura

Es **la del 29-09-2025**, la primera con al menos 3 alertas nuevas al 5% (la misma de `demo-bundle-gru:v1`). Son 4,
las 4 de autos que fallaron después: VEH_0451, VEH_0563, VEH_0566 y VEH_0583. Dos tienen hábitos que nombrar (aviso al
conductor) y dos van al concesionario. Con la suave abría el 25-08.

## Los agentes

Sin cambios en los prompts, la política ni el verificador. Se precalentaron con `gpt-5.4-mini-2026-03-17`, sembrando
la caché con la de los bundles anteriores: los pedidos idénticos no se volvieron a pagar. Después de podar, quedan
**1.674 respuestas**.

| | semanas | eventos | textos del agente | resúmenes del agente | intentos rechazados |
|---|---|---|---|---|---|
| 5% | 36 | 60 | 60 | 36 | 6 |
| 10% | 50 | 95 | 95 | 50 | 9 |
| 15% | 53 | 134 | 134 | 53 | 11 |
| 20% | 57 | 176 | 176 | 57 | 14 |

- **No quedó ninguna plantilla:** el agente redactó los 465 textos y los 196 resúmenes, todos aprobados por el
  verificador.
- **La carrera del precalentado en paralelo pasó una vez** (10%, semana del 19-05-2025). Se rehízo en un solo
  proceso, y la pasada `--mode cache_only --prune` confirmó que todas las semanas salen enteras de la caché.

## Verificación

- `scripts/check_setup.py`: 218 chequeos en verde.
- `load_bundle` carga los cuatro puntos con el modelo nuevo en la meta (`v2all-seeds3-gru-trips-estaticas`).
- AppTest de Streamlit, sin clave de OpenAI y con `DEMO_LLM_MODE=cache_only`: la app, los cuatro puntos de la perilla
  y las tres pantallas (bandeja, vehículo, resultados) corren sin excepciones, y ningún texto nombra a la etiqueta
  suave.

## Decisiones

1. **El ensamble del reporte v2 (folds `splits_r3`, semillas 42 / 1 / 2)**, el mismo que se midió en test. No se
   reentrenó: la GRU no es bit a bit entre corridas ([f11-test-resultado.md](f11-test-resultado.md) §2), y estas son
   las corridas de las que salieron los números de dev.
2. **Un bundle nuevo, `demo-bundle-gru-final`.** `demo-bundle-gru:v1` es el mismo modelo, pero con el formato de un
   solo punto, que el código de la perilla rechaza. `demo-bundle-gru-suave` no se toca.
3. **La perilla y los textos de la app no cambian.** El nombre y la familia del modelo viajan en la meta del bundle
   («GRU, red recurrente, ensamble de 3 semillas»).

## Límites

- Los de [f9-demo-gru.md](f9-demo-gru.md) siguen valiendo:
  - el porqué es una comparación con la flota;
  - parte de lo que se detecta es composición;
  - la muestra está enriquecida en fallas;
  - la efectividad de los avisos no está medida.
- **No le gana con evidencia a la celda mercado × motor**, ni en dev ni en test (+27,3 [−0,8; 46,1]).
- **(a′) +0,001:** el modelo sabe sobre todo *qué auto* y poco *cuándo*, como todos los de v2.
