# Explicabilidad de K2: por qué un auto tiene riesgo alto y qué parte de su uso lo explica

**Fecha:** 2026-09-24 · **Fase:** F4 (dashboard) · **Rama:** `feat/explicabilidad-k2`
**Alcance:** K2 (`f6-ss-hw-r3`), solo dev, las 3 repeticiones de `splits_r3.json`, más 6 repeticiones
con otra semilla para las réplicas de V3. **Test sin tocar**: una guarda falla si algún output nombra
un vehículo de test.
**Preregistro:** `configs/explain_k2.yaml` (3882919), commiteado antes de reentrenar un fold y de
mirar un SHAP.

**No es un candidato.** Explica al finalista, no gasta presupuesto de comparaciones y no cambia
ninguna métrica oficial: el score, el umbral y las alertas son los de K2 tal cual.

```bash
export FORD_DATA_DIR=$PWD/data/rebuild-0921 WANDB_MODE=disabled
python scripts/train.py --config configs/exp_ss_hw_r3.yaml            # si no está experiments/f6-ss-hw-r3
python scripts/explain_k2.py --config configs/explain_k2.yaml         # ~90 s; --figures-only rehace figuras y cases.md
python scripts/decision_layer.py --config configs/exp_decision_k2.yaml   # la usa el resto del dashboard
streamlit run scripts/dashboard_k2/app.py                               # Vehículo → sección "Por qué"
```

Deja en `experiments/explain-k2/` `shap_rows.parquet`, `shap_vehicle.parquet`, `messages.parquet`,
`explain_eval.json`, `cases.md`, `splits_extra.json` y `figures/` (beeswarm, estabilidad y un
waterfall por caso).

## En una línea

**Gana V3 (TreeSHAP del hazard promediado entre 3 repeticiones).** Las tres variantes elegibles
pasan la fidelidad, y V3 es la más estable: Jaccard del top 3 de 0,73, contra 0,57 de V1 y 0,48 de V2.
- **Lo que K2 aprendió coincide con la física del DPF en 10 de las 15 accionables con hipótesis**:
  motor que no llega a temperatura, idle, velocidad baja, motor frío al arrancar.
- **El control de calendario no da vuelta ningún signo.** "Motor frío" es el auto, no el invierno.
- **Cuatro van al revés de la física** (viajes más largos y temperatura máxima más alta ⇒ más
  riesgo) y quedan fuera del texto al cliente, igual que las cinco sin hipótesis.
- **De 32 alertas (auto × repetición, con V), 30 nombran al menos un hábito.** En las otras 2 el
  mensaje dice que el riesgo no se explica por hábitos de uso.

## Tres límites que el trabajo respeta (y el texto al cliente también)

1. **SHAP explica al modelo, no al auto.** Es asociación: (a′) de K2 es +0,019, así que el modelo
   sabe sobre todo *qué auto* y poco del *cuándo*. Además, "mucho idle ⇒ falla antes en km" era
   km/día disfrazado ([f3-reloj-y-ventana-del-evento.md](f3-reloj-y-ventana-del-evento.md) §2). El
   mensaje dice "tu uso se parece al de los autos que fallaron", nunca "falla porque hacés X".
2. **El horizonte es de 3.000 km con 500 km de gap, no "4 meses".** Se expresa en km y en semanas al
   ritmo del auto (`feat_km_per_day`). Ver "Límites".
3. **Síntomas y contexto no son "lo que hacés mal".** La clasificación de las 56 columnas del modelo
   está en el preregistro: 20 accionables (15 con signo físico), 28 síntomas y 8 de contexto.
   - **Síntomas:** toda la familia B (regeneración, DPF, filtro), los mensajes del filtro, la vida
     de aceite y el consumo.
   - **Contexto:** odómetro, mercado, bin del hazard, controles de ventana y el ritmo por día
     calendario (km/día, viajes/día, horas entre viajes), que traduce el reloj de edad al de km.
   - El signo esperado sale de la física del DPF, del mapa del 18-09 (escrito antes del primer
     modelo) y de F2 §3.2.

## Cómo se hizo

**§3 · Reconstrucción.** `train.py` no guarda los modelos, así que se reentrenan los 15 folds con
`src/eval/explain.py::refit_folds`. Es el cuerpo del loop de `run_cv`, con los mismos helpers de
split (`iter_repeats`), columnas, preprocesado y target. **Reproduce `predictions.parquet` bit a
bit: máx |Δ| = 0 en los 15 folds**, así que no hizo falta el respaldo de ρ = 1. Cada auto se explica
con el modelo del fold que no lo vio.

**Variantes** (`src/eval/explain.py`, funciones puras):
- **V1**: apila cada fila en los 6 tramos del score, igual que `predict_proba`. Pide
  `pred_contrib=True` al LightGBM del hazard y promedia los 6 vectores. Es aditivo exacto sobre el
  log-odds medio del hazard, y el riesgo que se recompone desde el booster es `predict_proba`
  (diferencia 0).
- **V2**: `shap.PermutationExplainer` sobre el score, con un fondo de un corte por sano del train del
  fold y del mismo mercado. Corre sobre 234 filas: los cortes explicados de los 19 autos que alertan,
  más el corte máximo de 40 que no alertan.
- **V3**: el promedio de los vectores de V1 de las 3 repeticiones, a nivel auto.
- **V4**: V3 sumado por familia.

**Agregación por vehículo, igual para las cuatro.** Es la alerta del dashboard (`vehicle_alerts` +
`operating_threshold`): k = 2 al 5% de falsas alarmas. Se explican los 2 cortes que la dispararon,
o el corte de score máximo si el auto no alerta. Reproduce `window_eval.json`: V detecta 7 / 10 / 6
de 45 y D, 8 / 10 / 6 de 53.

**Réplicas para la estabilidad.** V1 y V2 explican una repetición, así que sus 3 réplicas son las 3
repeticiones. V3 promedia 3, así que sus réplicas son 3 tripletes disjuntos: el oficial más dos
tripletes con repeticiones sorteadas por `make_splits` (misma estratificación, semilla 20260925).
Sin eso, la estabilidad de V3 no se puede medir.

## Resultados

### 1 · Qué variante funcionó mejor (V, 5%, 19 autos que alertan en alguna repetición)

| | V1 · TreeSHAP del hazard | V2 · Permutation del score | **V3 · V1 medio de 3 rep.** | V4 · V3 por familia |
|---|---|---|---|---|
| aditividad (máx.) | 1,9e-14 | 8,9e-16 | 7,1e-15 | 7,1e-15 |
| ρ Σφ + base vs. score (sin base) | 0,99998 (0,998) | 1 por construcción (0,985) | 0,9988 (0,998) | 0,9988 |
| borrado: caída del score (nulo, p95) | 0,234 (0,042; 0,064) | 0,248 (0,042; 0,064) | 0,199 (0,042; 0,070) | 0,191 (0,140; 0,162)¹ |
| borrado: p (200 sorteos) | 0,005 | 0,005 | 0,005 | 0,005¹ |
| Jaccard del top 3 entre réplicas | 0,574 ± 0,171 | 0,477 ± 0,182 | **0,730 ± 0,228** | 0,895 (top 1)² |
| acuerdo de signo | 0,82 | 0,83 | **0,97** | 1,00² |
| accionables alineadas con la física | 10 / 15 | 8 / 15 | 10 / 15 | — |
| costo por auto explicado | 0,0001 s | 0,24 s | 0,0003 s (3 modelos) | 0,0003 s |
| ¿pasa la fidelidad? | sí | sí | **sí · elegida** | no elegible |

¹ Informativo: reemplaza todas las accionables de la familia de mayor contribución (casi siempre A,
17 features) contra el mismo número al azar.
² No elegible por preregistro: solo A y C tienen accionables, así que el top 3 sería trivial.

- **La regla elige V3.** Pasan la fidelidad V1, V2 y V3; V3 le saca 0,16 de Jaccard a V1, más que
  la tolerancia de 0,05. V1 con las 9 repeticiones (informativo) da 0,605: el 0,574 es
  representativo.
- **V1 y V2 no discrepan.** La mediana del ρ entre sus vectores en las 32 alertas es 0,85 (mínimo
  0,59), por encima del umbral de hallazgo de 0,7. V2 es aditivo sobre el score pero tiene ruido de
  Monte Carlo, y por eso es el menos estable.
- **El borrado es un contrafactual del modelo, no del auto.** Reemplazar las 3 accionables de mayor
  contribución de un auto alertado por la mediana de los sanos de su mercado × mes baja su score
  0,20–0,25; tres accionables al azar lo bajan 0,04. No dice qué le pasaría al auto si cambiara de
  hábitos.

### 2 · Qué aprendió el modelo, globalmente (V3, 2.029 cortes de dev fuera de fold)

Peso medio |φ| por clase: accionables 60%, contexto 22%, síntomas 18%. Top 10 accionables por |φ|:

| # | accionable | esperado | ρ(valor, φ) | ρ contra la flota del mes | ¿al texto? |
|---|---|---|---|---|---|
| 1 | viajes que no llegan a temperatura | + | +0,85 | +0,76 | sí |
| 2 | velocidad media | − | −0,71 | −0,53 | sí |
| 3 | arranques sin moverse (idle) | + | +0,78 | +0,76 | sí |
| 4 | duración típica de los viajes | 0 | +0,93 | +0,88 | no: sin hipótesis |
| 5 | temperatura media del motor | − | −0,88 | −0,80 | sí |
| 6 | temperatura máxima del motor | − | **+0,63** | +0,54 | no: contraria a la física |
| 7 | viajes en movimiento con el motor frío | + | +0,78 | +0,75 | sí |
| 8 | temperatura del motor al arrancar | − | −0,86 | −0,78 | sí |
| 9 | largo típico de los viajes | − | **+0,73** | +0,72 | no: contraria a la física |
| 10 | viajes lentos (< 30 km/h) | + | +0,67 | +0,35 | sí |

- **Fuera del texto, además:**
  - contrarias a la física: refrigerante al terminar el viaje (+0,24) y viajes más cortos (p25,
    +0,73);
  - efecto débil: arranques en frío (ρ = −0,01, inestable);
  - sin hipótesis: amplitud térmica, viajes encadenados y las dos tendencias.
- **Entran al texto 10 accionables:** las 7 de arriba marcadas "sí", más motor encendido sin moverse
  cada 1.000 km (+0,83), viajes de menos de 5 km (+0,72) y viajes por cada 1.000 km (+0,23).
- **El control de calendario no cambia ningún signo ni ninguna decisión de texto.**
  - Normalizar contra los sanos del mismo mercado × mes (`FleetReferenceNormalizer`, train del fold)
    casi no toca a las temperaturas ni al idle (−0,88 → −0,80; +0,78 → +0,76): el modelo sigue al
    auto que está más frío que sus pares del mismo mes, no al invierno.
  - Lo que más se achica es la velocidad (−0,71 → −0,53) y los viajes lentos (+0,67 → +0,35): una
    parte de esa asociación varía con el mercado × mes.
- **Las contrarias a la física no tienen explicación medida.** Una lectura posible: largo y lento es
  tráfico (la duración, sin hipótesis, es la cuarta más importante con ρ = +0,93), y con features
  correlacionadas TreeSHAP reparte el crédito de formas poco intuitivas. No se probó: sería una
  hipótesis para preregistrar, no algo para decirle a un cliente.

### 3 · Mensajes

- De las 32 alertas con V, el mensaje nombra:
  - 3 factores en 25;
  - 2 en 2;
  - 1 en 3;
  - ninguno en 2, y dice "el riesgo no se explica por hábitos de uso".
- **Los factores que más aparecen** son idle (22), motor que no llega a temperatura (19), velocidad
  (16), temperatura media del motor (8) y viajes lentos (7).
- **En las 32 alertas los síntomas suman riesgo**, y la línea "lo que ve el filtro" lo dice. Entre
  los 19 autos alertados, la mediana de la suma es 1,00 de log-odds para las accionables y 0,37 para
  los síntomas. Los síntomas pesan más en 3 de 19.

### 4 · Casos (repetición 0, V, 5%, umbral 0,330; regla del YAML)

Tabla y mensajes completos en `experiments/explain-k2/cases.md`; waterfalls en `figures/`.

| auto | categoría | score máx. | ¿alertó? | ¿falló? | top 3 accionables (valor vs. mediana sana) | factores del mensaje |
|---|---|---|---|---|---|---|
| VEH_0451 | detectado | 0,618 | sí | sí | duración +0,28 (157 vs 20 min) · amplitud +0,16 · velocidad +0,11 (26 vs 19 km/h, del lado sano) | ninguno: "no se explica por hábitos" (síntomas +1,01) |
| VEH_0487 | detectado | 0,583 | sí | sí | velocidad +0,48 (16 vs 18 km/h) · duración +0,40 · largo de viaje +0,16 (contraria) | velocidad, viajes lentos |
| VEH_0513 | detectado | 0,559 | sí | sí | velocidad +0,44 · sin temperatura +0,28 (25% vs 11%) · idle +0,16 (28% vs 26%) | los tres |
| VEH_0374 | no detectado | 0,559 | no (medio) | sí | velocidad +0,72 (14 vs 19 km/h) · duración +0,18 · sin temperatura +0,16 (32% vs 10%) | velocidad, sin temperatura, idle |
| VEH_0535 | falsa alarma | 0,747 | sí | no | idle por km +0,46 (244 vs 16 cada 1.000 km) · idle +0,29 (61% vs 22%) · sin temperatura +0,21 | los tres |
| VEH_0275 | sano, riesgo bajo | 0,017 | no | no | temperatura media +0,18 · al arrancar +0,15 · motor frío en movimiento +0,09 | ninguno (riesgo bajo) |

**El detectado con el score más alto (VEH_0451) es el caso que justifica los filtros.**
- Hace viajes de 100 km a 224 km/día, y su riesgo viene de los síntomas (+1,01) y de accionables sin
  hipótesis.
- La velocidad le suma riesgo en el modelo aunque va más rápido que sus pares.
- Un texto sin filtros le habría dicho "manejás lento".

## Límites

- **Es asociación, no causa.** El borrado mueve al modelo, no al auto. Con (a′) = +0,019 el porqué
  es sobre todo "qué autos se parecen a los que fallaron", y no "qué cambió antes de fallar": no hay
  trayectoria previa al evento (f3-reloj §5).
- **Un riesgo "a 4 meses" no está validado, y este trabajo no lo calcula.**
  - El score de K2 es la probabilidad de que el evento caiga entre 500 y 3.500 km desde el corte.
    Pasarlo a semanas con el km/día del auto es una conversión de unidades, no una predicción: el
    ritmo cambia, y el evento sigue la edad del auto más que los km (f3-reloj §1–2).
  - Para medir un riesgo a 4 meses haría falta:
    1. un panel con el horizonte en días (H = 120 días desde el corte);
    2. una etiqueta corregida que solo cuente los cortes cuyo horizonte entero cae dentro de la
       ventana del registro;
    3. la calibración a la prevalencia real de la flota, porque el muestreo por cohortes infla la
       tasa.
  - Con la ventana actual (01-09-2025 → 11-03-2026, poco más de 6 meses), un horizonte de 4 meses
    deja evaluables solo los cortes de los primeros ~2 meses de la ventana: muy pocos eventos. Hace
    falta que Ford confirme una ventana más larga y la prevalencia.
- **La elección descansa en 19 autos alertados.** La estabilidad tiene desvíos de ~0,2 entre autos, y
  la ventaja de V3 sobre V1 (0,16) es clara en la media pero no se testeó formalmente.
- **La mediana "de los sanos comparables"** es la del train del fold (mercado × mes, con respaldo a
  mes y a global si la celda tiene menos de 10 sanos). En producción habría que fijarla con toda la
  flota sana del período.
- **La clasificación accionable / síntoma / contexto es una decisión**, preregistrada pero
  discutible. Ejemplos: el consumo como síntoma y el ritmo por día como contexto. Cambiarla cambia
  qué llega al texto; hay que hacerlo en el YAML, antes de mirar.

## Desviaciones y precisiones respecto del preregistro

- **Con riesgo bajo el mensaje no lista factores**, y se guardan vacíos. El preregistro no lo decía:
  el objetivo es explicar un riesgo alto, y decirle "se parece a los que fallaron" a un auto de
  riesgo bajo sería engañoso.
- **El borrado de V4** (informativo, V4 no es elegible) reemplaza todas las accionables de la familia
  de mayor contribución, contra el mismo número al azar. El preregistro no lo especificaba.
- **`render_vehicle_message` también descarta las accionables sin signo físico**, aunque se las
  pasen. Es una defensa extra: `message_factors` ya las excluía.
- **El waterfall siempre abre los factores que llegan al mensaje**, aunque queden fuera del top 6
  por |φ|. Es de presentación.
- **La capa de decisión** (`experiments/decision-k2/decision_layer.json`) no estaba en el checkout y
  se regeneró con su script, sin cambios, para que el dashboard cargue.
- **`results.py log` no aplica**: no hay una corrida nueva con métricas de modelo.

## Código nuevo

- `configs/explain_k2.yaml` (preregistro) y `configs/explain_texts.yaml` (textos al cliente y nombres
  de las figuras).
- `src/eval/explain.py`: reconstrucción, V1–V4, agregación, referencia de flota, criterios, mensaje y
  guarda de dev.
- `scripts/explain_k2.py` (punta a punta) y `scripts/explain_k2_report.py` (figuras y `cases.md`).
- Dashboard: sección "Por qué" en `scripts/dashboard_k2/app_pages/vehiculo.py`, con
  `load_explanations` / `vehicle_why` en `src/eval/dashboard_data.py`; clave `explain_config` en
  `configs/dashboard_k2.yaml`.
- `scripts/check_setup.py`: 10 chequeos nuevos (190 en total). Seis mutaciones los hacen fallar: V1
  sin base, un solo corte por auto, referencia con fallados, render sin filtro, guarda sin raise y
  factores sin coherencia con la mediana.
