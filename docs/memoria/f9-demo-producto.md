# Demo de producto: Radar DPF, K2 con agentes de triage

> **27-09: la demo ya no usa K2.** Pasó a la GRU + TripSummary + estática completa sobre la entrega
> v2, con un porqué descriptivo (comparación con los sanos del mercado): [f9-demo-gru.md](f9-demo-gru.md).
> Esta ficha queda como registro de la versión con K2 (`demo-bundle` en wandb, sin tocar).

**Fecha:** 2026-09-26 · **Fase:** F9 (producto, para el pitch del 02-10) · **Rama:** `feat/demo-agentes`
**Alcance:** K2 (`f6-ss-hw-r3`), solo dev, R1, etiqueta V, 5% de falsas alarmas. **Test sin tocar**: el
bundle falla si trae un vehículo de test.

**No es un candidato ni cambia ninguna métrica.** Muestra el score, el umbral, la alerta y el porqué de
K2 tal cual. Lo nuevo es la capa de producto: qué pasa con una alerta, quién se entera y con qué texto.

```bash
export FORD_DATA_DIR=$PWD/data/rebuild-0921
# insumos (si no están): K2, su window_eval, la capa de decisión y la explicabilidad
python scripts/train.py --config configs/exp_ss_hw_r3.yaml
python scripts/eval_window_label.py --config configs/exp_ss_hw_r3.yaml
python scripts/decision_layer.py --config configs/exp_decision_k2.yaml
python scripts/explain_k2.py --config configs/explain_k2.yaml
# la demo
python scripts/build_demo_bundle.py --config configs/demo.yaml          # experiments/demo-bundle/
python scripts/warm_demo_cache.py --config configs/demo.yaml --prune    # agentes en todas las semanas (.env con OPENAI_API_KEY)
python scripts/build_demo_bundle.py --config configs/demo.yaml --publish-only   # wandb Artifact demo-bundle
streamlit run scripts/demo_app/app.py
```

## En una línea

**La flota de dev se reproduce semana a semana en el calendario del registro, y cada alerta de K2 llega
a la bandeja de la gerente de posventa ya priorizada y redactada por dos agentes.** Un verificador en
código controla cada texto antes de mostrarlo.
- **El replay (R1, V, 5%) da 9 alertas y 2 escalamientos entre el 29-09 y el 21-12-2025.** De 45 autos
  que fallan, 7 reciben la alerta antes, con una mediana de 15 semanas desde la alerta confirmada hasta
  la falla. 2 de 95 sanos reciben una alerta de más y 38 fallas no se ven.
- **Los números oficiales no cambian**: 17,0% ± 3,8 al 5% (15,6% fuera de muestra) y 26,7% al 10%
  ([f8-capa-decision-k2.md](f8-capa-decision-k2.md)). La página «Qué pasó después» los muestra al lado
  del replay.
- **Los agentes redactaron las 11 piezas de las 8 semanas y todos los resúmenes pasaron el verificador**
  (`gpt-5.4-mini-2026-03-17`, 46 llamadas en la caché).

## Cómo funciona

**Principio: el modelo decide, el agente comunica.** El LLM nunca produce un score, un factor ni un
número. Lee los hechos del evento (`src/agents/facts.py`), que salen del bundle y solo tienen lo que se
sabía ese día: nunca el desenlace.

| pieza | dónde | qué hace |
|---|---|---|
| bundle | `scripts/build_demo_bundle.py` | Precalcula con el stack completo y las mismas funciones del dashboard de F4 los cortes, la alerta, los factores, el mensaje preregistrado, el waterfall, las señales del filtro contra los sanos del mercado y los números oficiales. La app no importa LightGBM, SHAP ni sklearn |
| calendario y política | `src/agents/policy.py` | Una alerta nueva se confirma en el `k`-ésimo corte seguido sobre el umbral. **Con hábitos que nombrar**, aviso al conductor; **sin hábitos**, directo al concesionario, porque no hay nada que pedirle al conductor; si el score **sigue sobre el umbral 2 revisiones después del aviso**, se escala. Es determinista |
| agente 1, redactor | `src/agents/drafter.py` | Salida estructurada con el mensaje al conductor, el resumen del taller y la línea de la bandeja. Las recomendaciones y los chequeos del taller se eligen por id de listas cerradas, y su texto lo pone el código |
| verificador | `src/agents/verifier.py` | Todo número tiene que estar en los hechos. Rechaza lenguaje causal, certezas, promesas de que el riesgo baja y probabilidades. Al conductor no le pueden llegar síntomas del filtro, y el texto tiene que ser coherente con la acción de la política. Si rechaza, se reintenta con la lista de problemas (hasta 2 veces); si no, queda la plantilla preregistrada |
| agente 2, triage | `src/agents/triage.py` | Loop de herramientas: eventos de la semana, acción de la política, redactar y contexto de la flota. `redactar_mensajes` rechaza una acción distinta de la política; un evento que el agente no redactó lo completa el código; el resumen pasa el verificador contra lo que devolvieron las herramientas |
| caché | `src/agents/llm.py` | Cada pedido se guarda por el hash de todo lo que lo define, y un loop de herramientas se reproduce sin red. `cache_first` en la app, `cache_only` para ensayar sin red. `live` es el botón «Regenerar en vivo», y no pisa la versión guardada |

**Qué atajó el verificador mientras se ajustaban los prompts** (esto no es una evaluación, es el
registro del desarrollo):
- "porque" en el mensaje al conductor, en varios intentos y sobre todo en los escalamientos (el prompt
  ahora sugiere "por eso" o "así que");
- una línea de la bandeja que proponía una visita al concesionario cuando la política había decidido un
  aviso (de ahí la regla de coherencia con la acción);
- la regla de "aviso" era demasiado amplia y se acotó a "avisos del filtro".

Además, un mensaje de invitación al concesionario decía que era "una verificación de rutina". El
verificador no lo atrapó: se corrigió en el prompt y quedó como regla (`de rutina` prohibido).

## Dos decisiones de la implementación

1. **Un solo umbral (5%), no dos niveles de 10% y 5%.** La explicabilidad preregistrada existe solo al
   5%: al 10%, los autos de ese tramo no tendrían hábitos que nombrar. Los dos niveles de acción salen
   de si hay hábitos y de si el riesgo persiste. Dónde operar (5–10%) sigue siendo una decisión de costo
   de Ford ([f8-costos-k2.md](f8-costos-k2.md)).
2. **El orden de los hábitos es V3 aunque use revisiones posteriores a la alerta en 7 de 9 autos.** V3
   promedia 3 repeticiones y alguna alerta más tarde. Con V1 de R1 sola, que no usa nada posterior, los
   hábitos nombrados coinciden:
   - los tres en 4 autos (0451, 0456, 0487, 0514);
   - dos de tres en 4 (0513, 0535, 0580, 0581);
   - uno de tres en VEH_0586.

   Todos son hábitos térmicos y de uso. Se mantiene V3 (la variante preregistrada y la más estable); los
   valores citados sí son los de los cortes de la alerta, y la página de resultados lo declara.

## Deploy (Railway)

La imagen (`Dockerfile`, `requirements-demo.txt`) lleva solo `src/`, `scripts/demo_app/` y dos
configs. Al arrancar baja `demo-bundle` en la versión fijada en `configs/demo.yaml` (hoy `v0`) y
levanta Streamlit en `$PORT`. Probado en local: construye, baja el bundle de wandb y sirve las tres
páginas sin LightGBM ni sklearn.

1. Railway → New Project → Deploy from GitHub repo → `octavioromagnoli/ford-fic` (autorizar la app de
   GitHub, el repo es privado) → la rama. Toma `railway.json`, que construye con el Dockerfile y solo
   redespliega si cambian los archivos de la demo.
2. Variables: `WANDB_API_KEY` (baja el bundle), `OPENAI_API_KEY` (solo para «Regenerar en vivo»;
   sin ella todo se sirve de la caché) y `DEMO_PASSWORD`.
3. Settings → Networking → Generate Domain. Settings → Serverless (on).
4. Antes del pitch, abrir la URL unos minutos antes: después de dormir, el primer request puede dar 502.

Para actualizar los datos o los textos: reconstruir, correr `warm_demo_cache.py --prune`, publicar con
`--publish-only`, cambiar `bundle.version` y hacer push.

## Límites

- **La muestra está enriquecida en fallas** (45 de 140). Los conteos de la bandeja no se trasladan a
  una flota real.
- **El replay es una repetición**; los números oficiales son el promedio de las 3.
- **El porqué es asociación**, con los límites de [f4-explicabilidad-k2.md](f4-explicabilidad-k2.md).
  El verificador controla el texto, no la verdad de la asociación.
- **Las señales del filtro para el taller** se comparan con la mediana de los sanos de dev del mismo
  mercado sobre toda la ventana. Es descriptivo, no es una referencia de producción.
- **La política y los chequeos del taller son de ejemplo.** Los define Ford: la política vive en
  `configs/agents.yaml` y se cambia ahí.
- **La efectividad de los avisos no está medida.** Nada en los datos dice si un conductor cambia de
  hábitos al recibirlo.
