# Product

<!-- impeccable:product-schema 1 -->

Alcance: la demo de producto de F9 (`scripts/demo_app/`), con la marca **Ford × SOG** (SOG es el equipo; el
nombre interno del producto es Ford DPF, antes Radar DPF). El dashboard de K2
(`scripts/dashboard_k2/`) y el del EDA son herramientas internas del equipo y quedan fuera.

## Platform

web

## Users

**Usuario principal: el jurado del Ford Innovation Challenge III** (gerentes de Ford). Entra solo,
con la URL y la clave, **después de haber visto el pitch del 02-10-2026**. Ya sabe qué es el modelo (la GRU), qué es
el filtro de partículas (DPF) y que la flota se reproduce en un replay. Recorre la demo a su ritmo,
sin nadie del equipo al lado, en una notebook o en el **celular** (el link llega por WhatsApp o por
mail). Lo que viene a comprobar: que lo que se contó en el pitch existe, se puede usar y es creíble.

**El personaje de la demo es quien gestiona la posventa.** Cada lunes abre la bandeja, revisa lo que
el agente de triage priorizó y redactó, y aprueba o descarta cada acción. El jurado usa el producto
en el lugar de esa persona.

**Destinatarios que no usan la interfaz y reciben textos:**
- el conductor, con un aviso en la app del vehículo;
- el concesionario o el taller, con un contacto y un resumen técnico.

## Product Purpose

Ford DPF convierte una alerta temprana de la GRU (el modelo final sobre la entrega v2) sobre la
degradación del filtro de partículas diésel en una acción de posventa: quién se entera, con qué texto
y qué revisa el taller.

La flota de desarrollo se reproduce semana a semana, de la primera revisión al fin de la extracción
de datos de Ford (03-03-2025 → 24-09-2026). Cada alerta de la GRU llega a la bandeja ya priorizada y
redactada por dos agentes:
- el **redactor**, con un verificador en código;
- el **triage**, con herramientas.

Existe porque en una hackathon el jurado mira lo que puede usar el usuario más que las métricas.
El plan define los criterios de evaluación: claridad de la visualización, explicabilidad y el pitch.

El éxito es que el jurado se vaya habiendo visto tres cosas:
1. una predicción que se vuelve una acción concreta;
2. textos en los que se puede confiar, porque un verificador los controla;
3. números que no se inflan.

## Positioning

**El modelo decide, el agente comunica.**
- Ningún LLM produce un score, un factor ni un número: todo lo que se muestra sale del bundle
  precalculado.
- La política de acción es determinista. El agente la justifica y la redacta, pero no la elige.
- Un verificador en código rechaza los números que no están en los hechos, el lenguaje causal, las
  certezas, las promesas de que el riesgo baja y los síntomas del filtro que le llegarían al conductor.

**Anticipa, no reacciona.** Ford ya tiene detección reactiva. Ford DPF alerta con una mediana de
15 semanas (~7.000 km) entre la alerta confirmada y la falla, a partir de datos que terminan antes
del evento (el gap de blanking).

## Operating Context

- **Acceso:** URL pública en Railway, protegida con `DEMO_PASSWORD`. El jurado tiene que entrar por
  la raíz: si arranca directo en `/vehiculo` o en `/resultados`, Streamlit 1.63 devuelve dos 404
  antes de conectar. El servicio es serverless, así que después de dormir el primer request puede
  dar 502. Como el jurado entra solo, nadie del equipo va a estar ahí para despertarlo antes.
- **Pantallas:**
  - **Bandeja:** el resumen del agente, una tarjeta por evento con el mensaje al conductor, el
    resumen del taller y los hechos con el verificador, y los botones aprobar, descartar y ver ficha.
  - **Vehículo:** el puntaje contra el umbral, la alerta y el porqué: en qué hábitos se aparta el auto
    de los autos sanos de su mercado (una comparación con la flota, no lo que usó el modelo).
  - **Qué pasó después:** el replay al lado de los números oficiales.
- **El ritual es semanal.** Se elige la semana en el calendario de la flota, arriba de la Bandeja y
  del Vehículo: una columna por semana con sus alertas. Se salta a la próxima con alertas y la demo
  abre en la del 29-09-2025.
- **«Cómo usar»** (arriba a la derecha) recorre la pantalla abierta paso a paso, en la notebook y en
  el celular.
- **Lo que pasó después no se mezcla con la semana.** La Bandeja y el Vehículo muestran solo lo
  que se sabía ese lunes; si cada auto alertado falló está en «Qué pasó después». Hasta el
  26-09-2026 había un interruptor que lo revelaba en las otras pantallas, y se sacó.
- **Las acciones posibles:**
  - aviso personalizado al conductor, si hay hábitos que nombrar;
  - contacto del concesionario con el resumen técnico, si no los hay.

  Se escala si el score sigue sobre el umbral 2 revisiones después del aviso. Una revisión es un
  corte cada 500 km.
- **Agentes:** `cache_first` por defecto. Las 36 semanas con eventos del replay están precalentadas: 302
  respuestas en la caché del bundle v1, con `gpt-5.4-mini-2026-03-17`. «Regenerar en vivo» llama a la API y no
  pisa la versión guardada. `cache_only` sirve para ensayar sin red.

## Capabilities and Constraints

- **Stack:** Streamlit 1.63 fijado, Python y gráficos de Altair. El estilo se inyecta desde
  `theme.css`, `tokens.css`, `presentation.py` y `.streamlit/config.toml`.
  `st.components.v1.html` está deprecado: hay que migrarlo a `st.iframe` antes de actualizar
  Streamlit.
- **Deploy:**
  - Dockerfile y `railway.json`.
  - La imagen lleva solo `src/`, `scripts/demo_app/` y dos configs.
  - La app baja el bundle `demo-bundle-gru` en la versión fijada en `configs/demo.yaml` y no importa
    torch, LightGBM ni sklearn.
- **Dispositivos:** tiene que funcionar bien **en celular** y en notebook o desktop (1366–1920 px).
- **Datos:** solo dev de la entrega v2. Son 426 autos: 135 con falla registrada y 291 sanos. El test no se muestra
  nunca, y el bundle falla si aparece un vehículo de test. La muestra está enriquecida en fallas,
  así que los conteos de la bandeja no se trasladan a una flota real.
- **Números que no se pueden contradecir:**
  - **Replay** (R1, etiqueta dura, 5%, umbral exacto): 53 alertas y 7 escalamientos, entre las
    semanas del 30-06-2025 y del 27-07-2026. De los 135 autos que fallan, 39 reciben la alerta antes
    y 96 no se ven. 14 de los 291 sanos reciben una alerta de más.
  - **Oficiales:** 28,1% ± 2,8 de detección al 5% de falsas alarmas (27,9% fuera de muestra) y
    42,5% al 10%. La tabla del equipo da 27,4% y 42,5%, y el pitch cita ~27–29% al 5%. La celda
    mercado × motor sola, sin mirar un viaje, detecta 18,0% al 5%.

  Las fuentes son `docs/memoria/f9-demo-gru.md` y `docs/memoria/f10-sweep-gru.md`.
- **Terminología:**
  - **alerta:** confirmada en el `k`-ésimo corte seguido sobre el umbral;
  - **escalamiento:** el paso al concesionario;
  - **revisión:** un corte, cada 500 km;
  - **hábitos:** lo que el conductor puede cambiar;
  - **señales del filtro:** síntomas, solo para el taller;
  - **umbral:** al 5% de falsas alarmas;
  - **replay:** la flota reproducida en el calendario.
- **Reglas de texto (`configs/agents.yaml`, `verifier`):**
  - nada causal: «porque», «causa», «provoca», «debido a»;
  - nada de certezas ni promesas de falla;
  - ni «probabilidad» (el score no está calibrado a la prevalencia real) ni «de rutina»;
  - al conductor no le llegan síntomas del filtro: carga, regeneraciones, aceite, consumo, avisos
    del filtro;
  - los hábitos son una comparación con autos sanos de su mercado: ni «se parece a autos que
    fallaron» ni «el sistema lo marcó por…».

  Los textos al conductor tutean. Toda la interfaz está en español.
- **Decisiones que no son nuestras:** la política de acción y los chequeos del taller son de
  ejemplo, y los define Ford (viven en `configs/agents.yaml`). Dónde operar, entre el 5% y el 10%,
  es una decisión de costo de Ford.
- **Sin medir:** si un conductor cambia de hábitos al recibir el aviso.

## Brand Commitments

- **Marca: Ford × SOG** (desde el 28-09-2026, decisión del usuario). SOG es el nombre del equipo, y
  «DPF» ya no va en la marca: no se entendía por qué la app lo decía.
  - La marca es el script de Ford, un × y «SOG». Aparece en la barra lateral, arriba en el celular y
    en el login.
  - La pestaña del navegador dice «SOG · Ford Innovation Challenge III» y el recorrido, «Así se usa
    la demo de SOG». Todo sale de `brand` en `configs/demo.yaml`.
  - «Ford DPF» (`product_name`) queda como nombre interno del bundle y de los prompts de los agentes
    (`configs/agents.yaml`), que ningún texto generado escribe. Cambiar los prompts invalida la caché
    del LLM. Antes se llamó Radar DPF (hasta el 26-09) y Ford DPF como marca (26-09 al 28-09).
- **Ford:** la palabra «Ford» de la marca es el **script de Ford** (el logo escrito), en blanco sobre
  el fondo oscuro. Lo autorizó el usuario el 26-09-2026 y reemplaza la regla anterior de no usar el
  logo.
  - El × es de dos trazos finos en el azul eléctrico de la app (dibujado, no el carácter), y «SOG»
    va en Manrope 800 blanco con espaciado amplio.
  - SOG se apoya sobre la línea de base del script y ocupa la altura de las minúsculas de «ord».
  - El archivo sale del «Ford logo flat.svg» de Wikimedia Commons: el archivo es de dominio público,
    pero la marca es de Ford.
  - El óvalo azul y el resto de la identidad oficial no se usan.
  - Se puede nombrar al Ford Innovation Challenge III (desafío *Data-Driven Powertrain
    Intelligence*).
- **Tipografía:** la corporativa de Ford (Ford Antenna) es propietaria y no está en el repo. Si el
  challenge entrega los archivos, puede reemplazar a Manrope; hasta entonces «SOG» y la interfaz van
  en Manrope.
- **Voz:** operativa y honesta. Comparación, no causa. Ninguna certeza que el modelo no tenga.

## Evidence on Hand

- **Bundle:** `oromagnoli-/ford-fic/demo-bundle-gru` en wandb (la versión fijada en `configs/demo.yaml`),
  con la copia local en `experiments/demo-bundle-gru/`. Trae los cortes, las alertas, los hábitos que
  se apartan de la flota sana, los mensajes, las señales del filtro, los números oficiales, el triage
  de las semanas con eventos y la caché del LLM. La demo anterior, con K2, sigue en `demo-bundle:v0`.
- **Textos reales:** los agentes redactaron los 60 textos y los 36 resúmenes, todos aprobados por
  el verificador (14 reintentos en el camino, en la v1). Lo que se corrigió después de revisar a mano la
  primera semana está en `docs/memoria/f9-demo-gru.md`.
- **El porqué:** una comparación con los autos sanos de dev del mismo mercado (`src/eval/fleet_profile.py`).
  No es atribución: la GRU no la tiene.
- **Revisión visual previa:** capturas a 1440×900 y 1920×1080 en `experiments/demo-ui-review/`
  (no versionado), con consola, accesibilidad e interacciones en `review.json`.
- **Lo que no existe y no se inventa:** clientes, testimonios, un despliegue real en Ford, la
  efectividad medida de los avisos, ahorros validados (los escenarios de costo son reporte), scores
  calibrados a la prevalencia real, ni otros activos de marca de Ford además del script
  (`scripts/demo_app/assets/ford-script.svg`).

## Product Principles

1. **El modelo decide, el agente comunica.** Todo número, score o factor que se muestra sale del
   bundle, y ningún texto generado llega a la pantalla sin pasar el verificador.
2. **Honestidad antes que impacto.** Los aciertos se muestran al lado de las fallas que no se ven y
   de las falsas alarmas, y el replay al lado de los números oficiales. El porqué es una comparación
   con la flota, no lo que usó el modelo.
3. **Anticipar, no reaccionar.** El valor está en la ventana entre la alerta y la falla. Lo que
   Ford ya tiene es detección reactiva.
4. **A cada uno, lo suyo.** El conductor ve hábitos que puede cambiar, el taller ve las señales del
   filtro, y quien gestiona la posventa ve la decisión, el texto y la traza del agente.
5. **Se recorre sin nosotros.** El jurado llega sabiendo qué es, pero sin guía: cada pantalla tiene
   que decir qué muestra y qué se puede hacer ahí, tanto en el celular como en la notebook.
