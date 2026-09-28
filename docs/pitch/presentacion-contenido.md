# Presentación Ford FIC III — Contenido

28-09-2026 · Export del doc vivo: https://claude.ai/code/artifact/1e8d7eae-8435-4819-be16-ab64715822e4
(el doc es la versión que se edita; este archivo es una foto para el repo).

## Estructura y storytelling

El relato va del problema al negocio. Un auto que se queda parado, los datos que lo anticipan, el modelo que lo detecta, lo que Ford hace con la alerta, cómo escala y cuánto vale. Los cinco bloques pedidos están todos. Los de la plantilla de Ford (valor diferencial, trabajo futuro, conclusiones) van en el cierre.

| # | bloque | pregunta que responde | bloque de la plantilla Ford | tiempo |
| --- | --- | --- | --- | --- |
| 1 | Apertura: el problema | ¿Por qué le importa a Ford? | 01 Descripción del desafío | 1 min |
| 2 | EDA: los datos | ¿Qué hay en la telemetría y qué trampas tenía? | 02 Descripción de la solución | 1,5 min |
| 3 | Modelo | ¿Qué probamos, qué ganó y cuánto detecta? | 02 Descripción de la solución | 2 min |
| 4 | Producto | ¿Qué pasa después de la alerta? (agentes + demo) | 04 Valor diferencial e innovación | 2 min |
| 5 | Escalabilidad | ¿Cómo llega a 1 millón de autos? | 05 Trabajo futuro | 1 min |
| 6 | Costos | ¿Qué pierde el cliente hoy y cuánto cuesta evitarlo? | 03 Factibilidad económica | 1,5 min |
| 7 | Cierre | ¿Qué le pedimos a Ford? | 04 Valor diferencial · 06 Conclusiones | 1 min |

**Por qué este orden.** El producto va antes que costos y escalabilidad. El jurado tiene que ver la alerta llegando al conductor antes de escuchar números de infraestructura. Costos cierra el cuerpo porque lo que pierde el cliente es el argumento de compra. El cierre termina con un pedido concreto: un piloto en sombra.

**El hilo que atraviesa todo:** "Hoy Ford se entera cuando el tablero ya avisó. Con la telemetría que ya recibe, lo sabe ~3 meses antes y sabe qué decirle al conductor."

**Reglas para citar números, en todo el deck:**

- Toda detección va al lado del azar y de la celda mercado × motor. "35%" solo no dice nada. "35% donde el azar da 10% y saber mercado y motor da 15%" sí.
- La anticipación se dice "lo marcamos con ~8.300 km de margen", nunca "predecimos que falla en 8.300 km".
- Nada de accuracy ni de PR-AUC en el cuerpo: van al backup.
- Los números de hoy son de dev (validación cruzada). El test se mide una sola vez antes del pitch y reemplaza la columna.

## Marca y mensaje: la confianza como producto

No vendemos un modelo que predice fallas: vendemos que el cliente sienta que su Ford lo cuida. La confianza entre el cliente y Ford es el producto; la GRU, los agentes y la telemetría son cómo se cumple.

**La gran idea:** *"Tu Ford te avisa antes."* Hoy el auto avisa cuando ya es tarde, con un testigo que nadie explica. Mañana avisa con tiempo, en lenguaje humano y con algo concreto para hacer.

**Nombre y tagline (a elegir):**

| nombre | tagline | qué transmite |
| --- | --- | --- |
| Ford Early Care | "Te avisamos antes de que lo notes." | cuidado, anticipación |
| Respira | "Tu motor, siempre en su mejor forma." | eficiencia, algo vivo que se cuida |
| Ford Aliado | "Manejás vos. Te acompañamos nosotros." | relación, compañía |
| DPF Guard | "Protección inteligente para tu diésel." | técnico; mejor para Ford Pro |

Recomendación: **Ford Early Care** para el cliente particular. Habla de cuidado, no de fallas, y funciona en inglés para Ford global.

**Los tres pilares de la confianza.** Cada decisión técnica del proyecto se cuenta como uno de ellos:

1. **Te aviso a tiempo.** ~3,5 meses de margen, no el testigo de último momento. (Sale del modelo y del gap de 500 km.)
2. **Te explico por qué y qué hacer.** Un hábito concreto, nunca un código de error ni una amenaza. (Sale del porqué contra la flota sana y de los agentes con verificador.)
3. **No te asusto de más.** Primero un consejo, después un turno solo si hace falta. Ford elige cuántas falsas alarmas tolera, y ninguna cuesta caro al cliente. (Sale del dial y de la política.)

**Voz y tono de todo lo que le llega al cliente:** cálido, concreto, sin miedo. Hablamos de cuidar el auto, no de que se va a romper. Frases prohibidas en el deck y en el producto: "tu auto va a fallar", "detectamos un problema", porcentajes de riesgo al cliente.

**Una frase ancla por bloque** (el título grande de la slide principal de cada uno):

| bloque | frase ancla |
| --- | --- |
| Apertura | "El auto avisa cuando ya es tarde. Y el cliente se entera solo." |
| EDA | "La forma de manejar deja huella meses antes." |
| Modelo | "1 de cada 3, tres meses antes, con 5% de falsas alarmas." |
| Producto | "El modelo decide. El agente habla como una persona. El código cuida lo que dice." |
| Escalabilidad | "Un millón de autos, sin un sensor nuevo." |
| Costos | "Cada aviso a tiempo es un cliente que confía más en Ford." |
| Cierre | "Del testigo en el tablero al aviso a tiempo." |

**Momentos memorables:**

- **Abrir con una persona, no con un gráfico** (slide 2) y cerrar con la misma persona, ahora avisada a tiempo (slide 35). Es el arco del relato.
- **Mostrar el mensaje real** que le llega al conductor, en una pantalla de celular, en la slide de producto. Es lo que el jurado recuerda.
- **Demo en vivo** del replay: el jurado ve a Ford avisar semanas antes de una falla que ya ocurrió en los datos.
- **Un solo número grande por slide.** El resto va al backup.

**Pruebas de confianza para el jurado:** lo que hace creer el mensaje es que no exageramos. Cada número se muestra al lado del azar, decimos lo que no podemos prometer y el LLM nunca inventa un número. La confianza que pedimos al jurado es la misma que el producto construye con el cliente.

## 1 · Apertura: el problema

El filtro de partículas (DPF) de los diésel de Ford se tapa por cómo se usa el auto, y hoy Ford se entera tarde. La falla es reactiva, cara y la sufre el cliente.

**Slide 1 · Portada.** Ford Early Care (nombre a confirmar) + "Tu Ford te avisa antes". Equipo SOG.

**Slide 2 · Una historia, no un gráfico.** Un conductor hace 4 km por día en ciudad. El motor casi nunca llega a temperatura y el filtro no termina de regenerar. En tres meses el testigo se prende, el auto entra en modo de protección y va en grúa al concesionario. Nadie le avisó. No pierde solo la reparación: pierde la confianza en su Ford. Auto ficticio, armado con el perfil real de un fallado del dataset. Ponerle nombre (por ejemplo, Laura) y volver a ella en el cierre.

**Slide 3 · Por qué duele.**

- Para el cliente: durante meses gasta más gasoil sin saberlo y, al final, auto parado, grúa y días sin vehículo.
- Para Ford: un cliente que siente que nadie le avisó, además del costo de garantía.
- Para flotas (Ford Pro): un vehículo de trabajo parado cuesta mucho más que la reparación.
- Hoy la detección es reactiva: el aviso llega cuando el filtro ya está saturado.

**Slide 4 · La promesa.** Con la telemetría que Ford ya recibe del auto conectado, sin sensores nuevos:

1. marcamos los autos que van camino a la falla con ~3 meses de margen;
2. decimos por qué, en hábitos que el conductor puede cambiar;
3. Ford elige cuántas falsas alarmas tolera según lo que le cuesta cada una.

**Nombre, tagline y tono:** ver "Marca y mensaje".

## 2 · EDA: qué dicen los datos

La telemetría anticipa la falla, pero solo si se lee bien: los datos traían trampas que, sin corregir, hacían que cualquier modelo pareciera mejor de lo que es.

**Slide 5 · Qué mandó Ford.** Tres tablas, ninguna con clave fila a fila entre viajes y señales:

| tabla | qué tiene | escala |
| --- | --- | --- |
| Estática | mercado, ciudad de venta, motor, modelo, producción, venta, fecha del evento | 990 vehículos, 5 mercados |
| Viajes (TripSummary) | km, duración, temperaturas, nivel de acumulación del DPF, regeneraciones | ~13 M de filas |
| Señales | mensajes del filtro (avisos, limpiezas) | por evento |

**Slide 6 · Las trampas que encontramos (y corregimos).** Es la slide que muestra rigor:

- **Dos listas muestreadas distinto.** Fallados y sanos vienen de períodos de producción distintos: la fecha de producción sola separaba las cohortes (AUC 0,87). Nos quedamos con el período común, 20-01 a 31-07-2025.
- **La fecha del evento es la del taller.** Cae ~2 semanas después de la intervención. La corremos 21 días; si no, el modelo "ve" el taller.
- **13 vehículos duplicados bajo dos códigos**, filas repetidas y una fila por evento.
- **Un contador que se apagó.** El marcador de regeneraciones se corta el 25-05-2026 para toda la flota. Lo reconstruimos desde las caídas del nivel del filtro.
- **La tasa de falla depende de cómo se armó la lista.** Va de 0% (PER) a 48% (COL) por mercado. Por eso todo se compara dentro del mercado.

**Slide 7 · El universo del estudio.** 557 vehículos, 177 eventos. 446 para desarrollar y 111 congelados como test, sorteados antes de mirar ninguna feature.

**Slide 8 · Qué anticipa la falla.** Hallazgos comparando fallados contra sanos del mismo mercado:

- **El 33% de los viajes son ralentí de 0 km** (motor encendido sin moverse). Es la señal que más anticipa.
- **Hay trayectoria.** El exceso de ralentí crece al acercarse el evento: AUC 0,60 → 0,64 → 0,76.
- **Se ve temprano.** A 60–90 días de la venta, el uso ya separa fallados de sanos (AUC 0,60–0,64).
- **Viajes cortos y motor frío.** El filtro no termina de regenerar y el hollín se acumula. Es la física del DPF, y el modelo la confirma (bloque 3).
- **El riesgo es de edad, no de kilómetros.** Sube hasta ~4 meses desde la venta y después es plano: ~3,3 eventos por 100 autos-mes.

**Slide 9 · Dónde se concentra.** Mercado × motor:

| mercado | autos | eventos | celda caliente |
| --- | --- | --- | --- |
| COL | 113 | 54 (48%) | ENG_2: 54 de 85 |
| CHL | 83 | 36 (43%) | ENG_2: 33 de 51 |
| BRA | 136 | 41 (30%) | ENG_3: 34 de 95 |
| ARG | 93 | 8 (9%) | — |
| PER | 21 | 0 | — |

Cifras de dev. La celda da un piso sin modelo: cualquier modelo tiene que ganarle a "saber mercado y motor".

**Backup:** altura de la ciudad de venta (va en la dirección física, no significativa), gasoil casi constante por país, auditoría de odometría.

## 3 · Modelo: qué probamos y qué ganó

El finalista es una red recurrente (GRU) sobre la secuencia de viajes con etiqueta suave lejos del evento: detecta el 35% de los autos que van a fallar con 5% de falsas alarmas, 3,6× el azar.

**Slide 10 · La pregunta, bien planteada.** Cada 500 km, el modelo mira los últimos 1.000 km del auto y responde: "¿este auto falla en los próximos 3.000 km?". Nunca ve los 500 km previos al evento (gap de blanking). Sin ese gap sería detección reactiva, lo que Ford ya tiene. Visual: línea de tiempo ventana → gap → horizonte.

**Slide 11 · Reglas que no negociamos.** Split por vehículo (un auto nunca está en train y validación a la vez), features solo hacia atrás, test congelado desde el día uno, y cada modelo auditado contra leakage: si se permutan las features, el score cae al azar.

**Slide 12 · El recorrido.** Más de 40 modelos medidos con la misma cuenta. Cada candidato nuevo se preregistró antes de correrlo, para no elegir ruido.

| familia | modelos | qué aprendimos |
| --- | --- | --- |
| Tabulares | logística, LightGBM sobre 53 features | aprenden qué auto falla, casi nada del cuándo |
| Supervivencia | survival stacking, cure model, K2 con ventana | fue el finalista con la entrega 1 (~15–17%); pierde contra los secuenciales en v2 |
| Series de tiempo | TimesFM zero-shot, MiniRocket | no suman o pierden con evidencia contra la GRU |
| Secuenciales | CNN-LSTM, GRU sobre TripSummary + estática | le ganan a supervivencia |
| Finalista | GRU con etiqueta suave | le gana a todo lo anterior con evidencia |

**Slide 13 · La idea que hizo la diferencia.** Los cortes de un auto que falla, pero lejos del evento, entrenan con 0,15 en vez de 0. El modelo deja de castigar la señal temprana y suma autos detectados sin perder el cuándo. Es el objetivo "lineal por tramos" de vida útil remanente (Heimes, PHM 2008), llevado a clasificación.

**Slide 14 · El resultado.** Detección de autos que fallan, según el porcentaje de sanos con falsa alarma. Confirmación preregistrada, folds y semillas nuevas, ensamble de 3 semillas:

|  | 5% | 10% | 15% | 20% |
| --- | --- | --- | --- | --- |
| **GRU con etiqueta suave (finalista)** | **34,6%** | **48,6%** | **60,2%** | **70,1%** |
| GRU anterior | 24,0% | 42,0% | 51,6% | 57,5% |
| Solo saber mercado × motor, sin modelo | 14,6% | 30,4% | 48,1% | 62,0% |
| Azar (p95, mismo historial) | 9,6% | 15,8% | 21,8% | 26,9% |

Visual: la curva detección vs. falsas alarmas con azar y celda dibujados.

- **+9,6 puntos contra la GRU anterior**, IC95 [3,0; 14,4], p = 0,0015.
- **Primer modelo que le gana con evidencia a "saber mercado y motor"**: +14,6 puntos [3,7; 22,5].
- **Fuera de muestra se sostiene:** con el umbral fijado en otros autos, 34,3% al 5,0% de falsas alarmas reales.
- Como referencia, survival stacking da ~14% al 5%.

**Slide 15 · Cuánto antes.** La primera alerta llega con una mediana de **~8.300 km, unos 3,5 meses (~106 días)** antes de la falla registrada. El score sube a medida que se acerca el evento.

**Slide 16 · En qué se apoya.** Permutando cada familia de señales:

- **estado del DPF** (acumulación, regeneraciones, avisos): −38 puntos de detección;
- **hábitos** (ralentí, motor frío, duración de viajes): −23 puntos. La duración de los viajes es el hábito que más pesa;
- **contexto** (mercado, motor, modelo): −48 puntos.

Es la lectura física: viajes cortos que no dejan terminar la regeneración.

**Pendiente antes del pitch:** medir el test (111 autos, un solo tiro, preregistrado en `f11-preregistro-test.md`). Si da distinto, se cita el test.

**Backup:** auditorías de leakage, (a′) y calendario, sweep bayesiano (130 trials, la semilla mueve más que los hiperparámetros), AUC dentro de mercado × motor 0,689, por qué no accuracy.

## 4 · Producto: qué pasa después de la alerta

Una alerta sola no evita ninguna falla. El producto convierte cada alerta en una acción: un consejo al conductor o un turno en el concesionario, redactado por agentes de IA y controlado por código.

**Slide 17 · El principio.** El modelo decide, el agente comunica. El LLM nunca produce un score, un número ni un diagnóstico: solo redacta a partir de hechos que el sistema le da.

**Slide 18 · El circuito, de la telemetría a la acción.** Visual: diagrama de flujo.

1. **Telemetría diaria** del auto conectado (viajes + señales del filtro).
2. **Score de la GRU** cada 500 km.
3. **Alerta confirmada** con 2 cortes seguidos sobre el umbral que Ford eligió.
4. **Por qué:** en qué hábitos el auto se aparta de los autos sanos de su mercado (supera al 75% de ellos). Solo hábitos accionables y coherentes con la física del DPF, hasta 3.
5. **Política (determinista, la define Ford):**
   - con hábitos para nombrar → aviso al conductor;
   - sin hábitos → contacto directo del concesionario;
   - riesgo que sigue alto 2 revisiones después → escalamiento.
6. **Agentes** redactan los textos y **un verificador** los controla.
7. **Retorno:** ¿hubo intervención en taller? Alimenta el monitoreo y el reentrenamiento.

**Slide 19 · Los agentes.**

| pieza | qué hace |
| --- | --- |
| Agente de triage | recorre los eventos de la semana con herramientas: pide la acción de la política, el contexto de la flota y encarga la redacción; arma el resumen semanal para posventa |
| Agente redactor | escribe el mensaje al conductor, el resumen para el taller y la línea de la bandeja; elige recomendaciones de listas cerradas |
| Verificador (código) | todo número tiene que estar en los hechos; rechaza lenguaje causal, certezas, promesas y probabilidades; al conductor no le llegan síntomas técnicos; si rechaza, se reintenta hasta 2 veces y si no, queda una plantilla aprobada |

**Slide 20 · Tres pantallas, tres usuarios.**

- **Gerente de posventa:** bandeja semanal priorizada con las alertas ya redactadas; aprueba o edita.
- **Concesionario:** ficha del auto con las señales del filtro contra los sanos del mercado y los chequeos sugeridos.
- **Conductor (app FordPass):** "Comparado con autos sanos de tu mercado, tu auto hace muchos viajes cortos. Un tramo de ruta de 20 minutos por semana ayuda al filtro a limpiarse".

**Slide 21 · Demo en vivo.** Replay de la flota semana a semana: un auto, su score subiendo, la alerta, el porqué y el mensaje. Con la GRU anterior, la demo muestra 53 alertas; 39 de los 135 autos que fallan reciben aviso antes, con una mediana de 15 semanas de margen.

**Slide 22 · Por qué es confiable.** Mientras se ajustaban los prompts, el verificador atajó 14 textos: lenguaje causal ("porque"), atribución ("lo marcó por…") y promesas. Ningún texto llega al cliente sin pasar esas reglas.

**Pendiente:** regenerar el bundle de la demo con la GRU de etiqueta suave y actualizar los números del replay.

**Límite que hay que decir:** el porqué es una comparación con la flota sana, no lo que el modelo usó. La efectividad de los avisos (¿el conductor cambia de hábitos?) se mide en el piloto.

## 5 · Escalabilidad: de 557 autos a toda la flota

Escala sin hardware nuevo: usa datos que Ford ya recibe, puntuar un millón de autos es un batch diario de 1–2 horas en CPU, y el límite real es la cantidad de eventos, no el cómputo.

**Slide 23 · Por qué escala.**

- **Cero sensores nuevos:** solo resúmenes por viaje y mensajes del filtro que el auto conectado ya manda.
- **Modelo liviano:** una GRU chica que corre en CPU en milisegundos por auto; sin GPU para servir ni para reentrenar.
- **Mismo código en entrenamiento y producción** para calcular las features. Evita que el piloto dé números distintos a los medidos.
- **Agentes solo sobre las alertas**, no sobre la flota: el costo de IA crece con las alertas, no con los autos.

**Slide 24 · Arquitectura.** Visual: diagrama de bloques.

1. Ingesta diaria de TripSummary + señales (el data lake de Ford).
2. Features por auto: ventana de 1.000 km, solo hacia atrás.
3. Scoring con el modelo empaquetado (pesos por semilla + export ONNX + referencia para el ensamble + umbrales).
4. Regla de alerta y política.
5. Agentes + verificador → bandeja de posventa, concesionario, app.
6. Retorno de taller → monitoreo y reentrenamiento.

El modelo se versiona como artefacto con alias candidato → sombra → producción. Un golden test re-puntúa 20 autos al desplegar y tiene que dar igual.

**Slide 25 · Cómo llega a producción.**

| etapa | qué | duración | criterio para pasar |
| --- | --- | --- | --- |
| 0 · Congelar | test final, reentrenar con los 557 autos, empaquetar | 1–2 semanas | test dentro del rango de dev |
| 1 · Piloto en sombra | puntuar la flota real de un mercado (COL o CHL) sin avisar a nadie | 3–6 meses | tasa de alertas ≈ presupuesto elegido; detección ≥ celda mercado × motor |
| 2 · Piloto activo | avisos en una región, con grupo control sin aviso | 6 meses | fallas evitadas contra el control; costo por alerta |
| 3 · Escala | todos los mercados, umbral recalibrado por mercado | continuo | monitoreo verde |

El piloto en sombra es el paso clave: es la primera vez que se mide la prevalencia real y las falsas alarmas sobre la flota completa, no sobre las listas de Ford.

**Slide 26 · Monitoreo y reentrenamiento.**

- Tasa de alertas por mercado contra el presupuesto: si se dispara, el umbral quedó viejo.
- Deriva de las features por mercado y mes (temperatura y estación primero).
- Detección realizada: cada falla en taller, ¿tuvo alerta antes y con cuánto margen?
- Datos que dejan de llegar: ya pasó una vez con el contador de regeneraciones.
- Reentrenamiento trimestral o con +50% de eventos nuevos, con las mismas auditorías y comparado en sombra contra el modelo vigente.

**Slide 27 · Riesgos y mitigación.**

| riesgo | mitigación |
| --- | --- |
| prevalencia real distinta a la del estudio | el umbral se recalibra en el piloto en sombra |
| fatiga de alertas | empezar con 5% de falsas alarmas y avisar solo con hábito explicable |
| cambio de firmware o telemetría | monitoreo de columnas nulas o cortadas por mercado |
| privacidad | solo agregados por viaje que Ford ya procesa, sin ubicación |
| mercados sin eventos (PER) | no se activa donde no se validó |

## 6 · Costos: lo que pierde el cliente y lo que cuesta el producto

El costo principal no es la reparación. Es lo que el cliente pierde mientras su motor pierde eficiencia sin que nadie le avise: gasoil, rendimiento y confianza. Evitarlo cuesta ~USD 0,10 por auto y por año.

**Slide 28 · Lo que paga el cliente hoy, sin saberlo.** La degradación es silenciosa durante meses:

- **Más gasoil.** El hollín acumulado sube la contrapresión del escape, y cada regeneración activa quema combustible extra. La literatura reporta una penalidad de más de 4%, y de 4,5–7% según la frecuencia de regeneración (valor a verificar, fuentes abajo).
- **En un auto típico** (15.000 km/año, 8 L/100 km, 1.200 L/año), 5% son **~60 litros de más por año** y **~160 kg de CO2** (2,68 kg por litro de diésel). Son supuestos redondos para ilustrar, no medidos en nuestros datos.
- **Menos rendimiento.** Regeneraciones interrumpidas y, al final, el modo de protección con potencia limitada.
- **Un testigo sin explicación** en el tablero y una visita al taller que no estaba en los planes.
- **La sensación de "nadie me avisó"**, que es lo que termina pesando en la relación con la marca.

**Slide 29 · Antes y después, desde el cliente.** Visual: dos líneas de tiempo paralelas.

| momento | hoy | con el producto |
| --- | --- | --- |
| ~3,5 meses antes | nada: el consumo sube y no se nota | aviso en la app con un hábito concreto ("un tramo de ruta de 20 minutos por semana") |
| mientras se degrada | gasta de más cada semana | corrige el hábito y el filtro se limpia en uso normal |
| si el riesgo sigue | testigo, modo de protección, grúa | el concesionario lo llama y le ofrece un turno |
| la relación | reclamo | "mi Ford me avisó a tiempo" |

**Slide 30 · Por qué una falsa alarma casi no le cuesta al cliente.** La primera acción es un consejo de manejo, no una visita. Si el auto no iba a fallar, el cliente recibió un consejo que igual le ahorra combustible. Por eso se puede operar con un presupuesto de falsas alarmas más alto que si cada alerta fuera un turno.

**Slide 31 · Lo que cuesta el producto.** Estimación propia para 1 millón de autos conectados, USD por año, a validar con Ford:

| componente | USD / año | supuesto |
| --- | --- | --- |
| Features + scoring diario | 2.500–6.000 | batch de 1–2 h en 8–16 vCPU |
| Agentes (LLM) | 1.000–3.000 | ~65.000 alertas/año × ~6 llamadas con un modelo chico; precio aproximado |
| Reentrenamiento + monitoreo | 5.000–10.000 | trimestral, en CPU |
| Equipo de operación | 60.000–80.000 | 1 persona ML/MLOps |
| **Total** | **~70.000–100.000** | **~USD 0,07–0,10 por auto y por año** |

Implementación, una vez: ~USD 70.000–110.000 (tres personas durante 6 meses, incluye el piloto en sombra). Almacenamiento adicional: casi nada, los viajes ya los guarda Ford.

**Slide 32 · Qué se mide en el piloto, del lado del cliente.**

- Consumo antes y después del aviso, en avisados contra un grupo control.
- Cuántos conductores cambian el hábito recomendado.
- Satisfacción (NPS) y retención en la red de servicio de los avisados.
- Visitas no planificadas y modos de protección evitados.

**Para Ford, en segundo plano:** además evita parte del costo de garantía. Los escenarios en USD de `f8-costos-k2.md` quedan en el backup, no en el cuerpo.

**Límite que hay que decir:** con estos datos no podemos medir bien la pérdida de consumo. El nivel de combustible solo permite estimarlo en el 33% de los viajes y no separa fallados de sanos. Por eso es la primera métrica del piloto.

Fuentes de la penalidad de consumo (resultados de búsqueda, no leídos completos): [ScienceDirect, filtro cargado de hollín y regeneración](https://www.sciencedirect.com/science/article/abs/pii/S0959652624020997) · [USPTO, regeneración de filtros de partículas](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/10378400) · [DieselNet, ceniza en DPF](https://dieselnet.com/tech/dpf_ash.php).

## 7 · Cierre: valor diferencial, trabajo futuro y conclusiones

El pedido a Ford es concreto: un piloto en sombra de 3 meses en un mercado, con los datos que ya tiene.

**Slide 33 · Valor diferencial e innovación.**

1. **Anticipación real, no reactiva.** Gap de 500 km y ~3,5 meses de margen, auditado contra leakage.
2. **Honestidad medida.** Cada número se compara contra el azar y contra saber mercado y motor. Es lo que hace creer el número en un piloto.
3. **Explica en hábitos y solo cuando la física acompaña.** Al conductor le llega algo que puede hacer, nunca un síntoma técnico.
4. **IA generativa con control.** Los agentes redactan, el código decide y verifica.
5. **Cero hardware nuevo.** Todo sale de la telemetría que Ford ya recibe.
6. **El punto de operación es una decisión de negocio.** Ford gira el dial según sus costos.

**Slide 34 · Trabajo futuro: qué necesitamos de Ford.**

- **Más eventos.** Hoy el límite son 177 fallas: la semilla mueve más que los hiperparámetros.
- **La fecha de la intervención**, no la del registro.
- **Dónde se usa el auto**, no solo dónde se vendió (altura, clima).
- **Costos reales** de inspección, limpieza y reemplazo por mercado.
- **La tasa real de falla por mercado × motor**, para separar la física de cómo se armaron las listas.
- **Extender a otras fallas** del postratamiento con el mismo circuito.

**Slide 35 · Conclusiones.** Las tres frases:

1. El DPF falla por cómo se usa el auto, y hoy Ford se entera tarde.
2. Con la telemetría que ya recibe, detectamos 1 de cada 3 autos que van a fallar, ~3,5 meses antes, con 5% de falsas alarmas.
3. Cada alerta se convierte en un consejo al conductor antes de que pierda eficiencia, o en un turno si el riesgo sigue, por ~USD 0,10 por auto y por año.

**Última slide, el arco cerrado.** Laura otra vez, tres meses antes. En la app le llega: "Comparado con autos sanos de tu zona, tu camioneta hace muchos viajes cortos. Un tramo de ruta de 20 minutos por semana ayuda a que el filtro se limpie solo". Hace el tramo. El testigo nunca se prende. Pantalla final: *"Tu Ford te avisa antes."* Ford Early Care.

**Límites que decimos antes de que los pregunten:** no predecimos la fecha exacta, la prevalencia real de la flota se mide en el piloto, y dentro de un mismo mercado y motor el orden entre autos es moderado (AUC 0,69).

**Preguntas probables del jurado (backup):**

| pregunta | respuesta corta |
| --- | --- |
| ¿No aprende solo qué mercado falla más? | Por eso lo comparamos contra saber mercado × motor, y le gana por +14,6 puntos |
| ¿Cómo sé que no es leakage? | Gap de 500 km, features hacia atrás, split por auto; con features permutadas cae al azar |
| ¿Por qué no accuracy? | Con ~6% de positivos, decir "nadie falla" da 94% |
| ¿Por qué una GRU y no algo más simple? | Probamos tabulares, supervivencia y series de tiempo; la GRU les gana con evidencia |
| ¿El LLM puede inventar algo? | No produce números ni diagnósticos; el verificador rechaza cualquier número que no esté en los hechos |
| ¿Qué pasa si el conductor ignora el aviso? | Se escala al concesionario si el riesgo sigue alto 2 revisiones después |

**Pendientes antes del 02-10:**

- [ ] Medir el test (un solo tiro) y reemplazar los números de dev.
- [ ] Regenerar la demo con la GRU de etiqueta suave.
- [ ] Correr los escenarios de costo con el finalista.
- [ ] Elegir el nombre del producto.
- [ ] Diseño visual del HTML.
