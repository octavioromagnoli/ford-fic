# Presentación Ford FIC III — Contenido y guion

29-09-2026 · Export del doc vivo: https://claude.ai/code/artifact/1e8d7eae-8435-4819-be16-ab64715822e4
(el doc es la versión que se edita; este archivo es una foto para el repo).

**01-10-2026:** el bloque 3, la demo, los costos y el cierre se actualizaron acá con el finalista medido en test,
la GRU con `label` de F10 (`docs/memoria/f11-test-resultado.md`). La GRU con etiqueta suave salió del pitch. El doc
vivo no se tocó. El mismo día, el guion siguió al deck: el hilo es **un usuario común** ("nuestro conductor", sin
nombre: Laura salió), y el cierre lo hace Octavio, sin "Si se llevan tres cosas" ni la caja "Lo que les pedimos".

## Estructura y storytelling

El relato va del problema al negocio. Un auto que se queda parado, los datos que lo anticipan, el modelo que lo detecta, lo que Ford hace con la alerta, cómo escala y cuánto vale. Los cinco bloques pedidos están todos. Los de la plantilla de Ford (valor diferencial, trabajo futuro, conclusiones) van en el cierre.

| # | bloque | pregunta que responde | bloque de la plantilla Ford | tiempo | orador |
| --- | --- | --- | --- | --- | --- |
| 1 | Apertura: el problema | ¿Por qué le importa a Ford? | 01 Descripción del desafío | 3 min | Gonzalo |
| 2 | EDA: los datos | ¿Qué hay en la telemetría y qué trampas tenía? | 02 Descripción de la solución | 4,5 min | Gonzalo |
| 3 | Modelo | ¿Qué probamos, qué ganó y cuánto detecta? | 02 Descripción de la solución | 6 min | Santino |
| 4 | Producto | ¿Qué pasa después de la alerta? (agentes + demo) | 04 Valor diferencial e innovación | 6 min (demo 2,5–3) | Octavio |
| 5 | Escalabilidad | ¿Cómo llega a 1 millón de autos? | 05 Trabajo futuro | 3 min | Octavio |
| 6 | Costos | ¿Qué pierde el cliente hoy y cuánto cuesta evitarlo? | 03 Factibilidad económica | 4 min | Gonzalo |
| 7 | Cierre | ¿Por qué es distinto y qué necesita de Ford? | 04 Valor diferencial · 05 Trabajo futuro · 06 Conclusiones | 2 min | Octavio |

Total: 28,5 min de exposición (Gonzalo 11,5 · Santino 6 · Octavio 11, con cuatro traspasos). Hablado, eso son \~3.800–4.100 palabras de guion (a \~130–140 palabras por minuto).

**Por qué este orden.** El producto va antes que costos y escalabilidad. El jurado tiene que ver la alerta llegando al conductor antes de escuchar números de infraestructura. Costos cierra el cuerpo porque lo que pierde el cliente es el argumento de compra. El cierre dice por qué es distinto y qué necesita de Ford, y termina con la marca: *"Tu Ford te avisa antes"*.

**El hilo que atraviesa todo:** "Hoy Ford se entera cuando el tablero ya avisó. Con la telemetría que ya recibe, lo sabe \~3 meses antes y sabe qué decirle al conductor."

**Reglas para citar números, en todo el deck:**

- Toda detección va al lado del azar. "31%" solo no dice nada. "31% donde el azar da 7%" sí.
- La anticipación se dice "lo marcamos con unos tres meses de margen", nunca "predecimos que falla en tres meses".
- Nada de accuracy ni de PR-AUC en el cuerpo: van al backup.
- El resultado del modelo se cita en test (111 autos del holdout, 32 fallados con datos): es lo que va en el deck, marcado [test]. Lo que no se pudo medir en test (la demo, la explicabilidad) va marcado [dev]. Con 32 fallas, se citan rangos: \~30% al 5%, \~60% al 20%.

**Cómo se escribe el guion:**

- **Cálido, como una historia contada.** Frases cortas, en segunda persona cuando se puede, con un usuario común como hilo ("nuestro conductor", sin nombre).
- **Cada bloque termina con una pregunta abierta** que el siguiente contesta. Nadie tiene que perder las ganas de ver lo que sigue.
- **Vender el producto pesa más que las métricas.** Los números importan, pero son la prueba de la historia, no la historia.
- **Las dificultades se cuentan.** Qué datos descartamos y por qué, qué panel elegimos, qué modelos fallaron y qué aprendimos de cada uno. Es lo que muestra que trabajamos en serio.
- **Marca temporal: Ford × SOG** (la misma de la demo). El nombre del producto se elige después; hasta entonces el guion usa *"tu Ford te avisa antes"*.

## Marca y mensaje: la confianza como producto

No vendemos un modelo que predice fallas: vendemos que el cliente sienta que su Ford lo cuida. La confianza entre el cliente y Ford es el producto; la GRU, los agentes y la telemetría son cómo se cumple.

**La gran idea:** *"Tu Ford te avisa antes."* Hoy el auto avisa cuando el filtro ya no da más, con un testigo que nadie explica. Mañana avisa con tiempo, en lenguaje humano y con algo concreto para hacer.

**Nombre y tagline (a elegir):**

| nombre | tagline | qué transmite |
| --- | --- | --- |
| Ford Early Care | "Te avisamos antes de que lo notes." | cuidado, anticipación |
| Respira | "Tu motor, siempre en su mejor forma." | eficiencia, algo vivo que se cuida |
| Ford Aliado | "Manejás vos. Te acompañamos nosotros." | relación, compañía |
| DPF Guard | "Protección inteligente para tu diésel." | técnico; mejor para Ford Pro |

Recomendación: **Ford Early Care** para el cliente particular. Habla de cuidado, no de fallas, y funciona en inglés para Ford global.

**Los tres pilares de la confianza.** Cada decisión técnica del proyecto se cuenta como uno de ellos:

1. **Te aviso a tiempo.** Más de tres meses de margen (mediana de los autos detectados), no el testigo de último momento. (Sale del modelo y del gap de 500 km.)
2. **Te explico por qué y qué hacer.** Un hábito concreto, nunca un código de error ni una amenaza. (Sale del porqué contra la flota sana y de los agentes con verificador.)
3. **No te asusto de más.** Primero un consejo, después un turno solo si hace falta. Ford elige cuántas falsas alarmas tolera, y ninguna cuesta caro al cliente. (Sale del dial y de la política.)

**Voz y tono de todo lo que le llega al cliente:** cálido, concreto, sin miedo. Hablamos de cuidar el auto, no de que se va a romper. Frases prohibidas en el deck y en el producto: "tu auto va a fallar", "detectamos un problema", porcentajes de riesgo al cliente.

**Una frase ancla por bloque** (el título grande de la slide principal de cada uno):

| bloque | frase ancla |
| --- | --- |
| Apertura | "El auto avisa cuando el filtro ya no da más. Y el cliente se entera solo." |
| EDA | "La forma de manejar deja huella meses antes." |
| Modelo | "Casi 1 de cada 3, tres meses antes, con 5% de falsas alarmas." |
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

**Slide 1 · Portada.** Ford × SOG (marca temporal, igual que la demo; el nombre del producto se elige después) + "Tu Ford te avisa antes". Equipo SOG.

**Slide 2 · Una historia, no un gráfico.** Un conductor hace viajes cortos en ciudad. El motor casi nunca llega a temperatura y el filtro no termina de regenerar. Durante meses el filtro se carga; un día se prende el testigo y, si sigue así, el auto limita la potencia para protegerse y termina en el taller sin haberlo planeado. Nadie le avisó. No pierde solo la reparación: pierde la confianza en su Ford. Auto ficticio, armado con los hábitos que distinguen a los fallados del dataset. **Sin grúa:** en modo de protección el auto suele llegar andando al taller, y los datos no dicen cómo llegó ninguno. Es **un usuario común**, sin nombre ("No es un cliente real"); después se lo retoma como "nuestro conductor" en el modelo, el producto y los costos.

**Slide 3 · Por qué duele.**

- Para el cliente: un testigo sin explicación y una visita al taller que no esperaba. El consumo extra es física conocida, pero con estos datos no lo medimos: no se dice.
- Para Ford: un cliente que siente que nadie le avisó, además del costo de garantía.
- Para flotas (Ford Pro): un vehículo de trabajo en el taller es trabajo que no se hace.
- Hoy la detección es reactiva: el aviso llega cuando el filtro ya está saturado.

**Slide 4 · La promesa.** Con la telemetría que Ford ya recibe del auto conectado, sin sensores nuevos:

1. marcamos los autos que van camino a la falla con \~3 meses de margen;
2. decimos por qué, en hábitos que el conductor puede cambiar;
3. Ford elige cuántas falsas alarmas tolera según lo que le cuesta cada una.

**Nombre, tagline y tono:** ver "Marca y mensaje".

### Guion · Gonzalo · 3 min

**[Portada, \~15 s]** Buenas. Somos Gonzalo, Santino y Octavio, el equipo SOG. Les venimos a contar cómo un Ford diésel puede avisarle a su dueño que algo no anda bien, meses antes de que tenga que ir al taller.

**[Un usuario común, \~60 s]** Pensemos en un usuario común. Vive en la ciudad y usa su camioneta para lo de todos los días: llevar a los chicos al colegio, ir al trabajo, hacer las compras. Viajes cortos, de pocos kilómetros, muchas veces con el motor en marcha y quieto en el tráfico.

No lo sabe, pero con ese uso el motor casi nunca llega a temperatura. Y durante meses, algo se va tapando sin que nadie lo note.

Un día se prende un testigo. Al tiempo, la camioneta pierde fuerza: se protege a sí misma. Y termina con un turno en el taller que no tenía planeado. No pierde solo una reparación: pierde la sensación de que su Ford lo cuidaba. Lo que más le molesta es que nadie le avisó.

No es un cliente real. Pero sus hábitos sí: son los que encontramos, en los datos de Ford, en los autos que fallaron.

**[Qué es el DPF, \~40 s]** Lo que se tapó es el filtro de partículas, el DPF. Atrapa el hollín del escape, y para limpiarse lo tiene que quemar: eso se llama regenerar. Para regenerar hace falta temperatura, un tramo de ruta, un motor caliente.

Con viajes cortos, la regeneración empieza y no termina. El hollín se acumula en silencio. El filtro no falla de golpe: se tapa de a poco, según cómo se usa el auto. Y eso, como vamos a ver, deja huella.

**[Por qué duele, \~35 s]** Hoy Ford se entera cuando el testigo ya se prendió. Para el cliente es un testigo que nadie le explicó, el auto en el taller y un día perdido. Para Ford, un cliente que siente que nadie le avisó, además del costo de garantía. Y si es un vehículo de trabajo, cada día en el taller es trabajo que no se hace.

**[La promesa, \~30 s]** Lo que proponemos usa solo la telemetría que Ford ya recibe de sus autos conectados, sin sensores nuevos. Con eso hacemos tres cosas. Marcamos los autos que van camino a la falla con una mediana de más de tres meses de margen. Le decimos al conductor por qué, en hábitos que puede cambiar. Y Ford decide cuántas falsas alarmas está dispuesto a tolerar.

Hoy el auto avisa cuando el filtro ya no da más. Nosotros queremos que tu Ford te avise antes.

**[Puente al EDA, \~10 s]** Para cumplir esa promesa, primero había que escuchar a los datos. Y la primera sorpresa fue que los datos, a veces, mienten.

## 2 · EDA: qué dicen los datos

La telemetría anticipa la falla, pero solo si se lee bien: los datos traían trampas que, sin corregir, hacían que cualquier modelo pareciera mejor de lo que es.

**Slide 5 · Qué mandó Ford.** Tres tablas, ninguna con clave fila a fila entre viajes y señales:

| tabla | qué tiene | escala |
| --- | --- | --- |
| Estática | mercado, ciudad de venta, motor, modelo, producción, venta, fecha del evento | 990 vehículos, 5 mercados |
| Viajes (TripSummary) | km, duración, temperaturas, nivel de acumulación del DPF, regeneraciones | \~13 M de filas |
| Señales | mensajes del filtro (avisos, limpiezas) | por evento |

**Slide 6 · Las trampas que encontramos (y corregimos).** Es la slide que muestra rigor:

- **Dos listas muestreadas distinto.** Fallados y sanos vienen de períodos de producción distintos: la fecha de producción sola separaba las cohortes (AUC 0,87). Nos quedamos con el período común, 20-01 a 31-07-2025.
- **La fecha del evento es la del taller.** Cae \~2 semanas después de la intervención. La corremos 21 días; si no, el modelo "ve" el taller.
- **13 vehículos duplicados bajo dos códigos**, filas repetidas y una fila por evento.
- **Un contador que se apagó.** El marcador de regeneraciones se corta el 25-05-2026 para toda la flota. Lo reconstruimos desde las caídas del nivel del filtro.
- **La tasa de falla depende de cómo se armó la lista.** Va de 0% (PER) a 48% (COL) por mercado. Por eso todo se compara dentro del mercado.

**Slide 7 · El universo del estudio.** 557 vehículos, 177 eventos. 446 para desarrollar y 111 congelados como test, sorteados antes de mirar ninguna feature.

**Slide 8 · Qué anticipa la falla.** Hallazgos comparando fallados contra sanos del mismo mercado:

- **El 33% de los viajes son ralentí de 0 km** (motor encendido sin moverse). Es la señal que más anticipa.
- **Hay trayectoria.** El exceso de ralentí crece al acercarse el evento: AUC 0,60 → 0,64 → 0,76.
- **Se ve temprano.** A 60–90 días de la venta, el uso ya separa fallados de sanos (AUC 0,60–0,64).
- **Viajes cortos y motor frío.** El filtro no termina de regenerar y el hollín se acumula. Es la física del DPF, y el modelo la confirma (bloque 3).
- **El riesgo es de edad, no de kilómetros.** Sube hasta \~4 meses desde la venta y después es plano: \~3,3 eventos por 100 autos-mes.

**Slide 9 · Dónde se concentra.** Mercado × motor:

| mercado | autos | eventos | celda caliente |
| --- | --- | --- | --- |
| COL | 113 | 54 (48%) | ENG\_2: 54 de 85 |
| CHL | 83 | 36 (43%) | ENG\_2: 33 de 51 |
| BRA | 136 | 41 (30%) | ENG\_3: 34 de 95 |
| ARG | 93 | 8 (9%) | — |
| PER | 21 | 0 | — |

Cifras de dev. La celda da un piso sin modelo: cualquier modelo tiene que ganarle a "saber mercado y motor".

**Backup:** altura de la ciudad de venta (va en la dirección física, no significativa), gasoil casi constante por país, auditoría de odometría.

### Guion · Gonzalo · 4,5 min

El guion escrito está completo. En la charla se elige qué decir (por ejemplo, tres trampas de cinco); el resto queda en el respaldo de abajo para las preguntas.

**[Lo que nos llegó, \~30 s]** Ford nos dio la telemetría de 990 autos diésel de cinco países: Argentina, Brasil, Chile, Colombia y Perú. Millones de viajes, cada uno con sus kilómetros, sus temperaturas y el nivel del filtro. Los mensajes del filtro. Y una lista de qué autos fallaron y cuándo.

Y en el medio del proyecto nos llegó una segunda entrega, con más países y más fallas. La recibimos con alegría y rehicimos todo. Varias cosas que creíamos saber dejaron de ser ciertas. Esa es la primera lección de este bloque: antes de creerle a un dato, hay que interrogarlo.

**[Los datos mienten: las trampas, \~1 min 30 s]** Les cuento algunas.

*El auto de 135.600 km/h.* Había viajes a esa velocidad, viajes de 230 km negativos y temperaturas de 73 grados bajo cero. Son pocos, pero alcanzan para arruinar un promedio. La velocidad la recalculamos desde los kilómetros y el tiempo, con topes físicos.

*El contador que se apagó.* Al principio, la mejor señal del análisis era el contador de regeneraciones. Hasta que vimos que el 25 de mayo de 2026 se apaga para toda la flota. Lo que medía no era el filtro, era el calendario: qué parte de la historia de cada auto caía antes de esa fecha. Lo descartamos y reconstruimos las regeneraciones desde el nivel del filtro, que sí se registra hasta el final.

*La fecha del taller.* En las dos semanas previas a la fecha de la falla ya aparecen el cambio de aceite y más ralentís largos, con el auto parado y el motor en marcha: la huella del taller y su regeneración forzada. La fecha que registra Ford llega unas dos semanas después de la intervención. Si no la corríamos, el modelo aprendía a ver el taller, no la falla. La corrimos 21 días.

*Dos listas que no se parecen.* Los autos que fallaron y los que no vienen de períodos de producción distintos. Tanto, que la fecha de fabricación sola ya separaba a unos de otros. Eso no es predecir, es reconocer la lista.

*Un país no es una señal.* La tasa de falla va de 0% en Perú a 48% en Colombia, y depende en parte de cómo se armaron las listas. Por eso todo lo que mostramos compara autos del mismo país.

**[Qué quedó adentro, \~45 s]** Con todo eso limpiamos. Unimos 13 autos que aparecían con dos códigos, sacamos filas repetidas y nos quedamos con el período de producción que las dos listas comparten: de enero a julio de 2025. Quedaron 557 autos y 177 fallas.

Antes de mirar un solo dato contra las fallas, guardamos 111 autos bajo llave como test. Trabajamos con los otros 446.

Y dejamos afuera lo que mide el calendario y no el auto: la fecha de fabricación, la de venta, la temperatura ambiente. Las usamos solo para auditar.

**[Cómo le preguntamos a los datos, \~45 s]** Después había que decidir la pregunta. Cada 500 km, miramos los últimos 1.000 km del auto y preguntamos: ¿este auto falla en los próximos 3.000? Y dejamos 500 km ciegos antes de la falla, porque sin ese margen estaríamos detectando lo que el tablero ya detecta.

Probamos ventanas de 1.000, 2.000 y 3.000 km: rinden lo mismo, y cada 500 km de más nos costaban unas tres fallas. Y a cada auto que falla lo comparamos con autos sanos del mismo kilometraje y del mismo mes. Si no, el modelo aprende en qué fecha estamos en vez de cómo se usa el auto.

**[La huella, \~1 min]** Entonces sí, la pregunta de fondo: ¿el uso anticipa la falla?

Sí. Comparando con autos sanos del mismo país y el mismo kilometraje, los que van a fallar pasan cada vez más tiempo parados con el motor en marcha, y terminan los viajes con el motor más frío. La huella crece a medida que se acerca la falla. Es la física del filtro que les contamos al principio, escrita en los datos.

También nos llevamos una sorpresa. La velocidad, que al principio parecía la señal más fuerte, desapareció al comparar dentro de cada país. No era la falla, era Colombia contra Chile.

Y lo más importante: la huella se ve temprano. A los 60 días de la venta, la huella ya asoma: tenue, pero está.

**[Puente a Santino, \~10 s]** La huella existe, pero es tenue: mirando una sola señal, acertamos un poco más que tirando una moneda. ¿Puede un modelo juntar todas y leerlas a tiempo? Eso se los cuenta Santino.

### Respaldo del bloque 2 (no se dice, se responde)

Todo sale de dev (446 autos); el test no se miró. Evidencia en `docs/memoria/f9-eda-v2.md`, `f9-universo-v2.md`, `f2-eda-revision-y-features.md`, `f2-calidad-columnas-dev.md` y `decisiones.md`.

**Las dos entregas.** La entrega 1 eran 364 autos de dos mercados parecidos y 60 eventos en dev. La v2 (26-09) trajo 557 autos de cinco mercados y 177 eventos. Qué cambió al rehacer todo:

| hallazgo con la entrega 1 | con la v2 |
| --- | --- |
| La velocidad baja antes de la falla (la señal más fuerte) | Desaparece dentro del mercado: era COL (lento, falla) contra CHL |
| No hay trayectoria: el exceso de ralentí no crece hacia el evento | Sí crece: 0,60 → 0,64 → 0,76 |
| El registro de eventos tiene una ventana (sep-2025 a mar-2026) | No hay ventana: era de la extracción vieja |
| El riesgo crece con la edad sin parar | Sube hasta \~4 meses y después es plano (\~3,3 por 100 autos-mes) |
| ENG\_3 nunca falla y queda afuera | ENG\_3 falla (34 de 95 en Brasil) y entra |
| La fecha es la intervención | La fecha cae \~2 semanas después: se corre 21 días |
| El rasgo temprano post-venta (AUC \~0,62–0,68) | Se sostiene con 3× los eventos y en 5 mercados (\~0,60–0,64) |

**Todas las trampas, cómo las vimos y qué hicimos:**

| trampa | cómo la vimos | qué hicimos |
| --- | --- | --- |
| Valores imposibles | 135.600 km/h, viajes de −230 km, −73 °C, viajes de 20 días (< 0,05% cada uno) | velocidad recalculada como km/duración, con topes físicos (200 km/h) |
| El nivel de combustible no es un porcentaje | supera 100 en el 10,5% de los viajes (hasta 103,5) | se trata como índice relativo, no se recorta a [0, 100] |
| El 33% de los viajes son ralentí de 0 km | la velocidad reportada es nula justo ahí | las fracciones "de viaje" se calculan entre los que se mueven; el ralentí se vuelve una señal propia (la que más anticipa) |
| El contador de regeneraciones se apaga | 0 marcadores desde el 25-05-2026 en toda la flota, contra \~2.000 caídas de nivel por mes en los viajes | fuera del modelo; se cuenta desde las caídas del nivel del filtro. Su correlación con la falla cae de 0,317 a 0,072: la mayor parte era calendario |
| La fecha del evento es la del taller | ralentí largo (≥ 15 min) 1,5–1,8× en las 2 semanas previas a la fecha; el día de la fecha el auto se usa más que nunca | referencia del evento = fecha − 21 días |
| Dos listas de períodos de producción distintos | la fecha de producción sola separa fallados de sanos (AUC 0,87 dentro del mercado) | universo = período común, 20-01 a 31-07-2025 |
| 13 autos con dos códigos | mismos viajes repetidos bajo dos identificadores | se colapsan antes de cualquier split; si no, un auto cae en train y validación a la vez |
| Formato de la v2 | columna del evento renombrada, fecha de producción de fallados corrida +538 días, una fila por evento, filas repetidas, extracción de fallados 10 días más larga | se corrige al leer, declarado en un solo archivo de configuración |
| La tasa por país depende de la lista | de 0% (PER) a 48% (COL), en el orden inverso de cuántos fallados sin fecha sacó Ford de cada país | toda comparación dentro del mercado (y del motor) |
| Después de la falla el uso cambia | bajo régimen baja en el 72% de los autos, la velocidad sube en el 79% | nada posterior al evento entra a ninguna ventana |

**Qué entró y qué no:**

| dato | decisión | por qué |
| --- | --- | --- |
| Viajes: ralentí, temperaturas, duración, nivel del filtro, regeneraciones reconstruidas | entra (53 features en 4 familias: térmica, regeneración, uso, severidad) | es el uso y el estado del filtro, hacia atrás |
| Mensajes del filtro (avisos, limpiezas) | entra como tasas por 1.000 km | severidad |
| Mercado, motor y modelo | entran | aportan la tasa de su celda; por eso todo se mide también dentro de mercado × motor |
| Fecha de producción, de venta, días hasta la venta | afuera (solo auditoría) | miden la cohorte, no el auto |
| Temperatura ambiente | afuera (solo auditoría) | deriva con el mes y el lugar |
| Contador de regeneraciones | afuera (solo auditoría) | se apaga en mayo de 2026 |
| Interrupciones de regeneración | no se construye | 1 mensaje de cada 100.000 |
| Vida del aceite dentro del viaje | se reemplaza por su pendiente | el delta dentro del viaje es siempre 0 |
| Altura, presión de neumáticos | no hay columna | la altura se probó desde la ciudad de venta (abajo) |
| Los 433 autos fuera del período común | afuera | se probó usarlos (990 autos) y los modelos ordenaron peor a los autos comparables |

**El universo:** 557 autos, 177 eventos. 446 en dev y 111 en test, sorteados con semilla 42 y estratificados por evento × mercado × motor, antes de mirar ninguna feature. 36 autos del test estaban en el dev de la entrega 1: el test se reporta también sin ellos.

**El panel y por qué esos números.** Una fila por (auto, corte). Cortes cada Δ = 500 km. Ventana W = 1.000 km hacia atrás, gap G = 500 km, horizonte H = 3.000 km.

- **W = 1.000.** Con la entrega 1, cada 500 km más de ventana costaba \~3 eventos, y W = 1.000, 2.000 y 3.000 daban la misma métrica dentro del ruido.
- **G = 500.** Es la regla anti-reactiva: el modelo nunca ve los km previos al evento.
- **H = 3.000.** Pone los positivos entre 500 y 3.500 km antes del evento, donde el perfil muestra que la señal existe.
- **Sanos emparejados por odómetro y mes.** Los eventos se concentran en ciertos meses. Sin emparejar por mes, agregar tres columnas de calendario subía el ROC de 0,67 a 0,76; emparejando, de 0,57 a 0,60 (entrega 1).
- **El evento se ordena por tiempo, no por km.** La dispersión (sd del log) es 0,38 en edad contra 0,91 en odómetro. Quien maneja más llega con más km, no antes.

**Correlaciones: el perfil alineado al evento.** Probabilidad de que un fallado supere a un sano del mismo mercado y el mismo tramo de odómetro (0,5 = no distingue):

| señal | 4–8k km antes | 2–4k | 0–2k | lectura |
| --- | --- | --- | --- | --- |
| ralentí (viajes de 0 km) | 0,60 | 0,64 | 0,76 | la que más anticipa, y crece |
| bajo régimen (motor < 70 °C) | 0,55 | 0,62 | 0,73 | crece |
| refrigerante al terminar | 0,47 | 0,39 | 0,29 | más frío cerca del evento |
| velocidad (entre los que se mueven) | 0,51 | 0,47 | 0,49 | no distingue dentro del mercado |
| viajes cortos (entre los que se mueven) | 0,46 | 0,47 | 0,47 | no distingue |
| nivel del DPF al terminar | 0,45 | 0,43 | 0,40 | levemente al revés |
| regeneraciones por 1.000 km | 0,45 | 0,40 | 0,41 | levemente menos |
| temperatura ambiente (control) | 0,45 | 0,42 | 0,38 | los fallados viven en lugares más fríos |

- **Lo que crece es estar parado con el motor en marcha y no llegar a temperatura, no manejar distinto.**
- **Comparar todo junto infla lo que es país:** filtro anormal pasa de 0,52 a 0,58, DPF saturado de 0,52 a 0,59, duración de 0,56 a 0,61.
- **Rasgo temprano:** con los primeros 60 días después de la venta, el ralentí ordena autos con AUC 0,636 dentro del mercado (125 fallados, 298 sanos). Las regeneraciones por km y el consumo van al revés (0,38 y 0,40).
- **Riesgo por edad:** eventos por 100 autos-mes de 0,34 (0–60 días desde la venta) a 2,11 (60–120) y \~3,1–3,9 después. El mes calendario no agrega nada (p = 0,16). El trimestre de venta sí (los de 2025Q1 fallan 2,1× más que los de 2025Q2 a igual edad): por eso las fechas quedan fuera.
- **Modelo y motor cuentan casi la misma historia:** MODEL\_2 es solo ENG\_2 (62 de 135 fallan).
- **Altura de la ciudad de venta (Colombia):** 1,7 eventos por 100 autos-mes bajo 1.000 m, 6,3 sobre 2.000 m. Ajustando por motor, RR 1,30 por cada 1.000 m [0,93; 1,81]: va en la dirección de la física, pero no se separa del cero.

**Lo que probamos en el EDA y no sirvió:** fechar a los fallados sin fecha desde la telemetría (cambio de aceite, días sin uso, DPF con motor apagado): no pasó su criterio. Usar los 990 autos: la fecha de producción dominaba. Tasa de marcadores de regeneración: era calendario.

## 3 · Modelo: qué probamos y qué ganó

El finalista es una red recurrente (GRU) que lee la secuencia de viajes y señales del filtro, más país, motor y modelo, con un ensamble de 3 semillas. En el test (111 autos bajo llave) avisa a 10 de los 32 autos que fallaron con 5% de falsas alarmas: 31%, 4,5× el azar.

**Slide 10 · La pregunta, bien planteada.** Cada 500 km, el modelo mira los últimos 1.000 km del auto y responde: "¿este auto falla en los próximos 3.000 km?". Nunca ve los 500 km previos al evento (gap de blanking). Sin ese gap sería detección reactiva, lo que Ford ya tiene. Visual: línea de tiempo ventana → gap → horizonte.

**Slide 11 · Reglas que no negociamos.** Split por vehículo (un auto nunca está en train y validación a la vez), features solo hacia atrás, test congelado desde el día uno (nunca se usa para entrenar ni para ajustar), y cada modelo auditado contra leakage: si se permutan las features, el score cae al azar.

**Slide 12 · El recorrido.** Más de 40 modelos medidos con la misma cuenta. Las barras son la detección **en test** al 5% de falsas alarmas, cada modelo entrenado con todo dev; la línea es el azar (7%).

| familia | modelo | test al 5% | qué aprendimos |
| --- | --- | --- | --- |
| Tabulares | logística sobre 53 features | 6% | aprenden qué auto falla, casi nada del cuándo |
| Tabulares | LightGBM | 16% | ídem |
| Supervivencia | survival stacking | 19% | fue el finalista con la entrega 1; con la v2 lo pasan los secuenciales |
| Supervivencia | con efecto aleatorio por auto | 9% | el efecto por auto absorbe justo lo que distingue autos |
| Series de tiempo | TimesFM zero-shot, MiniRocket | — | no sumaron en dev; no llegaron al test |
| **Finalista** | **GRU + viajes + estática ×3** | **31%** | lee la secuencia en orden; con 3× eventos pasa adelante |

**Slide 13 · Cómo funciona la GRU.** Cada 500 km toma los últimos 1.000 km en 20 tramos de 50 km. En cada tramo ve cómo se viaja (duración, ralentí, motor frío, velocidad) y qué hace el filtro (acumulación, regeneraciones, avisos); además sabe país, motor y modelo. Lee los tramos en orden y pasa una memoria de uno al siguiente: nota si el hollín viene subiendo o si los viajes se acortan, no solo el promedio. Da un puntaje por auto; tres redes con semillas distintas (42, 1 y 2) votan por rango. Es chica (24 unidades de memoria) y corre en CPU. Visual: la red desenrollada sobre los tramos.

**Slide 14 · El resultado, en test.** Detección de autos que fallaron, según el porcentaje de sanos con falsa alarma. 103 autos con datos (32 fallaron, 71 sanos), nunca usados para entrenar:

|  | 5% | 10% | 15% | 20% |
| --- | --- | --- | --- | --- |
| **GRU (finalista)** | **31,2%** | **46,9%** | **62,5%** | **62,5%** |
| Survival stacking (finalista entrega 1) | 18,8% | 21,9% | 21,9% | 43,8% |
| Azar (media, mismo historial) | 7,0% | 13,7% | 18,6% | 24,4% |
| Azar (p95) | 15,6% | 25,0% | 31,2% | 40,6% |

Visual: la curva detección vs. falsas alarmas con azar, survival stacking y la GRU.

- **10 de 32 al 5%: casi 1 de cada 3**, 4,5× el azar y por encima de su p95.
- **El doble que survival stacking** en promedio entre 5% y 20% (50,8 contra 26,6).
- **Con el umbral fijado antes, en dev:** 20 · 34 · 57% al 5 · 10 · 20%, con falsas alarmas reales de 5,8 · 8,9 · 18,2% en test.
- **Repitió dev:** en validación cruzada daba 27 · 42 · 61% al 5 · 10 · 20%.

**Slide 15 · Cuánto antes.** La primera alerta llega con una mediana de **\~4.600 km, unos 3 meses (\~100 días)** antes de la falla registrada (test, al 10%). El score sube a medida que se acerca el evento: rango percentil medio 0,44 en sanos, 0,58 a más de 10.000 km, 0,61 a 3.500–10.000 km y 0,72 a menos de 3.500 (test).

**Slide 16 · En qué se apoya** (dev, semilla 42, permutando cada señal en la validación; detección base 52%):

- **duración de los viajes:** −24 puntos. Es el hábito que más pesa, por delante de km por viaje (−10), velocidad (−6), motor frío (−5) y ralentí (−2);
- **cuánto sube el hollín en el filtro:** −22 puntos (nivel medio −10, máximo −9, regeneraciones −9);
- **país:** −28 puntos; motor −15.

Es la lectura física: viajes cortos que no dejan terminar la regeneración y el hollín que se acumula.

**Backup:** auditorías de leakage, (a′) y calendario, sweep bayesiano (130 trials, la semilla mueve más que los hiperparámetros), AUC dentro de mercado × motor, por qué no accuracy.

### Guion · Santino · 6 min

Los números marcados **[test]** son de los 111 autos del holdout (103 con datos, 32 fallados), con cada modelo entrenado con todo dev. Los marcados **[dev]** son de validación cruzada sobre los 446 autos de desarrollo. El panel ya lo planteó Gonzalo: acá se retoma en una línea.

**[Retomar, \~15 s]** Gonzalo les dejó una pregunta: cada 500 km, mirando los últimos 1.000, ¿este auto falla en los próximos 3.000? La huella existe, pero es tenue. Mi trabajo fue encontrar un modelo que la lea a tiempo. Y antes de contarles cuál, les cuento cómo nos cuidamos de engañarnos.

**[Las reglas, \~45 s]** Cuatro reglas que no negociamos. Primero, el modelo nunca ve los 500 km antes de la falla: si los viera, estaría detectando lo que el tablero ya detecta. Segundo, un mismo auto nunca está a la vez en entrenamiento y en validación. Tercero, los 111 autos del test están bajo llave desde el primer día: nunca se usan para entrenar ni para ajustar, solo para medir. Y cuarto, cada modelo se audita: si le mezclamos los datos al azar y el resultado no cae al azar, hay una fuga, y el modelo se descarta.

**[El recorrido, \~2 min]** Probamos más de 40 modelos, todos medidos con la misma cuenta.

Empezamos por lo clásico: una regresión logística y árboles de decisión sobre 53 indicadores del uso. Daban números que parecían buenos. Hasta que nos preguntamos: ¿y si en vez de predecir, el modelo solo reconoce qué auto falla? Probamos darle a cada auto un puntaje fijo, sin nada del cuándo, y la métrica clásica daba casi lo mismo. Ese día cambiamos cómo medimos. Desde entonces la pregunta es una sola: **de los autos que van a fallar, ¿a cuántos avisamos a tiempo, si solo le permitimos una cantidad fija de falsas alarmas?** Y cada respuesta se compara con el azar: con 5% de falsas alarmas, un puntaje al azar avisa a un 7%.

Lo que ven son los números en el test. La logística y LightGBM quedan cerca del azar. Después vinieron los modelos de supervivencia, los que usa la medicina para estimar cuánto falta para un evento: con la primera entrega de Ford fueron nuestros finalistas, y en el test llegan a 19%. Probamos también TimesFM, un modelo fundacional de Google, y MiniRocket, un clasificador de series muy rápido: en desarrollo no sumaron.

Cuando llegó la segunda entrega, con tres veces más fallas, las redes que leen la secuencia de viajes pasaron adelante. **Nuestro finalista es una GRU: en el test avisa a 31% de las fallas con 5% de falsas alarmas**, más de cuatro veces el azar y el doble de survival stacking.

**[Cómo funciona, \~1 min]** La GRU es una red recurrente. Cada 500 km toma los últimos 1.000 km del auto, cortados en 20 tramos. En cada tramo ve cómo se viaja, la duración de los viajes, el ralentí, el motor frío, la velocidad, y qué hace el filtro: cuánto hollín acumula, cuándo regenera, qué avisos da. Además sabe el país, el motor y el modelo.

Lo importante es que lee los tramos en orden y recuerda. No mira un promedio: pasa una memoria de un tramo al siguiente, así que nota si el hollín viene subiendo o si los viajes se vienen acortando. Al final da un puntaje por auto. Entrenamos tres redes con semillas distintas y las hacemos votar, porque con pocas fallas una sola red depende demasiado de la suerte.

**[El resultado, \~1 min]** Este es el resultado en el test: 111 autos que estuvieron bajo llave todo el proyecto. Con 5% de falsas alarmas, el modelo avisa a **10 de los 32 autos que fallaron: casi 1 de cada 3 [test]**. El azar avisa a 7%. Si Ford acepta 20% de falsas alarmas, llegamos a **6 de cada 10 [test]**.

Le duplica la detección a survival stacking, nuestro finalista de la primera entrega. Y el test repitió lo que veíamos en desarrollo. Una cosa más, la que importa para operar: si fijamos el umbral antes, con los autos de desarrollo, en el test las falsas alarmas quedan en 5,8%.

**[Cuánto antes, \~20 s]** La primera alerta llega con una mediana de **unos tres meses de margen [test]**: unos 4.600 km antes de la falla. Y el puntaje sube a medida que se acerca la falla: el modelo no solo sabe qué auto, también nota que el momento se acerca.

**[En qué se apoya, \~30 s]** Si le escondemos al modelo la duración de los viajes, detecta 24 puntos menos [dev]: es el hábito que más pesa. Si le escondemos cuánto sube el hollín en el filtro, 22. Es la misma física que les contó Gonzalo: viajes cortos que no dejan terminar la limpieza del filtro. El país también pesa: el riesgo no es el mismo en todos los mercados. No le dimos ninguna regla del filtro: esa relación la aprendió de los datos.

**[Puente a Octavio, \~10 s]** Ahora el modelo marcó el auto de nuestro conductor, con meses de margen. Pero una alerta en un servidor no le sirve al conductor. ¿Qué le decimos, y cómo, sin asustarlo? Eso se los cuenta Octavio.

### Respaldo del bloque 3 (no se dice, se responde)

Evidencia en `docs/memoria/f11-test-resultado.md` (test), `f9-remedicion-completa-v2.md` (dev), `f10-sweep-gru.md`, la primera entrada de `decisiones.md`; la explicabilidad, en `configs/explain_gru_final.yaml`.

**Cómo medimos, y por qué cambiamos.** El PR-AUC por fila no alcanza: en este panel, puntuar cada fila con "¿este auto falla?" (sin nada del cuándo) ya da PR-AUC 0,177 y lift 3×, el "techo de cohorte". Por eso medimos la **detección por auto**: la alerta es 2 cortes seguidos sobre un umbral, y el umbral se fija para que solo el X% de los autos sanos tenga una falsa alarma. Cada número va contra el nulo (el mismo puntaje mezclado entre filas, conservando el largo del historial de cada auto) y contra el modelo anterior, con bootstrap pareado por vehículo.

**El finalista, completo.** `gru_seq`: GRU de 1 capa (24 unidades) con pooling por atención + rama estática + cabeza, dropout 0,3. Entrada: secuencia de señales + TripSummary en 20 bins de km de la ventana, más país, motor y modelo. Entrenado con la etiqueta dura (`label`), 40 épocas, semillas 42, 1 y 2, ensamble por rango. Modelo final: cada semilla con todo dev (`configs/exp_v2all_gru_trips_estaticas{,_s1,_s2}_r3.yaml`).

**Todos los modelos en test** (103 autos, 32 fallados; cada uno entrenado con todo dev; descriptivo, no elige):

| modelo | 5% | 10% | 15% | 20% | AUC auto | dev (5 · 10 · 20%) |
| --- | --- | --- | --- | --- | --- | --- |
| **GRU ×3 (42/1/2)** | **31,2** | **46,9** | **62,5** | **62,5** | 0,794 | 27 · 42 · 61 |
| Survival stacking (F3) | 18,8 | 21,9 | 21,9 | 43,8 | 0,710 | 14 · 24 · 42 |
| LightGBM (control, 53 features) | 15,6 | 25,0 | 28,1 | 34,4 | 0,725 | 18 · 28 · 42 |
| Logística L2 (53 features) | 6,2 | 12,5 | 31,2 | 34,4 | 0,679 | 15 · 23 · 37 |
| SS + efecto aleatorio por auto | 9,4 | 9,4 | 9,4 | 12,5 | 0,363 | 6 · 9 · 17 |
| Azar (media / p95, GRU) | 7,0 / 15,6 | 13,7 / 25,0 | 18,6 / 31,2 | 24,4 / 40,6 | — | — |

- **GRU contra survival stacking:** +24,2 puntos de detección media al 5–20%, IC95 [0,8; 43,0]. No estaba preregistrado.
- **Con 32 fallas, cada auto mueve 3 puntos.** La GRU se leyó tres veces en test con la misma configuración (31 · 34 · 28% al 5%): se cita **\~30% al 5%, \~40–50% al 10%, \~50–60% al 20%**.
- **Con el umbral fijado en dev** (lo que haría Ford en operación): 19,6 · 34,2 · 57,3% al 5 · 10 · 20%, con falsas alarmas reales de 5,8 · 8,9 · 18,2%.
- **Sin los 36 autos de test que estaban en el dev de la entrega 1** (22 fallados): 36,4 · 50,0 · 59,1% al 5 · 10 · 20%.
- **Anticipación** al 10%: mediana de \~4.600 km (p25–p75: 2.400–9.300), \~100 días aproximados con los km por día de cada auto.
- **AUC por auto** 0,794; dentro de mercado × motor 0,675. Entre autos del mismo país y motor, el orden es moderado.

**Auditorías (dev, semilla 42):**

| auditoría | qué mide | resultado |
| --- | --- | --- |
| (a0) | features permutadas entre todas las filas: ¿cae a la tasa base? | 0,061 contra 0,058: sin fuga |
| (a′) | colapsar el score al promedio del auto: ¿cuánto sabía del cuándo? | +0,001: chico pero positivo |
| (b) | ¿sumar el calendario mueve el ROC? | +0,004: no marca |

**Lo que probamos sobre la GRU y no sirvió:**

- Sweep bayesiano de 130 trials (Optuna): el mejor dio 0,492 contra 0,475, pero con semillas nuevas empata (0,469 contra 0,471). La semilla mueve más que los hiperparámetros, y por eso todo se reporta como ensamble de 3 semillas.
- Suavizar el puntaje en el tiempo (media acumulada, EWMA, CUSUM): resta 2–7 puntos, porque la detección vive en los picos cerca del evento.
- La historia completa del auto como estáticas, y 11 canales más en la secuencia: no suman o restan.
- Ensambles por rango (SS + LightGBM, SS + CNN-LSTM, celda + GRU): no le ganan al mejor miembro con evidencia.

**Preguntas probables del bloque 3:**

| pregunta | respuesta corta |
| --- | --- |
| ¿No aprende solo qué país falla más? | En parte: el país pesa (−28 puntos si se lo escondemos). Pero la duración de los viajes y el hollín pesan casi lo mismo, y dentro de un mismo país y motor ordena autos con AUC 0,68 [test] |
| ¿Por qué una GRU y no algo más simple? | Lo simple lo probamos primero: tabulares y supervivencia detectan la mitad en el test |
| ¿Por qué no accuracy? | Con \~6% de filas positivas, decir "nadie falla" da 94% |
| ¿40 modelos no es buscar hasta encontrar? | Se eligió en dev, y el test (111 autos que nunca se usaron para entrenar ni ajustar) repitió lo de dev |
| ¿Por qué 31% y no un número más redondo? | Son 32 fallas: cada auto mueve 3 puntos. Por eso citamos \~30% |

**Para estudiar (Santino): lo que no hay que exagerar**

- **Son 32 fallas.** El IC de cualquier diferencia es ancho: contra survival stacking, [0,8; 43,0]. Decir "el doble", nunca "el doble con certeza".
- **El modelo sabe poco del cuándo.** (a′) es +0,001: positivo pero chico, como en todos los modelos de v2. Lo que se puede decir es que el score sube hacia el evento (rango 0,58 → 0,61 → 0,72 en test). Nunca "predecimos cuándo falla".
- **El país pesa.** Si preguntan por "saber país y motor sin modelo": en test esa regla detecta 0% al 5% y 50% al 20%; la GRU le saca +27 puntos de media, IC95 [−0,8; 46,1], en el límite. No decir que "le gana con evidencia".
- **El test no se miró una sola vez.** La configuración de la GRU se leyó tres veces en test (una antes de preregistrar una variante con etiqueta suave lejos del evento, que en test no se sostuvo y se descartó). Por eso se cita un rango y no un número. La GRU no se ajustó mirando test.
- **La anticipación en km del test (4.600) es menor que la de dev (7.500)**, pero en días es parecida (\~100): los autos detectados del test andan menos km por día. Decir "unos tres meses".

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

**Slide 21 · Demo en vivo.** Replay de la flota semana a semana: un auto, su score subiendo, la alerta, el porqué y el mensaje. Con la GRU finalista, al 5%: 53 alertas y 7 escalamientos; 39 de los 135 autos que fallan reciben aviso antes, con una mediana de 15 semanas (\~7.000 km); 14 de 291 sanos reciben un aviso de más. La perilla va de 5% a 20%.

**Slide 22 · Por qué es confiable.** Mientras se ajustaban los prompts, el verificador atájó 14 textos: lenguaje causal ("porque"), atribución ("lo marcó por…") y promesas. Ningún texto llega al cliente sin pasar esas reglas.

**Bundle:** `demo-bundle-gru-final` (01-10), la GRU finalista con la perilla 5 · 10 · 15 · 20%.

**Límite que hay que decir:** el porqué es una comparación con la flota sana, no lo que el modelo usó. La efectividad de los avisos (¿el conductor cambia de hábitos?) se mide en el piloto.

### Guion · Octavio · 6 min (demo 2,5–3)

Los números de la demo son de su replay (una repetición de dev, 426 autos). Los oficiales, promedio de 3 repeticiones, van marcados **[dev]**.

**Ojo con la anticipación (para no errarle):** Santino ya dijo "unos tres meses" (el oficial es \~4.600 km, \~100 días, en test al 10%). En la demo se ve otro número: 15 semanas (\~7.000 km). No es un error, se miden distinto (ver el respaldo). Octavio dice siempre **"en esta temporada, unas 15 semanas: más de tres meses"**, nunca "4.600 km". Si preguntan por qué no coincide: *"el oficial es el test al 10% de falsas alarmas; la demo es una temporada de desarrollo al 5%"*.

**[Retomar, \~15 s]** Santino nos dejó con una alerta: el auto de nuestro conductor parece ir camino a la falla. Pero una alerta en un servidor no evita nada. Lo que evita la falla es que alguien haga algo. Y eso es lo que construimos.

**[El principio, \~30 s]** Hay una regla que atraviesa todo el producto: **el modelo decide, el agente comunica, y el código cuida lo que se dice.** Usamos inteligencia artificial generativa para escribir los mensajes, porque escribe como una persona. Pero nunca le dejamos decidir nada: el LLM no produce un número, ni un puntaje, ni un diagnóstico. Solo redacta con los hechos que el sistema le da.

**[El circuito, \~1 min]** Funciona así. Todos los días llega la telemetría que Ford ya recibe. Cada 500 km, la red le pone un puntaje a cada auto. Si el puntaje queda dos veces seguidas sobre el umbral que eligió Ford, hay alerta.

Después buscamos el porqué: en qué hábitos este auto se aparta de los autos sanos de su mismo país. Y acá hay una política simple, que define Ford. Si hay un hábito para nombrar, le avisamos al conductor con un consejo. Si no lo hay, no tiene sentido pedirle nada al conductor, y lo llama el concesionario. Y si dos revisiones después el riesgo sigue alto, escalamos al concesionario.

Recién ahí entran los agentes: uno redacta, otro organiza la semana. Y un verificador, escrito en código, revisa cada texto antes de que llegue a nadie.

**[Tres personas, tres mensajes, \~40 s]** Cada persona recibe lo suyo.

El conductor recibe esto en la app de su Ford: *"Comparado con autos sanos de tu zona, tu camioneta hace muchos viajes cortos. Un tramo de ruta de 20 minutos por semana ayuda a que el filtro se limpie solo."* Sin códigos de error, sin porcentajes, sin miedo. Algo que puede hacer el sábado.

El taller recibe las señales técnicas del filtro y qué revisar. Y la persona de posventa de Ford recibe, cada lunes, una bandeja con todo priorizado y redactado, lista para aprobar.

**[Demo en vivo, \~2 min 30 s]** Les muestro. *(Abrir la demo en la bandeja.)*

Esta es la bandeja del lunes 29 de septiembre de 2025. Estamos reproduciendo, semana a semana, los autos reales del estudio, y el sistema solo sabe lo que se sabía ese día. Esta semana hay cuatro avisos: dos son un consejo al conductor y en dos lo llama el concesionario. *(Abrir una tarjeta.)* Acá está el mensaje al conductor, el resumen para el taller y los hechos en los que se apoya el texto. Este tilde verde es el verificador: el texto pasó todas las reglas.

*(Ver ficha.)* Esta es la ficha del auto. El puntaje fue subiendo, cruzó el umbral y se confirmó la alerta. Y acá está el porqué: en estos hábitos, el auto se aparta de los sanos de su país.

*(Mover la perilla de 5% a 20%.)* Esta perilla es la decisión de Ford. Con 5% de falsas alarmas avisamos a más de un cuarto de las fallas [dev]. Con 20%, a seis de cada diez [dev], pero con cuatro veces más avisos de más. Como el primer aviso es un consejo y no una visita al taller, Ford puede permitirse girarla.

*(Abrir "Qué pasó después".)* Y como esto ya pasó, podemos ver cómo terminó. Los cuatro avisos de esa semana eran de autos que después fallaron. No siempre es así, y lo mostramos igual: en esta temporada, al 5%, avisamos a 39 de los 135 autos que fallaron, unas 15 semanas antes: más de tres meses. Y 14 de 291 autos sanos recibieron un aviso de más.

**[Por qué es confiable, \~30 s]** En toda la temporada, los agentes escribieron 465 textos. El verificador rechazó 103 intentos en el camino: frases causales, promesas, atribuciones. Los agentes corrigieron y volvieron a intentar, y siempre lo lograron. Si un día no lo logran, el sistema no fuerza nada: usa una plantilla aprobada. Ningún texto le llega a un cliente sin pasar esas reglas.

**[Puente a escalabilidad, \~10 s]** Todo esto lo vieron con 426 autos. La pregunta de Ford es otra: ¿funciona con un millón? Y la respuesta es sí, sin un solo sensor nuevo.

### Respaldo del bloque 4 (no se dice, se responde)

Fuentes: el bundle `demo-bundle-gru-final` (01-10), `docs/memoria/f9-demo-gru.md`, `f9-demo-producto.md` y `PRODUCT.md`. Todo es dev v2: 426 autos, 135 que fallan y 291 sanos. El test no se muestra nunca, y el bundle falla si aparece un auto de test.

**Por qué hay dos números de anticipación:**

|  | oficial (Santino) | demo (Octavio) |
| --- | --- | --- |
| número | \~4.600 km, \~100 días (\~3 meses) | 15 semanas, \~7.000 km (más de 3 meses) |
| falsas alarmas | 10% | 5% |
| autos | test (111, 32 fallados con datos) | dev, una repetición (426, 135 fallados) |
| qué mide | primera alerta → falla registrada | alerta confirmada → falla, en el calendario del replay |

Los dos son correctos. Los autos son otros, y el replay cuenta en calendario: los km por día cambian de auto a auto. Con más tolerancia, la alerta llega antes: 16 semanas al 10% y al 15%, 21 al 20%.

**El replay por punto de la perilla:**

|  | 5% | 10% | 15% | 20% |
| --- | --- | --- | --- | --- |
| detección oficial [dev] | 28,1% | 42,5% | 53,1% | 61,0% |
| fuera de muestra: detección / falsas alarmas reales | 27,9 / 4,9% | 42,5 / 10,2% | 52,3 / 15,1% | 60,7 / 19,9% |
| azar (mismo historial) | 5,4% | 10,7% | 15,7% | 20,7% |
| fallas anticipadas en el replay (de 135) | 39 | 56 | 69 | 82 |
| sanos con aviso de más (de 291) | 14 | 29 | 43 | 58 |
| alertas / escalamientos | 53 / 7 | 85 / 10 | 112 / 22 | 140 / 36 |
| alertas con hábito para nombrar | 48 de 53 | 73 de 85 | 101 de 112 | 125 de 140 |
| anticipación mediana | 15 sem · 7.000 km | 16 sem · 7.200 km | 16 sem · 7.500 km | 21 sem · 8.100 km |

La demo usa dev porque el test nunca se muestra. Por eso su 28% al 5% no coincide con el 31% del test: son otros autos.

**La semana de apertura (29-09-2025, al 5%):** 4 alertas nuevas, las cuatro de autos que fallaron después. Dos van con consejo al conductor (VEH\_0563 y VEH\_0566, con tres hábitos cada uno) y dos van al concesionario porque no hay hábito para nombrar (VEH\_0451 y VEH\_0583). Para la ficha conviene abrir VEH\_0563 o VEH\_0566.

**Las piezas:**

| pieza | qué hace |
| --- | --- |
| Bundle | Precalcula todo lo numérico: cortes, puntaje, umbral, alerta, hábitos contra los sanos, señales del filtro y números oficiales. La app no importa torch, LightGBM ni sklearn |
| Política (código) | Determinista, la define Ford (`configs/agents.yaml`). Con hábito: aviso al conductor. Sin hábito: concesionario. Riesgo alto 2 revisiones después del aviso: escalamiento. Una revisión es un corte, cada 500 km |
| Agente redactor | Salida estructurada: mensaje al conductor, resumen para el taller y línea de la bandeja. Las recomendaciones y los chequeos se eligen de listas cerradas, y su texto lo pone el código |
| Agente de triage | Loop con herramientas: eventos de la semana, acción de la política, redactar y contexto de la flota. No puede elegir una acción distinta de la política |
| Verificador (código) | Todo número tiene que estar en los hechos. Rechaza lenguaje causal ("porque", "causa", "debido a"), certezas, promesas de que el riesgo baja, "probabilidad" y "de rutina". Al conductor no le llegan síntomas del filtro (carga, regeneraciones, aceite, consumo). Nada de "se parece a autos que fallaron" ni "el sistema lo marcó por…". Si rechaza, se reintenta hasta 2 veces con la lista de problemas; si no, queda una plantilla aprobada |
| Caché | Cada pedido al LLM se guarda por el hash de lo que lo define: la demo se reproduce sin red. "Regenerar en vivo" llama a la API y no pisa lo guardado |

**Los textos, en números:** en los cuatro puntos de la perilla, los agentes redactaron 515 de 517 textos y los 211 resúmenes semanales, todos aprobados por el verificador, con 106 intentos rechazados en el camino. Las reglas que más rechazan: "para comparar", la atribución ("lo marcó por…") y el lenguaje causal. Los 2 textos en plantilla son del mismo auto (VEH\_0001, semana del 04-08-2025, al 15% y al 20%): no tenía hábitos y el resumen del taller insistió tres veces con una frase prohibida. Modelo: `gpt-5.4-mini-2026-03-17`, \~3,0 M tokens de entrada y 0,24 M de salida para toda la temporada.

**Límites que hay que decir si preguntan:**

- **El porqué es una comparación con la flota sana del mismo mercado, no lo que usó el modelo.** La GRU no tiene atribución por auto.
- **La muestra está enriquecida en fallas** (135 de 426). Los conteos de la bandeja no se trasladan a una flota real.
- **La efectividad de los avisos no está medida:** nada en los datos dice si el conductor cambia de hábitos. Se mide en el piloto.
- **La política y los chequeos del taller son de ejemplo:** los define Ford.
- **El modelo sabe sobre todo qué auto, y poco cuándo** ((a′) = +0,0006).

**Operación de la demo:** URL en Railway con clave (`DEMO_PASSWORD`). Hay que entrar por la raíz, porque arrancar en `/vehiculo` da 404. Es serverless: después de dormir, el primer pedido puede dar 502, así que hay que abrirla unos minutos antes del pitch. Marca: Ford × SOG.

**Preguntas probables del bloque 4:**

| pregunta | respuesta corta |
| --- | --- |
| ¿El LLM puede inventar algo? | No produce números ni diagnósticos. El verificador rechaza cualquier número que no esté en los hechos, y si falla, queda una plantilla |
| ¿Por qué usar un LLM y no plantillas? | Porque escribe cada aviso para ese auto y esa semana, y organiza la bandeja. Las plantillas quedan de red de seguridad |
| ¿Qué pasa si el conductor ignora el aviso? | Si el riesgo sigue alto dos revisiones después, se escala al concesionario |
| ¿El porqué es lo que vio el modelo? | No: es en qué hábitos el auto se aparta de los sanos de su país. Por eso el texto dice "comparado con", nunca "porque" |
| ¿Cuánto cuesta el LLM? | Solo corre sobre las alertas, no sobre la flota: toda la temporada fueron \~3 M tokens |
| ¿Por qué una falsa alarma no es grave? | El primer contacto es un consejo de manejo, no un turno; si el auto no iba a fallar, el consejo igual le sirve |

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

### Guion · Octavio · 3 min

**[Nada nuevo en el auto, \~50 s]** Lo primero que hay que saber es que para llegar a un millón de autos no hay que tocar ni uno. No agregamos sensores ni pedimos datos nuevos. Todo lo que vieron sale de los resúmenes de viaje y los mensajes del filtro que el auto conectado ya le manda a Ford.

El modelo es chico: una red que corre en una computadora común, sin placas de video, en milisegundos por auto. Puntuar un millón de autos es un proceso nocturno de una o dos horas. Y los agentes no leen la flota entera: solo trabajan sobre las alertas. El costo de la inteligencia artificial crece con los avisos, no con los autos.

Y algo que parece un detalle, pero no lo es: el código que calcula los datos del auto en producción va a ser exactamente el mismo con el que medimos todo lo que les mostramos. Así, los números del piloto se pueden comparar con los de hoy.

**[El paso clave: la sombra, \~1 min]** ¿Cómo llega a la calle? En cuatro pasos. Primero, congelamos el modelo, que ya medimos en el test, y lo reentrenamos con los 557 autos. Después, el paso que más nos importa: un **piloto en sombra**. Durante tres a seis meses, el sistema puntúa todos los autos reales de un país, Colombia o Chile, pero no le avisa a nadie. Solo mira.

¿Por qué tanto cuidado? Porque todo lo que medimos salió de las listas que Ford armó para este desafío. En la sombra vamos a ver por primera vez cuántos autos fallan de verdad en la flota, y cuántas falsas alarmas daríamos. Recién con eso, un piloto activo en una región, con un grupo que recibe avisos y otro que no, para medir cuántas fallas se evitan. Y después, el resto de los países.

**[Lo que puede salir mal, \~50 s]** Pensamos en lo que puede salir mal, y en cómo lo vemos venir. Si la cantidad de alertas de un país se dispara, el umbral quedó viejo y se recalibra. Si un dato deja de llegar, nos damos cuenta: ya nos pasó con el contador de regeneraciones, que se apagó para toda la flota. Y cada falla que llega al taller se vuelve una pregunta: ¿la habíamos avisado? ¿con cuánto margen? Esa respuesta alimenta el reentrenamiento, cada tres meses o cuando haya muchas fallas nuevas, siempre con las mismas auditorías.

Y donde no pudimos validar, como en Perú, que no tuvo fallas en los datos, no se enciende.

**[Puente a Gonzalo, \~10 s]** Escala, y no necesita hardware nuevo. Pero la pregunta que importa es otra: ¿cuánto pierde hoy el cliente, y cuánto cuesta evitarlo? Eso se los cuenta Gonzalo.

### Respaldo del bloque 5 (no se dice, se responde)

Las estimaciones de cómputo son propias y hay que validarlas con el equipo de datos de Ford. Fuente: `docs/pitch/guion-presentacion.md` §4–§6, y las fichas de la demo en `feat/demo-gru`.

**Por qué escala, con números.** Supuestos: 1 millón de autos conectados, \~3 viajes por día, batch diario.

| componente | estimación | por qué |
| --- | --- | --- |
| Features + scoring diario | 1–2 h en 8–16 vCPU | \~3 M de filas de viajes por día; la GRU puntúa en milisegundos por auto, en CPU |
| Almacenamiento | \~1 GB/día de viajes, que Ford ya guarda | no se agrega ningún dato que Ford no tenga |
| Reentrenamiento | menos de 1 h de CPU | hoy: 56 corridas en \~25 min en 24 núcleos; las 12 de F11 en \~20 min. Sin GPU |
| Agentes (LLM) | proporcional a las alertas | toda la temporada de la demo (426 autos, cuatro puntos de la perilla) fueron \~3,0 M tokens de entrada y 0,24 M de salida |

El costo en dólares está en el bloque 6. **El costo real no es la computadora:** es integrar el sistema con Ford (ingesta, app, concesionarios) y el costo de cada alerta.

**Arquitectura, de punta a punta:**

1. **Ingesta diaria** de TripSummary + señales, desde el data lake de Ford.
2. **Features por auto** con el mismo código del entrenamiento (`src/features`, `src/data/panel.py`): un corte cada 500 km, ventana de 1.000 km, solo hacia atrás. Si producción reimplementara las features, aparecería el *training-serving skew* y los números del piloto dejarían de ser los medidos.
3. **Scoring** con el modelo empaquetado: 3 semillas de la GRU; cada una se pasa a percentil contra la referencia de los sanos y se promedian (el ensamble por rango).
4. **Regla de alerta:** 2 cortes seguidos sobre el umbral del presupuesto elegido por Ford.
5. **Política + agentes + verificador** → bandeja de posventa, concesionario, app del conductor.
6. **Retorno del taller:** ¿hubo intervención? Alimenta el monitoreo y el reentrenamiento.

**El modelo empaquetado.** Hoy `train.py` guarda predicciones, no modelos: **falta escribir `scripts/export_model.py`** (no toca nada de lo medido). El paquete:

| archivo | qué tiene |
| --- | --- |
| `manifest.json` | commit, YAML completo, hash de las features, orden de columnas, W/G/H/Δ, k de la alerta, semillas, versiones de librerías, lista de autos de entrenamiento, métricas de dev y test |
| `preprocess.joblib` | imputación y escalado, ajustados con el train final |
| `model/` | pesos de cada semilla (`state_dict`) + export ONNX para servir sin torch |
| `score_reference.parquet` | distribución del score de los sanos por semilla: sin ella, el ensamble por rango no funciona en producción |
| `thresholds.json` | umbral por presupuesto de falsas alarmas (fijado fuera de muestra) y el elegido |
| `golden.parquet` | 20 autos con su score esperado: al cargar se re-puntúan y tienen que dar igual (a 1e-6) |

- **Nada de pickle del objeto entero:** se rompe al cambiar la versión de una librería.
- Se versiona como wandb Artifact (`oromagnoli-/ford-fic`) con alias candidato → sombra → producción.
- **El número del test sale del modelo entrenado con dev; el que se despliega se reentrena con los 557 autos** y la misma configuración.

**Las etapas:**

| etapa | qué | cuánto | criterio para pasar |
| --- | --- | --- | --- |
| 0 · Congelar | test final, reentrenar con 557 autos, empaquetar | 1–2 semanas | test dentro del rango de dev |
| 1 · Piloto en sombra | puntuar la flota real de COL o CHL (los que más eventos tienen) sin avisar a nadie | 3–6 meses | tasa de alertas ≈ presupuesto elegido; detección realizada ≥ celda mercado × motor |
| 2 · Piloto activo | avisos en una región, con grupo control sin aviso | 6 meses | fallas evitadas contra el control; costo por alerta; cambio de hábitos |
| 3 · Escala | todos los mercados validados, umbral recalibrado por mercado | continuo | monitoreo en verde |

**Por qué la sombra es el paso clave:** las listas de Ford vienen muestreadas en dos cohortes, así que no conocemos la prevalencia real de la flota. La sombra es la primera medición de la tasa real de fallas por mercado × motor y de las falsas alarmas sobre la flota completa. También responde lo que quedó abierto: qué parte de lo que el modelo aprende es física y qué parte es cómo se armó la lista.

**Monitoreo:**

- Tasa de alertas por mercado contra el presupuesto: si se dispara, el umbral quedó viejo.
- Deriva de las features por mercado y mes, contra la referencia del entrenamiento. La temperatura ambiente y la estación son las primeras sospechosas.
- Detección realizada: cada falla en taller, ¿tuvo alerta antes? ¿con cuánto margen?
- Datos que dejan de llegar: columnas nulas o cortadas por mercado. Ya pasó con el marcador de regeneraciones (25-05-2026, toda la flota): un corte así cambia el score sin que el auto cambie.
- Textos: tasa de rechazos del verificador y de plantillas por semana.

**Reentrenamiento:** trimestral, o cuando haya +50% de eventos nuevos (el límite hoy es la cantidad de eventos, no el modelo). Cada modelo nuevo pasa las mismas auditorías (`audit_model.py`) y el mismo reporte (`report_v2_models.py`), y se compara en sombra contra el vigente antes de reemplazarlo.

**Riesgos, completo:**

| riesgo | mitigación |
| --- | --- |
| Prevalencia real distinta a la del estudio | el umbral se recalibra en la sombra, no se hereda de dev |
| Fatiga de alertas | empezar al 5%; avisar solo con hábito explicable; escalar solo si el riesgo sigue |
| Cambio de firmware o de telemetría | monitoreo de columnas nulas o cortadas por mercado |
| Ciudad de venta ≠ ciudad de uso | la ciudad no entra como feature hasta tener la ubicación de uso |
| Privacidad | solo agregados por viaje que Ford ya procesa, sin ubicación |
| Mercados sin eventos (PER) | no se activa donde no se validó |
| Motores o modelos nuevos | no se activan hasta tener eventos propios; entran por la celda |
| El LLM cambia de versión | la versión va fijada; los textos pasan por el verificador igual; si falla, plantilla |

**Preguntas probables del bloque 5:**

| pregunta | respuesta corta |
| --- | --- |
| ¿Necesitan GPU? | No, ni para servir ni para reentrenar |
| ¿Qué datos nuevos le piden a Ford? | Ninguno para operar. Para mejorar: más eventos, la fecha de la intervención y la ubicación de uso (bloque 7) |
| ¿Por qué no arrancar directo con avisos? | Porque la prevalencia real y las falsas alarmas reales solo se ven en la flota completa; la sombra no le cuesta nada al cliente |
| ¿Por qué Colombia o Chile? | Tienen más eventos (54 y 36 en dev), así que la sombra mide algo en pocos meses |
| ¿Cada cuánto se reentrena? | Trimestral o con +50% de eventos, siempre con las mismas auditorías y comparado en sombra |
| ¿Qué pasa con un auto nuevo, sin historia? | Necesita 1.000 km para el primer puntaje; antes, solo la tasa de su celda |

## 6 · Costos: lo que pierde el cliente y lo que cuesta el producto

El costo principal no es la reparación. Es lo que el cliente pierde cuando la falla lo toma por sorpresa: una visita que no planeaba, días de uso y confianza. Evitarlo cuesta \~USD 0,10 por auto y por año.

**Slide 28 · Lo que paga el cliente hoy, sin saberlo.** La degradación es silenciosa durante meses:

- **Probablemente más gasoil.** El hollín acumulado sube la contrapresión del escape, y cada regeneración activa quema combustible extra. Con estos datos no lo pudimos medir (el consumo no separa fallados de sanos), así que no va en el pitch.
- **Menos rendimiento.** Regeneraciones interrumpidas y, al final, el modo de protección con potencia limitada.
- **Un testigo sin explicación** en el tablero y una visita al taller que no estaba en los planes.
- **La sensación de "nadie me avisó"**, que es lo que termina pesando en la relación con la marca.

**Slide 29 · Antes y después, desde el cliente.** Visual: dos líneas de tiempo paralelas.

| momento | hoy | con el producto |
| --- | --- | --- |
| \~3,5 meses antes | nada: el filtro se carga y no se nota | aviso en la app con un hábito concreto ("un tramo de ruta de 20 minutos por semana") |
| mientras se degrada | el filtro sigue cargándose | corrige el hábito y el filtro se limpia en uso normal |
| si el riesgo sigue | testigo, modo de protección, visita al taller sin planificar | el concesionario lo llama y le ofrece un turno |
| la relación | reclamo | "mi Ford me avisó a tiempo" |

**Slide 30 · Por qué una falsa alarma casi no le cuesta al cliente.** La primera acción es un consejo de manejo, no una visita. Si el auto no iba a fallar, el cliente recibió un consejo que igual le ahorra combustible. Por eso se puede operar con un presupuesto de falsas alarmas más alto que si cada alerta fuera un turno.

**Slide 31 · Lo que cuesta el producto.** Estimación propia para 1 millón de autos conectados, USD por año, a validar con Ford:

| componente | USD / año | supuesto |
| --- | --- | --- |
| Features + scoring diario | 2.500–6.000 | batch de 1–2 h en 8–16 vCPU |
| Agentes (LLM) | 1.000–3.000 | \~65.000 alertas/año × \~6 llamadas con un modelo chico; precio aproximado |
| Reentrenamiento + monitoreo | 5.000–10.000 | trimestral, en CPU |
| Equipo de operación | 60.000–80.000 | 1 persona ML/MLOps |
| **Total** | **\~70.000–100.000** | **\~USD 0,07–0,10 por auto y por año** |

Implementación, una vez: \~USD 70.000–110.000 (tres personas durante 6 meses, incluye el piloto en sombra). Almacenamiento adicional: casi nada, los viajes ya los guarda Ford.

**Slide 32 · Qué se mide en el piloto, del lado del cliente.**

- Consumo antes y después del aviso, en avisados contra un grupo control.
- Cuántos conductores cambian el hábito recomendado.
- Satisfacción (NPS) y retención en la red de servicio de los avisados.
- Visitas no planificadas y modos de protección evitados.

**Para Ford, en segundo plano:** además evita parte del costo de garantía. Los escenarios en USD de `f8-costos-k2.md` quedan en el backup, no en el cuerpo.

**Límite que hay que decir:** con estos datos no podemos medir bien la pérdida de consumo. El nivel de combustible solo permite estimarlo en el 33% de los viajes y no separa fallados de sanos. Por eso es la primera métrica del piloto.

Fuentes de la penalidad de consumo, por si se verifica más adelante (resultados de búsqueda, no leídos completos; no se citan en el pitch): [ScienceDirect, filtro cargado de hollín y regeneración](https://www.sciencedirect.com/science/article/abs/pii/S0959652624020997) · [USPTO, regeneración de filtros de partículas](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/10378400) · [DieselNet, ceniza en DPF](https://dieselnet.com/tech/dpf_ash.php).

### Guion · Gonzalo · 4 min

**El eje del bloque (decidido el 29-09):** el costo real no es la reparación, que no conocemos. Es la fricción con el cliente y el riesgo de perderlo. Por eso el bloque pone en la balanza el costo de los avisos contra la confianza del cliente. Que además ayuda a prevenir averías va después, como ganancia extra. **No se cita ningún costo de falla en dólares.**

**[El costo que no está en la factura, \~40 s]** Volvamos a nuestro conductor. ¿Cuánto le costó a Ford su falla? La reparación tiene un precio, y Ford lo conoce mejor que nosotros. Pero hay un costo que no aparece en ninguna factura: lo que dijo al retirar su camioneta del taller. *"Nadie me avisó."*

Ese es el costo que nos importa. Un cliente que se enteró tarde se pregunta si su Ford lo cuida, y eso no aparece en ninguna factura. Cada falla sin aviso se paga en confianza.

**[Avisar es barato, \~40 s]** La buena noticia es que avisar cuesta muy poco. Con la telemetría que Ford ya tiene, operar esto para un millón de autos cuesta alrededor de **10 centavos de dólar por auto y por año**: las computadoras, los agentes que escriben los mensajes y una persona que lo opera.

Entonces la pregunta no es cuánto cuesta avisar. Es otra: **¿cuántos avisos soporta la confianza de un cliente?**

**[La balanza, \~1 min 20 s]** Porque un aviso también puede gastar confianza. Si le avisamos a un conductor y su auto no iba a fallar, eso es fricción. Y si mandáramos el mismo consejo a toda la flota, nadie lo leería, y el día que importa tampoco.

En un plato de la balanza está el aviso de menos: la falla sin aviso, el "nadie me avisó". En el otro, el aviso de más: un mensaje que no hacía falta.

Diseñamos todo para que el aviso de más pese lo menos posible. Casi siempre, el primer contacto no es "lleve su auto al taller". Es un consejo: *"un tramo de ruta de 20 minutos por semana"*. Si el auto no iba a fallar, el conductor recibió un consejo que igual le hace bien a su motor. Si no hay un hábito para nombrar, o el riesgo sigue, la llama el concesionario. En la temporada que vieron en la demo, casi todas las alertas fueron consejos, no turnos.

Y el que decide cuánto pesa cada plato es Ford, con la perilla. Al 5%, uno de cada veinte autos sanos recibe un consejo de más, y avisamos a casi un tercio de las fallas. Al 20%, avisamos a seis de cada diez [test], con cuatro veces más consejos de más. No hay un número correcto: hay un punto que Ford elige según cuánto confía en sus clientes y cuánto le duele una falla.

**[Y encima, una chance de esquivar la avería, \~40 s]** Todo esto sin contar lo que ya les conté: un filtro que se tapa termina en el modo de protección y en una visita que nadie planificó. Cada aviso a tiempo es una oportunidad de evitarlo: un hábito que cambia, o un turno programado en vez de una urgencia. Cuántas esquiva, lo va a decir el piloto. Para una flota de trabajo de Ford Pro, cada día que el vehículo no para vale todavía más.

**[Lo que vamos a medir, \~30 s]** No les pedimos que nos crean. En el piloto medimos exactamente esto, con un grupo que recibe avisos y otro que no: si los clientes avisados confían más en Ford, si cambian el hábito, si vuelven al concesionario, y cuántas fallas se evitan.

**[Puente a Octavio, \~10 s]** Un producto barato, que cuida la confianza del cliente y encima le da una chance de esquivar la avería. ¿Qué le pedimos a Ford para empezar? Eso se los cuenta Octavio.

### Respaldo del bloque 6 (no se dice, se responde)

**Qué citamos y qué no:**

- **Sí:** lo que cuesta operar el producto (estimación propia, abajo) y la fricción medida en el replay (cuántos avisos de más, cuántos son consejos y cuántos turnos).
- **No:** lo que le cuesta a Ford una falla. No lo conocemos. Los escenarios de `f8-costos-k2.md` usan precios públicos de EE. UU. (USD 800 a 15.000 por falla) y el modelo K2 de la entrega 1. Solo se mencionan si preguntan, con esas dos aclaraciones.
- **No, hasta verificarla:** la penalidad de consumo del 4–7%. Sus fuentes (abajo) son resultados de búsqueda que no leímos completos. En el guion, "gasta más combustible" va sin número.

**Lo que cuesta el producto.** Estimación propia para 1 millón de autos conectados, USD por año, a validar con Ford:

| componente | USD / año | supuesto |
| --- | --- | --- |
| Features + scoring diario | 2.500–6.000 | batch de 1–2 h en 8–16 vCPU |
| Agentes (LLM) | 1.000–3.000 | \~65.000 alertas por año × \~6 llamadas con un modelo chico; precio aproximado |
| Reentrenamiento + monitoreo | 5.000–10.000 | trimestral, en CPU |
| Equipo de operación | 60.000–80.000 | 1 persona de ML/MLOps |
| **Total** | **\~70.000–100.000** | **\~USD 0,07–0,10 por auto y por año** |

Implementación, una sola vez: \~USD 70.000–110.000 (tres personas durante 6 meses, incluye el piloto en sombra). El 80% del costo anual es la persona que lo opera, no la computadora: agregar autos casi no lo mueve.

**La fricción en el replay** (dev, una repetición; 135 autos que fallan y 291 sanos):

| punto de la perilla | 5% | 10% | 15% | 20% |
| --- | --- | --- | --- | --- |
| sanos con un aviso de más (de 291) | 14 | 29 | 43 | 58 |
| fallas avisadas antes (de 135) | 39 | 56 | 69 | 82 |
| alertas nuevas | 53 | 85 | 112 | 140 |
| de esas, consejos al conductor (con hábito) | 48 (91%) | 73 (86%) | 101 (90%) | 125 (89%) |
| escalamientos al concesionario | 7 | 10 | 22 | 36 |

La fricción es mínima donde más importa: al 5%, 91 de cada 100 primeros contactos son un consejo, no un turno.

**Ojo: en una flota real, la mayoría de los avisos van a autos que no iban a fallar.** La muestra de Ford está enriquecida (135 de 426 fallan). En la flota, la prevalencia es mucho más baja y no la conocemos. Cuenta ilustrativa por cada 1.000 autos, con la detección del test (31% al 5%, 62% al 20%):

| prevalencia supuesta | perilla | fallas avisadas | consejos de más | de cada 10 avisos, cuántos son de un auto que iba a fallar |
| --- | --- | --- | --- | --- |
| 2% | 5% | 6 de 20 | 49 | \~1 |
| 5% | 5% | 16 de 50 | 48 | \~2 |
| 10% | 5% | 31 de 100 | 45 | \~4 |
| 5% | 20% | 31 de 50 | 190 | \~1 |

Por eso el primer contacto tiene que ser un consejo que le sirva a cualquiera, y por eso el umbral se recalibra en la sombra con la prevalencia real. Es el argumento a favor del diseño, no en contra: si el aviso de más fuera un turno en el taller, la fricción sería inaceptable.

**La trampa: "si el consejo es casi gratis, ¿por qué no mandárselo a todos?"** Nuestro propio análisis de costos dice que, si el aviso cuesta \~0, alertar a todos le gana a cualquier modelo. La respuesta:

1. El aviso no es gratis: cuesta confianza. Un consejo genérico a toda la flota se ignora, y cada aviso de más gasta la credibilidad del siguiente.
2. Lo que vale es la personalización. *"Tu auto hace muchos viajes cortos, comparado con los sanos de tu zona"* solo se puede decir si es cierto para ese auto. Un mensaje a todos no puede decirlo.
3. El segundo paso sí cuesta: que el concesionario llame y ofrezca un turno. Ahí el modelo decide a quién llamar.

**Lo que se mide en el piloto (grupo avisado contra grupo control):**

- **Confianza:** satisfacción (NPS) y reclamos de los avisados contra el control.
- **Fricción:** tasa de apertura de los avisos, bajas de notificaciones y quejas por aviso.
- **Retención:** vuelta a la red de servicio de Ford.
- **Hábito:** cuántos conductores cambian el hábito recomendado (se ve en la misma telemetría).
- **Averías:** modos de protección y visitas no planificadas evitadas; consumo antes y después del aviso.
- **Costo:** costo por alerta y por falla evitada, ya con los costos reales de Ford.

**El consumo, lo que sabemos:** con estos datos no se puede medir bien. El nivel de combustible solo permite estimarlo en el 33% de los viajes y no separa fallados de sanos. Por eso es una métrica del piloto y no un número del pitch.

**Preguntas probables del bloque 6:**

| pregunta | respuesta corta |
| --- | --- |
| ¿Cuánto ahorra Ford? | Depende del costo de una falla y de la prevalencia real, que no tenemos. El piloto lo mide con sus costos. Lo que sí sabemos es que operarlo cuesta \~10 centavos por auto y por año |
| ¿No molesta recibir avisos? | Por eso el primero es un consejo, no un turno, y Ford elige cuántos tolera. Al 5%, 91 de cada 100 primeros contactos son un consejo |
| ¿Por qué no mandarle el consejo a todos? | Un aviso genérico se ignora; el nuestro dice algo cierto de ese auto. Y el llamado del concesionario sí cuesta |
| ¿Cuánto combustible se ahorra? | No lo podemos medir con estos datos; es la primera métrica del piloto |
| ¿Por qué tan barato? | Usa datos que Ford ya tiene, corre en CPU, y los agentes solo trabajan sobre las alertas. El 80% del costo es la persona que lo opera |
| ¿Y si la falla es barata? | Entonces el valor está casi todo en la confianza, no en la garantía. Por eso lo medimos así en el piloto |

## 7 · Cierre: valor diferencial, trabajo futuro y conclusiones

Dos slides, las dos de Octavio. La primera dice por qué es distinto y qué solo Ford tiene; la segunda es la marca.

**Slide 33 · Por qué es distinto.** *"Del testigo en el tablero al aviso a tiempo."*

1. **Anticipa de verdad.** Nunca ve los últimos 500 km y aun así avisa con meses de margen.
2. **No exagera.** Cada número, al lado del azar, y medido en autos que nunca vio.
3. **Habla como una persona, sin inventar.** La IA redacta; el código decide y verifica.
4. **No toca ni un auto.** La telemetría que Ford ya recibe. El dial lo gira Ford.

Abajo, **para que mejore más rápido · lo que solo Ford tiene:**

- **Más fallas registradas** (hoy son 177).
- **La fecha real de la intervención** (hoy, −21 días a ojo).
- **Dónde se usa el auto**, no dónde se vendió.
- **La tasa real por país × motor**, para separar la física de cómo se armó la lista.

**Slide 34 · Final.** *"Tu Ford te avisa antes."* Gracias · Equipo SOG.

**Límites que decimos antes de que los pregunten:** no predecimos la fecha exacta, la prevalencia real de la flota se mide en el piloto, y dentro de un mismo mercado y motor el orden entre autos es moderado (AUC 0,68 en test).

**Preguntas probables del jurado (backup):**

| pregunta | respuesta corta |
| --- | --- |
| ¿No aprende solo qué mercado falla más? | En parte: el país pesa. Pero la duración de los viajes y el hollín pesan casi lo mismo, y dentro de un mismo país y motor ordena autos con AUC 0,68 [test] |
| ¿Cómo sé que no es leakage? | Gap de 500 km, features hacia atrás, split por auto; con features permutadas cae al azar |
| ¿Por qué no accuracy? | Con \~6% de positivos, decir "nadie falla" da 94% |
| ¿Por qué una GRU y no algo más simple? | Probamos tabulares, supervivencia y series de tiempo; en el test detectan la mitad que la GRU |
| ¿El LLM puede inventar algo? | No produce números ni diagnósticos; el verificador rechaza cualquier número que no esté en los hechos |
| ¿Qué pasa si el conductor ignora el aviso? | Se escala al concesionario si el riesgo sigue alto 2 revisiones después |

### Guion · Octavio · 2 min

**[Por qué es distinto, \~1 min 45 s]** Para cerrar, les cuento por qué creemos que esto es distinto.

Primero, **anticipa de verdad.** El modelo nunca ve los últimos 500 km antes de la falla, y aun así avisa con meses de margen.

Segundo, **no exageramos.** Cada número está al lado del azar, y lo medimos en autos que el modelo nunca vio. Les contamos qué descartamos, qué falló y dónde no conviene usarlo.

Tercero, **habla como una persona,** y un verificador en código controla que no invente.

Y cuarto, **no hay que tocar ni un auto,** y el que decide cuántos avisos manda es Ford.

Y si quieren que esto mejore más rápido, hay cuatro cosas que solo Ford tiene.

**[Final, \~10 s]** *(Pantalla final.)* **Tu Ford te avisa antes.** Gracias.

### Respaldo del bloque 7 (no se dice, se responde)

**Si preguntan por los números, cómo se dicen para que coincidan con el resto:**

- **"Casi uno de cada tres, con 5% de falsas alarmas"** = 31,2% [test], 10 de 32.
- **"Unos tres meses antes"** vale en los dos puntos: en test, al 10%, la mediana es de \~4.600 km (\~100 días) y en la demo, al 5%, de 15 semanas.
- **"Unos centavos por auto y por año"** = USD 0,07–0,10 (bloque 6).
- **La sombra:** ya no está en el cierre, pero sí en el plan (bloque 5), con 3–6 meses. Si preguntan: tres meses alcanzan para medir la tasa de alertas y las falsas alarmas en COL o CHL, y seis para contar fallas con precisión.

**Qué destraba cada pedido a Ford:**

| pedido | qué destraba |
| --- | --- |
| Piloto en sombra de 3 meses en un país | la prevalencia real, las falsas alarmas reales y la detección realizada, sobre la flota completa y no sobre las listas |
| Más fallas registradas | hoy el límite son 177 eventos: la semilla mueve más que los hiperparámetros, y más eventos bajan la varianza |
| La fecha real de la intervención | hoy la corremos 21 días a ojo, porque la fecha registrada llega \~2 semanas después del taller |
| Dónde se usa el auto | la altura y el clima van en la dirección de la física, pero con la ciudad de venta no se separan del cero |
| La tasa real de falla por país × motor | separa la física de cómo se armaron las listas: es la duda más grande de todo el proyecto |
| Los costos reales (reparación, inspección, retención) | convierte la perilla en una decisión con números de Ford |

**Trabajo futuro, si preguntan:** extender el mismo circuito a otras fallas del postratamiento (el costo de la segunda falla cubierta es marginal), reentrenar con los 557 autos, escribir el empaquetado del modelo (`export_model.py`) y medir en el piloto si los avisos cambian hábitos.

**Lo que no prometemos** (decirlo antes de que lo pregunten, si hay tiempo):

- No predecimos la fecha exacta de la falla: la marcamos con margen.
- No conocemos la prevalencia real de la flota: se mide en la sombra.
- Dentro de un mismo país y motor, el orden entre autos es moderado (AUC 0,68 en test).
- La efectividad de los avisos no está medida.
- Con 32 fallas en el test, cada auto mueve 3 puntos: por eso se citan rangos.

**Pendientes antes del 02-10:**

- [x] Medir el test y reemplazar los números de dev (01-10: el finalista es la GRU de F10).
- [x] Regenerar la demo con la GRU finalista (`demo-bundle-gru-final`).
- [ ] Correr los escenarios de costo con el finalista.
- [ ] Elegir el nombre del producto.
- [ ] Diseño visual del HTML.
