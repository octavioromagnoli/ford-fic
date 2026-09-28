# Ford × SOG · presentación de la demo

```sh
source .venv/bin/activate
DEMO_LLM_MODE=cache_only streamlit run scripts/demo_app/app.py
```

Abrir la raíz de la app y navegar con la barra lateral. «Cómo usar», arriba a la derecha, recorre la pantalla abierta paso a paso. El modo `cache_only` permite ensayar el pitch con los textos del bundle sin llamadas nuevas al modelo.

## Diseño

- `tokens.css`: el sistema, compartido por la app y el iframe de los contadores.
  - Superficies: un panel, el panel del agente, un inset y los controles.
  - Dos radios: 14 px para los paneles y 10 px para lo que va adentro.
  - Una escala tipográfica: 11 / 13 / 15 / 18 / 22 / 32.

  Toda tarjeta, métrica, tabla, gráfico y expander usa esas superficies.
- `assets/ford-script.svg`: el script de Ford, extraído del «Ford logo flat.svg» de Wikimedia Commons (el archivo es de dominio público; la marca es de Ford). `presentation.brand_html` arma la marca «Ford × SOG»: el script, un × dibujado en el azul eléctrico y «SOG» (el equipo, `brand.team` en `configs/demo.yaml`) en Manrope 800 espaciada, apoyada en la línea de base del script. Los tamaños de cada lugar (barra lateral, celular, login) son variables `--brand-*` en `theme.css`. `brand_block` le agrega qué hace y de dónde viene (barra lateral y login).
- **Arriba de cada página**, una barra con la etiqueta del replay y «Cómo usar», debajo la **perilla de falsas alarmas** y después el **calendario de la flota** (no va en «Qué pasó después», que muestra la temporada entera). El calendario y el recorrido son componentes de `st.components.v2`: el JS corre en el documento de la app, sin iframe, y los estilos viven en `theme.css`.
  - La perilla (`common.budget_knob`, un `st.segmented_control`) elige el punto de operación: 5, 10, 15 o 20% de los autos sanos con una alerta de más. Cada punto trae en el bundle su umbral exacto, sus alertas, sus hábitos, sus mensajes y su triage, así que mover la perilla cambia el calendario, la bandeja, la ficha y la temporada. Al lado dice cuánto se anticipa en ese punto (el número oficial). La semana elegida se queda; las decisiones de la bandeja se guardan por punto. En «Qué pasó después», la tabla oficial muestra los cuatro puntos, con el elegido marcado.
  - `timeline.js`: una columna por semana del replay, con una barra por las alertas nuevas (ámbar) y los escalamientos (rojo) de esa semana; el segmento de abajo es el avance del replay y la columna con borde es la semana abierta. Elegir una columna, una flecha o «Próxima con alertas» manda el lunes a Python como el trigger `week` (`app.py::_from_timeline`). Los datos salen de `common.timeline_data`.
  - `tour.js` y `guide.py`: el recorrido. `guide.py` escribe los pasos de la página abierta (todo número sale del bundle), y cada paso nombra selectores en orden de preferencia: gana el primero visible, así el mismo paso apunta a la barra lateral en una notebook y al dock en un celular. Un paso sin blanco en la página se saltea. Oscurece la app, rodea la parte que explica y se maneja con los botones, las flechas del teclado y Esc; en un celular la tarjeta es una hoja arriba o abajo, del lado con más lugar. Corre en el navegador, sin rerun. El botón brilla siempre (un halo que respira y un reflejo que lo cruza cada 5 s), y hasta el primer recorrido suma un anillo que late (se recuerda en `localStorage`). La tarjeta de bienvenida del recorrido («Así se usa la demo de SOG») brilla igual. Con movimiento reducido queda solo el halo, quieto.
- **Colores de los badges, uno por significado:**
  - ámbar o rojo: el tipo de evento;
  - azul: la acción de la política;
  - verde: lo que pasó un control (el verificador o la aprobación);
  - gris: lo neutro.
- `theme.css`: superficies, navegación, tarjetas, foco y animaciones. Respeta `prefers-reduced-motion`.
- `presentation.py`: estilos, encabezados, mensajes literales, contadores y tema de gráficos.
- `.streamlit/config.toml`: tema nativo de controles y tablas; el Dockerfile lo incluye.
- **Celular y ventanas angostas.** A 768 px o menos, Streamlit colapsa la barra lateral, y con el header oculto no hay forma de abrirla. Por eso `app.py` repite sus controles en un modo compacto, que `theme.css` muestra solo mientras la barra está colapsada:
  - un dock abajo, con la semana (anterior, siguiente, «Próxima») y las tres páginas; el calendario de arriba esconde sus botones, que ya están en el dock, y queda como gráfico para tocar;
  - las notas de la muestra, al final.

  Las grillas de métricas se reordenan con container queries, porque con la barra abierta en una tablet quedan tan angostas como en un celular. Las tablas son HTML estático (`presentation.table`) y no `st.dataframe`, que en un celular obliga a desplazarse de costado.

Los contadores animan únicamente enteros y terminan en la cadena exacta recibida. Los mensajes se escapan como HTML y preservan sus saltos de línea. Las operaciones del bundle, los agentes, el verificador y los formatos numéricos existentes no se modifican.

## Revisión visual

**Demo con la GRU (28-09).** Chrome headless manejado por Playwright, a 1440×900 y a 390×844 táctil. Espera a que Streamlit termine de dibujar por websocket (una captura directa de Chrome headless se queda en la pantalla de carga) y recorre el contenedor que scrollea, que no es el documento. Revisa:
- las tres páginas y el calendario: «Próxima», las flechas, una columna y el dock;
- las cuatro tarjetas de la semana de apertura con sus tres pestañas;
- la ficha de VEH_0563, abierta desde «Ver ficha»;
- la línea de tiempo, la tabla oficial y «Qué no promete»;
- el recorrido «Cómo usar»;
- en cada página, la consola, los requests fallidos y el desborde horizontal.

Qué se corrigió y por qué está en `docs/memoria/f9-demo-gru.md` («Revisión visual y bundle v1»). El contenedor se probó igual, con la clave de la demo.

**Demo anterior, con K2.** Capturas en Chrome aislado a 1440×900, 1920×1080, 390×844, 844×390 y 820×1180 en `experiments/demo-ui-review/` (no versionado), con el recorrido completo en el celular.

## Límites de Streamlit observados

- `st.switch_page` (el botón «Ver ficha») descarta el estado de los widgets, y la ficha abría en la primera semana. Por eso la semana vive en `_week` (`common.py`), una clave que no pertenece a ningún widget. El calendario y el dock solo la reflejan y la actualizan con sus callbacks. El punto de operación hace lo mismo con `_budget`: la perilla lo refleja y lo cambia con su callback, y `required=True` evita que un segundo toque la deje sin valor.
- La navegación normal desde la raíz no produce errores de consola. Al arrancar directamente en `/vehiculo` o `/resultados`, Streamlit 1.63 prueba primero `/<página>/_stcore/health` y `host-config`: devuelve dos 404 y luego conecta correctamente con la raíz. Para el pitch, abrir la raíz.
- axe detecta `aria-expanded` en el `<section>` del sidebar nativo, cuyo rol no admite ese atributo. No se manipula el DOM interno del framework para ocultar el hallazgo. No detectó infracciones de contraste WCAG AA en las pantallas revisadas.
- `st.components.v1.html`, solicitado para los contadores, funciona en la versión fijada, pero Streamlit avisa de su deprecación en el log del servidor. Revisar la migración a `st.iframe` antes de actualizar Streamlit. Su iframe deja además, en la consola del navegador, nueve avisos de Chrome sobre los permisos y el sandbox que le pone Streamlit.
- Los gráficos con fechas dejan en la consola avisos de Vega («Infinite extent for field…»). No son errores: Streamlit crea la vista vacía y le inserta los datos después, para poder agregar filas. Un dominio de fechas fijo no los evita, porque Vega-Lite lo compila como una señal que también se evalúa tarde.
- Con `autosize: fit` (lo que pone `presentation.chart_style` por defecto), el alto del gráfico incluye título, ejes y leyenda. En los gráficos de una fila por ítem (el porqué, la línea de tiempo) va `fit="fit-x"`: con `fit`, una leyenda o un título de eje largos achican las filas.
- Manrope se carga desde Google Fonts, con Arial como alternativa si no hay conexión.
