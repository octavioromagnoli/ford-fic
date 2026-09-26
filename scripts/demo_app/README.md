# Ford DPF · presentación de la demo

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
- `assets/ford-script.svg`: el script de Ford, extraído del «Ford logo flat.svg» de Wikimedia Commons (el archivo es de dominio público; la marca es de Ford). `presentation.brand_html` arma la marca «Ford DPF»: el script, un filete vertical y «DPF» en Manrope, centrados sobre un eje. `brand_block` le agrega qué es y de dónde viene (barra lateral y login).
- **Arriba de cada página**, una barra con la etiqueta del replay y «Cómo usar», y debajo el **calendario de la flota** (no va en «Qué pasó después», que muestra la temporada entera). Los dos son componentes de `st.components.v2`: el JS corre en el documento de la app, sin iframe, y los estilos viven en `theme.css`.
  - `timeline.js`: una columna por semana del replay, con una barra por las alertas nuevas (ámbar) y los escalamientos (rojo) de esa semana; el segmento de abajo es el avance del replay y la columna con borde es la semana abierta. Elegir una columna, una flecha o «Próxima con alertas» manda el lunes a Python como el trigger `week` (`app.py::_from_timeline`). Los datos salen de `common.timeline_data`.
  - `tour.js` y `guide.py`: el recorrido. `guide.py` escribe los pasos de la página abierta (todo número sale del bundle), y cada paso nombra selectores en orden de preferencia: gana el primero visible, así el mismo paso apunta a la barra lateral en una notebook y al dock en un celular. Un paso sin blanco en la página se saltea. Oscurece la app, rodea la parte que explica y se maneja con los botones, las flechas del teclado y Esc; en un celular la tarjeta es una hoja arriba o abajo, del lado con más lugar. Corre en el navegador, sin rerun. El botón brilla siempre (un halo que respira y un reflejo que lo cruza cada 5 s), y hasta el primer recorrido suma un anillo que late (se recuerda en `localStorage`). La tarjeta de bienvenida del recorrido («Así se usa Ford DPF») brilla igual. Con movimiento reducido queda solo el halo, quieto.
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

Capturas de las tres pantallas y sus detalles en Chrome aislado, a 1440×900 y 1920×1080, en `experiments/demo-ui-review/` (no versionado). El informe `review.json` registra consola, accesibilidad e interacciones.

Después se agregaron capturas a 390×844 (celular), 844×390 (celular apaisado) y 820×1180 (tablet), con dos controles:
- que ningún elemento se salga del ancho y ninguna métrica quede truncada;
- que los controles del dock midan al menos 44 px.

En el celular se probó además el recorrido completo: navegar desde el dock, saltar a la próxima semana con alertas y revelar el desenlace.

Se revisaron la navegación desde la Bandeja, el cambio de semana, la selección de vehículo, el revelado del desenlace, el movimiento reducido, los valores finales de los contadores y la igualdad literal de los mensajes con la caché del bundle.

## Límites de Streamlit observados

- `st.switch_page` (el botón «Ver ficha») descarta el estado de los widgets, y la ficha abría en la primera semana. Por eso la semana vive en `_week` (`common.py`), una clave que no pertenece a ningún widget. El calendario y el dock solo la reflejan y la actualizan con sus callbacks.
- La navegación normal desde la raíz no produce errores de consola. Al arrancar directamente en `/vehiculo` o `/resultados`, Streamlit 1.63 prueba primero `/<página>/_stcore/health` y `host-config`: devuelve dos 404 y luego conecta correctamente con la raíz. Para el pitch, abrir la raíz.
- axe detecta `aria-expanded` en el `<section>` del sidebar nativo, cuyo rol no admite ese atributo. No se manipula el DOM interno del framework para ocultar el hallazgo. No detectó infracciones de contraste WCAG AA en las pantallas revisadas.
- `st.components.v1.html`, solicitado para los contadores, funciona en la versión fijada, pero Streamlit avisa de su deprecación en el log del servidor. Revisar la migración a `st.iframe` antes de actualizar Streamlit.
- Manrope se carga desde Google Fonts, con Arial como alternativa si no hay conexión.
