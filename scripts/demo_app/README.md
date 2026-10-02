# Demo GRU · Ford × SOG

Aplicación de Streamlit que reproduce la flota de desarrollo y muestra el
circuito de alertas de posventa: bandeja semanal, ficha del vehículo y resultados
del período. El modelo es la GRU con atención y ensamble de tres semillas.

## Abrir la aplicación

Desde la raíz del repositorio, con Python 3.12 y `.venv/` activo:

```bash
python -m pip install -r requirements-demo.txt
DEMO_LLM_MODE=cache_only streamlit run scripts/demo_app/app.py
```

En PowerShell:

```powershell
$env:DEMO_LLM_MODE = "cache_only"
streamlit run scripts/demo_app/app.py
```

Antes de iniciar, copiar el bundle de la entrega, incluida su caché de textos,
en `experiments/demo-bundle-gru-final/`. `DEMO_BUNDLE_DIR` permite indicar otra
ubicación. Los datos del bundle no se versionan. No hace falta entrenar ni
instalar las dependencias de modelado para abrir un bundle ya construido.

Como alternativa, con acceso autorizado al artefacto de W&B y `WANDB_API_KEY`
configurada en el entorno:

```bash
python scripts/demo_app/fetch_bundle.py
```

El nombre y la versión del artefacto están fijados en `configs/demo.yaml`.
Abrir la dirección local que imprime Streamlit. El botón «Cómo usar» explica
la pantalla activa. Elegir una semana y un presupuesto de falsas alarmas;
la bandeja, las fichas y los resultados se actualizan para ese punto de operación.

## Textos y agentes

- `cache_only`: reproduce textos guardados sin llamadas a la API; si falta una
  respuesta, se utiliza la alternativa de plantilla.
- `cache_first`: usa la caché y consulta la API si falta una respuesta.
- `live`: consulta la API sin sustituir la caché guardada.

Seleccionar el modo mediante `DEMO_LLM_MODE`. Los modos con llamadas requieren
`OPENAI_API_KEY`; los parámetros del modelo y la política se declaran en
`configs/agents.yaml`. `DEMO_PASSWORD`, si está definida, habilita una clave de acceso.
No incluir claves ni contraseñas en archivos versionados.

La política determina la acción. Los agentes redactan y organizan los mensajes;
un verificador contrasta los textos contra los hechos y las reglas configuradas.
El perfil del vehículo se compara con vehículos sanos del mismo mercado: es una
comparación descriptiva, no una explicación causal de la predicción.

## Construir el bundle

Se necesitan las dependencias de modelado, los datos v2, los splits congelados,
el panel agregado `panel_v2_estaticas.parquet` y las tres corridas de la GRU con
su ensamble, según el [README principal](../../README.md) y la
[guía de reproducción](../../docs/reproducibilidad.md).

```bash
python scripts/build_demo_bundle.py --config configs/demo.yaml
```

El constructor usa solamente desarrollo y rechaza vehículos de prueba. Conserva
los scores del ensamble y calcula los puntos de operación configurados.
Para generar la caché de textos se requiere acceso a la API y se realizan llamadas:

```bash
python scripts/warm_demo_cache.py --config configs/demo.yaml
```

La app lee el bundle resultante; no reentrena la GRU al navegar.

## Implementación

- `app.py`, `app_pages/`: navegación y vistas.
- `common.py`: estado de semana y presupuesto, carga de configuración y bundle.
- `presentation.py`, `theme.css`, `tokens.css`: presentación y estilos.
- `timeline.js`, `tour.js`, `guide.py`: calendario y recorrido de ayuda.
- `assets/ford-script.svg`: marca Ford; el archivo original es de dominio público,
  y la marca pertenece a Ford.
- `src/agents/`: política, hechos, caché, redacción, triage y verificación.
- `Dockerfile`, `railway.json`: configuración del contenedor y despliegue.

La navegación admite escritorio y celular. La tipografía Manrope se carga desde
Google Fonts, con Arial como alternativa sin conexión. Para comprobar el pipeline
y las validaciones de agentes, ejecutar `python scripts/check_setup.py` con las
dependencias completas instaladas.
