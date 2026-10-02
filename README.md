# Ford FIC · Predicción temprana de saturación del DPF

Ford Innovation Challenge III — *Data-Driven Powertrain Intelligence*.
Anticipamos la saturación del filtro de partículas diésel a partir de los
resúmenes de viaje y las señales de postratamiento de vehículos conectados.

**Modelo finalista: GRU con atención y ensamble por rango de tres semillas.**
Recibe los últimos 1.000 km como 20 tramos de 50 km × 15 canales, más país,
motor y serie. Estima un puntaje de riesgo para un evento entre 500 y 3.500 km
por delante del corte. El puntaje ordena riesgo; no es una probabilidad calibrada.

- [Informe final (PDF)](docs/informe/informe.pdf)
- [Fuentes y compilación del informe](docs/informe/README.md)
- [Reproducción y artefactos necesarios](docs/reproducibilidad.md)

## Resultados del informe

El universo contiene 557 vehículos: 446 de desarrollo y 111 de prueba;
103 de estos últimos tienen cortes evaluables, 32 con evento.

| Presupuesto de falsas alarmas por vehículo | Detección en prueba |
|---|---:|
| 5 % | ≈30 % |
| 10 % | ≈40–50 % |
| 20 % | ≈50–60 % |

La mediana de anticipación es de aproximadamente 4.600 km. Estos valores
corresponden a la curva de evaluación; con umbrales fijados en desarrollo,
las falsas alarmas realizadas en prueba fueron 5,8 %, 8,9 % y 18,2 %.
El informe detalla las métricas, la incertidumbre y las limitaciones.
El conjunto de prueba ya fue evaluado y no se usa para ajustar nuevas variantes.

## Cómo usar el repositorio

Se necesitan Git y Python 3.12. Abrir una terminal y descargar el proyecto:

```bash
git clone https://github.com/octavioromagnoli/ford-fic.git
cd ford-fic
```

Si se recibió un ZIP, descomprimirlo y abrir una terminal en la carpeta que
contiene `README.md`, `configs/` y `scripts/`. Todos los comandos siguientes se
ejecutan desde esa carpeta. Si Git solicita acceso, usar una cuenta autorizada
para el repositorio.

### 1. Instalar el entorno

En macOS o Linux:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-dl.txt
export WANDB_MODE=disabled
```

En Windows (PowerShell):

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt -r requirements-dl.txt
$env:WANDB_MODE = "disabled"
```

W&B es opcional: la ejecución local no requiere una cuenta de ese servicio.
En cada terminal nueva, volver a activar `.venv` y configurar `WANDB_MODE`.

### 2. Comprobar la instalación sin datos de Ford

```bash
python scripts/check_setup.py
python scripts/make_dummy.py --config configs/data/dummy_v1.yaml
python scripts/train.py --config configs/exp_dummy.yaml
```

El ejemplo sintético verifica el pipeline con un predictor de tasa base.
Sus métricas no son resultados del modelo finalista. La suite debe terminar
con todos los chequeos aprobados. La corrida de ejemplo se guarda en
`experiments/f0-dummy-baserate/`.

### 3. Preparar los datos reales

Los CSV y los splits congelados se reciben por separado; no vienen con el clon.
Copiar los CSV a `data/raw/` y los archivos `test_split.json` y `splits_r3.json`
a `data/processed/`. Los nombres exactos y la alternativa de recibir el panel
ya construido están en la [guía de reproducción](docs/reproducibilidad.md).
Con esos insumos, construir los paneles:

```bash
python scripts/build_dataset.py --config configs/data/panel_v2_estaticas.yaml
python scripts/build_seq_panel.py --config configs/data/panel_seq_trips_v2_estaticas.yaml
```

Si se recibió el panel secuencial con su metadata y los splits, omitir esos dos
comandos. Sin datos reales se puede completar el ejemplo del paso anterior.

### 4. Entrenar la GRU y combinar las semillas

```bash
python scripts/train.py --config configs/exp_v2all_gru_trips_estaticas_r3.yaml
python scripts/train.py --config configs/exp_v2all_gru_trips_estaticas_s1_r3.yaml
python scripts/train.py --config configs/exp_v2all_gru_trips_estaticas_s2_r3.yaml
python scripts/ensemble_rank.py --config configs/exp_v2all_seeds3_gru_trips_estaticas.yaml
```

Las tres corridas usan las semillas 42, 1 y 2 y los mismos folds agrupados por
vehículo: cinco folds, tres repeticiones, solo sobre desarrollo. El ensamble
combina predicciones fuera de muestra. La evaluación final del informe entrena
cada semilla con todo desarrollo; es distinta de esta validación cruzada.
La GRU puede variar entre ejecuciones incluso con la misma semilla.

Cada corrida guarda configuración, métricas y predicciones en `experiments/`.
Los paths, semillas e hiperparámetros se declaran en `configs/`.

### 5. Consultar las salidas

Cada semilla produce una carpeta `experiments/<nombre-de-corrida>/`. El ensamble
queda en `experiments/v2all-seeds3-gru-trips-estaticas/`. Allí se guardan las
métricas y las predicciones fuera de muestra; no confundirlas con las cifras del
test publicadas en el informe.

Para consultar la evaluación final, abrir [el PDF](docs/informe/informe.pdf).
Para regenerar sus figuras desde las salidas guardadas, seguir la
[guía del informe](docs/informe/README.md). No hace falta reentrenar para leerlo.

Si aparece `FileNotFoundError`, verificar los insumos y las rutas del YAML.
Si falla la validación de splits, usar el panel y los splits correspondientes;
no desactivar la comprobación ni sortear un nuevo conjunto de prueba.

## Contenido del repositorio

| Directorio | Contenido |
|---|---|
| `src/` | Datos, features, modelos, entrenamiento y evaluación |
| `configs/` | Configuración del finalista y experimentos anteriores |
| `scripts/` | Construcción de paneles, entrenamiento, evaluación y figuras |
| `results/` | Registro histórico de resultados y configuraciones |
| `notebooks/` | Exploración y auditoría de datos |
| `docs/informe/` | Informe entregado, fuentes, figuras y tablas |

Las implementaciones anteriores se conservan para respaldar las comparaciones.
El circuito de producto descrito en el informe incluye componentes de una demo
desarrollada en otra rama; este checkout contiene el pipeline de investigación.
MiniRocket, mencionado en las comparaciones del informe, tampoco está incluido
en este checkout.

Los datos de Ford, modelos entrenados, credenciales y outputs locales no se
versionan. Para trabajar en el código, consultar el [contrato de datos y reglas
antileakage](CLAUDE.md) y [AGENTS.md](AGENTS.md).
