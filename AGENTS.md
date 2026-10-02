# Instrucciones para trabajar en el repositorio

Leer [README.md](README.md) para ejecutar el proyecto y [CLAUDE.md](CLAUDE.md)
para el contrato de datos y las reglas de validación. Los insumos necesarios se
describen en [docs/reproducibilidad.md](docs/reproducibilidad.md).

- Usar el entorno `.venv/`.
- Declarar rutas, semillas e hiperparámetros en YAML dentro de `configs/`.
- Conservar los splits por vehículo y ajustar el preprocesamiento solo en train.
- Crear una rama por cambio y ejecutar `python scripts/check_setup.py` antes de un PR.
- No versionar `data/`, `experiments/`, `wandb/`, entornos ni credenciales.
- Documentar resultados y limitaciones con precisión; distinguir desarrollo y prueba.
