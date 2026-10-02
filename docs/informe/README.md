# Informe final · Ford FIC III

El informe que se entrega a Ford (PDF desde LaTeX). Tiene formato de informe técnico (clase `article`, Latin Modern,
ecuaciones numeradas, referencias numeradas) y sigue las secciones del documento del desafío, con los criterios de la
§5.2 dentro de la sección 5.

```bash
# 1 · figuras y tablas generadas (lee experiments/ y configs/, no entrena ni abre el test)
python scripts/make_report_figures.py --config configs/report_figures.yaml
# 2 · el PDF (tectonic baja los paquetes la primera vez; en Windows: scoop install tectonic)
cd docs/informe && tectonic informe.tex
```

- `informe.tex`: preámbulo y portada; cada sección está en `secciones/` (`08-referencias.tex` es la bibliografía).
- `figuras/` y `tablas/`: salidas de `scripts/make_report_figures.py`, más `figuras/ford_logo.pdf` (el óvalo de la
  plantilla oficial del desafío). No se editan a mano: se cambia el YAML o la ficha de origen y se regeneran.
- **El PDF no cita el repositorio.** La fuente de cada número (`docs/memoria/`, `docs/pitch/`, `experiments/*/report.json`)
  queda en un comentario `% fuente:` al principio de cada sección o al lado de la figura o la tabla.
- [test] es el holdout (111 autos, 103 con cortes, 32 fallados); [dev], la CV sobre los 446 autos de desarrollo.
