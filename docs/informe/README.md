# Informe final · Ford FIC III

El informe que se entrega a Ford (PDF desde LaTeX). Sigue las siete secciones del documento del desafío y mapea
cada criterio de la §5.2.

```bash
# 1 · figuras y tablas generadas (lee experiments/ y configs/, no entrena ni abre el test)
python scripts/make_report_figures.py --config configs/report_figures.yaml
# 2 · el PDF (tectonic baja los paquetes la primera vez)
cd docs/informe && tectonic informe.tex
```

- `informe.tex`: preámbulo, portada e índice; cada sección está en `secciones/`.
- `figuras/` y `tablas/`: salidas de `scripts/make_report_figures.py`. No se editan a mano: se cambia el YAML o
  la ficha de origen y se regeneran.
- Los números salen de `docs/memoria/`, del pitch (`docs/pitch/`) o de `experiments/*/report.json`. Cada figura y
  tabla trae su fuente.
- [test] es el holdout (111 autos, 103 con cortes, 32 fallados); [dev], la CV sobre los 446 autos de desarrollo.
