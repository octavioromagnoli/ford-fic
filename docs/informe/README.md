# Informe final · Ford FIC III

[Leer el informe entregado](informe.pdf).

El modelo finalista es una GRU con atención y ensamble por rango de tres semillas.
El documento incluye el problema, el tratamiento de datos, las comparaciones,
la evaluación y el circuito de producto propuesto.

## Compilar el PDF

Las figuras y tablas necesarias ya están versionadas. Con Tectonic instalado:

```bash
cd docs/informe
tectonic informe.tex
```

Tectonic descarga paquetes LaTeX cuando no están en su caché.
`informe.tex` contiene el preámbulo y la portada; `secciones/` contiene el texto
y la bibliografía. `figuras/` y `tablas/` contienen los recursos incluidos.

## Regenerar figuras y tablas

Desde la raíz del repositorio, con los artefactos descritos en la
[guía de reproducción](../reproducibilidad.md):

```bash
python scripts/make_report_figures.py --config configs/report_figures.yaml
```

El script lee resultados guardados, sin entrenar ni volver a evaluar el test.
Los archivos de resultados y las tablas auxiliares se declaran en
`configs/report_figures.yaml`. Las figuras y tablas incluidas permiten compilar
el documento directamente.

En el informe, [test] identifica el holdout (111 vehículos, 103 con cortes,
32 con evento); [dev], la validación cruzada sobre desarrollo (446 vehículos).
