"""Capa de producto sobre K2: el bundle de la demo, los hechos de cada alerta, la política y los agentes.

Nada de acá entrena, puntúa ni explica: todo lee el bundle que arma `scripts/build_demo_bundle.py`
con el stack completo. Por eso estos módulos solo importan pandas, numpy, yaml y openai, y la app
desplegada no necesita LightGBM, SHAP ni sklearn.
"""
