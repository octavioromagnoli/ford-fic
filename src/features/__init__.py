"""Features de ventana móvil (F2, Track A).

Acá vive `windows.py`: una sola primitiva que recibe una lista de columnas y una
lista de agregadores y devuelve columnas `feat_*`. Agregar una feature tiene que
ser agregar una línea en `configs/data/panel_v1.yaml`, no escribir código nuevo.

Regla no negociable: toda agregación mira hacia atrás desde el punto de corte.
Ninguna feature puede usar información posterior al corte.
"""
