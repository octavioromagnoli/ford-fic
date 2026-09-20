# Las caídas de 5 puntos son ruido; una regeneración exige al menos 15

**Fecha:** 2026-09-20 · **Fase:** F2 (corrección del contrato) · **Configuración:**
`thresholds.regen_drop_points: 15.0` en `configs/data/features_v1.yaml`.
**Reproduce el panel:**

```bash
python scripts/build_eda_cache.py --config configs/data/eda_cache.yaml --force
python scripts/eda_gaps.py --config configs/data/eda_cache.yaml
python scripts/build_dataset.py --config configs/data/panel_v1.yaml
```

## El problema

La definición anterior contaba como regeneración cualquier caída mayor a 5 puntos de
`AirRegeneration` dentro de un viaje (o de `Acumulation` entre señales en el panel
secuencial). Ese umbral capturaba oscilaciones del nivel: el saldo de las caídas no
seguía la acumulación medida, con correlación **0,005**.

Al exigir una caída mínima de **15 puntos**, la correlación mediana dentro de cada
vehículo sube a **0,45**. Es una definición mucho más coherente con una descarga real
del filtro y pasa a ser el umbral canónico del panel agregado, el panel secuencial y
el EDA.

## Lo que todavía limita la interpretación

Incluso con 15 puntos, el detector cuenta aproximadamente **2,5 veces** más eventos
que el marcador literal `Regeneration` durante el período en que ambos existen. El
marcador tampoco es una verdad completa porque deja de registrarse el 25-05-2026.
Por eso 15 es el umbral operativo actual, no una validación definitiva contra ground
truth; conviene volver a barrerlo si Ford aclara la semántica o entrega una etiqueta
de regeneración confiable.

## Consecuencia

Cambian las features de conteo, tendencia, distancia y recencia de regeneraciones,
además del canal `regen_drops` del CNN-LSTM. Los paneles y las corridas construidos
con 5 puntos quedan como resultados históricos y deben regenerarse antes de comparar
números nuevos.
