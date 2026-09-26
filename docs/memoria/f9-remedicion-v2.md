# Re-medición sobre la entrega v2: más señal agrupada, la misma señal de uso dentro de cada celda

**Fecha:** 2026-09-26 · **Fase:** F9 · **Alcance:** dev v2, R = 3, test sin tocar, sin wandb
(`WANDB_MODE=disabled`). **No es un candidato**: son los mismos modelos y el mismo diseño de
panel medidos sobre los datos nuevos. El presupuesto de comparaciones lo permite porque hace
comparable una fila existente. **Reproduce:**

```bash
python scripts/build_dataset.py --config configs/data/panel_v2.yaml
python scripts/build_survival_panel.py --config configs/data/panel_survival_v2.yaml
python scripts/build_dataset.py --config configs/data/panel_v2_estaticas.yaml
python scripts/build_survival_panel.py --config configs/data/panel_survival_v2_estaticas.yaml
python scripts/make_splits.py --config configs/data/splits_panel_v2_r3.yaml
WANDB_MODE=disabled python scripts/train.py --config configs/exp_v2_<x>.yaml     # baserate, lgbm_r3, ss_r3, ss_hfull_r3, ss_hfull_r3_estaticas
WANDB_MODE=disabled python scripts/audit_model.py --config configs/exp_v2_ss_r3.yaml
```

## El panel v2

Es el diseño del panel v1 (W = 1000, G = 500, H = 3000, Δ = 500, emparejado de sanos por
odómetro × mes, 53 features) con tres cambios de datos:
- el universo v2;
- la referencia del evento = fecha registrada − 21 d ([f9-entrega-v2.md](f9-entrega-v2.md) §6);
- folds por label × mercado × motor.

| | panel v1 (dev) | **panel v2 (dev)** |
|---|---|---|
| filas | 2.029 | **11.703** |
| vehículos (con evento / con positivo) | 171 (53 / 53) | **426 (135 / 133)** |
| filas positivas | 254 | **682** |
| tasa base por fila | 0,125 | **0,058** |
| techo de cohorte (PR-AUC / lift) | 0,263 / 2,10× | **0,177 / 3,03×** |
| sanos que conserva el emparejado | 1.241 de 7.308 filas (17%) | 9.517 de 15.688 (61%) |
| cortes por bolsa: sano / fallado (media) | 9,0 / 18,3 | **26,9 / 28,8** |

173 de los 177 eventos del universo quedan interpolados sobre los viajes (incertidumbre mediana
1 km); 4 caen después del último viaje. Con los eventos repartidos en 13 meses, el emparejado por
odómetro × mes conserva muchas más filas sanas, y **las bolsas quedan casi simétricas**. Era el
pendiente de F3 ("reemparejar a nivel vehículo para que las bolsas sean simétricas",
`decisiones.md` 20-09), y en v2 se resuelve solo. El tamaño de bolsa ya no separa cohortes, pero
sí hace más difícil el presupuesto de falsas alarmas (punto 7 de abajo).

## Resultados

| dev v2, R = 3 | PR-AUC (lift) | ROC fila | Brier | (a′) | lift entre fallados | lift vehículo | ROC vehículo |
|---|---|---|---|---|---|---|---|
| tasa base | 0,057 (0,97×) | 0,489 | 0,055 | −0,002 | 0,97× | 0,97× | 0,49 |
| LightGBM (control) | 0,112 ± 0,005 (1,92×) | 0,690 | 0,167 | **−0,019 ± 0,003** | 1,37 ± 0,05× | 1,67× | 0,72 |
| **survival stacking** (F3) | **0,125 ± 0,006 (2,14×)** | **0,709** | **0,053** | **+0,016 ± 0,004** | **1,44 ± 0,04×** | 1,65× | 0,73 |
| SS con horizonte completo (K2 sin ventana) | 0,111 ± 0,001 (1,90×) | 0,685 | 0,054 | −0,006 ± 0,007 | 1,26 ± 0,01× | 1,75× | 0,75 |
| ídem + motor y modelo como `static_` | 0,128 ± 0,002 (2,20×) | 0,723 | 0,053 | +0,003 ± 0,007 | 1,26 ± 0,01× | 1,95× | 0,80 |

Las mismas corridas sobre la entrega 1 (panel v1, dev, R = 3):

| | PR-AUC (lift) | ROC fila | (a′) | lift entre fallados | lift vehículo |
|---|---|---|---|---|---|
| LightGBM (control) | 0,160 (1,27×) | 0,586 | −0,006 ± 0,004 | 1,09× | 1,49× |
| survival stacking | 0,172 (1,37×) | 0,597 | +0,016 ± 0,003 | 1,10× | 1,62× |
| K2 | 0,175 (1,40×) | 0,604 | +0,020 | 1,10× | 1,67× |

**Auditorías** (`scripts/audit_model.py`):

| | (a0) | (a′) | (b) calendario | veredicto |
|---|---|---|---|---|
| survival stacking | PR-AUC 0,056 contra 0,058: pasa | **+0,018** (ROC dentro del vehículo 0,63): pasa | ROC +0,008: no marca | **aprobada** |
| SS con horizonte completo | 0,062 contra 0,058: pasa | **−0,004**: falla | ROC +0,016: no marca | no aprobada |

## Lo que dicen, y lo que no

1. **La metodología de F3 se sostiene.**
   - El control ordena los cortes al revés ((a′) negativo, como en v1).
   - Survival stacking aprende algo del *cuándo* (+0,016, el mismo número que en v1) y aprueba
     (a0) y (a′).
   - Ninguno supera el techo de cohorte: el PR-AUC por fila sigue midiendo sobre todo *qué auto*.
2. **La (b) de calendario casi desaparece** (+0,008 contra +0,049 en v1). Encaja con que no haya
   ventana del registro: el atajo era la ventana.
3. **El *cuándo* dentro de los fallados es más fuerte que en v1**: lift entre fallados 1,44×
   contra 1,10×. Coincide con la trayectoria que la EDA encontró en v2
   ([f9-eda-v2.md](f9-eda-v2.md) §E). Falta ver cuánto es síntoma: depende de dónde está la
   intervención real, que en v2 se estima poblacionalmente.
4. **La separación de autos subió, pero casi todo es mercado y motor.** AUC por vehículo (score
   medio del auto, promedio de las 3 repeticiones):

   | | agrupado | dentro del mercado | dentro de mercado × motor |
   |---|---|---|---|
   | LightGBM | 0,72 | 0,61 | 0,58 |
   | survival stacking | 0,73 | 0,62 | 0,57 |
   | SS horizonte completo | 0,75 | 0,64 | 0,60 |
   | ídem + motor y modelo | 0,80 | 0,71 | 0,60 |

   **Dentro de cada celda de mercado × motor, el uso ordena autos con AUC ~0,57–0,60**: la
   misma "señal honesta débil" de la entrega 1 (ROC ≈ 0,58). Lo que v2 agrega es la diferencia
   entre celdas, y esa diferencia está cruzada con cómo Ford armó la lista (fallados sacados por
   mercado, ENG_3 fallados solo en BRA). **Toda cifra de v2 se reporta con su versión dentro de
   mercado × motor al lado.**
5. **Motor y modelo aportan la tasa de su celda, no orden entre autos.** Suben el AUC dentro del
   mercado (0,64 → 0,71) y no dentro de la celda (0,60 → 0,60). Su motivo de exclusión (ENG_3 =
   0% de los fallados) ya no existe, pero lo que suman es la parte de la señal que menos se
   puede defender afuera.
6. **Con horizonte completo, K2 gana en ordenar autos y pierde el *cuándo***, igual que en F6:
   lift por vehículo 1,75× contra 1,65×, y (a′) −0,006 contra +0,016. En v2 no hay ventana que
   corregir, así que K2 no tiene su justificación original. Sin ventana, la variante de F6 queda
   en survival stacking con horizonte completo, y esa no aprueba (a′).
7. **La detección a ≤ 50 falsas alarmas/1.000 no se compara con la de v1.**
   - Los sanos tienen ~26 cortes por bolsa en v2 contra 9 en v1, y "dos cortes seguidos sobre el
     umbral" es mucho más fácil de cumplir con 26 cortes: el mismo presupuesto obliga a un umbral
     más alto.
   - Da 9% ± 6 con survival stacking y 0% con horizonte completo.
   - Antes de citarla hace falta el nulo que conserva el tamaño de bolsa
     (`scripts/audit_detection_null.py`, regla 6), o re-emparejar por vehículo.

## Qué no decide esto

- **Cuál es el finalista sobre v2.** K2 se eligió sobre la entrega 1, con su etiqueta corregida
  por la ventana. Sus números (15–17%) no valen para v2, y su variante sin ventana no aprueba
  (a′). Elegir sobre v2 es un preregistro nuevo.
- **Si la referencia −21 d es la correcta.** Es la del panel; la sensibilidad (0 / 14 / 21 /
  30 d) es lo primero que hay que medir al fijar el panel v2.
- **Si motor y modelo entran al set base**: están medidos, no decididos (ver `decisiones.md`,
  26-09).
