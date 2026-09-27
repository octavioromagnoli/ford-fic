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

## K2 sobre v2: cómo da el mejor modelo hasta ahora (26-09)

```bash
python scripts/build_positional_panel.py --config configs/data/panel_positional_v2.yaml
WANDB_MODE=disabled python scripts/train.py --config configs/exp_v2_positional_r3.yaml    # piso posicional
python scripts/decision_layer.py --config configs/exp_decision_v2_k2.yaml                 # -> experiments/decision-v2-k2/
python scripts/decision_layer.py --config configs/exp_decision_v2_ss.yaml                 # -> experiments/decision-v2-ss/
```

K2 (F6) es survival stacking con horizonte completo **y** el conjunto en riesgo de la ventana
del registro. En v2 no hay ventana, así que K2 queda en su única parte que se puede portar:
`exp_v2_ss_hfull_r3`. Se mide igual que en F8 (`decision_layer.py`, con el nulo de tamaño de bolsa
y el umbral fijado fuera de muestra), solo sobre dev y con la etiqueta dura (sin ventana no hay V).
El test no se tocó. No es un candidato: todo es monótono en el score.

| | K2, entrega 1 (V) | **K2 sobre v2** | survival stacking sobre v2 |
|---|---|---|---|
| fallados / sanos en dev (por auto) | 45 / 95 | 135 / 291 | 135 / 291 |
| (a′) · auditoría | +0,020 · pasa | −0,004 · **falla** | +0,018 · pasa |
| lift entre fallados | 1,10× | 1,26× | **1,44×** |
| AUC por auto dentro de mercado × motor | — | **0,60** | 0,57 |
| **5% FA, grilla de `train.py`** | 17,0% ± 3,8 (nulo 7,5) | **0% (la grilla salta de 0% a 7% FA)** | 9,4% ± 6,7 (17 / 0 / 21; nulo 3,9) |
| 5% FA, umbral exacto (informativo) | — | 11,6% ± 1,8 (nulo 5,4) | **14,1% ± 3,2** (nulo 5,6) |
| 5% FA, fuera de muestra (empírico) | 15,6% a 3,5% FA | 0% a 0% FA | **11,4% ± 4,6 a 4,2% FA** |
| ≤ 10% garantizado (Neyman-Pearson) | 17,0% a 4,2% FA | 14,3% a 7,1% FA | **16,8% a 7,3% FA** |
| 10% FA, grilla (nulo) | 26,7% (16,2) | 16,8% ± 4,9 (8,4) | **23,2% ± 2,4** (8,5) |
| anticipación mediana al 5% | 7.438 km | — | 8.662 km |

"Umbral exacto" es el mismo criterio de alerta (dos cortes seguidos), pero con el umbral puesto
en el cuantil exacto de los sanos y no en la grilla de 50 cuantiles de filas. En v2, con ~27
cortes por sano, un paso de la grilla mueve muchos autos juntos: eso le cuesta a K2 todo su 5%.
Se reporta al lado, sin reemplazar al número oficial.

**Lectura:**
1. **Sin su ventana, K2 no es el mejor modelo sobre v2.** Ordena autos un poco mejor que survival
   stacking (0,60 contra 0,57 dentro de la celda), pero falla (a′), detecta menos en todos los
   puntos y su 5% oficial es 0.
2. **Survival stacking (F3) es el más sólido sobre v2 con la regla del repo.** Aprueba (a0) y
   (a′), tiene el mejor lift entre fallados y la mejor detección fuera de muestra: ~11–14% al 5%,
   ~23% al 10%, con ~8.500 km de anticipación. Pero **su (a′) no le gana al piso posicional**
   (+0,016 contra +0,024, abajo) y su 5% en la grilla tiene una repetición en 0.
3. **Los números de detección de v2 no son los de la entrega 1.** Son otra etiqueta (dura, con la
   referencia −21 d) y otra población (3× los fallados, cinco mercados), y en v2 **todos los
   modelos detectan cerca del nulo más 6–12 puntos al 5%**. El "~15–17%" del pitch es de la
   entrega 1; sobre v2 el rango honesto hoy es **~11–14% al 5%**.
4. **Al 10% la detección es sobre todo composición.** La tasa de eventos de la celda mercado ×
   motor sola, sin modelo, en la muestra y dejando al auto afuera, detecta 24,8% al 10% de
   falsas alarmas. Al 5% no detecta nada, porque cada celda riesgosa tiene demasiados sanos. **El
   5% es donde el uso agrega algo que la celda no da.**
5. **El control LightGBM detecta más que los dos (umbral exacto: 17,5% ± 4,6 al 5%, 27,7% al
   10%)** con (a′) negativo. Es la regla 6 en su versión por auto: la detección por vehículo
   premia saber *qué auto*, no *cuándo*. Es una observación, no un candidato: el presupuesto de
   comparaciones sigue agotado.

**Pisos del *cuándo*** (`exp_v2_positional_r3`, solo `feat_cut_odo`):

| | (a′) | lift entre fallados | 5% FA, umbral exacto |
|---|---|---|---|
| piso posicional | +0,024 | 1,24× | 3,2% (bajo el nulo) |
| survival stacking | +0,016 | 1,44× | 14,1% |

El odómetro solo explica todo el (a′) de survival stacking en v2, pero no su lift entre fallados
ni su detección. El *cuándo* de v2 se lee en el lift entre fallados, no en (a′).

## Qué no decide esto

- **Cuál es el finalista sobre v2.** K2 se eligió sobre la entrega 1, con su etiqueta corregida
  por la ventana. Sus números (15–17%) no valen para v2, y su variante sin ventana no aprueba
  (a′). Elegir sobre v2 es un preregistro nuevo.
- **Si la referencia −21 d es la correcta.** Es la del panel; la sensibilidad (0 / 14 / 21 /
  30 d) es lo primero que hay que medir al fijar el panel v2.
- **Si motor y modelo entran al set base**: están medidos, no decididos (ver `decisiones.md`,
  26-09).
