---
name: compare
description: Comparar corridas de entrenamiento de este repo (Ford FIC) y mostrar en el chat una tabla de modelo vs. métricas más una interpretación corta. Usar cuando se pida comparar modelos o experimentos, ver resultados, qué modelo va ganando, o revisar las corridas de experiments/.
---

# Comparar corridas

## 1. Correr el script

```bash
python scripts/compare.py                          # todas las corridas de experiments/
python scripts/compare.py --runs exp_a exp_b       # solo estas corridas (nombre del directorio)
python scripts/compare.py --model lgbm baserate    # solo las corridas de estos modelos
python scripts/compare.py --dir <otro_directorio>  # comparar corridas de otro lado
```

`--runs` y `--model` fallan con exit 1 y listan lo disponible si el nombre no
existe: si pedías tres corridas y la tabla trae dos, es un error, no un filtro.

Devuelve la tabla en markdown ya armada. **Pegala tal cual en la respuesta**: los
números salen de `metrics.json`, que es lo que escribió `train.py`. No los
recalcules, no los redondees distinto y no reordenes las filas — la tabla tiene
que poder contrastarse contra wandb sin discrepancias.

Si el script no encuentra corridas, la respuesta es que todavía no hay nada que
comparar y hay que correr `train.py` (skill `train`), no inventar una tabla.

## 2. Escribir la interpretación

Debajo de la tabla, **tres o cuatro frases**, sin viñetas decorativas ni repetir
los números que ya están arriba. Responde en este orden:

1. **Quién gana y si la diferencia es real.** Compará el PR-AUC contra el
   intervalo bootstrap del pie: si los intervalos de dos corridas se solapan, la
   diferencia no está demostrada y hay que decirlo así, no elegir igual al de
   media más alta.
2. **Contra el piso, y el piso es el techo de cohorte.** No es la tasa base.
   En este panel las 254 filas positivas de dev están **todas** dentro de los 967
   cortes de vehículos fallados, así que un modelo que solo sabe *qué* autos
   fallan —sin la menor idea de *cuándo*— saca **PR-AUC 0,263 y lift 2,10×**. Ese
   es el techo de cohorte y lo imprime la columna "Techo cohorte" con un `✓`/`✗`.

   De ahí salen tres cosas que hay que decir con todas las letras:

   - **Un PR-AUC por fila por debajo del techo no demuestra timing.** Marca `✗`:
     identificar la cohorte habría dado más. No invalida la corrida —el modelo
     puede seguir siendo el mejor de la tabla— pero no se puede vender como
     anticipación.
   - **Un lift de 1,6–2× no demuestra anticipación**, aunque sea la meta del
     plan: el techo ya está en 2,10×, o sea que esa meta se alcanza sin anticipar
     nunca. Si alguien celebra el lift, esto va en la respuesta.
   - **La tasa base sigue sirviendo solo para el piso de abajo** (`baserate`): un
     modelo que no le gana está roto. Pero ganarle no es señal de nada.

   La decisión entre modelos se apoya entonces en, por este orden:
   **(i) PR-AUC entre fallados** —medido solo sobre esos 967 cortes, donde el
   nivel del vehículo ya no ordena nada—, leído como **lift sobre 0,263**, nunca
   como número pelado; **(ii) el aporte del cuándo (a')**, que es lo que se pierde
   al colapsar el score al promedio de cada vehículo: negativo significa que el
   orden dentro del auto **resta** y que el modelo anda mejor sin él; y
   **(iii) la anticipación al punto de operación** (punto 3).

   Si dos modelos empatan en PR-AUC por fila, la respuesta es cuál gana en (i) y
   (ii), no cuál tiene la media tercer decimal más alta.
3. **El número del pitch.** Detección y anticipación mediana al punto de
   operación: es lo que se le muestra a Ford, y es la única de las tres que mide
   el *cuándo* en las unidades del producto. Un modelo con mejor PR-AUC pero peor
   anticipación no es el mejor modelo para este desafío. `—` en detección
   significa que ningún umbral entró en el presupuesto de falsas alarmas.
4. **Qué haría falta para decidir**, si la comparación no alcanza.

## 3. Cuándo la comparación no vale

Si el script imprime un aviso **No comparable** (panel, splits o presupuesto
distintos), eso va **primero**, antes de cualquier conclusión: modelos evaluados
sobre folds distintos no se comparan, y el ranking de la tabla en ese caso no
significa nada.

**Antes y después de 1eab4a1 no se comparan.** Ese commit corrigió el umbral de
regeneraciones y reconstruyó `panel.parquet`: mismas filas, mismos vehículos,
mismos folds, pero otros valores en las 7 columnas `feat_*regen*`. El control
pasó de 0,165 a **0,1612**, y la tabla separa corridas por ~0,005, o sea menos
que ese movimiento: el orden de las filas puede no sobrevivir al cambio de panel.

La huella de `splits.json` **no lo detecta** —es sobre ids de vehículo, no sobre
features—, así que `compare.py` lo infiere comparando la mtime de cada
`metrics.json` con la del panel que declara usar, y avisa "medidas contra una
versión anterior del panel". Cuando aparezca ese aviso:

- decilo antes de la tabla, igual que cualquier otro **No comparable**;
- no mezcles el ranking de corridas pre y post en una sola conclusión;
- lo que corresponde es re-correr la corrida vieja contra el panel actual, no
  ajustar el número a mano.

Una columna en `—` es otra cosa distinta: es una corrida anterior a que
`train.py` guardara esa métrica. No invalida la fila, solo significa que de esa
corrida no se puede decir nada sobre el techo ni sobre el cuándo hasta
re-correrla.

Tampoco se celebra un PR-AUC alto sin auditarlo (regla 6 de `CLAUDE.md`): si un
modelo salta muy por encima del resto, lo primero que se revisa es el gap de
blanking y las features casi-definicionales del evento, no el hiperparámetro.

## Ejemplo de respuesta

> | Corrida | Modelo | PR-AUC (oof) | Lift vs. base | Techo cohorte | PR-AUC entre fallados | Aporte del cuándo (a') | … |
> |---|---|---|---|---|---|---|---|
> | `f3-survival-stacking` | survival_stacking | 0.161 | 1.29× | 0.263 ✗ | 0.276 (1.05×) | +0.0152 | … |
> | `f3-lgbm-panel-v1` | lgbm | 0.161 | 1.29× | 0.263 ✗ | 0.287 (1.09×) | -0.0067 | … |
>
> Los dos empatan en PR-AUC por fila (0,161) y los intervalos por fold se solapan
> de punta a punta, así que por ese número no hay diferencia que declarar. Ninguno
> llega al techo de cohorte de 0,263: **el lift de 1,29× no es anticipación**, es
> reconocer qué autos fallan, y la meta de 1,6–2× del plan tampoco lo sería,
> porque el techo ya está en 2,10×. Donde sí se separan es en (a'): survival
> stacking gana 0,0152 por ordenar los cortes dentro del auto y el LightGBM
> **pierde** 0,0067 —colapsarle el score al promedio del vehículo lo mejora—, o
> sea que de los dos, uno anticipa un poco y el otro tiene el orden interno en
> contra. Ojo con que entre fallados el orden se invierte (1,05× contra 1,09×):
> son dos cortes distintos de la misma pregunta y acá no coinciden, así que
> conviene decir las dos y no elegir la que conviene. Para decidir falta saber si
> esos 0,0152 superan la dispersión entre repeticiones: con `splits.n_repeats: 3`
> y `dispersion()` se contesta en una corrida.
