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
2. **Contra el piso.** PR-AUC cerca de la tasa base = sin señal, por más que sea
   el mejor de la tabla. `baserate` es el piso: un modelo que no le gana está
   roto o el problema no tiene señal con esas features.
3. **El número del pitch.** Detección y anticipación mediana al punto de
   operación: es lo que se le muestra a Ford. Un modelo con mejor PR-AUC pero
   peor anticipación no es el mejor modelo para este desafío. `—` en detección
   significa que ningún umbral entró en el presupuesto de falsas alarmas.
4. **Qué haría falta para decidir**, si la comparación no alcanza.

## 3. Cuándo la comparación no vale

Si el script imprime un aviso **No comparable** (panel, splits o presupuesto
distintos), eso va **primero**, antes de cualquier conclusión: modelos evaluados
sobre folds distintos no se comparan, y el ranking de la tabla en ese caso no
significa nada.

Tampoco se celebra un PR-AUC alto sin auditarlo (regla 6 de `CLAUDE.md`): si un
modelo salta muy por encima del resto, lo primero que se revisa es el gap de
blanking y las features casi-definicionales del evento, no el hiperparámetro.

## Ejemplo de respuesta

> | Corrida | Modelo | PR-AUC (oof) | … |
> |---|---|---|---|
> | `exp-lgbm-w20` | lgbm | 0.184 | … |
> | `f0-dummy-baserate` | baserate | 0.021 | … |
>
> LightGBM le gana claramente al piso de tasa base (0.184 vs. 0.021) y el
> intervalo por fold [0.121, 0.242] no toca al del baseline, así que la
> diferencia es real y no ruido de partición. Al punto de operación detecta el
> 41% de los vehículos con evento con 7.320 km de anticipación mediana, que es
> el número presentable. Antes de darlo por bueno conviene confirmar que el gap
> de blanking esté aplicado: un salto de 9× sobre la tasa base en el primer
> intento es exactamente el patrón que produce una feature casi-definicional.
