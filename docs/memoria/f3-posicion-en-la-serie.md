# La posición del corte dentro de su serie es casi la etiqueta

**Fecha:** 2026-09-19 · **Fase:** F3 · **Alcance:** dev · **Reproduce:**

```bash
python scripts/audit_sequence.py --panel data/processed/panel.parquet   # bloque 2
```

Este es el hallazgo más importante del 19-09 y **no es sobre ninguna feature en
particular**: es una propiedad del panel que afecta a cualquier columna que se construya
de acá en adelante.

## El número

En dev, P(fila positiva > fila sana) de columnas que **no son features** sino descriptores
de dónde cae la fila dentro de la serie de cortes de su propio vehículo:

| columna | qué es | P(pos > sano) |
|---|---|---|
| **`_frac`** | posición relativa: `rank(cut_odo) / nº de cortes del vehículo` | **0,829** |
| `_rank` | posición absoluta del corte en la serie | 0,651 |
| `_n` | cuántos cortes tiene el vehículo | 0,435 |
| — *referencia* — | `feat_idle_per_1000km`, la mejor feature de nivel | 0,589 |

La posición relativa separa **mucho mejor que la mejor feature del panel**, y la absoluta
también le gana. Ninguna de las dos está en el panel; se calculan en `audit_sequence.py`
justamente para poder medir esto.

## Por qué pasa: es construcción, no física

Sale directo de cómo se etiqueta (`src/data/panel.py`):

- Para un vehículo con evento en `E`, los cortes llegan **hasta `E − G`** y se detienen ahí.
- `label = 1` si `c ≥ E − G − H`, o sea en los **últimos `H/Δ` cortes** de esa serie.

Entonces "estoy al final de la serie de este vehículo" ⇒ "soy positiva", casi por
definición. En los sanos no pasa: su serie termina donde termina la observación, y las
filas están etiquetadas 0 en toda su extensión.

No es leakage del panel —el panel no expone la posición— pero **sí es un canal que
cualquier feature acumulativa puede importar sin querer**.

## Qué columnas están en riesgo

Cualquier cosa **monótona en el índice de corte** dentro de un vehículo:

- acumuladores y sumas corridas (fue el caso de `feat_degradation_cusum`, corr 0,37 con
  la posición; ver [f3-secuencia-zself-cusum.md](f3-secuencia-zself-cusum.md));
- contadores del tipo "cuántas veces pasó X desde el principio";
- `km_since_last_*` y cualquier "hace cuánto que...", si el evento que reinicia el
  contador es raro;
- estadísticos con ventana expandida (media/desvío del historial completo), que se
  estabilizan con el tiempo y por eso cambian sistemáticamente con la posición.

Las features de ventana `(c − W, c]` del set base **no** tienen el problema: miran
siempre los mismos 1.000 km, no el historial acumulado. Por eso el panel v1 sobrevive
a esta revisión sin cambios.

## Cómo se audita una feature nueva

Dos chequeos, los dos en `scripts/audit_sequence.py`:

1. **Correlación con la posición.** Si `|ρ(feature, _rank)| ≥ 0,2`, la feature tiene el
   atajo adentro y hay que mirarla con el chequeo 2 antes de creerle.
2. **Separación dentro de estratos de posición.** Se parte dev en cuartiles de `_rank` y
   se mide P(pos > sano) en cada uno. Una feature que separa por física **sostiene el
   número en los cuatro**; una que mide posición, oscila alrededor de 0,5. Ejemplo real
   (panel Δ=250):

   | estrato | `feat_idle_per_1000km` (nivel) | `feat_degradation_cusum` |
   |---|---|---|
   | q1 (temprano) | 0,745 | 0,424 |
   | q2 | 0,621 | 0,515 |
   | q3 | 0,654 | 0,572 |
   | q4 (tarde) | 0,563 | 0,481 |

3. **Para cualquier gradiente "hacia el evento"**: la referencia no puede ser "los sanos",
   tiene que ser **los sanos en el mismo rango de posición**. Sin eso, cualquier
   acumulador dibuja un gradiente perfecto que es el paso del tiempo.

## Por qué no se "arregla" el panel

Se podría igualar la longitud de las series (truncar a todos al mismo número de cortes),
pero eso tira los vehículos con historia larga, que son justamente los que más ventanas
aportan. La respuesta correcta es **no construir features monótonas en la posición**, y
auditar las que se construyan. Es la regla 6 de `CLAUDE.md` aplicada a un canal concreto.
