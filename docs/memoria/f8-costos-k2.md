# Cuánto vale operar K2: escenarios de costo con fuente

**Fecha:** 2026-09-24 · **Fase:** F8/F4 (reporte, no consume presupuesto ni elige modelo)
**Alcance:** las curvas fuera de fold de K2 con la etiqueta V (45 fallados, 95 sanos), R = 3.
**Test sin tocar.**

```bash
export FORD_DATA_DIR=$PWD/data/rebuild-0921
python scripts/cost_scenarios.py --config configs/cost_scenarios_k2.yaml   # necesita K2 y la capa de decisión
```

## La cuenta

Por vehículo, desde el punto de vista de Ford, en USD:

| lo que pasa | costo |
|---|---|
| falsa alarma | `C_insp` (un diagnóstico que no hacía falta) |
| detectado a tiempo | `C_insp + C_prev + (1 − e)·C_fail` (la prevención evita una fracción `e` de las fallas) |
| no detectado | `C_fail` (la falla completa) |

- Alertar a un auto que va a fallar ahorra `B = e·C_fail − C_prev − C_insp`.
- Lo que decide dónde operar es el cociente `B / C_insp`, junto con la **prevalencia real** `π`.
- La prevalencia de dev (~32% con V) **no sirve**: está inflada por el muestreo en dos cohortes.

Las dos políticas triviales, no alertar a nadie y alertar a todos, entran como extremos de la curva.

## Los escenarios

Los rangos salen de precios públicos de EE. UU. El detalle y las URL están en
`configs/cost_scenarios_k2.yaml`. **No son costos de Ford ni de los mercados del estudio.**

| escenario | `C_insp` | `C_prev` | `C_fail` | de dónde sale |
|---|---|---|---|---|
| bajo | 100 | 150 | 800 | la falla casi siempre se resuelve con una limpieza (~500) + grúa (~110) + diagnóstico |
| medio | 150 | 300 | 2.000 | 30% reemplazo del DPF (~3.500 en concesionario Ford) y 70% limpieza |
| alto | 250 | 500 | 4.500 | reemplazo + grúa + auto de cortesía |
| aviso remoto | 10 | 0 | 2.000 | la alerta es un mensaje al conductor, sin visita (supuesto) |
| flota comercial | 250 | 500 | 15.000 | cota alta: parada y lucro cesante de un vehículo de trabajo |

`e` ∈ {0,5; 0,8} y `π` ∈ {2%, 5%, 10%}. Son supuestos: ni Ford ni los datos los dan.

**La grúa es un supuesto pesimista (nota del 29-09).** Los escenarios bajo, medio y alto la suman a toda falla, pero en modo de protección el auto suele llegar andando al taller, y los datos no dicen cómo llegó ninguno. Sin la grúa, `C_fail` baja ~110 por escenario. La tabla no se rehízo porque no se cita en el pitch.

## Lo que da

La tabla completa está en `experiments/decision-k2/cost_scenarios.csv`. Una selección:

| escenario | e | π | falsas alarmas óptimas | detección | ahorro contra la mejor trivial | USD / 1.000 autos | de eso, sobre el azar |
|---|---|---|---|---|---|---|---|
| bajo | 0,5 | 5% | 0% | 0% | 0% (no alertar) | 0 | 0 |
| medio | 0,8 | 5% | 4,9% | 23% | 6% | 6.200 | 5.300 |
| medio | 0,8 | 10% | 30% ⚠ | 66% | 18% | 35.100 | 9.600 |
| alto | 0,5 | 5% | 4,9% | 23% | 2% | 5.600 | 5.300 |
| alto | 0,8 | 5% | 15% ⚠ | 42% | 11% | 25.200 | 16.800 |
| flota comercial | 0,8 | 2% | 35% ⚠ | 72% | 25% | 74.800 | 18.600 |
| flota comercial | 0,5 | 5% | 49% ⚠ | 83% | 10% (alertar a todos) | 64.200 | 0 |
| aviso remoto | 0,5 | 5% | 100% | 100% | 0% (alertar a todos) | 0 | 0 |

⚠ = el óptimo cae por encima del 10% de falsas alarmas, fuera de lo verificado fuera de muestra
([f8-capa-decision-k2.md](f8-capa-decision-k2.md)).

**"Sobre el azar"** es lo que ahorra K2 por encima de un score permutado que conserva el largo de
cada historial (regla 6), en los presupuestos de la capa de decisión. Es la parte del ahorro que sale
del modelo.

## Qué significa

1. **Si la falla es barata (escenario bajo), K2 no paga:** lo óptimo es no alertar. Una limpieza de
   DPF cuesta poco más que un diagnóstico, y la prevención no evita todas las fallas.
2. **En la zona validada (≤ 10% de falsas alarmas), K2 ahorra poco:** de 0 a 6% del costo esperado,
   unos USD 0–6 por vehículo, y casi todo es mérito del modelo sobre el azar. Es el escenario medio o
   alto con prevalencia de 2–5%.
3. **Los ahorros grandes piden operar a 15–35% de falsas alarmas**, con fallas caras y prevención
   efectiva. Pero ahí los números son optimistas (el umbral se eligió sobre las mismas predicciones),
   no están verificados fuera de muestra y el azar ya detecta mucho.
4. **Si la falla es muy cara o la alerta es muy barata, gana alertar a todos y el modelo sobra.** Con
   un aviso remoto de costo ~0, mandarlo a toda la flota le gana a cualquier umbral. K2 solo vuelve a
   servir si la alerta tiene un costo real: fatiga del conductor, pérdida de confianza en los avisos.
5. **El valor de K2 vive en una franja:** `B / C_insp` de ~5 a ~15 con `π` de 2–5%. Fuera de esa
   franja, una política trivial gana o empata.

**Para el pitch:** no hay un ahorro único que citar. Lo que se puede mostrar es la franja donde el
modelo paga, y pedirle a Ford los dos números que la ubican: cuánto le cuesta una falla contra un
diagnóstico, y la prevalencia real del evento en la flota.

## Límites

- Todos los costos son supuestos con fuente pública, no datos del caso.
- `e` (cuánto evita la prevención) es el supuesto más fuerte y no se puede medir con estos datos.
- Son 45 fallados: el ahorro de cada escenario tiene el mismo ruido que la detección (±2 autos).
