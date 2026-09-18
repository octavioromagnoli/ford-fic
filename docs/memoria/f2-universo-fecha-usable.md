# El universo del estudio son 364 vehículos, no 1081

**Fecha:** 2026-09-17 · **Fase:** F2 (previo) · **Reproduce:**
`python scripts/make_test_split.py --config configs/data/test_split.yaml --force`
(imprime el cuadro por mercado y deja el recorte en `data/processed/test_split.json`).
El recorrido completo en [`notebooks/eda-exhaustivo-dev.ipynb`](../../notebooks/eda-exhaustivo-dev.ipynb) §3.3.

Este archivo **resuelve** el bloqueante de
[f2-identificationdate-igual-a-venta.md](f2-identificationdate-igual-a-venta.md).
Ese doc describe el problema; éste, el criterio que se aplicó y lo que costó.

## El hecho, en una línea

`IdentificationDate` trae **dos convenciones de registro mezcladas**, y cuál de las
dos le tocó a un vehículo **depende del mercado en el que se vendió**, no del
vehículo.

## Que la fecha por defecto es administrativa y no física

Sobre los 365 positivos del dataset completo:

| | n | % |
|---|---|---|
| `IdentificationDate == daysUntilSale` (exacto) | **284** | 77,8% |
| `IdentificationDate > daysUntilSale` | 81 | 22,2% |
| `IdentificationDate < daysUntilSale` | **0** | **0%** |

El tercer renglón es el que cierra el argumento. Si la fecha marcara el momento
real de la falla, alguna caería antes de la venta —un vehículo puede fallar en el
concesionario—. Que no pase **ni una vez en 365** dice que la columna se completa
con la fecha de venta cuando no hay otra cosa que poner.

En los 81 con fecha real el delta contra la venta tiene mediana de **148 días** y
solo 3 están por debajo de 30: no son eventos "casi en la venta", son eventos de un
vehículo que circuló.

## El corolario que obligó a tirar más de lo previsto

| mercado | fecha por defecto | fecha real | % usable | sanos |
|---|---|---|---|---|
| CNTRY_1 | 126 | **0** | 0,0% | 199 |
| CNTRY_2 | 104 | 1 | 1,0% | 199 |
| CNTRY_3 | 27 | 19 | 41,3% | 100 |
| CNTRY_4 | 12 | **61** | 83,6% | 184 |
| CNTRY_5 | 15 | **0** | 0,0% | 34 |

En CNTRY_1 y CNTRY_5 **ningún** evento tiene fecha utilizable; en CNTRY_2, uno de
105. La convención es del mercado.

**Por qué no alcanzaba con filtrar los positivos.** Tirando solo los 284 quedaban
797 vehículos con 81 eventos, pero los sanos seguían enteros: CNTRY_1 aportaba 199
negativos y 0 positivos, CNTRY_2 199 y 1, CNTRY_5 34 y 0. **343 de los 633
vehículos de dev (54%) habrían venido de mercados donde un evento es invisible por
construcción.**

Eso no es un desbalance de clases, es **selección sobre el resultado**: se descarta
un vehículo según lo que le pasó, y los negativos dejan de venir de la misma
población que los positivos. La tasa base queda inventada y cualquier feature
correlacionada con el mercado hereda el sesgo. Excluir `SalesCountry_cd` del set de
features no lo arregla —el problema está en quién entró a la muestra, no en qué
columna ve el modelo—.

## El criterio

Dos condiciones, declaradas en `universe` de `configs/data/test_split.yaml` e
implementadas en `src/data/usable.py`:

1. `require_usable_event_date: true` — un positivo entra solo si
   `IdentificationDate > daysUntilSale`. Los sanos no se tocan: no tienen fecha.
2. `keep_markets: [CNTRY_3, CNTRY_4]` — se conservan los mercados donde el evento
   **se puede observar**.

La simetría es lo que lo hace defendible: un vehículo sano de CNTRY_4 es un
negativo legítimo porque, si hubiera fallado, lo habríamos visto. Uno de CNTRY_1
no.

## Lo que quedó

| | vehículos | con evento | tasa |
|---|---|---|---|
| dev | **290** | **60** | 0,2069 |
| test | **74** | **20** | 0,2703 |
| fuera del universo | 717 | 285 | — |

De 1081 vehículos y 365 eventos quedan 364 y 80. La tasa de eventos sube de 0,338 a
0,207 —baja, porque los descartados eran casi todos positivos—, y la CV de 5 folds
queda con ~12 eventos por fold.

## El recorte se aplicó SOBRE el sorteo, no antes

`scripts/make_test_split.py` corre en dos etapas y el orden no es intercambiable:

1. **Sortear** los 1081 con `make_test_split()`, semilla 42. Idéntico al sorteo de
   siempre: la huella del reparto (`ba9aa4d290bc6610`) queda guardada en `parent` y
   coincide con la del holdout congelado el 2026-09-16.
2. **Recortar** con `restrict_test_split()`, que conserva el lado que le tocó a cada
   vehículo sobreviviente.

Sortear el universo ya recortado habría sido un dado nuevo, tirado **después** de
haber mirado los datos que motivaron el recorte. Verificado: ningún vehículo cambió
de lado, `test` nuevo ⊆ `test` viejo y `dev` nuevo ⊆ `dev` viejo.

**Lo que se paga por no re-sortear:** la estratificación se hizo sobre la población
vieja, así que las tasas quedan desparejas (0,207 dev contra 0,270 test) y el test
se lleva el 25% de los positivos en vez del 20%. Se acepta: forzar el balance a
posteriori es elegir el test. Y el desvío juega a favor del número de portada —20
eventos de test en vez de los ~16 que habría dado un sorteo nuevo—.

Un sorteo nuevo con semilla 42 sobre los 364 habría dado dev 637/65 y test 160/16,
con solo 34 vehículos en común con el test actual. No se hizo.

## El holdout ahora guarda tres listas

```json
{"dev_vehicles": [...290], "test_vehicles": [...74], "excluded_vehicles": [...717],
 "universe": {...}, "parent": {...}}
```

`excluded_vehicles` no es redundante. Sin esa lista, `test_split_masks()` no puede
distinguir dos situaciones muy distintas:

| situación | qué significa | qué hacer |
|---|---|---|
| el panel trae un vehículo **excluido** | se construyó con el universo viejo | regenerar el panel |
| el panel trae un vehículo **desconocido** | el universo cambió abajo del holdout | regenerar el holdout |

Sin distinguirlas, las dos terminan en dev por descarte y ninguna métrica lo
delata. Las dos fallan ruidosamente, con mensajes distintos.

Y `dev_mask` dejó de ser el complemento de `test_mask`: con un universo recortado,
"no es test" incluye a los excluidos. Ahora es pertenencia explícita a
`dev_vehicles`.

## Lo que este recorte NO arregla

- **El panel ya no se construye con los 1081.** El contrato cambió: se construye con
  los 364 del holdout. `test_split_masks()` falla si aparece un excluido.
- **Los 290 de dev son dos mercados, no cinco.** Toda conclusión vale para CNTRY_3 y
  CNTRY_4. Miden parecido —9 de 10 variables del ranking apuntan en la misma
  dirección en los dos— pero difieren en nivel: CNTRY_4 es 8 °C más cálido, circula
  a la mitad de velocidad y tiene `msg_full_per_1000km` **5,45x** más alto. Las
  tasas absolutas arrastran ese nivel.
- **2 de los 60 eventos de dev pasan el filtro y siguen sin historial**: `VEH_0056`
  y `VEH_0499`, con delta de 2 y 1 día y el evento en 5 y 8 km de odómetro. El
  criterio es necesario, no suficiente; los descarta
  `sampling.min_trips_in_window` al armar el panel.
- **No responde qué es `IdentificationDate` para los 284 descartados.** Sigue
  pendiente preguntárselo a Ford: si la respuesta permite recuperarlos, el universo
  se amplía cambiando dos claves del YAML.
