# Relaciones entre features: dos ganan, y la literatura de DPF no aporta

**Fecha:** 2026-09-19 · **Fase:** F3

> ⚠️ **Todos los números de este archivo son provisorios.** El 19-09 se supo que **hay
> una falla en los eventos y la mentora los va a corregir**. Toda cifra que se compare
> contra `label` —acá y en
> [f3-esfuerzo-de-control-y-dosis.md](f3-esfuerzo-de-control-y-dosis.md)— se mide contra
> una etiqueta que va a cambiar. Lo que **no** cambia es la maquinaria: las columnas, el
> bloque `derived:` y los agregadores son independientes de la etiqueta, y re-medir todo
> esto con los eventos corregidos es un `build_dataset.py` y el comando de auditoría.

Reproduce:

```bash
python scripts/build_dataset.py --config configs/data/panel_v4.yaml
python scripts/audit_sequence.py --panel data/processed/panel_v4.parquet --columns all
```

---

## 1 · Lo que faltaba no era una columna, era poder escribir un cociente

Hasta acá cada feature era **una** columna cruda con **un** agregador. Con esa primitiva
no se puede escribir "cuántas regeneraciones hace el sistema por cada minuto que el motor
estuvo encendido sin moverse": son dos agregados distintos de la misma ventana.

`src/features/derived.py` agrega el bloque `derived:` al YAML de features: `ratio`,
`product` y `diff` entre dos columnas `feat_*`/`aux_*` de la **misma fila**, con un piso
de denominador (`min_b`) para que un cociente con denominador ~0 no se convierta en el
outlier que domina el modelo. No leakea y no hay nada que fitear por fold: los insumos ya
son solo hacia atrás, y esto es aritmética fila a fila. Lo cubren 7 chequeos nuevos de
`check_setup.py` (41 en total), incluidos el del denominador ~0 y el de que barajar el
panel no cambia ningún valor.

## 2 · El resultado: el idle improductivo

Dos derivadas le ganan a todo lo que había en el panel, y son la misma idea:

| | P(pos>sano) | IC95% (bootstrap por vehículo) | nulos |
|---|---|---|---|
| **`feat_regen_per_idle_min`** | **0,370** | **[0,316 – 0,423]** | 2,5% |
| **`feat_dpf_load_per_idle_min`** | **0,371** | **[0,311 – 0,420]** | 2,5% |
| `feat_idle_frac` (lo mejor que había) | 0,597 | [0,536 – 0,664] | 0% |
| `feat_regenerations_per_1000km` (el numerador) | 0,467 | [0,400 – 0,532] | 0% |

Se lee así: **por cada minuto de motor encendido sin moverse, los vehículos que fallan
regeneran menos y cargan menos el filtro**. No es el numerador (0,467, que no separa) ni
el denominador solo (0,597): es la relación. Separa con |0,5 − P| = 0,130 contra 0,097 de
lo mejor que teníamos.

**Y aguanta las tres auditorías**, que es lo que la [regla de la
posición](f3-posicion-en-la-serie.md) obliga a mirar antes de festejar:

- correlación con la posición del corte en su serie: **0,119** (el CUSUM tenía 0,37);
- dentro de los cuatro estratos de posición: **0,290 / 0,422 / 0,376 / 0,354** — sostiene
  el signo y la magnitud en los cuatro, a diferencia de `regenerations_per_1000km`, que
  **se da vuelta** (0,622 / 0,654 / 0,394 / 0,422) y por eso su número pooled no
  significa nada;
- **a nivel vehículo**, que es la unidad de muestreo: P = 0,372 con **p = 0,0074** sobre
  53 vehículos con evento contra 118 sanos. No son unas pocas filas de un vehículo raro.

Que el mecanismo térmico apareciera como "idle" ya se sabía; lo nuevo es **la dosis de
tiempo** (los minutos de idle nunca estaban medidos: `idle_frac` e `idle_per_1000km`
cuentan *cuántos* idle hay, y un idle de 40 minutos pesaba igual que uno de 1) y **el idle
puesto como denominador del trabajo del postratamiento**.

El resto de la familia F queda entre 0,53 y 0,60 —`idle_min_per_1000km` (0,597) empata
con `idle_frac` y `idle_time_frac` (0,584) no le gana—, así que lo que vale la pena
defender son esas dos derivadas, no las veinte columnas.

## 3 · La literatura de DPF: buenas hipótesis, cero señal

Se buscaron los mecanismos publicados de degradación de DPF y se construyó una feature
para cada uno (familia G). **Ninguno separa**: la mejor llega a |0,5 − P| = 0,049.

| mecanismo (fuente) | cómo se midió | P |
|---|---|---|
| La ceniza no se quema y **reduce la capacidad** del filtro | `capacity_km_per_point` = km entre ciclos ÷ puntos que remueve cada ciclo | 0,541 |
| La **contrapresión** sube y se lee en el **consumo** | `fuel_slope_per_1000km` (la pendiente, no el nivel que ya estaba) | 0,475 |
| Lo primero que muere es la **regeneración pasiva** | bajadas chicas del nivel en viajes calientes: `passive_share_of_load`, `passive_per_active` | 0,538 |
| Las regeneraciones **diluyen el aceite** con combustible | `oil_drop_per_regen`, `oil_drop_per_cold_km` | 0,482 / 0,547 |
| Un filtro cargado necesita **corridas más largas** para completar el ciclo | `regen_trip_km_mean`, `regen_trip_km_vs_typical` | 0,451 / 0,486 |

Es consistente con lo que ya había dado
[f3-esfuerzo-de-control-y-dosis.md](f3-esfuerzo-de-control-y-dosis.md): **el subsistema
del filtro no anticipa nada en estos datos**. Cinco mecanismos independientes, todos con
fundamento publicado, todos planos. Lo que sí separa —el idle improductivo— separa por el
lado del **uso**, no por el lado de la pieza.

Fuentes de los mecanismos: [DieselNet, *Ash Accumulation in Diesel Particulate
Filters*](https://dieselnet.com/tech/dpf_ash.php) · [Fleet Equipment, telemática y
mantenimiento predictivo](https://www.fleetequipmentmag.com/10-ways-predictive-maintenance-with-telematics-data-can-boost-fuel-efficiency/)
· [Fluid Life, dilución de aceite por
combustible](https://www.fluidlife.com/resource-center/data-interpretation/documents/diesel-fuel-dilution-bad-for-engine/).

## 4 · Qué queda

- **El panel canónico sigue siendo el v1.** v2/v3/v4 son specs acumulativos: v4 = v1 + las
  16 físicas + las 20 de relaciones + las 10 de literatura. Cada uno reproduce al anterior
  exacto en las features compartidas (diferencia máxima 0), así que la comparación es limpia.
- **Nada se promueve hasta que los eventos estén corregidos.** Promover
  `regen_per_idle_min` ahora es elegir una feature con la etiqueta equivocada. Cuando
  llegue la corrección: reconstruir el v4, re-correr las tres auditorías, y si aguanta,
  esas dos derivadas entran a `features_v1.yaml`.
- **Lo que no depende de la etiqueta y ya está hecho:** el bloque `derived:`, las 11
  columnas de la familia F/G en `trips.py`, el agregador `slope_per_1000km`, y
  `audit_sequence.py --columns`.
