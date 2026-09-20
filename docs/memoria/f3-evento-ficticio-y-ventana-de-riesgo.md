# El evento ficticio cierra el atajo de posición — y corrige a quién culpábamos del leakage

**Fecha:** 2026-09-19 · **Fase:** F3

> ⚠️ Provisorio: los eventos tienen una falla conocida (error de SQL) y la mentora los va
> a corregir. El diseño y el código no dependen de eso; los números se re-miden.

Reproduce:

```bash
python scripts/build_dataset.py --config configs/data/panel_pseudo.yaml
python scripts/train.py --config configs/exp_logistic_pseudo.yaml
python scripts/audit_sequence.py --panel data/processed/panel_pseudo.parquet --columns all
python scripts/train.py --config configs/exp_l1_pseudo_abl_static.yaml   # la ablación fina
```

---

## 1 · Dos piezas, y la primera sola no alcanza

**El evento ficticio** (`assign_pseudo_events`, la asignación de fecha índice del sesgo de
tiempo inmortal): a cada sano se le sortea un punto `P` de la distribución de odómetros de
evento de los positivos —solo de dev— y se le corta la serie en `P − G`, igual que a un
positivo se le corta en `E − G`. `P` tiene que ser factible (`first + W + G ≤ P ≤ last − H`);
242 de 284 sanos lo consiguen, 42 se descartan y se cuentan.

**Y no alcanzó.** Con el evento ficticio solo, la posición seguía separando con P = 0,88
(contra 0,84 del v1), y el reparto de posición de las sanas no se movió: `[0,28 0,30 0,27
0,15]` contra `[0,01 0,09 0,13 0,77]` de las positivas. La medición mostró que el problema
no era dónde **termina** la serie:

> **Dentro de un vehículo, la etiqueta ES la posición.** Las filas positivas son, por
> definición, los últimos `H/Δ` cortes. Un sano —con evento ficticio o sin él— aporta filas
> en *todas* las posiciones de su serie; un positivo aporta las positivas solo en la última.

**La segunda pieza** (`label.window_only`) es la que cierra: se conserva de cada vehículo
**solo la ventana de riesgo**, los `H/Δ` cortes previos al evento real o ficticio. Con eso
los dos grupos aportan exactamente los mismos cortes relativos y la posición deja de decir
de qué grupo es la fila. El precio son los negativos tempranos de los vehículos con evento:
el panel pasa a ser un caso-control emparejado a nivel ventana.

## 2 · Los tres ejes cerrados a la vez, que era lo imposible

| panel | filas dev | tasa | **P(posición)** | TV mes | TV odómetro |
|---|---|---|---|---|---|
| v1 canónico | 2.029 | 0,125 | **0,836** | 0,186 | 0,162 |
| ficticio, sin ventana de riesgo | 1.256 | 0,202 | 0,884 | 0,307 | — |
| ficticio + ventana, sin emparejar | 1.119 | 0,227 | **0,491** | 0,519 | 0,277 |
| **ficticio + ventana + emparejado** | **455** | **0,558** | **0,489** | **0,091** | 0,234 |

La última fila es `configs/data/panel_pseudo.yaml`. El
[trade-off estructural](f3-emparejado-posicion-vs-calendario.md) —o posición o calendario—
desaparece cuando la posición se arregla **al construir las series** en vez de al muestrear
celdas: el emparejado por odómetro y mes recupera su trabajo (TV de mes 0,091, **mejor que
el v1**) porque ya no compite con una tercera dimensión.

La grilla de emparejado se reajustó a 8.000 km × mes con 40% de déficit: con el pool sano
mucho más chico, la del v1 dejaba 98 negativos y esta deja 246 (107 vehículos).

## 3 · El modelo sobre el panel limpio

| modelo | PR-AUC (tasa base 0,558) | lift | ROC-AUC |
|---|---|---|---|
| `baserate` | 0,557 | 1,00× | 0,496 |
| **`logistic`** | **0,732** | **1,31×** | **0,703** |
| `gbm` | 0,721 | 1,29× | 0,692 |
| `logistic_l1` | 0,710 | 1,27× | 0,681 |

**ROC 0,703 contra 0,617 del v1**, y esta vez sobre un panel donde la posición no separa
(0,489). Es el mejor número del proyecto y el primero que no se apoya en el artefacto. Con
la advertencia del tamaño: 455 filas, 140 vehículos, 53 con evento. El intervalo es ancho y
el PR-AUC **no** se compara con el de los otros paneles (la tasa base es 0,558, no 0,125).

## 4 · La corrección importante: el leakage no era el calendario

La ablación gruesa (`extra_prefixes: [aux_]`) mezclaba cosas distintas. Separada por
familia, sobre `logistic_l1` y en los tres paneles:

| panel | base | + `aux_air_temp` | + `aux_regen_marker` | + `aux_static_` | + todo |
|---|---|---|---|---|---|
| v1 | 0,617 | 0,624 | 0,617 | **0,710** | 0,719 |
| Δ=250 + posición | 0,670 | 0,664 | **0,742** | **0,759** | 0,922 |
| ficticio + ventana | 0,681 | 0,688 | 0,681 | **0,845** | 0,842 |

Tres lecturas:

1. **La temperatura ambiente no filtra en ningún panel** (+0,007 como máximo). El
   emparejado por mes hace su trabajo. En la vuelta anterior escribí que "el panel v1
   también filtra calendario": **era una atribución equivocada**, el salto era otra cosa.
2. **Lo que filtra en todos lados son las estáticas** (`Engine`, `ModelSeries`,
   `ProductionDay`, `daysUntilSale`): +0,09 en v1, +0,16 en el panel del evento ficticio.
   Es el sesgo de muestreo conocido desde F1 —`ENG_3` es el 0% de los fallados— y es
   exactamente por eso que están fuera del set base. Ahora está cuantificado.
3. **El marcador `Regenerations` sí filtra calendario, pero solo donde el mes no se
   empareja**: +0,072 en el panel Δ=250+posición y 0,000 en los otros dos. Se corta el
   25-05-2026 para toda la flota, así que en cuanto el mes queda libre se vuelve un reloj.
   Confirma por qué queda como `aux_`.

Moraleja de método: **la ablación gruesa detecta que hay un atajo, la fina dice cuál es.**
Vale correrla por familia y no en bloque.

## 5 · Qué quedó

- `src/data/panel.py`: `assign_pseudo_events()` (sorteo factible, solo referencia dev,
  determinístico por semilla) y `LabelConfig.window_only`. Los dos con su bloque en el YAML
  (`pseudo_event:` y `label.window_only`), apagados por default: el panel canónico no cambia.
- 11 chequeos nuevos en `check_setup.py` (52 en total): que el ficticio va solo a los sanos,
  que sale de la distribución de los positivos, que el sano sin rango factible se descarta y
  se cuenta, que la etiqueta del sano sigue en 0, que la serie termina en `P − G`, que un
  positivo y un sano con el mismo punto producen **los mismos cortes**, y que un vehículo no
  puede tener evento y evento ficticio a la vez.
- `configs/data/panel_pseudo{,_nomatch}.yaml` y los `configs/exp_*_pseudo.yaml`, más las
  nueve corridas de ablación fina (`exp_l1_*_abl_*.yaml`).

## 6 · Qué falta

- ~~Más negativos~~ **hecho, y con un resultado que no esperaba** (`draws_per_vehicle`):
  sortear k eventos ficticios por sano sí recupera negativos, pero **degrada el balance de
  posición**, porque el emparejado por odómetro se queda con las ventanas *tempranas* de
  cada sano mientras las positivas siguen en la suya. Y el ROC no mejora:

  | k | negativos | vehículos | P(posición) | ROC |
  |---|---|---|---|---|
  | **1** | **246** | **107** | **0,489** | **0,703** |
  | 2 | 408 | 155 | 0,577 | 0,691 |
  | 5 | 985 | 143 | 0,654 | 0,657 |

  Queda en **k = 1**: pagar desbalance por filas que no mejoran el número es exactamente el
  trueque que esta fase viene evitando. La clave queda como knob, con la curva medida.
- **Re-medir todo con los eventos corregidos**, que es lo que va a cambiar el diagnóstico.
- Decidir si el panel canónico pasa a ser este. No se toca hasta que la etiqueta esté bien.
