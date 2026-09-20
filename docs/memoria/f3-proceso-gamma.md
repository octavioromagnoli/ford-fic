# Proceso gamma de degradación: la premisa no se sostiene en esta flota

**Fecha:** 2026-09-20 · **Fase:** F3 · **Rama:** `exp/gamma-degradation` ·
**Reproduce:** `python scripts/audit_gamma_monotonia.py --config configs/data/gamma_monotonia.yaml`
(deja los CSV por vehículo y `resumen.csv` en `experiments/gamma/dev/`)

**Resultado: hallazgo negativo. El modelo no se implementó.** No hay en estos datos un
proxy de carga irreversible que sea monótono en km *y* cuya tasa separe a los que
fallan de los sanos. Sin eso, un proceso gamma no tiene estado latente que estimar, y
lo que quedaría es una reparametrización cara de las features de familia B que el panel
v1 ya tiene.

## Qué se iba a construir y por qué era razonable

Un proceso gamma (Lawless & Crowder, *Covariates and Random Effects in a Gamma Process
Model with Application to Degradation and Failure*, Lifetime Data Analysis 2004) modela
un daño acumulado `D(km)` **monótono no decreciente**, con incrementos gamma cuya tasa
depende de covariables de uso, y define la falla como el **primer cruce de un umbral**.
El score es `P(primer cruce en [corte+G, corte+G+H])` y entra a `lead_time_curve()` como
cualquier otro.

La física del postratamiento encaja casi palabra por palabra: la **ceniza** del filtro
es irreversible —la regeneración quema hollín, no ceniza—, ocupa capacidad, y obliga a
regenerar cada vez más seguido hasta que el filtro no da más. Los trayectos cortos, que
es la hipótesis causal del enunciado, aceleran el proceso porque no dejan completar la
regeneración. Eso es exactamente lo que miden las 15 features de familia B
(`configs/data/features_v1.yaml`), que hoy el repo usa sueltas; la rama iba a ascenderlas
a modelo.

## Las dos condiciones, y cómo se midieron

Antes de escribir el modelo hay que medir lo que lo sostiene:

- **(a) monotonía** — el proxy de carga sube con el odómetro *dentro* del vehículo;
- **(b) separación** — la tasa a la que sube distingue fallados de sanos.

Todo sobre **dev** (290 vehículos), sobre los crudos de `trips`, y con el historial
**truncado en el evento**: después de `IdentificationDate` hay una intervención y el
nivel vuelve a cero por un motivo que no es el uso (17,1% de los viajes descartados).
El nivel de carga es `AirRegenerationEnd`, que es `signals.Acumulation` al 99,7%
([f2-columnas-airfilter-airregeneration.md](f2-columnas-airfilter-airregeneration.md)) y
viene con el odómetro casi completo.

Cuatro proxies, cada uno con su orientación (qué dirección predice la hipótesis):

| bloque | proxy | qué sería la ceniza |
|---|---|---|
| A | nivel resumido por bins de 2.000 km (min, p10, media) | el piso que deja la oscilación carga/descarga |
| B | residuo con el que **termina** cada regeneración | la ceniza literal: lo que el fuego no se lleva |
| B | nivel de disparo de la regeneración | el umbral se alcanza antes si hay ceniza |
| C | km entre regeneraciones (↓) y tasa local (↑) | menos capacidad libre ⇒ ciclos más cortos |

## (a) Monotonía: es el asentamiento, no la ceniza

Sobre el historial completo el nivel **sí** sube, y con números que a primera vista
cierran: mediana de Spearman +0,29 y el **81,5%** de los vehículos con ρ > 0 para el
nivel medio (Wilcoxon p = 1,2e−24); +0,22 y 70,9% para el piso del bin.

El bloque D lo desarma. Medido en ventanas de odómetro, casi toda esa tendencia vive en
los primeros mil kilómetros:

| ventana de odómetro | ρ mediano (nivel medio) | % vehículos con ρ > 0 | pendiente mediana |
|---|---|---|---|
| 0 – 6.000 km | **+0,457** | 89,9% | +4,73 puntos / 1.000 km |
| 0 – 8.000 km | +0,310 | 85,3% | +2,19 |
| **1.000 – 8.000 km** | **−0,036** | **45,0%** | −0,64 |
| **4.000 – 16.000 km** | **−0,009** | **47,0%** | −0,02 |

Sacando el arranque, la tendencia no es débil: **no existe**. El 45–47% de los
vehículos con ρ > 0 es exactamente lo que da una moneda. Lo que el bloque A veía era el
**primer llenado del filtro**, un transitorio de asentamiento que termina alrededor del
km 1.000 y que le pasa a todos los vehículos igual.

El proxy que la física señala como la ceniza literal —el residuo con el que termina la
regeneración— no muestra nada ni siquiera sin recortar: ρ mediano **+0,013**, 52,4% de
vehículos positivos, p = 0,52. Tiene además un problema de medición que conviene saber:
**el 51,1% de las 14.695 regeneraciones termina en 0 exacto**. La escala es entera y en
pasos de 5; el piso de ceniza que buscamos, si existe a estos kilometrajes, cae por
debajo de la resolución del sensor.

## (b) Separación: no, y el único efecto fuerte apunta al revés

| proxy | ventana | AUC de la tasa | p |
|---|---|---|---|
| km entre regeneraciones (↓) | historial completo | **0,306** | 0,0001 |
| tasa local de regeneración | historial completo | 0,358 | 0,004 |
| piso robusto (p10) | 0 – 8.000 km | 0,519 | 0,73 |
| nivel medio | 1.000 – 8.000 km | 0,402 | 0,074 |
| km entre regeneraciones (↓) | 1.000 – 8.000 km | 0,436 | 0,25 |
| nivel medio | 4.000 – 16.000 km | 0,380 | 0,12 |

Dos lecturas, y las dos importan.

**El único efecto grande y significativo tiene el signo invertido.** Sobre el historial
completo, la distancia entre regeneraciones **crece** con el odómetro en los vehículos
que fallan (+9,9 km de gap por cada 1.000 km recorridos) y baja levemente en los sanos:
AUC 0,306, p = 0,0001. Orientado hacia la hipótesis eso es un AUC de 0,69 *en contra*.
La hipótesis de ceniza predice lo opuesto.

**Y ese efecto es exposición, no física.** El historial de un fallado se corta en el
evento —span mediano **7.439 km** contra **15.508 km** de los sanos— así que una
pendiente sobre el historial completo compara tramos distintos de la vida del vehículo,
y el tramo temprano es justo donde está el asentamiento. Igualando la ventana de
odómetro, el AUC se mueve a 0,385–0,467 y ninguna ventana queda significativa
(p ≥ 0,13): el efecto grande desaparece y lo que sobra no tiene signo estable. Es el
mismo mecanismo que ya nos mordió con el marcador `Regenerations`
([f2-eda-revision-y-features.md](f2-eda-revision-y-features.md) §2.1): una tasa sin
exposición igualada mide el calendario, y el calendario mide la etiqueta.

Con la exposición igualada, **ningún proxy llega a 0,60 de AUC en la dirección de la
hipótesis**: el rango completo va de 0,380 a 0,519.

## Por qué pasa: la flota es demasiado joven

No es que la física esté mal, es que la escala de tiempo no es esta. La acumulación de
ceniza en un DPF es un fenómeno de **más de 100.000 km**: viene del aceite y de los
aditivos, y necesita decenas de miles de kilómetros para ocupar una fracción medible de
la capacidad. En esta flota:

- el evento cae en una **mediana de 7.987 km**;
- el historial observado tiene un span mediano de 15.508 km (sanos) y 7.439 km (fallados);
- **7 vehículos de 290 pasan los 50.000 km**.

A estos kilometrajes el canal irreversible todavía no tuvo lugar para diferenciar
vehículos. Lo que sí varía —y mucho— es el canal **reversible**: hollín, régimen
térmico, ciclos que no completan. Ese es el que el panel v1 ya mide con las familias A y
B, y el que sostiene el lift que tenemos.

## Qué significa para el repo

1. **No se implementó `src/models/gamma_process.py`.** La ruta de la rama decía parar si
   el paso 1 no se sostenía, y no se sostiene. El costo de la auditoría fue una tarde; el
   del modelo, dos días para un estimador de 5–10 parámetros construido sobre un estado
   latente que los datos no muestran.
2. **La auditoría queda en el repo y es ejecutable.** Si aparece una extracción con
   vehículos más viejos, `scripts/audit_gamma_monotonia.py` vuelve a contestar la
   pregunta sin rehacer nada: el criterio de veredicto está declarado en
   `configs/data/gamma_monotonia.yaml` (`veredicto:`), no escondido en el script, así que
   no se puede ablandar después de ver el resultado.
3. **Cuidado con la pendiente de la distancia entre regeneraciones como feature.**
   Medida sobre el historial completo separa con AUC **0,694** contra la etiqueta (sin
   orientar), que es más de lo que consigue cualquier `feat_*` del panel v1. Es de las
   cosas que un GBM encuentra solo. Mide cuánto duró el registro: con la exposición
   igualada baja a **0,564** y deja de ser significativa. Si aparece con importancia
   alta en algún modelo, es esto.
4. **Se confirma que `regen_residual` es un proxy pobre**, por resolución del sensor: la
   mitad de las regeneraciones termina en 0 exacto. `feat_regen_residual_mean` sigue en
   el panel —no molesta— pero no hay que leerla como "nivel de ceniza".

## Lo que haría falta para reabrir esto

- Vehículos con **más de 50.000 km** de historial, o una extracción con contrapresión
  diferencial del filtro (`DifferentialPressure`), que es la medición directa de la
  capacidad perdida y no depende de la resolución de una escala 0–100 en pasos de 5.
- O bien mover el modelo al canal que sí se mueve: un proceso de degradación sobre el
  **régimen térmico** (deuda térmica acumulada) en vez de sobre la ceniza. Eso no es esta
  rama, y el estado latente ahí no es monótono, así que el proceso gamma tampoco sería la
  herramienta.

## Comandos

```bash
python scripts/audit_gamma_monotonia.py --config configs/data/gamma_monotonia.yaml
python scripts/check_setup.py
```
