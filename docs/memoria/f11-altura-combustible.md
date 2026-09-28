# Altura y calidad del gasoil por zona: ¿entran como features?

**Fecha:** 2026-09-28 · **Fase:** F11 (factibilidad, no consume presupuesto ni corre modelos)
**Alcance:** lo ya medido en la EDA de dev v2 (`f9-eda-v2.md` §I), `configs/data/city_elevation.yaml`
y normativa pública de combustibles por país. **Test sin tocar.** No se entrenó nada: esto decide si
vale la pena gastar un preregistro, no el resultado.

## En una línea

**La altura sí se puede meter y tiene sentido físico, pero con estos autos su efecto no se separa
del cero; el gasoil por zona, casi no se puede: en cuatro de los cinco mercados es una constante del
país, o sea el mercado otra vez.** Si se prueba algo, es una sola ablación preregistrada de la altura
(§4), leída dentro de mercado × motor y contra su permutación.

## 1 · Por qué podrían importar (la física)

- **Altura:** menos presión de aire → menos oxígeno por ciclo → combustión más rica, más hollín por
  km y regeneraciones que cuestan más (menos O₂ para quemar el hollín). Es el ángulo del "ingreso de
  oxígeno" del título del desafío, que F2 dio por perdido porque TripSummary no trae elevación ni GPS
  (`f2-diccionario-trips-incompleto.md`). La entrega v2 lo reabre por la ciudad de venta.
- **Azufre del gasoil:** el azufre deja sulfatos y ceniza que la regeneración **no** quema; el DPF
  se tapa de forma irreversible y la contrapresión sube. El biodiésel (B7–B15 según el país) diluye
  el aceite durante las post-inyecciones de la regeneración.

## 2 · Altura: qué hay y qué dio

`scripts/build_city_elevation.py` ya dejó la altura de 151 de 153 ciudades (GeoNames, vía
Open-Meteo). Faltan CAPITAL (ARG, ambigua, 14 autos) y Punta Arenas (9999 en GeoNames); 22 autos no
traen ciudad.

| mercado | ciudades | altura mín · mediana · máx (m) | ¿sirve dentro del mercado? |
|---|---|---|---|
| COL | 17 | 14 · 758 · 2.582 | **sí**: Bogotá/Chía contra la costa, y es el mercado con más eventos |
| BRA | 91 | 4 · 371 · 1.222 | algo: rango chico, pero muchas ciudades |
| ARG | 37 | 1 · 67 · 1.054 | poco: casi todo pampa, más CAPITAL sin dato |
| CHL | 2 | 556 (Santiago) | no: una sola altura |
| PER | 5 | 29 · 2.397 · 3.596 | no: el rango es enorme pero hay 0 eventos en dev |

Lo que ya midió la EDA (`f9-eda-v2.md` §I, Poisson sobre autos-mes de dev):

| modelo | RR por cada 1.000 m | IC 95% |
|---|---|---|
| todos los mercados, ajustado por días desde la venta + mercado + motor | 1,30 | [0,93; 1,81] |
| solo COL, solo ENG_2 | 1,21 | [0,82; 1,78] |

- **La dirección es la que predice la física, y el crudo de COL es grande** (1,7 → 5,0 → 6,3 eventos
  cada 100 autos-mes de < 1.000 m a ≥ 2.000 m). Pero en COL solo falla ENG_2 y la mezcla de motores
  cambia con la ciudad: ajustado, el intervalo cruza el 1.
- **La temperatura ambiente apunta a lo mismo:** los fallados están más fríos que los sanos del mismo
  mercado y mes (AUC 0,37–0,45, `f9-eda-v2.md` §E.3). En un mismo mercado, lugar frío es altura.
  Parte del "bajo régimen térmico" puede ser la ciudad.
- **Entre países no se puede usar:** la tasa por mercado está cruzada con cómo Ford armó las listas.
- **Es una constante por auto.** Como el motor y el modelo, a lo sumo aporta la tasa de su grupo de
  ciudades y ningún orden del *cuándo*: (a′) no la puede ver.
- **Es la ciudad de venta, no donde se maneja.** Un auto vendido en Bogotá que vive en la costa queda
  mal medido. Sin GPS no hay cómo corregirlo.

## 3 · Gasoil por zona: qué dice la normativa

Los autos tienen DPF: son diésel, así que la variable es el gasoil, no la nafta.

| mercado | azufre del gasoil (2025–26) | ¿varía dentro del país? |
|---|---|---|
| CHL | ≤ 15 ppm en todo el país | no |
| COL | norma ≤ 50 ppm; Ecopetrol entrega ~10 ppm en todo el país desde 2025 | no (Bogotá ≤ 10 ppm, igual que el resto) |
| PER | 50 ppm en Lima y parte del país | sí, pero 0 eventos en dev |
| ARG | Grado 3 ≤ 10 ppm; Grado 2 ≤ 350 ppm (transición de la Res. SE 492/2023 hasta fin de 2025) | **por surtidor, no por zona**: lo elige el conductor y no lo sabemos |
| BRA | S10 y S500 conviven; el S500 es ~27–30% de las ventas, más en el Norte y zonas remotas; la ANP suspendió en 2025 el calendario para retirarlo | **sí, por región/estado** |

Lo que eso implica:

1. **En CHL, COL y PER es una constante del país:** agregarla es agregar `static_SalesCountry_cd` con
   otro nombre. Y COL, con el gasoil más limpio de la tabla, es el mercado con más fallas (48% en dev):
   el azufre no explica la diferencia entre países.
2. **En ARG la variación existe pero es del conductor**, no de la ciudad. No hay dato por auto.
3. **Solo BRA tiene una variación regional medible**: la participación del S500 en las ventas por
   estado (ANP publica ventas por producto y UF). Los livianos diésel nuevos (PROCONVE L6+) deberían
   cargar S10; la participación regional mide, como mucho, la probabilidad de que no lo hagan.
4. **El modelo ya mide el efecto por auto:** la familia B (regeneraciones por 1.000 km, distancia
   entre regeneraciones, nivel residual del DPF) es la consecuencia del combustible y de la altura en
   *ese* auto. Un promedio regional imputado es un proxy peor de algo que ya se observa.

## 4 · Qué se propone (para preregistrar, no corrido)

Una sola comparación, con la lista cerrada de antemano:

- **Candidato:** el modelo de referencia v2 que se elija en el preregistro del finalista (hoy, GRU +
  TripSummary + estática completa ×3 semillas) **más** `static_elevation_m` (altura de la ciudad de
  venta; NaN sin ciudad o ambigua, imputación dentro del `Pipeline`).
- **Opcional, solo si hay tiempo:** `static_s500_share_state` para BRA (NaN fuera de BRA), con la
  serie de la ANP del período de uso. Es una segunda comparación y gasta presupuesto.
- **Se lee:** detección por auto a 5 · 10 · 20% de falsas alarmas con umbral exacto
  (`report_v2_models.py`), AUC por auto **dentro de mercado × motor**, bootstrap pareado contra el
  mismo modelo sin la altura.
- **Nulo:** la misma corrida con la altura permutada **entre autos del mismo mercado × motor**
  (conserva la distribución y rompe la relación). La mejora se mide contra ese nulo, no contra 0
  (`ford-nulo-antes-de-buscar`).
- **Regla de adopción:** entra solo si el bootstrap pareado separa la mejora del cero al 10% **y**
  el AUC dentro de la celda sube. Si no, queda como hallazgo de la EDA y como frase del pitch.

**Expectativa honesta:** con ~140 eventos en dev y un RR de ~1,2–1,3 por 1.000 m, lo más probable es
que no se separe. El valor más seguro de la altura es narrativo: explica el mecanismo en COL y es un
dato que Ford sí tiene (la ciudad del concesionario) para una versión productiva con más autos.

## Qué queda abierto

- **Bloqueo local (28-09):** esta máquina tiene los viajes v2 de fallados sin `VehicleCode`
  (`TripSummary_Failed_SelectionVins_v2_20260924_142629.csv`); el config espera
  `TripSummary_Failed_SelectionVins_vehiclecode_v2 (2).csv`. Para correr §4 hay que traer ese archivo.
- Pedirle a Ford la ciudad o región **de uso** (no de venta) o un GPS agregado por viaje: es lo que
  convierte la altura en una feature de verdad.

## Fuentes

- Altura: `configs/data/city_elevation.yaml` (GeoNames, CC BY 4.0) y `f9-eda-v2.md` §I.
- ARG: [Res. SE 492/2023, Boletín Oficial](https://www.boletinoficial.gob.ar/detalleAviso/primera/288023/20230609);
  [DieselNet, Argentina](https://dieselnet.com/standards/ar/fuel.php).
- BRA: [DieselNet, Brasil](https://dieselnet.com/standards/br/fuel.php);
  [ANP, participación del S500 por región (Minaspetro)](https://minaspetro.com.br/anp-segue-acompanhando-a-participacao-do-diesel-s500-em-todas-as-regioes-do-pais/);
  [Canal Rural, ventas de S500 en el 1er semestre de 2026](https://www.canalrural.com.br/economia/vendas-de-diesel-s500-caem-51-no-primeiro-semestre-de-2026/);
  [BiodieselBR, la ANP desiste del calendario](https://www.biodieselbr.com/noticias/regulacao/r/anp-desiste-de-criar-calendario-para-suspender-venda-de-diesel-mais-poluente-120925).
- COL: [Ecopetrol, calidad de combustibles](https://www.ecopetrol.com.co/wps/portal/Home/sostecnibilidad/ambiental/aire-limpio/calidad-combustibles);
  [Boyacá 7 Días, diésel de 10 ppm (02-2025)](https://boyaca7dias.com.co/2025/02/24/en-colombia-ya-se-produce-diesel-de-alta-calidad/).
- CHL, PER: [NRDC, *Dumping Dirty Diesels in Latin America*](https://www.nrdc.org/sites/default/files/latin-america-diesel-pollution-report.pdf).
