# Guion de la presentación — Ford Innovation Challenge III

**Estado:** borrador de contenido (28-09). No es el deck: es lo que el deck tiene que decir, en qué
orden y con qué número. La versión marketinera (§8) se arma después, sobre este guion.

**Cómo se edita:** todos los números viven en la tabla de §1 con un id `⟦M-xx⟧`. El resto del
documento los cita por id. **Si alguien mejora una métrica, cambia la fila de §1 (valor, fuente,
fecha) y nada más.** Ningún número del deck se copia a mano de otro lado: sale de §1, y §1 sale de
`docs/memoria/`.

---

## 0 · El mensaje en tres frases

1. **El problema:** el filtro de partículas (DPF) de los diésel de Ford se tapa por cómo se usa el
   auto —viajes cortos, motor que no llega a temperatura, mucho ralentí— y hoy Ford se entera cuando
   el tablero ya avisó: la falla es reactiva, cara y la sufre el cliente.
2. **Lo que hacemos:** con la telemetría que Ford **ya recibe** del auto conectado, marcamos los
   autos que van camino a la falla con **⟦M-06⟧ de margen**, y decimos **por qué** en términos de
   hábitos que el cliente puede cambiar.
3. **Lo que vale:** convertir una falla en garantía o una grúa en una visita programada o un
   consejo de manejo, con un nivel de falsas alarmas que Ford elige según cuánto le cuesta cada una.

---

## 1 · Métricas (tabla abierta: se actualiza acá)

Todas sobre **dev v2, fuera de fold** (446 autos, 135 fallados con cortes, R = 3). **El test (111
autos) no se tocó** y no hay finalista elegido sobre v2. Cuando lo haya, se agrega la columna de test
y esa pasa a ser la que se cita.

| id | qué | valor hoy | fuente | fecha | dueño |
|---|---|---|---|---|---|
| ⟦M-01⟧ | detección al **5%** de falsas alarmas (mejor modelo) | 27% | GRU + TripSummary + estática ×3 semillas, `f9-remedicion-completa-v2.md` | 26-09 | Track B |
| ⟦M-02⟧ | detección al **10%** de falsas alarmas | 42% ± 2 | ídem | 26-09 | Track B |
| ⟦M-03⟧ | detección al **20%** de falsas alarmas | 61% | ídem | 26-09 | Track B |
| ⟦M-04⟧ | lo mismo, **al azar** (score permutado, bolsa conservada) | 5 · 11 · 21% | ídem, fila "Nulo" | 26-09 | Track C |
| ⟦M-05⟧ | lo mismo, **solo sabiendo mercado × motor** (sin modelo) | 18 · 33 · 62% | ídem, fila "Tasa de la celda" | 26-09 | Track C |
| ⟦M-06⟧ | anticipación mediana de la primera alerta (al 10%) | ~7.500 km (~80–90 días) | `report_v2_leads.py`, ídem §7 | 26-09 | Track C |
| ⟦M-07⟧ | dispersión de la anticipación (p25–p75) | 3.700–12.300 km | ídem | 26-09 | Track C |
| ⟦M-08⟧ | AUC por auto, agrupado / dentro de mercado × motor | 0,82 / 0,65 | ídem | 26-09 | Track C |
| ⟦M-09⟧ | ¿el umbral se sostiene fuera de muestra? | sí en K2 (5% pedido → 3,5% real); **falta re-medir en v2** | `f8-capa-decision-k2.md` | 24-09 | Track C |
| ⟦M-10⟧ | explicaciones que coinciden con la física del DPF | 10 de 15 hábitos accionables; 30 de 32 alertas nombran un hábito (K2) | `f4-explicabilidad-k2.md` | 24-09 | Track C |
| ⟦M-11⟧ | ahorro esperado por 1.000 autos (escenario medio, e = 0,8, π = 5%) | ~USD 6.200, de los cuales ~5.300 son mérito del modelo (K2) | `f8-costos-k2.md` | 24-09 | Track C |
| ⟦M-12⟧ | test final (holdout, una sola vez) | **pendiente** | — | — | todos |

**Reglas para citar (que no se negocian en el deck):**
- Todo número de detección va **al lado de ⟦M-04⟧ y ⟦M-05⟧**. Un 42% solo no dice nada; "42% donde
  el azar da 11% y saber el mercado y el motor da 33%" sí.
- La anticipación se dice como **"lo marcamos con ~7.500 km de margen"**, nunca "predecimos que
  falla en 7.500 km": la dispersión (⟦M-07⟧) es enorme.
- Nada de accuracy. Nada del PR-AUC por fila en el deck (se explica en el backup si preguntan).
- El 15–17% de pitches anteriores es de la entrega 1: **no se usa más**.
- ⟦M-09⟧, ⟦M-10⟧ y ⟦M-11⟧ son de K2 (entrega 1). Si el finalista v2 es otro, hay que re-medirlos
  antes de citarlos, o citarlos como "en la versión anterior".

---

## 2 · Estructura del deck (versión técnico-comercial, ~12 slides, 10 minutos)

| # | slide | qué dice | visual | número |
|---|---|---|---|---|
| 1 | Portada | "Del aviso en el tablero al aviso a tiempo" | foto de un DPF / tablero con testigo | — |
| 2 | El problema | La falla del DPF es de uso, no de fábrica; Ford se entera tarde | historia de un auto: viajes cortos → DPF saturado → grúa | costo de una falla (rango de §6.2) |
| 3 | Por qué es difícil | Pocos eventos, dos listas muestreadas distinto, la fecha del evento es la del taller | línea de tiempo de un auto con el gap de 500 km | — |
| 4 | La idea | Mirar la telemetría hasta 500 km antes y preguntar "¿este auto va camino a fallar?" | diagrama ventana → gap → horizonte | — |
| 5 | Qué mira el modelo | Térmica, ralentí, regeneraciones del DPF, severidad | 4 familias con íconos | — |
| 6 | Resultado | Curva detección vs. falsas alarmas, con azar y celda dibujados | la curva (dashboard) | ⟦M-01⟧–⟦M-05⟧ |
| 7 | Cuánto antes | Distribución de la anticipación | histograma en km y días | ⟦M-06⟧, ⟦M-07⟧ |
| 8 | Por qué alerta | Un auto real: waterfall SHAP + el mensaje al cliente | captura del dashboard (pestaña Vehículo) | ⟦M-10⟧ |
| 9 | Cuánto vale | El dial: Ford elige el punto de operación según sus costos | pestaña Costos del dashboard | ⟦M-11⟧ |
| 10 | Cómo llega a producción | Arquitectura + piloto en sombra | diagrama de §4 | — |
| 11 | Qué hace falta de Ford | Más eventos, fecha de intervención, ciudad de uso, costos reales | lista corta | — |
| 12 | Cierre | Las tres frases de §0 | — | ⟦M-02⟧, ⟦M-06⟧ |

**Backup (no se presentan, se tienen a mano):** auditorías de leakage (gap, (a0), (a′)), por qué el
techo de cohorte es el piso, la celda mercado × motor, la altura y el gasoil
(`f11-altura-combustible.md`), el sweep de la GRU, los límites del muestreo.

---

## 3 · Cómo se vende

### 3.1 · A quién le sirve dentro de Ford

| quién | qué gana | cómo lo usa |
|---|---|---|
| **Posventa / garantía** | menos fallas en garantía y menos grúas; costo por evento más bajo | lista semanal de autos en riesgo por concesionario |
| **Red de concesionarios** | visitas programadas en vez de urgencias; ingreso de servicio | turno ofrecido al cliente con el motivo |
| **Cliente (FordPass / app)** | no quedarse parado; un consejo concreto de manejo | notificación: "hacé un tramo de ruta de 20 min" |
| **Ford Pro (flotas)** | un vehículo de trabajo parado cuesta mucho más que la reparación | tablero de flota con el riesgo por unidad |
| **Ingeniería de producto** | dónde se concentra el riesgo (mercado × motor) y con qué uso | la celda y los hábitos agregados, sin modelo por auto |

**El cliente principal es posventa**: es quien paga la falla y quien tiene el canal (concesionario +
app). Ford Pro es el caso de negocio más fuerte (§6.2: con falla cara el ahorro se multiplica).

### 3.2 · La propuesta de valor, en una línea por audiencia

- **Jurado técnico:** anticipación real, con gap de blanking y auditada contra el azar y contra
  saber el mercado; no es detección reactiva disfrazada.
- **Negocio:** un dial. Ford decide cuántas falsas alarmas tolera y el modelo le dice cuántas fallas
  evita y cuánto ahorra.
- **Cliente final:** "tu auto te avisa antes, y te dice qué cambiar".

### 3.3 · Los diferenciales (lo que otro equipo probablemente no tenga)

1. **Usa solo datos que Ford ya recibe.** Cero sensores nuevos, cero hardware.
2. **Explica en hábitos, y solo cuando la física está de acuerdo.** Las explicaciones que van contra
   la física del DPF no llegan al cliente (⟦M-10⟧).
3. **Honestidad medida.** Cada número está contra el azar y contra la composición de la muestra. Eso
   es lo que hace creíble el número en un piloto.
4. **El punto de operación es una decisión de negocio, no del modelo** (dashboard, pestaña Costos).

### 3.4 · Lo que no hay que prometer

- No predecimos **cuándo** exacto falla un auto; lo marcamos con margen.
- No sabemos la **prevalencia real** de la flota (las listas de Ford vienen muestreadas en dos
  cohortes). El ahorro en dinero es un rango por escenario hasta que Ford lo mida.
- Dentro de un mismo mercado y motor el orden entre autos es moderado (⟦M-08⟧). Hay que decirlo
  antes de que lo pregunten, junto con cómo se mejora: más eventos.

---

## 4 · Cómo se lleva a producción

### 4.1 · Arquitectura

```
Auto conectado ──► ingesta TripSummary + señales (lo que Ford ya recibe, batch diario)
                        │
                        ▼
           features por auto (mismo código que el entrenamiento: src/features, src/data/panel.py)
           · un corte cada Δ = 500 km · ventana W = 1.000 km · solo hacia atrás
                        │
                        ▼
           scoring (bundle serializado, §7) ──► score por auto y corte
                        │
                        ▼
           regla de alerta: 2 cortes seguidos ≥ τ(presupuesto elegido por Ford)
                        │
          ┌─────────────┼──────────────────────┐
          ▼             ▼                      ▼
   concesionario   app del cliente       tablero de flota / posventa
   (turno)         (hábito a cambiar)    (riesgo por unidad, por celda)
                        │
                        ▼
   retorno: ¿hubo intervención? (IdentificationDate) ──► monitoreo y reentrenamiento
```

**El mismo código calcula las features en entrenamiento y en producción.** Es la regla 3 del repo
llevada a operación: si producción reimplementa las features, aparece el *training-serving skew* y
los números del piloto dejan de ser los de este documento.

### 4.2 · Etapas

| etapa | qué | cuánto | criterio para pasar |
|---|---|---|---|
| 0 · Congelar | test final (⟦M-12⟧), refit con los 557 autos, bundle versionado | 1–2 semanas | test dentro del intervalo de dev |
| 1 · Piloto en sombra | se puntúa la flota real de 1 mercado (COL o CHL: más eventos), **sin avisar a nadie** | 3–6 meses | tasa de alertas ≈ presupuesto; detección realizada ≥ celda |
| 2 · Piloto activo | aviso a concesionarios de una región; grupo control sin aviso | 6 meses | fallas evitadas vs. control; costo por alerta |
| 3 · Escala | todos los mercados, recalibración de τ por mercado | — | — |

**El piloto en sombra es el paso que más importa:** es la primera vez que se mide la prevalencia
real y la tasa de falsas alarmas sobre la flota completa, no sobre las listas de Ford.

### 4.3 · Monitoreo

- **Tasa de alertas por mercado** contra el presupuesto elegido (si se dispara, el τ quedó viejo).
- **Deriva de las features** por mercado y mes contra la referencia del entrenamiento (temperatura
  ambiente y estación son las primeras sospechosas: ver `aux_` en CLAUDE.md).
- **Detección realizada**: cada evento nuevo en taller, ¿tuvo alerta antes? ¿con cuánto margen?
- **Datos que dejan de llegar**: el marcador `Regenerations` ya se cortó una vez para toda la flota
  (25-05-2026); un corte así cambia el score sin que el auto cambie.

### 4.4 · Reentrenamiento

- Trimestral, o cuando haya +50% de eventos nuevos (hoy el límite es la cantidad de eventos, no el
  modelo).
- Cada modelo nuevo pasa las mismas auditorías (`scripts/audit_model.py`) y el mismo reporte
  (`scripts/report_v2_models.py`) antes de reemplazar al anterior, y se compara en sombra contra él.

### 4.5 · Riesgos

| riesgo | mitigación |
|---|---|
| prevalencia real distinta de la del estudio | τ se recalibra en el piloto en sombra, no se hereda del dev |
| fatiga de alertas (el cliente ignora los avisos) | presupuesto de falsas alarmas bajo al empezar (5%); alertas solo con hábito explicable |
| ciudad de venta ≠ ciudad de uso | no usar la ciudad como feature hasta tener ubicación de uso |
| cambio de firmware/telemetría | monitoreo de columnas nulas o cortadas por mercado |
| privacidad | solo agregados por viaje que Ford ya procesa; sin ubicación |

---

## 5 · Serialización del modelo

**Qué hay hoy:** `scripts/train.py` reentrena por fold y guarda predicciones, no modelos (salvo la
incidencia externa, que usa `joblib`). **Falta un `scripts/export_model.py`** que entrene la
configuración final con todos los autos y deje el bundle de abajo. Es trabajo de Track B y no toca
nada de lo medido.

### 5.1 · El bundle

```
ford-dpf-<modelo>-<versión>/
  manifest.json          git sha, YAML del experimento completo, hash de features_v1.yaml,
                         lista y orden de columnas feat_/static_, W/G/H/Δ, k de la alerta,
                         origen del calendario (panel_meta.json), semillas, versiones de las
                         librerías, hash de la lista de vehículos de entrenamiento, métricas dev y test
  preprocess.joblib      el Pipeline de sklearn (imputación + escalado) ajustado con el train final
  model/                 LightGBM: booster.save_model() en texto (estable entre versiones)
                         GRU / CNN-LSTM: state_dict por semilla + export ONNX para servir sin torch
  score_reference.parquet  distribución del score de los sanos del train por semilla: convierte
                         cada score a percentil, que es lo que promedia el ensamble por rango
  thresholds.json        τ por presupuesto de falsas alarmas (fijado fuera de muestra) y el elegido
  explain/               lo que necesita el porqué (explain_texts.yaml, qué features son accionables)
  golden.parquet         20 autos con su score esperado: al cargar el bundle se re-puntúan y tienen
                         que dar igual a 1e-6
```

### 5.2 · Reglas

- **Nada de `pickle` del objeto entero** para guardar a largo plazo: se rompe al cambiar la versión de
  la librería. Formato nativo de cada modelo + `joblib` solo para el pipeline de sklearn, con las
  versiones fijadas en el manifest.
- **El ensamble por rango necesita la referencia congelada.** En producción no hay "el resto de la
  fila" para rankear: cada semilla se pasa a percentil contra `score_reference` y se promedia.
- **Se publica como wandb Artifact** tipo `model` en `oromagnoli-/ford-fic` (regla 9), con alias
  `candidate` → `shadow` → `production`. Nunca un archivo suelto en un drive.
- **El golden test corre en CI y al desplegar.** Es la misma garantía que `rescore_run.py` da hoy
  para las predicciones.
- **Qué se reporta y qué se despliega:** el número del test sale del modelo entrenado con dev; el
  modelo desplegado se reentrena con los 557 y la misma configuración (sin tocar hiperparámetros).

---

## 6 · Costos

### 6.1 · Costo de operar el modelo (infraestructura)

**Orden de magnitud, a validar con el equipo de datos de Ford.** Supuestos: 1 millón de autos
conectados, ~3 viajes por día, batch diario.

| componente | estimación | por qué |
|---|---|---|
| features + scoring diario | 1–2 h de un cluster chico (8–16 vCPU) | ~3 M de filas de viajes por día; el modelo puntúa en milisegundos |
| almacenamiento | ~1 GB/día de viajes, ya lo guarda Ford | no se agrega nada que Ford no tenga |
| reentrenamiento trimestral | < 1 h de CPU (hoy: 56 corridas en ~25 min en 24 núcleos) | no hace falta GPU |
| **total infra** | **del orden de cientos de USD por mes** (≪ USD 0,01 por auto y mes) | |

**El costo real no es la computadora:** es la integración con los sistemas de Ford (ingesta, app,
concesionarios) y, sobre todo, **el costo de cada alerta** (una inspección, un turno). Eso es lo que
decide el punto de operación.

### 6.2 · Cuánto vale (costo–beneficio)

La cuenta por auto (de `f8-costos-k2.md`):

- falsa alarma → cuesta una inspección `C_insp`;
- falla detectada a tiempo → `C_insp + C_prev + (1 − e)·C_fail`;
- falla no detectada → `C_fail`.

**El modelo paga cuando `B / C_insp` está entre ~5 y ~15** (`B = e·C_fail − C_prev − C_insp`) con
prevalencia real de 2–5%. Con falla barata no conviene alertar; con aviso casi gratis conviene
alertar a todos. Ejemplo citable: ⟦M-11⟧. Con flota comercial (falla de USD 15.000 con lucro
cesante) el ahorro sube a decenas de miles de USD por 1.000 autos, pero a presupuestos de falsas
alarmas no verificados fuera de muestra.

**Pendiente:** re-correr `scripts/cost_scenarios.py` con el finalista v2 y, si Ford da costos reales,
reemplazar los escenarios de precios públicos de EE. UU.

---

## 7 · Preguntas probables del jurado

| pregunta | respuesta corta | dónde está |
|---|---|---|
| ¿No es que el modelo aprende qué mercado falla más? | Por eso lo comparamos contra saber solo mercado × motor (⟦M-05⟧) y medimos el AUC dentro de la celda (⟦M-08⟧) | `f9-remedicion-completa-v2.md` |
| ¿Cómo sé que no es leakage? | Gap de 500 km, features solo hacia atrás, split por auto, y la permutación de features cae al azar | `scripts/audit_model.py` |
| ¿Por qué no accuracy? | Con ~6% de positivos, decir "nadie falla" da 94% | CLAUDE.md regla 5 |
| ¿Cuántos autos hacen falta para mejorar? | El límite es la cantidad de eventos; la semilla mueve más que los hiperparámetros | `f10-sweep-gru.md` |
| ¿Y la altura, el combustible? | La altura va en la dirección de la física, pero con estos autos no se separa del cero; el gasoil es casi constante por país | `f11-altura-combustible.md` |
| ¿Qué le piden a Ford? | Más eventos, la fecha de intervención (no la de registro), ciudad de uso y costos reales | §2 slide 11 |

---

## 8 · La versión marketinera (para más adelante)

Se arma **después** de que haya finalista v2 y test. Ideas para no perder:

- **Nombre del producto** (a elegir): algo del estilo *DPF Guard*, *Ford Early Care*, *Respira*. Que
  hable del cliente, no del modelo.
- **Arranque con una historia**, no con un gráfico: "María hace 4 km por día para llevar a los chicos
  al colegio. En tres meses su camioneta va a quedar en la banquina. Hoy lo sabemos." (auto ficticio,
  construido con el perfil real de un fallado del dashboard).
- **Un solo número grande por slide.** El resto va al backup.
- **Demo en vivo del dashboard** (pestaña Vehículo): un auto, su score subiendo, la alerta, el porqué y
  el mensaje. Es lo que más convence (plan §7).
- **Antes / después:** línea de tiempo de hoy (testigo → grúa → garantía) contra la de mañana (alerta
  → consejo → turno programado).
- **El dial** como metáfora visual: Ford gira la perilla de falsas alarmas y ve fallas evitadas y USD.
- **Cierre con el pedido**: piloto en sombra de 3 meses en un mercado.
- Formato: deck de 8–10 slides con identidad visual cercana a Ford (sin usar su marca registrada en
  materiales que parezcan oficiales), más un video de 60 s del dashboard.

---

## 9 · Pendientes para cerrar el guion

- [ ] Preregistro y elección del finalista v2 → actualizar ⟦M-01⟧–⟦M-08⟧.
- [ ] Test final → ⟦M-12⟧.
- [ ] Capa de decisión, costos y explicabilidad sobre el finalista v2 → ⟦M-09⟧–⟦M-11⟧.
- [ ] Si el finalista es la GRU: el porqué hoy existe solo para K2 (TreeSHAP). Hay que decidir si se
      explica la GRU (otro método) o si el mensaje sale de un modelo de árboles al lado.
- [ ] `scripts/export_model.py` y el golden test (§5).
- [ ] Figuras finales desde `src/eval/plots.py` (nada de capturas con números a mano).
