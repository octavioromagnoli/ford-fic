# Datar eventos desde la telemetría (Fase 0): ninguna marca de intervención fecha el evento

**Fecha:** 2026-09-24 · **Fase:** F8 (exploratoria) · **Rama:** `exp/datar-eventos`
**Alcance:** solo dev, sin entrenar y sin cambiar etiquetas. Son 56 fallados con telemetría a ±30 d
de su fecha y 230 sanos. **No se tocaron los excluidos ni el test.**
**Criterio:** `configs/data/event_dating.yaml`, commiteado antes de medir (7e1d817).

```bash
export FORD_DATA_DIR=$PWD/data/rebuild-0921
python scripts/audit_event_dating.py --config configs/data/event_dating.yaml --diagnose
```

## Por qué se intentó

Sin Ford, la única forma de sumar los 33 fallados sin fecha de CNTRY_3/4 (+55% de eventos, en la
misma población) es fecharlos desde los datos. La pista venía de
[f2-eda-revision-y-features.md](f2-eda-revision-y-features.md) §3.3: después de la fecha, el uso
cambia en ~80% de los fallados, así que la fecha marca una intervención real.

**Regla de diseño:** se fecha solo con marcas de intervención discretas entre viajes, nunca con
features del panel. Fechar donde "bajan los viajes bajo régimen" haría que el período anterior
tenga más viajes bajo régimen por construcción, y el modelo aprendería la regla de fechado.

## En una línea

**Ninguna marca pasa las tres condiciones: la Fase 1 no corre.** La más cercana es la caída del
DPF con el motor apagado. Se concentra alrededor del evento (79% a ±14 d, contra 48% en los días
lejanos del mismo auto, p < 0,001), pero es demasiado frecuente para fechar: la regla "primera marca
en la ventana" acierta a ±14 d en el 14% de los casos.

## Los números (tolerancia de decisión: 14 d)

| marca | acierto | nulo (mismo auto, lejos del evento) | exceso | p | regla a ±14 d | pasa |
|---|---|---|---|---|---|---|
| **DPF baja con motor apagado** (≥ 15 pts) | 0,786 | 0,476 | **+0,309** | < 0,001 | 0,143 | no (iii) |
| días sin uso ≥ 3 | 0,714 | 0,618 | +0,097 | 0,059 | 0,161 | no |
| días sin uso ≥ 7 | 0,304 | 0,393 | −0,089 | 0,95 | 0,143 | no |
| reseteo de aceite (≥ +30 pts) | 0,161 | 0,093 | +0,067 | 0,072 | 0,125 | no |
| km sin telemetría (≥ 20 km) | 0,304 | 0,246 | +0,058 | 0,15 | 0,071 | no |
| limpieza manual | 0,018 | 0,009 | +0,009 | 0,42 | 0,018 | no |
| filtro al límite | 0,054 | 0,039 | +0,015 | 0,38 | 0,018 | no |
| cualquier intervención | 0,893 | 0,841 | +0,052 | 0,17 | 0,143 | no |

Las condiciones eran (i) acierto ≥ 0,50, (ii) exceso ≥ 0,25 con p < 0,01 y (iii) la regla de fechado
a ±14 d en ≥ 50%. Todas las tolerancias (7/14/30 d) están en
`experiments/audit_event_dating/summary_by_marker.csv`.

**Lo que dicen:**
- **No hay un "paso por el taller" visible.** El aceite no se resetea en la fecha (16%), el auto no
  deja de usarse más de lo normal (los huecos de ≥ 7 d caen *menos* cerca del evento que lejos) y
  los mensajes de limpieza manual son casi inexistentes (71 en todo dev).
- **Lo único que se mueve es el DPF entre viajes.** Es consistente con §3.3 (después del evento, el
  nivel del filtro al final del viaje sube en el 84%). Pero hay 6.907 de estas caídas en 253 de los
  290 autos: son parte del funcionamiento normal, y el evento solo las hace un poco más probables.

## Diagnóstico posterior al veredicto (no preregistrado)

Se probó una regla alternativa: la caída del DPF **más grande** dentro de la ventana, en vez de la
primera. **No cambia el veredicto.** Está escrito para que el próximo preregistro no la redescubra
sin su contexto.

- **Como fecha:** a ±7/14/30 d en 19% / 31% / 41%. Está centrada en el evento (error con signo:
  mediana +2,9 d), pero es muy ruidosa (p25–p75 de −56 a +32 d). No sirve para poner el gap G =
  500 km en el lugar correcto.
- **Como firma del vehículo** ("¿este auto es un fallado?"): AUC 0,769. Pero **el largo de la
  ventana solo da 0,759** (regla 6: el máximo sobre una ventana más larga es más grande por
  construcción, y los fallados tienen más días dentro de la ventana del registro). Contra un nulo
  que permuta la etiqueta dentro de terciles de largo: media 0,599, p95 0,659, p = 0,0005. **Algo
  queda por encima del largo, pero el estrato es grueso** y parte del 0,769 sigue siendo exposición.
- **No es un predictor.** Mira toda la ventana, incluido lo posterior al evento: es una firma de la
  etiqueta. No puede entrar como feature.

## Qué queda abierto (ideas, no corridas)

- **Usar la firma para contestar "¿es el mismo evento?", no para fechar.** Si los fallados sin
  fecha de CNTRY_3/4 (y de CNTRY_1/2/5) tienen el mismo exceso de caídas del DPF que los fechados,
  contra sus sanos y con un nulo que conserve el largo, eso sostiene que son el mismo evento. Con
  eso podrían entrar **censurados por intervalo** (el evento cae en algún momento entre la venta y
  el fin de la ventana), sin fecha. Necesita su propio preregistro, y el nulo tiene que ser más fino
  que los terciles de este diagnóstico (emparejar por días en ventana y por mercado).
- **Preguntas para Ford**, que siguen siendo la vía directa: qué es la fecha por defecto, y si
  existe un registro de taller o de reclamo con fecha.

## Límites

- Los umbrales de las marcas (3/7 d, 30 pts, 15 pts, 20 km) se fijaron antes de medir y no se
  barrieron. Otro umbral podría acercar alguna marca al criterio, pero barrerlos sobre 56 eventos
  sería elegir ruido.
- Los mensajes de `signals` se usaron por timestamp. El marcador `Regenerations` no se usó (está
  cortado el 25-05-2026).
