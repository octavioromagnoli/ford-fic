"""Qué vehículo entra al estudio, y por qué los demás no.

Este módulo decide el **universo**: el conjunto de vehículos sobre el que se
construye el panel, se entrena y se reporta. Es una decisión distinta de la del
split (`src/eval/splits.py`, que reparte ese universo en dev/test) y va antes.

Existe porque la cohorte `failed` no es homogénea. `IdentificationDate` —la
columna que ubica el evento en el tiempo— viene con dos convenciones de registro
mezcladas, y solo una de las dos sirve para predecir:

- **Fecha real**: `IdentificationDate > daysUntilSale`, con una mediana de 148
  días entre la venta y el evento. El vehículo circuló, se degradó y falló: hay
  historial de uso antes del evento y se puede anticipar.
- **Fecha por defecto**: `IdentificationDate == daysUntilSale` exactamente. El
  evento queda pegado al día de la venta, con el odómetro en ~13 km y el 2,3% del
  historial por delante. No hay ventana W que agregar ni gap G que blanquear.

Que la segunda es una convención administrativa y no física lo cierra un conteo:
**`IdentificationDate < daysUntilSale` no pasa nunca** (0 de 365). Si la fecha
fuera el momento real de la falla, alguna caería antes de la venta.

## El corolario que no es obvio: la convención es del mercado, no del vehículo

| mercado | positivos con fecha por defecto | con fecha real | sanos |
|---|---|---|---|
| CNTRY_1 | 126 | **0** | 199 |
| CNTRY_2 | 104 | 1 | 199 |
| CNTRY_3 | 27 | 19 | 100 |
| CNTRY_4 | 12 | **61** | 184 |
| CNTRY_5 | 15 | **0** | 34 |

En CNTRY_1 y CNTRY_5 **ningún** evento tiene fecha utilizable; en CNTRY_2, uno de
105. Por eso no alcanza con filtrar los positivos: si se los tira y se dejan los
sanos, esos tres mercados aportan 432 negativos y 1 positivo. Eso no es un
desbalance, es **selección sobre el resultado** —se descarta un vehículo según lo
que le pasó— y deja a los negativos viniendo de una población distinta de la de
los positivos. Cualquier feature correlacionada con el mercado hereda el sesgo, y
excluir `SalesCountry_cd` del set de features no lo arregla: el sesgo está en
quién entró a la muestra, no en qué columna se le muestra al modelo.

La salida es simétrica: se conservan los mercados donde el evento **se puede
observar**, y dentro de ellos, los vehículos cuyo evento se puede ubicar. Un
vehículo sano de CNTRY_4 es un negativo legítimo porque, si hubiera fallado, lo
habríamos visto. Uno de CNTRY_1 no.

Los números de arriba se reproducen con `market_usability()`, y el detalle está en
`docs/reproducibilidad.md`.

## Entrega v2 (26-09-2026): la selección pasó del mercado a la fecha de producción

En v2 ningún fallado tiene la fecha por defecto: Ford sacó a esos 278 autos en vez de
corregirles la fecha, y los cinco mercados tienen eventos fechados. El criterio 1 ya
no tira a nadie y el 2 deja de ser necesario (se declaran los mercados con sanos).

Pero apareció el mismo problema en otro eje. Las dos cohortes se muestrearon sobre
**períodos de producción distintos**:

| producción | fallados | sanos |
|---|---|---|
| ago-2024 → 19-01-2025 | 93 | **0** |
| 20-01-2025 → 31-07-2025 | 177 | 380 |
| ago-2025 → dic-2025 | **0** | 341 |

Los 93 fallados producidos en 2024 no tienen sanos de su época: son positivos de otra
población. Y ningún auto producido desde agosto de 2025 figura como fallado, aunque
el hazard por días desde la venta de los producidos en ene-jun 2025 predice ~50
eventos entre ellos (28 en ago-sep, 22 en oct-dic). Cero observados contra ~50
esperados no es falta de exposición: es que sus fallas no están en la muestra. Un
sano de octubre de 2025 no es un negativo verificable, igual que no lo era uno de
CNTRY_1 en la entrega 1. Es la misma regla simétrica, en otro eje: se conserva el
período de producción donde las dos cohortes se muestrearon (criterio 3,
`production_day_window`). El detalle está en `docs/reproducibilidad.md`.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

ID_COL = "vehicle_id"
EVENT_COL = "event_observed"
EVENT_DAY_COL = "event_day_since_production"
SALE_DAY_COL = "static_daysUntilSale"
MARKET_COL = "static_SalesCountry_cd"
PRODUCTION_DAY_COL = "static_ProductionDay"


def event_date_usable(vehicles: pd.DataFrame) -> pd.Series:
    """`True` si el evento del vehículo se puede ubicar sobre el eje de km.

    Vale solo para los positivos: un vehículo sin evento no tiene fecha que ubicar,
    así que devuelve `False` y el criterio de mercado es el que decide si entra.
    Usar esta serie sola para filtrar tiraría toda la cohorte sana.

    El criterio es una desigualdad estricta sobre dos columnas crudas, a propósito:
    se puede verificar sin reconstruir el anclaje temporal ni el cache del EDA.
    """
    _require(vehicles, [EVENT_COL, EVENT_DAY_COL, SALE_DAY_COL])
    event_day = pd.to_numeric(vehicles[EVENT_DAY_COL], errors="coerce")
    sale_day = pd.to_numeric(vehicles[SALE_DAY_COL], errors="coerce")
    return vehicles[EVENT_COL].eq(1) & event_day.gt(sale_day)


def market_usability(vehicles: pd.DataFrame) -> pd.DataFrame:
    """Por mercado: cuántos positivos tienen fecha real, cuántos la tienen por defecto.

    Es el cuadro que justifica el criterio de mercado. `frac_usable` es la
    proporción de positivos con fecha utilizable: donde da 0, un evento es
    invisible por construcción y los sanos de ese mercado no son comparables.
    """
    usable = event_date_usable(vehicles)
    grouped = vehicles.assign(_usable=usable).groupby(MARKET_COL, observed=True)
    out = grouped.agg(
        n_vehicles=(ID_COL, "size"),
        n_healthy=(EVENT_COL, lambda s: int((s == 0).sum())),
        n_events=(EVENT_COL, "sum"),
        n_usable=("_usable", "sum"),
    )
    out["n_default_date"] = out["n_events"] - out["n_usable"]
    out["frac_usable"] = out["n_usable"] / out["n_events"].where(out["n_events"] > 0)
    return out.reset_index()[
        [
            MARKET_COL,
            "n_vehicles",
            "n_healthy",
            "n_events",
            "n_usable",
            "n_default_date",
            "frac_usable",
        ]
    ]


def production_by_cohort(
    vehicles: pd.DataFrame, window: tuple[float, float] | list[float] | None = None
) -> pd.DataFrame:
    """Vehículos por cohorte antes, dentro y después de la ventana de producción.

    Es el cuadro que justifica el criterio 3: si una cohorte tiene vehículos en un
    tramo de producción donde la otra no tiene ninguno, ese tramo es de otra población.
    """
    _require(vehicles, [ID_COL, EVENT_COL, PRODUCTION_DAY_COL])
    day = pd.to_numeric(vehicles[PRODUCTION_DAY_COL], errors="coerce")
    if window is None:
        band = pd.Series("todo", index=vehicles.index)
    else:
        lo, hi = float(window[0]), float(window[1])
        band = pd.Series("dentro", index=vehicles.index).mask(day < lo, "antes").mask(day > hi, "después")
    out = (
        vehicles.assign(_band=band, _cohort=vehicles[EVENT_COL].map({1: "fallados", 0: "sanos"}))
        .groupby(["_band", "_cohort"]).size().unstack(fill_value=0)
    )
    return out.reindex([b for b in ("antes", "dentro", "después", "todo") if b in out.index])


def select_universe(
    vehicles: pd.DataFrame,
    *,
    require_usable_event_date: bool = True,
    keep_markets: list[str] | None = None,
    production_day_window: tuple[float, float] | list[float] | None = None,
) -> tuple[pd.Series, dict[str, Any]]:
    """Máscara del universo del estudio + el informe de por qué quedó así.

    Los criterios se componen con AND y ninguno es un default silencioso: salen del
    YAML (`configs/data/test_split.yaml`, sección `universe`), así que volver al
    universo completo es cambiar una clave, no editar código.

    - `require_usable_event_date`: tira los positivos con fecha por defecto. Los
      sanos no se tocan —no tienen fecha—.
    - `keep_markets`: se queda con los mercados listados. `None` los deja todos, que
      es lo que reintroduce la selección sobre el resultado descrita arriba.
    - `production_day_window`: `[desde, hasta]` en `ProductionDay` (inclusive), el
      período de producción donde las dos cohortes se muestrearon. `None` no recorta.
      Un vehículo sin `ProductionDay` queda afuera si hay ventana.

    Devuelve `(mask, report)`. El informe se serializa dentro del holdout: quien
    lea `test_split.json` dentro de seis meses tiene que poder ver cuántos
    vehículos se descartaron y por cuál criterio, sin correr nada.
    """
    _require(vehicles, [ID_COL, EVENT_COL, MARKET_COL])
    keep = pd.Series(True, index=vehicles.index)

    usable = event_date_usable(vehicles)
    if require_usable_event_date:
        # Solo se juzga a los positivos: `~event_observed | usable`.
        keep &= vehicles[EVENT_COL].ne(1) | usable
    dropped_by_date = int((~keep).sum())

    if keep_markets is not None:
        in_market = vehicles[MARKET_COL].astype(str).isin([str(m) for m in keep_markets])
        before = keep.copy()
        keep &= in_market
        dropped_by_market = int((before & ~keep).sum())
    else:
        dropped_by_market = 0

    dropped_by_production = {"fallados": 0, "sanos": 0}
    if production_day_window is not None:
        _require(vehicles, [PRODUCTION_DAY_COL])
        lo, hi = float(production_day_window[0]), float(production_day_window[1])
        day = pd.to_numeric(vehicles[PRODUCTION_DAY_COL], errors="coerce")
        before = keep.copy()
        keep &= day.between(lo, hi)
        dropped = before & ~keep
        dropped_by_production = {
            "fallados": int((dropped & vehicles[EVENT_COL].eq(1)).sum()),
            "sanos": int((dropped & vehicles[EVENT_COL].ne(1)).sum()),
        }

    report = {
        "require_usable_event_date": bool(require_usable_event_date),
        "keep_markets": list(keep_markets) if keep_markets is not None else None,
        "production_day_window": (
            [float(production_day_window[0]), float(production_day_window[1])]
            if production_day_window is not None else None
        ),
        "n_input": int(len(vehicles)),
        "n_kept": int(keep.sum()),
        "n_dropped_no_usable_date": dropped_by_date,
        "n_dropped_market": dropped_by_market,
        "n_dropped_production": int(sum(dropped_by_production.values())),
        "dropped_production_by_cohort": dropped_by_production,
        "events_input": int(vehicles[EVENT_COL].sum()),
        "events_kept": int(vehicles.loc[keep, EVENT_COL].sum()),
        "markets": market_usability(vehicles).to_dict(orient="records"),
    }
    if production_day_window is not None:
        table = production_by_cohort(vehicles, production_day_window)
        report["production_by_cohort"] = {
            str(band): {str(k): int(v) for k, v in row.items()} for band, row in table.iterrows()
        }
    logger.info(
        "Universo: %d -> %d vehículos (%d sin fecha usable, %d por mercado, %d por producción); "
        "eventos %d -> %d",
        report["n_input"],
        report["n_kept"],
        dropped_by_date,
        dropped_by_market,
        report["n_dropped_production"],
        report["events_input"],
        report["events_kept"],
    )
    return keep, report


def _require(frame: pd.DataFrame, columns: list[str]) -> None:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise KeyError(f"La tabla de vehículos no tiene las columnas requeridas: {missing}")
