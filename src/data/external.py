"""Conjunto externo para la incidencia (F5 §3.2, Track A): los fallados sin fecha y sus sanos.

El universo del estudio (`src/data/usable.py`) dejó afuera 717 vehículos, y **no por la
etiqueta**: 284 fallados cuyo evento tiene la fecha por defecto (`IdentificationDate ==
daysUntilSale`) y 433 de mercados donde ningún evento tiene fecha. Para la pregunta
"¿este auto falla?" la fecha no hace falta, así que esos vehículos pueden enseñar
*qué auto* con muchos más eventos que los 60 de dev
(`CLAUDE.md`).

Este módulo decide **quién entra** a ese conjunto, y nada más:

1. **El lado del sorteo padre.** `test_split.json` guarda solo las listas recortadas;
   el lado que le tocó a cada excluido en el sorteo de los 1081 se reproduce con la
   misma `make_test_split` (semilla y `test_size` del holdout) y se **verifica contra la
   huella** `parent.test_vehicles_sha256_16`. Si no coincide, no se sigue: sería otro
   sorteo.
2. **Solo el lado dev del padre.** Los excluidos que el sorteo mandó a test no se tocan,
   ni se leen sus viajes. El test congelado (74) no cambia en ningún caso.
3. **Solo mercados con fallados y sanos en el conjunto.** El mercado es el estrato del
   modelo y la celda de la referencia de flota; un mercado con solo fallados (los de
   fecha por defecto de CNTRY_3/CNTRY_4, cuyos sanos están en dev) no aporta nada dentro
   del estrato y no tiene referencia sin usar a dev.

Qué es positivo y qué negativo, y con qué exposición, no se decide acá: es del panel
(`src/data/landmark.py`) y del preregistro
(`docs/reproducibilidad.md`).
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from src.eval.splits import _digest, make_test_split

logger = logging.getLogger(__name__)

ID_COL = "vehicle_id"
EVENT_COL = "event_observed"
EVENT_DAY_COL = "event_day_since_production"
SALE_DAY_COL = "static_daysUntilSale"
MARKET_COL = "static_SalesCountry_cd"
SIDES = ("dev", "test")


def reproduce_parent_split(vehicles: pd.DataFrame, test_split: dict[str, Any]) -> dict[str, list[str]]:
    """Las listas dev/test del sorteo de los 1081, verificadas contra la huella congelada.

    `vehicles` es la tabla estática deduplicada (`load_vehicle_static`): una fila por
    vehículo con `event_observed`. Se usan la semilla y el `test_size` que declara el
    holdout, no los del YAML, para que no haya dos fuentes.
    """
    parent = test_split.get("parent") or {}
    expected = parent.get("test_vehicles_sha256_16")
    if not expected:
        raise KeyError("`test_split.json` no trae `parent.test_vehicles_sha256_16`: no hay contra qué verificar")
    split = make_test_split(
        vehicles,
        test_size=float(test_split["test_size_requested"]),
        seed=int(test_split["seed"]),
        group_column=test_split.get("group_column", ID_COL),
        event_column=test_split.get("event_column", EVENT_COL),
    )
    got = _digest(split["test_vehicles"])
    if got != expected:
        raise ValueError(
            f"El sorteo reproducido tiene huella {got} y el congelado {expected}: no es el mismo sorteo. "
            "¿Cambió la tabla estática o el dedupe?"
        )
    if parent.get("n_vehicles") is not None and len(split["dev_vehicles"]) + len(split["test_vehicles"]) != int(parent["n_vehicles"]):
        raise ValueError("El sorteo reproducido no tiene la cantidad de vehículos del padre")
    dev, test = set(split["dev_vehicles"]), set(split["test_vehicles"])
    kept_dev, kept_test = set(test_split["dev_vehicles"]), set(test_split["test_vehicles"])
    if not kept_dev <= dev or not kept_test <= test:
        raise ValueError("El holdout recortado no es un subconjunto del sorteo reproducido")
    logger.info("Sorteo padre reproducido: %d dev · %d test · huella %s", len(dev), len(test), got)
    return {"dev": sorted(dev), "test": sorted(test), "sha256_16": got}


def default_event_date(vehicles: pd.DataFrame) -> pd.Series:
    """`True` si el vehículo tiene evento y la fecha es la de venta (la convención por defecto)."""
    event_day = pd.to_numeric(vehicles[EVENT_DAY_COL], errors="coerce")
    sale_day = pd.to_numeric(vehicles[SALE_DAY_COL], errors="coerce")
    return vehicles[EVENT_COL].eq(1) & event_day.eq(sale_day)


def select_external(
    vehicles: pd.DataFrame,
    test_split: dict[str, Any],
    parent: dict[str, list[str]],
    *,
    side: str = "dev",
    markets: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Los vehículos del conjunto externo y el informe de por qué quedó así.

    Entran los excluidos del universo que el sorteo padre puso del lado `side`, de los
    mercados de `markets`. Devuelve `(vehicles filtrado, informe)`; el informe cuenta
    cada descarte por mercado y cohorte, y falla si algún vehículo del conjunto aparece
    en dev o en test del holdout (la regla 2 entre el conjunto externo y dev).
    """
    if side not in SIDES:
        raise ValueError(f"`side` tiene que ser uno de {SIDES}")
    excluded = {str(v) for v in test_split["excluded_vehicles"]}
    kept = {str(v) for v in test_split["dev_vehicles"]} | {str(v) for v in test_split["test_vehicles"]}
    on_side = {str(v) for v in parent[side]}
    ids = vehicles[ID_COL].astype(str)

    base = vehicles.loc[ids.isin(excluded & on_side)].copy()
    base["default_event_date"] = default_event_date(base)
    in_market = base[MARKET_COL].astype(str).isin([str(m) for m in markets]) if markets else pd.Series(True, index=base.index)
    out = base.loc[in_market].copy()

    leaked = sorted(set(out[ID_COL].astype(str)) & kept)
    if leaked:
        raise ValueError(f"{len(leaked)} vehículo(s) del conjunto externo están en dev o test del holdout: {leaked[:3]}")

    def table(frame: pd.DataFrame) -> list[dict[str, Any]]:
        grouped = frame.groupby(MARKET_COL, observed=True)
        return [
            {"market": str(m), "vehicles": int(len(g)), "healthy": int(g[EVENT_COL].eq(0).sum()),
             "failed": int(g[EVENT_COL].eq(1).sum()), "failed_default_date": int(g["default_event_date"].sum()),
             "failed_dated": int((g[EVENT_COL].eq(1) & ~g["default_event_date"]).sum())}
            for m, g in grouped
        ]

    report = {
        "side": side,
        "markets": list(markets) if markets else None,
        "excluded_on_side": table(base),
        "excluded_on_other_side_untouched": int(len(excluded - on_side)),
        "dropped_by_market": table(base.loc[~in_market]),
        "kept": table(out),
        "n_vehicles": int(len(out)),
        "n_failed": int(out[EVENT_COL].sum()),
        "n_healthy": int(out[EVENT_COL].eq(0).sum()),
    }
    logger.info("Conjunto externo (lado %s del padre, mercados %s): %d vehículos, %d fallados",
                side, markets, report["n_vehicles"], report["n_failed"])
    return out.sort_values(ID_COL, ignore_index=True), report
