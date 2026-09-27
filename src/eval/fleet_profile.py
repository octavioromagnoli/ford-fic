"""El "por qué" descriptivo de la demo: en qué hábitos se aparta un auto de los autos sanos de su mercado.

**No es atribución.** El modelo de la demo (la GRU de F3) no tiene explicabilidad, y lee sobre todo las señales del
propio filtro. Esto no dice qué usó para alertar: compara el perfil de uso del auto en los cortes que dispararon la
alerta con los autos sanos de dev de su mercado, que es la misma referencia que las señales del filtro del taller
(`scripts/build_demo_bundle.py::technician_signals`). Por eso el texto dice "comparado con autos sanos de su mercado,
este auto…", nunca "el modelo alertó por…" (`docs/memoria/f9-demo-gru.md`).

Tres reglas, las del mensaje de F4 (`docs/memoria/f4-explicabilidad-k2.md`):

- **Solo hábitos accionables de la lista cerrada:** clase `accionable` con signo físico ≠ 0 en la clasificación
  preregistrada (`configs/explain_k2.yaml`, física del DPF) y con frase y recomendación en los textos del mensaje
  (`configs/explain_texts.yaml`). Síntomas del filtro y contexto no se nombran nunca.
- **Solo del lado que la física señala como riesgoso:** más idle es peor, más velocidad no. Un hábito del lado de
  los sanos no se nombra.
- **"Se aparta" es estar más allá de `min_share` de los sanos del mercado** (0,75: más extremo que 3 de cada 4), y
  se nombran hasta `max_factors`, de mayor a menor.

Todo es puro: recibe DataFrames y devuelve DataFrames.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

KEY = ["vehicle_id", "cut_odo"]
MARKET = "static_SalesCountry_cd"
ACTIONABLE = "accionable"


def healthy_market_reference(rows: pd.DataFrame, columns: Sequence[str], *,
                             market_column: str = MARKET) -> tuple[pd.DataFrame, pd.DataFrame]:
    """La flota sana de `rows`: la mediana de cada auto sano (índice mercado × auto) y, por mercado, la mediana de
    esas medianas. Un auto con muchos cortes no pesa más que uno con pocos."""
    cols = list(columns)
    healthy = rows.loc[rows["event_observed"].eq(0)]
    per_vehicle = healthy.groupby([market_column, "vehicle_id"])[cols].median()
    return per_vehicle, per_vehicle.groupby(level=0).median()


def window_values(rows: pd.DataFrame, cuts: pd.DataFrame, columns: Sequence[str], *,
                  market_column: str = MARKET) -> pd.DataFrame:
    """El promedio de `columns` en los cortes de `cuts` de cada auto, con su mercado (índice: vehicle_id). `cuts`
    son los cortes que dispararon la alerta, o el de score máximo si el auto no alertó (`explained_cuts`)."""
    cols = list(columns)
    values = cuts[KEY].merge(rows[KEY + [market_column] + cols], on=KEY, how="left", validate="one_to_one")
    return values.groupby("vehicle_id").agg({market_column: "first", **{c: "mean" for c in cols}})


def nameable_habits(features: Sequence[str], classification: Mapping[str, Mapping[str, Any]],
                    texts: Mapping[str, Any]) -> dict[str, int]:
    """`{hábito: signo físico}` de los que el "por qué" puede nombrar, en el orden de `features`.

    `classification` es el bloque `features` del preregistro de F4 y `texts` los textos del mensaje: entra un hábito
    accionable, con signo ≠ 0 y con frase y recomendación. Lo demás (síntomas, contexto, sin hipótesis física, sin
    texto) no puede llegar al conductor aunque se lo pida en `features`.
    """
    feature_texts = texts.get("features") or {}
    out: dict[str, int] = {}
    for feature in features:
        spec = classification.get(feature) or {}
        text = feature_texts.get(feature) or {}
        sign = int(spec.get("sign", 0))
        if spec.get("class") == ACTIONABLE and sign != 0 and text.get("phrase") and text.get("recommendation"):
            out[feature] = sign
    return out


def fleet_deviations(values: pd.DataFrame, per_vehicle: pd.DataFrame, reference: pd.DataFrame,
                     habits: Mapping[str, int], *, min_healthy: int = 10,
                     market_column: str = MARKET) -> pd.DataFrame:
    """Una fila por (auto, hábito nombrable), en el orden de `values` y de `habits`:

    - `value`: el promedio del auto en sus cortes; `reference`: la mediana de los sanos de su mercado;
    - `share`: la parte de los sanos del mercado que el auto supera **hacia el lado riesgoso** (un empate cuenta
      medio): 0,5 es estar en la mediana, 1 es ser más extremo que todos;
    - `spread`: (valor − mediana)·signo en rangos intercuartiles de esos sanos (desempata a `share`);
    - `risky`: (valor − mediana)·signo > 0, el único lado que se puede nombrar.

    Falla si algún mercado tiene menos de `min_healthy` autos sanos: la referencia no alcanzaría para comparar.
    """
    counts = per_vehicle.groupby(level=0).size()
    thin = {m: int(counts.get(m, 0)) for m in values[market_column].unique() if counts.get(m, 0) < min_healthy}
    if thin:
        raise ValueError(f"Mercados con menos de {min_healthy} autos sanos de referencia: {thin}")
    records = []
    for vid, row in values.iterrows():
        market = row[market_column]
        healthy = per_vehicle.xs(market, level=0)
        for feature, sign in habits.items():
            value, ref = float(row[feature]), float(reference.loc[market, feature])
            sample = healthy[feature].dropna().to_numpy(dtype=float)
            share = spread = float("nan")
            if np.isfinite(value) and len(sample):
                d = (value - sample) * sign
                share = float(((d > 0).sum() + 0.5 * (d == 0).sum()) / len(sample))
                iqr = float(np.subtract(*np.quantile(sample, [0.75, 0.25])))
                spread = float((value - ref) * sign / iqr) if iqr > 0 else float("nan")
            records.append({"vehicle_id": vid, "feature": feature, "sign": int(sign), "value": value,
                            "reference": ref, "share": share, "spread": spread,
                            "risky": bool(np.isfinite(value) and np.isfinite(ref) and (value - ref) * sign > 0),
                            "n_healthy": int(len(sample))})
    return pd.DataFrame(records)


def fleet_factors(deviations: pd.DataFrame, *, min_share: float, max_factors: int) -> pd.DataFrame:
    """`deviations` con `named` (se nombra) y `rank` (1…max_factors entre los nombrados).

    Se nombra un hábito del lado riesgoso con `share ≥ min_share`: hasta `max_factors` por auto, de mayor a menor
    `share` (desempate: `spread`, después el orden de los hábitos). Si ninguno pasa, el auto no tiene hábitos que
    nombrar, y se dice así: no se rellena.
    """
    out = deviations.copy()
    eligible = out.loc[out["risky"] & out["share"].ge(min_share)].assign(_order=lambda f: np.arange(len(f)))
    ranked = eligible.sort_values(["vehicle_id", "share", "spread", "_order"], ascending=[True, False, False, True],
                                  na_position="last", kind="stable")
    top = ranked.groupby("vehicle_id", sort=False).head(int(max_factors))
    rank = top.groupby("vehicle_id", sort=False).cumcount() + 1
    out["named"] = out.index.isin(top.index)
    out["rank"] = rank.reindex(out.index)
    return out
