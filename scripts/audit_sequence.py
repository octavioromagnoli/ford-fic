#!/usr/bin/env python
"""¿Una feature separa, o mide la posición del corte en la serie?

    python scripts/audit_sequence.py --panel data/processed/panel.parquet
    python scripts/audit_sequence.py --panel data/processed/panel_delta250.parquet
    python scripts/audit_sequence.py --panel data/processed/panel_v2.parquet --columns all
    python scripts/audit_sequence.py --panel data/processed/panel_v2.parquet \
        --columns feat_regen_efficiency_mean,feat_regen_residual_slope

Nació para la familia E (secuencia) y `--columns` lo abre a cualquier candidata: la
pregunta del bloque 3 —¿sigue separando entre filas que están en el mismo punto de su
serie?— es la misma para todas.

Corre **solo sobre dev** (recorta con el holdout congelado, igual que `train.py`) y no
escribe nada: imprime la auditoría. Existe porque la familia E (`src/features/sequence.py`)
dio un resultado negativo que hay que poder re-verificar sin reconstruir el panel, y
porque el bloque 2 descubrió un atajo que aplica a **cualquier** feature acumulativa.

Cuatro bloques:

1. **Separación por fila**: P(fila positiva > fila sana) de cada columna, que es el AUC
   de esa columna sola. 0,5 = no separa. Se listan las features de secuencia junto a las
   de nivel de las que salen, porque la pregunta no es "¿separa?" sino "¿separa más que
   la columna que ya teníamos?".
2. **El atajo de posición**: `_frac` es en qué punto de la serie de cortes de su propio
   vehículo cae la fila. En este panel separa sola con P ≈ 0,83, y no es señal: los
   cortes de un vehículo con evento llegan hasta `E − G` y las positivas son las últimas
   `H/Δ`, así que "qué tan avanzado estoy" *es* la etiqueta. Cualquier columna monótona
   en el índice de corte hereda parte de ese atajo.
3. **Separación dentro de estratos de posición**: si una columna separa por física, tiene
   que seguir separando entre filas que están en el mismo punto de su serie. Si solo
   separa mirando el panel entero, lo que mide es la posición.
4. **Gradiente contra el evento, con referencia comparable**: el agregado de los
   vehículos con evento por tramo de km hasta el evento, contra el de los sanos **en el
   mismo rango de posición**. Sin esa referencia, cualquier acumulador dibuja un
   gradiente perfecto que es solo el paso del tiempo.

El detalle está en `docs/memoria/f3-secuencia-zself-cusum.md` y
`docs/memoria/f3-posicion-en-la-serie.md`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.config import resolve_path  # noqa: E402
from src.eval.splits import load_test_split, test_split_masks  # noqa: E402

# Las cuatro componentes del índice, con el nivel del que sale cada `_zself`.
INDEX_COMPONENTS = [
    "feat_idle_per_1000km",
    "feat_trips_below_regime_temp_frac",
    "feat_speed_kmh_mean",
    "feat_coolant_temp_end_mean",
]


def p_positive_gt_healthy(frame: pd.DataFrame, column: str, *, min_n: int = 10) -> float:
    """P(fila positiva > fila sana). Es el AUC de esa sola columna; 0,5 = no separa."""
    from scipy.stats import mannwhitneyu

    a = frame.loc[frame["label"].eq(1), column].dropna()
    b = frame.loc[frame["label"].eq(0), column].dropna()
    if len(a) < min_n or len(b) < min_n:
        return float("nan")
    u = mannwhitneyu(a, b, alternative="two-sided").statistic
    return float(u / (len(a) * len(b)))


def select_columns(dev: pd.DataFrame, spec: str, seq_cols: list[str]) -> list[str]:
    """Qué columnas se auditan: las de secuencia (default), todas las `feat_*`, o una lista."""
    if spec == "sequence":
        cols = [c for c in INDEX_COMPONENTS if c in dev.columns] + seq_cols
    elif spec == "all":
        cols = [c for c in dev.columns if c.startswith("feat_")]
    else:
        cols = [c.strip() for c in spec.split(",") if c.strip()]
        missing = [c for c in cols if c not in dev.columns]
        if missing:
            raise SystemExit(f"El panel no tiene estas columnas: {missing}")
    numeric = [c for c in cols if pd.api.types.is_numeric_dtype(dev[c])]
    if not numeric:
        raise SystemExit(f"Ninguna columna numérica para auditar (`--columns {spec}`)")
    return numeric


def with_position(dev: pd.DataFrame) -> pd.DataFrame:
    """Agrega la posición del corte dentro de la serie de su vehículo.

    Si el panel trae `aux_cut_position` (calculado antes de muestrear los sanos), esa es
    la posición buena y se usa esa. Recalcularla acá sobre el panel ya muestreado la
    distorsiona: de un vehículo con evento se conservan todos los cortes y de un sano
    solo un subconjunto, así que el rango del sano se comprime y dos filas con el mismo
    `_frac` no están en el mismo punto de sus historias. Los paneles viejos no tienen la
    columna y caen al cálculo de antes, que es aproximado y conservador.
    """
    out = dev.copy()
    grouped = out.groupby("vehicle_id")["cut_odo"]
    out["_n"] = grouped.transform("size")
    if "aux_cut_position" in out.columns:
        out["_frac"] = out["aux_cut_position"]
        out["_rank"] = out["_frac"] * out["_n"]
    else:
        out["_rank"] = grouped.rank(method="first")
        out["_frac"] = out["_rank"] / out["_n"]
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--panel", default="data/processed/panel.parquet")
    parser.add_argument("--test-split", default="data/processed/test_split.json")
    parser.add_argument("--columns", default="sequence",
                        help="`sequence` (las de secuencia + los niveles del índice), `all` "
                             "(todas las feat_*) o una lista separada por comas.")
    parser.add_argument("--top", type=int, default=25,
                        help="Cuántas columnas se imprimen en los bloques 1 y 3, por |0,5 − P|.")
    args = parser.parse_args()

    panel = pd.read_parquet(resolve_path(args.panel))
    split = load_test_split(resolve_path(args.test_split))
    dev_mask, _ = test_split_masks(panel, split, strict=True)
    dev = with_position(panel.loc[dev_mask].reset_index(drop=True))
    print(f"panel: {args.panel}")
    print(f"dev: {len(dev)} filas · {dev['vehicle_id'].nunique()} vehículos · "
          f"{int(dev['label'].sum())} positivas (tasa {dev['label'].mean():.4f})")

    seq_cols = [c for c in dev.columns if c.endswith("_zself") or "degradation" in c]
    if not seq_cols and args.columns == "sequence":
        print("\nEste panel no tiene features de secuencia (`sequence.enabled: false`). "
              "Se audita igual el atajo de posición, que no depende de ellas.")

    audited = select_columns(dev, args.columns, seq_cols)

    # 1 ------------------------------------------------------------------------------
    print("\n== 1 · separación por fila (dev) ==")
    rows = []
    for c in audited:
        p = p_positive_gt_healthy(dev, c)
        rows.append({"columna": c, "P(pos>sano)": p, "|0,5-P|": abs(p - 0.5),
                     "nulos": float(dev[c].isna().mean())})
    table = pd.DataFrame(rows).set_index("columna").sort_values("|0,5-P|", ascending=False)
    print(table.head(args.top).round(3).to_string())
    if len(table) > args.top:
        print(f"  ... {len(table) - args.top} columnas más (se listan las {args.top} que más separan)")

    # 2 ------------------------------------------------------------------------------
    print("\n== 2 · el atajo de posición ==")
    for c in ("_frac", "_rank", "_n"):
        print(f"  {c:8s} P(pos>sano) = {p_positive_gt_healthy(dev, c):.3f}")
    print("  (`_frac` alto es la etiqueta por construcción: las positivas son los "
          "últimos H/Δ cortes de un vehículo cuya serie termina en E − G)")
    flagged = 0
    for c in audited:
        rho = dev[[c, "_rank"]].corr(method="spearman").iloc[0, 1]
        if abs(rho) >= 0.2:
            print(f"  corr({c}, posición) = {rho:.2f}  <-- hereda parte del atajo")
            flagged += 1
    if not flagged:
        print("  Ninguna de las columnas auditadas correlaciona |ρ| >= 0,2 con la posición.")

    # 3 ------------------------------------------------------------------------------
    print("\n== 3 · separación DENTRO de estratos de posición ==")
    dev["_stratum"] = pd.qcut(dev["_rank"], 4, labels=["q1", "q2", "q3", "q4"], duplicates="drop")
    cols = list(table.head(args.top).index)
    strata = []
    for name, sub in dev.groupby("_stratum", observed=True):
        row = {"estrato": name, "n": len(sub), "positivas": int(sub["label"].sum())}
        row.update({c: p_positive_gt_healthy(sub, c) for c in cols})
        strata.append(row)
    # Traspuesta: con `--columns all` son 25 columnas y una fila por estrato no se lee.
    print(pd.DataFrame(strata).set_index("estrato").T.round(3).to_string())
    print("  Una columna que separa por física sostiene el número en los cuatro estratos.")

    # 4 ------------------------------------------------------------------------------
    if seq_cols:
        print("\n== 4 · gradiente al evento, contra sanos en la misma posición ==")
        ev = dev[dev["event_observed"].eq(1) & dev["time_to_event_km"].notna()].copy()
        ev["tramo"] = pd.cut(ev["time_to_event_km"], [0, 2000, 4000, 8000, np.inf],
                             labels=["0-2k", "2-4k", "4-8k", ">8k"])
        healthy = dev[dev["event_observed"].eq(0)]
        target = [c for c in ("feat_degradation_cusum", "feat_degradation_index") if c in dev.columns]
        out = []
        for tramo, sub in ev.groupby("tramo", observed=True):
            lo, hi = sub["_rank"].quantile(0.25), sub["_rank"].quantile(0.75)
            ref = healthy[healthy["_rank"].between(lo, hi)]
            row = {"tramo": tramo, "n": len(sub), "rank_p50": float(sub["_rank"].median())}
            for c in target:
                row[f"evento_{c.split('_')[-1]}"] = float(sub[c].median())
                row[f"sano_{c.split('_')[-1]}"] = float(ref[c].median()) if len(ref) > 20 else np.nan
            out.append(row)
        print(pd.DataFrame(out).set_index("tramo").round(3).to_string())
        print("  Si la columna del evento no supera a la del sano comparable, el gradiente "
              "que se ve sin esta referencia es el paso del tiempo, no la degradación.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
