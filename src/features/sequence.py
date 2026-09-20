"""Features de secuencia: el vehículo contra sí mismo, y la evidencia acumulada.

Todo lo que hay acá se calcula **sobre el panel ya construido**, agrupando por
`vehicle_id` y recorriendo los cortes en orden de `cut_odo`. Dos motivos por los que
esto es una familia aparte y no un agregador más de `src/features/windows.py`:

* `windows.py` resume *una* ventana `(c − W, c]`. Acá la unidad es la **serie de
  cortes** de un vehículo, que solo existe cuando el panel ya está armado.
* El EDA mostró que la señal es **progresiva** —`idle_frac` pasa de P = 0,58 a 0,74 a
  medida que el corte se acerca al evento— y que las features que anticipan son
  justamente las de ICC bajo, las que varían *dentro* del vehículo
  (`below_regime_frac` 0,31). Un modelo de nivel confunde "vehículo urbano" con
  "vehículo que se está degradando"; el desvío contra su propio historial separa las
  dos cosas (`docs/memoria/f2-eda-revision-y-features.md` §3.2 y §3.6).

## Hacia atrás, siempre (regla 3)

Cada estadístico de un corte `t` usa **solo los cortes anteriores del mismo
vehículo**: `expanding().shift(1)`. Nunca el corte actual, nunca los siguientes, nunca
otros vehículos. Por eso estas columnas no necesitan fitearse por fold: no hay ningún
estadístico compartido entre vehículos que pueda filtrar del train a la validación.

El precio es que los primeros cortes de cada vehículo salen NaN (no hay historial con
qué compararse). Se dejan NaN a propósito: el imputador del `Pipeline` los resuelve
con la mediana del train del fold, y fabricar un cero sería afirmar "no se desvía".

## Qué produce

- `feat_<x>_zself` — desvío estandarizado de la feature contra su propio pasado.
- `feat_degradation_index` — media de los `zself` de las features del índice, con el
  signo de cada una puesto por la física (`+1` = más riesgo cuando sube).
- `feat_degradation_cusum` — CUSUM unilateral del índice: la evidencia acumulada.
- `feat_degradation_run` — cuántos cortes seguidos viene el índice por encima de `k`.

Las tres últimas contestan la pregunta operativa: un corte aislado con el índice en
+0,3 no significa nada, pero diez cortes seguidos empujando en la misma dirección sí.
El CUSUM es la forma estándar de decir eso con dos parámetros (`k`, `h`) en vez de un
umbral inventado por corte.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ID_COL = "vehicle_id"
ORDER_COL = "cut_odo"
FEATURE_PREFIX = "feat_"
ZSELF_SUFFIX = "_zself"

INDEX_NAME = "degradation"


def _ordered_groups(panel: pd.DataFrame) -> pd.DataFrame:
    """Índice original del panel, ordenado por vehículo y corte. No reordena el panel."""
    missing = [c for c in (ID_COL, ORDER_COL) if c not in panel.columns]
    if missing:
        raise ValueError(f"El panel no tiene {missing}: no se puede armar la secuencia por vehículo")
    return panel.sort_values([ID_COL, ORDER_COL], kind="mergesort")


def self_deviation(
    panel: pd.DataFrame,
    columns: Iterable[str],
    *,
    min_history: int = 3,
    min_std: float = 1e-9,
    robust: bool = True,
) -> pd.DataFrame:
    """Desvío estandarizado del corte contra el pasado del mismo vehículo.

    `min_history` es cuántos cortes previos hace falta tener para que el desvío
    signifique algo: con 2 puntos el desvío muestral es una diferencia disfrazada y el
    z-score explota. Los cortes que no llegan salen NaN.

    Un vehículo con la feature constante en todo su pasado (escala 0) también sale NaN:
    dividir por cero daría ±inf, y "no varió nunca" no es lo mismo que "se desvió mucho".

    **`robust=True` (default): mediana e IQR/1,349 en vez de media y desvío.** No es un
    gusto, es un arreglo medido: las features de este panel son asimétricas
    —`idle_per_1000km` tiene skew 3,47 y el 72% de sus valores cae por debajo de su
    propia media— así que la z clásica tiene mediana **−0,34** en vez de 0. Con esa
    base, un CUSUM con holgura `k > 0` no acumula nunca salvo por un outlier: en el
    panel Δ=500 solo el 18% de las filas llegaba a `S > 0`. Con la versión robusta la
    mediana vuelve a ≈ 0 (−0,07) y el acumulador mide tendencia en vez de cola.
    """
    columns = [c for c in columns if c in panel.columns]
    if not columns:
        raise ValueError("Ninguna de las columnas pedidas para `zself` está en el panel")

    ordered = _ordered_groups(panel)
    grouped = ordered.groupby(ID_COL, observed=True, sort=False)[columns]
    # shift(1): el corte actual NO entra en su propia referencia (regla 3).
    past = grouped.shift(1)
    expanding = past.groupby(ordered[ID_COL].to_numpy(), sort=False).expanding()
    if robust:
        centre = expanding.median().reset_index(level=0, drop=True)
        q1 = expanding.quantile(0.25).reset_index(level=0, drop=True)
        q3 = expanding.quantile(0.75).reset_index(level=0, drop=True)
        scale = (q3 - q1) / 1.349          # IQR -> desvío equivalente de una normal
    else:
        centre = expanding.mean().reset_index(level=0, drop=True)
        scale = expanding.std().reset_index(level=0, drop=True)
    count = expanding.count().reset_index(level=0, drop=True)

    z = (ordered[columns] - centre) / scale.where(scale > min_std)
    z = z.where(count >= min_history)
    z = z.replace([np.inf, -np.inf], np.nan)
    z.columns = [f"{c}{ZSELF_SUFFIX}" for c in columns]
    return z.reindex(panel.index)


def degradation_index(
    zself: pd.DataFrame,
    weights: Mapping[str, float],
    *,
    min_components: int = 2,
) -> pd.Series:
    """Media de los `zself` del índice, con el signo físico de cada componente.

    El peso es el **sentido**, no una importancia aprendida: `+1` en las features que
    suben cuando el vehículo se degrada (idle, viajes que no llegan a régimen) y `−1`
    en las que bajan (velocidad, temperatura de refrigerante). Es la misma codificación
    que las `monotone_constraints` del LGBM, y vale lo mismo: regularización que no
    cuesta datos.

    Se promedia ignorando los NaN pero exigiendo `min_components` componentes vivos:
    un índice armado con una sola feature es esa feature, no un índice.
    """
    present = {f"{c}{ZSELF_SUFFIX}": float(w) for c, w in weights.items()
               if f"{c}{ZSELF_SUFFIX}" in zself.columns}
    if len(present) < min_components:
        raise ValueError(
            f"El índice necesita al menos {min_components} componentes presentes en el panel; "
            f"hay {len(present)} de {len(weights)}. Revisá `sequence.index.weights`."
        )
    signed = zself[list(present)].mul(pd.Series(present), axis=1)
    n_alive = signed.notna().sum(axis=1)
    return signed.mean(axis=1, skipna=True).where(n_alive >= min_components)


def cusum(
    panel: pd.DataFrame,
    index: pd.Series,
    *,
    k: float = 0.5,
) -> pd.DataFrame:
    """CUSUM unilateral del índice por vehículo: `S_t = max(0, S_{t−1} + (idx_t − k))`.

    `k` es la holgura: cuánto desvío se considera ruido normal y no se acumula. Con
    `k = 0,5` hacen falta ~4 cortes consecutivos a +0,75 para llegar a 1,0 de evidencia
    acumulada, y un corte aislado a +0,3 no mueve nada. Ese es exactamente el
    comportamiento que se quiere: más historial coherente, más confianza.

    Devuelve además `run`, la racha de cortes consecutivos con el índice por encima de
    `k`. Es la versión legible del mismo hecho ("viene cuatro ventanas empujando"), y
    la que se cuenta en la demo.

    Los NaN del índice (cortes sin historial) no cortan la racha ni suman evidencia:
    se tratan como "no hay medición", que es lo que son.
    """
    ordered = _ordered_groups(panel)
    idx = index.reindex(ordered.index)
    vehicles = ordered[ID_COL].to_numpy()

    values = idx.to_numpy(dtype=float)
    n = len(values)
    s_out = np.full(n, np.nan)
    run_out = np.full(n, np.nan)
    s = 0.0
    run = 0.0
    prev = None
    for i in range(n):
        if vehicles[i] != prev:
            s, run, prev = 0.0, 0.0, vehicles[i]
        v = values[i]
        if not np.isnan(v):
            s = max(0.0, s + (v - k))
            run = run + 1.0 if v > k else 0.0
        s_out[i] = s
        run_out[i] = run

    out = pd.DataFrame({f"{FEATURE_PREFIX}{INDEX_NAME}_cusum": s_out,
                        f"{FEATURE_PREFIX}{INDEX_NAME}_run": run_out},
                       index=ordered.index)
    return out.reindex(panel.index)


def add_sequence_features(panel: pd.DataFrame, cfg: Mapping[str, Any] | None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Agrega las columnas de secuencia al panel. Sin `cfg`, devuelve el panel intacto.

    Se llama **entre** `build_panel()` y `match_healthy_cuts()`: la serie de un vehículo
    tiene que estar completa cuando se calcula el desvío contra su propio pasado. Si se
    calculara después del emparejado, el "historial previo" de un corte sería el que
    sobrevivió al muestreo de sanos, que es una serie con agujeros y distinta para cada
    configuración de `sampling`.
    """
    if not cfg or not cfg.get("enabled", False):
        return panel, {"enabled": False}

    min_history = int(cfg.get("min_history", 3))
    index_cfg = dict(cfg.get("index") or {})
    weights: dict[str, float] = {str(k): float(v) for k, v in (index_cfg.get("weights") or {}).items()}
    if not weights:
        raise ValueError("`sequence.index.weights` está vacío: el índice no tiene componentes")

    declared = list(cfg.get("zself_columns") or [])
    columns = declared or sorted(weights)
    unknown = [c for c in columns if c not in panel.columns]
    if unknown:
        raise ValueError(
            f"`sequence` pide columnas que el panel no tiene: {unknown}. "
            "Los nombres son los del panel (`feat_*`), no los de `features_v1.yaml`."
        )

    robust = bool(cfg.get("robust", True))
    z = self_deviation(panel, columns, min_history=min_history, robust=robust)
    idx = degradation_index(z, weights, min_components=int(index_cfg.get("min_components", 2)))
    acc = cusum(panel, idx, k=float(cfg.get("k", 0.5)))

    out = panel.copy()
    for name, values in z.items():
        out[name] = values
    out[f"{FEATURE_PREFIX}{INDEX_NAME}_index"] = idx
    for name, values in acc.items():
        out[name] = values

    summary = {
        "enabled": True,
        "min_history": min_history,
        "robust": robust,
        "k": float(cfg.get("k", 0.5)),
        "zself_columns": columns,
        "index_weights": weights,
        "n_columns_added": int(len(z.columns) + 3),
        "index_null_frac": float(idx.isna().mean()),
        "rows_with_run_ge_3": int((out[f"{FEATURE_PREFIX}{INDEX_NAME}_run"] >= 3).sum()),
    }
    logger.info("Secuencia: +%d columnas · índice nulo en el %.1f%% de las filas (cortes sin historial)",
                summary["n_columns_added"], 100 * summary["index_null_frac"])
    return out, summary
