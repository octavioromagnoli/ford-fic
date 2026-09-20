"""TimesFM zero-shot sobre las series por km de cada vehículo, en los cortes del panel v1.

TimesFM es un forecaster de series, no un clasificador: no entra por `registry.py`
porque a los modelos de ahí les llega una fila tabular de `feat_*` por corte, y lo
que TimesFM necesita es la historia del vehículo. Este módulo arma esa historia y
convierte el pronóstico en un score y en resúmenes por corte:

1. **Series por km** (`build_km_series`). Bins de `bin_km` sobre el odómetro, con
   tres canales del filtro, todos desde las tablas que ya deriva el panel:

   - `accumulation`: nivel medio de `AirRegenerationEnd` de los viajes que terminan
     en el bin (`trips`; es la misma variable que `signals.Acumulation`, con el
     odómetro completo);
   - `regen_count`: regeneraciones del bin, contadas como caídas del nivel
     (`regen_drop` de `src/features/trips.py`);
   - `bad_msg_frac`: fracción de mensajes "malos" de `signals`, ubicados por
     `OdometerValue` como en el panel.

2. **Cortes: los del panel v1** (`src/data/panel.py`). Etiqueta, gap, horizonte, QC
   de ventana, universo y emparejado de sanos por odómetro **y mes** los decide el
   panel, no este módulo: así el número es comparable 1:1 con `train.py`.

3. **Score** (`score_cuts`). Un cuantil del pronóstico agregado solo sobre `[G, G+H]`:
   lo que cae dentro del gap no cuenta (regla 1). El contexto son los bins
   enteramente anteriores al corte (regla 3).

Qué cambió respecto de la primera versión, por el EDA de F2
(`docs/memoria/f3-timesfm-zeroshot.md`): el canal de regeneraciones contaba el
marcador `signals.Regenerations`, que se corta el 25-05-2026 y mide el calendario;
los cortes se armaban con el universo de 1081 (incluidos los mercados sin eventos
observables y los positivos con fecha de evento = fecha de venta) y los sanos se
emparejaban solo por odómetro, que deja el calendario como atajo.

No se entrena nada, así que no hay nada que ajustar por fold. Los folds se usan
solo para dar un intervalo a las métricas.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ID_COL = "vehicle_id"
CHANNELS = ("accumulation", "regen_count", "bad_msg_frac")
QUANTILE_LEVELS = tuple(round(0.1 * i, 1) for i in range(1, 10))  # salida de TimesFM 3.0


# --------------------------------------------------------------------------- #
# Series por km
# --------------------------------------------------------------------------- #
def build_km_series(trips: pd.DataFrame, signals: pd.DataFrame, *, bin_km: float,
                    bad_slugs: Sequence[str]) -> pd.DataFrame:
    """Una fila por `(vehicle_id, bin)` con los canales de `CHANNELS`, sin huecos.

    `trips` y `signals` son las tablas de `derive_trip_columns` / `derive_signal_columns`.
    El bin `i` cubre `[i·bin_km, (i+1)·bin_km)`; un viaje cae en el bin de su
    `OdometerTripEnd` y una señal en el de su `OdometerValue`. La serie de cada
    vehículo empieza en su primer bin con viajes: no se inventa historia previa.

    Un bin sin viajes (o sin señales, para `bad_msg_frac`) no es "cero": es falta de
    dato, y arrastra el último valor observado. El relleno es **solo hacia adelante**:
    interpolar metería en el contexto de un corte un valor posterior al corte (regla 3).
    """
    t = trips[[ID_COL, "OdometerTripEnd", "AirRegenerationEnd", "regen_drop"]].copy()
    t["bin"] = np.floor(t["OdometerTripEnd"].to_numpy(dtype=float) / bin_km).astype(int)
    by_trip = t.groupby([ID_COL, "bin"], observed=True).agg(
        accumulation=("AirRegenerationEnd", "mean"),
        regen_count=("regen_drop", "sum"),
        n_trips=("regen_drop", "size"),
    )
    s = signals[[ID_COL, "OdometerValue"]].copy()
    s["bad"] = signals[[f"msg_{slug}" for slug in bad_slugs]].any(axis=1).to_numpy()
    s["bin"] = np.floor(s["OdometerValue"].to_numpy(dtype=float) / bin_km).astype(int)
    by_signal = s.groupby([ID_COL, "bin"], observed=True).agg(bad_msg_frac=("bad", "mean"),
                                                              n_signals=("bad", "size"))
    agg = by_trip.join(by_signal, how="outer")

    series = []
    for vid, g in agg.groupby(level=0, sort=True, observed=True):
        g = g.droplevel(0)
        with_trips = g.index[g["n_trips"].fillna(0) > 0]
        if with_trips.empty:
            continue
        full = pd.RangeIndex(int(with_trips.min()), int(g.index.max()) + 1, name="bin")
        v = g.reindex(full)
        # Sin viajes en el bin: el conteo de regeneraciones es falta de dato, no cero.
        v.loc[v["n_trips"].fillna(0) == 0, "regen_count"] = np.nan
        v[["n_trips", "n_signals"]] = v[["n_trips", "n_signals"]].fillna(0).astype(int)
        v[list(CHANNELS)] = v[list(CHANNELS)].ffill()
        v["bad_msg_frac"] = v["bad_msg_frac"].fillna(0.0)  # antes de la primera señal: sin mensajes malos
        v.insert(0, ID_COL, str(vid))
        series.append(v.reset_index())
    out = pd.concat(series, ignore_index=True)
    logger.info("Series: %d vehículos, %d bins de %.0f km (%.1f%% sin viajes, relleno causal)",
                out[ID_COL].nunique(), len(out), bin_km, 100 * (out["n_trips"] == 0).mean())
    return out


# --------------------------------------------------------------------------- #
# Forecaster y scores
# --------------------------------------------------------------------------- #
def load_forecaster(params: dict[str, Any]):
    """TimesFM 3.0 con backend MLX (Apple silicon) o torch. Import adentro: dependencia opcional."""
    backend = params.get("backend", "mlx")
    checkpoint = params.get("checkpoint", "google/timesfm-3.0-pytorch")
    batch = int(params.get("batch_size", 64))
    if backend == "mlx":
        from timesfm3.mlx import TimesFM3Forecaster

        return TimesFM3Forecaster.from_pretrained(checkpoint, per_core_batch_size=batch)
    if backend == "torch":
        from timesfm3 import ModelConfig, TimesFM3Evaluator

        return TimesFM3Evaluator(
            ModelConfig(checkpoint_path=checkpoint, per_core_batch_size=batch, device=params.get("device", "cpu"))
        )
    raise ValueError(f"backend `{backend}` no soportado (mlx | torch)")


def _aggregate(values: np.ndarray, how: str) -> np.ndarray:
    return {"sum": np.sum, "mean": np.mean, "max": np.max}[how](values, axis=-1)


def _bins(km: float, bin_km: float, name: str) -> int:
    if km % bin_km:
        raise ValueError(f"`{name}`={km} tiene que ser múltiplo de `bin_km`={bin_km}")
    return int(round(km / bin_km))


def score_cuts(
    cuts: pd.DataFrame,
    series: pd.DataFrame,
    forecaster,
    *,
    bin_km: float,
    window_km: float,
    gap_km: float,
    horizon_km: float,
    target: str,
    quantile: float,
    agg: str,
    max_context_bins: int,
    batch_size: int,
    forecast_kwargs: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """`cuts` (clave `vehicle_id`, `cut_odo`) con el score de TimesFM, el naive y los resúmenes.

    * `score`: cuantil `quantile` del pronóstico, agregado con `agg` sobre `[G, G+H]`.
    * `score_naive`: persistencia, "lo que pasó en los últimos W km se repite". Es el
      piso que importa: si el pronóstico no le gana, TimesFM no aporta nada.
    * `ctx_mean_w`, `ctx_slope_w`: la serie mirada hacia atrás (media en W, pendiente
      en W por 1.000 km).
    * `fc_q10`, `fc_q50`, `fc_q90`: media de cada cuantil pronosticado en `[G, G+H]`;
      `fc_q90_max`: el peor bin del cuantil alto.

    Un corte sin contexto (vehículo sin serie) sale en NaN.
    """
    if target not in CHANNELS:
        raise ValueError(f"target `{target}` no es un canal: {CHANNELS}")
    if round(quantile, 1) not in QUANTILE_LEVELS:
        raise ValueError(f"quantile={quantile}: TimesFM 3.0 devuelve deciles {QUANTILE_LEVELS}")
    q_idx = QUANTILE_LEVELS.index(round(quantile, 1))
    w = _bins(window_km, bin_km, "window_km")
    g0 = _bins(gap_km, bin_km, "gap_km")
    g1 = g0 + _bins(horizon_km, bin_km, "horizon_km")

    values = {vid: (int(g["bin"].iloc[0]), g[target].to_numpy(dtype=np.float32))
              for vid, g in series.groupby(ID_COL, sort=False)}

    n = len(cuts)
    rows, contexts = [], []
    naive, ctx_mean_w, ctx_slope_w = (np.full(n, np.nan) for _ in range(3))
    for i, (vid, cut) in enumerate(zip(cuts[ID_COL].astype(str).to_numpy(), cuts["cut_odo"].to_numpy())):
        if vid not in values:
            continue
        first_bin, arr = values[vid]
        # Bins < cut_bin terminan antes del corte: nada posterior al corte entra (regla 3).
        end = _bins(float(cut), bin_km, "cut_odo") - first_bin
        ctx = arr[max(0, end - max_context_bins):max(end, 0)]
        ctx = ctx[~np.isnan(ctx)]
        if ctx.size == 0:
            continue
        rows.append(i)
        contexts.append(ctx)
        recent = ctx[-w:]
        ctx_mean_w[i] = recent.mean()
        if recent.size >= 2:
            ctx_slope_w[i] = np.polyfit(np.arange(recent.size) * bin_km / 1000.0, recent, 1)[0]
        naive[i] = _aggregate(np.full((1, g1 - g0), ctx_mean_w[i]), agg)[0]

    summaries = {k: np.full(n, np.nan) for k in ("score", "fc_q10", "fc_q50", "fc_q90", "fc_q90_max")}
    kwargs = {"return_quantiles": True, **(forecast_kwargs or {})}
    for start in range(0, len(rows), batch_size):
        outs = forecaster.predict_batch(contexts[start:start + batch_size], horizon=g1, **kwargs)
        for i, out in zip(rows[start:start + batch_size], outs):
            window = out.quantiles[g0:g1]  # (H en bins, 9): solo [G, G+H], el gap no cuenta
            summaries["score"][i] = _aggregate(window[None, :, q_idx], agg)[0]
            summaries["fc_q10"][i] = window[:, 0].mean()
            summaries["fc_q50"][i] = window[:, 4].mean()
            summaries["fc_q90"][i] = window[:, 8].mean()
            summaries["fc_q90_max"][i] = window[:, 8].max()
        if (start // batch_size) % 10 == 0:
            logger.info("TimesFM: %d / %d cortes", min(start + batch_size, len(rows)), len(rows))

    out = cuts[[ID_COL, "cut_odo"]].copy()
    out["score_naive"] = naive
    out["ctx_mean_w"] = ctx_mean_w
    out["ctx_slope_w"] = ctx_slope_w
    for key, arr in summaries.items():
        out[key] = arr
    logger.info("Cortes pronosticados: %d de %d", len(rows), n)
    return out
