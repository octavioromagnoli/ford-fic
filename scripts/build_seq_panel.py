#!/usr/bin/env python
"""Materializa el panel secuencial: las filas del panel v1 con la ventana en bins de km.

    python scripts/build_seq_panel.py --config configs/data/panel_seq_v1.yaml

Es la entrada de los modelos secuenciales (hoy `cnn_lstm`, el baseline de la tutora).
No decide nada del problema: universo, cortes, etiqueta, gap, horizonte, QC y sanos
emparejados son los del panel v1, que se lee ya construido. Lo único nuevo es cómo se
describe la ventana `(c − L, c]`: en vez de 53 agregados, `T` bins de `b` km con `C`
canales cada uno (`src/features/sequences.py`).

Hace, en este orden:

1. Lee el panel v1 (`base_panel` → `output.panel`) y se queda con sus claves, su
   cabecera, sus `static_*` y, si el YAML lo pide, sus `aux_*`. Las `feat_*` de
   ventana **no** viajan: el modelo recibe la secuencia y las estáticas, nada más.
2. Lee las tablas crudas que piden los canales (`signals` y/o `trips`) para los
   vehículos del panel, canonizadas y sin filas repetidas (`src/data/subset.py`), con
   las mismas derivadas que usa el panel v1.
3. Arma el tensor `(filas, T, C)` y lo aplana a `feat_seq_*`.
4. Escribe el panel y un `_meta.json` con la forma del tensor, que es lo que lee el
   modelo para reconstruirlo.

Mismas filas que el panel v1 ⇒ misma huella ⇒ el `splits.json` del panel v1 sirve sin
regenerarse. Las filas de test se escriben (igual que en el panel v1) y
`scripts/train.py` las recorta antes de la CV.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from scripts.build_dataset import SIGNAL_COLUMNS, TRIP_COLUMNS  # noqa: E402  (mismas columnas que el panel v1)
from src.config import ensure_dir, load_config, repo_root, resolve_path  # noqa: E402
from src.data.subset import read_table_for_vehicles  # noqa: E402
from src.eval.splits import panel_fingerprint  # noqa: E402
from src.features.sequences import (  # noqa: E402
    SEQ_PREFIX,
    bin_source,
    build_sequences,
    derive_signal_sequence_columns,
    flatten_sequences,
    load_channel_specs,
    sequence_column_names,
)
from src.features.signals import derive_signal_columns  # noqa: E402
from src.features.trips import derive_trip_columns  # noqa: E402

logger = logging.getLogger("build_seq_panel")

ID = "vehicle_id"
KEY = [ID, "cut_odo"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/data/panel_seq_v1.yaml")
    return parser.parse_args()


def load_source(name: str, vehicles: set[str], base_cfg: dict, spec_cfg: dict, chunksize: int) -> pd.DataFrame:
    """Tabla cruda + derivadas, con el mismo recorte de columnas que `build_dataset.py`."""
    # Fila completa para leer (el dedupe exacto la necesita); se recorta después.
    frame, _ = read_table_for_vehicles(name, None, vehicles, sources=base_cfg["sources"],
                                       dedupe=base_cfg["dedupe"], chunksize=chunksize)
    thresholds = spec_cfg.get("thresholds") or {}
    if name == "signals":
        keep = {ID, *SIGNAL_COLUMNS, "eventTimestamp", "Acumulation"}
        frame, _ = derive_signal_columns(frame[[c for c in frame.columns if c in keep]])
        return derive_signal_sequence_columns(frame, regen_drop_points=float(thresholds.get("regen_drop_points", 5.0)))
    frame = frame[[c for c in frame.columns if c in {ID, *TRIP_COLUMNS}]]
    frame, _ = derive_trip_columns(frame, thresholds=thresholds, clip=spec_cfg.get("clip"))
    return frame


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    args = parse_args()
    cfg = load_config(args.config)
    base_cfg = load_config(cfg["base_panel"])
    spec_cfg = load_config(base_cfg["features"]["spec"])
    seq_cfg = cfg["sequence"]
    channels = load_channel_specs(seq_cfg["channels"])
    lookback_km = float(seq_cfg["lookback_km"])
    bin_km = float(seq_cfg["bin_km"])
    chunksize = int((base_cfg.get("scan") or {}).get("chunksize", 500_000))

    # 1 · filas del panel v1 --------------------------------------------------------
    base_path = resolve_path(base_cfg["output"]["panel"])
    base = pd.read_parquet(base_path)
    if base.duplicated(KEY).any():
        raise ValueError(f"{base_path.name} tiene claves (vehicle_id, cut_odo) repetidas")
    window_km = float(base["window_km"].iloc[0])
    if lookback_km > window_km:
        # Con L > W los cortes tempranos tienen bins antes del primer registro, y cuántos
        # hay es una función del odómetro del corte: el atajo que el emparejado cierra.
        logger.warning("lookback_km=%g > W=%g: los bins previos al primer registro codifican el odómetro",
                       lookback_km, window_km)
    keep_cols = [c for c in base.columns if not c.startswith("feat_")]
    if not cfg.get("keep_aux", True):
        keep_cols = [c for c in keep_cols if not c.startswith("aux_")]
    vehicles = set(base[ID].astype(str))
    logger.info("Panel base: %s | %d filas | %d vehículos", base_path.name, len(base), len(vehicles))

    # 2 · crudos + bins por tabla ---------------------------------------------------
    binned_parts = []
    for source in sorted({c.source for c in channels}):
        frame = load_source(source, vehicles, base_cfg, spec_cfg, chunksize)
        binned_parts.append(bin_source(frame, source=source, bin_km=bin_km, channels=channels))
        del frame
    binned = pd.concat(binned_parts, axis=1) if len(binned_parts) > 1 else binned_parts[0]

    # 3 · tensor y panel ------------------------------------------------------------
    tensor, coverage = build_sequences(base[KEY], binned, channels, lookback_km=lookback_km, bin_km=bin_km)
    flat = flatten_sequences(tensor, channels, index=base.index)
    panel = pd.concat([base[keep_cols], flat], axis=1)

    # Lo que el modelo da por sentado al reconstruir el tensor: las únicas `feat_*` son
    # la secuencia y, ordenadas como las ordena el panel, quedan en orden (t, canal).
    feat_cols = [c for c in panel.columns if c.startswith("feat_")]
    expected = sequence_column_names(coverage["seq_len"], channels)
    if feat_cols != expected or sorted(feat_cols) != expected:
        raise AssertionError("Las `feat_*` del panel secuencial no son exactamente la secuencia en orden (t, canal)")
    if panel_fingerprint(panel) != panel_fingerprint(base):
        raise AssertionError("El panel secuencial cambió el set de filas/vehículos del panel v1")

    out_path = resolve_path(cfg["output"]["panel"])
    ensure_dir(out_path.parent)
    panel.to_parquet(out_path, index=False)

    meta = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": cfg.get("_config_path"),
        "base_panel": {"config": base_cfg.get("_config_path"), "path": str(base_cfg["output"]["panel"]),
                       "fingerprint": panel_fingerprint(base)},
        "lookback_km": lookback_km,
        "bin_km": bin_km,
        "seq_len": coverage["seq_len"],
        "n_channels": coverage["n_channels"],
        "channels": [c.__dict__ for c in channels],
        "seq_prefix": SEQ_PREFIX,
        "columns": expected,
        "coverage": coverage,
    }
    meta_path = resolve_path(cfg["output"]["meta"])
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n== Panel secuencial ==")
    print(f"base           : {base_path.name} ({len(base)} filas, {len(vehicles)} vehículos) · W={window_km:g} km")
    print(f"secuencia      : L={lookback_km:g} km en T={coverage['seq_len']} bins de {bin_km:g} km · "
          f"C={coverage['n_channels']} canales ({', '.join(c.name for c in channels)})")
    print(f"cobertura      : {coverage['empty_bin_frac']:.1%} de bins sin filas · "
          f"{coverage['rows_all_bins_empty']} filas con la secuencia vacía · "
          f"{coverage['rows_with_nan_after_fill']} con NaN tras el relleno (los imputa el fold)")
    print(f"columnas       : {len(expected)} feat_seq_ + "
          f"{sum(c.startswith('static_') for c in panel.columns)} static_ + "
          f"{sum(c.startswith('aux_') for c in panel.columns)} aux_")
    print(f"escrito        : {_rel(out_path)} · {meta_path.name}")
    return 0


def _rel(path: Path) -> Path:
    return path.relative_to(repo_root()) if path.is_relative_to(repo_root()) else path


if __name__ == "__main__":
    raise SystemExit(main())
