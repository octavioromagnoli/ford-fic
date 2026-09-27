#!/usr/bin/env python
"""Pega a un panel las `aux_` de otro panel con las mismas filas, y verifica que lo sean.

    FORD_DATA_DIR=$PWD/data/rebuild-0921 python scripts/join_window_columns.py --config configs/data/panel_seq_kmw.yaml

Lo usa la demo de la GRU (F9): la GRU lee `panel_seq_v1`, que no trae la etiqueta corregida por la ventana del
registro (`aux_eval_in_window`); esa columna la armó `build_km_window_panel.py` sobre el panel de agregados. No se
recalcula nada: se copia por `(vehicle_id, cut_odo)`, y antes de escribir se verifica, fallando si no se cumple:

- las dos claves son únicas y los dos paneles tienen el mismo conjunto de filas (el merge es 1:1 y no pierde ni
  agrega ninguna);
- las columnas de `same` coinciden fila a fila (etiqueta, evento, horizonte, fecha, mercado);
- las filas de dev son las mismas en los dos paneles (`test_split_masks`) y su huella es la de `splits`, así que
  los folds congelados sirven tal cual;
- solo se agregan `aux_`, y las `feat_*`/`static_*` de la salida son las de la base en el mismo orden: el modelo
  recibe exactamente la misma entrada.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.config import ensure_dir, load_config, repo_root, resolve_path  # noqa: E402
from src.eval.splits import load_test_split, panel_fingerprint, test_split_masks  # noqa: E402
from src.training.cv import select_feature_columns  # noqa: E402

logger = logging.getLogger("join_window_columns")

KEY = ["vehicle_id", "cut_odo"]


def _same(a: pd.Series, b: pd.Series) -> np.ndarray:
    """Igualdad fila a fila con NaN = NaN (y NaT = NaT)."""
    return (a.to_numpy() == b.to_numpy()) | (a.isna().to_numpy() & b.isna().to_numpy())


def join_by_key(base: pd.DataFrame, source: pd.DataFrame, columns: Sequence[str],
                same: Sequence[str] = ()) -> pd.DataFrame:
    """`base` con las `columns` de `source` al final, por `(vehicle_id, cut_odo)`. Falla si las filas no son las
    mismas, si alguna columna de `same` difiere, si se pide algo que no es `aux_` o si la entrada del modelo cambia."""
    columns, same = list(columns), list(same)
    bad = [c for c in columns if not c.startswith("aux_")]
    if bad:
        raise ValueError(f"Solo se pegan columnas aux_ (fuera del modelo): {bad}")
    clash = [c for c in columns if c in base.columns]
    if clash:
        raise ValueError(f"La base ya trae {clash}")
    missing = [c for c in [*columns, *same] if c not in source.columns] + [c for c in same if c not in base.columns]
    if missing:
        raise KeyError(f"Faltan columnas: {missing}")
    for name, frame in (("base", base), ("fuente", source)):
        if frame.duplicated(KEY).any():
            raise ValueError(f"La {name} tiene pares {KEY} repetidos")
    if len(base) != len(source):
        raise ValueError(f"La base tiene {len(base)} filas y la fuente {len(source)}: no son las mismas filas")

    right = source[KEY + columns + same].rename(columns={c: f"__src__{c}" for c in same})
    out = base.merge(right, on=KEY, how="left", validate="one_to_one", indicator=True)
    unmatched = int(out["_merge"].ne("both").sum())
    if unmatched:
        raise ValueError(f"{unmatched} fila(s) de la base sin pareja en la fuente: no son las mismas filas")
    for c in same:
        ok = _same(out[c], out[f"__src__{c}"])
        if not ok.all():
            raise ValueError(f"`{c}` no coincide en {int((~ok).sum())} fila(s) entre la base y la fuente")
    out = out[list(base.columns) + columns]
    if select_feature_columns(out) != select_feature_columns(base):
        raise RuntimeError("Las feat_/static_ de la salida no son las de la base: cambiaría la entrada del modelo")
    return out


def _sha16(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _rel(path: Path) -> str:
    return str(path.relative_to(repo_root())) if path.is_relative_to(repo_root()) else str(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/data/panel_seq_kmw.yaml")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    cfg = load_config(args.config)
    base_path, source_path = resolve_path(cfg["base"]), resolve_path(cfg["source"])
    base, source = pd.read_parquet(base_path), pd.read_parquet(source_path)
    out = join_by_key(base, source, cfg["columns"], cfg.get("same") or [])

    # Mismas filas de dev en los dos paneles, y con la huella de los folds congelados.
    split = load_test_split(cfg["test_split"])
    dev_out = out.loc[test_split_masks(out, split)[0]]
    dev_src = source.loc[test_split_masks(source, split)[0]]
    if not dev_out[KEY].sort_values(KEY).reset_index(drop=True).equals(
            dev_src[KEY].sort_values(KEY).reset_index(drop=True)):
        raise ValueError("Las filas de dev no son las mismas en la base y en la fuente")
    splits = json.loads(resolve_path(cfg["splits"]).read_text(encoding="utf-8"))
    fingerprint = panel_fingerprint(dev_out)
    if fingerprint != splits["panel"]:
        raise ValueError(f"La huella de dev {fingerprint} no es la de {Path(cfg['splits']).name}: {splits['panel']}")

    evaluable = dev_out["aux_eval_in_window"].eq(1) if "aux_eval_in_window" in out else pd.Series(dtype=bool)
    out_path = resolve_path(cfg["output"]["panel"])
    ensure_dir(out_path.parent)
    out.to_parquet(out_path, index=False)
    meta: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": cfg["_config_path"],
        "base": {"path": _rel(base_path), "sha256_16": _sha16(base_path)},
        "source": {"path": _rel(source_path), "sha256_16": _sha16(source_path)},
        "columns": list(cfg["columns"]),
        "verified": {
            "rows": int(len(out)), "vehicles": int(out["vehicle_id"].nunique()),
            "same_rows_and_key": True, "same_columns": list(cfg.get("same") or []),
            "model_columns_unchanged": len(select_feature_columns(out)),
            "dev_rows_equal": True, "dev_fingerprint": fingerprint, "splits": _rel(resolve_path(cfg["splits"])),
        },
        "dev": {"rows": int(len(dev_out)), "vehicles": int(dev_out["vehicle_id"].nunique()),
                "positives": int(dev_out["label"].sum()),
                "evaluable_rows": int(evaluable.sum()),
                "evaluable_vehicles": int(dev_out.loc[evaluable, "vehicle_id"].nunique()) if len(evaluable) else 0},
    }
    meta_path = resolve_path(cfg["output"]["meta"])
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n== Panel con las aux_ de la ventana ==")
    print(f"base           : {base_path.name} ({len(base)} filas) + {len(cfg['columns'])} aux_ de {source_path.name}")
    print(f"verificado     : mismas filas y clave · {cfg.get('same')} iguales fila a fila · "
          f"{meta['verified']['model_columns_unchanged']} columnas del modelo sin cambios")
    print(f"dev            : {meta['dev']['rows']} filas · {meta['dev']['vehicles']} vehículos · "
          f"{meta['dev']['positives']} positivas · huella = {Path(cfg['splits']).name}")
    print(f"etiqueta V     : {meta['dev']['evaluable_rows']} filas evaluables · {meta['dev']['evaluable_vehicles']} vehículos")
    print(f"escrito        : {_rel(out_path)} · {meta_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
