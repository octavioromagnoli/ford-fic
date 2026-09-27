#!/usr/bin/env python
"""Geocodifica cada ciudad de venta y congela su altura en `configs/data/city_elevation.yaml`.

    python scripts/build_city_elevation.py --config configs/data/city_geocode.yaml

Una consulta por par (país, ciudad) a la API de Open-Meteo (GeoNames). Entre los
resultados del país se elige el lugar poblado de mayor jerarquía y, a igual
jerarquía, el más poblado. Cada entrada del YAML de salida dice qué se consultó, qué
lugar se eligió y con qué estado (`ok`, `alias`, `ambiguous`, `not_found`), para que
una elección dudosa se vea sin volver a correr nada. Los números de la ficha salen
de ese archivo, no de la API.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.config import load_config, resolve_path  # noqa: E402

logger = logging.getLogger("build_city_elevation")


def query(api: dict, name: str, country2: str) -> list[dict]:
    params = {"name": name, "count": api["count"], "language": api["language"], "format": "json",
              "countryCode": country2}
    url = f"{api['url']}?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url, timeout=api["timeout_s"]) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return [r for r in payload.get("results", []) if r.get("country_code") == country2]


def pick(results: list[dict], feature_codes: list[str], elevation_range: list[float]) -> dict | None:
    """El lugar poblado de mayor jerarquía (y más poblado) con una altura plausible.

    GeoNames usa 9999 como "sin dato" (Punta Arenas lo trae): una altura fuera de
    `elevation_range` descarta el candidato en vez de colarse como 9.999 m.
    """
    rank = {code: i for i, code in enumerate(feature_codes)}
    lo, hi = float(elevation_range[0]), float(elevation_range[1])
    candidates = [
        r for r in results
        if r.get("feature_code") in rank and r.get("elevation") is not None and lo <= float(r["elevation"]) <= hi
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda r: (rank[r["feature_code"]], -(r.get("population") or 0)))


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/data/city_geocode.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)

    vehicles = pd.read_parquet(resolve_path(cfg["vehicle_table"]))
    pairs = (
        vehicles[[cfg["country_column"], cfg["city_column"]]].dropna().drop_duplicates()
        .sort_values([cfg["country_column"], cfg["city_column"]]).itertuples(index=False, name=None)
    )
    aliases = cfg.get("aliases") or {}
    ambiguous = cfg.get("ambiguous") or {}
    entries = []
    for country, city in pairs:
        entry = {"country": country, "city": city}
        if city in (ambiguous.get(country) or {}):
            entry.update({"status": "ambiguous", "note": ambiguous[country][city], "elevation_m": None})
            entries.append(entry)
            continue
        name = (aliases.get(country) or {}).get(city, city)
        results = query(cfg["api"], name, cfg["countries"][country])
        time.sleep(float(cfg["api"]["pause_s"]))
        best = pick(results, cfg["feature_codes"], cfg["elevation_range_m"])
        if best is None:
            entry.update({"query": name, "status": "not_found", "elevation_m": None})
        else:
            entry.update(
                {
                    "query": name,
                    "status": "alias" if name != city else "ok",
                    "name": best.get("name"),
                    "admin1": best.get("admin1"),
                    "feature_code": best.get("feature_code"),
                    "population": best.get("population"),
                    "geonames_id": best.get("id"),
                    "latitude": best.get("latitude"),
                    "longitude": best.get("longitude"),
                    "elevation_m": best.get("elevation"),
                    "n_candidates": len(results),
                }
            )
        entries.append(entry)
        logger.info("%s / %s -> %s (%s m, %s)", country, city, entry.get("name"), entry.get("elevation_m"),
                    entry["status"])

    header = (
        "# Altura de la ciudad de venta. GENERADO por scripts/build_city_elevation.py con\n"
        f"# configs/data/city_geocode.yaml el {date.today().isoformat()}; no se edita a mano (se\n"
        "# corrigen los alias del config y se regenera). Fuente: Open-Meteo / GeoNames (CC BY 4.0).\n"
    )
    out = resolve_path(cfg["output"])
    with out.open("w", encoding="utf-8") as fh:
        fh.write(header)
        yaml.safe_dump({"source": "open-meteo geocoding (GeoNames)", "cities": entries}, fh,
                       allow_unicode=True, sort_keys=False, width=120)
    status = pd.Series([e["status"] for e in entries]).value_counts().to_dict()
    print(f"\n{len(entries)} ciudades -> {out} · estados: {status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
