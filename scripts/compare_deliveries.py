#!/usr/bin/env python
"""Compara la entrega 1 de Ford con la v2, archivo por archivo y vehículo por vehículo.

    python scripts/compare_deliveries.py --config configs/data/compare_deliveries.yaml

Seis secciones, cada una con su CSV en `output_dir` y sus números en `summary.json`:

1. **Archivos**: hash, filas, columnas. Qué cambió y qué es byte a byte igual.
2. **Estática**: filas y códigos por cohorte, columnas nuevas o renombradas, códigos
   repetidos dentro de un archivo, el mapa de países, y qué le pasó a cada código
   (fallado → fallado, fallado → ausente, sano → fallado...).
3. **Anclaje**: `primer viaje − ProductionDay` por cohorte, con y sin el corrimiento
   de −538 días que declara `raw_sources.yaml` para los fallados.
4. **Etiquetas**: cómo se movieron las fechas de los fallados que siguen, qué eran los
   que desaparecieron (fecha por defecto, mercado, período de producción).
5. **Telemetría y producción**: si la telemetría vieja de los fallados está dentro de
   la nueva, desde cuándo viene la de los fallados nuevos, y si el corte de producción
   de la lista de fallados es exposición o selección (hazard esperado contra observado).
6. **Qué marca la fecha nueva**: cambio de aceite y mensajes de sobrecarga del filtro
   alrededor de la fecha v1 y de la v2. Sin autos de test.

Lee los CSV crudos directo (es una auditoría de archivos, no del pipeline). La lectura
está en `docs/reproducibilidad.md`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.config import ensure_dir, load_config, resolve_path  # noqa: E402
from src.data.dedupe import canonical_map  # noqa: E402
from src.data.loader import table_parts  # noqa: E402
from src.eval.splits import load_test_split  # noqa: E402

logger = logging.getLogger("compare_deliveries")
pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 200)
ENC = "utf-8-sig"


def show(title: str, obj: Any) -> None:
    print(f"\n== {title} ==")
    print(obj.to_string() if isinstance(obj, (pd.DataFrame, pd.Series)) else obj)


def md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def parts(config: str) -> dict[tuple[str, str], Path]:
    return {(t, p.name): p.path for t in ("vehicles", "trips", "signals") for p in table_parts(t, config)}


# ======================================================================================
# 1 · Archivos
# ======================================================================================
def files_section(cfg: dict[str, Any]) -> pd.DataFrame:
    p1, p2 = parts(cfg["sources_v1"]), parts(cfg["sources_v2"])
    rows = []
    for key in sorted(p1):
        a, b = p1[key], p2[key]
        ha, hb = md5(a), md5(b)
        cols_a = pd.read_csv(a, nrows=0, encoding=ENC).columns.tolist()
        cols_b = pd.read_csv(b, nrows=0, encoding=ENC).columns.tolist()
        n_a = sum(1 for _ in a.open("rb")) - 1
        n_b = sum(1 for _ in b.open("rb")) - 1
        rows.append({
            "tabla": key[0], "cohorte": key[1], "archivo_v2": b.name, "md5_v1": ha, "md5_v2": hb,
            "igual_byte_a_byte": ha == hb, "filas_v1": n_a, "filas_v2": n_b,
            "columnas_nuevas": sorted(set(cols_b) - set(cols_a)), "columnas_que_faltan": sorted(set(cols_a) - set(cols_b)),
        })
    return pd.DataFrame(rows)


# ======================================================================================
# 2 · Estática
# ======================================================================================
def read_static(cfg: dict[str, Any], version: str) -> pd.DataFrame:
    p = parts(cfg[f"sources_{version}"])
    frames = []
    for cohort in ("failed", "not_failed"):
        f = pd.read_csv(p[("vehicles", cohort)], encoding=ENC)
        f = f.rename(columns={"IdentificationDaysSinceProduction": "IdentificationDate"})
        f["cohort"] = cohort
        frames.append(f)
    return pd.concat(frames, ignore_index=True)


def static_section(cfg: dict[str, Any], s1: pd.DataFrame, s2: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for version, s in (("v1", s1), ("v2", s2)):
        for cohort, g in s.groupby("cohort"):
            dup_codes = g[g["VehicleCode"].duplicated(keep=False)]
            out[f"{version}_{cohort}"] = {
                "filas": len(g), "codigos": int(g["VehicleCode"].nunique()),
                "codigos_repetidos": int(dup_codes["VehicleCode"].nunique()),
                "filas_identicas": int(g.drop(columns="cohort").duplicated().sum()),
                "filas_identicas_salvo_evento": int(g.drop(columns=["cohort", "IdentificationDate"]).duplicated().sum()),
            }
    def per_code(s: pd.DataFrame) -> pd.DataFrame:
        return s.groupby("VehicleCode").agg(
            coh=("cohort", lambda x: "".join(sorted({"F" if c == "failed" else "N" for c in x}))),
            country=("SalesCountry_cd", "first"), dus=("daysUntilSale", "first"), eng=("Engine", "first"),
            model=("ModelSeries", "first"), n_events=("IdentificationDate", lambda x: x.notna().sum()),
            id_first=("IdentificationDate", "min"))
    c1, c2 = per_code(s1), per_code(s2)
    both = c1.join(c2, lsuffix="_v1", rsuffix="_v2", how="inner")
    out["codigos_comunes"] = len(both)
    out["iguales_en_comunes"] = {
        "daysUntilSale": int(((both.dus_v1 == both.dus_v2) | (both.dus_v1.isna() & both.dus_v2.isna())).sum()),
        "Engine": int((both.eng_v1 == both.eng_v2).sum()), "ModelSeries": int((both.model_v1 == both.model_v2).sum()),
    }
    countries = pd.crosstab(both["country_v1"], both["country_v2"])
    trans = pd.Series(
        [f"{c1.coh.get(code, '-')}->{c2.coh.get(code, '-')}" for code in sorted(set(c1.index) | set(c2.index))]
    ).value_counts()
    out["_paises"] = countries
    out["_transiciones_codigo"] = trans
    return out, c1, c2


# ======================================================================================
# 3 · Anclaje
# ======================================================================================
def first_trips(path: Path, id_col: str = "VehicleCode") -> pd.Series:
    t = pd.read_csv(path, usecols=[id_col, "TripDatetimeStart"], encoding=ENC)
    ts = pd.to_datetime(t["TripDatetimeStart"], format="ISO8601", utc=True).dt.tz_convert(None)
    first = ts.groupby(t[id_col]).min()
    return (first.dt.normalize() - pd.Timestamp("1970-01-01")).dt.days


def anchor_section(cfg: dict[str, Any], s2: pd.DataFrame) -> pd.DataFrame:
    p = parts(cfg["sources_v2"])
    first = pd.concat([first_trips(p[("trips", "failed")]).rename("F"),
                       first_trips(p[("trips", "not_failed")]).rename("N")], axis=1)
    rows = []
    for cohort, key in (("failed", "F"), ("not_failed", "N")):
        pdays = s2[s2["cohort"] == cohort].groupby("VehicleCode")["ProductionDay"].first()
        for offset in (0, int(cfg["production_offset_failed_v2"])) if cohort == "failed" else (0,):
            origin = (first[key] - (pdays + offset)).dropna()
            rows.append({"cohorte": cohort, "corrimiento": offset, "n": len(origin), "mediana": origin.median(),
                         "iqr": origin.quantile(0.75) - origin.quantile(0.25), "std": round(origin.std(), 3),
                         "rango": origin.max() - origin.min()})
    return pd.DataFrame(rows)


# ======================================================================================
# 4 · Etiquetas
# ======================================================================================
def label_section(cfg: dict[str, Any], s1: pd.DataFrame, s2: pd.DataFrame, c1: pd.DataFrame,
                  c2: pd.DataFrame) -> dict[str, Any]:
    origin = pd.Timestamp(cfg["origin_date"])
    out: dict[str, Any] = {}
    f1 = s1[s1["cohort"] == "failed"].copy()
    f1["default"] = f1["IdentificationDate"] == f1["daysUntilSale"]
    f2 = s2[s2["cohort"] == "failed"].copy()
    out["v1_fecha_por_defecto"] = int(f1["default"].sum())
    out["v2_fecha_por_defecto"] = int((f2["IdentificationDate"] == f2["daysUntilSale"]).sum())
    out["v2_evento_antes_de_venta"] = int((f2["IdentificationDate"] < f2["daysUntilSale"]).sum())

    # fallados que siguen: cuánto se movió la fecha
    first_v2 = f2.groupby("VehicleCode")["IdentificationDate"].min()
    common = f1.set_index("VehicleCode").join(first_v2.rename("id_v2"), how="inner")
    delta = (common["id_v2"] - common["IdentificationDate"])
    out["fallados_que_siguen"] = int(len(common))
    out["delta_fecha_dias"] = delta.describe().round(1).to_dict()
    out["delta_negativo"] = int((delta < 0).sum())

    # ausentes: qué eran
    gone = f1[~f1["VehicleCode"].isin(set(s2["VehicleCode"]))].copy()
    gone["prod"] = origin + pd.to_timedelta(gone["ProductionDay"], unit="D")
    lo, hi = pd.Timestamp(cfg["production_test"]["reference_window"][0]), pd.Timestamp("2025-07-31")
    gone["en_ventana_produccion"] = gone["prod"].between(lo, hi)
    out["_ausentes"] = pd.crosstab([gone["SalesCountry_cd"]], [gone["en_ventana_produccion"], gone["default"]], margins=True)
    out["ausentes"] = int(len(gone))
    return out


# ======================================================================================
# 5 · Telemetría y producción
# ======================================================================================
def telemetry_section(cfg: dict[str, Any]) -> dict[str, Any]:
    p1, p2 = parts(cfg["sources_v1"]), parts(cfg["sources_v2"])
    canon = canonical_map(cfg["dedupe"])
    cols = ["TripDatetimeStart", "OdometerTripEnd"]
    t1 = pd.read_csv(p1[("trips", "failed")], usecols=["VehicleCode", *cols], encoding=ENC)
    # La fila completa: los viajes de los clones vienen bajo los dos códigos y se colapsan
    # canonizando y deduplicando la fila entera, como el pipeline (por clave se perderían
    # viajes legítimos con el mismo inicio y odómetro).
    t2 = pd.read_csv(p2[("trips", "failed")], encoding=ENC, low_memory=False)
    t1["VehicleCode"] = t1["VehicleCode"].replace(canon)
    t2["VehicleCode"] = t2["VehicleCode"].replace(canon)
    t2 = t2.drop_duplicates()[["VehicleCode", *cols]]
    for t in (t1, t2):
        t["ts"] = pd.to_datetime(t["TripDatetimeStart"], format="ISO8601", utc=True).dt.tz_convert(None)
    common = sorted(set(t1["VehicleCode"]) & set(t2["VehicleCode"]))
    k1 = set(zip(t1["VehicleCode"], t1["ts"], t1["OdometerTripEnd"]))
    k2 = set(zip(t2["VehicleCode"], t2["ts"], t2["OdometerTripEnd"]))
    k1c = {k for k in k1 if k[0] in set(common)}
    k2c = {k for k in k2 if k[0] in set(common)}
    new_codes = sorted(set(t2["VehicleCode"]) - set(t1["VehicleCode"]))
    g_new = t2[t2["VehicleCode"].isin(new_codes)].groupby("VehicleCode")["ts"].agg(["min", "max"])
    n_raw = pd.read_csv(p2[("trips", "not_failed")], usecols=["TripDatetimeStart"], encoding=ENC)
    healthy_end = pd.to_datetime(n_raw["TripDatetimeStart"], format="ISO8601", utc=True).max()
    return {
        "fallados_comunes_con_viajes": len(common),
        "viajes_v1_en_v2": len(k1c & k2c), "viajes_v1_que_faltan_en_v2": len(k1c - k2c),
        "viajes_v2_nuevos_en_comunes": len(k2c - k1c),
        "fin_viajes_fallados_v1": str(t1["ts"].max()), "fin_viajes_fallados_v2": str(t2["ts"].max()),
        "fin_viajes_sanos": str(healthy_end.tz_convert(None)),
        "viajes_fallados_v2_despues_del_fin_sanos": int((t2["ts"] > healthy_end.tz_convert(None)).sum()),
        "fallados_nuevos_con_viajes": len(new_codes),
        "fallados_nuevos_primer_viaje": g_new["min"].describe().astype(str).to_dict(),
    }


def production_test(cfg: dict[str, Any], s2: pd.DataFrame) -> pd.DataFrame:
    """Hazard por días desde la venta de los producidos en la ventana de referencia,
    aplicado a la exposición de los producidos después: eventos esperados contra observados."""
    origin, end = pd.Timestamp(cfg["origin_date"]), pd.Timestamp(cfg["extraction_end"])
    canon = canonical_map(cfg["dedupe"])
    s = s2.copy()
    s["vid"] = s["VehicleCode"].replace(canon)
    s["pd"] = s["ProductionDay"] + np.where(s["cohort"] == "failed", cfg["production_offset_failed_v2"], 0)
    f = s[s["cohort"] == "failed"].sort_values("IdentificationDate").drop_duplicates("vid")
    h = s[(s["cohort"] == "not_failed") & ~s["vid"].isin(set(f["vid"]))].drop_duplicates("vid")
    v = pd.concat([f.assign(y=1), h.assign(y=0)], ignore_index=True)
    v["prod"] = origin + pd.to_timedelta(v["pd"], unit="D")
    v["sale"] = v["prod"] + pd.to_timedelta(v["daysUntilSale"], unit="D")
    v["event"] = v["prod"] + pd.to_timedelta(v["IdentificationDate"], unit="D")
    v["exit"] = v["event"].where(v["y"].eq(1) & v["event"].le(end), end)
    v["y"] = (v["y"].eq(1) & v["event"].le(end)).astype(int)
    v = v.dropna(subset=["sale"])
    v["dss_exit"] = (v["exit"] - v["sale"]).dt.days
    bins = cfg["production_test"]["dss_bins"]

    def exposure(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        e = np.array([(df["dss_exit"].clip(upper=b1) - b0).clip(lower=0).sum() for b0, b1 in zip(bins[:-1], bins[1:])])
        d = np.array([((df["y"] == 1) & (df["dss_exit"] > b0) & (df["dss_exit"] <= b1)).sum()
                      for b0, b1 in zip(bins[:-1], bins[1:])])
        return e, d
    lo, hi = cfg["production_test"]["reference_window"]
    e_ref, d_ref = exposure(v[v["prod"].between(lo, hi)])
    rate = np.divide(d_ref, e_ref, out=np.zeros_like(e_ref, dtype=float), where=e_ref > 0)
    rows = [{"ventana": f"referencia {lo} → {hi}", "autos": int(v["prod"].between(lo, hi).sum()),
             "observados": int(d_ref.sum()), "esperados": float(d_ref.sum())}]
    for name, (a, b) in cfg["production_test"]["late_windows"].items():
        sub = v[v["prod"].between(a, b)]
        e, _ = exposure(sub)
        rows.append({"ventana": name, "autos": len(sub), "observados": int(sub["y"].sum()),
                     "esperados": float((e * rate).sum())})
    by_q = v.assign(q=v["prod"].dt.to_period("Q").astype(str)).groupby(["q", "y"]).size().unstack(fill_value=0)
    return pd.DataFrame(rows), by_q


# ======================================================================================
# 6 · Qué marca la fecha nueva (sin test)
# ======================================================================================
def signatures_section(cfg: dict[str, Any], s1: pd.DataFrame, s2: pd.DataFrame) -> dict[str, Any]:
    origin = pd.Timestamp(cfg["origin_date"])
    test = set(load_test_split(cfg["test_split"])["test_vehicles"]) | set(load_test_split(cfg["test_split_v1"])["test_vehicles"])
    canon = canonical_map(cfg["dedupe"])
    p2 = parts(cfg["sources_v2"])
    f2 = s2[s2["cohort"] == "failed"].copy()
    f2["vid"] = f2["VehicleCode"].replace(canon)
    f2["v2_date"] = origin + pd.to_timedelta(f2["ProductionDay"] + cfg["production_offset_failed_v2"]
                                             + f2["IdentificationDate"], unit="D")
    first = f2.sort_values("v2_date").drop_duplicates("vid").set_index("vid")["v2_date"]
    f1 = s1[s1["cohort"] == "failed"].copy()
    f1["vid"] = f1["VehicleCode"].replace(canon)
    f1["v1_date"] = origin + pd.to_timedelta(f1["ProductionDay"] + f1["IdentificationDate"], unit="D")
    v1_dates = f1.drop_duplicates("vid").set_index("vid")["v1_date"]
    keep = sorted(set(first.index) - test)
    common = sorted(set(keep) & set(v1_dates.index))

    t = pd.read_csv(p2[("trips", "failed")], usecols=["VehicleCode", "TripDatetimeStart", "EngineOilLifePCStart",
                                                      "EngineOilLifePCEnd"], encoding=ENC)
    t["vid"] = t.pop("VehicleCode").replace(canon)
    t = t.drop_duplicates()
    t = t[t["vid"].isin(set(keep))]
    t["day"] = pd.to_datetime(t["TripDatetimeStart"], format="ISO8601", utc=True).dt.tz_convert(None).dt.normalize()
    t = t.sort_values(["vid", "TripDatetimeStart"])
    t["oil_reset"] = (t["EngineOilLifePCStart"] - t.groupby("vid")["EngineOilLifePCEnd"].shift()) >= cfg["signatures"]["oil_reset_points"]
    s = pd.read_csv(p2[("signals", "failed")], usecols=["VehicleCode", "eventTimestamp", "Message"], encoding=ENC)
    s["vid"] = s["VehicleCode"].replace(canon)
    s = s[s["vid"].isin(set(keep))]
    s["day"] = pd.to_datetime(s["eventTimestamp"], format="ISO8601", utc=True).dt.tz_convert(None).dt.normalize()
    s["over"] = s["Message"].isin(cfg["signatures"]["bad_messages"])
    daily_msg = s.groupby(["vid", "day"])["over"].max().reset_index()
    active = t.groupby("vid")["day"].apply(set)
    bins = cfg["signatures"]["bins_days"]

    def profile(refs: pd.Series, vehicles: list[str]) -> pd.DataFrame:
        rows = []
        for vid in vehicles:
            ref = refs[vid]
            rel_oil = (t.loc[(t["vid"] == vid) & t["oil_reset"], "day"] - ref).dt.days
            m = daily_msg[daily_msg["vid"] == vid]
            rel_msg = (m["day"] - ref).dt.days
            rows.append(pd.DataFrame({"rel": rel_oil, "kind": "cambio_aceite", "val": 1.0}))
            rows.append(pd.DataFrame({"rel": rel_msg, "kind": "sobrecarga", "val": m["over"].astype(float).to_numpy()}))
        x = pd.concat(rows)
        x = x[x["rel"].between(bins[0] + 1, bins[-1])]
        x["bin"] = pd.cut(x["rel"], bins)
        oil = x[x["kind"] == "cambio_aceite"].groupby("bin", observed=False).size() / len(vehicles)
        over = x[x["kind"] == "sobrecarga"].groupby("bin", observed=False)["val"].mean()
        width = pd.Series([b1 - b0 for b0, b1 in zip(bins[:-1], bins[1:])], index=oil.index)
        return pd.DataFrame({"cambios_aceite_por_auto_dia": oil / width, "frac_dias_con_sobrecarga": over})

    between = []
    for vid in common:
        a, b = v1_dates[vid], first[vid]
        n = (b - a).days
        if n <= 0:
            continue
        days = active.get(vid, set())
        between.append({"L": n, "activo_entre": np.mean([(a + pd.Timedelta(days=k)) in days for k in range(n)]),
                        "activo_antes": np.mean([(a - pd.Timedelta(days=k)) in days for k in range(1, n + 1)])})
    between = pd.DataFrame(between)
    return {
        "n_fallados_sin_test": len(keep), "n_comunes_sin_test": len(common),
        "_perfil_v2": profile(first, keep), "_perfil_v1_comunes": profile(v1_dates, common),
        "_perfil_v2_comunes": profile(first, common),
        "actividad_entre_v1_y_v2": float(between["activo_entre"].mean()),
        "actividad_mismo_largo_antes_de_v1": float(between["activo_antes"].mean()),
        "lag_v2_menos_v1_dias": between["L"].describe().round(1).to_dict(),
    }


# ======================================================================================
def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/data/compare_deliveries.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    out_dir = ensure_dir(cfg["output_dir"])
    summary: dict[str, Any] = {}

    files = files_section(cfg)
    show("1 · archivos", files.drop(columns=["md5_v1", "md5_v2"]))
    files.to_csv(out_dir / "1_archivos.csv", index=False)
    summary["archivos"] = files.to_dict(orient="records")

    s1, s2 = read_static(cfg, "v1"), read_static(cfg, "v2")
    st, c1, c2 = static_section(cfg, s1, s2)
    for key, value in st.items():
        show(f"2 · {key.lstrip('_')}", value if isinstance(value, (pd.DataFrame, pd.Series)) else json.dumps(value, ensure_ascii=False))
        if isinstance(value, (pd.DataFrame, pd.Series)):
            value.to_csv(out_dir / f"2_{key.lstrip('_')}.csv")
    summary["estatica"] = {k: v for k, v in st.items() if not k.startswith("_")}

    anchor = anchor_section(cfg, s2)
    show("3 · anclaje (primer viaje − ProductionDay, días desde epoch)", anchor)
    anchor.to_csv(out_dir / "3_anclaje.csv", index=False)
    summary["anclaje"] = anchor.to_dict(orient="records")

    labels = label_section(cfg, s1, s2, c1, c2)
    for key, value in labels.items():
        show(f"4 · {key.lstrip('_')}", value if isinstance(value, (pd.DataFrame, pd.Series)) else json.dumps(value, ensure_ascii=False, default=float))
    labels["_ausentes"].to_csv(out_dir / "4_ausentes.csv")
    summary["etiquetas"] = {k: v for k, v in labels.items() if not k.startswith("_")}

    tel = telemetry_section(cfg)
    show("5 · telemetría", json.dumps(tel, indent=1, ensure_ascii=False, default=str))
    summary["telemetria"] = tel
    prod, by_q = production_test(cfg, s2)
    show("5 · corte de producción: eventos esperados (hazard de la referencia) contra observados", prod.round(1))
    show("5 · autos por trimestre de producción y cohorte", by_q)
    prod.to_csv(out_dir / "5_corte_produccion.csv", index=False)
    by_q.to_csv(out_dir / "5_produccion_por_trimestre.csv")
    summary["corte_produccion"] = prod.round(2).to_dict(orient="records")

    sig = signatures_section(cfg, s1, s2)
    for key, value in sig.items():
        show(f"6 · {key.lstrip('_')}", value.round(4) if isinstance(value, pd.DataFrame) else value)
        if isinstance(value, pd.DataFrame):
            value.to_csv(out_dir / f"6_{key.lstrip('_')}.csv")
    summary["fecha_v2"] = {k: v for k, v in sig.items() if not k.startswith("_")}

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\nEscrito en {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
