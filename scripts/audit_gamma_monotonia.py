#!/usr/bin/env python
"""Paso 1 del proceso gamma: ¿hay un proxy de carga irreversible en estos datos?

    python scripts/audit_gamma_monotonia.py --config configs/data/gamma_monotonia.yaml

Un proceso gamma de degradación (Lawless & Crowder 2004) supone un daño acumulado
D(km) **monótono no decreciente** con incrementos gamma cuya tasa depende del uso, y
define la falla como el primer cruce de un umbral. La física del DPF encaja: la
ceniza no se quema en la regeneración, ocupa capacidad del filtro y obliga a
regenerar cada vez más seguido.

Antes de implementar el modelo hay que medir las dos condiciones que lo sostienen,
sobre dev y sobre los crudos:

  **(a) monotonía** — el proxy de carga sube con el odómetro *dentro* del vehículo;
  **(b) separación** — la tasa a la que sube distingue fallados de sanos.

Sin (a) no hay estado latente que estimar; sin (b), el estado existe pero no dice
nada de la falla y el proceso gamma es una reparametrización cara de la tasa base.

Cuatro bloques, cuatro proxies, y en todos la misma precaución: **la exposición no es
comparable entre cohortes**. El historial de un fallado se corta en el evento (mediana
~10.500 km) y el de un sano llega a donde llegan los datos (~16.800 km), así que una
pendiente estimada sobre el historial completo compara tramos distintos de la vida del
vehículo. Por eso el bloque D repite la medición sobre ventanas de odómetro que las dos
cohortes cubren enteras: es la diferencia entre medir física y medir cuánto duró el
registro.

  A · nivel del postratamiento resumido por bins de odómetro (piso, p10 y media)
  B · residuo con el que termina cada regeneración — la ceniza, según la física
  C · distancia entre regeneraciones y tasa local — el síntoma de capacidad perdida
  D · A y C otra vez, con la exposición igualada

Deja los CSV por vehículo en `experiments/gamma/dev/` y el resumen en `resumen.csv`.
El resultado y su lectura están en docs/memoria/f3-proceso-gamma.md.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ensure_dir, load_config, repo_root, set_seed  # noqa: E402
from src.data.anchor import estimate_origin_day, event_dates, project_dates_to_odometer  # noqa: E402
from src.data.join import load_vehicle_static  # noqa: E402
from src.data.subset import read_table_for_vehicles  # noqa: E402
from src.eval.splits import load_test_split  # noqa: E402
from src.features.trips import derive_trip_columns  # noqa: E402

logger = logging.getLogger("audit-gamma")

ID = "vehicle_id"
ODO = "OdometerTripEnd"


# --------------------------------------------------------------------------------------
# primitivas
# --------------------------------------------------------------------------------------
def trend(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """`(rho de Spearman, pendiente por 1.000 km)` de `y` contra `x`.

    Los dos juntos a propósito: rho dice si la relación es monótona (que es lo que
    el proceso gamma exige) y la pendiente dice cuánto, en unidades de la escala.
    Con `y` constante —pasa: la escala es entera y en pasos de 5— rho no existe y
    la pendiente es 0; se devuelve NaN y 0 en vez de romper.
    """
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan"), 0.0
    rho = float(stats.spearmanr(x, y).statistic)
    slope = float(np.polyfit(x, y, 1)[0] * 1000)
    return rho, slope


def monotonia(values: pd.Series) -> dict[str, float]:
    """Condición (a): ¿la tendencia intra-vehículo es positiva *en la flota*?

    El test es sobre los vehículos, no sobre las filas: un Wilcoxon de signo sobre
    los rho por vehículo. Un rho mediano de +0,3 con el 80% de los vehículos positivo
    es monotonía débil pero real; uno de +0,00 con el 50% positivo es ruido.
    """
    s = values.dropna()
    if len(s) < 5:
        return {"n": len(s), "mediana": float("nan"), "frac_positiva": float("nan"), "p": float("nan")}
    return {
        "n": int(len(s)),
        "mediana": float(s.median()),
        "frac_positiva": float((s > 0).mean()),
        "p": float(stats.wilcoxon(s, alternative="two-sided").pvalue),
    }


def separacion(frame: pd.DataFrame, column: str, *, label: str = "event_observed") -> dict[str, float]:
    """Condición (b): AUC de Mann-Whitney de `column` entre fallados y sanos.

    AUC y no diferencia de medianas porque es la métrica con la que se compara contra
    el resto del repo (`src/eval/metrics.py`), y porque una tasa con cola larga mueve
    la media sin mover el orden. 0,5 es "no separa".
    """
    ev = frame.loc[frame[label] == 1, column].dropna()
    hl = frame.loc[frame[label] == 0, column].dropna()
    if len(ev) < 5 or len(hl) < 5:
        return {"n_evento": len(ev), "n_sano": len(hl), "auc": float("nan"), "p": float("nan"),
                "mediana_evento": float("nan"), "mediana_sano": float("nan")}
    u = stats.mannwhitneyu(ev, hl, alternative="two-sided")
    return {
        "n_evento": int(len(ev)),
        "n_sano": int(len(hl)),
        "mediana_evento": float(ev.median()),
        "mediana_sano": float(hl.median()),
        "auc": float(u.statistic / (len(ev) * len(hl))),
        "p": float(u.pvalue),
    }


# --------------------------------------------------------------------------------------
# datos
# --------------------------------------------------------------------------------------
def load_pre_event_trips(cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """`trips` de dev, con derivadas, **truncados en el evento**.

    El truncamiento no es opcional: después de `IdentificationDate` hay una
    intervención (limpieza o cambio de filtro) y el nivel del postratamiento vuelve a
    cero por un motivo que no es el uso. Dejar esas filas adentro mete una caída
    artificial justo en los vehículos con evento, que es la mitad de la comparación.
    """
    split = load_test_split(cfg["test_split"])
    dev = {str(v) for v in split["dev_vehicles"]}
    logger.info("Universo de la auditoría: %d vehículos de dev", len(dev))

    spec_cfg = load_config(load_config(cfg["panel"])["features"]["spec"])
    trips, counters = read_table_for_vehicles(
        "trips", None, dev, sources=cfg["sources"], dedupe=cfg["dedupe"],
        chunksize=int((cfg.get("scan") or {}).get("chunksize", 500_000)),
    )
    trips, _ = derive_trip_columns(trips, thresholds=spec_cfg.get("thresholds"), clip=spec_cfg.get("clip"))

    intrusos = set(trips[ID].astype(str).unique()) - dev
    if intrusos:
        raise AssertionError(
            f"`trips`: {len(intrusos)} vehículo(s) que no son de dev (ej.: {sorted(intrusos)[:3]})"
        )

    static = load_vehicle_static(config_path=cfg["sources"], dedupe_config=cfg["dedupe"],
                                 panel_config=cfg["panel"])
    static = static[static[ID].astype(str).isin(dev)].set_index(ID)
    first_trip = trips.groupby(ID, observed=True)["TripDatetimeStart"].min()
    origin_day, anchor_stats = estimate_origin_day(first_trip.reindex(static.index),
                                                   static["static_ProductionDay"])
    is_event = static["event_observed"].eq(1)
    events = project_dates_to_odometer(event_dates(static.loc[is_event], origin_day), trips)
    logger.info("Eventos ubicados en km: %d · mediana %.0f km", len(events), events["event_odo_km"].median())

    trips = trips.merge(events["event_odo_km"], left_on=ID, right_index=True, how="left")
    trips["event_observed"] = trips[ID].map(static["event_observed"]).astype(int)
    n_before = len(trips)
    trips = trips[trips["event_odo_km"].isna() | (trips[ODO] <= trips["event_odo_km"])]
    trips = trips.sort_values([ID, ODO])
    logger.info("Viajes pre-evento: %d de %d (%.1f%% descartado como post-evento)",
                len(trips), n_before, 100 * (1 - len(trips) / n_before))

    exposure = trips.groupby(ID).agg(
        span_km=(ODO, lambda s: float(np.ptp(s))), n_trips=(ODO, "size"),
        event_observed=("event_observed", "first"),
    )
    meta = {"counters": counters, "anchor": anchor_stats, "origin_day": origin_day,
            "n_events": int(len(events))}
    return trips, exposure, meta


# --------------------------------------------------------------------------------------
# bloques
# --------------------------------------------------------------------------------------
def level_trend_by_vehicle(
    trips: pd.DataFrame, audit: dict[str, Any], *, bin_km: float, min_trips_bin: int,
    min_bins: int, min_trips: int, window: tuple[float, float] | None = None,
    min_coverage: float = 0.0,
) -> pd.DataFrame:
    """Bloque A/D · tendencia del nivel del postratamiento, por vehículo.

    El nivel crudo oscila con cada ciclo de carga y descarga; lo que el proceso gamma
    modela es el **piso** que esa oscilación deja. Se estima resumiendo el nivel en
    bins de odómetro y siguiendo tres estadísticos: `min` (el piso literal, que la
    escala entera en pasos de 5 vuelve grueso), `p10` (el piso robusto) y `mean` (el
    nivel medio, que mezcla ceniza y hollín).

    Con `window` se mide solo el tramo `[lo, hi]` del odómetro y se exigen vehículos
    que lo cubran: es lo que iguala la exposición entre cohortes.
    """
    lo, hi = window if window else (-np.inf, np.inf)
    column = audit["level_column"]
    rows = []
    for vid, g in trips.groupby(ID):
        km = g[ODO].to_numpy(float)
        level = g[column].to_numpy(float)
        ok = np.isfinite(km) & np.isfinite(level) & (km >= lo) & (km <= hi)
        if ok.sum() < min_trips:
            continue
        km, level = km[ok], level[ok]
        if window and np.ptp(km) < min_coverage * (hi - lo):
            continue
        if not window and np.ptp(km) < min_bins * bin_km:
            continue
        binned = pd.DataFrame({"bin": np.floor(km / bin_km).astype(int), "level": level, "km": km})
        agg = binned.groupby("bin").agg(
            piso=("level", "min"),
            p10=("level", lambda s: s.quantile(0.10)),
            medio=("level", "mean"),
            km=("km", "mean"),
            n=("level", "size"),
        )
        agg = agg[agg["n"] >= min_trips_bin]
        if len(agg) < min_bins:
            continue
        row: dict[str, Any] = {ID: vid, "n_bins": int(len(agg)), "span_km": float(np.ptp(km)),
                               "event_observed": int(g["event_observed"].iloc[0])}
        for stat in ("piso", "p10", "medio"):
            row[f"rho_{stat}"], row[f"slope_{stat}"] = trend(agg["km"].to_numpy(), agg[stat].to_numpy())
        rows.append(row)
    return pd.DataFrame(rows)


def regen_trend_by_vehicle(
    trips: pd.DataFrame, audit: dict[str, Any],
    *, window: tuple[float, float] | None = None, min_coverage: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Bloques B y C · la ceniza medida donde la física la pone.

    B · **residuo**: el nivel con el que *termina* una regeneración. La regeneración
    quema hollín y no ceniza, así que lo que queda es la ceniza. Si hay acumulación
    irreversible, ese residuo sube con el odómetro.

    C · **ciclo**: distancia entre regeneraciones consecutivas y tasa local (1.000/gap).
    Es el síntoma indirecto: menos capacidad libre ⇒ el filtro se llena antes ⇒ se
    regenera más seguido. Es el mismo mecanismo que las features de familia B del
    panel, medido acá como tendencia dentro del vehículo en vez de como nivel.

    Con `window` se repite todo sobre un tramo de odómetro que las dos cohortes
    cubren entera. No es un refinamiento: es el control que decide si una diferencia
    entre cohortes es física o es cuánto duró el registro.
    """
    lo, hi = window if window else (-np.inf, np.inf)
    reg = trips[trips["regen_drop"].fillna(False)].copy()
    reg = reg[(reg[ODO] >= lo) & (reg[ODO] <= hi)]
    rows = []
    for vid, g in reg.groupby(ID):
        km = g[ODO].to_numpy(float)
        if len(km) < audit["regen_min_events"] or np.ptp(km) < audit["regen_min_span_km"]:
            continue
        if window and np.ptp(km) < min_coverage * (hi - lo):
            continue
        residual = g["regen_residual"].to_numpy(float)
        start = g["regen_start_level"].to_numpy(float)
        gap = np.diff(km)
        mid = 0.5 * (km[1:] + km[:-1])
        rate = 1000.0 / np.clip(gap, 1.0, None)
        row: dict[str, Any] = {ID: vid, "n_regen": int(len(km)), "span_km": float(np.ptp(km)),
                               "event_observed": int(g["event_observed"].iloc[0]),
                               "residual_medio": float(np.nanmean(residual)),
                               "residual_cero_frac": float(np.nanmean(residual == 0)),
                               "gap_medio_km": float(gap.mean()),
                               "tasa_global_por_1000km": float(1000.0 * len(km) / np.ptp(km))}
        row["rho_residual"], row["slope_residual"] = trend(km, residual)
        row["rho_disparo"], row["slope_disparo"] = trend(km, start)
        row["rho_gap"], row["slope_gap"] = trend(mid, gap)
        row["rho_tasa"], row["slope_tasa"] = trend(mid, rate)
        rows.append(row)
    return reg, pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# salida
# --------------------------------------------------------------------------------------
def summary_rows(
    bloque: str, frame: pd.DataFrame, proxies: dict[str, tuple[str, int]],
    *, window: tuple[float, float] | None = None,
) -> list[dict[str, Any]]:
    """Una fila por proxy con las dos condiciones lado a lado: monotonía y separación.

    Cada proxy trae su **orientación**: +1 si la hipótesis dice que crece hacia la
    falla (el nivel, el residuo de ceniza, la tasa de regeneración) y −1 si decrece
    (los km entre regeneraciones, que se acortan a medida que la ceniza ocupa
    capacidad). Se aplica antes de resumir, así "rho positivo" y "AUC > 0,5" quieren
    decir lo mismo en toda la tabla y el veredicto puede leerlas juntas. Sin esto, el
    proxy con el AUC más alto del bloque C entraría como evidencia a favor cuando
    apunta exactamente al revés.
    """
    out = []
    for prefix, (nombre, orientacion) in proxies.items():
        oriented = frame.assign(
            _rho=orientacion * frame[f"rho_{prefix}"],
            _slope=orientacion * frame[f"slope_{prefix}"],
        )
        mono = monotonia(oriented["_rho"])
        sep = separacion(oriented, "_slope")
        out.append({
            "bloque": bloque, "proxy": nombre, "orientacion": orientacion,
            "n_vehiculos": len(frame),
            "win_lo": window[0] if window else float("nan"),
            "win_hi": window[1] if window else float("nan"),
            "rho_mediano": mono["mediana"], "frac_rho_positivo": mono["frac_positiva"],
            "wilcoxon_p": mono["p"],
            "slope_mediano_evento": sep["mediana_evento"], "slope_mediano_sano": sep["mediana_sano"],
            "auc_slope": sep["auc"], "auc_p": sep["p"],
            "n_evento": sep["n_evento"], "n_sano": sep["n_sano"],
        })
    return out


def veredicto(tabla: pd.DataFrame, reglas: dict[str, Any]) -> dict[str, Any]:
    """Las dos condiciones del paso 1, con el criterio que declara el YAML.

    Dos cuidados que la versión ingenua de este chequeo no tiene, y que acá cambian
    el resultado:

    * la **monotonía** se juzga fuera del asentamiento. En la ventana que arranca en 0
      casi todos los vehículos suben, y eso es el primer llenado del filtro, no ceniza;
    * la **separación** solo cuenta con el signo de la hipótesis (AUC > 0,5: el que
      falla acumula más rápido) y con `p` chico. Un AUC de 0,40 significativo sería
      evidencia *en contra*, y redondearlo a "0,10 de distancia del azar" lo daría por
      bueno al revés.
    """
    lo, hi = reglas["monotonia_window"]
    mono_rows = tabla[(tabla["win_lo"] == lo) & (tabla["win_hi"] == hi)]
    if mono_rows.empty:
        raise ValueError(
            f"`veredicto.monotonia_window` = [{lo}, {hi}] no está en `audit.matched_windows`"
        )
    mono_best = float(mono_rows["frac_rho_positivo"].max())

    matched = tabla[tabla["win_lo"].notna()]
    a_favor = matched[(matched["auc_slope"] >= float(reglas["min_auc"]))
                      & (matched["auc_p"] < float(reglas["max_p"]))]
    return {
        "monotonia": mono_best >= float(reglas["min_frac_rho_positivo"]),
        "monotonia_frac": mono_best,
        "monotonia_window": (lo, hi),
        "separacion": not a_favor.empty,
        "separacion_auc_max": float(matched["auc_slope"].max()),
        "separacion_auc_min": float(matched["auc_slope"].min()),
    }


def log_table(rows: list[dict[str, Any]]) -> None:
    logger.info("%-26s %-28s %7s %7s %7s %9s %7s %7s", "bloque", "proxy", "rho_med", "%rho>0",
                "wilcox", "slope_ev", "auc", "p")
    for r in rows:
        logger.info("%-26s %-28s %+7.3f %6.1f%% %7.2g %+9.3f %7.3f %7.3f",
                    r["bloque"], r["proxy"], r["rho_mediano"], 100 * r["frac_rho_positivo"],
                    r["wilcoxon_p"], r["slope_mediano_evento"], r["auc_slope"], r["auc_p"])


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/data/gamma_monotonia.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))
    audit = cfg["audit"]
    out_dir = ensure_dir(cfg["output"]["dir"])

    trips, exposure, meta = load_pre_event_trips(cfg)
    exposure.to_csv(out_dir / "exposicion_por_vehiculo.csv")
    # Con qué se midió: el anclaje del calendario y los contadores del dedupe. Sin esto,
    # dos corridas de la auditoría en momentos distintos no son distinguibles.
    (out_dir / "meta.json").write_text(
        json.dumps({"config": cfg.get("_config_path"), **meta}, indent=2, default=float),
        encoding="utf-8",
    )
    for label, name in ((1, "con evento"), (0, "sanos")):
        span = exposure.loc[exposure["event_observed"] == label, "span_km"]
        logger.info("Exposición %-10s n=%3d · span mediano %7.0f km (p25 %6.0f, p75 %6.0f)",
                    name, len(span), span.median(), span.quantile(0.25), span.quantile(0.75))

    resumen: list[dict[str, Any]] = []

    # A · nivel del postratamiento por bins de odómetro -------------------------------
    nivel = level_trend_by_vehicle(
        trips, audit, bin_km=float(audit["level_bin_km"]),
        min_trips_bin=int(audit["level_bin_min_trips"]), min_bins=int(audit["level_min_bins"]),
        min_trips=int(audit["level_min_trips"]),
    )
    nivel.to_csv(out_dir / "bloque_a_nivel.csv", index=False)
    resumen += summary_rows(
        f"A · nivel/{audit['level_bin_km']:.0f}km", nivel,
        {"piso": ("piso del bin (min)", +1), "p10": ("piso robusto (p10)", +1),
         "medio": ("nivel medio", +1)},
    )

    # B y C · residuo de la regeneración y ciclo --------------------------------------
    reg, ciclo = regen_trend_by_vehicle(trips, audit)
    ciclo.to_csv(out_dir / "bloque_bc_regeneraciones.csv", index=False)
    logger.info("Regeneraciones pre-evento: %d en %d vehículos · residuo == 0 en el %.1f%%",
                len(reg), reg[ID].nunique(), 100 * float((reg["regen_residual"] == 0).mean()))
    resumen += summary_rows("B · residuo de regeneración", ciclo,
                            {"residual": ("residuo (ceniza)", +1),
                             "disparo": ("nivel de disparo", +1)})
    # El gap se acorta hacia la falla: entra con orientación −1 y por eso su rho y su
    # AUC se leen con el mismo signo que los del resto de la tabla.
    resumen += summary_rows("C · ciclo de regeneración", ciclo,
                            {"gap": ("km entre regeneraciones (−)", -1),
                             "tasa": ("tasa local", +1)})

    # D · lo mismo, con la exposición igualada ----------------------------------------
    for lo, hi in audit["matched_windows"]:
        emparejado = level_trend_by_vehicle(
            trips, audit, bin_km=float(audit["matched_bin_km"]),
            min_trips_bin=int(audit["matched_bin_min_trips"]), min_bins=int(audit["matched_min_bins"]),
            min_trips=int(audit["matched_min_trips"]), window=(float(lo), float(hi)),
            min_coverage=float(audit["matched_min_coverage"]),
        )
        emparejado.to_csv(out_dir / f"bloque_d_nivel_{lo:.0f}_{hi:.0f}.csv", index=False)
        resumen += summary_rows(f"D · {lo:.0f}-{hi:.0f} km", emparejado,
                                {"p10": ("piso robusto (p10)", +1), "medio": ("nivel medio", +1)},
                                window=(float(lo), float(hi)))
        # El mismo control para el ciclo de regeneración: es el proxy con el AUC más
        # alto del bloque C y el que más hay que desconfiar.
        _, ciclo_w = regen_trend_by_vehicle(
            trips, audit, window=(float(lo), float(hi)),
            min_coverage=float(audit["matched_min_coverage"]),
        )
        if not ciclo_w.empty:
            ciclo_w.to_csv(out_dir / f"bloque_d_ciclo_{lo:.0f}_{hi:.0f}.csv", index=False)
            resumen += summary_rows(f"D · {lo:.0f}-{hi:.0f} km", ciclo_w,
                                    {"gap": ("km entre regeneraciones (−)", -1)},
                                    window=(float(lo), float(hi)))

    tabla = pd.DataFrame(resumen)
    tabla.to_csv(out_dir / "resumen.csv", index=False)
    log_table(resumen)

    # El veredicto no se deja a interpretación: las dos condiciones, explícitas.
    v = veredicto(tabla, cfg["veredicto"])
    logger.info("")
    logger.info("(a) MONOTONÍA  : %s · %.1f%% de los vehículos con rho > 0 en %.0f-%.0f km "
                "(fuera del asentamiento)",
                "SE SOSTIENE" if v["monotonia"] else "NO se sostiene", 100 * v["monotonia_frac"],
                *v["monotonia_window"])
    logger.info("(b) SEPARACIÓN : %s · con la exposición igualada el AUC de la tasa va de "
                "%.3f a %.3f",
                "SE SOSTIENE" if v["separacion"] else "NO se sostiene",
                v["separacion_auc_min"], v["separacion_auc_max"])
    logger.info("Veredicto del paso 1: %s",
                "seguir al paso 2" if (v["monotonia"] and v["separacion"]) else
                "PARAR — el proceso gamma no tiene premisa en estos datos")
    logger.info("Outputs: %s", out_dir.relative_to(repo_root()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
