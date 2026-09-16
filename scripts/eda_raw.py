#!/usr/bin/env python
"""EDA de las tablas crudas: el diagnóstico de F1 antes de construir el panel.

    python scripts/eda_raw.py
    python scripts/eda_raw.py --chunksize 250000 --out experiments/eda

No dibuja nada bonito: responde las preguntas que bloquean F2.

1. Inventario y join: ¿cierran los vehículos entre las tres tablas?
2. Etiqueta y censura: cuántos eventos, cuánta cobertura por vehículo.
3. Odómetro: ¿sirve como eje? (nulos, retrocesos, saltos)
4. Anclaje temporal: `IdentificationDate` está en días desde producción y los
   viajes en calendario. Acá se mide si `ProductionDay` alcanza para alinearlos,
   que es lo que decide si el evento se puede ubicar sobre el eje de km.
5. Señal del postratamiento: cuánto separan los niveles de `Message` a las dos
   cohortes (el insumo para elegir el gap de blanking).

Todo lo que calcula queda en CSV bajo `--out` para que el informe y el dashboard
no tengan que volver a leer 1,2 GB.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.config import ensure_dir  # noqa: E402
from src.data.loader import DEFAULT_SOURCES_CONFIG, iter_table, load_table  # noqa: E402

logger = logging.getLogger("eda")

VEHICLE_COL = "VehicleCode"


def h(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def per_vehicle_trips(config: str, chunksize: int) -> pd.DataFrame:
    """Agregados por vehículo de la tabla de viajes, en una pasada por chunks."""
    columns = [
        VEHICLE_COL,
        "TripDatetimeStart",
        "TripDatetimeEnd",
        "OdometerTripStart",
        "OdometerTripEnd",
        "TripNumber",
    ]
    parts = []
    for chunk in iter_table("trips", config, chunksize=chunksize, columns=columns):
        chunk["trip_km"] = chunk["OdometerTripEnd"] - chunk["OdometerTripStart"]
        chunk["dur_h"] = (
            chunk["TripDatetimeEnd"] - chunk["TripDatetimeStart"]
        ).dt.total_seconds() / 3600.0
        parts.append(
            chunk.groupby([VEHICLE_COL, "cohort"], observed=True).agg(
                n_trips=("OdometerTripEnd", "size"),
                odo_min=("OdometerTripStart", "min"),
                odo_max=("OdometerTripEnd", "max"),
                km_sum=("trip_km", "sum"),
                km_max_trip=("trip_km", "max"),
                km_neg=("trip_km", lambda s: int((s < 0).sum())),
                dur_h_sum=("dur_h", "sum"),
                date_min=("TripDatetimeStart", "min"),
                date_max=("TripDatetimeStart", "max"),
                trip_num_max=("TripNumber", "max"),
            )
        )
    agg = pd.concat(parts)
    out = agg.groupby(level=[0, 1], observed=True).agg(
        n_trips=("n_trips", "sum"),
        odo_min=("odo_min", "min"),
        odo_max=("odo_max", "max"),
        km_sum=("km_sum", "sum"),
        km_max_trip=("km_max_trip", "max"),
        km_neg=("km_neg", "sum"),
        dur_h_sum=("dur_h_sum", "sum"),
        date_min=("date_min", "min"),
        date_max=("date_max", "max"),
        trip_num_max=("trip_num_max", "max"),
    )
    return out.reset_index()


def per_vehicle_signals(config: str, chunksize: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Agregados por vehículo + conteo de niveles de `Message` por cohorte."""
    columns = [
        VEHICLE_COL,
        "eventTimestamp",
        "OdometerValue",
        "Acumulation",
        "Message",
        "Regenerations",
        "DistanceBetweenRegenerations",
    ]
    veh_parts, msg_parts = [], []
    for chunk in iter_table("signals", config, chunksize=chunksize, columns=columns):
        veh_parts.append(
            chunk.groupby([VEHICLE_COL, "cohort"], observed=True).agg(
                n_signals=("OdometerValue", "size"),
                odo_null=("OdometerValue", lambda s: int(s.isna().sum())),
                odo_min=("OdometerValue", "min"),
                odo_max=("OdometerValue", "max"),
                date_min=("eventTimestamp", "min"),
                date_max=("eventTimestamp", "max"),
                n_regen=("Regenerations", "count"),
                acum_max=("Acumulation", "max"),
            )
        )
        msg_parts.append(
            chunk.groupby(["cohort", "Message"], observed=True, dropna=False).size().rename("n")
        )
    veh = (
        pd.concat(veh_parts)
        .groupby(level=[0, 1], observed=True)
        .agg(
            n_signals=("n_signals", "sum"),
            odo_null=("odo_null", "sum"),
            odo_min=("odo_min", "min"),
            odo_max=("odo_max", "max"),
            date_min=("date_min", "min"),
            date_max=("date_max", "max"),
            n_regen=("n_regen", "sum"),
            acum_max=("acum_max", "max"),
        )
        .reset_index()
    )
    msg = pd.concat(msg_parts).groupby(level=[0, 1], observed=True, dropna=False).sum()
    msg = msg.unstack("cohort", fill_value=0)
    return veh, msg


def duplicate_report(vehicles: pd.DataFrame, trips: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Dos formas de contar el mismo vehículo dos veces, y las dos rompen el split.

    a) El mismo `VehicleCode` aparece en las dos cohortes: la etiqueta se contradice
       y su historia de viajes viene duplicada fila por fila.
    b) Dos códigos distintos con la misma historia (mismo conteo de viajes, mismo
       odómetro, mismas fechas): es un vehículo clonado bajo otro id. Si uno cae en
       train y el otro en validación, la regla 2 se viola sin que el split se entere.
    """
    dup_cohorte = (
        vehicles[vehicles.duplicated(VEHICLE_COL, keep=False)]
        .sort_values(VEHICLE_COL)
        .loc[:, [VEHICLE_COL, "cohort", "IdentificationDate", "ProductionDay", "Engine"]]
    )

    fingerprint = ["n_trips", "odo_max", "km_sum", "date_min", "date_max", "trip_num_max"]
    huellas = trips.drop_duplicates(VEHICLE_COL).copy()
    grupos = huellas.groupby(fingerprint, dropna=False)[VEHICLE_COL].apply(list)
    clones = pd.DataFrame(
        [
            {"grupo": i, "vehiculos": " ".join(sorted(codes)), "n": len(codes)}
            for i, codes in enumerate(g for g in grupos if len(g) > 1)
        ]
    )
    return dup_cohorte, clones


def anchor_analysis(vehicles: pd.DataFrame, trips: pd.DataFrame) -> pd.DataFrame:
    """¿Se puede llevar `IdentificationDate` (días) al calendario de los viajes?

    Si `ProductionDay` son días desde un origen común, entonces
    `primer_viaje - ProductionDay` tiene que ser aproximadamente constante entre
    vehículos. La dispersión de ese residuo es la medida de si el anclaje existe.
    """
    df = vehicles.merge(trips[[VEHICLE_COL, "date_min"]], on=VEHICLE_COL, how="left")
    first_trip_day = df["date_min"].map(
        lambda t: t.tz_localize(None).toordinal() if pd.notna(t) else np.nan
    )
    rows = []
    hipotesis = [
        ("primer_viaje - ProductionDay", df["ProductionDay"]),
        (
            "primer_viaje - (ProductionDay + daysUntilSale)",
            df["ProductionDay"] + df["daysUntilSale"],
        ),
        ("primer_viaje (sin offset)", pd.Series(0.0, index=df.index)),
    ]
    for name, offset in hipotesis:
        resid = first_trip_day - offset
        rows.append(
            {
                "hipotesis": name,
                "n": int(resid.notna().sum()),
                "std_dias": float(resid.std()),
                "iqr_dias": float(resid.quantile(0.75) - resid.quantile(0.25)),
                "rango_dias": float(resid.max() - resid.min()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", default=DEFAULT_SOURCES_CONFIG)
    parser.add_argument("--out", default="experiments/eda")
    parser.add_argument("--chunksize", type=int, default=500_000)
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    out_dir = Path(ensure_dir(args.out))

    vehicles = load_table("vehicles", args.sources)
    vehicles["event_observed"] = (vehicles["cohort"] == "failed").astype(int)

    h("1. INVENTARIO Y JOIN")
    trips = per_vehicle_trips(args.sources, args.chunksize)
    signals, messages = per_vehicle_signals(args.sources, args.chunksize)
    inv = pd.DataFrame(
        [
            {
                "tabla": "vehicles",
                "filas": len(vehicles),
                "vehiculos": vehicles[VEHICLE_COL].nunique(),
            },
            {
                "tabla": "trips",
                "filas": int(trips["n_trips"].sum()),
                "vehiculos": trips[VEHICLE_COL].nunique(),
            },
            {
                "tabla": "signals",
                "filas": int(signals["n_signals"].sum()),
                "vehiculos": signals[VEHICLE_COL].nunique(),
            },
        ]
    )
    print(inv.to_string(index=False))
    v_set, t_set, s_set = (set(d[VEHICLE_COL]) for d in (vehicles, trips, signals))
    print("\nvehiculos en las tres tablas: %d" % len(v_set & t_set & s_set))
    print("en vehicles y no en trips: %d | no en signals: %d" % (len(v_set - t_set), len(v_set - s_set)))
    print("en trips y no en vehicles: %d | en signals y no en vehicles: %d"
          % (len(t_set - v_set), len(s_set - v_set)))

    h("1b. VEHICULOS CONTADOS DOS VECES")
    dup_cohorte, clones = duplicate_report(vehicles, trips)
    print("filas en static: %d | VehicleCode unicos: %d"
          % (len(vehicles), vehicles[VEHICLE_COL].nunique()))
    print("codigos presentes en las DOS cohortes: %d" % dup_cohorte[VEHICLE_COL].nunique())
    if len(dup_cohorte):
        print(dup_cohorte.head(8).to_string(index=False))
    print("\ngrupos de codigos distintos con historia de viajes identica: %d "
          "(vehiculos involucrados: %d)"
          % (len(clones), int(clones["n"].sum()) if len(clones) else 0))
    if len(clones):
        print(clones.head(15).to_string(index=False))

    h("2. ETIQUETA Y CENSURA")
    print(vehicles.groupby("cohort").size().rename("vehiculos").to_frame().to_string())
    print("\ntasa de eventos: %.4f" % vehicles["event_observed"].mean())
    print("\nIdentificationDate (solo cohorte failed):")
    print(vehicles.loc[vehicles.cohort == "failed", "IdentificationDate"].describe().to_string())
    print("\nfraccion de nulos en static, por cohorte:")
    print(vehicles.groupby("cohort").apply(lambda d: d.isna().mean().round(4)).T.to_string())
    print("\ncategoricas por cohorte (proporcion dentro de la cohorte):")
    for col in ["Engine", "ModelSeries", "SalesCountry_cd"]:
        tab = pd.crosstab(vehicles[col], vehicles["cohort"], normalize="columns").round(3)
        tab["n_failed"] = pd.crosstab(vehicles[col], vehicles["cohort"])["failed"]
        print("\n-- %s" % col)
        print(tab.to_string())

    h("3. COBERTURA Y ODOMETRO POR VEHICULO")
    cov = vehicles[[VEHICLE_COL, "cohort", "event_observed"]].merge(
        trips, on=[VEHICLE_COL, "cohort"], how="left"
    )
    cov = cov.merge(signals, on=[VEHICLE_COL, "cohort"], how="left", suffixes=("_trip", "_sig"))
    cov["odo_span_km"] = cov["odo_max_trip"] - cov["odo_min_trip"]
    cov["span_dias"] = (cov["date_max_trip"] - cov["date_min_trip"]).dt.days
    cols = ["n_trips", "n_signals", "odo_min_trip", "odo_max_trip", "odo_span_km", "km_sum", "span_dias"]
    print(cov.groupby("cohort")[cols].describe().T.round(1).to_string())
    print("\nviajes con km negativo (retroceso de odometro): %d" % int(cov["km_neg"].sum()))
    print("vehiculos con al menos un retroceso: %d" % int((cov["km_neg"] > 0).sum()))
    print("señales con OdometerValue nulo: %d (%.2f%%)"
          % (int(cov["odo_null"].sum()), 100 * cov["odo_null"].sum() / cov["n_signals"].sum()))
    print("\ndesfase odo_max(signals) - odo_max(trips) [km]:")
    print((cov["odo_max_sig"] - cov["odo_max_trip"]).describe().round(1).to_string())

    h("4. ANCLAJE TEMPORAL (regla 4)")
    anchor = anchor_analysis(vehicles, trips)
    print(anchor.round(1).to_string(index=False))
    print("\nProductionDay va de %.0f a %.0f dias." % (vehicles["ProductionDay"].min(), vehicles["ProductionDay"].max()))
    print("Si alguna hipotesis tiene dispersion chica frente a ese rango, el evento")
    print("se puede ubicar en calendario y de ahi sobre el eje de km.")

    h("5. SEÑAL DEL POSTRATAMIENTO")
    msg = messages.copy()
    msg.columns = ["n_%s" % c for c in msg.columns]
    total = msg.sum()
    for c in list(msg.columns):
        msg[c.replace("n_", "pct_")] = (msg[c] / total[c] * 100).round(4)
    km_por_cohorte = cov.groupby("cohort")["km_sum"].sum()
    for c in ["failed", "not_failed"]:
        if "n_%s" % c in msg:
            msg["por_1000km_%s" % c] = (msg["n_%s" % c] / km_por_cohorte[c] * 1000).round(4)
    print(msg.sort_values(msg.columns[0], ascending=False).to_string())
    print("\nregeneraciones por vehiculo:")
    print(cov.groupby("cohort")["n_regen"].describe().round(1).to_string())
    print("\nnivel maximo de Acumulation por vehiculo:")
    print(pd.crosstab(cov["acum_max"], cov["cohort"]).to_string())

    salidas = [
        ("per_vehicle.csv", cov),
        ("messages.csv", msg.reset_index()),
        ("anchor.csv", anchor),
        ("inventario.csv", inv),
        ("dup_cohorte.csv", dup_cohorte),
        ("clones.csv", clones),
    ]
    for name, frame in salidas:
        frame.to_csv(out_dir / name, index=False)
    print("\nCSVs en %s" % out_dir)


if __name__ == "__main__":
    main()
