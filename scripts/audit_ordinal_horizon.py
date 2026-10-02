#!/usr/bin/env python
"""Las tres auditorías obligatorias del plan (§9 / f3-modelos-candidatos §0) para una corrida.

    FORD_DATA_DIR=data/v364 python scripts/audit_ordinal_horizon.py \
        --config configs/exp_ordinal_horizon_ext.yaml

No es específica del target ordinal: toma cualquier YAML de experimento y repite su
CV bajo tres controles. Nació con los bins extendidos porque ahí la auditoría (a) deja
de ser trámite y pasa a ser la pregunta central.

**(a) Permutación de la etiqueta, con dos nulos.** Se baraja el desenlace (`label` y
`time_to_event_km` juntos, para que el target ordinal siga siendo coherente con la
etiqueta binaria) y se reentrena todo, de dos maneras distintas:

* **global**, entre todas las filas: destruye la cohorte y el orden interno. Es el
  nulo clásico y su PR-AUC tiene que caer a la tasa base. Si no cae, el que tiene un
  problema es el pipeline, no el modelo.
* **dentro de cada vehículo**: conserva la cohorte —`event_observed` y el
  multiconjunto de distancias al evento de cada vehículo— y destruye solo la relación
  entre las features de un corte y *cuándo* cae el evento respecto de ese corte.

**El nulo dentro del vehículo no cae a la tasa base en este panel, y no puede caer.**
Un vehículo sano no tiene ninguna fila positiva, así que permutar adentro lo deja con
cero positivas: la etiqueta permutada sigue siendo perfectamente predecible desde la
cohorte. El piso correcto para leer ese nulo no es la tasa base sino el propio nulo, y
lo que hay que mirar es si el modelo **le gana**. Si no le gana, su PR-AUC es
separación de cohortes —que en este panel *es* la etiqueta (CLAUDE.md)— y no
anticipación adentro del vehículo. Con bins ordinales extendidos más allá de la
ventana positiva ésta es la auditoría central, y por eso se corre también sobre el
control binario: sin ese contraste no se sabe si el efecto es del target o del panel.

**(b) Los `aux_` de calendario como features.** Se agregan `aux_air_temp_avg`,
`aux_regen_marker_per_1000km` y `aux_static_ProductionDay` con prefijo `feat_` y se
vuelve a correr. Los eventos caen entre sep-2025 y mar-2026 y la exposición sana en
2026: si el ROC salta, el modelo encontró el calendario y el emparejado por odómetro ×
mes no alcanzó.

**(c) Importancias contra la hipótesis física.** Ganancia media por fold del modelo del
`Pipeline`, con los nombres que salen del preprocesador. La hipótesis es térmica y de
uso, progresiva en los últimos ~4.000 km: idle, no llegar a régimen, más lento
(CLAUDE.md).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.train import load_or_make_splits, select_dev, split_options  # noqa: E402
from src.config import load_config, resolve_path, set_seed  # noqa: E402
from src.eval.metrics import classification_metrics  # noqa: E402
from src.eval.splits import iter_folds  # noqa: E402
from src.models.registry import get_model  # noqa: E402
from src.training.cv import build_preprocessor, run_cv, select_feature_columns  # noqa: E402
from src.training.targets import build_target  # noqa: E402

logger = logging.getLogger("audit")

# Las tres del plan: una de calendario, una del marcador cortado el 25-05-2026 y la
# estática con más deriva temporal. Se declaran acá porque son *la* auditoría, no un
# parámetro del experimento; el YAML no las puede apagar.
AUDIT_AUX_COLUMNS = (
    "aux_air_temp_avg",
    "aux_regen_marker_per_1000km",
    "aux_static_ProductionDay",
)
# Familias de la hipótesis física, por el prefijo con el que viven en el panel.
PHYSICAL_FAMILIES = {
    "A · térmica / trayectos cortos": (
        "idle", "below_regime", "cold_start", "short_trip", "chained_trip",
        "engine_temp", "coolant_temp", "trip_distance", "trip_duration",
    ),
    "B · regeneración / DPF": (
        "regen", "dpf", "filter", "distance_between",
    ),
    "C · uso": (
        "speed", "urban", "trips_per", "km_per_day", "rest",
    ),
    "D · severidad / mensajes": (
        "msg", "oil_life", "consumption", "fuel",
    ),
}


def cv_once(panel: pd.DataFrame, splits: dict, cfg: dict) -> dict[str, float]:
    """Una pasada completa de CV con el mismo contrato que `scripts/train.py`."""
    options = split_options(cfg)
    predictions, _ = run_cv(
        panel,
        splits,
        model_name=cfg["model"]["name"],
        model_params=dict(cfg["model"].get("params", {})),
        target=(cfg.get("target") or None),
        strict_splits=bool(cfg.get("splits", {}).get("strict", True)),
        min_valid_positives=options["min_valid_positives"],
    )
    return classification_metrics(
        predictions["label"].to_numpy(dtype=int), predictions["score"].to_numpy(dtype=float)
    )


# El desenlace se mueve **en bloque**. Reasignar `label` sin `time_to_event_km` —o sin
# `event_observed`— deja el panel internamente incoherente (una fila censurada con
# distancia al evento, o al revés) y el nulo terminaría midiendo esa incoherencia en
# vez de la ausencia de señal: el target ordinal directamente se niega a construirse.
OUTCOME_COLUMNS = ("label", "time_to_event_km", "event_observed")


def permute_outcome(
    panel: pd.DataFrame, rng: np.random.Generator, *, scope: str
) -> pd.DataFrame:
    """Baraja el desenlace entre filas; las features quedan donde estaban.

    `scope="global"` permuta entre todas las filas y destruye también la cohorte. Deja
    vehículos con `event_observed` mezclado, que no existen en la realidad: es a
    propósito, porque este nulo no es un panel plausible sino el control del pipeline.

    `scope="within_vehicle"` permuta solo entre los cortes de un mismo vehículo.
    `event_observed` es constante dentro del grupo, así que permutarlo ahí no lo
    cambia, y el multiconjunto de distancias al evento de cada vehículo se conserva:
    la cohorte sobrevive intacta y lo único que se rompe es *cuándo*, dentro de la
    historia de ese vehículo, cae el evento respecto de cada corte.
    """
    shuffled = panel.copy()
    columns = list(OUTCOME_COLUMNS)
    if scope == "global":
        positions = rng.permutation(len(panel))
    elif scope == "within_vehicle":
        positions = np.arange(len(panel))
        for index in panel.groupby("vehicle_id", sort=False).indices.values():
            if len(index) < 2:
                continue
            positions[index] = rng.permutation(index)
    else:
        raise ValueError(f"Alcance de permutación desconocido: {scope!r}")
    shuffled[columns] = panel[columns].to_numpy()[positions]
    return shuffled


def audit_permutation(
    panel: pd.DataFrame, splits: dict, cfg: dict, *, scope: str, n_permutations: int, seed: int
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    runs = []
    for i in range(n_permutations):
        metrics = cv_once(permute_outcome(panel, rng, scope=scope), splits, cfg)
        runs.append(metrics)
        logger.info(
            "  %s %d/%d | PR-AUC=%.4f | ROC-AUC=%.4f",
            scope, i + 1, n_permutations, metrics["pr_auc"], metrics["roc_auc"],
        )
    return {
        "scope": scope,
        "n_permutations": n_permutations,
        "pr_auc_mean": float(np.mean([m["pr_auc"] for m in runs])),
        "pr_auc_std": float(np.std([m["pr_auc"] for m in runs], ddof=0)),
        "pr_auc_max": float(np.max([m["pr_auc"] for m in runs])),
        "roc_auc_mean": float(np.mean([m["roc_auc"] for m in runs])),
        "roc_auc_std": float(np.std([m["roc_auc"] for m in runs], ddof=0)),
        "base_rate": runs[0]["base_rate"],
        "runs": runs,
    }


def audit_aux_leakage(
    panel: pd.DataFrame, splits: dict, cfg: dict
) -> dict[str, object]:
    missing = [c for c in AUDIT_AUX_COLUMNS if c not in panel.columns]
    if missing:
        raise KeyError(f"El panel no tiene las columnas de la auditoría (b): {missing}")
    promoted = panel.copy()
    renamed = {}
    for column in AUDIT_AUX_COLUMNS:
        # `feat_` es lo único que mira `select_feature_columns`; el nombre deja
        # rastro de que la columna entró por la auditoría y no por el set base.
        new = f"feat_auditaux_{column.removeprefix('aux_')}"
        promoted[new] = panel[column]
        renamed[column] = new
    metrics = cv_once(promoted, splits, cfg)
    return {"promoted_columns": renamed, **metrics}


def audit_importances(
    panel: pd.DataFrame, splits: dict, cfg: dict, *, top: int
) -> dict[str, object]:
    """Ganancia media por fold, con los nombres reales del preprocesador."""
    from sklearn.pipeline import Pipeline

    feature_columns = select_feature_columns(panel)
    X = panel[feature_columns]
    y = panel["label"].astype(int).to_numpy()
    target_cfg = cfg.get("target") or {}
    target_name = target_cfg.get("name")
    target_params = dict(target_cfg.get("params") or {})

    totals: dict[str, list[float]] = {}
    for _, train_mask, _ in iter_folds(panel, splits):
        pipeline = Pipeline(
            [
                ("prep", build_preprocessor(X)),
                ("model", get_model(cfg["model"]["name"], dict(cfg["model"].get("params", {})))),
            ]
        )
        fit_target = (
            build_target(target_name, panel, train_mask, **target_params).y
            if target_name
            else y[train_mask]
        )
        pipeline.fit(X.loc[train_mask], fit_target)
        model = pipeline.named_steps["model"]
        if not hasattr(model, "feature_importances_"):
            raise TypeError(
                f"El modelo `{cfg['model']['name']}` no expone `feature_importances_`; "
                "la auditoría (c) necesita un modelo de árboles."
            )
        names = list(pipeline.named_steps["prep"].get_feature_names_out())
        for name, value in zip(names, model.feature_importances_):
            totals.setdefault(name, []).append(float(value))

    table = (
        pd.DataFrame(
            [
                {"feature": name, "gain": float(np.mean(values)), "n_folds": len(values)}
                for name, values in totals.items()
            ]
        )
        .sort_values("gain", ascending=False)
        .reset_index(drop=True)
    )
    total_gain = float(table["gain"].sum()) or 1.0
    table["share"] = table["gain"] / total_gain

    families = {}
    for family, keys in PHYSICAL_FAMILIES.items():
        mask = table["feature"].str.contains("|".join(keys), case=False, regex=True)
        families[family] = float(table.loc[mask, "share"].sum())
    families["sin familia física (estáticas, control de ventana, otras)"] = float(
        1.0 - sum(families.values())
    )
    return {"top": table.head(top).to_dict("records"), "families": families}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="YAML del experimento a auditar")
    parser.add_argument("--n-permutations", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--top", type=int, default=15)
    parser.add_argument("--out", default=None, help="JSON de salida (default: experiments/<name>/audit.json)")
    parser.add_argument("--skip", nargs="*", default=[], choices=["a", "b", "c"])
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    logging.getLogger("train").setLevel(logging.WARNING)
    logging.getLogger("src.training.cv").setLevel(logging.WARNING)

    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))
    panel = pd.read_parquet(resolve_path(cfg["data"]["panel"]))
    panel = select_dev(panel, cfg)
    splits = load_or_make_splits(panel, cfg)
    run_name = cfg.get("name", Path(args.config).stem)

    report: dict[str, object] = {"run": run_name, "config": args.config}
    base = cv_once(panel, splits, cfg)
    report["base"] = base
    print()
    print(f"# Auditorías · {run_name}")
    print(f"Corrida base: PR-AUC={base['pr_auc']:.4f} (lift {base['pr_auc_lift']:.2f}×) | "
          f"ROC-AUC={base['roc_auc']:.4f} | tasa base={base['base_rate']:.4f} | "
          f"{base['n']} filas, {base['n_positive']} positivas")

    if "a" not in args.skip:
        print()
        print("## (a) Permutación de la etiqueta: dos nulos")
        nulls = {
            scope: audit_permutation(
                panel, splits, cfg,
                scope=scope, n_permutations=args.n_permutations, seed=args.seed,
            )
            for scope in ("global", "within_vehicle")
        }
        report["permutation"] = nulls
        glob, within = nulls["global"], nulls["within_vehicle"]
        print()
        print(f"| {'':<34} | PR-AUC          | ROC-AUC |")
        print(f"|{'-' * 36}|-----------------|---------|")
        print(f"| {'modelo':<34} | {base['pr_auc']:.4f}          | {base['roc_auc']:.4f}  |")
        print(f"| {'nulo global (cohorte destruida)':<34} | "
              f"{glob['pr_auc_mean']:.4f} ± {glob['pr_auc_std']:.4f} | {glob['roc_auc_mean']:.4f}  |")
        print(f"| {'nulo intra-vehículo (cohorte viva)':<34} | "
              f"{within['pr_auc_mean']:.4f} ± {within['pr_auc_std']:.4f} | {within['roc_auc_mean']:.4f}  |")
        print(f"| {'tasa base':<34} | {base['base_rate']:.4f}          | 0.5000  |")
        print()
        # El nulo global es el chequeo del pipeline: tiene que caer a la tasa base.
        global_gap = glob["pr_auc_mean"] - base["base_rate"]
        pipeline_ok = global_gap <= 2 * max(glob["pr_auc_std"], 1e-9)
        print(f"Nulo global: {global_gap:+.4f} sobre la tasa base.", end=" ")
        print("Cae donde tiene que caer: el pipeline no filtra."
              if pipeline_ok else
              "⚠ NO cae a la tasa base: el problema es el pipeline, antes que el modelo.")
        # El nulo intra-vehículo es el piso real del modelo: conserva la cohorte y en
        # este panel ningún vehículo sano tiene filas positivas, así que no puede caer
        # a la tasa base. Lo que importa es si el modelo le gana.
        cohort_gap = base["pr_auc"] - within["pr_auc_mean"]
        z = cohort_gap / max(within["pr_auc_std"], 1e-9)
        print(f"Nulo intra-vehículo: {within['pr_auc_mean'] - base['base_rate']:+.4f} sobre la tasa "
              f"base —es cohorte pura, y es el piso real—. El modelo le saca {cohort_gap:+.4f} "
              f"({z:+.2f} desvíos del nulo).")
        if cohort_gap > 2 * max(within["pr_auc_std"], 1e-9):
            print("El modelo le gana a su nulo de cohorte: hay señal de anticipación intra-vehículo.")
        else:
            print("⚠ El modelo NO le gana a su nulo de cohorte: lo que mide el PR-AUC es "
                  "separación de vehículos, no anticipación adentro del vehículo.")
        report["permutation_verdict"] = {
            "pipeline_clean": bool(pipeline_ok),
            "beats_cohort_null": bool(cohort_gap > 2 * max(within["pr_auc_std"], 1e-9)),
            "cohort_gap": float(cohort_gap),
            "cohort_gap_z": float(z),
        }

    if "b" not in args.skip:
        print()
        print("## (b) Los `aux_` de calendario promovidos a `feat_`")
        aux = audit_aux_leakage(panel, splits, cfg)
        report["aux_leakage"] = aux
        print(f"Promovidas: {', '.join(AUDIT_AUX_COLUMNS)}")
        print()
        print(f"| {'':<14} | PR-AUC | ROC-AUC | Brier  |")
        print(f"|{'-' * 16}|--------|---------|--------|")
        print(f"| {'set base':<14} | {base['pr_auc']:.4f} | {base['roc_auc']:.4f}  | {base['brier']:.4f} |")
        print(f"| {'+ aux_':<14} | {aux['pr_auc']:.4f} | {aux['roc_auc']:.4f}  | {aux['brier']:.4f} |")
        print()
        jump = aux["roc_auc"] - base["roc_auc"]
        print(f"Δ ROC = {jump:+.4f}.", end=" ")
        print("⚠ El ROC salta: el modelo encuentra el calendario." if jump > 0.03
              else "El ROC no salta: el emparejado por odómetro × mes aguanta.")

    if "c" not in args.skip:
        print()
        print("## (c) Importancias contra la hipótesis física")
        imp = audit_importances(panel, splits, cfg, top=args.top)
        report["importances"] = imp
        print(f"Ganancia media sobre los {len(splits['folds'])} folds, normalizada.")
        print()
        print(f"| # | {'feature':<44} | share |")
        print(f"|---|{'-' * 46}|-------|")
        for i, row in enumerate(imp["top"], 1):
            print(f"| {i:>1} | {row['feature']:<44} | {100 * row['share']:>4.1f}% |")
        print()
        print(f"| {'familia':<52} | share |")
        print(f"|{'-' * 54}|-------|")
        for family, share in imp["families"].items():
            print(f"| {family:<52} | {100 * share:>4.1f}% |")

    out = Path(args.out) if args.out else (
        Path(resolve_path(cfg.get("output_dir", "experiments"))) / run_name / "audit.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    print()
    print(f"JSON: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
