#!/usr/bin/env python
"""Las tres auditorías obligatorias de F3, sobre el YAML de cualquier experimento.

    python scripts/audit_model.py --config configs/exp_survival_stacking.yaml

`docs/f3-modelos-candidatos.md` §0.4 pide tres cosas de todo modelo antes de creerle un
número. Son las tres formas conocidas de sacar PR-AUC sin haber aprendido nada:

**(a) Permutación dentro del vehículo.** Se barajan, entre los cortes de un mismo
vehículo, las **features**: cada ventana queda pegada a otro corte del mismo auto.
Queda intacto *qué* vehículos fallan y se destruye *cuándo*. El PR-AUC tiene que caer
cerca de la tasa base.

Barajar las features y no la etiqueta es a propósito, y no es un atajo: reordenar los
pares `(X, y)` de un vehículo es la misma reasignación de cualquiera de los dos lados,
pero así **la etiqueta contra la que se mide no se mueve**, y los dos PR-AUC son el
mismo número medido sobre el mismo conjunto. Permutando la etiqueta, el modo de
supervivencia ni se enteraría (su objetivo sale de `time_to_event_km`, no de `label`) y
la referencia y la auditoría quedarían medidas contra etiquetas distintas — que fue lo
que pasó la primera vez que se escribió esto: la permutación daba *más* PR-AUC, porque
"¿es un vehículo que falla?" es un problema más fácil que "¿falla en los próximos H km?".

Ojo con la lectura: en este panel **esta permutación no es un null y el PR-AUC sube**.
No es un bug: deja intacto qué vehículos fallan —que es de donde sale casi todo el
PR-AUC— y de paso le saca a cada vehículo el ruido de qué ventana le tocó, así que lo
que entrena es un ordenador de vehículos más limpio. Medido con tres semillas y con los
dos modelos (`exp_lgbm_panel_v1` incluido) da siempre 0,17–0,21 contra 0,161 de
referencia. Por eso van dos auditorías más alrededor:

* **(a0) el null de verdad**: permutar las features entre *todas* las filas, que rompe
  también el vínculo vehículo ↔ ventana. Ahí sí el PR-AUC tiene que caer a la tasa base,
  y si no cae hay leakage.
* **(a') el aporte del *cuándo***: tomar las predicciones de la referencia y
  reemplazar el score de cada fila por el promedio de su vehículo, sin reentrenar. Lo
  que se pierde es exactamente lo que el modelo sabía del *cuándo*, con todo lo demás
  fijo. Es la cuenta que la permutación no puede hacer.

**(b) Las `aux_` de calendario como `feat_`.** `aux_air_temp_avg`,
`aux_regen_marker_per_1000km` y `aux_static_ProductionDay` miden el calendario, y el
calendario mide la etiqueta (`docs/memoria/f2-eda-revision-y-features.md` §2.1). Si el
ROC salta al dárselas, el emparejado de sanos no alcanzó y el modelo encuentra el atajo.

**(c) Importancias contra la hipótesis física.** Qué mira el modelo, en el orden en que
lo mira. Si arriba hay una columna que no tiene por qué anticipar una degradación de
combustión, hay algo que auditar antes de celebrar (CLAUDE.md, regla 6).

Todo corre con el mismo panel, los mismos folds y el mismo `target:` que el YAML
declara, y solo sobre dev.
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

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.config import ensure_dir, load_config, resolve_path, set_seed  # noqa: E402
from src.eval.metrics import classification_metrics, concordance_index_oof  # noqa: E402
from src.models.registry import get_model  # noqa: E402
from src.training.cv import build_preprocessor, run_cv, select_feature_columns  # noqa: E402
from src.training.targets import build_target  # noqa: E402

# El recorte a dev y la carga de folds salen de `train.py`: si la auditoría los
# reimplementara, podría estar auditando un conjunto que no es el que se entrena.
from scripts.train import load_or_make_splits, select_dev  # noqa: E402

logger = logging.getLogger("audit_model")

#: Las tres columnas de la auditoría (b). Son `aux_` justamente porque miden calendario.
CALENDAR_AUX = ["aux_air_temp_avg", "aux_regen_marker_per_1000km", "aux_static_ProductionDay"]


def permute_all(panel: pd.DataFrame, *, seed: int = 0) -> pd.DataFrame:
    """Baraja las features entre **todas** las filas: el null de verdad.

    Rompe también el vínculo vehículo ↔ ventana, así que no queda nada que aprender y
    el PR-AUC tiene que caer a la tasa base. Si no cae, hay leakage por algún lado y
    ningún otro número de la corrida vale. La permutación *dentro* del vehículo no
    sirve para esto (ver el docstring del módulo): acá no es un null.
    """
    block = [c for c in panel.columns if c.startswith(("feat_", "static_"))]
    rng = np.random.default_rng(seed)
    out = panel.copy()
    out[block] = panel[block].to_numpy()[rng.permutation(len(panel))]
    return out.astype({c: panel[c].dtype for c in block})


def permute_within_vehicle(panel: pd.DataFrame, *, seed: int = 0) -> pd.DataFrame:
    """Baraja las features entre los cortes de cada vehículo; el objetivo no se toca.

    Después de esto, la ventana de un corte es la de otro corte del mismo auto. Lo que
    identifica al vehículo (mercado, nivel de uso, cuántos cortes tiene) sobrevive
    intacto; lo que se rompe es la correspondencia entre la ventana y lo que pasa
    después de ella. Un modelo que solo sabía decir "este auto es de los que fallan"
    puntúa igual; uno que anticipaba, no.

    El objetivo se queda quieto a propósito: así la referencia y la auditoría se miden
    contra exactamente la misma etiqueta (ver el docstring del módulo).
    """
    block = [c for c in panel.columns if c.startswith(("feat_", "static_"))]
    if not block:
        raise ValueError("El panel no tiene columnas `feat_`/`static_` para permutar")
    rng = np.random.default_rng(seed)
    out = panel.copy()
    positions = [out.columns.get_loc(c) for c in block]
    values = panel[block].to_numpy()
    for index in panel.groupby("vehicle_id", observed=True, sort=False).indices.values():
        out.iloc[index, positions] = values[rng.permutation(index)]
    return out.astype({c: panel[c].dtype for c in block})


def promote_aux(panel: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Renombra `aux_x` a `feat_aux_x` para que `cv.py` la seleccione por prefijo."""
    present = [c for c in columns if c in panel.columns]
    return panel.rename(columns={c: f"feat_{c}" for c in present}), present


def evaluate(
    panel: pd.DataFrame, splits: dict, cfg: dict, *, label: str
) -> tuple[dict, pd.DataFrame]:
    model_cfg = cfg["model"]
    predictions, _ = run_cv(
        panel,
        splits,
        model_name=model_cfg["name"],
        model_params=model_cfg.get("params", {}),
        target=cfg.get("target"),
        strict_splits=bool(cfg.get("splits", {}).get("strict", True)),
    )
    metrics = score_metrics(predictions, label=label)
    return metrics, predictions


def score_metrics(predictions: pd.DataFrame, *, label: str) -> dict:
    metrics = classification_metrics(predictions["label"], predictions["score"])
    if "aux_km_observed_after_cut" in predictions:
        metrics["c_index"] = concordance_index_oof(predictions)["c_index"]
    metrics["audit"] = label
    logger.info(
        "%-52s PR-AUC=%.4f (base %.4f) ROC=%.4f",
        label, metrics["pr_auc"], metrics["base_rate"], metrics["roc_auc"],
    )
    return metrics


def within_vehicle_auc(predictions: pd.DataFrame) -> dict[str, float]:
    """ROC del score contra `label` **dentro** de cada vehículo, promediado.

    Es la pregunta que el pitch vende y que el PR-AUC global no aísla: dado un auto que
    va a fallar, ¿el modelo pone más arriba los cortes cercanos al evento que los
    lejanos? Solo entran los vehículos que tienen cortes de las dos clases. Por debajo
    de 0,5 el modelo ordena al revés.
    """
    from sklearn.metrics import roc_auc_score

    values = []
    for _, group in predictions.groupby("vehicle_id", observed=True):
        y = group["label"].to_numpy(dtype=int)
        if len(np.unique(y)) < 2:
            continue
        values.append(roc_auc_score(y, group["score"].to_numpy(dtype=float)))
    return {
        "within_vehicle_roc_mean": float(np.mean(values)) if values else float("nan"),
        "n_vehicles": int(len(values)),
    }


def vehicle_mean_scores(predictions: pd.DataFrame) -> pd.DataFrame:
    """Cada fila con el score **promedio de su vehículo**: el modelo sin el "cuándo"."""
    out = predictions.copy()
    out["score"] = out.groupby("vehicle_id", observed=True)["score"].transform("mean")
    return out


def importances(panel: pd.DataFrame, cfg: dict, *, top: int = 20) -> pd.DataFrame | None:
    """Importancias de un modelo fiteado sobre **todo dev**, con los nombres del preproceso.

    No sale de un fold: es la foto de qué mira el modelo, para contrastarla con la
    hipótesis física. Los números por fold ya están en la CV.
    """
    model = get_model(cfg["model"]["name"], cfg["model"].get("params", {}))
    if not hasattr(model, "feature_importances_") and not hasattr(type(model), "feature_importances_"):
        logger.warning("El modelo `%s` no expone `feature_importances_`", cfg["model"]["name"])
        return None

    X = panel[select_feature_columns(panel)]
    prep = build_preprocessor(X)
    Xt = prep.fit_transform(X)
    names = list(prep.get_feature_names_out())

    target_cfg = cfg.get("target")
    all_rows = np.ones(len(panel), dtype=bool)
    y = (
        build_target(target_cfg["name"], panel, all_rows, **(target_cfg.get("params") or {})).y
        if target_cfg
        else panel["label"].astype(int).to_numpy()
    )
    model.fit(Xt, y)
    names += list(getattr(model, "extra_feature_names_", []))

    values = np.asarray(model.feature_importances_, dtype=float)
    if len(values) != len(names):
        logger.warning("%d importancias para %d nombres: no se puede alinear", len(values), len(names))
        return None
    frame = pd.DataFrame({"feature": names, "importance": values})
    frame["share"] = frame["importance"] / frame["importance"].sum()
    return frame.sort_values("importance", ascending=False).head(top).reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=0, help="Semilla de la permutación (auditoría a)")
    parser.add_argument("--top", type=int, default=20, help="Cuántas features imprime la auditoría (c)")
    parser.add_argument("--out", default=None, help="Dónde dejar el JSON (default: experiments/<run>/audit.json)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(message)s")
    logger.setLevel(logging.INFO)
    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))

    panel = pd.read_parquet(resolve_path(cfg["data"]["panel"]))
    panel = select_dev(panel, cfg)
    splits = load_or_make_splits(panel, cfg)

    reference, ref_predictions = evaluate(panel, splits, cfg, label="referencia (el modelo tal cual)")
    null, _ = evaluate(
        permute_all(panel, seed=args.seed),
        splits, cfg, label=f"(a0) features permutadas entre todos (seed {args.seed})",
    )
    permuted, _ = evaluate(
        permute_within_vehicle(panel, seed=args.seed),
        splits, cfg, label=f"(a) features permutadas dentro del vehículo (seed {args.seed})",
    )
    # Sin retrain: el mismo modelo, con el "cuándo" borrado del score. Separa cuánto del
    # PR-AUC sale de ordenar vehículos y cuánto de ordenar cortes dentro de un vehículo.
    collapsed = score_metrics(
        vehicle_mean_scores(ref_predictions), label="(a') score promediado por vehículo"
    )
    promoted, present = promote_aux(panel, CALENDAR_AUX)
    calendar, _ = evaluate(
        promoted, splits, cfg, label=f"(b) +{len(present)} columnas aux_ de calendario"
    )
    results = [reference, null, permuted, collapsed, calendar]
    within = within_vehicle_auc(ref_predictions)

    table = pd.DataFrame(results)[
        [c for c in ("audit", "pr_auc", "pr_auc_lift", "roc_auc", "brier", "c_index", "base_rate") if c in reference]
    ]
    top = importances(panel, cfg, top=args.top)

    print("\n== Auditorías ==")
    print(table.round(4).to_string(index=False))
    delta_roc = calendar["roc_auc"] - reference["roc_auc"]
    delta_null = null["pr_auc"] - null["base_rate"]
    delta_perm = reference["pr_auc"] - permuted["pr_auc"]
    delta_when = reference["pr_auc"] - collapsed["pr_auc"]
    print(f"\n(a0) null: PR-AUC {null['pr_auc']:.4f} vs tasa base {null['base_rate']:.4f} "
          f"({delta_null:+.4f}) — {'OK, no hay leakage' if abs(delta_null) < 0.02 else 'NO CAE: hay leakage'}")
    print(f"(a) permutación dentro del vehículo: {delta_perm:+.4f} de PR-AUC. "
          + ("El *cuándo* aporta."
             if delta_perm > 0
             else "Sube. En este panel esta permutación NO es un null: deja intacto el "
                  "nivel del vehículo y le saca el ruido de la ventana, así que entrena "
                  "un ordenador de vehículos mejor. Se lee con (a'), no sola."))
    print(f"(a') borrar el *cuándo* del score (sin retrain): {delta_when:+.4f} de PR-AUC — "
          + ("el orden dentro del vehículo suma" if delta_when > 0 else "el orden dentro del vehículo resta"))
    print(f"     ROC dentro del vehículo: {within['within_vehicle_roc_mean']:.4f} sobre "
          f"{within['n_vehicles']} vehículos con cortes de las dos clases "
          f"({'ordena al revés' if within['within_vehicle_roc_mean'] < 0.5 else 'ordena bien'})")
    print(f"(b) las aux_ de calendario mueven el ROC {delta_roc:+.4f} "
          f"({'hay atajo, revisar' if delta_roc > 0.02 else 'no hay salto'})")
    if top is not None:
        print("\n(c) importancias sobre dev:")
        print(top.round(4).to_string(index=False))

    run_name = cfg.get("name") or Path(args.config).stem
    out = Path(args.out) if args.out else ensure_dir(Path(resolve_path(cfg.get("output_dir", "experiments"))) / run_name) / "audit.json"
    out.write_text(
        json.dumps(
            {"config": cfg.get("_config_path"), "seed": args.seed, "audits": results,
             "within_vehicle": within,
             "importances": top.to_dict("records") if top is not None else None},
            indent=2, default=float,
        ),
        encoding="utf-8",
    )
    print(f"\nescrito: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
