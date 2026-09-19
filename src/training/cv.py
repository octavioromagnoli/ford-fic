"""Loop de validación cruzada. Único camino por el que se entrena un modelo.

Dos garantías que el loop impone y que ningún experimento puede saltearse:

* **Selección de features por prefijo**: entran al modelo las columnas `feat_*` y
  `static_*` y ninguna otra. Así, agregar una feature al panel no requiere tocar
  este archivo ni coordinar con nadie (plan §2, "regla fija").
* **Preprocesamiento ajustado solo con el train del fold**: imputación, escalado y
  one-hot se fitean dentro del pipeline, por fold. Ajustarlos sobre el panel
  entero es leakage aunque no lo parezca.

Los folds no se arman acá: se piden a `src/eval/splits.py` (CLAUDE.md, regla 2).
Este loop solo los recorre —incluidas las repeticiones de la CV repetida, que son
R juegos de folds y no R modelos distintos—.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.eval.metrics import classification_metrics
from src.eval.splits import iter_repeats
from src.models.registry import get_model

logger = logging.getLogger(__name__)

FEATURE_PREFIXES = ("feat_", "static_")
ID_COLUMNS = ("vehicle_id", "cut_odo", "cut_date", "horizon_km", "gap_km")
TARGET_COLUMN = "label"


def select_feature_columns(
    panel: pd.DataFrame, *, prefixes: tuple[str, ...] = FEATURE_PREFIXES
) -> list[str]:
    """Columnas que entran al modelo, por prefijo. Nunca por lista hardcodeada."""
    columns = [c for c in panel.columns if c.startswith(prefixes)]
    if not columns:
        raise ValueError(
            f"El panel no tiene ninguna columna con prefijo {prefixes}. "
            "Revisá el contrato de datos (CLAUDE.md)."
        )
    return columns


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    """Numéricas: mediana + estandarizado. Categóricas: moda + one-hot tolerante."""
    numeric = features.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = [c for c in features.columns if c not in numeric]
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
                ),
                numeric,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=5)),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def run_cv(
    panel: pd.DataFrame,
    splits: dict[str, Any],
    *,
    model_name: str,
    model_params: dict[str, Any] | None = None,
    target_column: str = TARGET_COLUMN,
    feature_prefixes: tuple[str, ...] = FEATURE_PREFIXES,
    strict_splits: bool = True,
    min_valid_positives: int | None = None,
) -> tuple[pd.DataFrame, list[dict[str, float]]]:
    """Corre la CV agrupada y devuelve `(predicciones out-of-fold, métricas por fold)`.

    Con `n_repeats > 1` en los splits, el panel se recorre R veces con folds
    distintos. Eso deja **R juegos de predicciones out-of-fold**, y hay que decidir
    qué se serializa.

    **Decisión: `predictions` tiene una fila por fila del panel, con `score` = promedio
    de las R repeticiones.** El motivo es `src/eval/metrics.py::lead_time_curve()`:
    agrupa por vehículo, ordena por `cut_odo` y busca `k_consecutive` cortes seguidos
    sobre el umbral. Si el parquet trajera R filas por corte, cada corte contaría R
    veces y una alerta aislada pasaría por sostenida — la anticipación saldría
    inflada justo en la figura del pitch. Con el promedio, `lead_time_curve()`
    funciona sin cambios y la curva es una sola.

    Las R pasadas no se pierden: con R > 1 se agregan las columnas `score_r{i}` y
    `fold_r{i}` (una por repetición) más `score_std`, así la dispersión es auditable
    desde el mismo archivo. Con R = 1 las columnas son exactamente las de siempre
    (`score`, `fold`).

    Ojo con qué se compara: el PR-AUC de selección es el **promedio de los PR-AUC de
    cada repetición** (`scripts/train.py`), no el PR-AUC de los scores promediados.
    Promediar scores es un ensamble encubierto y da mejor de lo que el modelo es.

    Las predicciones traen las columnas que necesitan las métricas de anticipación
    (`vehicle_id`, `cut_odo`, `score`, `event_observed`, `time_to_event_km`), así
    F4 consume el parquet sin volver a tocar el panel.
    """
    if target_column not in panel.columns:
        raise KeyError(f"El panel no tiene la columna objetivo `{target_column}`")

    feature_columns = select_feature_columns(panel, prefixes=feature_prefixes)
    logger.info("Features seleccionadas por prefijo: %d", len(feature_columns))

    X = panel[feature_columns]
    y = panel[target_column].astype(int).to_numpy()

    repeat_scores: list[np.ndarray] = []
    repeat_folds: list[np.ndarray] = []
    fold_metrics: list[dict[str, float]] = []

    for repeat, masks in iter_repeats(
        panel, splits, strict=strict_splits, min_valid_positives=min_valid_positives
    ):
        oof_score = np.full(len(panel), np.nan, dtype=float)
        oof_fold = np.full(len(panel), -1, dtype=int)

        for fold, train_mask, valid_mask in masks:
            pipeline = Pipeline(
                [
                    ("prep", build_preprocessor(X)),
                    ("model", get_model(model_name, model_params)),
                ]
            )
            pipeline.fit(X.loc[train_mask], y[train_mask])
            scores = _predict_scores(pipeline, X.loc[valid_mask])

            oof_score[valid_mask] = scores
            oof_fold[valid_mask] = fold

            metrics = classification_metrics(y[valid_mask], scores)
            metrics["fold"] = fold
            metrics["repeat"] = repeat
            fold_metrics.append(metrics)
            logger.info(
                "rep %d | fold %d | n=%d | pos=%d | PR-AUC=%.4f | ROC-AUC=%.4f",
                repeat,
                fold,
                metrics["n"],
                metrics["n_positive"],
                metrics["pr_auc"],
                metrics["roc_auc"],
            )

        if np.isnan(oof_score).any():
            n_missing = int(np.isnan(oof_score).sum())
            raise RuntimeError(
                f"Repetición {repeat}: {n_missing} filas quedaron sin predicción "
                "out-of-fold. Los folds no cubren el panel."
            )
        repeat_scores.append(oof_score)
        repeat_folds.append(oof_fold)

    if not repeat_scores:
        raise RuntimeError("Los splits no tienen ninguna repetición: no se entrenó nada")

    predictions = panel[
        [c for c in (*ID_COLUMNS, "event_observed", "time_to_event_km", target_column) if c in panel]
    ].copy()
    stacked = np.vstack(repeat_scores)
    predictions["score"] = stacked.mean(axis=0)

    if len(repeat_scores) == 1:
        predictions["fold"] = repeat_folds[0]
    else:
        predictions["score_std"] = stacked.std(axis=0, ddof=0)
        for repeat, (scores_r, folds_r) in enumerate(zip(repeat_scores, repeat_folds)):
            predictions[f"score_r{repeat}"] = scores_r
            predictions[f"fold_r{repeat}"] = folds_r
        logger.info(
            "CV repetida: %d repeticiones | `score` es el promedio (ver docstring de run_cv)",
            len(repeat_scores),
        )
    return predictions, fold_metrics


def _predict_scores(pipeline: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Probabilidad de la clase positiva; `decision_function` como fallback."""
    if hasattr(pipeline, "predict_proba"):
        proba = pipeline.predict_proba(X)
        classes = list(pipeline.classes_)
        if 1 in classes:
            return proba[:, classes.index(1)].astype(float)
        return proba[:, -1].astype(float)
    return np.asarray(pipeline.decision_function(X), dtype=float)
