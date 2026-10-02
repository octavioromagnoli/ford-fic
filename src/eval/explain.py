"""Explicabilidad del finalista K2: por qué un auto tiene riesgo alto, en términos de su uso.

**Explica al modelo, no al auto.** Todo lo de acá es asociación: dice qué columnas empujaron el
score de K2 hacia arriba, no qué causa la falla. (a′) de K2 es +0,019 —el modelo sabe sobre todo
*qué auto* y poco del *cuándo*— y el "mucho idle ⇒ falla antes en km" resultó ser km/día disfrazado
(`docs/reproducibilidad.md`). Por eso el texto al cliente dice "tu uso se
parece al de los autos que fallaron" y solo muestra una feature cuyo efecto global coincide con la
física del DPF. El preregistro con todas las reglas es `configs/explain_k2.yaml`.

## Las cuatro variantes

* **V1 · TreeSHAP del hazard.** K2 es un LightGBM sobre filas apiladas `(fila, tramo de 500 km)`
  y el score es `1 − Π (1 − h_k)` sobre los 6 tramos de H. V1 apila cada fila en esos 6 tramos
  —igual que `DiscreteSurvivalStacker.predict_proba`—, pide `pred_contrib=True` (log-odds del
  hazard por tramo) y promedia los 6 vectores. Es aditivo exacto sobre el **log-odds medio del
  hazard**, que ordena casi igual que el score pero no es el score.
* **V2 · SHAP del score.** `shap.PermutationExplainer` sobre `predict_proba[:, 1]` del stacker, con
  un fondo de sanos del train del mismo fold (y del mismo mercado si alcanza). Aditivo exacto sobre
  el score, pero caro y con ruido de Monte Carlo: va sobre una muestra.
* **V3 · V1 promediado entre repeticiones.** El promedio de los vectores de V1 de las R
  repeticiones de la CV, a nivel auto. Baja el ruido del sorteo de folds.
* **V4 · V3 por familia.** Suma las contribuciones por familia de `configs/data/features_v1.yaml`
  (A, B, C, D) para las accionables, y en dos bolsas aparte el contexto y los síntomas.

## Qué no es

No es un candidato ni una métrica: no reentrena nada distinto de K2 y no cambia el score, el umbral
ni las alertas. La reconstrucción (`refit_folds`) es el mismo loop que `src/training/cv.py::run_cv`
—mismos helpers de split y de columnas— y `scripts/explain_k2.py` verifica que reproduzca las
predicciones de la corrida antes de explicar nada.

Todo acá es puro: recibe paneles, splits y modelos ya entrenados y devuelve arrays, DataFrames y
dicts. El I/O lo hace `scripts/explain_k2.py`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import spearmanr
from sklearn.pipeline import Pipeline

from src.eval.dashboard_data import vehicle_alerts
from src.eval.splits import iter_repeats
from src.models.registry import get_model
from src.training.cv import FEATURE_PREFIXES, TARGET_COLUMN, build_preprocessor, select_feature_columns
from src.training.targets import build_target, decode_predictions

KEY = ["vehicle_id", "cut_odo"]
ACTIONABLE, SYMPTOM, CONTEXT = "accionable", "sintoma", "contexto"
CLASSES = (ACTIONABLE, SYMPTOM, CONTEXT)
#: Nombre de la familia de V4 de cada bloque de `configs/data/features_v1.yaml`.
FAMILY_KEYS = {"A_thermal": "A", "B_regeneration": "B", "C_usage": "C", "D_severity": "D"}
RISK_LEVELS = ("alto", "medio", "bajo")


# --------------------------------------------------------------------------------------------- #
# Clasificación de las columnas del modelo
# --------------------------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FeatureSpec:
    """Qué es una columna del modelo para el texto al cliente (preregistro, bloque `features`)."""

    name: str
    klass: str                  # accionable | sintoma | contexto
    sign: int                   # signo físico esperado de una accionable (+1, −1, 0 = sin hipótesis)
    family: str                 # A/B/C/D según features_v1.yaml, o "contexto" si no está ahí
    source: tuple[str, ...] = ()
    note: str = ""

    @property
    def group(self) -> str:
        """Bolsa de V4: la familia si es accionable; si no, su clase."""
        return self.family if self.klass == ACTIONABLE else self.klass


def feature_families(features_v1: Mapping[str, Any], *, prefix: str = "feat_") -> dict[str, str]:
    """`{feat_<name>: 'A'|'B'|'C'|'D'}` desde `configs/data/features_v1.yaml` (las `aux` no entran)."""
    out: dict[str, str] = {}
    for block, items in (features_v1.get("families") or {}).items():
        key = FAMILY_KEYS.get(block, block)
        for item in items or []:
            if not item.get("aux", False):
                out[f"{prefix}{item['name']}"] = key
    return out


def feature_specs(explain_cfg: Mapping[str, Any], families: Mapping[str, str]) -> dict[str, FeatureSpec]:
    """Las `FeatureSpec` del preregistro, validadas: clase conocida y signo en {−1, 0, +1}."""
    specs: dict[str, FeatureSpec] = {}
    for name, entry in (explain_cfg.get("features") or {}).items():
        klass = str(entry.get("class"))
        if klass not in CLASSES:
            raise ValueError(f"`{name}`: clase `{klass}` desconocida (opciones: {CLASSES})")
        if klass == ACTIONABLE and "sign" not in entry:
            raise ValueError(f"`{name}` es accionable y no declara `sign`: el signo físico es obligatorio")
        sign = int(entry.get("sign", 0))
        if sign not in (-1, 0, 1):
            raise ValueError(f"`{name}`: signo {sign} fuera de {{−1, 0, +1}}")
        specs[name] = FeatureSpec(
            name=name, klass=klass, sign=sign, family=families.get(name, CONTEXT),
            source=tuple(entry.get("source") or ()), note=str(entry.get("note", "")),
        )
    return specs


def check_coverage(units: Sequence[str], specs: Mapping[str, FeatureSpec]) -> None:
    """Toda columna que el modelo ve tiene clase. Una sin clasificar no puede caer en el texto."""
    missing = [u for u in units if u not in specs]
    if missing:
        raise KeyError(f"Columnas del modelo sin clasificar en el preregistro: {missing}")


def names_of(specs: Mapping[str, FeatureSpec], klass: str, *, signed: bool = False) -> list[str]:
    return [n for n, s in specs.items() if s.klass == klass and (not signed or s.sign != 0)]


# --------------------------------------------------------------------------------------------- #
# §3 · Reconstrucción fuera de fold (el mismo loop que run_cv)
# --------------------------------------------------------------------------------------------- #
@dataclass
class FoldModel:
    """Un fold reentrenado: el Pipeline ajustado, sus máscaras y el score de su validación."""

    repeat: int
    fold: int
    pipeline: Pipeline
    train_mask: np.ndarray
    valid_mask: np.ndarray
    valid_score: np.ndarray

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.pipeline.predict_proba(X), dtype=float)[:, 1]


def refit_folds(
    panel: pd.DataFrame,
    splits: dict[str, Any],
    *,
    model_name: str,
    model_params: dict[str, Any] | None = None,
    target: dict[str, Any] | None = None,
    strict_splits: bool = True,
    min_valid_positives: int | None = None,
    feature_prefixes: tuple[str, ...] = FEATURE_PREFIXES,
    repeat_offset: int = 0,
) -> tuple[list[str], list[FoldModel]]:
    """Reentrena cada fold de cada repetición y **devuelve los modelos**, que `run_cv` no guarda.

    Es el cuerpo del loop de `src/training/cv.py::run_cv` con `preprocessing: standard`, paso por
    paso: las máscaras salen de `iter_repeats` (src/eval/splits.py), las columnas de
    `select_feature_columns`, el preprocesado de `build_preprocessor`, el modelo del registry y el
    `y` de `build_target`. Si alguien cambia `run_cv`, el chequeo de `check_setup.py` que compara
    los dos sobre el panel dummy falla; y `scripts/explain_k2.py` compara contra las predicciones
    reales antes de explicar nada.

    `repeat_offset` corre el número de repetición (las extra de la estabilidad de V3 van después
    de las oficiales).
    """
    if TARGET_COLUMN not in panel.columns:
        raise KeyError(f"El panel no tiene la columna objetivo `{TARGET_COLUMN}`")
    feature_columns = select_feature_columns(panel, prefixes=feature_prefixes)
    X = panel[feature_columns]
    y = panel[TARGET_COLUMN].astype(int).to_numpy()
    target_name = (target or {}).get("name")
    target_params = dict((target or {}).get("params") or {})

    models: list[FoldModel] = []
    for repeat, masks in iter_repeats(panel, splits, strict=strict_splits, min_valid_positives=min_valid_positives):
        for fold, train_mask, valid_mask in masks:
            model = get_model(model_name, model_params)
            named = bool(getattr(model, "wants_feature_names", False))
            pipeline = Pipeline([("prep", build_preprocessor(X, named_output=named)), ("model", model)])
            fit_target = (
                build_target(target_name, panel, train_mask, **target_params).y if target_name else y[train_mask]
            )
            pipeline.fit(X.loc[train_mask], fit_target)
            score = decode_predictions(target_name, pipeline, X.loc[valid_mask], **target_params).score
            models.append(FoldModel(repeat=int(repeat) + repeat_offset, fold=int(fold), pipeline=pipeline,
                                    train_mask=np.asarray(train_mask, bool), valid_mask=np.asarray(valid_mask, bool),
                                    valid_score=np.asarray(score, dtype=float)))
    return feature_columns, models


def oof_scores(models: Sequence[FoldModel], n_rows: int) -> dict[int, np.ndarray]:
    """`{repetición: score fuera de fold por fila}`; falla si una fila quedó sin predicción."""
    out: dict[int, np.ndarray] = {}
    for m in models:
        scores = out.setdefault(m.repeat, np.full(n_rows, np.nan))
        scores[m.valid_mask] = m.valid_score
    for repeat, scores in out.items():
        if np.isnan(scores).any():
            raise RuntimeError(f"Repetición {repeat}: {int(np.isnan(scores).sum())} filas sin predicción fuera de fold")
    return out


def reproduction_report(
    models: Sequence[FoldModel], panel: pd.DataFrame, predictions: pd.DataFrame, *, atol: float
) -> dict[str, Any]:
    """¿El modelo reconstruido es el de la corrida? Fold por fold, contra `score_r{i}`.

    Alinea por `(vehicle_id, cut_odo)`, no por posición. Devuelve la diferencia absoluta máxima,
    el ρ de Spearman por fold y los dos veredictos del preregistro: exacto (≤ `atol`) y, si no,
    de rango (ρ = 1 en cada fold).
    """
    keyed = predictions.set_index(KEY)
    records = []
    for m in models:
        keys = pd.MultiIndex.from_frame(panel.loc[m.valid_mask, KEY])
        column = f"score_r{m.repeat}" if f"score_r{m.repeat}" in keyed else "score"
        reference = keyed[column].reindex(keys).to_numpy(dtype=float)
        if np.isnan(reference).any():
            raise KeyError(f"rep {m.repeat} fold {m.fold}: filas sin score en las predicciones de la corrida")
        diff = np.abs(m.valid_score - reference)
        rho = float(spearmanr(m.valid_score, reference).statistic) if len(reference) > 2 else float("nan")
        records.append({"repeat": m.repeat, "fold": m.fold, "n": int(len(reference)),
                        "max_abs_diff": float(diff.max()), "spearman": rho})
    max_abs = max(r["max_abs_diff"] for r in records)
    exact = bool(max_abs <= atol)
    rank = bool(all(r["spearman"] >= 1.0 - 1e-12 for r in records))
    return {"atol": float(atol), "max_abs_diff": float(max_abs), "exact": exact, "rank_identical": rank,
            "passed": exact or rank, "mode": "exact" if exact else ("spearman_1" if rank else "failed"),
            "folds": records}


# --------------------------------------------------------------------------------------------- #
# §4 · Variantes
# --------------------------------------------------------------------------------------------- #
@dataclass
class Explanation:
    """Contribuciones por fila: `output ≈ base + Σ phi` en la escala que explica la variante."""

    phi: np.ndarray                      # (n, n_units)
    units: list[str]
    base: np.ndarray                     # (n,)
    output: np.ndarray                   # (n,) log-odds medio del hazard (V1) o score (V2)
    risk: np.ndarray | None = None       # (n,) 1 − S(H|x) reconstruido (V1) o el score (V2)
    scale: str = "log-odds medio del hazard"

    def frame(self, index: pd.Index | None = None) -> pd.DataFrame:
        return pd.DataFrame(self.phi, columns=list(self.units), index=index)

    @property
    def additivity_error(self) -> np.ndarray:
        return np.abs(self.phi.sum(axis=1) + self.base - self.output)


def output_names(pipeline: Pipeline) -> list[str]:
    """Nombres de las columnas del preprocesado que ve el hazard, en orden (el bin lo agrega quien llama)."""
    prep, model = pipeline[:-1], pipeline[-1]
    names = [str(n) for n in prep.get_feature_names_out()]
    index = getattr(model, "_column_index_", None)
    if index is not None:
        names = [names[i] for i in index]
    return names


def unit_of(names: Sequence[str], feature_columns: Sequence[str]) -> list[str]:
    """La columna del panel que corresponde a cada columna del booster.

    Una numérica es ella misma; una dummy del one-hot (`static_SalesCountry_cd_CNTRY_3`) es su
    categórica; lo que no es del panel (el bin del hazard) queda con su nombre.
    """
    columns = list(feature_columns)
    longest_first = sorted(columns, key=len, reverse=True)
    out = []
    for name in names:
        if name in columns:
            out.append(name)
            continue
        parent = next((c for c in longest_first if name.startswith(f"{c}_")), None)
        out.append(parent if parent is not None else name)
    return out


def collapse_units(values: np.ndarray, names: Sequence[str], feature_columns: Sequence[str]) -> tuple[list[str], np.ndarray]:
    """Suma las columnas que son la misma unidad (las dummies de una categórica). Es aditivo."""
    units = unit_of(names, feature_columns)
    order = list(dict.fromkeys(units))
    position = {u: i for i, u in enumerate(order)}
    out = np.zeros((values.shape[0], len(order)))
    for j, unit in enumerate(units):
        out[:, position[unit]] += values[:, j]
    return order, out


def score_stack(model: Any, Xt: Any) -> tuple[np.ndarray, int]:
    """Las filas apiladas con que `DiscreteSurvivalStacker.predict_proba` compone el score.

    Una por `(fila, tramo de H)`, con el borde izquierdo del tramo como última columna. Es la
    misma construcción que el modelo; `tree_shap_hazard` verifica además que el riesgo que sale de
    acá reproduzca `predict_proba`.
    """
    X = model._select(Xt)
    edges = model._train_edges_[: model._score_bins_]
    n_rows, n_bins = len(X), len(edges)
    rows = np.repeat(np.arange(n_rows), n_bins)
    return np.column_stack([X[rows], np.tile(edges, n_rows)]), n_bins


def tree_shap_hazard(pipeline: Pipeline, X: pd.DataFrame, feature_columns: Sequence[str]) -> Explanation:
    """V1: TreeSHAP del hazard por tramo, promediado sobre los tramos que componen el score."""
    prep, model = pipeline[:-1], pipeline[-1]
    if getattr(model, "backend", None) != "lightgbm" or not hasattr(model, "_score_bins_"):
        raise TypeError("V1 necesita un DiscreteSurvivalStacker ajustado con backend lightgbm")
    stacked, n_bins = score_stack(model, prep.transform(X))
    contrib = np.asarray(model.booster_.predict(stacked, pred_contrib=True), dtype=float)
    raw = np.asarray(model.booster_.predict(stacked, raw_score=True), dtype=float).reshape(len(X), n_bins)
    phi = contrib[:, :-1].reshape(len(X), n_bins, -1).mean(axis=1)
    base = contrib[:, -1].reshape(len(X), n_bins).mean(axis=1)
    hazard = np.clip(expit(raw), 0.0, 1 - 1e-12)
    risk = 1.0 - np.exp(np.log1p(-hazard).sum(axis=1))
    names = output_names(pipeline) + list(getattr(model, "extra_feature_names_", []))
    units, phi_units = collapse_units(phi, names, feature_columns)
    return Explanation(phi=phi_units, units=units, base=base, output=raw.mean(axis=1), risk=risk)


def permutation_shap_score(
    pipeline: Pipeline,
    X: pd.DataFrame,
    background: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    max_evals: int | str = "auto",
    seed: int = 0,
) -> Explanation:
    """V2: `shap.PermutationExplainer` sobre el score `1 − S(H|x)`, en el espacio preprocesado.

    El fondo es el que se le pase (sanos del train del mismo fold; lo arma el script). Se explica
    en el espacio que sale del preprocesado del fold —imputado y escalado, que son transformaciones
    por columna y crecientes— para que el enmascarado no tenga que lidiar con la categórica en texto.
    """
    import shap  # import adentro: dependencia pesada, solo para V2

    prep, model = pipeline[:-1], pipeline[-1]
    Xt = np.asarray(prep.transform(X), dtype=float)
    Bt = np.asarray(prep.transform(background), dtype=float)

    def predict(Z: np.ndarray) -> np.ndarray:
        return np.asarray(model.predict_proba(Z), dtype=float)[:, 1]

    explainer = shap.PermutationExplainer(predict, shap.maskers.Independent(Bt, max_samples=len(Bt)), seed=seed)
    result = explainer(Xt, max_evals=max_evals, silent=True)
    units, phi = collapse_units(np.asarray(result.values, dtype=float), output_names(pipeline), feature_columns)
    score = predict(Xt)
    return Explanation(phi=phi, units=units, base=np.asarray(result.base_values, dtype=float).reshape(-1),
                       output=score, risk=score, scale="score (1 − S(3.000 km | x))")


def family_sums(phi: pd.DataFrame, specs: Mapping[str, FeatureSpec]) -> pd.DataFrame:
    """V4: las contribuciones sumadas por bolsa (familia de las accionables, contexto, síntoma)."""
    check_coverage(list(phi.columns), specs)
    groups = pd.Series({u: specs[u].group for u in phi.columns})
    return phi.T.groupby(groups, sort=True).sum().T


def healthy_background(
    train: pd.DataFrame, market: str, *, min_vehicles: int, rng: np.random.Generator,
    market_column: str = "static_SalesCountry_cd",
) -> pd.Index:
    """Fondo de V2: un corte al azar por vehículo sano del train, del mismo mercado si alcanza.

    Si el mercado tiene menos de `min_vehicles` sanos en el train, se completa con sanos del otro
    mercado (en orden aleatorio). Devuelve el índice de las filas elegidas.
    """
    healthy = train.loc[train["event_observed"].eq(0)]
    if healthy.empty:
        raise ValueError("El train del fold no tiene sanos: no hay fondo para V2")
    # Barajar y quedarse con la primera fila de cada vehículo = un corte al azar por vehículo.
    picks = healthy.iloc[rng.permutation(len(healthy))].groupby("vehicle_id", sort=True).head(1)
    picks = picks.sort_values("vehicle_id", kind="stable")
    same = picks.index[picks[market_column].astype(str).eq(str(market)).to_numpy()]
    if len(same) >= min_vehicles:
        return pd.Index(same)
    other = picks.index[picks[market_column].astype(str).ne(str(market)).to_numpy()]
    fill = rng.permutation(np.asarray(other))[: max(0, min_vehicles - len(same))]
    return pd.Index(np.concatenate([np.asarray(same), fill]))


# --------------------------------------------------------------------------------------------- #
# Agregación por vehículo: los cortes que dispararon la alerta
# --------------------------------------------------------------------------------------------- #
def explained_cuts(rows: pd.DataFrame, repeat: int, threshold: float, k: int) -> pd.DataFrame:
    """Qué cortes explican a cada auto, con la regla de alerta de la curva y del dashboard.

    La alerta es la de `src/eval/dashboard_data.py::vehicle_alerts` —la misma cuenta que
    `first_alert_lead_times`—: el primer corte que arranca una racha de `k` cortes con score ≥
    umbral. Si el auto alerta se explican esos `k` cortes; si no, su corte de score máximo.
    Devuelve una fila por `(auto, corte explicado)` con el estado del auto y su nivel de riesgo:
    alto (alertó), medio (algún corte sobre el umbral, sin racha) o bajo.
    """
    score = f"score_r{repeat}"
    alerts = vehicle_alerts(rows, repeat, threshold, k).set_index("vehicle_id")
    records = []
    for vid, g in rows.groupby("vehicle_id", sort=False):
        g = g.sort_values("cut_odo", kind="stable")
        cuts = g["cut_odo"].to_numpy(dtype=float)
        s = g[score].to_numpy(dtype=float)
        info = alerts.loc[vid]
        if bool(info["alerted"]):
            start = int(np.flatnonzero(cuts == float(info["alert_cut_odo"]))[0])
            chosen = cuts[start:start + k]
        else:
            chosen = cuts[[int(np.argmax(s))]]
        level = "alto" if bool(info["alerted"]) else ("medio" if s.max() >= threshold else "bajo")
        for cut in chosen:
            records.append({"vehicle_id": vid, "cut_odo": float(cut), "alerted": bool(info["alerted"]),
                            "failed": bool(info["failed"]), "outcome": info["outcome"],
                            "max_score": float(info["max_score"]), "risk_level": level,
                            "threshold": float(threshold), "repeat": int(repeat)})
    return pd.DataFrame(records)


def vehicle_means(row_values: pd.DataFrame, cuts: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """El promedio de `columns` sobre los cortes explicados de cada auto (índice: vehicle_id)."""
    merged = cuts[KEY].merge(row_values[KEY + list(columns)], on=KEY, how="left", validate="one_to_one")
    if merged[list(columns)].isna().all(axis=1).any():
        raise KeyError("Hay cortes explicados sin fila en las contribuciones: ¿otra repetición u otro panel?")
    return merged.groupby("vehicle_id", sort=True)[list(columns)].mean()


# --------------------------------------------------------------------------------------------- #
# Referencia de flota (borrado, mensaje y control de calendario)
# --------------------------------------------------------------------------------------------- #
def _fleet_wide(frame: pd.DataFrame, values: pd.DataFrame, features: Sequence[str], months: Sequence[str],
                month: pd.Series, market_column: str) -> pd.DataFrame:
    from src.data.landmark import FM_PREFIX, FM_WEIGHT

    columns: dict[str, np.ndarray] = {}
    for feature in features:
        v = values[feature].to_numpy(dtype=float)
        for m in months:
            here = (month == m).to_numpy()
            columns[f"{FM_PREFIX}{feature}__{m}"] = np.where(here, v, np.nan)
            columns[f"{FM_PREFIX}{FM_WEIGHT}{feature}__{m}"] = here.astype(float)
    out = pd.DataFrame(columns, index=frame.index)
    out[market_column] = frame[market_column].astype(str).to_numpy()
    return out


def fleet_reference(
    train: pd.DataFrame,
    target: pd.DataFrame,
    features: Sequence[str],
    *,
    market_column: str = "static_SalesCountry_cd",
    month_column: str = "cut_date",
    min_ref_vehicles: int = 10,
    levels: Sequence[Sequence[str]] = (("market", "month"), ("month",), ()),
) -> pd.DataFrame:
    """Mediana de los sanos del **train** por mercado × mes del corte, para cada fila de `target`.

    Es `src/training/transformers.py::FleetReferenceNormalizer` tal cual —la referencia se ajusta
    con los vehículos sanos del train y nada de validación, un voto por vehículo y mes, con la
    jerarquía mercado × mes → mes → global—, alimentado con una celda por fila: el mes de su corte.
    La mediana misma sale de transformar un valor 0 (el desvío de 0 es −referencia), así que
    existe aunque el auto no tenga dato en esa feature. Un corte sin fecha no tiene mes: cae a la
    mediana global de los sanos del train (una por vehículo). En el panel de K2 no hay ninguno.
    """
    from src.training.transformers import OUTPUT_PREFIX, FleetReferenceNormalizer

    def month_of(frame: pd.DataFrame) -> pd.Series:
        return pd.to_datetime(frame[month_column]).dt.strftime("%Y-%m")

    train_month, target_month = month_of(train), month_of(target)
    months = sorted(set(train_month.dropna()) | set(target_month.dropna()))
    healthy = train["event_observed"].to_numpy(dtype=int) == 0
    groups = train["vehicle_id"].astype(str).to_numpy()
    normalizer = FleetReferenceNormalizer(market_column=market_column, min_ref_vehicles=int(min_ref_vehicles),
                                          levels=tuple(tuple(level) for level in levels))
    normalizer.fit(_fleet_wide(train, train, features, months, train_month, market_column), (healthy, groups))
    zeros = pd.DataFrame(0.0, index=target.index, columns=list(features))
    deviation = normalizer.transform(_fleet_wide(target, zeros, features, months, target_month, market_column))
    reference = -deviation[[f"{OUTPUT_PREFIX}{f}" for f in features]]
    reference.columns = list(features)
    undated = target_month.isna().to_numpy()
    if undated.any():
        per_vehicle = train.loc[healthy].groupby("vehicle_id")[list(features)].median()
        reference.loc[undated] = reference.loc[undated].fillna(per_vehicle.median())
    return reference


def dev_only_guard(vehicle_columns: Sequence[pd.Series], test_split: Mapping[str, Any]) -> dict[str, Any]:
    """Falla si algún output nombra un vehículo de test o de fuera de dev (CLAUDE.md, regla 2).

    Recibe las columnas `vehicle_id` de todo lo que se va a escribir. Devuelve el resumen para el
    JSON cuando pasa.
    """
    seen = set().union(*(set(map(str, c.dropna())) for c in vehicle_columns))
    test_vehicles = set(map(str, test_split.get("test_vehicles", [])))
    dev_vehicles = set(map(str, test_split.get("dev_vehicles", [])))
    leaked = sorted(seen & test_vehicles)
    outside = sorted(seen - dev_vehicles) if dev_vehicles else []
    if leaked or outside:
        raise RuntimeError(f"Vehículos fuera de dev en los outputs: test={leaked[:5]} · fuera de dev={outside[:5]}")
    return {"vehicles_in_outputs": len(seen), "test_vehicles_found": leaked, "outside_dev": outside}


# --------------------------------------------------------------------------------------------- #
# §5 · Criterios
# --------------------------------------------------------------------------------------------- #
def spearman(a: Sequence[float], b: Sequence[float]) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or np.ptp(a[ok]) == 0 or np.ptp(b[ok]) == 0:
        return float("nan")
    return float(spearmanr(a[ok], b[ok]).statistic)


def top_k(vector: pd.Series, candidates: Sequence[str], k: int) -> frozenset[str]:
    """Las `k` candidatas de mayor contribución **positiva** (las que suben el riesgo)."""
    v = vector.reindex(list(candidates)).dropna()
    v = v[v > 0].sort_values(ascending=False, kind="stable")
    return frozenset(v.index[:k])


def jaccard(a: frozenset, b: frozenset) -> float:
    """|A ∩ B| / |A ∪ B|, con Jaccard(∅, ∅) = 1 (dos réplicas que no señalan nada coinciden)."""
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def stability_table(
    replicates: Sequence[pd.DataFrame], vehicles: Sequence[str], candidates: Sequence[str], k: int
) -> pd.DataFrame:
    """Por auto: Jaccard medio del top `k` entre pares de réplicas y acuerdo de signo.

    Acuerdo de signo: sobre la unión de los top `k` de las réplicas, la fracción de unidades cuya
    contribución tiene el mismo signo (no nulo) en todas. Una réplica es un DataFrame indexado por
    vehicle_id con una columna por unidad.
    """
    records = []
    for vid in vehicles:
        vectors = [r.loc[vid] for r in replicates]
        tops = [top_k(v, candidates, k) for v in vectors]
        union = sorted(set().union(*tops))
        if union:
            signs = np.array([[np.sign(v.get(u, 0.0)) for u in union] for v in vectors])
            agree = float(((signs == signs[0]).all(axis=0) & (signs[0] != 0)).mean())
        else:
            agree = 1.0
        records.append({"vehicle_id": vid,
                        "jaccard": float(np.mean([jaccard(a, b) for a, b in combinations(tops, 2)])),
                        "sign_agreement": agree, "n_union": len(union),
                        "tops": [sorted(t) for t in tops]})
    return pd.DataFrame(records)


def association(values: pd.DataFrame, phi: pd.DataFrame, features: Sequence[str]) -> pd.Series:
    """ρ de Spearman entre el valor de cada feature y su contribución (filas con valor finito)."""
    return pd.Series({f: spearman(values[f], phi[f]) for f in features}, dtype=float)


def plausibility_table(
    rho: pd.Series, rho_by_repeat: pd.DataFrame, specs: Mapping[str, FeatureSpec], *, min_abs_rho: float
) -> pd.DataFrame:
    """Signo global observado de cada accionable contra el físico esperado, y si entra al texto.

    Entra si: signo esperado ≠ 0, signo observado = esperado, |ρ| ≥ `min_abs_rho` y el ρ de cada
    repetición tiene el mismo signo (estable). Si no, `reason` dice por qué queda fuera.
    """
    records = []
    for feature, value in rho.items():
        spec = specs[feature]
        observed = int(np.sign(value)) if np.isfinite(value) else 0
        per = rho_by_repeat[feature].to_numpy(dtype=float) if feature in rho_by_repeat else np.array([value])
        stable = bool(observed != 0 and np.all(np.sign(per) == observed))
        if spec.sign == 0:
            reason = "sin hipótesis física"
        elif not np.isfinite(value) or abs(value) < min_abs_rho:
            reason = "efecto débil (|ρ| < umbral)"
        elif observed != spec.sign:
            reason = "contraria a la física"
        elif not stable:
            reason = "inestable entre repeticiones"
        else:
            reason = ""
        records.append({"feature": feature, "expected_sign": spec.sign, "rho": float(value),
                        "observed_sign": observed, "rho_min": float(np.nanmin(per)) if per.size else np.nan,
                        "rho_max": float(np.nanmax(per)) if per.size else np.nan, "stable": stable,
                        "aligned": reason == "", "reason": reason})
    return pd.DataFrame(records).set_index("feature")


@dataclass
class ErasureBlock:
    """Los cortes explicados de un auto en una repetición, con el modelo y la referencia de su fold."""

    predict: Callable[[pd.DataFrame], np.ndarray]
    X: pd.DataFrame          # columnas crudas del panel que ve el Pipeline
    reference: pd.DataFrame  # medianas de flota de esas filas, columnas accionables


@dataclass
class ErasureUnit:
    key: tuple
    top: list[str]
    blocks: list[ErasureBlock] = field(default_factory=list)


def _drops(block: ErasureBlock, subsets: Sequence[Sequence[str]]) -> np.ndarray:
    """Caída media del score de los cortes del bloque al reemplazar cada subconjunto por la referencia."""
    n, m = len(block.X), len(subsets)
    base = block.predict(block.X)
    Z = block.X.iloc[np.tile(np.arange(n), m)].reset_index(drop=True)
    which = np.repeat(np.arange(m), n)
    for feature in sorted({f for s in subsets for f in s}):
        chosen = np.array([feature in s for s in subsets])[which]
        replacement = np.tile(block.reference[feature].to_numpy(dtype=float), m)
        Z[feature] = np.where(chosen, replacement, Z[feature].to_numpy(dtype=float))
    after = block.predict(Z).reshape(m, n)
    return (base[None, :] - after).mean(axis=1)


def erasure_test(units: Sequence[ErasureUnit], candidates: Sequence[str], *, n_null: int, seed: int) -> dict[str, Any]:
    """Prueba de borrado: ¿sacar las top accionables baja el score más que sacar unas al azar?

    Es un contrafactual **del modelo**, no del auto: dice cuánto cambia la predicción si esas
    columnas valieran la mediana de los sanos comparables, no qué le pasaría al auto. El
    estadístico es la caída media sobre las unidades; el nulo reemplaza, en cada unidad, la misma
    cantidad de accionables elegidas al azar entre `candidates`. p = (1 + #{nulo ≥ obs}) / (1 + n).
    """
    rng = np.random.default_rng(seed)
    usable = [u for u in units if u.top]
    if not usable:
        return {"n_units": 0, "observed": float("nan"), "p_value": float("nan"), "null_mean": float("nan")}
    candidates = list(candidates)
    observed = np.zeros(len(usable))
    null = np.zeros((n_null, len(usable)))
    for i, unit in enumerate(usable):
        draws = [[str(f) for f in rng.choice(candidates, size=len(unit.top), replace=False)] for _ in range(n_null)]
        per_block = np.array([_drops(b, [unit.top] + draws) for b in unit.blocks])  # (blocks, 1 + n_null)
        mean = per_block.mean(axis=0)
        observed[i], null[:, i] = mean[0], mean[1:]
    obs, null_means = float(observed.mean()), null.mean(axis=1)
    return {"n_units": len(usable), "n_skipped": len(units) - len(usable), "observed": obs,
            "null_mean": float(null_means.mean()), "null_p95": float(np.quantile(null_means, 0.95)),
            "p_value": float((1 + np.sum(null_means >= obs)) / (1 + n_null)),
            "unit_drops": observed.tolist()}


# --------------------------------------------------------------------------------------------- #
# §6 · Mensaje al cliente
# --------------------------------------------------------------------------------------------- #
@dataclass
class MessageFactor:
    feature: str
    contribution: float
    value: float
    reference: float


def message_factors(
    contribution: pd.Series,
    *,
    values: pd.Series,
    references: pd.Series,
    specs: Mapping[str, FeatureSpec],
    allowed: set[str] | frozenset[str],
    replicate_contributions: Sequence[pd.Series] = (),
    max_factors: int = 3,
) -> list[MessageFactor]:
    """Hasta `max_factors` factores que pasan todos los filtros del preregistro, por contribución.

    Un factor entra si: es accionable; tiene signo físico ≠ 0; está en `allowed` (plausible y
    estable en el global); su contribución es > 0 y del mismo signo en todas las réplicas; su valor
    está observado y del lado riesgoso de la mediana sana comparable ((valor − mediana)·signo > 0).
    """
    factors = []
    for feature, phi in contribution.sort_values(ascending=False, kind="stable").items():
        spec = specs.get(feature)
        if spec is None or spec.klass != ACTIONABLE or spec.sign == 0 or feature not in allowed:
            continue
        if not (np.isfinite(phi) and phi > 0):
            continue
        if any(not (np.isfinite(r.get(feature, np.nan)) and r.get(feature) > 0) for r in replicate_contributions):
            continue
        value, reference = float(values.get(feature, np.nan)), float(references.get(feature, np.nan))
        if not (np.isfinite(value) and np.isfinite(reference)) or (value - reference) * spec.sign <= 0:
            continue
        factors.append(MessageFactor(feature, float(phi), value, reference))
        if len(factors) >= max_factors:
            break
    return factors


#: `format` de explain_texts.yaml → (escala, decimales, unidad).
VALUE_FORMATS = {"pct": (100.0, 0, "%"), "km1": (1.0, 1, " km"), "kmh0": (1.0, 0, " km/h"),
                 "degc0": (1.0, 0, " °C"), "per1000": (1.0, 0, " cada 1.000 km"), "int": (1.0, 0, ""),
                 "min0": (1.0, 0, " min"), "pp1": (100.0, 1, " pp")}


def format_value(value: float, fmt: str) -> str:
    """Formato de un valor para el cliente: coma decimal y punto de miles en el número, y la unidad."""
    if not np.isfinite(value):
        return "—"
    scale, digits, unit = VALUE_FORMATS.get(fmt, (1.0, 2, ""))
    number = f"{value * scale:,.{digits}f}".replace(",", "_").replace(".", ",").replace("_", ".")
    return number + unit


def feature_label(unit: str, texts: Mapping[str, Any]) -> str:
    """Nombre legible de una unidad: el `label` de la accionable o el de `figure_labels`."""
    entry = (texts.get("features") or {}).get(unit) or {}
    return entry.get("label") or (texts.get("figure_labels") or {}).get(unit) or unit.removeprefix("feat_")


def value_pair(value: float, reference: float, fmt: str) -> str:
    """`16 vs 19 km/h`: el valor del auto y la mediana sana, con la unidad una sola vez."""
    if fmt == "pct":
        return f"{format_value(value, fmt)} vs {format_value(reference, fmt)}"
    unit = VALUE_FORMATS.get(fmt, (1.0, 2, ""))[2]
    return f"{format_value(value, fmt).removesuffix(unit)} vs {format_value(reference, fmt).removesuffix(unit)}{unit}"


def waterfall_steps(
    vector: pd.Series,
    classes: Mapping[str, str],
    *,
    top: int,
    allowed_in_message: set[str] | frozenset[str],
    texts: Mapping[str, Any],
    values: pd.Series | None = None,
    references: pd.Series | None = None,
) -> pd.DataFrame:
    """Los escalones del waterfall de un auto: las `top` accionables por |φ|, el resto, el contexto
    y los síntomas (una barra cada uno). La misma cuenta para la figura del informe y el dashboard.

    `classes` es `{unidad: accionable | contexto | sintoma}`. Contexto y síntomas nunca se abren:
    no son "lo que hacés", y el gráfico no los nombra uno por uno. Las accionables que llegan al
    mensaje (`allowed_in_message`) se abren siempre, aunque queden fuera del top por |φ|: el gráfico
    tiene que mostrar cada factor que el mensaje nombra.
    """
    actionable = vector[[u for u in vector.index if classes.get(u) == ACTIONABLE]]
    ranked = list(actionable.abs().sort_values(ascending=False, kind="stable").index)
    opened = ranked[:top] + [u for u in ranked[top:] if u in allowed_in_message]
    order = pd.Index(opened + [u for u in ranked if u not in opened])
    top = len(opened)
    steps = []
    for unit in order[:top]:
        label = feature_label(unit, texts)
        fmt = ((texts.get("features") or {}).get(unit) or {}).get("format")
        if fmt and values is not None and references is not None and np.isfinite(values.get(unit, np.nan)):
            label += f" ({value_pair(float(values[unit]), float(references[unit]), fmt)})"
        steps.append({"step": label, "kind": ACTIONABLE, "value": float(actionable[unit]),
                      "in_message": unit in allowed_in_message, "unit": unit})
    rest = order[top:]
    if len(rest):
        steps.append({"step": f"otras {len(rest)} accionables", "kind": ACTIONABLE,
                      "value": float(actionable[rest].sum()), "in_message": False, "unit": ""})
    for klass, label in ((CONTEXT, "contexto (odómetro, mercado, ritmo, tramo)"),
                         (SYMPTOM, "síntomas (estado del filtro, mensajes, aceite, consumo)")):
        members = [u for u in vector.index if classes.get(u) == klass]
        steps.append({"step": label, "kind": klass, "value": float(vector[members].sum()), "in_message": False,
                      "unit": ""})
    return pd.DataFrame(steps)


def horizon_weeks(gap_km: float, horizon_km: float, km_per_day: float) -> tuple[int, int] | None:
    """El horizonte `[G, G + H]` en semanas al ritmo del auto; None si el ritmo no se conoce."""
    if not (np.isfinite(km_per_day) and km_per_day > 0):
        return None
    lo = max(1, int(round(gap_km / km_per_day / 7.0)))
    hi = max(lo, int(round((gap_km + horizon_km) / km_per_day / 7.0)))
    return lo, hi


def render_vehicle_message(
    *,
    risk_level: str,
    factors: Sequence[MessageFactor],
    specs: Mapping[str, FeatureSpec],
    texts: Mapping[str, Any],
    k: int,
    gap_km: float,
    horizon_km: float,
    km_per_day: float,
    symptom_contribution: float | None,
    symptom_min: float = 0.0,
) -> str:
    """El mensaje al cliente, en texto plano: riesgo, horizonte, factores, recomendación, aviso.

    Solo las accionables pueden aparecer: un factor de contexto o de síntoma que llegue acá se
    descarta aunque se lo pase (los síntomas tienen su línea aparte, sin nombrar columnas). Con
    riesgo bajo no se listan factores. Si el auto tiene riesgo y ningún factor pasó los filtros,
    se dice, no se rellena.
    """
    if risk_level not in RISK_LEVELS:
        raise ValueError(f"nivel de riesgo `{risk_level}` desconocido: {RISK_LEVELS}")
    feature_texts = texts.get("features") or {}
    lines = [texts["risk_levels"][risk_level].format(k=k)]
    weeks = horizon_weeks(gap_km, horizon_km, km_per_day)
    start_km, end_km = format_value(gap_km, "int"), format_value(gap_km + horizon_km, "int")
    if weeks is None:
        lines.append(texts["horizon_no_rate"].format(gap_km=start_km, end_km=end_km))
    else:
        lines.append(texts["horizon"].format(gap_km=start_km, end_km=end_km, w_lo=weeks[0], w_hi=weeks[1],
                                             kmpd=f"{km_per_day:.0f}"))
    if risk_level != "bajo":
        # Última defensa: solo accionables con signo físico y con frase; contexto, síntomas y las
        # accionables sin hipótesis no se nombran aunque lleguen acá.
        shown = [f for f in factors
                 if specs.get(f.feature) is not None and specs[f.feature].klass == ACTIONABLE
                 and specs[f.feature].sign != 0 and "phrase" in (feature_texts.get(f.feature) or {})]
        if shown:
            lines.append(texts["factors_intro"])
            recommendations: list[str] = []
            for f in shown:
                t = feature_texts[f.feature]
                lines.append("- " + t["phrase"].format(value=format_value(f.value, t["format"]),
                                                       reference=format_value(f.reference, t["format"])))
                rec = texts["recommendations"][t["recommendation"]]
                if rec not in recommendations:
                    recommendations.append(rec)
            lines.append(texts["recommendations_intro"])
            lines.extend(f"- {r}" for r in recommendations)
        else:
            lines.append(texts["no_factors"])
        if symptom_contribution is not None and np.isfinite(symptom_contribution):
            key = "up" if symptom_contribution > symptom_min else "down"
            lines.append(texts["symptom_line"][key])
    lines.append(texts["disclaimer"])
    return "\n".join(lines)
