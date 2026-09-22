"""Mixture cure model por hito post-venta: *qué auto* (incidencia) × *cuándo* (latencia).

Fase 5 del cure model (Track B). La especificación es la del preregistro
(`docs/memoria/f3-preregistro-cure.md` §4, P0 y P1). Este archivo la implementa y no
decide nada que no esté escrito ahí.

## Qué modela

Para el vehículo i en el hito L (una fila del panel de hitos):

- **incidencia** π_i = π_L(x_i): probabilidad de que el vehículo sea de los que van a ser
  identificados, dado que llegó al hito sin evento. Es una logística sobre las cuatro
  features desviadas contra la flota;
- **latencia** S_u(t) = exp(−(t/λ)^k): cuándo, entre los susceptibles, en días desde la
  venta. Es una Weibull **sin covariables**: la latencia con covariables quedó fuera del
  preregistro;
- **exposición** (e_i, x_i]: el tramo en que el registro de eventos podía anotarlo
  (`aux_dss_entry`, `aux_dss_exit`). Fuera de la ventana no hay riesgo observable.

Con S̃_i = S_u(x_i)/S_u(e_i) (la entrada tardía), la verosimilitud de una fila es

    L_i = [π_i · h_u(x_i) · S̃_i]^δ_i · [1 − π_i + π_i · S̃_i]^(1−δ_i)

Las filas con x_i ≤ e_i no entran: no informan sobre el evento. Se ajusta un modelo por
hito (`per_landmark`).

## EM (Sy & Taylor 2000; Peng & Dear 2000)

El dato que falta es si un sano es susceptible. El paso E le da a cada fila el peso

    w_i = δ_i + (1 − δ_i) · π_i·S̃_i / (1 − π_i + π_i·S̃_i)

y el paso M se separa en dos problemas que no comparten parámetros:

- **incidencia:** una logística con respuesta fraccionaria w_i. Es lo mismo que la
  regresión ponderada sobre datos aumentados del prompt: cada censurado aporta (1, w_i)
  y (0, 1 − w_i), y cada evento (1, 1);
- **latencia:** el máximo de Σ δ·log h_u(x) − Σ w·[(x/λ)^k − (e/λ)^k] en (k, λ). Para
  k fijo, θ = λ^(−k) tiene óptimo cerrado, así que se busca en una sola dimensión
  (log k) sobre la verosimilitud perfilada.

Los dos pasos M arrancan del valor anterior y **nunca aceptan uno peor**: es un EM
generalizado, y por eso el objetivo no puede bajar. Se verifica en cada iteración, y si
baja, el ajuste falla en vez de devolver un número.

## Firth + FLIC (P1)

La incidencia penaliza con Firth (1993): maximiza Q + ½·log det(XᵀVX), con
V = diag(π(1 − π)). Hay un detalle que hace que esto sea un EM de verdad: en los datos
aumentados, los dos pesos de un censurado suman 1, así que XᵀVX **no depende de w**. La
penalización es una función fija de β (el prior de Jeffreys de la logística), y el EM
maximiza ℓ + ½·log det(XᵀVX) con la misma garantía de monotonía que sin penalizar.

El paso M es el Newton de Heinze & Schemper (2002): la dirección es
(XᵀVX)⁻¹·Xᵀ[(w − π) + h∘(½ − π)], con h la diagonal de V^½X(XᵀVX)⁻¹XᵀV^½, y el paso se
parte a la mitad hasta que el objetivo no baje.

**FLIC** (Puhr et al. 2017) va al final. Firth corre las probabilidades hacia ½, así
que con el EM ya convergido las pendientes quedan fijas (como offset) y el intercepto
se reestima por máxima verosimilitud **sin** penalizar. La latencia, que nunca estuvo
penalizada, se vuelve a ajustar en ese mismo EM: sus pesos w dependen de π, y dejarla
fija sería maximizar con pesos de otro modelo. En esa segunda fase lo que no puede bajar
es ℓ sin penalizar.

## Pesos unitarios (P0)

π = sigmoid(a + b·s), con s = Σ signo_j·z_j (z estandarizada con el train del hito) y
solo a y b libres (Dawes 1979). **P0 se evalúa con s**, que sale como la columna auxiliar
`unit_weight_score`. El cure solo lo calibra, y dentro de un hito es monótono en s si
b > 0.

(a, b) se estiman con el **mismo Firth + FLIC que P1**, no por máxima verosimilitud a
secas. Con una ventana de 191 días no alcanza el seguimiento para separar "susceptible
que todavía no falló" de "curado". La MV se va al borde (π → 1, información singular) en
2 de 11 submuestras del 80% de dev en el hito de 30 días, y Firth en ninguna: su
penalización vale −∞ en ese borde. Es un diagnóstico de convergencia, sin β ni D1/D2.
Solo cambia la calibración que se reporta de P0, porque sus D1/D2 salen de s.

## Qué no esperar

Con latencia sin covariables, **dentro de un hito el orden es el de π_L**. El cure model
no discrimina mejor que su incidencia. Lo que aporta es otra cosa:

- qué sanos cuentan como negativos y cuánto (el peso w);
- probabilidades comparables entre hitos (para la primera alerta);
- que el riesgo de un sano baje solo a medida que la ventana pasa sin evento.

El score es P(identificación en (L+G, L+G+H] | vivo en el hito, x) =
π_L(x)·[1 − S_u(L+G+H)/S_u(L+G)], con la latencia del reloj post-venta **sin** ventana:
es lo que valdría en producción. La ventana es del registro de este dataset.

## Aproximación declarada

π_L(x) no depende de la entrada e_i. Es la verosimilitud del preregistro, que es exacta
para las filas que entran en L + G. Una fila que entra más tarde (la ventana se abrió
después de L + G) usa el mismo π_L, aunque parte de sus susceptibles ya habría pasado
su evento antes de que el registro existiera.

## Pipeline interno

Con `preprocessing: none` en el YAML, `cv.py` no preprocesa nada y el modelo trae lo
suyo, ajustado con el train del fold:

    FleetReferenceNormalizer → por hito: SimpleImputer(mediana) → StandardScaler → EM

La imputación y la escala son **por hito**, con todas las filas de train de ese hito
(preregistro §3).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from src.training.targets import CURE_FIELDS
from src.training.transformers import FleetReferenceNormalizer

logger = logging.getLogger(__name__)

INCIDENCES = ("unit_weight", "firth", "tabpfn")
LATENCIES = ("weibull",)
LANDMARK_COLUMN = "feat_landmark_day"
UNIT_WEIGHT_SCORE = "unit_weight_score"
# Cotas de la forma de la Weibull en la búsqueda del paso M. No son un hiperparámetro:
# están lejos de cualquier forma plausible (la Fase 1 da un riesgo que crece con los
# días, k > 1) y existen para que x^k no desborde. Si el óptimo cae en una, se avisa.
SHAPE_BOUNDS = (0.05, 50.0)
# Cuánto puede bajar el objetivo por redondeo antes de declarar roto el EM (relativo).
MONOTONE_RTOL = 1e-9


# --------------------------------------------------------------------------- #
# Logística (con respuesta fraccionaria y pesos), con o sin Firth
# --------------------------------------------------------------------------- #
def _log_sigmoids(eta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """`(log p, log(1 − p))` sin perder precisión en las colas."""
    return -np.logaddexp(0.0, -eta), -np.logaddexp(0.0, eta)


def _information(X: np.ndarray, p: np.ndarray, weights: np.ndarray) -> np.ndarray:
    v = weights * p * (1.0 - p)
    return (X * v[:, None]).T @ X


def firth_penalty(X: np.ndarray, p: np.ndarray, weights: np.ndarray | None = None) -> float:
    """½·log det(XᵀVX), V = diag(m·p(1 − p)). −inf si la información es singular."""
    weights = np.ones(len(p)) if weights is None else weights
    sign, logdet = np.linalg.slogdet(_information(X, p, weights))
    return 0.5 * logdet if sign > 0 else -np.inf


def logistic_objective(
    beta: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    *,
    weights: np.ndarray | None = None,
    offset: np.ndarray | None = None,
    firth: bool = False,
) -> float:
    """Σ m·[y·log p + (1 − y)·log(1 − p)] (+ la penalización de Firth)."""
    weights = np.ones(len(y)) if weights is None else weights
    eta = X @ beta + (0.0 if offset is None else offset)
    log_p, log_q = _log_sigmoids(eta)
    value = float(np.sum(weights * (y * log_p + (1.0 - y) * log_q)))
    if firth:
        value += firth_penalty(X, expit(eta), weights)
    return value


@dataclass(frozen=True)
class LogisticFit:
    coef: np.ndarray
    objective: float
    n_iter: int
    converged: bool


def fit_logistic(
    X: np.ndarray,
    y: np.ndarray,
    *,
    weights: np.ndarray | None = None,
    offset: np.ndarray | None = None,
    firth: bool = False,
    start: np.ndarray | None = None,
    max_iter: int = 100,
    tol: float = 1e-10,
    max_halvings: int = 40,
) -> LogisticFit:
    """Newton con paso a la mitad sobre `logistic_objective` (Heinze & Schemper 2002).

    `y` puede ser fraccionaria (el paso M del EM la usa con y = w). La dirección es
    (XᵀVX)⁻¹·U*, con U* = Xᵀ[m∘(y − p) + h∘(½ − p)] el gradiente exacto del objetivo
    penalizado. Como XᵀVX es definida positiva, es una dirección de ascenso, y partir el
    paso hasta que el objetivo no baje garantiza que **nunca devuelve algo peor que
    `start`**: eso es lo que hace monótono al EM.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    weights = np.ones(len(y)) if weights is None else np.asarray(weights, dtype=float)
    offset_ = np.zeros(len(y)) if offset is None else np.asarray(offset, dtype=float)
    beta = np.zeros(X.shape[1]) if start is None else np.asarray(start, dtype=float).copy()

    def objective(b: np.ndarray) -> float:
        return logistic_objective(b, X, y, weights=weights, offset=offset_, firth=firth)

    current = objective(beta)
    converged = False
    n_iter = 0
    for n_iter in range(1, max_iter + 1):
        p = expit(X @ beta + offset_)
        info = _information(X, p, weights)
        try:
            info_inv = np.linalg.inv(info)
        except np.linalg.LinAlgError as exc:
            raise np.linalg.LinAlgError(
                "La información de la logística es singular: hay una covariable constante "
                "o colineal con el intercepto en este hito."
            ) from exc
        score = X.T @ (weights * (y - p))
        if firth:
            v = weights * p * (1.0 - p)
            leverage = v * np.einsum("ij,jk,ik->i", X, info_inv, X)
            score = score + X.T @ (leverage * (0.5 - p))
        step = info_inv @ score
        size = 1.0
        for _ in range(max_halvings):
            candidate = beta + size * step
            value = objective(candidate)
            if value >= current:
                break
            size *= 0.5
        else:
            # Ni el paso más chico mejora: estamos en el máximo, a precisión de máquina.
            converged = True
            break
        change = float(np.max(np.abs(size * step)))
        beta, current = candidate, value
        if change < tol:
            converged = True
            break
    return LogisticFit(coef=beta, objective=current, n_iter=n_iter, converged=converged)


# --------------------------------------------------------------------------- #
# Latencia Weibull con entrada tardía
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class WeibullLatency:
    """S_u(t) = exp(−(t/λ)^k), en días desde la venta."""

    shape: float
    scale: float

    def log_survival(self, t: np.ndarray | float) -> np.ndarray:
        return -np.power(np.asarray(t, dtype=float) / self.scale, self.shape)

    def log_hazard(self, t: np.ndarray | float) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        return np.log(self.shape) - np.log(self.scale) + (self.shape - 1.0) * (np.log(t) - np.log(self.scale))

    def conditional_survival(self, start: np.ndarray | float, end: np.ndarray | float) -> np.ndarray:
        """S_u(end)/S_u(start)."""
        return np.exp(self.log_survival(end) - self.log_survival(start))


def latency_objective(
    latency: WeibullLatency, entry: np.ndarray, exit_: np.ndarray, event: np.ndarray, weight: np.ndarray
) -> float:
    """Parte de latencia del paso M: Σ δ·log h(x) + Σ w·[log S(x) − log S(e)] (w = 1 en eventos)."""
    log_ratio = latency.log_survival(exit_) - latency.log_survival(entry)
    return float(np.sum(np.where(event == 1, latency.log_hazard(exit_), 0.0)) + np.sum(weight * log_ratio))


def fit_weibull_latency(
    entry: np.ndarray,
    exit_: np.ndarray,
    event: np.ndarray,
    weight: np.ndarray,
    *,
    start: WeibullLatency | None = None,
    shape_bounds: tuple[float, float] = SHAPE_BOUNDS,
) -> WeibullLatency:
    """Máximo de `latency_objective` en (k, λ), por la verosimilitud perfilada en k.

    Con θ = λ^(−k), el objetivo es D·log k + (k − 1)·Σδ·log x + D·log θ − θ·A(k), con
    D = Σδ y A(k) = Σ w·(x^k − e^k). Para k fijo, θ̂ = D/A(k), y queda una búsqueda
    acotada en log k. Los tiempos se reescalan por la mediana de las salidas con evento
    para que x^k no desborde; el óptimo no cambia.

    Con `start`, nunca devuelve algo peor que `start` (paso M generalizado).
    """
    event = np.asarray(event, dtype=int)
    n_events = int(event.sum())
    if n_events == 0:
        raise ValueError("La latencia no se puede estimar sin eventos en riesgo en este hito")
    reference = float(np.median(exit_[event == 1]))
    xs, es = np.asarray(exit_, float) / reference, np.asarray(entry, float) / reference
    sum_log_x = float(np.sum(np.log(xs[event == 1])))

    def accumulated(k: float) -> float:
        return float(np.sum(weight * (np.power(xs, k) - np.power(es, k))))

    def negative_profile(log_k: float) -> float:
        k = float(np.exp(log_k))
        total = accumulated(k)
        if not total > 0:
            return np.inf
        return -(n_events * np.log(k) + (k - 1.0) * sum_log_x + n_events * np.log(n_events / total) - n_events)

    low, high = np.log(shape_bounds[0]), np.log(shape_bounds[1])
    result = minimize_scalar(negative_profile, bounds=(low, high), method="bounded",
                             options={"xatol": 1e-10, "maxiter": 500})
    shape = float(np.exp(result.x))
    if min(abs(result.x - low), abs(result.x - high)) < 1e-6:
        logger.warning("latencia Weibull: la forma k=%.3g quedó en la cota de la búsqueda %s", shape, shape_bounds)
    theta = n_events / accumulated(shape)
    candidate = WeibullLatency(shape=shape, scale=float(reference * theta ** (-1.0 / shape)))
    if start is not None:
        if latency_objective(candidate, entry, exit_, event, weight) < latency_objective(start, entry, exit_, event, weight):
            return start
    return candidate


# --------------------------------------------------------------------------- #
# El EM
# --------------------------------------------------------------------------- #
def cure_loglik(
    eta: np.ndarray, latency: WeibullLatency, entry: np.ndarray, exit_: np.ndarray, event: np.ndarray
) -> float:
    """ℓ observada: Σ δ·[log π + log h(x) + log S̃] + (1 − δ)·log(1 − π + π·S̃)."""
    log_pi, log_not_pi = _log_sigmoids(eta)
    log_ratio = latency.log_survival(exit_) - latency.log_survival(entry)
    with_event = log_pi + latency.log_hazard(exit_) + log_ratio
    censored = np.logaddexp(log_not_pi, log_pi + log_ratio)
    return float(np.sum(np.where(event == 1, with_event, censored)))


def susceptible_weights(
    eta: np.ndarray, latency: WeibullLatency, entry: np.ndarray, exit_: np.ndarray, event: np.ndarray
) -> np.ndarray:
    """Paso E: w = δ + (1 − δ)·π·S̃ / (1 − π + π·S̃)."""
    log_pi, log_not_pi = _log_sigmoids(eta)
    log_ratio = latency.log_survival(exit_) - latency.log_survival(entry)
    log_susceptible = log_pi + log_ratio
    posterior = np.exp(log_susceptible - np.logaddexp(log_not_pi, log_susceptible))
    return np.where(event == 1, 1.0, posterior)


@dataclass
class CureFit:
    """Resultado de un ajuste por hito: parámetros, verosimilitud y la traza del EM."""

    coef: np.ndarray
    latency: WeibullLatency
    loglik: float
    history: list[float] = field(default_factory=list)
    flic_history: list[float] = field(default_factory=list)
    n_iter: int = 0
    n_iter_flic: int = 0
    converged: bool = False
    converged_flic: bool | None = None
    firth: bool = False


def _em(
    design: np.ndarray,
    entry: np.ndarray,
    exit_: np.ndarray,
    event: np.ndarray,
    *,
    coef: np.ndarray,
    latency: WeibullLatency,
    firth: bool,
    free: slice,
    max_iter: int,
    tol: float,
    inner_max_iter: int,
    inner_tol: float,
) -> tuple[np.ndarray, WeibullLatency, list[float], int, bool]:
    """Iteraciones E/M hasta |Δ objetivo| < tol. `free` son los coeficientes que se mueven.

    El objetivo es ℓ (+ ½·log det(XᵀVX) con `firth`). Los coeficientes que no son
    `free` entran como offset: es la fase FLIC.
    """
    X_free = design[:, free]

    def objective(b: np.ndarray, lat: WeibullLatency) -> float:
        eta = design @ b
        value = cure_loglik(eta, lat, entry, exit_, event)
        if firth:
            value += firth_penalty(design, expit(eta))
        return value

    current = objective(coef, latency)
    history = [current]
    converged = False
    n_iter = 0
    for n_iter in range(1, max_iter + 1):
        eta = design @ coef
        weights = susceptible_weights(eta, latency, entry, exit_, event)
        offset = eta - X_free @ coef[free]
        step = fit_logistic(X_free, weights, offset=offset, firth=firth, start=coef[free],
                            max_iter=inner_max_iter, tol=inner_tol)
        coef = coef.copy()
        coef[free] = step.coef
        latency = fit_weibull_latency(entry, exit_, event, weights, start=latency)
        value = objective(coef, latency)
        if value < current - MONOTONE_RTOL * max(1.0, abs(current)):
            raise RuntimeError(
                f"El EM bajó el objetivo en la iteración {n_iter} ({current:.10f} → {value:.10f}): "
                "algún paso M no es un ascenso. No se puede confiar en este ajuste."
            )
        history.append(value)
        delta, current = abs(value - current), value
        if delta < tol:
            converged = True
            break
    return coef, latency, history, n_iter, converged


def fit_cure_em(
    design: np.ndarray,
    entry: np.ndarray,
    exit_: np.ndarray,
    event: np.ndarray,
    *,
    firth: bool = False,
    flic: bool = False,
    max_iter: int = 1000,
    tol: float = 1e-7,
    inner_max_iter: int = 100,
    inner_tol: float = 1e-10,
) -> CureFit:
    """EM del mixture cure model sobre las filas **en riesgo** de un hito.

    `design` trae el intercepto en la primera columna. Inicialización del preregistro:
    la latencia por máxima verosimilitud con solo los eventos (con entrada tardía), y π
    igual a la proporción de eventos entre las filas en riesgo.
    """
    design = np.asarray(design, dtype=float)
    entry, exit_ = np.asarray(entry, dtype=float), np.asarray(exit_, dtype=float)
    event = np.asarray(event, dtype=int)
    if not np.allclose(design[:, 0], 1.0):
        raise ValueError("La primera columna del diseño tiene que ser el intercepto")
    if (exit_ <= entry).any():
        raise ValueError("`fit_cure_em` recibe solo filas en riesgo (salida > entrada)")
    n_events = int(event.sum())
    if n_events == 0 or n_events == len(event):
        raise ValueError(f"El hito tiene {n_events} eventos en {len(event)} filas en riesgo: no hay mezcla que estimar")

    coef = np.zeros(design.shape[1])
    rate = n_events / len(event)
    coef[0] = np.log(rate / (1.0 - rate))
    latency = fit_weibull_latency(entry, exit_, event, event.astype(float))
    options = dict(max_iter=max_iter, tol=tol, inner_max_iter=inner_max_iter, inner_tol=inner_tol)

    coef, latency, history, n_iter, converged = _em(
        design, entry, exit_, event, coef=coef, latency=latency, firth=firth,
        free=slice(None), **options)
    fit = CureFit(coef=coef, latency=latency, loglik=cure_loglik(design @ coef, latency, entry, exit_, event),
                  history=history, n_iter=n_iter, converged=converged, firth=firth)
    if firth and flic:
        coef, latency, flic_history, n_flic, converged_flic = _em(
            design, entry, exit_, event, coef=coef, latency=latency, firth=False,
            free=slice(0, 1), **options)
        fit.coef, fit.latency = coef, latency
        fit.loglik = cure_loglik(design @ coef, latency, entry, exit_, event)
        fit.flic_history, fit.n_iter_flic, fit.converged_flic = flic_history, n_flic, converged_flic
    return fit


# --------------------------------------------------------------------------- #
# El estimador
# --------------------------------------------------------------------------- #
@dataclass
class LandmarkModel:
    """Lo ajustado en un hito: imputación, escala y el EM."""

    imputer: SimpleImputer
    scaler: StandardScaler
    fit: CureFit
    n_rows: int
    n_at_risk: int
    n_events: int
    susceptible_fraction: float


def _unpack_cure_target(y: Any) -> dict[str, np.ndarray]:
    y = np.asarray(y)
    names = y.dtype.names
    if names is None or any(f not in names for f in CURE_FIELDS):
        raise TypeError(
            "`cure_mixture` espera el target estructurado de `cure_window`, no `label`. "
            "Poné `target: {name: cure_window}` en el YAML del experimento."
        )
    return {name: y[name] for name in CURE_FIELDS}


class CureMixtureModel(ClassifierMixin, BaseEstimator):
    """Cure model por hito con normalización contra la flota adentro.

    Parámetros (todos desde el YAML del experimento):

    - `incidence`: `firth` (P1), `unit_weight` (P0) o `tabpfn` (P2, condicional).
    - `unit_weight_signs`: `{columna: ±1}` para `unit_weight`. Las claves son las columnas
      que salen del normalizador (`fleet_<feature>`) y tienen que ser exactamente esas.
    - `latency`: solo `weibull`.
    - `per_landmark`: un ajuste por hito (el preregistrado). Con `False`, uno solo para
      todas las filas.
    - `landmark_column`: la variable de diseño que separa los hitos. No es covariable.
    - `normalizer_params`: se pasan a `FleetReferenceNormalizer`.
    - `firth_flic`: reestimar el intercepto al final. Vale para `firth` y para la
      calibración de `unit_weight`, que también se penaliza (ver arriba).
    - `max_iter`, `tol`: del EM (sobre el objetivo). `inner_max_iter`, `inner_tol`: del
      Newton de la incidencia.
    - `random_state`: P0 y P1 no tienen azar; queda para P2.
    """

    def __init__(
        self,
        *,
        incidence: str = "firth",
        unit_weight_signs: Mapping[str, float] | None = None,
        latency: str = "weibull",
        per_landmark: bool = True,
        landmark_column: str = LANDMARK_COLUMN,
        normalizer_params: Mapping[str, Any] | None = None,
        firth_flic: bool = True,
        max_iter: int = 1000,
        tol: float = 1e-7,
        inner_max_iter: int = 100,
        inner_tol: float = 1e-10,
        random_state: int = 42,
    ) -> None:
        self.incidence = incidence
        self.unit_weight_signs = unit_weight_signs
        self.latency = latency
        self.per_landmark = per_landmark
        self.landmark_column = landmark_column
        self.normalizer_params = normalizer_params
        self.firth_flic = firth_flic
        self.max_iter = max_iter
        self.tol = tol
        self.inner_max_iter = inner_max_iter
        self.inner_tol = inner_tol
        self.random_state = random_state

    # -- interno -------------------------------------------------------------------

    def _key(self, landmark: float) -> float | None:
        return float(landmark) if self.per_landmark else None

    def _signs(self, covariates: Sequence[str]) -> np.ndarray:
        signs = dict(self.unit_weight_signs or {})
        if set(signs) != set(covariates):
            raise ValueError(
                f"`unit_weight_signs` tiene que traer un signo por covariable: faltan "
                f"{sorted(set(covariates) - set(signs))}, sobran {sorted(set(signs) - set(covariates))}"
            )
        values = np.asarray([float(signs[c]) for c in covariates])
        if not np.isin(values, (-1.0, 1.0)).all():
            raise ValueError("Los pesos unitarios son +1 o −1 (Dawes 1979): nada se ajusta")
        return values

    def _standardized(self, key: float | None, features: pd.DataFrame) -> np.ndarray:
        model = self.landmarks_[key]
        return model.scaler.transform(model.imputer.transform(features.to_numpy(dtype=float)))

    def _design(self, standardized: np.ndarray) -> np.ndarray:
        ones = np.ones((len(standardized), 1))
        if self.incidence == "unit_weight":
            return np.hstack([ones, (standardized @ self.signs_)[:, None]])
        return np.hstack([ones, standardized])

    # -- API de sklearn -------------------------------------------------------------

    def fit(self, X: pd.DataFrame, y: Any) -> "CureMixtureModel":
        if self.incidence not in INCIDENCES:
            raise ValueError(f"`incidence` tiene que ser uno de {INCIDENCES}, no `{self.incidence}`")
        if self.latency not in LATENCIES:
            raise ValueError(f"`latency` tiene que ser uno de {LATENCIES}, no `{self.latency}`")
        if self.incidence == "tabpfn":
            raise NotImplementedError(
                "P2 (TabPFN v2 en la incidencia) corre solo si P1 le gana a P0 (preregistro §4) "
                "y todavía no está implementado: tabpfn no es dependencia del repo."
            )
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "`cure_mixture` necesita el panel crudo (con las columnas `feat_fm__*` y el mercado): "
                "poné `preprocessing: none` en el YAML del experimento."
            )
        target = _unpack_cure_target(y)
        if len(target["entry"]) != len(X):
            raise ValueError(f"X tiene {len(X)} filas y el target {len(target['entry'])}")
        if self.landmark_column not in X:
            raise KeyError(f"Falta la columna de diseño `{self.landmark_column}`")
        landmarks = X[self.landmark_column].to_numpy(dtype=float)
        if not np.allclose(landmarks, target["landmark"].astype(float)):
            raise ValueError(f"`{self.landmark_column}` no coincide con el hito del target: X e y no están alineados")

        self.normalizer_ = FleetReferenceNormalizer(**dict(self.normalizer_params or {})).fit(
            X, (target["healthy"], target["group"]))
        normalized = self.normalizer_.transform(X)
        self.covariates_ = [c for c in normalized.columns if c != self.landmark_column]
        if not self.covariates_:
            raise ValueError("No quedó ninguna covariable de incidencia después del normalizador")
        self.signs_ = self._signs(self.covariates_) if self.incidence == "unit_weight" else None

        # La ventana del score es del diseño del panel, no del ajuste: una por hito.
        self.score_windows_ = {}
        for landmark in np.unique(landmarks):
            rows = landmarks == landmark
            starts, ends = np.unique(target["score_start"][rows]), np.unique(target["score_end"][rows])
            if len(starts) != 1 or len(ends) != 1:
                raise ValueError(f"El hito {landmark:g} mezcla ventanas de score: {starts} / {ends}")
            self.score_windows_[float(landmark)] = (float(starts[0]), float(ends[0]))

        entry = target["entry"].astype(float)
        exit_ = target["exit"].astype(float)
        event = target["event"].astype(int)
        at_risk = target["at_risk"].astype(bool)
        if (event.astype(bool) & ~at_risk).any():
            raise ValueError("Hay filas con evento fuera de riesgo: el target está mal armado")

        self.landmarks_: dict[float | None, LandmarkModel] = {}
        keys = sorted({self._key(v) for v in landmarks}, key=lambda k: -1.0 if k is None else k)
        for key in keys:
            rows = np.ones(len(X), dtype=bool) if key is None else landmarks == key
            raw = normalized.loc[rows, self.covariates_]
            empty = [c for c in self.covariates_ if raw[c].isna().all()]
            if empty:
                raise ValueError(f"Hito {key}: {empty} no tienen ningún valor en train, no hay mediana que imputar")
            # Imputación y escala con TODAS las filas de train del hito (también las que no
            # están en riesgo): son features, no desenlaces. El EM, solo con las en riesgo.
            imputer = SimpleImputer(strategy="median").fit(raw.to_numpy(dtype=float))
            scaler = StandardScaler().fit(imputer.transform(raw.to_numpy(dtype=float)))
            design = self._design(scaler.transform(imputer.transform(raw.to_numpy(dtype=float))))
            risk = at_risk[rows]
            fit = fit_cure_em(
                design[risk], entry[rows][risk], exit_[rows][risk], event[rows][risk],
                firth=True, flic=bool(self.firth_flic),
                max_iter=int(self.max_iter), tol=float(self.tol),
                inner_max_iter=int(self.inner_max_iter), inner_tol=float(self.inner_tol),
            )
            model = LandmarkModel(
                imputer=imputer, scaler=scaler, fit=fit, n_rows=int(rows.sum()), n_at_risk=int(risk.sum()),
                n_events=int(event[rows][risk].sum()),
                susceptible_fraction=float(expit(design[risk] @ fit.coef).mean()),
            )
            self.landmarks_[key] = model
            if not fit.converged or fit.converged_flic is False:
                logger.warning("cure | hito %s | el EM no convergió en %d iteraciones", key, int(self.max_iter))
            logger.info(
                "cure (%s) | hito %s | %d filas, %d en riesgo, %d eventos | π̄ %.3f | k %.2f λ %.0f d | "
                "%d iteraciones%s",
                self.incidence, "todos" if key is None else f"{key:g}", model.n_rows, model.n_at_risk,
                model.n_events, model.susceptible_fraction, fit.latency.shape, fit.latency.scale, fit.n_iter,
                f" + {fit.n_iter_flic} FLIC" if fit.flic_history else "",
            )

        self.classes_ = np.array([0, 1])
        self.n_features_in_ = X.shape[1]
        return self

    def predict_components(self, X: pd.DataFrame) -> dict[str, np.ndarray]:
        """`score`, `pi_incidence`, `p_horizon` y la latencia del hito (y `unit_weight_score` con P0).

        `latency_shape` y `latency_scale` son constantes dentro de cada hito y fold: viajan
        por fila para que las métricas recalculen S_u sobre la exposición real de cada
        vehículo (calibración, seguimiento suficiente) sin reabrir el modelo.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("`cure_mixture` predice sobre el panel crudo: `preprocessing: none`")
        landmarks = X[self.landmark_column].to_numpy(dtype=float)
        unknown = sorted(set(landmarks) - set(self.score_windows_))
        if unknown:
            raise ValueError(f"Hitos que no estaban en train: {unknown}")
        normalized = self.normalizer_.transform(X)
        out = {name: np.full(len(X), np.nan)
               for name in ("score", "pi_incidence", "p_horizon", "latency_shape", "latency_scale")}
        if self.incidence == "unit_weight":
            out[UNIT_WEIGHT_SCORE] = np.full(len(X), np.nan)
        for landmark in np.unique(landmarks):
            rows = landmarks == landmark
            model = self.landmarks_[self._key(landmark)]
            standardized = self._standardized(self._key(landmark), normalized.loc[rows, self.covariates_])
            pi = expit(self._design(standardized) @ model.fit.coef)
            start, end = self.score_windows_[float(landmark)]
            p_horizon = 1.0 - float(model.fit.latency.conditional_survival(start, end))
            out["pi_incidence"][rows] = pi
            out["p_horizon"][rows] = p_horizon
            out["latency_shape"][rows] = model.fit.latency.shape
            out["latency_scale"][rows] = model.fit.latency.scale
            out["score"][rows] = pi * p_horizon
            if self.incidence == "unit_weight":
                out[UNIT_WEIGHT_SCORE][rows] = standardized @ self.signs_
        return out

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        score = self.predict_components(X)["score"]
        return np.column_stack([1.0 - score, score])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    @property
    def summary_(self) -> dict[str, dict[str, Any]]:
        """Por hito: coeficientes con nombre, (k, λ), convergencia y fracción susceptible."""
        names = ["intercept", UNIT_WEIGHT_SCORE] if self.incidence == "unit_weight" else ["intercept", *self.covariates_]
        summary = {}
        for key, model in self.landmarks_.items():
            fit = model.fit
            summary["all" if key is None else f"{key:g}"] = {
                "coef": dict(zip(names, map(float, fit.coef))),
                "latency_shape": fit.latency.shape,
                "latency_scale_days": fit.latency.scale,
                "loglik": fit.loglik,
                "n_iter": fit.n_iter,
                "n_iter_flic": fit.n_iter_flic,
                "converged": bool(fit.converged and fit.converged_flic is not False),
                "n_rows": model.n_rows,
                "n_at_risk": model.n_at_risk,
                "n_events": model.n_events,
                "susceptible_fraction": model.susceptible_fraction,
            }
        return summary
