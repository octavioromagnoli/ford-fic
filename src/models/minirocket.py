"""MiniRocket multivariado sobre la secuencia de la ventana (F13).

Convoluciones **fijas** + un lineal regularizado, en vez de convoluciones aprendidas: es la
prueba de si el techo de F12 (GRU, CNN-LSTM y sus ensambles, AUC ~0,65 dentro de mercado ×
motor) es del modelo o de la información. Preregistro: `docs/memoria/f13-preregistro-minirocket.md`.

Implementación en numpy de Dempster, Schmidt y Webb (KDD 2021), variante multivariada, sin
`aeon`/`sktime` (los dos bajan numpy y scipy, que están fijados):

- **84 kernels** de largo 9: todas las combinaciones de 3 posiciones con peso 2 y el resto −1
  (suman 0). La convolución es `−Σ_{9 taps} x + 3·Σ_{3 taps} x`.
- **Dilataciones** `⌊2^linspace(0, log2((T−1)/8))⌋`, sin repetidas. Con T = 20 dan {1, 2}.
- **Padding alternado:** con padding, el PPV se cuenta sobre todo el largo; sin padding, solo
  donde el kernel entra entero.
- **Canales por kernel:** un subconjunto al azar de tamaño `⌊2^U(0, log2(min(C, 9) + 1))⌋`,
  sumados antes de convolucionar.
- **Sesgos:** cuantiles (sucesión de baja discrepancia con la razón áurea) de la convolución de
  una fila de train al azar por (dilatación, kernel). Se fijan en `fit`, nunca con validación.
- **Feature:** PPV, la proporción de posiciones donde la convolución supera el sesgo.

Después: `StandardScaler` sobre los PPV, las `static_*` concatenadas al final y
`RidgeClassifierCV`. El score es la función de decisión pasada por una sigmoide: la detección
y el ensamble por rango solo usan el orden.

Cómo llega la secuencia: igual que en `gru_seq.py`. `cv.py` imputa y estandariza la matriz con
el train del fold; las `feat_seq_*` vienen primero en orden `(t, canal)` y `T`/`C` salen de
`sequence_meta`.
"""

from __future__ import annotations

import json
from itertools import combinations
from typing import Any

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import RidgeClassifierCV
from sklearn.preprocessing import StandardScaler

from src.config import resolve_path

KERNEL_LENGTH = 9
KERNEL_INDICES = np.array(list(combinations(range(KERNEL_LENGTH), 3)), dtype=np.int64)  # 84 × 3
GOLDEN = (np.sqrt(5.0) - 1.0) / 2.0


def _dilations(seq_len: int, num_features: int, max_dilations_per_kernel: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """Dilataciones y cuántas features lleva cada una (como el MiniRocket de referencia)."""
    n_kernels = len(KERNEL_INDICES)
    per_kernel = max(1, num_features // n_kernels)
    true_max = min(per_kernel, max_dilations_per_kernel)
    multiplier = per_kernel / true_max
    max_exponent = np.log2(max(1.0, (seq_len - 1) / (KERNEL_LENGTH - 1)))
    dil, counts = np.unique(np.floor(2 ** np.linspace(0, max_exponent, true_max)).astype(np.int64),
                            return_counts=True)
    counts = (counts * multiplier).astype(np.int64)
    remainder = per_kernel - counts.sum()
    i = 0
    while remainder > 0:
        counts[i] += 1
        remainder -= 1
        i = (i + 1) % len(counts)
    return dil, counts


def _convolve(x: np.ndarray, kernel: np.ndarray, dilation: int) -> np.ndarray:
    """Convolución "same" con ceros: `x` (n, T) → (n, T)."""
    n, T = x.shape
    pad = (KERNEL_LENGTH - 1) * dilation // 2
    xp = np.pad(x, ((0, 0), (pad, pad)))
    taps = np.stack([xp[:, j * dilation: j * dilation + T] for j in range(KERNEL_LENGTH)])
    return -taps.sum(0) + 3.0 * taps[kernel].sum(0)


class MiniRocketClassifier(ClassifierMixin, BaseEstimator):
    """MiniRocket + `RidgeClassifierCV` sobre la matriz de `cv.py`. Todo se pisa desde el YAML.

    | param | default | por qué |
    |---|---|---|
    | `num_features` | 10.000 | el default del paper; no se barre (preregistro F13) |
    | `alphas` | logspace(−3, 3, 10) | el rango del paper para `RidgeClassifierCV` |
    | `class_weight` | `balanced` | ~6% de filas positivas, sin re-muestreo |
    | `random_state` | 42 | sesgos y canales al azar; una semilla no es un resultado |
    """

    def __init__(
        self,
        *,
        sequence_meta: str | None = None,
        seq_len: int | None = None,
        n_channels: int | None = None,
        num_features: int = 10_000,
        max_dilations_per_kernel: int = 32,
        alphas: tuple[float, ...] | list[float] | None = None,
        class_weight: str | None = "balanced",
        random_state: int = 42,
    ) -> None:
        self.sequence_meta = sequence_meta
        self.seq_len = seq_len
        self.n_channels = n_channels
        self.num_features = num_features
        self.max_dilations_per_kernel = max_dilations_per_kernel
        self.alphas = alphas
        self.class_weight = class_weight
        self.random_state = random_state

    # ------------------------------------------------------------------ forma del tensor
    def _resolve_shape(self) -> tuple[int, int]:
        seq_len, n_channels = self.seq_len, self.n_channels
        if self.sequence_meta is not None:
            meta = json.loads(resolve_path(self.sequence_meta).read_text(encoding="utf-8"))
            for name, given, stored in (("seq_len", seq_len, meta["seq_len"]),
                                        ("n_channels", n_channels, meta["n_channels"])):
                if given is not None and int(given) != int(stored):
                    raise ValueError(f"`{name}`={given} en el YAML, pero el panel secuencial dice {stored}")
            seq_len, n_channels = int(meta["seq_len"]), int(meta["n_channels"])
        if seq_len is None or n_channels is None:
            raise ValueError("`minirocket` necesita `sequence_meta` (o `seq_len` y `n_channels`) en model.params")
        if seq_len < KERNEL_LENGTH:
            raise ValueError(f"MiniRocket necesita T ≥ {KERNEL_LENGTH}; el panel tiene T={seq_len}")
        return int(seq_len), int(n_channels)

    def _split(self, X: Any) -> tuple[np.ndarray, np.ndarray]:
        X = X.toarray() if sparse.issparse(X) else np.asarray(X)
        X = X.astype(np.float64, copy=False)
        width = self.seq_len_ * self.n_channels_
        if X.shape[1] < width:
            raise ValueError(f"X tiene {X.shape[1]} columnas y la secuencia ocupa {width}: ¿panel secuencial?")
        if np.isnan(X).any():
            raise ValueError("X trae NaN: el imputador del Pipeline tendría que haberlos resuelto")
        # (n, T, C) → (n, C, T): cada canal es una serie en km
        seq = X[:, :width].reshape(len(X), self.seq_len_, self.n_channels_).transpose(0, 2, 1)
        return np.ascontiguousarray(seq), np.ascontiguousarray(X[:, width:])

    # ------------------------------------------------------------------ transformación
    def _fit_transform_params(self, seq: np.ndarray, rng: np.random.Generator) -> None:
        n, C, T = seq.shape
        self.dilations_, self.features_per_dilation_ = _dilations(T, self.num_features,
                                                                  self.max_dilations_per_kernel)
        max_exp = np.log2(min(C, KERNEL_LENGTH) + 1)
        combos: list[dict[str, Any]] = []
        q_offset = 0
        for d_idx, (dilation, n_feat) in enumerate(zip(self.dilations_, self.features_per_dilation_)):
            for k_idx, kernel in enumerate(KERNEL_INDICES):
                n_ch = max(1, int(2 ** rng.uniform(0, max_exp)))
                channels = rng.choice(C, size=min(n_ch, C), replace=False)
                example = seq[rng.integers(n)][channels].sum(0)[None, :]
                conv = _convolve(example, kernel, int(dilation))[0]
                quantiles = ((np.arange(n_feat) + q_offset + 1) * GOLDEN) % 1.0
                q_offset += n_feat
                combos.append({"dilation": int(dilation), "kernel": kernel, "channels": channels,
                               "biases": np.quantile(conv, quantiles),
                               "padded": (d_idx + k_idx) % 2 == 0})
        self.combos_ = combos

    def _transform(self, seq: np.ndarray) -> np.ndarray:
        out = []
        for c in self.combos_:
            conv = _convolve(seq[:, c["channels"]].sum(1), c["kernel"], c["dilation"])
            if not c["padded"]:
                pad = (KERNEL_LENGTH - 1) * c["dilation"] // 2
                conv = conv[:, pad: conv.shape[1] - pad]
            out.append((conv[:, :, None] > c["biases"][None, None, :]).mean(1, dtype=np.float32))
        return np.concatenate(out, axis=1)

    # ------------------------------------------------------------------ API sklearn
    def fit(self, X: Any, y: Any) -> MiniRocketClassifier:
        self.seq_len_, self.n_channels_ = self._resolve_shape()
        seq, static = self._split(X)
        y = np.asarray(y).astype(int)
        if not set(np.unique(y)) <= {0, 1}:
            raise ValueError("`minirocket` es binario: `y` tiene que ser 0/1")
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = X.shape[1]
        rng = np.random.default_rng(self.random_state)
        self._fit_transform_params(seq, rng)
        feats = self._transform(seq)
        self.scaler_ = StandardScaler().fit(feats)
        Z = np.hstack([self.scaler_.transform(feats), static])
        alphas = np.logspace(-3, 3, 10) if self.alphas is None else np.asarray(self.alphas, dtype=float)
        self.ridge_ = RidgeClassifierCV(alphas=alphas, class_weight=self.class_weight).fit(Z, y)
        self.n_rocket_features_ = feats.shape[1]
        return self

    def decision_function(self, X: Any) -> np.ndarray:
        seq, static = self._split(X)
        Z = np.hstack([self.scaler_.transform(self._transform(seq)), static])
        return self.ridge_.decision_function(Z)

    def predict_proba(self, X: Any) -> np.ndarray:
        p = 1.0 / (1.0 + np.exp(-self.decision_function(X)))
        return np.column_stack([1.0 - p, p])

    def predict(self, X: Any) -> np.ndarray:
        return (self.decision_function(X) > 0).astype(int)
