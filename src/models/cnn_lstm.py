"""Baseline de la tutora: CNN-LSTM con una rama dinámica y una rama estática.

Lo que sabemos de su solución son tres frases:

    "Usó una red convolucional long short memory"
    "Pero usó solo dos datasets"
    "Una capa dinámica que procesaba los datos y después una capa estática que no le
    llegaban todos los datos"

Cómo se traduce a este repo:

- **Dos datasets** = `DynamicInformation` (`signals`) y `StaticInformation`
  (`vehicles`). `TripSummary` no entra.
- **Rama dinámica** = Conv1D → LSTM sobre la secuencia de `signals` de la ventana
  `(c − W, c]`, partida en bins de odómetro (`src/features/sequences.py`). La
  convolución extrae patrones locales entre bins vecinos; la LSTM resume el orden y
  entrega su último estado oculto.
- **Rama estática** = una capa densa que recibe *solo* las `static_*` del vehículo
  —no la secuencia— y se concatena con el estado final de la LSTM antes de la cabeza
  de clasificación (fusión tardía). Es la lectura de "una capa estática a la que no le
  llegaban todos los datos".

Con respecto a lo que ella pudo haber usado, hay dos diferencias, y las impone lo que
encontramos en F1/F2 (CLAUDE.md): de `signals` no entran `Regenerations` ni
`DistanceBetweenRegenerations` (se cortan el 25-05-2026 y miden el calendario), y de
`StaticInformation` entra solo `SalesCountry_cd` (`Engine`, `ModelSeries`,
`ProductionDay` y `daysUntilSale` son sesgo de muestreo o exposición).

**Cómo llega la secuencia sin tocar `cv.py`.** El panel secuencial
(`scripts/build_seq_panel.py`) trae la secuencia aplanada en columnas
`feat_seq_t{t}_c{k}_*` y ninguna otra `feat_*`. El preprocesador de `cv.py` las imputa
y estandariza con el train del fold, y pone primero las numéricas —en el orden del
panel, que es `(t, canal)`— y después el one-hot de las categóricas. Así, las primeras
`T·C` columnas de `X` son la secuencia y el resto son las estáticas. `T` y `C` salen de
`sequence_meta`, el JSON que escribe el builder del panel, para que no haya dos lugares
donde declararlos.

Lo que este modelo NO hace, a propósito: early stopping. El modelo no recibe
`vehicle_id`, así que no puede apartar una validación interna agrupada por vehículo, y
una partición por filas mezclaría cortes del mismo vehículo (pararía tarde, premiando
la memorización). Las épocas son un hiperparámetro del YAML.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, ClassifierMixin

from src.config import resolve_path


def _torch():
    """Import diferido: el registry se importa sin torch; solo el que entrena lo necesita."""
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depende del entorno
        raise ImportError(
            "`cnn_lstm` necesita PyTorch, que es una dependencia opcional: "
            "`uv pip install -r requirements-dl.txt`"
        ) from exc
    return torch


def _build_net(n_channels: int, n_static: int, *, conv_channels: int, kernel_size: int, lstm_hidden: int,
               static_hidden: int, head_hidden: int, dropout: float):
    torch = _torch()
    nn = torch.nn

    class CNNLSTMNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            # Rama dinámica: la convolución corre sobre el eje de bins (padding "same").
            self.conv = nn.Sequential(
                nn.Conv1d(n_channels, conv_channels, kernel_size, padding=kernel_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.lstm = nn.LSTM(conv_channels, lstm_hidden, batch_first=True)
            # Rama estática: solo las `static_*`, se une después de la LSTM.
            self.static = (nn.Sequential(nn.Linear(n_static, static_hidden), nn.ReLU())
                           if n_static > 0 and static_hidden > 0 else None)
            fused = lstm_hidden + (static_hidden if self.static is not None else 0)
            self.head = nn.Sequential(
                nn.Linear(fused, head_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(head_hidden, 1),
            )

        def forward(self, seq, static):
            x = self.conv(seq.transpose(1, 2)).transpose(1, 2)    # (B, T, C) -> (B, T, F)
            _, (hidden, _) = self.lstm(x)
            z = hidden[-1]                                        # último estado: (B, H)
            if self.static is not None:
                z = torch.cat([z, self.static(static)], dim=1)
            return self.head(z).squeeze(-1)                       # logits

    return CNNLSTMNet()


class CNNLSTMClassifier(ClassifierMixin, BaseEstimator):
    """Clasificador sklearn-compatible: `fit(X, y)` / `predict_proba(X)` sobre la matriz de `cv.py`.

    Los hiperparámetros por defecto son chicos a propósito: ~1.600 filas de train y
    ~200 positivas por fold (53 vehículos con evento en dev). La red tiene ~3.000
    parámetros; más capacidad memoriza vehículos.
    """

    def __init__(
        self,
        *,
        sequence_meta: str | None = None,
        seq_len: int | None = None,
        n_channels: int | None = None,
        conv_channels: int = 16,
        kernel_size: int = 3,
        lstm_hidden: int = 16,
        static_hidden: int = 4,
        head_hidden: int = 16,
        dropout: float = 0.3,
        epochs: int = 40,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        class_weight: str | None = "balanced",
        grad_clip: float | None = 1.0,
        device: str = "cpu",
        random_state: int = 42,
    ) -> None:
        self.sequence_meta = sequence_meta
        self.seq_len = seq_len
        self.n_channels = n_channels
        self.conv_channels = conv_channels
        self.kernel_size = kernel_size
        self.lstm_hidden = lstm_hidden
        self.static_hidden = static_hidden
        self.head_hidden = head_hidden
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.class_weight = class_weight
        self.grad_clip = grad_clip
        self.device = device
        self.random_state = random_state

    # ------------------------------------------------------------------ forma del tensor
    def _resolve_shape(self) -> tuple[int, int]:
        """`(T, C)` desde `sequence_meta`; `seq_len`/`n_channels` explícitos tienen que coincidir."""
        seq_len, n_channels = self.seq_len, self.n_channels
        if self.sequence_meta is not None:
            meta = json.loads(resolve_path(self.sequence_meta).read_text(encoding="utf-8"))
            for name, given, stored in (("seq_len", seq_len, meta["seq_len"]),
                                        ("n_channels", n_channels, meta["n_channels"])):
                if given is not None and int(given) != int(stored):
                    raise ValueError(f"`{name}`={given} en el YAML, pero el panel secuencial dice {stored}")
            seq_len, n_channels = int(meta["seq_len"]), int(meta["n_channels"])
        if seq_len is None or n_channels is None:
            raise ValueError("`cnn_lstm` necesita `sequence_meta` (o `seq_len` y `n_channels`) en model.params")
        return int(seq_len), int(n_channels)

    def _split(self, X: Any) -> tuple[np.ndarray, np.ndarray]:
        X = X.toarray() if sparse.issparse(X) else np.asarray(X)
        X = X.astype(np.float32, copy=False)
        width = self.seq_len_ * self.n_channels_
        if X.shape[1] < width:
            raise ValueError(
                f"X tiene {X.shape[1]} columnas y la secuencia sola ocupa {width} (T={self.seq_len_} × "
                f"C={self.n_channels_}). ¿El panel es el secuencial y `sequence_meta` es el suyo?"
            )
        if np.isnan(X).any():
            raise ValueError("X trae NaN: el imputador del Pipeline tendría que haberlos resuelto")
        seq = X[:, :width].reshape(len(X), self.seq_len_, self.n_channels_)
        return np.ascontiguousarray(seq), np.ascontiguousarray(X[:, width:])

    # ------------------------------------------------------------------ API sklearn
    def fit(self, X: Any, y: Any) -> CNNLSTMClassifier:
        torch = _torch()
        self.seq_len_, self.n_channels_ = self._resolve_shape()
        seq, static = self._split(X)
        y = np.asarray(y).astype(np.float32)
        self.classes_ = np.array([0, 1])
        if not set(np.unique(y)) <= {0.0, 1.0}:
            raise ValueError("`cnn_lstm` es binario: `y` tiene que ser 0/1")
        self.n_features_in_ = seq.shape[1] * seq.shape[2] + static.shape[1]
        self.n_static_ = static.shape[1]

        torch.manual_seed(self.random_state)
        rng = np.random.default_rng(self.random_state)
        device = torch.device(self.device)
        self.net_ = _build_net(
            self.n_channels_, self.n_static_, conv_channels=self.conv_channels, kernel_size=self.kernel_size,
            lstm_hidden=self.lstm_hidden, static_hidden=self.static_hidden, head_hidden=self.head_hidden,
            dropout=self.dropout,
        ).to(device)

        # Desbalance: peso de los positivos = negativos/positivos del train del fold
        # (el equivalente de `class_weight="balanced"`; nunca re-muestreo, plan §6).
        n_pos = float(y.sum())
        pos_weight = None
        if self.class_weight == "balanced" and 0 < n_pos < len(y):
            pos_weight = torch.tensor((len(y) - n_pos) / n_pos, device=device)
        elif self.class_weight not in (None, "balanced"):
            raise ValueError(f"class_weight={self.class_weight!r}: opciones None o 'balanced'")
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(self.net_.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        seq_t = torch.from_numpy(seq).to(device)
        static_t = torch.from_numpy(static).to(device)
        y_t = torch.from_numpy(y).to(device)
        self.loss_history_ = []
        self.net_.train()
        for _ in range(int(self.epochs)):
            order = rng.permutation(len(y))
            total = 0.0
            for start in range(0, len(order), int(self.batch_size)):
                idx = torch.from_numpy(order[start:start + int(self.batch_size)]).to(device)
                optimizer.zero_grad()
                loss = loss_fn(self.net_(seq_t[idx], static_t[idx]), y_t[idx])
                loss.backward()
                if self.grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.net_.parameters(), float(self.grad_clip))
                optimizer.step()
                total += float(loss.detach()) * len(idx)
            self.loss_history_.append(total / len(y))
        self.net_.eval()
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        torch = _torch()
        seq, static = self._split(X)
        if static.shape[1] != self.n_static_:
            raise ValueError(f"X trae {static.shape[1]} columnas estáticas y el modelo se ajustó con {self.n_static_}")
        device = next(self.net_.parameters()).device
        with torch.no_grad():
            logits = self.net_(torch.from_numpy(seq).to(device), torch.from_numpy(static).to(device))
            p = torch.sigmoid(logits).cpu().numpy().astype(float)
        return np.column_stack([1.0 - p, p])

    def predict(self, X: Any) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
