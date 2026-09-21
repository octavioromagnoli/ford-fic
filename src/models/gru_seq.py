"""GRU sobre la secuencia de la ventana: ¿el orden en km aporta algo que los agregados no?

El panel v1 resume `(c − W, c]` en 53 números y el LightGBM saca de ahí PR-AUC 0,165.
El panel secuencial entrega la misma ventana en `T` bins de km × `C` canales, y la
pregunta de este modelo es si **el orden** —que el agregado tira— vale algo. La
CNN-LSTM de la tutora (`src/models/cnn_lstm.py`) ya midió una respuesta parcial
(0,153); acá se cambian tres cosas y se deja todo lo demás igual para que la
comparación sea limpia:

1. **GRU en vez de Conv1D + LSTM.** Una capa recurrente sola, sin la convolución
   previa: con 20 bins la convolución mezcla vecinos antes de que el recurrente vea
   nada, y con ~250 filas positivas cada bloque extra es capacidad que se gasta en
   memorizar. La GRU tiene 3 compuertas contra las 4 de la LSTM: 25% menos
   parámetros para la misma tarea.
2. **Unidireccional, siempre.** Nada de `bidirectional`: el eje es km recorridos y
   leer la ventana al revés equivale a que el estado en el km 200 dependa de lo que
   pasó en el km 900. No hay lectura física de eso, y en una ventana que termina
   pegada al corte es la forma más barata de inventar señal.
3. **Pooling configurable sobre la secuencia** (`pooling`), que es la hipótesis que
   este modelo viene a probar:

   - `last`: el último estado oculto, que es lo que hace la CNN-LSTM. Le da todo el
     peso al bin más cercano al corte.
   - `mean`: promedio de los estados. Ignora dónde pasó cada cosa.
   - `attention` (**default**): un solo head, `α_t = softmax(v·tanh(W·h_t))`, y el
     resumen es `Σ α_t·h_t`. Si la señal está en el medio de la ventana —una
     regeneración anómala en el km 400 de 1.000—, `last` la tiene que arrastrar por
     todos los bins siguientes y la atención la toma directo. Es el default porque
     es la única de las tres que puede ganarle a `last` cuando la señal no está al
     final, y `attention_weights()` deja auditar dónde mira (si mira siempre el
     último bin, es el atajo de posición y no la física).

Lo que **no** cambia respecto de la CNN-LSTM, a propósito, porque si no la
comparación no mide la arquitectura:

- **Rama estática con fusión tardía.** Una densa que recibe solo las `static_*` (hoy
  el one-hot de `SalesCountry_cd`) y se concatena con el resumen de la secuencia
  antes de la cabeza. `static_hidden: 0` la apaga, que es la ablación obligatoria:
  el mercado solo ya da ROC 0,566 (docs/memoria/f3-cnn-lstm-tutora.md), así que un
  salto que venga de ahí es el sesgo de muestreo de F1, no el vehículo.
- **Sin early stopping.** El modelo no recibe `vehicle_id`, así que no puede apartar
  una validación interna agrupada; una partición por filas mezclaría cortes del
  mismo vehículo y pararía tarde, premiando la memorización. Las épocas son un
  hiperparámetro del YAML (regla 7).
- **`pos_weight` del train de cada fold** (`class_weight: balanced`), nunca
  re-muestreo: el re-muestreo fuera del `Pipeline` es leakage (plan §6).
- **Chico a propósito.** Dev tiene 53 vehículos con evento. Los defaults dan ~3.400
  parámetros (`n_parameters_` lo reporta después de `fit`); más capacidad memoriza
  vehículos.

**W, G, H y Δ no se implementan acá.** Son del panel (`configs/data/panel_v1.yaml`):
el gap ya está aplicado y la etiqueta ya es a horizonte cuando la fila llega. La red
solo ve la ventana `(c − W, c]` que el panel secuencial le arma, y `T`, `C`,
`lookback_km` y `bin_km` salen de `sequence_meta` —el `_meta.json` que escribe
`scripts/build_seq_panel.py`—, nunca de una constante. Si algún día un
hiperparámetro necesitara el horizonte, va al YAML como cualquier otro.

**Cómo llega la secuencia sin tocar `cv.py`.** Igual que la CNN-LSTM: el panel
secuencial trae la ventana aplanada en `feat_seq_t{t}_c{k}_*` y ninguna otra
`feat_*`. El preprocesador de `cv.py` las imputa y estandariza con el train del fold
y las deja primeras en la matriz, en orden `(t, canal)`; acá se rearma el tensor con
un reshape y el resto de las columnas son las estáticas.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, ClassifierMixin

from src.config import resolve_path

POOLINGS = ("attention", "last", "mean")


def _torch():
    """Import diferido: el registry se importa sin torch; solo el que entrena lo necesita."""
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depende del entorno
        raise ImportError(
            "`gru_seq` necesita PyTorch, que es una dependencia opcional: "
            "`uv pip install -r requirements-dl.txt`"
        ) from exc
    return torch


def _build_net(n_channels: int, n_static: int, *, hidden: int, num_layers: int, pooling: str,
               attention_hidden: int, static_hidden: int, head_hidden: int, dropout: float):
    torch = _torch()
    nn = torch.nn

    class GRUSeqNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            # Rama dinámica: una GRU unidireccional recorre los bins en orden de km.
            # `dropout` entre capas solo tiene efecto con num_layers > 1 (lo dice torch).
            self.gru = nn.GRU(n_channels, hidden, num_layers=num_layers, batch_first=True,
                              dropout=dropout if num_layers > 1 else 0.0)
            self.drop = nn.Dropout(dropout)
            # Atención de un head sobre los estados: score escalar por bin.
            self.attn = (nn.Sequential(nn.Linear(hidden, attention_hidden), nn.Tanh(),
                                       nn.Linear(attention_hidden, 1, bias=False))
                         if pooling == "attention" else None)
            # Rama estática: solo las `static_*`, se une después del pooling.
            self.static = (nn.Sequential(nn.Linear(n_static, static_hidden), nn.ReLU())
                           if n_static > 0 and static_hidden > 0 else None)
            fused = hidden + (static_hidden if self.static is not None else 0)
            self.head = nn.Sequential(
                nn.Linear(fused, head_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(head_hidden, 1),
            )

        def pool(self, states):
            """`(B, T, H)` -> `(B, H)` + los pesos por bin (uniformes o one-hot si no hay atención)."""
            if self.attn is not None:
                weights = torch.softmax(self.attn(states).squeeze(-1), dim=1)   # (B, T)
                return torch.einsum("bt,bth->bh", weights, states), weights
            if pooling == "mean":
                weights = states.new_full(states.shape[:2], 1.0 / states.shape[1])
                return states.mean(dim=1), weights
            weights = states.new_zeros(states.shape[:2])
            weights[:, -1] = 1.0
            return states[:, -1, :], weights

        def forward(self, seq, static, *, return_weights: bool = False):
            states, _ = self.gru(seq)                        # (B, T, C) -> (B, T, H)
            z, weights = self.pool(self.drop(states))
            if self.static is not None:
                z = torch.cat([z, self.static(static)], dim=1)
            logits = self.head(z).squeeze(-1)
            return (logits, weights) if return_weights else logits

    return GRUSeqNet()


class GRUSeqClassifier(ClassifierMixin, BaseEstimator):
    """Clasificador sklearn-compatible: `fit(X, y)` / `predict_proba(X)` sobre la matriz de `cv.py`.

    Cada default y por qué (todos se pisan desde `model.params` del YAML, nunca
    editando esta clase):

    | param | default | por qué |
    |---|---|---|
    | `pooling` | `attention` | la hipótesis del modelo: la señal puede no estar en el último bin. `last` reproduce a la CNN-LSTM, `mean` es el control que ignora el orden de llegada |
    | `hidden` | 24 | con C=9 son ~2.500 parámetros en la GRU; el total queda en ~3.400, el orden de la CNN-LSTM (~3.000). Con 53 vehículos con evento, más capacidad memoriza |
    | `num_layers` | 1 | 20 bins no justifican una segunda capa; la dejo configurable para el barrido de ventana (T=40 y T=60 en W=2.000/3.000) |
    | `attention_hidden` | 16 | la proyección del score de atención. Más chica que `hidden`: es un escalar por bin, no una representación |
    | `static_hidden` | 4 | lo mismo que la CNN-LSTM, para que la ablación `0` sea comparable entre los dos modelos |
    | `head_hidden` | 16 | ídem CNN-LSTM |
    | `dropout` | 0.3 | fuerte, por el tamaño de dev. Se aplica a los estados antes del pooling y dentro de la cabeza |
    | `epochs` | 40 | fijas, sin early stopping (ver el docstring del módulo). Es el valor con el que se midió la CNN-LSTM |
    | `batch_size` | 64 | ~1.600 filas de train por fold ⇒ 25 pasos por época |
    | `learning_rate` | 1e-3 | Adam por defecto; con 40 épocas y este tamaño converge sin oscilar |
    | `weight_decay` | 1e-4 | regularización adicional barata |
    | `grad_clip` | 1.0 | recurrentes y gradientes largos |
    | `class_weight` | `balanced` | `pos_weight` = negativos/positivos **del train del fold**. Nunca re-muestreo |
    | `device` | `cpu` | determinista y sobra para este tamaño |
    | `random_state` | 42 | la semilla sale del YAML. Una sola semilla no es un resultado: la varianza de inicialización es ±0,01 de PR-AUC |
    """

    def __init__(
        self,
        *,
        sequence_meta: str | None = None,
        seq_len: int | None = None,
        n_channels: int | None = None,
        pooling: str = "attention",
        hidden: int = 24,
        num_layers: int = 1,
        attention_hidden: int = 16,
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
        self.pooling = pooling
        self.hidden = hidden
        self.num_layers = num_layers
        self.attention_hidden = attention_hidden
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
        """`(T, C)` desde `sequence_meta`; `seq_len`/`n_channels` explícitos tienen que coincidir.

        De paso guarda `lookback_km_` y `bin_km_` (= L / T): la geometría de la ventana
        se lee del panel, no se hardcodea (regla 7). El modelo no la usa para decidir
        nada —es el panel el que fija W—, pero sin ella los pesos de atención de
        `attention_weights()` no se pueden leer en km.
        """
        seq_len, n_channels = self.seq_len, self.n_channels
        self.lookback_km_ = self.bin_km_ = None
        if self.sequence_meta is not None:
            meta = json.loads(resolve_path(self.sequence_meta).read_text(encoding="utf-8"))
            for name, given, stored in (("seq_len", seq_len, meta["seq_len"]),
                                        ("n_channels", n_channels, meta["n_channels"])):
                if given is not None and int(given) != int(stored):
                    raise ValueError(f"`{name}`={given} en el YAML, pero el panel secuencial dice {stored}")
            seq_len, n_channels = int(meta["seq_len"]), int(meta["n_channels"])
            self.lookback_km_ = float(meta["lookback_km"])
            self.bin_km_ = float(meta.get("bin_km", self.lookback_km_ / seq_len))
        if seq_len is None or n_channels is None:
            raise ValueError("`gru_seq` necesita `sequence_meta` (o `seq_len` y `n_channels`) en model.params")
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
    def fit(self, X: Any, y: Any) -> GRUSeqClassifier:
        torch = _torch()
        if self.pooling not in POOLINGS:
            raise ValueError(f"pooling={self.pooling!r}: opciones {list(POOLINGS)}")
        self.seq_len_, self.n_channels_ = self._resolve_shape()
        seq, static = self._split(X)
        y = np.asarray(y).astype(np.float32)
        self.classes_ = np.array([0, 1])
        if not set(np.unique(y)) <= {0.0, 1.0}:
            raise ValueError("`gru_seq` es binario: `y` tiene que ser 0/1")
        self.n_features_in_ = seq.shape[1] * seq.shape[2] + static.shape[1]
        self.n_static_ = static.shape[1]

        torch.manual_seed(self.random_state)
        rng = np.random.default_rng(self.random_state)
        device = torch.device(self.device)
        self.net_ = _build_net(
            self.n_channels_, self.n_static_, hidden=self.hidden, num_layers=self.num_layers,
            pooling=self.pooling, attention_hidden=self.attention_hidden, static_hidden=self.static_hidden,
            head_hidden=self.head_hidden, dropout=self.dropout,
        ).to(device)
        self.n_parameters_ = sum(p.numel() for p in self.net_.parameters() if p.requires_grad)

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

    def _forward(self, X: Any, *, return_weights: bool) -> Any:
        torch = _torch()
        seq, static = self._split(X)
        if static.shape[1] != self.n_static_:
            raise ValueError(f"X trae {static.shape[1]} columnas estáticas y el modelo se ajustó con {self.n_static_}")
        device = next(self.net_.parameters()).device
        with torch.no_grad():
            return self.net_(torch.from_numpy(seq).to(device), torch.from_numpy(static).to(device),
                             return_weights=return_weights)

    def predict_proba(self, X: Any) -> np.ndarray:
        torch = _torch()
        p = torch.sigmoid(self._forward(X, return_weights=False)).cpu().numpy().astype(float)
        return np.column_stack([1.0 - p, p])

    def predict(self, X: Any) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def attention_weights(self, X: Any) -> np.ndarray:
        """`(N, T)` con el peso que el pooling le dio a cada bin. Es una auditoría, no una métrica.

        Con `pooling: attention` son los `α_t` aprendidos; con `last` y `mean` devuelve
        lo que esos dos hacen implícitamente (todo el peso en el último bin, o uniforme),
        así la lectura es la misma para las tres variantes. Si la atención se concentra
        siempre en el bin pegado al corte, lo que la red aprendió es la posición en la
        serie, no la física (regla 6).
        """
        return self._forward(X, return_weights=True)[1].cpu().numpy().astype(float)
