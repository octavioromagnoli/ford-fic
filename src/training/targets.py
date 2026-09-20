"""Qué se le pasa al estimador como `y`. Una función por modo, registro por nombre.

`src/training/cv.py` evalúa **siempre** contra la etiqueta dura `label` (CLAUDE.md,
regla 5). Lo que este archivo decide es otra cosa: con qué objetivo se *entrena*. La
etiqueta binaria de una ventana tira información que el panel ya tiene —a qué
distancia cae el evento, hasta dónde se observó un vehículo sano, de qué vehículo es
la fila— y cada modo de acá la recupera de una forma distinta.

**El registro es el contrato.** `cv.py` no sabe qué modos existen: pide
`build_target(name, panel, train_mask, **params)` y usa lo que vuelva. Agregar un modo
es escribir una función con `@register_target("...")` en este archivo y nombrarla en el
`target:` del YAML del experimento; no se toca `cv.py` ni `train.py`, y dos ramas pueden
agregar modos distintos sin pisarse.

**Reglas que todo modo tiene que respetar:**

1. El target se construye **solo con las filas de train del fold** (`train_mask`). El
   panel entero entra como argumento porque hay modos que necesitan agrupar por
   vehículo, pero cualquier estadístico se estima sobre el train (CLAUDE.md, regla 3).
2. `TargetSpec.y` está alineado 1:1 con `panel.loc[train_mask]`, en ese orden. Es lo
   que `Pipeline.fit` recibe como `y`, así que el estimador del `model:` del YAML y el
   modo del `target:` van de a pares.
3. Ningún modo puede mirar los km del gap de blanking (regla 1). El de supervivencia lo
   cumple corriendo el origen de la duración a `c + G`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

GROUP_COLUMN = "vehicle_id"
FOLLOWUP_COLUMN = "aux_km_observed_after_cut"


@dataclass(frozen=True)
class TargetSpec:
    """Lo que un modo le entrega a `cv.py`: el `y` de train y con qué se construyó.

    `y` es lo único que `cv.py` usa; `name` y `params` viajan para que la corrida quede
    trazable en el log y en la metadata del experimento.
    """

    y: np.ndarray
    name: str
    params: dict[str, Any] = field(default_factory=dict)
    info: dict[str, Any] = field(default_factory=dict)


TargetBuilder = Callable[..., TargetSpec]

_REGISTRY: dict[str, TargetBuilder] = {}


def register_target(name: str) -> Callable[[TargetBuilder], TargetBuilder]:
    def decorator(builder: TargetBuilder) -> TargetBuilder:
        if name in _REGISTRY:
            raise ValueError(f"El target `{name}` ya está registrado")
        _REGISTRY[name] = builder
        return builder

    return decorator


def available_targets() -> list[str]:
    return sorted(_REGISTRY)


def build_target(
    name: str, panel: pd.DataFrame, train_mask: np.ndarray, **params: Any
) -> TargetSpec:
    """Construye el target de train del modo `name`. Único punto de entrada."""
    if name not in _REGISTRY:
        raise KeyError(f"Target `{name}` no registrado. Disponibles: {available_targets()}")
    spec = _REGISTRY[name](panel, train_mask, **params)
    n_train = int(np.asarray(train_mask).sum())
    if len(spec.y) != n_train:
        raise RuntimeError(
            f"El target `{name}` devolvió {len(spec.y)} valores para {n_train} filas de "
            "train. Tiene que estar alineado 1:1 con `panel.loc[train_mask]`."
        )
    return spec


# --------------------------------------------------------------------------- #
# Supervivencia en tiempo discreto ("survival stacking")
# --------------------------------------------------------------------------- #
#: Campos del `y` estructurado que consume `src/models/survival_stacking.py`.
SURVIVAL_FIELDS = ("duration_km", "event", "at_risk", "group")


@register_target("discrete_survival")
def build_discrete_survival_target(
    panel: pd.DataFrame,
    train_mask: np.ndarray,
    *,
    gap_km: float | None = None,
    followup_column: str = FOLLOWUP_COLUMN,
    group_column: str = GROUP_COLUMN,
) -> TargetSpec:
    """`(duración, evento, vehículo)` por fila de train, con el origen corrido al gap.

    El panel ya es un dataset persona-período: una fila por `(vehículo, corte)`. Lo que
    falta para tratarlo como supervivencia es decir, para cada fila, **cuánto se la
    observó y si terminó en evento** (Craig, Zhong & Tibshirani 2021, arXiv:2107.13480).

    * **Origen.** La duración no se cuenta desde el corte `c` sino desde `c + G`: el
      modelo no puede ver los km del gap de blanking (CLAUDE.md, regla 1). Una fila que
      "sobrevivió 0 km" es una fila cuyo evento cae justo al terminar el blanking.
    * **Filas con evento** (`event_observed == 1`): duración `time_to_event_km − G`,
      evento 1. El panel las corta en `E − G`, así que la duración es ≥ 0 por
      construcción.
    * **Filas sanas** (`event_observed == 0`): duración `aux_km_observed_after_cut − G`,
      evento 0. No son ceros limpios: son "llegó hasta acá sin fallar". Con
      `censored_policy: drop` esa duración es ≥ H, así que dentro del horizonte se
      comportan igual que un negativo; la diferencia aparece cuando el estimador mira
      más allá de H (`max_horizon_km`), que es justamente lo que la etiqueta binaria no
      puede usar.
    * **Duración < 0** (`at_risk = False`): la fila no está en riesgo en ningún km
      visible, porque el evento cayó dentro del gap. El estimador la descarta al apilar;
      usarla sería mirar los km que la regla 1 prohíbe. El panel v1 no tiene ninguna,
      pero el dummy sí, porque genera cortes hasta el evento. Una duración de
      exactamente 0 **sí** está en riesgo: es el evento justo al terminar el blanking, y
      es un `label = 1`.

    La etiqueta dura sale de esto sin residuo: `label = 1 ⇔ at_risk y evento y duración
    ∈ [0, H]`. No es una reparametrización libre, es la misma definición escrita sobre
    el eje correcto — y `scripts/check_setup.py` lo verifica fila a fila.

    `group` viaja para el efecto aleatorio por vehículo (GPBoost): los ~5 cortes de un
    mismo vehículo no son observaciones independientes.
    """
    mask = np.asarray(train_mask, dtype=bool)
    train = panel.loc[mask]

    missing = [c for c in ("time_to_event_km", "event_observed", group_column) if c not in train]
    if missing:
        raise KeyError(f"El panel no tiene {missing}: no se puede armar el target de supervivencia")
    if followup_column not in train:
        raise KeyError(
            f"El panel no tiene `{followup_column}`: los censurados no traen hasta dónde se "
            "observaron. Reconstruí el panel con `python scripts/build_dataset.py "
            "--config configs/data/panel_v1.yaml` (src/data/panel.py lo emite)."
        )

    gap = float(train["gap_km"].iloc[0]) if gap_km is None else float(gap_km)
    if gap_km is None and train["gap_km"].nunique() != 1:
        raise ValueError(
            "El panel mezcla varios `gap_km`: pasá `target.params.gap_km` explícito."
        )

    event = train["event_observed"].to_numpy().astype(np.int8)
    tte = train["time_to_event_km"].to_numpy(dtype=float)
    followup = train[followup_column].to_numpy(dtype=float)
    # Con evento, el seguimiento termina en el evento; sin evento, en el último km visto.
    raw = np.where(event == 1, tte, followup) - gap
    if np.isnan(raw).any():
        n = int(np.isnan(raw).sum())
        raise ValueError(
            f"{n} filas de train sin duración: `time_to_event_km` nulo en filas con evento "
            f"o `{followup_column}` nulo en censuradas."
        )

    at_risk = raw >= 0
    duration = np.where(at_risk, raw, 0.0)
    event = np.where(at_risk, event, 0).astype(np.int8)

    groups = train[group_column].astype(str).to_numpy()
    width = max(1, max((len(g) for g in groups), default=1))
    y = np.empty(
        len(train),
        dtype=[("duration_km", "f8"), ("event", "i1"), ("at_risk", "?"), ("group", f"U{width}")],
    )
    y["duration_km"] = duration
    y["event"] = event
    y["at_risk"] = at_risk
    y["group"] = groups

    info = {
        "gap_km": gap,
        "n_rows": int(len(train)),
        "n_events": int(event.sum()),
        "n_not_at_risk": int((~at_risk).sum()),
        "duration_km_median": float(np.median(duration[at_risk])) if at_risk.any() else float("nan"),
    }
    logger.debug("target discrete_survival | %s", info)
    return TargetSpec(
        y=y,
        name="discrete_survival",
        params={"gap_km": gap, "followup_column": followup_column, "group_column": group_column},
        info=info,
    )
