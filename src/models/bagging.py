"""Bagging por vehículo: el mismo modelo entrenado sobre remuestreos de **autos**, no de filas.

Es la pieza de E2 del preregistro del ensamble
(`docs/memoria/f3-preregistro-landmark-ensamble.md`): cada finalista se reentrena envuelto
en esto, con `n_bags` remuestreos bootstrap de los vehículos del train de cada fold, y el
score es el promedio de las bolsas. Lo único que promete es bajar la varianza del
entrenamiento, que con 53 vehículos con evento es de donde sale buena parte de la
dispersión entre repeticiones.

**Por qué vehículos y no filas.** Los cortes de un mismo auto comparten conductor, ruta y
casi todas las features; remuestrear filas deja cortes del mismo vehículo en bolsas que
se creen independientes y subestima la varianza que se quiere promediar. Un vehículo
sorteado dos veces entra con todos sus cortes, dos veces.

**De dónde sale el vehículo.** El `Pipeline` del fold le pasa al modelo la matriz ya
preprocesada, sin `vehicle_id`. El vehículo viaja en el `y` estructurado que arma
`src/training/targets.py`: el campo `group` del target `discrete_survival`, o el de
`grouped_label` para los modelos binarios. El modelo interno recibe lo que recibiría sin
el envoltorio: el `y` estructurado entero si lo usa (supervivencia), o solo `label`.

El preprocesamiento del fold se ajusta una vez, con el train entero del fold, antes de
esto; lo que se remuestrea es solo el ajuste del modelo. No cambia qué ve cada bolsa:
todo sale del train del fold (regla 3).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

GROUP_FIELD = "group"
LABEL_FIELD = "label"


class VehicleBaggingClassifier(ClassifierMixin, BaseEstimator):
    """Promedio de `n_bags` copias de un modelo del registry, cada una con un bootstrap de vehículos.

    Parámetros
    ----------
    base_model
        Nombre del modelo interno en `src/models/registry.py`.
    base_params
        Sus hiperparámetros, **los mismos** que la corrida sin bagging (preregistro: nada
        se re-tunea).
    n_bags
        Cuántos remuestreos. El preregistro fija 10.
    random_state
        Semilla del sorteo de vehículos. Las semillas internas del modelo no se tocan:
        la diversidad entre bolsas sale de los datos, no de la inicialización.
    """

    def __init__(
        self,
        *,
        base_model: str,
        base_params: dict[str, Any] | None = None,
        n_bags: int = 10,
        random_state: int = 42,
    ) -> None:
        self.base_model = base_model
        self.base_params = base_params
        self.n_bags = n_bags
        self.random_state = random_state

    def fit(self, X: Any, y: Any) -> VehicleBaggingClassifier:
        from src.models.registry import get_model  # import adentro: registry importa este módulo

        y = np.asarray(y)
        names = y.dtype.names
        if names is None or GROUP_FIELD not in names:
            raise TypeError(
                "`vehicle_bagging` necesita saber de qué vehículo es cada fila: el `y` tiene que "
                f"traer el campo `{GROUP_FIELD}`. Con un modelo binario, poné "
                "`target: {name: grouped_label}` en el YAML; `discrete_survival` ya lo trae."
            )
        if int(self.n_bags) < 1:
            raise ValueError(f"n_bags={self.n_bags}: tiene que ser ≥ 1")
        # Binario: el modelo interno recibe `label`. Cualquier otro target (supervivencia)
        # lo recibe entero, que es lo que recibiría sin el envoltorio.
        inner_y = y[LABEL_FIELD] if set(names) == {LABEL_FIELD, GROUP_FIELD} else y

        groups = y[GROUP_FIELD].astype(str)
        rng = np.random.default_rng(int(self.random_state))
        is_matrix = hasattr(X, "shape") and not hasattr(X, "iloc")
        self.estimators_ = []
        self.bag_vehicles_ = []
        for _ in range(int(self.n_bags)):
            rows, n_distinct = vehicle_bootstrap_rows(groups, rng)
            X_bag = X[rows] if is_matrix else X.iloc[rows]
            model = get_model(self.base_model, dict(self.base_params or {}))
            model.fit(X_bag, inner_y[rows])
            self.estimators_.append(model)
            self.bag_vehicles_.append(n_distinct)
        self.classes_ = np.array([0, 1])
        self.n_vehicles_ = int(len(np.unique(groups)))
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        risk = np.mean([_positive_proba(model, X) for model in self.estimators_], axis=0)
        return np.column_stack([1.0 - risk, risk])

    def predict(self, X: Any) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def vehicle_bootstrap_rows(groups: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, int]:
    """Filas de un bootstrap de vehículos: se sortean N autos con reposición y cada uno entra entero.

    Devuelve `(índices de fila, cuántos vehículos distintos salieron)`. Un auto sorteado
    dos veces aporta todos sus cortes dos veces; uno que no salió no aporta ninguno.
    """
    vehicles, codes = np.unique(np.asarray(groups).astype(str), return_inverse=True)
    rows_by_vehicle = [np.flatnonzero(codes == v) for v in range(len(vehicles))]
    drawn = rng.integers(0, len(vehicles), size=len(vehicles))
    rows = np.concatenate([rows_by_vehicle[v] for v in drawn])
    return rows, int(len(np.unique(drawn)))


def _positive_proba(model: Any, X: Any) -> np.ndarray:
    proba = np.asarray(model.predict_proba(X), dtype=float)
    classes = list(getattr(model, "classes_", [0, 1]))
    return proba[:, classes.index(1)] if 1 in classes else proba[:, -1]
