"""Modelos de corrección del pronóstico base.

Todos comparten la interfaz `fit(train) -> self` y `predict(df) -> np.ndarray`, donde
`df` es la tabla de `features.build_features`. La predicción está en unidades.
"""

from __future__ import annotations

from typing import Literal, Protocol

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import RANDOM_STATE
from .features import FEATURE_COLS, LOG_FEATURE_COLS

RATIO_CLIP: tuple[float, float] = (0.5, 2.0)
FeatureSet = Literal["raw", "log"]


class ForecastModel(Protocol):
    name: str

    def fit(self, train: pd.DataFrame) -> "ForecastModel": ...

    def predict(self, df: pd.DataFrame) -> np.ndarray: ...


class BaselineModel:
    """Pronóstico base sin modificar (referencia)."""

    name = "Base"

    def fit(self, train: pd.DataFrame) -> "BaselineModel":
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return df["pronostico"].to_numpy(dtype=float)


class RollingBiasCorrector:
    """Corrección multiplicativa por el sesgo reciente observado de cada CD.

    pred = pronóstico × Σreal/Σpron (últimos `window` días realizados). Si el ratio no
    está disponible (historia insuficiente) se usa 1.0, es decir, el pronóstico base.
    """

    def __init__(self, window: int = 28) -> None:
        self.window = window
        self.name = f"Corrección sesgo {window}d"

    def fit(self, train: pd.DataFrame) -> "RollingBiasCorrector":
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        ratio = df[f"ratio_sesgo_{self.window}"].fillna(1.0).clip(*RATIO_CLIP)
        return (df["pronostico"] * ratio).to_numpy(dtype=float)


class MLCorrector:
    """Modelo global (todos los CD) que aprende el log-ratio real/pronóstico.

    target = log(real / pronóstico)  →  pred = pronóstico × exp(ŷ)

    Modelar el ratio (y no las unidades) hace que el modelo sea agnóstico a la escala
    de cada CD y lo vuelve aplicable a centros nuevos vía las features de sesgo móvil.

    Parámetros:
        estimator: "ridge" | "hgb".
        feature_set: "raw" (features en niveles, iteración 1) | "log" (señales relativas
            al pronóstico base en escala log, iteración 2).
        per_horizon: si True entrena un modelo por horizonte (D+1..D+4): la confianza
            óptima en el pronóstico base vs. la historia cambia con la anticipación.
        include_cd: agrega one-hot del CD (no transferible a CDs nuevos).
    """

    def __init__(
        self,
        estimator: Literal["ridge", "hgb"] = "ridge",
        feature_set: FeatureSet = "log",
        per_horizon: bool = True,
        include_cd: bool = False,
        alpha: float = 1.0,
        name: str | None = None,
    ) -> None:
        self.estimator = estimator
        self.feature_set = feature_set
        self.per_horizon = per_horizon
        self.include_cd = include_cd
        self.alpha = alpha
        self.features: list[str] = list(LOG_FEATURE_COLS if feature_set == "log" else FEATURE_COLS)
        if not per_horizon and feature_set == "log":
            self.features.append("h")
        default = f"ML {estimator} {feature_set}" + (" por horizonte" if per_horizon else " global") + (" + CD" if include_cd else "")
        self.name = name or default
        self.pipelines_: dict[int, Pipeline] = {}

    def _build(self) -> Pipeline:
        transformers: list[tuple] = [(
            "num",
            Pipeline([("imp", SimpleImputer(strategy="median", keep_empty_features=True)), ("sc", StandardScaler())]),
            self.features,
        )]
        if self.include_cd:
            transformers.append(("cd", OneHotEncoder(handle_unknown="ignore"), ["centro_distribucion"]))
        if self.estimator == "ridge":
            est = Ridge(alpha=self.alpha)
        elif self.estimator == "hgb":
            est = HistGradientBoostingRegressor(
                max_depth=3, learning_rate=0.05, max_iter=300, min_samples_leaf=20,
                l2_regularization=1.0, random_state=RANDOM_STATE,
            )
        else:
            raise ValueError(f"estimator desconocido: {self.estimator}")
        return Pipeline([("prep", ColumnTransformer(transformers)), ("model", est)])

    def _groups(self, df: pd.DataFrame) -> dict[int, pd.Index]:
        if self.per_horizon:
            return {int(h): g.index for h, g in df.groupby("h")}
        return {0: df.index}

    def fit(self, train: pd.DataFrame) -> "MLCorrector":
        if train.empty:
            raise ValueError("train vacío")
        y = np.log(train["real"] / train["pronostico"])
        self.pipelines_ = {k: self._build().fit(train.loc[idx], y.loc[idx]) for k, idx in self._groups(train).items()}
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if not self.pipelines_:
            raise RuntimeError("Modelo no entrenado")
        log_ratio = pd.Series(0.0, index=df.index)
        for k, idx in self._groups(df).items():
            pipe = self.pipelines_.get(k)
            if pipe is None:  # horizonte no visto en train → sin corrección ML
                continue
            log_ratio.loc[idx] = pipe.predict(df.loc[idx])
        log_ratio = log_ratio.clip(*np.log(RATIO_CLIP))
        return (df["pronostico"] * np.exp(log_ratio)).to_numpy(dtype=float)

    def coefficients(self) -> pd.DataFrame:
        """Coeficientes estandarizados (solo Ridge): una columna por modelo/horizonte."""
        if not self.pipelines_ or self.estimator != "ridge":
            raise RuntimeError("Disponible solo para Ridge entrenado")
        cols = {}
        for k, pipe in self.pipelines_.items():
            names = [n.replace("num__", "") for n in pipe.named_steps["prep"].get_feature_names_out()]
            cols[f"D+{k}" if self.per_horizon else "global"] = pd.Series(pipe.named_steps["model"].coef_, index=names)
        return pd.DataFrame(cols)
