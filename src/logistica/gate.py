"""Gate de historia mínima (Cold Start) y forecaster resiliente.

Regla:
  * historia continua >= `min_days`  → modo MODEL (se entrena/usa el modelo complementario)
  * historia continua <  `min_days`  → modo FALLBACK (resguardo seguro, sin excepciones)

Cadena de fallback (de más a menos informada):
  1. Corrección por sesgo móvil del propio CD (si hay >= 7 días de errores observados).
  2. Pronóstico base sin modificar.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from .config import MIN_HISTORY_DAYS
from .models import MLCorrector, RATIO_CLIP

logger = logging.getLogger(__name__)

Mode = Literal["MODEL", "FALLBACK"]


@dataclass(frozen=True)
class GateDecision:
    centro_distribucion: str
    dias_historia_continua: int
    modo: Mode
    motivo: str


def continuous_history_days(actuals: pd.DataFrame, centro: str, ref_date: pd.Timestamp) -> int:
    """Días consecutivos con dato válido (no nulo, >= 0) que terminan en `ref_date - 1`.

    Un hueco (día sin registro o valor inválido) corta la racha: la historia "continua"
    es la que alimenta las ventanas rolling sin interpolaciones.
    """
    s = actuals.loc[actuals["centro_distribucion"] == centro].set_index("fecha")["real"]
    s = s[s.index < ref_date]
    if s.empty:
        return 0
    idx = pd.date_range(end=ref_date - pd.Timedelta(days=1), periods=(ref_date - s.index.min()).days, freq="D")
    valid = s.reindex(idx).pipe(lambda x: x.notna() & (x >= 0)).to_numpy()[::-1]
    return int(np.argmin(valid)) if not valid.all() else int(len(valid))


class HistoryGate:
    """Evalúa la cobertura histórica por CD y decide el modo de operación."""

    def __init__(self, min_days: int = MIN_HISTORY_DAYS) -> None:
        self.min_days = min_days

    def evaluate(self, actuals: pd.DataFrame, centros: list[str], ref_date: pd.Timestamp) -> list[GateDecision]:
        decisions = []
        for cd in centros:
            try:
                n = continuous_history_days(actuals, cd, ref_date)
                if n >= self.min_days:
                    decisions.append(GateDecision(cd, n, "MODEL", f"{n} días >= {self.min_days}"))
                else:
                    decisions.append(GateDecision(cd, n, "FALLBACK", f"{n} días < {self.min_days}"))
            except Exception as exc:  # el gate nunca debe romper el pipeline
                logger.warning("Gate error en %s: %s", cd, exc)
                decisions.append(GateDecision(cd, 0, "FALLBACK", f"error evaluando historia: {exc}"))
        return decisions

    @staticmethod
    def to_frame(decisions: list[GateDecision]) -> pd.DataFrame:
        return pd.DataFrame([d.__dict__ for d in decisions])


class ResilientForecaster:
    """Orquesta gate + modelo + fallback. `predict` nunca lanza excepciones."""

    def __init__(self, gate: HistoryGate | None = None, model: MLCorrector | None = None) -> None:
        self.gate = gate or HistoryGate()
        self.model = model or MLCorrector("ridge", feature_set="log", per_horizon=True)
        self.decisions_: dict[str, GateDecision] = {}
        self.model_trained_: bool = False

    def fit(self, features: pd.DataFrame, actuals: pd.DataFrame, ref_date: pd.Timestamp) -> "ResilientForecaster":
        centros = sorted(features["centro_distribucion"].unique())
        self.decisions_ = {d.centro_distribucion: d for d in self.gate.evaluate(actuals, centros, ref_date)}
        eligible = [cd for cd, d in self.decisions_.items() if d.modo == "MODEL"]
        train = features[
            features["centro_distribucion"].isin(eligible)
            & (features["fecha_despacho"] < ref_date)
            & features["real"].notna()
        ].dropna(subset=["roll_mean_14", "ratio_sesgo_14"])
        try:
            if train.empty:
                raise ValueError("ningún CD habilitado para entrenar")
            self.model.fit(train)
            self.model_trained_ = True
        except Exception as exc:
            logger.warning("Entrenamiento omitido → todo en FALLBACK: %s", exc)
            self.model_trained_ = False
        return self

    @staticmethod
    def fallback_predict(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Sesgo móvil 14d si existe; si no, pronóstico base."""
        ratio = df["ratio_sesgo_14"]
        pred = np.where(ratio.notna(), df["pronostico"] * ratio.clip(*RATIO_CLIP), df["pronostico"])
        source = np.where(ratio.notna(), "FALLBACK_SESGO", "FALLBACK_BASE")
        return pred.astype(float), source

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df[["centro_distribucion", "fecha_emision", "fecha_despacho", "horizonte", "pronostico"]].copy()
        pred, source = self.fallback_predict(df)
        use_model = df["centro_distribucion"].map(
            lambda cd: cd in self.decisions_ and self.decisions_[cd].modo == "MODEL"
        ).to_numpy() & self.model_trained_
        if use_model.any():
            try:
                pred[use_model] = self.model.predict(df.loc[use_model])
                source[use_model] = "MODEL"
            except Exception as exc:
                logger.warning("Fallo en inferencia del modelo → FALLBACK: %s", exc)
        out["pronostico_corregido"] = pred
        out["fuente"] = source
        return out
