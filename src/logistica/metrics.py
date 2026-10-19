"""Métricas de error de pronóstico (definiciones del enunciado)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def wape(real: Sequence[float] | pd.Series, pred: Sequence[float] | pd.Series) -> float:
    """WAPE = Σ|real − pred| / Σ|real|."""
    r, p = np.asarray(real, dtype=float), np.asarray(pred, dtype=float)
    denom = np.abs(r).sum()
    return float(np.abs(r - p).sum() / denom) if denom > 0 else np.nan


def bias(real: Sequence[float] | pd.Series, pred: Sequence[float] | pd.Series) -> float:
    """Bias = Σ(real − pred) / Σ|real|. Positivo = subestimación, negativo = sobreestimación."""
    r, p = np.asarray(real, dtype=float), np.asarray(pred, dtype=float)
    denom = np.abs(r).sum()
    return float((r - p).sum() / denom) if denom > 0 else np.nan


def error_table(
    df: pd.DataFrame,
    by: str | list[str] | None,
    real_col: str = "real",
    pred_col: str = "pronostico",
) -> pd.DataFrame:
    """Tabla de WAPE, Bias y volumen agrupada por `by` (None = total)."""

    def _agg(g: pd.DataFrame) -> pd.Series:
        return pd.Series({
            "WAPE": wape(g[real_col], g[pred_col]),
            "Bias": bias(g[real_col], g[pred_col]),
            "n": len(g),
            "unidades_reales": g[real_col].sum(),
        })

    if by is None:
        return _agg(df).to_frame("Total").T
    return df.groupby(by)[[real_col, pred_col]].apply(_agg)
