"""Ingeniería de características sin fuga de información (leakage-safe).

Regla de disponibilidad: para un pronóstico emitido en `fecha_emision = E`, solo se
conoce la demanda real de fechas estrictamente anteriores a E (hasta E-1), y solo
los errores de pronósticos cuya `fecha_despacho` ya ocurrió (<= E-1).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLS: list[str] = [
    "log_pronostico",
    "h",
    "dow_sin",
    "dow_cos",
    "es_martes",
    "es_dia_pico",
    "indice_estacional_dow",
    "roll_mean_7",
    "roll_mean_14",
    "roll_std_7",
    "lag_mismo_dow",
    "mean_mismo_dow_4s",
    "ratio_pron_vs_media7",
    "ratio_sesgo_14",
    "ratio_sesgo_28",
]

# Set log-relativo (modelo campeón): señales históricas relativas al pronóstico base.
LOG_FEATURE_COLS: list[str] = [
    "log_sesgo_14",
    "log_sesgo_28",
    "log_media7_vs_pron",
    "log_media14_vs_pron",
    "log_dow4s_vs_pron",
    "log_lag_dow_vs_pron",
    "log_indice_estacional",
    "cv_7",
    "dow_sin",
    "dow_cos",
    "es_martes",
    "es_dia_pico",
]


def _actuals_matrix(actuals: pd.DataFrame) -> pd.DataFrame:
    """Pivot fecha x CD con calendario completo (días sin dato quedan NaN)."""
    mat = actuals.pivot(index="fecha", columns="centro_distribucion", values="real")
    full_idx = pd.date_range(mat.index.min(), mat.index.max(), freq="D")
    return mat.reindex(full_idx)


def _rolling_bias_ratio(fva: pd.DataFrame, window: int) -> pd.DataFrame:
    """Ratio Σreal/Σpronóstico de los últimos `window` días *ya realizados* (por CD).

    Se indexa por fecha de conocimiento: el ratio en la fecha E usa pronósticos con
    fecha_despacho en [E-window, E-1], de todos los horizontes.
    """
    daily = (
        fva.dropna(subset=["real"])
        .groupby(["centro_distribucion", "fecha_despacho"])[["real", "pronostico"]].sum()
        .unstack("centro_distribucion")
    )
    full_idx = pd.date_range(daily.index.min(), daily.index.max() + pd.Timedelta(days=1), freq="D")
    daily = daily.reindex(full_idx)
    real_sum = daily["real"].rolling(window, min_periods=max(3, window // 2)).sum().shift(1)
    pron_sum = daily["pronostico"].rolling(window, min_periods=max(3, window // 2)).sum().shift(1)
    ratio = (real_sum / pron_sum).stack(future_stack=True).rename(f"ratio_sesgo_{window}")
    ratio.index.names = ["fecha_emision", "centro_distribucion"]
    return ratio.reset_index()


def build_features(fva: pd.DataFrame, actuals: pd.DataFrame) -> pd.DataFrame:
    """Agrega features a la tabla pronóstico-vs-real (una fila por pronóstico emitido).

    Features:
      * Calendario: día de la semana (codificación cíclica), flag martes.
      * Pico: `es_dia_pico` (DOW con mayor demanda media en la historia conocida a E)
        e `indice_estacional_dow` (media del mismo DOW / media 28d).
      * Rolling: media 7/14d, desvío 7d, mismo DOW semana anterior, media mismo DOW 4 sem.
      * Sesgo móvil del pronóstico base: Σreal/Σpron últimos 14 y 28 días.
    """
    df = fva.copy()
    mat = _actuals_matrix(actuals)

    # Estadísticos rolling hasta E-1 → se calculan en la serie y se desplazan 1 día.
    def _stack(frame: pd.DataFrame, name: str) -> pd.Series:
        s = frame.stack(future_stack=True).rename(name)
        s.index.names = ["fecha_emision", "centro_distribucion"]
        return s

    known = mat.shift(1)  # valor en E = real de E-1
    roll = pd.concat([
        _stack(known.rolling(7, min_periods=5).mean(), "roll_mean_7"),
        _stack(known.rolling(14, min_periods=10).mean(), "roll_mean_14"),
        _stack(known.rolling(28, min_periods=14).mean(), "roll_mean_28"),
        _stack(known.rolling(7, min_periods=5).std(), "roll_std_7"),
    ], axis=1).reset_index()
    df = df.merge(roll, on=["fecha_emision", "centro_distribucion"], how="left")

    # Mismo día de semana (referido a la fecha objetivo D). Con h<=4, D-7 <= E-3: conocido.
    same_dow = pd.concat([
        _stack(mat.shift(7), "lag_mismo_dow"),
        _stack(pd.concat([mat.shift(k) for k in (7, 14, 21, 28)]).groupby(level=0).mean(), "mean_mismo_dow_4s"),
    ], axis=1)
    same_dow.index.names = ["fecha_despacho", "centro_distribucion"]
    df = df.merge(same_dow.reset_index(), on=["fecha_despacho", "centro_distribucion"], how="left")

    # Día pico según la historia conocida a la fecha de emisión (expanding, sin fuga).
    dow_mean_hist = _expanding_dow_peak(actuals)
    df = df.merge(dow_mean_hist, on="fecha_emision", how="left")

    dow = df["fecha_despacho"].dt.dayofweek
    df["dow"] = dow
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    df["es_martes"] = (dow == 1).astype(int)
    df["es_dia_pico"] = (dow == df["dow_pico"]).astype(float).where(df["dow_pico"].notna())
    df["indice_estacional_dow"] = df["mean_mismo_dow_4s"] / df["roll_mean_28"]
    df["log_pronostico"] = np.log(df["pronostico"])
    df["ratio_pron_vs_media7"] = df["pronostico"] / df["roll_mean_7"]

    for w in (14, 28):
        df = df.merge(_rolling_bias_ratio(fva, w), on=["fecha_emision", "centro_distribucion"], how="left")

    # Versión log-relativa: cada señal histórica expresada como log(señal / pronóstico base).
    # Con target log(real/base), el modelo lineal se vuelve una combinación ponderada
    # (en escala log) entre el pronóstico base, su sesgo reciente y la historia propia.
    df["log_sesgo_14"] = np.log(df["ratio_sesgo_14"])
    df["log_sesgo_28"] = np.log(df["ratio_sesgo_28"])
    df["log_media7_vs_pron"] = np.log(df["roll_mean_7"] / df["pronostico"])
    df["log_media14_vs_pron"] = np.log(df["roll_mean_14"] / df["pronostico"])
    df["log_dow4s_vs_pron"] = np.log(df["mean_mismo_dow_4s"] / df["pronostico"])
    df["log_lag_dow_vs_pron"] = np.log(df["lag_mismo_dow"] / df["pronostico"])
    df["log_indice_estacional"] = np.log(df["indice_estacional_dow"])
    df["cv_7"] = df["roll_std_7"] / df["roll_mean_7"]

    return df.drop(columns=["dow_pico"])


def _expanding_dow_peak(actuals: pd.DataFrame, min_days: int = 14) -> pd.DataFrame:
    """Para cada fecha E: DOW con mayor demanda media total usando reales < E."""
    tot = actuals.groupby("fecha")["real"].sum().sort_index()
    rows = []
    for e in pd.date_range(tot.index.min(), tot.index.max() + pd.Timedelta(days=1), freq="D"):
        hist = tot[tot.index < e]
        peak = hist.groupby(hist.index.dayofweek).mean().idxmax() if len(hist) >= min_days else np.nan
        rows.append({"fecha_emision": e, "dow_pico": peak})
    return pd.DataFrame(rows)
