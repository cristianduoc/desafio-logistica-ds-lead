"""Validación cruzada con origen rodante (rolling-origin / expanding window)."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import pandas as pd

from .metrics import error_table, wape
from .models import ForecastModel

ModelFactory = Callable[[], ForecastModel]

WARMUP_COLS: list[str] = ["roll_mean_14", "ratio_sesgo_14"]


def make_origins(df: pd.DataFrame, first_origin_day: int = 21, step_days: int = 7) -> list[pd.Timestamp]:
    """Orígenes semanales a partir del día `first_origin_day` de la serie."""
    start = df["fecha_emision"].min()
    end = df["fecha_emision"].max()
    return list(pd.date_range(start + pd.Timedelta(days=first_origin_day), end, freq=f"{step_days}D"))


def rolling_origin_backtest(
    df: pd.DataFrame,
    factories: Sequence[ModelFactory],
    origins: Sequence[pd.Timestamp],
    step_days: int = 7,
    train_window_days: int | None = None,
) -> pd.DataFrame:
    """Entrena en cada origen T y predice los pronósticos emitidos en [T, T+step).

    * Train: filas con `fecha_despacho < T` (real ya observado) y features de warm-up.
      Si `train_window_days` se define, solo los últimos N días (ventana deslizante).
    * Test: filas con `fecha_emision` en [T, T+step). Sus features solo usan info < E.

    Devuelve un DataFrame largo con una columna por modelo y la columna `fold`.
    """
    base = df.dropna(subset=["real"])
    outputs = []
    for k, origin in enumerate(origins):
        train = base[(base["fecha_despacho"] < origin)].dropna(subset=WARMUP_COLS)
        if train_window_days is not None:
            train = train[train["fecha_despacho"] >= origin - pd.Timedelta(days=train_window_days)]
        test = base[(base["fecha_emision"] >= origin) & (base["fecha_emision"] < origin + pd.Timedelta(days=step_days))]
        if train.empty or test.empty:
            continue
        out = test[["centro_distribucion", "fecha_emision", "fecha_despacho", "horizonte", "h", "real", "pronostico"]].copy()
        out["fold"] = k
        out["origen"] = origin
        for factory in factories:
            model = factory()
            out[model.name] = model.fit(train).predict(test)
        outputs.append(out)
    return pd.concat(outputs, ignore_index=True)


def summarize_backtest(preds: pd.DataFrame, model_names: Sequence[str], by: str | list[str] | None = None) -> pd.DataFrame:
    """WAPE y Bias de cada modelo sobre exactamente las mismas filas de test."""
    tables = []
    for name in model_names:
        t = error_table(preds, by=by, pred_col=name)[["WAPE", "Bias"]]
        t.columns = pd.MultiIndex.from_product([[name], t.columns])
        tables.append(t)
    return pd.concat(tables, axis=1)


def history_sensitivity(
    fva: pd.DataFrame,
    actuals: pd.DataFrame,
    factories: Sequence[ModelFactory],
    lengths: Sequence[int],
    test_start: pd.Timestamp,
    test_days: int = 14,
) -> pd.DataFrame:
    """Curva de aprendizaje vs. historia disponible (insumo para fijar el umbral del gate).

    Para cada longitud L se simula que *todos* los CD tienen solo L días de historia
    antes de `test_start`: se truncan reales y pronósticos, se recalculan las features
    (las ventanas rolling también quedan truncadas), se entrena con lo observado y se
    evalúa sobre los pronósticos emitidos en [test_start, test_start + test_days).
    """
    from .features import build_features  # import local para evitar ciclo

    rows = []
    for L in lengths:
        cutoff = test_start - pd.Timedelta(days=L)
        a_t = actuals[actuals["fecha"] >= cutoff]
        f_t = fva[(fva["fecha_emision"] >= cutoff) & (fva["fecha_despacho"] >= cutoff)]
        feats = build_features(f_t, a_t).dropna(subset=["real"])
        train = feats[feats["fecha_despacho"] < test_start]
        test = feats[(feats["fecha_emision"] >= test_start) & (feats["fecha_emision"] < test_start + pd.Timedelta(days=test_days))]
        row: dict[str, float] = {"dias_historia": L, "filas_train": len(train)}
        for factory in factories:
            model = factory()
            try:
                pred = model.fit(train).predict(test)
                row[model.name] = wape(test["real"], pred)
            except Exception:
                row[model.name] = float("nan")
        rows.append(row)
    return pd.DataFrame(rows).set_index("dias_historia")
