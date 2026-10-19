"""Tests de las piezas críticas: métricas, anti-fuga de features, gate y resiliencia."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from logistica.data import daily_actuals, forecast_vs_actual, load_data
from logistica.features import build_features
from logistica.gate import HistoryGate, ResilientForecaster, continuous_history_days
from logistica.metrics import bias, wape


@pytest.fixture(scope="module")
def pkg():
    return load_data()


@pytest.fixture(scope="module")
def frames(pkg):
    actuals = daily_actuals(pkg.pedidos)
    fva = forecast_vs_actual(pkg.pronostico, actuals)
    return actuals, fva, build_features(fva, actuals)


def test_metrics_definitions():
    real, pred = np.array([100.0, 200.0]), np.array([90.0, 230.0])
    assert wape(real, pred) == pytest.approx(40 / 300)
    assert bias(real, pred) == pytest.approx(-20 / 300)  # negativo = sobreestimación
    assert np.isnan(wape([0.0], [1.0]))


def test_forecast_join_is_complete(frames):
    _, fva, _ = frames
    assert fva["real"].notna().all()
    assert ((fva["fecha_despacho"] - fva["fecha_emision"]).dt.days == fva["h"]).all()


def test_features_have_no_leakage(frames):
    """Alterar la demanda real de E en adelante no debe cambiar features emitidas en E."""
    actuals, fva, feats = frames
    cut = pd.Timestamp("2026-02-10")
    tampered = actuals.copy()
    tampered.loc[tampered["fecha"] >= cut, "real"] *= 10
    fva_t = forecast_vs_actual(fva[["centro_distribucion", "fecha_emision", "fecha_despacho", "horizonte", "h", "pronostico"]]
                               .rename(columns={"pronostico": "unidades_pronosticadas"}), tampered)
    feats_t = build_features(fva_t, tampered)
    cols = ["roll_mean_7", "roll_mean_14", "lag_mismo_dow", "mean_mismo_dow_4s", "ratio_sesgo_14", "ratio_sesgo_28"]
    key = ["centro_distribucion", "fecha_emision", "fecha_despacho"]
    a = feats[feats["fecha_emision"] == cut].set_index(key)[cols].sort_index()
    b = feats_t[feats_t["fecha_emision"] == cut].set_index(key)[cols].sort_index()
    pd.testing.assert_frame_equal(a, b)


def test_continuous_history_breaks_on_gap(frames):
    actuals, _, _ = frames
    ref = pd.Timestamp("2026-02-15")
    assert continuous_history_days(actuals, "CD_NORTE", ref) == 45
    gap = actuals[~((actuals["centro_distribucion"] == "CD_NORTE") & (actuals["fecha"] == "2026-02-10"))]
    assert continuous_history_days(gap, "CD_NORTE", ref) == 4
    assert continuous_history_days(actuals, "CD_INEXISTENTE", ref) == 0


def test_gate_threshold(frames):
    actuals, _, _ = frames
    centros = sorted(actuals["centro_distribucion"].unique())
    assert {d.modo for d in HistoryGate(45).evaluate(actuals, centros, pd.Timestamp("2026-02-15"))} == {"MODEL"}
    assert {d.modo for d in HistoryGate(45).evaluate(actuals, centros, pd.Timestamp("2026-02-14"))} == {"FALLBACK"}


def test_resilient_forecaster_never_raises(frames):
    actuals, _, feats = frames
    ref = pd.Timestamp("2026-02-15")
    for a_bad in (actuals.iloc[0:0], actuals.assign(real=np.nan)):
        out = ResilientForecaster().fit(feats, a_bad, ref).predict(feats[feats["fecha_emision"] >= ref])
        assert out["pronostico_corregido"].notna().all()
        assert out["fuente"].str.startswith("FALLBACK").all()


def test_model_beats_baseline_in_backtest(frames):
    from logistica.backtest import make_origins, rolling_origin_backtest
    from logistica.models import BaselineModel, MLCorrector

    _, _, feats = frames
    bt = rolling_origin_backtest(feats, [BaselineModel, lambda: MLCorrector("ridge", "log", True)], make_origins(feats))
    assert wape(bt["real"], bt["ML ridge log por horizonte"]) < wape(bt["real"], bt["Base"])
