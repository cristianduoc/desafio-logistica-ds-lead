"""Traducción del error de pronóstico a impacto operativo (flota y fletes de emergencia).

Supuesto operativo (parametrizable): la flota se reserva con el pronóstico del
horizonte de planificación, acotado por la capacidad máxima del CD.
  * Demanda real > reserva  → unidades a cubrir con flete de emergencia (subestimación).
  * Reserva > demanda real  → capacidad reservada ociosa (sobreestimación).
"""

from __future__ import annotations

import pandas as pd


def fleet_impact(
    preds: pd.DataFrame,
    capacity: pd.DataFrame,
    model_cols: list[str],
    horizonte: str = "D+2",
    unidades_por_camion: float = 300.0,
) -> pd.DataFrame:
    """Unidades de emergencia, ociosas y camiones-equivalentes por modelo."""
    df = preds[preds["horizonte"] == horizonte].merge(
        capacity.rename(columns={"fecha": "fecha_despacho"}), on=["centro_distribucion", "fecha_despacho"], how="left"
    )
    rows = []
    for col in model_cols:
        reserva = df[col].clip(upper=df["capacidad_maxima_unidades"])
        emergencia = (df["real"] - reserva).clip(lower=0)
        ociosa = (reserva - df["real"]).clip(lower=0)
        rows.append({
            "modelo": col,
            "dias_cd": len(df),
            "dias_con_flete_emergencia": int((emergencia > 0).sum()),
            "unidades_emergencia": emergencia.sum(),
            "unidades_ociosas": ociosa.sum(),
            "camiones_emergencia_eq": emergencia.sum() / unidades_por_camion,
            "camiones_ociosos_eq": ociosa.sum() / unidades_por_camion,
            "nivel_servicio_sin_emergencia": 1 - emergencia.sum() / df["real"].sum(),
        })
    return pd.DataFrame(rows).set_index("modelo")
