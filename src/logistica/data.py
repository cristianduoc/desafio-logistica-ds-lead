"""Carga, validación y armado de tablas base."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import DATA_DIR, FILES


@dataclass(frozen=True)
class DataPackage:
    """Contenedor de los cuatro datasets del desafío, ya tipados."""

    pedidos: pd.DataFrame
    pronostico: pd.DataFrame
    capacidad: pd.DataFrame
    plan: pd.DataFrame


_DATE_COLS: dict[str, list[str]] = {
    "pedidos": ["fecha_despacho"],
    "pronostico": ["fecha_emision", "fecha_despacho"],
    "capacidad": ["fecha"],
    "plan": ["fecha_plan", "fecha_compra_real"],
}

_REQUIRED_COLS: dict[str, set[str]] = {
    "pedidos": {"centro_distribucion", "fecha_despacho", "unidades_reales", "canal_venta"},
    "pronostico": {"centro_distribucion", "fecha_emision", "fecha_despacho", "horizonte", "unidades_pronosticadas"},
    "capacidad": {"centro_distribucion", "fecha", "capacidad_maxima_unidades"},
    "plan": {
        "cliente_id", "centro_distribucion", "canal_venta", "fecha_plan", "unidades_plan",
        "fecha_compra_real", "unidades_reales", "fuera_de_plan",
    },
}


def load_data(data_dir: Path = DATA_DIR) -> DataPackage:
    """Lee los CSV, parsea fechas y valida el esquema mínimo esperado."""
    frames: dict[str, pd.DataFrame] = {}
    for key, fname in FILES.items():
        df = pd.read_csv(data_dir / fname, parse_dates=_DATE_COLS[key])
        missing = _REQUIRED_COLS[key] - set(df.columns)
        if missing:
            raise ValueError(f"{fname}: faltan columnas {sorted(missing)}")
        frames[key] = df
    frames["pronostico"]["h"] = frames["pronostico"]["horizonte"].str.replace("D+", "", regex=False).astype(int)
    return DataPackage(**frames)


def quality_report(pkg: DataPackage) -> pd.DataFrame:
    """Resumen de calidad: filas, nulos, duplicados y rango de fechas por dataset."""
    rows = []
    for name in FILES:
        df: pd.DataFrame = getattr(pkg, name)
        date_col = _DATE_COLS[name][-1] if name != "pronostico" else "fecha_despacho"
        rows.append({
            "dataset": name,
            "filas": len(df),
            "columnas": df.shape[1],
            "nulos": int(df.isna().sum().sum()),
            "duplicados": int(df.duplicated().sum()),
            "fecha_min": df[date_col].min().date(),
            "fecha_max": df[date_col].max().date(),
        })
    return pd.DataFrame(rows)


def daily_actuals(pedidos: pd.DataFrame) -> pd.DataFrame:
    """Demanda real diaria por CD (suma de canales). Columnas: centro_distribucion, fecha, real."""
    return (
        pedidos.groupby(["centro_distribucion", "fecha_despacho"], as_index=False)["unidades_reales"].sum()
        .rename(columns={"fecha_despacho": "fecha", "unidades_reales": "real"})
    )


def forecast_vs_actual(pronostico: pd.DataFrame, actuals: pd.DataFrame) -> pd.DataFrame:
    """Une cada pronóstico emitido con la demanda real de su fecha objetivo."""
    out = pronostico.merge(
        actuals.rename(columns={"fecha": "fecha_despacho"}),
        on=["centro_distribucion", "fecha_despacho"],
        how="left",
        validate="many_to_one",
    )
    return out.rename(columns={"unidades_pronosticadas": "pronostico"})
