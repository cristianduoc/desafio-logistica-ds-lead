"""Tarea 2: análisis del plan de entrega a clientes y simulación de aplanamiento."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DIAS_SEMANA


def prepare_plan(plan: pd.DataFrame) -> pd.DataFrame:
    """Agrega desvíos de fecha y volumen a nivel de línea de plan."""
    df = plan.copy()
    df["desvio_dias"] = (df["fecha_compra_real"] - df["fecha_plan"]).dt.days
    df["desvio_unidades"] = df["unidades_reales"] - df["unidades_plan"]
    df["dow_plan"] = pd.Categorical(df["fecha_plan"].dt.dayofweek.map(dict(enumerate(DIAS_SEMANA))), DIAS_SEMANA, ordered=True)
    df["dow_real"] = pd.Categorical(df["fecha_compra_real"].dt.dayofweek.map(dict(enumerate(DIAS_SEMANA))), DIAS_SEMANA, ordered=True)
    df["tipo_desvio_fecha"] = np.select(
        [df["desvio_dias"] < 0, df["desvio_dias"] > 0], ["Adelantada", "Atrasada"], "En fecha"
    )
    return df


def plan_bias(df: pd.DataFrame, by: str | list[str] | None = None) -> pd.DataFrame:
    """Sesgo del plan en volumen (misma convención del enunciado) y en fecha."""

    def _agg(g: pd.DataFrame) -> pd.Series:
        real = g["unidades_reales"].sum()
        return pd.Series({
            "lineas": len(g),
            "unidades_plan": g["unidades_plan"].sum(),
            "unidades_reales": real,
            "bias_volumen": (real - g["unidades_plan"].sum()) / real,
            "wape_volumen": (g["unidades_reales"] - g["unidades_plan"]).abs().sum() / real,
            "pct_lineas_real_mayor_plan": (g["desvio_unidades"] > 0).mean(),
            "desvio_dias_medio": g["desvio_dias"].mean(),
            "desvio_dias_medio_fuera_plan": g.loc[g["fuera_de_plan"] == 1, "desvio_dias"].mean(),
            "pct_atrasadas": (g["desvio_dias"] > 0).mean(),
            "pct_adelantadas": (g["desvio_dias"] < 0).mean(),
        })

    if by is None:
        return _agg(df).to_frame("Total").T
    return df.groupby(by, observed=True).apply(_agg, include_groups=False)


def off_plan_rate(df: pd.DataFrame, by: str | list[str] | None = None) -> pd.DataFrame:
    """% de compras fuera de plan (por líneas y ponderado por unidades)."""

    def _agg(g: pd.DataFrame) -> pd.Series:
        return pd.Series({
            "lineas": len(g),
            "fuera_de_plan": int(g["fuera_de_plan"].sum()),
            "pct_fuera_plan": g["fuera_de_plan"].mean(),
            "pct_unidades_fuera_plan": g.loc[g["fuera_de_plan"] == 1, "unidades_reales"].sum() / g["unidades_reales"].sum(),
        })

    if by is None:
        return _agg(df).to_frame("Total").T
    return df.groupby(by, observed=True).apply(_agg, include_groups=False)


def client_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Métricas por cliente: adhesión, variación de fecha y volumen."""
    g = df.groupby(["cliente_id", "centro_distribucion", "canal_venta"], observed=True)
    out = g.apply(lambda x: pd.Series({
        "lineas": len(x),
        "adhesion_fecha": 1 - x["fuera_de_plan"].mean(),
        "desvio_dias_abs_medio": x["desvio_dias"].abs().mean(),
        "bias_volumen": (x["unidades_reales"].sum() - x["unidades_plan"].sum()) / x["unidades_reales"].sum(),
        "wape_volumen": (x["unidades_reales"] - x["unidades_plan"]).abs().sum() / x["unidades_reales"].sum(),
        "unidades_reales": x["unidades_reales"].sum(),
        "dia_plan_habitual": x["dow_plan"].mode().iat[0],
    }), include_groups=False).reset_index()
    return out.sort_values("adhesion_fecha")


def concentration_curve(values: pd.Series) -> tuple[pd.DataFrame, float]:
    """Curva de Lorenz/Pareto y coeficiente de Gini de un vector no negativo."""
    v = np.sort(values.to_numpy(dtype=float))[::-1]
    cum = np.cumsum(v) / v.sum() if v.sum() > 0 else np.zeros_like(v)
    curve = pd.DataFrame({"pct_clientes": np.arange(1, len(v) + 1) / len(v), "pct_acumulado": cum})
    asc = v[::-1]
    n = len(asc)
    gini = float((2 * np.sum(np.arange(1, n + 1) * asc) / (n * asc.sum())) - (n + 1) / n) if asc.sum() > 0 else 0.0
    return curve, gini


def load_profile(actuals: pd.DataFrame, capacity: pd.DataFrame) -> pd.DataFrame:
    """Une demanda diaria real con capacidad y calcula utilización/sobrecarga."""
    df = actuals.merge(capacity, on=["centro_distribucion", "fecha"], how="left")
    df["utilizacion"] = df["real"] / df["capacidad_maxima_unidades"]
    df["exceso"] = (df["real"] - df["capacidad_maxima_unidades"]).clip(lower=0)
    df["dow"] = df["fecha"].dt.dayofweek
    return df


def weekly_profile(load: pd.DataFrame, col: str = "real") -> pd.DataFrame:
    """Perfil por día de semana usando **promedios diarios** (60 días no tienen el mismo
    número de lunes que de jueves, así que sumar sesgaría el índice)."""
    daily = load.groupby("fecha").agg(dem=(col, "sum"), cap=("capacidad_maxima_unidades", "sum"))
    daily["dow"] = daily.index.dayofweek
    over = load.assign(sobre=load[col] > load["capacidad_maxima_unidades"]).groupby("dow")["sobre"].sum()
    prof = daily.groupby("dow").agg(demanda_media=("dem", "mean"), capacidad_media=("cap", "mean"), n_dias=("dem", "size"))
    prof["indice_vs_promedio"] = prof["demanda_media"] / prof["demanda_media"].mean()
    prof["utilizacion"] = prof["demanda_media"] / prof["capacidad_media"]
    prof["dias_cd_sobre_capacidad"] = over
    prof.index = [DIAS_SEMANA[i] for i in prof.index]
    return prof


def peak_kpis(load: pd.DataFrame, col: str = "real") -> pd.Series:
    """KPIs de forma de la curva: pico/promedio (por DOW, con medias), CV diario, días y
    unidades sobre capacidad, utilización p95."""
    prof = weekly_profile(load, col)
    daily_cv = load.groupby("centro_distribucion")[col].apply(lambda s: s.std() / s.mean()).mean()
    exceso = (load[col] - load["capacidad_maxima_unidades"]).clip(lower=0)
    return pd.Series({
        "ratio_pico_promedio_dow": prof["demanda_media"].max() / prof["demanda_media"].mean(),
        "participacion_martes": prof.loc["Martes", "demanda_media"] / prof["demanda_media"].sum(),
        "cv_diario_medio": daily_cv,
        "dias_cd_sobre_capacidad": int((exceso > 0).sum()),
        "unidades_sobre_capacidad": exceso.sum(),
        "utilizacion_p95": (load[col] / load["capacidad_maxima_unidades"]).quantile(0.95),
    })


def simulate_peak_shift(load: pd.DataFrame, shift_share: float, target_util: float = 0.85) -> pd.DataFrame:
    """Palanca estática: desplaza `shift_share` de la demanda de cada martes hacia
    lunes/miércoles/jueves de la misma semana, priorizando el día con más holgura
    (capacidad × target_util). El volumen no se pierde, se mueve."""
    df = load.copy()
    df["sim"] = df["real"].astype(float)
    df["semana"] = df["fecha"].dt.to_period("W-SUN")
    for _, wk in df.groupby(["centro_distribucion", "semana"]):
        tue = wk[wk["dow"] == 1]
        recv = wk[wk["dow"].isin([0, 2, 3])]
        if tue.empty or recv.empty:
            continue
        i_tue = tue.index[0]
        to_move = df.at[i_tue, "sim"] * shift_share
        df.at[i_tue, "sim"] -= to_move
        _distribute(df, recv.index, to_move, target_util)
    return df.drop(columns="semana")


def simulate_dynamic_slotting(
    load: pd.DataFrame, max_util: float = 0.90, flex_share: float = 0.30, window: int = 2
) -> pd.DataFrame:
    """Palanca dinámica (la que se propone): cualquier día cuya demanda supere
    `max_util` × capacidad recomienda moverse a ±`window` días con holgura, hasta un
    máximo de `flex_share` del volumen del día (clientes flexibles). Solo mueve lo que
    cabe en la holgura de los días receptores: nunca crea una nueva sobrecarga."""
    df = load.sort_values(["centro_distribucion", "fecha"]).copy()
    df["sim"] = df["real"].astype(float)
    for _, g in df.groupby("centro_distribucion"):
        idx = g.index.to_list()
        for pos, i in enumerate(idx):
            limit = df.at[i, "capacidad_maxima_unidades"] * max_util
            excess = min(df.at[i, "sim"] - limit, flex_share * df.at[i, "real"])
            if excess <= 0:
                continue
            neigh = [idx[p] for p in range(max(0, pos - window), min(len(idx), pos + window + 1)) if p != pos]
            for j in sorted(neigh, key=lambda j: df.at[j, "sim"] / df.at[j, "capacidad_maxima_unidades"]):
                slack = df.at[j, "capacidad_maxima_unidades"] * max_util - df.at[j, "sim"]
                move = min(max(slack, 0.0), excess)
                df.at[j, "sim"] += move
                df.at[i, "sim"] -= move
                excess -= move
                if excess <= 1e-9:
                    break
    return df.sort_index()


def _distribute(df: pd.DataFrame, recv_idx: pd.Index, amount: float, target_util: float) -> None:
    """Reparte `amount` en `recv_idx` llenando primero la holgura y luego en partes iguales."""
    for j in sorted(recv_idx, key=lambda j: df.at[j, "sim"] / df.at[j, "capacidad_maxima_unidades"]):
        slack = df.at[j, "capacidad_maxima_unidades"] * target_util - df.at[j, "sim"]
        move = min(max(slack, 0.0), amount)
        df.at[j, "sim"] += move
        amount -= move
    if amount > 1e-9:
        df.loc[recv_idx, "sim"] += amount / len(recv_idx)


def day_migration(df_plan: pd.DataFrame) -> pd.DataFrame:
    """Matriz día planificado → día real de compra (solo líneas fuera de plan)."""
    off = df_plan[df_plan["fuera_de_plan"] == 1]
    return pd.crosstab(off["dow_plan"], off["dow_real"], margins=True, margins_name="Total")


def shiftable_tuesday_volume(df_plan: pd.DataFrame) -> pd.Series:
    """Volumen martes planificado de clientes que ya demostraron flexibilidad
    (al menos una compra fuera de plan): pool realista para incentivar el cambio de día."""
    flex_clients = df_plan.groupby("cliente_id")["fuera_de_plan"].max()
    flex_clients = flex_clients[flex_clients == 1].index
    tue = df_plan[df_plan["dow_plan"] == "Martes"]
    return pd.Series({
        "unidades_plan_martes": tue["unidades_plan"].sum(),
        "unidades_martes_clientes_flexibles": tue.loc[tue["cliente_id"].isin(flex_clients), "unidades_plan"].sum(),
        "pct_martes_flexible": tue.loc[tue["cliente_id"].isin(flex_clients), "unidades_plan"].sum() / tue["unidades_plan"].sum(),
        "clientes_flexibles": len(flex_clients),
        "clientes_totales": df_plan["cliente_id"].nunique(),
    })
