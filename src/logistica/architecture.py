"""Diagrama conceptual de la arquitectura AWS (renderizado con matplotlib)."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from .viz import COLORS

_W, _H = 2.3, 0.82  # ancho y alto de cada caja
_LANES: list[tuple[str, float, str]] = [  # (título, x, color de fondo)
    ("Fuentes", 0.2, "#f0efec"),
    ("Persistencia · Data Lake", 2.9, "#e6f0fc"),
    ("Batch / Re-entrenamiento", 5.6, "#e7f6ef"),
    ("Recomendación / Inferencia", 8.3, "#fdeee6"),
    ("Métricas, estados y monitoreo", 11.0, "#f3effc"),
]
_BOXES: dict[str, tuple[float, float, str]] = {  # (x, y esquina inferior izquierda, texto)
    "erp": (0.2, 5.0, "ERP / OMS\npedidos · capacidad · plan"),
    "fb": (0.2, 3.6, "Pronóstico base\n(sistema actual)"),
    "s3r": (2.9, 5.0, "S3 raw"),
    "glue": (2.9, 3.6, "Glue Jobs + Data Catalog\nvalidación de calidad"),
    "s3c": (2.9, 2.2, "S3 curated\nParquet por fecha / CD"),
    "sf": (5.6, 5.55, "EventBridge Scheduler\n+ Step Functions"),
    "gate": (5.6, 4.4, "Lambda · Gate 45 días\nhistoria continua por CD"),
    "smp": (5.6, 3.2, "SageMaker Pipelines\nfeatures → train → backtest"),
    "reg": (5.6, 2.0, "Model Registry\npromueve si WAPE ≤ campeón"),
    "btj": (5.6, 0.8, "Batch Transform\npronóstico corregido D+1..D+4"),
    "app": (8.3, 5.55, "App B2B / Web\nclientes y agentes"),
    "apigw": (8.3, 4.4, "API Gateway\n(Cognito / IAM)"),
    "svc": (8.3, 3.2, "Lambda / Fargate\nmotor de fecha recomendada"),
    "ddb": (8.3, 0.8, "DynamoDB\npronóstico · capacidad · modo CD"),
    "cw": (11.0, 4.4, "CloudWatch + Model Monitor\nalarmas · drift · % fallback"),
    "rds": (11.0, 3.2, "Aurora / RDS\nWAPE·Bias · decisiones gate"),
    "qs": (11.0, 2.0, "QuickSight\ndashboard de KPIs"),
}
_EDGES: list[tuple[str, str, str]] = [
    ("erp", "s3r", ""), ("fb", "s3r", ""), ("s3r", "glue", ""), ("glue", "s3c", ""),
    ("s3c", "gate", ""), ("sf", "gate", ""), ("gate", "smp", "CDs ≥ 45d"), ("smp", "reg", ""), ("reg", "btj", ""),
    ("btj", "ddb", "diario"),
    ("app", "apigw", ""), ("apigw", "svc", ""), ("svc", "ddb", "lectura ms"),
    ("svc", "rds", "decisiones"), ("svc", "cw", "logs"), ("rds", "qs", ""),
]


def _anchor(key: str, other: str) -> tuple[float, float]:
    x, y, _ = _BOXES[key]
    ox, oy, _ = _BOXES[other]
    if abs(ox - x) > 0.5:
        return (x + _W, y + _H / 2) if ox > x else (x, y + _H / 2)
    return (x + _W / 2, y + _H) if oy > y else (x + _W / 2, y)


def draw_architecture(figsize: tuple[float, float] = (15, 7)) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 13.6); ax.set_ylim(0.1, 7.25); ax.axis("off")
    for title, x, color in _LANES:
        ax.add_patch(FancyBboxPatch((x - 0.12, 0.6), _W + 0.24, 6.25, boxstyle="round,pad=0.02,rounding_size=0.15",
                                    fc=color, ec="none", zorder=0))
        ax.text(x + _W / 2, 6.6, title, ha="center", va="center", fontsize=10, weight="bold", color=COLORS["ink"])
    for x, y, label in _BOXES.values():
        ax.add_patch(FancyBboxPatch((x, y), _W, _H, boxstyle="round,pad=0.02,rounding_size=0.08",
                                    fc="white", ec="#c3c2b7", lw=1, zorder=2))
        ax.text(x + _W / 2, y + _H / 2, label, ha="center", va="center", fontsize=7.6, color=COLORS["ink"], zorder=3)
    for a, b, lab in _EDGES:
        p0, p1 = _anchor(a, b), _anchor(b, a)
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=10, lw=1, color=COLORS["ink2"], zorder=1))
        if lab:
            vertical = abs(p0[0] - p1[0]) < 0.01
            ax.text((p0[0] + p1[0]) / 2 + (0.08 if vertical else 0), (p0[1] + p1[1]) / 2 + (0 if vertical else 0.08), lab,
                    fontsize=6.8, color=COLORS["ink2"], ha="left" if vertical else "center",
                    va="center" if vertical else "bottom", zorder=4)
    # Lazo de control: alarma de monitoreo → re-entrenamiento / rollback
    x0, y0 = _BOXES["cw"][0] + _W / 2, _BOXES["cw"][1] + _H
    x1, y1 = _BOXES["sf"][0] + _W / 2, _BOXES["sf"][1] + _H
    top = 6.98
    ax.plot([x0, x0, x1], [y0, top, top], color=COLORS["sub"], lw=1, ls="--", zorder=1)
    ax.add_patch(FancyArrowPatch((x1, top), (x1, 6.72), arrowstyle="-|>", mutation_scale=10, lw=1, color=COLORS["sub"], zorder=1))
    ax.text((x0 + x1) / 2, top + 0.05, "alarma WAPE / drift → re-entrena o rollback", fontsize=7, color=COLORS["sub"], ha="center", va="bottom")
    ax.text(0.1, 0.32, "Resiliencia: si no hay pronóstico corregido vigente, DynamoDB sirve el fallback (sesgo móvil o base). "
            "El servicio nunca depende del modelo en la ruta crítica.", fontsize=7.5, color=COLORS["ink2"])
    ax.set_title("Arquitectura AWS propuesta — pronóstico corregido + gate + recomendación de fecha", loc="left")
    return fig
