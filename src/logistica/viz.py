"""Estilo gráfico común (notebook y deck).

El tema se elige con la variable de entorno `LOGISTICA_THEME`:
  * "default": paleta neutra azul/naranja (la del notebook).
  * "andina": paleta corporativa aproximada (rojo #F40009, negro, grises) para el deck.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt

from .config import OUTPUT_DIR

_THEMES: dict[str, dict[str, str]] = {
    "default": {
        "base": "#898781",       # pronóstico base: neutro
        "modelo": "#2a78d6",     # modelo corregido: azul
        "fallback": "#eb6834",   # fallback: naranja
        "alt": "#1baf7a",        # tercera serie
        "light": "#86b6ef",      # variantes secundarias del modelo
        "highlight": "#cde2fb",  # resaltado de celdas en tablas
        "sub": "#e34948",        # subestimación (bias > 0)
        "sobre": "#2a78d6",      # sobreestimación (bias < 0)
        "sub_name": "rojo",
        "sobre_name": "azul",
        "capacidad": "#0b0b0b",
        "ink": "#0b0b0b",
        "ink2": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "surface": "#fcfcfb",
        "cmap_seq": "Blues",
        "cmap_seq2": "Reds",
        "cmap_div": "RdBu_r",
        "font": "Segoe UI",
    },
    "andina": {
        "base": "#A6A6A6",
        "modelo": "#F40009",
        "fallback": "#1E1E1E",
        "alt": "#7A7A7A",
        "light": "#F7A1A4",
        "highlight": "#FDE0E1",
        "sub": "#F40009",
        "sobre": "#1E1E1E",
        "sub_name": "rojo",
        "sobre_name": "negro",
        "capacidad": "#1E1E1E",
        "ink": "#1E1E1E",
        "ink2": "#4D4D4D",
        "muted": "#7A7A7A",
        "grid": "#E6E6E6",
        "surface": "#FFFFFF",
        "cmap_seq": "Greys",
        "cmap_seq2": "Reds",
        "cmap_div": "RdGy_r",
        "font": "Arial",
    },
}
_SERIES: dict[str, list[str]] = {
    "default": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"],
    "andina": ["#F40009", "#1E1E1E", "#9E9E9E", "#A30006", "#F7A1A4", "#5C5C5C"],
}
_RAMP: dict[str, list[str]] = {  # rampa ordinal (p.ej. horizontes D+1..D+4)
    "default": ["#b7d3f6", "#6da7ec", "#2a78d6", "#184f95"],
    "andina": ["#F7A1A4", "#F2545A", "#F40009", "#8F0005"],
}

THEME: str = os.environ.get("LOGISTICA_THEME", "default")
if THEME not in _THEMES:
    THEME = "default"
COLORS: dict[str, str] = _THEMES[THEME]
SERIES: list[str] = _SERIES[THEME]
RAMP: list[str] = _RAMP[THEME]


def set_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": COLORS["surface"],
        "axes.facecolor": COLORS["surface"],
        "axes.edgecolor": "#c3c2b7",
        "axes.labelcolor": COLORS["ink2"],
        "axes.titlecolor": COLORS["ink"],
        "axes.titleweight": "bold",
        "axes.titlesize": 12,
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": COLORS["grid"],
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "xtick.color": COLORS["muted"],
        "ytick.color": COLORS["muted"],
        "xtick.labelcolor": COLORS["ink2"],
        "ytick.labelcolor": COLORS["ink2"],
        "lines.linewidth": 2,
        "legend.frameon": False,
        "font.family": [COLORS["font"], "DejaVu Sans", "sans-serif"],
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    })


def save(fig: plt.Figure, name: str, out_dir: Path = OUTPUT_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path)
    return path
