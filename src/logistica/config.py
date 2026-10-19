"""Configuración central del proyecto (rutas y constantes de negocio)."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = PROJECT_ROOT / "data_package_desafio_logistica"
OUTPUT_DIR: Path = Path(os.environ.get("LOGISTICA_OUTPUT_DIR", PROJECT_ROOT / "outputs"))

FILES: dict[str, str] = {
    "pedidos": "pedidos_historicos_genericos.csv",
    "pronostico": "pronostico_demanda_generico.csv",
    "capacidad": "capacidad_logistica_generica.csv",
    "plan": "plan_entrega_clientes_generico.csv",
}

CENTROS: list[str] = ["CD_NORTE", "CD_SUR", "CD_CENTRO", "CD_ESTE", "CD_OESTE", "CD_PACIFICO"]
HORIZONTES: list[str] = ["D+1", "D+2", "D+3", "D+4"]
DIAS_SEMANA: list[str] = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]

MIN_HISTORY_DAYS: int = 45
RANDOM_STATE: int = 42
