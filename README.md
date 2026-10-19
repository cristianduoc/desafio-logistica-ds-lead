# Desafío Técnico — Data Scientist Lead · Optimización Logística

Diagnóstico del pronóstico de demanda en 6 CDs, análisis del plan de entrega, modelo complementario de corrección,
gate de historia mínima (Cold Start) y arquitectura AWS.

## Resultados principales

| Tema | Resultado |
|---|---|
| Pronóstico base | WAPE 20.8%. SUR (+32%), ESTE (+25%) y NORTE (+8%) **subestiman**; OESTE, PACÍFICO y CENTRO (≈ −21%) **sobreestiman**. El sesgo no cambia de signo en ninguna semana |
| Horizonte | WAPE 19.8% (D+1) → 22.5% (D+4): degradación moderada; el problema dominante es el sesgo de nivel |
| Plan de entrega | Bias de volumen +10.6% (subestima); 17.5% de las compras ocurren fuera de plan, **60% de ellas terminan en martes** |
| Modelo complementario | Ridge con features log-relativas, un modelo por horizonte: **WAPE 20.7% → 6.6%** en el backtest rolling-origin (6 folds); bias global +0.3% |
| Gate 45 días | Con < 21 días el modelo es peor que el fallback; con 45 días alcanza su mejor WAPE (6.1%) y su menor varianza |
| Impacto (reserva con D+2) | −79% de unidades en flete de emergencia; nivel de servicio 86% → 97% |

## Estructura

```
desafio_logistica.ipynb        Notebook ejecutable: genera todas las tablas y figuras (outputs/)
src/logistica/
  config.py                    Rutas y constantes (umbral del gate, CDs, horizontes)
  data.py                      Carga, validación de esquema y uniones base
  metrics.py                   WAPE y Bias (definiciones del enunciado)
  features.py                  Features sin fuga de información (rolling, picos, sesgo móvil, log-relativas)
  models.py                    Base, corrección por sesgo móvil y MLCorrector (Ridge/HGB, global o por horizonte)
  backtest.py                  Backtest rolling-origin y sensibilidad a la historia disponible
  gate.py                      HistoryGate (historia continua) + ResilientForecaster (fallback sin excepciones)
  plan.py                      Tarea 2: bias del plan, adhesión, concentración, simulación de aplanamiento
  impact.py                    Traducción a flota: fletes de emergencia y capacidad ociosa
  architecture.py, viz.py      Diagrama AWS y estilo gráfico
tests/                         Tests de métricas, anti-fuga, gate y resiliencia
docs/arquitectura_aws.drawio   Diagrama AWS editable (draw.io, íconos oficiales) + .png/.svg exportados
docs/arquitectura.md           Arquitectura AWS (Mermaid + descripción de componentes)
outputs/                       Tablas (.csv) y figuras (.png) generadas por el notebook
deck/deck_desafio_logistica.pdf Presentación ejecutiva (9 diapositivas); fuente en deck_desafio_logistica.html
deck/figuras/                  Gráficos del deck (notebook ejecutado con LOGISTICA_THEME=andina)
```

## Cómo ejecutar

**Local (Python ≥ 3.11)**
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    ·   Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt pytest
pytest -q
jupyter nbconvert --to notebook --execute --inplace desafio_logistica.ipynb   # o abrirlo en Jupyter y "Run All"
```

**Docker**
```bash
docker build -t desafio-logistica .
docker run --rm -v "$(pwd)/outputs:/app/outputs" desafio-logistica
```
La ejecución completa toma ~1–2 minutos. Los CSV de entrada se leen desde `data_package_desafio_logistica/`.

## Regenerar el deck
```bash
LOGISTICA_THEME=andina LOGISTICA_OUTPUT_DIR=deck/figuras jupyter nbconvert --to notebook --execute desafio_logistica.ipynb --output-dir /tmp
# luego imprimir deck/deck_desafio_logistica.html a PDF (1920×1080, sin márgenes) desde Chrome/Edge
```

## Decisiones y supuestos
- **Disponibilidad de información:** un pronóstico emitido en E solo usa la demanda real hasta E-1. Esto se verifica con un test.
- **Target del modelo:** `log(real / pronóstico_base)`. El modelo aprende la *corrección* del base, sin usar el identificador del CD, para que sea transferible a CDs nuevos.
- **Selección de modelo:** WAPE en el backtest rolling-origin, más un bootstrap por semanas de la diferencia contra el fallback y contra la versión v1.
- **Impacto:** se supone que la flota se reserva con el pronóstico D+2, acotado a la capacidad. El camión equivalente (300 u) es un supuesto ilustrativo.

## Fuera de alcance
Estacionalidad mensual y feriados (solo hay 60 días de historia), intervalos de predicción para reservar por cuantil, corrección por canal, código de infraestructura y tuning exhaustivo de hiperparámetros.
