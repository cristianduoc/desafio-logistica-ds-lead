# Imagen reproducible: instala dependencias, corre tests y ejecuta el notebook completo.
#   docker build -t desafio-logistica .
#   docker run --rm -v "$(pwd)/outputs:/app/outputs" desafio-logistica
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

WORKDIR /app

COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt pytest

COPY src/ src/
COPY tests/ tests/
COPY docs/ docs/
COPY data_package_desafio_logistica/ data_package_desafio_logistica/
COPY desafio_logistica.ipynb ./
RUN pip install --no-cache-dir -e .

# Por defecto: tests + ejecución del notebook (los resultados quedan en outputs/ y en el .ipynb ejecutado)
CMD ["sh", "-c", "pytest -q && jupyter nbconvert --to notebook --execute desafio_logistica.ipynb --output outputs/desafio_logistica_ejecutado.ipynb --ExecutePreprocessor.timeout=900"]
