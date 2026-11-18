FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml ./
COPY src ./src
COPY scripts ./scripts
COPY evals ./evals

RUN pip install --upgrade pip && pip install --no-cache-dir ".[ui]" \
    && groupadd --system appgroup \
    && useradd --system --gid appgroup --create-home appuser \
    && mkdir -p /app/data \
    && chown -R appuser:appgroup /app

USER appuser

EXPOSE 8302 8502

CMD ["sh", "-c", "python -m scripts.seed_demo_db && uvicorn orderops.api.main:app --host 0.0.0.0 --port 8302"]
