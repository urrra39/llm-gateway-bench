FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy

WORKDIR /app

# libgomp1 for the torch CPU wheel pulled in by the embeddings extra.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists \
    && pip install --no-cache-dir uv

COPY pyproject.toml uv.lock README.md Makefile ./
COPY config/ config/
COPY src/ src/
# Committed workload sample so `make pilot` works from a clean clone without
# any download. Embedding weights (~90 MB) are NOT baked in; they download to
# data/models on first embed (needs network).
COPY data/workload/ data/workload/

RUN uv sync --frozen --no-dev --extra embeddings

EXPOSE 8000

CMD [".venv/bin/python", "-m", "lgb", "serve"]
