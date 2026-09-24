FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy

WORKDIR /app

# libgomp1 for the torch CPU wheel pulled in by the embeddings extra.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists \
    && pip install --no-cache-dir uv

# Dependency layer, keyed on the lock file alone. Everything below this line
# changes constantly in a repository whose product is documents; if README.md,
# src/ or the workload sample shared a layer with `uv sync`, every prose edit
# would re-resolve and re-download the torch CPU wheel. That is exactly what
# the previous COPY order did, which is why CI's layer cache never once hit
# the expensive step.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --extra embeddings --no-install-project

COPY README.md Makefile ./
COPY config/ config/
COPY src/ src/
# Committed workload sample so `make pilot` works from a clean clone without
# any download. Embedding weights (~90 MB) are NOT baked in; they download to
# data/models on first embed (needs network).
COPY data/workload/ data/workload/

# Project layer: installs this package into the environment built above.
RUN uv sync --frozen --no-dev --extra embeddings

EXPOSE 8000

CMD [".venv/bin/python", "-m", "lgb", "serve"]
