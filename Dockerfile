FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock README.md ./
COPY config/ config/
COPY src/ src/

RUN uv sync --frozen --no-dev

EXPOSE 8000

CMD [".venv/bin/python", "-m", "lgb", "serve"]
