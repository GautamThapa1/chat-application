# syntax=docker/dockerfile:1
#
# One image runs BOTH Python processes (the gRPC "service" and the FastAPI
# "app"); docker compose picks the command. Multi-stage: the builder installs
# dependencies, the runtime image only receives the finished virtualenv.

# ---------- Stage 1: install dependencies ----------
FROM python:3.12-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN pip install --no-cache-dir uv==0.13.9

WORKDIR /app

# Copy ONLY the dependency files first: Docker caches this layer, so code
# changes do not reinstall every package.
COPY pyproject.toml uv.lock ./
# --frozen: install exactly what uv.lock says (fails if it is out of date).
RUN uv sync --frozen --no-dev --no-install-project

# ---------- Stage 2: small runtime image ----------
FROM python:3.12-slim AS runtime

# Run as an unprivileged user. /data is where the database volume is mounted.
RUN useradd --create-home --uid 10001 app \
    && mkdir /data \
    && chown app:app /data

WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY backend ./backend
COPY shared_common ./shared_common
COPY frontend ./frontend

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DB_PATH=/data/app.db

USER app
EXPOSE 8000

# Default command = the web app. IMPORTANT: keep ONE worker. Running replies
# are tracked in this process's memory (see backend/orchestrator/websocket.py),
# so several workers would not see each other's generations.
CMD ["uvicorn", "backend.orchestrator.main:app", "--host", "0.0.0.0", "--port", "8000"]
