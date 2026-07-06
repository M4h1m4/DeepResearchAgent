# ── Stage 1: install dependencies ──────────────────────────────────────────
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim AS builder

WORKDIR /app

# Cache dependency install separately from app code
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Now copy app and install project itself
COPY . .
RUN uv sync --frozen --no-dev

# ── Stage 2: lean runtime image ─────────────────────────────────────────────
FROM python:3.11-slim-bookworm AS runtime

WORKDIR /app

# Copy the virtualenv built in stage 1
COPY --from=builder /app/.venv /app/.venv
# Copy application source
COPY --from=builder /app /app

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app"
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# HuggingFace/datasets cache lives under a writable HOME (HF Spaces runs as UID 1000)
ENV HOME="/home/appuser"
ENV HF_HOME="/home/appuser/.cache/huggingface"

# HF Spaces (and good practice everywhere) run the container as a non-root UID 1000.
# Create that user and give it ownership of /app so SQLite + uploaded docs are writable.
RUN useradd -m -u 1000 appuser \
    && mkdir -p data/documents \
    && chown -R appuser:appuser /app /home/appuser

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/v1/health')"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
