# Build context: repository root (see docker-compose.yml)
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.10.0 /uv /uvx /bin/
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT=/srv/.venv \
    PYTHONUNBUFFERED=1 MODELS_DIR=/models SHARED_SRC=/srv

WORKDIR /srv/backend
COPY backend/pyproject.toml backend/uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Corruption operators are shared with training (NumPy only) - single source of truth.
COPY src/__init__.py /srv/src/__init__.py
COPY src/data/__init__.py src/data/corruptions.py /srv/src/data/
COPY backend/app ./app
COPY backend/samples ./samples

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=5 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/api/health').status==200 else 1)"
CMD ["/srv/.venv/bin/uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
