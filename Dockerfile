# ─────────────────────────────────────────────
# UrbanFlow API — Dockerfile
#
# Multi-stage build:
#   Stage 1: Install Python dependencies
#   Stage 2: Copy app code + trained models
#
# Build:   docker build -t urbanflow-api .
# Run:     docker run -p 8000:8000 urbanflow-api
# ─────────────────────────────────────────────

FROM python:3.11-slim AS base

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# ── Stage 1: Dependencies ────────────────────
FROM base AS deps

# Install system-level build dependencies for scientific packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgeos-dev \
    libproj-dev \
    gdal-bin \
    libgdal-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --default-timeout=1000 --retries 10 --prefer-binary --no-cache-dir -r requirements.txt

# ── Stage 2: Application ─────────────────────
FROM base AS app

# Install runtime-only system libs (GEOS for shapely, GDAL for geopandas)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgeos-dev \
    libproj-dev \
    gdal-bin \
    libgdal-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from deps stage
COPY --from=deps /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=deps /usr/local/bin /usr/local/bin

# Copy application source code
COPY config.py .
COPY api/ api/
COPY db/ db/
COPY ml/ ml/
COPY data/ data/
COPY alembic.ini .

# Create output directories (models must be committed via Git LFS to be available)
RUN mkdir -p outputs/models outputs/shap outputs/reports

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Run with uvicorn
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
