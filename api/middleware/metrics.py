"""
UrbanFlow -- Prometheus Metrics Middleware

Tracks per-endpoint request counts, latencies, and HTTP status codes.
Metrics are exposed at GET /metrics in Prometheus text format.

Counters:
    urbanflow_requests_total         -- total requests by method/path/status
    urbanflow_request_errors_total   -- 4xx/5xx errors

Histograms:
    urbanflow_request_duration_seconds -- latency by method/path

Gauges:
    urbanflow_requests_in_progress   -- currently-inflight requests
"""

import time

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
)

# ─────────────────────────────────────────────
# Prometheus metric definitions
# ─────────────────────────────────────────────

REQUEST_COUNT = Counter(
    "urbanflow_requests_total",
    "Total HTTP requests",
    ["method", "path", "status"],
)

REQUEST_ERRORS = Counter(
    "urbanflow_request_errors_total",
    "Total HTTP error responses (4xx + 5xx)",
    ["method", "path", "status"],
)

REQUEST_DURATION = Histogram(
    "urbanflow_request_duration_seconds",
    "Request latency in seconds",
    ["method", "path"],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)

REQUESTS_IN_PROGRESS = Gauge(
    "urbanflow_requests_in_progress",
    "Number of requests currently being processed",
    ["method"],
)


# ─────────────────────────────────────────────
# Helper — normalise URL paths to avoid high-cardinality labels
# ─────────────────────────────────────────────

def _normalise_path(path: str) -> str:
    """
    Collapse dynamic path segments to keep Prometheus label cardinality bounded.

    Examples:
        /api/v1/explain/zone/161  →  /api/v1/explain/zone/{zone_id}
        /api/v1/pricing/history/42 →  /api/v1/pricing/history/{zone_id}
    """
    parts = path.rstrip("/").split("/")
    normalised = []
    for i, part in enumerate(parts):
        if part.isdigit():
            # Replace integer path segments with a placeholder
            if i > 0 and normalised:
                prev = normalised[-1]
                if prev in ("zone", "history"):
                    normalised.append("{zone_id}")
                    continue
            normalised.append("{id}")
        else:
            normalised.append(part)
    return "/".join(normalised) or "/"


# ─────────────────────────────────────────────
# Starlette Middleware
# ─────────────────────────────────────────────

class MetricsMiddleware(BaseHTTPMiddleware):
    """ASGI middleware that records Prometheus metrics for every request."""

    async def dispatch(self, request: Request, call_next) -> Response:
        method = request.method
        path = _normalise_path(request.url.path)

        # Skip metrics endpoint itself to avoid recursion noise
        if request.url.path == "/metrics":
            return await call_next(request)

        REQUESTS_IN_PROGRESS.labels(method=method).inc()
        start = time.perf_counter()

        try:
            response = await call_next(request)
        except Exception:
            REQUEST_ERRORS.labels(method=method, path=path, status="500").inc()
            REQUEST_COUNT.labels(method=method, path=path, status="500").inc()
            REQUESTS_IN_PROGRESS.labels(method=method).dec()
            raise

        duration = time.perf_counter() - start
        status = str(response.status_code)

        REQUEST_COUNT.labels(method=method, path=path, status=status).inc()
        REQUEST_DURATION.labels(method=method, path=path).observe(duration)

        if response.status_code >= 400:
            REQUEST_ERRORS.labels(method=method, path=path, status=status).inc()

        REQUESTS_IN_PROGRESS.labels(method=method).dec()
        return response


def get_metrics_text() -> str:
    """Generate the Prometheus exposition text."""
    return generate_latest().decode("utf-8")
