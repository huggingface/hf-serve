import logging
import time
from typing import Callable, List, Optional

from fastapi import Request, Response
from prometheus_client import Counter, Histogram, Gauge
from starlette.types import ASGIApp
from starlette.middleware.base import BaseHTTPMiddleware

REQUEST_COUNT = Counter("http_request_total", "Total HTTP Requests", ["method", "status", "path"])
REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP Request Duration",
    ["method", "status", "path"],
)
REQUEST_IN_PROGRESS = Gauge("http_requests_in_progress", "HTTP Requests in progress", ["method", "path"])

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PrometheusMiddleware(BaseHTTPMiddleware):
    def __init__(
        self,
        app: ASGIApp,
        *,
        exclude_paths: Optional[List[str]] = None,
    ):
        super().__init__(app)

        self.exclude_paths = exclude_paths or []

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        method = request.method
        path = request.url.path

        if path in self.exclude_paths:
            return await call_next(request)

        REQUEST_IN_PROGRESS.labels(method=method, path=path).inc()
        start_time = time.time()

        try:
            response = await call_next(request)
            status = response.status_code
        except Exception as e:
            status = 500
            raise e from None
        finally:
            duration = time.time() - start_time
            REQUEST_COUNT.labels(method=method, status=status, path=path).inc()  # type: ignore
            REQUEST_LATENCY.labels(method=method, status=status, path=path).observe(  # type: ignore
                duration
            )
            REQUEST_IN_PROGRESS.labels(method=method, path=path).dec()

            logger.info(
                f"Request: {method} {path} - Status: {status} - Duration: {duration:.4f}s"  # type: ignore
            )

        return response
