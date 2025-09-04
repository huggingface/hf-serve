import time
from typing import Callable, List, Optional

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from hf_serve.logging import logger


class LoggingMiddleware(BaseHTTPMiddleware):
    def __init__(
        self,
        app: ASGIApp,
        *,
        exclude_paths: Optional[List[str]] = None,
        inference_paths: Optional[List[str]] = None,
    ) -> None:
        super().__init__(app)

        self.exclude_paths = exclude_paths or []
        self.inference_paths = inference_paths or []

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        method = request.method
        path = request.url.path

        if path in self.exclude_paths:
            return await call_next(request)

        start_time = time.perf_counter()

        if path in self.inference_paths:
            request_id = getattr(request.state, "request_id", None)
            logger.info(f"[{request_id}] Request: {method} {path}")

            response = await call_next(request)

            process_time = (time.perf_counter() - start_time) * 1000
            response.headers["X-Process-Time"] = f"{process_time:.2f}ms"
            request_id = getattr(request.state, "request_id", None)
            logger.info(
                f"[{request_id}] Response: Status: {response.status_code} - Duration: {process_time:.2f}ms"
            )
        else:
            logger.info(f"Request: {method} {path}")
            response = await call_next(request)

            process_time = (time.perf_counter() - start_time) * 1000
            response.headers["X-Process-Time"] = f"{process_time:.2f}ms"
            logger.info(f"Response: Status: {response.status_code} - Duration: {process_time:.2f}ms")

        return response
