import time
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from huggingface_inference_toolkit.logging import logger


class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()

        if request.method == "POST":
            ct = request.headers.get("content-type", "")
            if "application/json" in ct:
                body = await request.body()
                logger.info(
                    f"Request: type=json path={request.url.path} method={request.method} - Body={body.decode()}"
                )
                # Reset body for downstream
                request._receive = lambda: {"type": "http.request", "body": body, "more_body": False}
            elif "multipart/form-data" in ct:
                logger.info(f"Request: type=multipart path={request.url.path} method={request.method}")

        response = await call_next(request)
        process_time = time.time() - start_time
        logger.info(
            f"Request: {request.method} {request.url.path} - Status: {response.status_code} - Duration: {process_time:.4f}s"
        )
        return response
