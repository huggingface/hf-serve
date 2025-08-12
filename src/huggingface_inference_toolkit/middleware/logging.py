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
            match ct:
                case "application/json":
                    body = await request.body()
                    logger.info(f"Request: {request.method} {request.url.path} - Body: {body.decode('utf-8')}")
                case "multipart/form-data":
                    form = await request.form()
                    file = form.get("file")
                    if file:
                        logger.info(f"Request: {request.method} {request.url.path} - File: {file.filename}")
                    else:
                        logger.info(f"Request: {request.method} {request.url.path} - No file in form data")


        response = await call_next(request)
        process_time = time.time() - start_time
        logger.info(
            f"Request: {request.method} {request.url.path} - Status: {response.status_code} - Duration: {process_time:.4f}s"
        )
        return response
