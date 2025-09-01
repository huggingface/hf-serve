from typing import Callable, List, Optional
from uuid import uuid4

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp


class RequestIdMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp, *, exclude_paths: Optional[List[str]] = None) -> None:
        super().__init__(app)

        self.exclude_paths = exclude_paths or []

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if request.url.path in self.exclude_paths:
            return await call_next(request)

        # Check for user-provided request ID in X-Request-Id header
        request_id = request.headers.get("X-Request-Id")
        if not request_id:
            request_id = uuid4().hex

        # Store request ID in request.state for request-scoped access
        request.state.request_id = request_id

        response = await call_next(request)

        response.headers["X-Request-Id"] = request_id

        return response
