import re
from collections.abc import Iterable
from typing import Any

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send


_ROUTE_ATTRIBUTE = re.compile(r"(?:^|[,;])\s*route\s*=\s*([^,;\s]+)", re.IGNORECASE)


def post_paths(routes: Iterable[Any]) -> set[str]:
    """Return the paths handled by a POST route in a FastAPI application."""
    from hf_serve.server_utils import iter_http_routes

    return {path for path, methods in iter_http_routes(routes) if "POST" in methods and path != "/invocations"}


def _requested_route(headers: Iterable[tuple[bytes, bytes]]) -> str | None:
    for name, value in headers:
        if name.lower() != b"x-amzn-sagemaker-custom-attributes":
            continue

        match = _ROUTE_ATTRIBUTE.search(value.decode("latin-1"))
        if match is None:
            return None

        route = match.group(1)
        return route if route.startswith("/") else f"/{route}"

    return None


class SageMakerRoutingMiddleware:
    """Internally route SageMaker's fixed endpoints to hf-serve endpoints."""

    def __init__(self, app: ASGIApp, invocation_paths: Iterable[str]) -> None:
        self.app = app
        self.invocation_paths = set(invocation_paths)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        target_path: str | None = None
        if scope["path"] == "/ping":
            target_path = "/health"
        elif scope["path"] == "/invocations":
            target_path = _requested_route(scope.get("headers", [])) or "/predict"
            if target_path not in self.invocation_paths:
                available_routes = ", ".join(sorted(self.invocation_paths)) or "none"
                response = JSONResponse(
                    {
                        "detail": (
                            f"Unsupported SageMaker invocation route: {target_path}. "
                            f"Available routes: {available_routes}"
                        )
                    },
                    status_code=400,
                )
                await response(scope, receive, send)
                return

        if target_path is not None:
            scope = {**scope, "path": target_path, "raw_path": target_path.encode("ascii")}

        await self.app(scope, receive, send)
