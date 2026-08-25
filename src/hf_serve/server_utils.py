from collections.abc import Iterable, Iterator
from typing import Any

from fastapi import FastAPI

from hf_serve.logging import logger


def _join_route_path(prefix: str, path: str) -> str:
    if not prefix:
        return path
    if path == "/":
        return f"{prefix.rstrip('/')}/"
    return f"{prefix.rstrip('/')}/{path.lstrip('/')}"


# NOTE: FastAPI may store routes added through `include_router()` in nested wrappers
# without a path or methods at the top level, so iterating over `app.routes` alone misses them.
def iter_http_routes(routes: Iterable[Any], prefix: str = "") -> Iterator[tuple[str, set[str]]]:
    """Yield concrete HTTP paths, including routes nested by FastAPI's include_router."""
    for route in routes:
        path = getattr(route, "path", None)
        methods = getattr(route, "methods", None)
        if isinstance(path, str) and methods:
            yield _join_route_path(prefix, path), {str(method).upper() for method in methods}

        original_router = getattr(route, "original_router", None)
        if original_router is not None:
            original_router_routes = getattr(original_router, "routes", ())
            include_context = getattr(route, "include_context", None)
            include_prefix = getattr(include_context, "prefix", "")
            original_router_prefix = _join_route_path(prefix, include_prefix)
            yield from iter_http_routes(original_router_routes, original_router_prefix)

        mounted_routes = getattr(route, "routes", None)
        if mounted_routes:
            mount_prefix = _join_route_path(prefix, path) if isinstance(path, str) else prefix
            yield from iter_http_routes(mounted_routes, mount_prefix)


def log_available_routes(app: FastAPI) -> None:
    logger.info("Available API routes:")

    route_groups = {
        "predict": ["/", "/predict", "/score"],
        "docs": ["/docs", "/docs/oauth2-redirect"],
        "openapi": ["/openapi.json", "/swagger.json", "/api-doc/openapi.json"],
    }

    logged = set()
    grouped_routes = {}

    for path, methods in iter_http_routes(app.routes):
        for method in sorted(method for method in methods if method != "HEAD"):
            group_found = False
            for group_name, group_paths in route_groups.items():
                if path in group_paths:
                    if group_name not in grouped_routes:
                        grouped_routes[group_name] = {"method": method, "paths": []}
                    if path not in grouped_routes[group_name]["paths"]:
                        grouped_routes[group_name]["paths"].append(path)
                    group_found = True
                    break

            if not group_found and path not in logged:
                logger.info(f"[{method:<4}] {path}")
                logged.add(path)

    for group_name, group_data in grouped_routes.items():
        if len(group_data["paths"]) > 1:
            paths_str = ", ".join(group_data["paths"])
            logger.info(f"[{group_data['method']:<4}] {paths_str}")
        else:
            logger.info(f"[{group_data['method']:<4}] {group_data['paths'][0]}")
        logged.update(group_data["paths"])
