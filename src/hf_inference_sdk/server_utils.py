from fastapi import FastAPI

from hf_inference_sdk.logging import logger


def log_available_routes(app: FastAPI) -> None:
    logger.info("Available API routes:")

    route_groups = {
        "predict": ["/", "/predict", "/score"],
        "docs": ["/docs", "/docs/oauth2-redirect"],
        "openapi": ["/openapi.json", "/swagger.json", "/api-doc/openapi.json"],
    }

    logged = set()
    grouped_routes = {}

    for route in app.routes:
        if hasattr(route, "methods") and hasattr(route, "path"):  # type: ignore
            path = route.path  # type: ignore
            methods = [m for m in sorted(route.methods) if m != "HEAD"]  # type: ignore

            for method in methods:
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
