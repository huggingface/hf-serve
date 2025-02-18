from fastapi import APIRouter
from fastapi.responses import Response
from prometheus_client import generate_latest, REGISTRY, CONTENT_TYPE_LATEST

router = APIRouter()


@router.get("/metrics")
async def metrics() -> Response:
    return Response(generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST, status_code=200)
