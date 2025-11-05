import os

from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter()


@router.get(os.getenv("AIP_HEALTH_ROUTE", "/health"))
def health() -> JSONResponse:
    return JSONResponse(content={"status": "ok"}, status_code=200)
