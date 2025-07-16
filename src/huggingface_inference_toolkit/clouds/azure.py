from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

router = APIRouter()


@router.get("/swagger.json")
async def swagger_json(request: Request) -> JSONResponse:
    return request.app.openapi()
