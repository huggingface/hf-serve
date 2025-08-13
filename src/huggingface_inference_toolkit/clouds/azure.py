from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, RedirectResponse

router = APIRouter()


@router.get("/swagger.json")
async def swagger_json(request: Request) -> JSONResponse:
    return request.app.openapi()


@router.post("/score")
async def score(_: Request):
    return RedirectResponse(url="/predict", status_code=307)
