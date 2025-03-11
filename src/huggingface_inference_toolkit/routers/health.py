from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter()


@router.get("/health")
def health() -> JSONResponse:
    return JSONResponse(content={"status": "ok"}, status_code=200)


# NOTE: in case returning a JSON instead of a plain text response is not compliant with what the Inference Endpoints
# expect, then use the implementation below instead
# from fastapi.responses import PlainTextResponse
#
# @router.get("/health", response_class=PlainTextResponse)
# def health() -> PlainTextResponse:
#     return "ok"
