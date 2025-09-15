from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

router = APIRouter()


# https://github.com/huggingface/text-embeddings-inference/blob/ebb63dfa7121705f1999a06d8e222581a5221c00/router/src/http/server.rs#L1834
@router.get("/api-doc/openapi.json")
async def openapi_json(request: Request) -> JSONResponse:
    return request.app.openapi()
