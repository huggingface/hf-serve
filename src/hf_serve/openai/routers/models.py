from fastapi import APIRouter
from fastapi.responses import JSONResponse


def router(model_id: str, timestamp: int) -> APIRouter:
    router = APIRouter()

    @router.get("/v1/models")
    def models() -> JSONResponse:
        return JSONResponse(
            {
                "object": "list",
                "data": [
                    {
                        "id": model_id,
                        "object": "model",
                        "created": timestamp,
                        "owned_by": model_id.split("/")[0]
                        if model_id is not None and model_id.__contains__("/")
                        else None,
                    }
                ],
            }
        )

    return router
