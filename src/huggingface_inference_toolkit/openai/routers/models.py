from fastapi import APIRouter
from fastapi.responses import JSONResponse

from huggingface_inference_toolkit.tasks.predictor import Predictor


def router(predictor: Predictor) -> APIRouter:
    router = APIRouter()

    @router.get("/v1/models")
    def models() -> JSONResponse:
        return JSONResponse(
            {
                "object": "list",
                "data": [
                    {
                        "id": predictor.model_id,
                        "object": "model",
                        "created": "",
                        "owned_by": predictor.model_id.split("/")[0]
                        if predictor.model_id is not None and predictor.model_id.__contains__("/")
                        else None,
                    }
                ],
            }
        )

    return router
