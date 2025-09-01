from typing import TYPE_CHECKING, Union

from fastapi import APIRouter
from fastapi.responses import JSONResponse

if TYPE_CHECKING:
    from huggingface_inference_toolkit.openai.tasks.chat_completions import ChatCompletions
    from huggingface_inference_toolkit.openai.tasks.images import Images


def router(predictor: Union["ChatCompletions", "Images"], timestamp: int) -> APIRouter:
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
                        "created": timestamp,
                        "owned_by": predictor.model_id.split("/")[0]
                        if predictor.model_id is not None and predictor.model_id.__contains__("/")
                        else None,
                    }
                ],
            }
        )

    return router
