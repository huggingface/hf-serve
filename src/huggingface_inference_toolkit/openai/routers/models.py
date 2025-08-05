from fastapi import APIRouter
from fastapi.responses import JSONResponse

from huggingface_inference_toolkit.openai.tasks.chat_completions import ChatCompletions


def router(predictor: ChatCompletions, timestamp: int) -> APIRouter:
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
