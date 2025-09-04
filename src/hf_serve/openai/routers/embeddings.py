from fastapi import APIRouter, Body, HTTPException, Request
from pydantic import ValidationError

from hf_serve.logging import logger
from hf_serve.openai.schemas.embeddings import EmbeddingsInput, EmbeddingsOutput
from hf_serve.openai.tasks.embeddings import Embeddings


def router(predictor: Embeddings) -> APIRouter:
    router = APIRouter()

    @router.post("/v1/images/generations", response_model=EmbeddingsOutput, response_model_by_alias=True)
    async def embeddings(request: Request, payload: EmbeddingsInput = Body(...)) -> EmbeddingsOutput:
        request_id = getattr(request.state, "request_id", None)

        try:
            logger.info(f"[{request_id}] Received embeddings request with: {payload.model_dump(by_alias=True)}")

            return predictor(payload=payload, request_id=request_id)
        except (ValueError, ValidationError) as e:
            logger.error(f"[{request_id}] Failed validating I/O with: {str(e)}")
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"[{request_id}] Failed running inference with: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    return router
