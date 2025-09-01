from fastapi import APIRouter, Body, Form, HTTPException, Request
from pydantic import ValidationError

from huggingface_inference_toolkit.logging import logger
from huggingface_inference_toolkit.openai.schemas.images_generations import (
    ImagesGenerationsInput,
    ImagesGenerationsOutput,
)
from huggingface_inference_toolkit.openai.tasks.images_generations import ImagesGenerations


def router(predictor: ImagesGenerations) -> APIRouter:
    router = APIRouter()

    @router.post("/v1/images/generations", response_model=ImagesGenerationsOutput)
    async def images_generations(
        request: Request,
        payload: ImagesGenerationsInput = Body(...),
    ) -> ImagesGenerationsOutput:
        request_id = getattr(request.state, "request_id", None)

        try:
            logger.info(f"[{request_id}] Received images generations request with: {payload.model_dump()}")

            if payload.stream is True:
                logger.error(
                    f"[{request_id}] Request to images generations failed as `{payload.stream=}` is not supported."
                )
                raise ValueError("`stream=True` is not supported for `/v1/images/generations`.")

            return predictor(payload=payload, request_id=request_id)
        except (ValueError, ValidationError) as e:
            logger.error(f"[{request_id}] Failed validating I/O with: {str(e)}")
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"[{request_id}] Failed running inference with: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    return router
