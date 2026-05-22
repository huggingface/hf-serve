from typing import Type, Union

from fastapi import APIRouter, Body, Request
from fastapi.exceptions import HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, ValidationError

from hf_inference_sdk import idle
from hf_inference_sdk.logging import logger
from hf_inference_sdk.tasks.predictor import Predictor


def router(
    predictor: Predictor,
    input_schema: Union[Type[BaseModel], Type[Union[BaseModel, ...]]],  # type: ignore
    output_schema: Union[Type[BaseModel], Type[Union[BaseModel, ...]]],  # type: ignore
) -> APIRouter:
    router = APIRouter()

    logger.info("Warming up the pipeline before starting the API...")
    predictor.warmup()
    logger.info("Warmup succeeded, exposing API routes...")

    # NOTE: for Inference Endpoints we also need to route to / for the /predict route, as
    # that's the endpoint being hit within the Inference API widgets
    @router.post("/", response_model=output_schema)
    @router.post("/predict", response_model=output_schema)
    async def predict(request: Request, payload: input_schema = Body(...)) -> output_schema:  # type: ignore
        request_id = getattr(request.state, "request_id", None)

        async with idle.request_tracker():
            if idle.caller_left(request):
                logger.info(f"[{request_id}] Caller already disconnected, skipping inference")
                return Response(status_code=204)
            try:
                # NOTE: The message below won't be printed whenever the validation fails given that will
                # be automatically handled by FastAPI as we're defining the input as `input_schema = Body(...)`
                logger.info(f"[{request_id}] Received request with: {payload.model_dump()}")
                return predictor(payload=payload)
            except (ValueError, ValidationError) as e:
                logger.error(f"[{request_id}] Failed validating I/O with: {str(e)}")
                raise HTTPException(status_code=400, detail=str(e))
            except Exception as e:
                logger.error(f"[{request_id}] Failed running inference with: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))

    return router
