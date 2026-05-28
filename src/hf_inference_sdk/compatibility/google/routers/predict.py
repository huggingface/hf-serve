import os
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
    inner_input_schema: Union[Type[BaseModel], Type[Union[BaseModel, ...]]],  # type: ignore
) -> APIRouter:
    router = APIRouter()

    @router.post(os.getenv("AIP_PREDICT_ROUTE", "/predict"), response_model=output_schema)
    async def predict(request: Request, payload: input_schema = Body(...)) -> output_schema:  # type: ignore
        request_id = getattr(request.state, "request_id", None)

        async with idle.request_tracker():
            if idle.caller_left(request):
                logger.info(f"[{request_id}] Caller already disconnected, skipping inference")
                return Response(status_code=204)
            try:
                logger.info(f"[{request_id}] Received request with: {payload.model_dump()}")

                parameters = {}
                if payload.parameters is not None:
                    parameters = payload.parameters.model_dump()

                predictions = []
                for instance in payload.instances:
                    if isinstance(instance, BaseModel):
                        instance = instance.model_dump()

                    payload = inner_input_schema(**{"inputs": instance, "parameters": parameters})
                    prediction = predictor(payload=payload)
                    if prediction is None:
                        logger.error(f"[{request_id}] Prediction failed unexpectedly as it produced None...")
                        raise HTTPException(
                            status_code=500, detail="Prediction failed unexpectedly and produced None..."
                        )

                    predictions.append(prediction.model_dump())
                return output_schema(predictions=predictions)
            except (ValueError, ValidationError) as e:
                logger.error(f"[{request_id}] Failed validating I/O with: {str(e)}")
                raise HTTPException(status_code=400, detail=str(e))
            except Exception as e:
                logger.error(f"[{request_id}] Failed running inference with: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))

    return router
