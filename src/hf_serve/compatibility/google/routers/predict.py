import os
from typing import Type, Union

from fastapi import APIRouter, Body, Request
from fastapi.exceptions import HTTPException
from pydantic import BaseModel, ValidationError

from hf_serve.logging import logger
from hf_serve.tasks.predictor import Predictor


def router(
    predictor: Predictor,
    input_schema: Union[Type[BaseModel], Type[Union[BaseModel, ...]]],  # type: ignore
    output_schema: Union[Type[BaseModel], Type[Union[BaseModel, ...]]],  # type: ignore
) -> APIRouter:
    router = APIRouter()

    @router.post(os.getenv("AIP_PREDICT_ROUTE", "/predict"), response_model=output_schema)
    async def predict(request: Request, payload: input_schema = Body(...)) -> output_schema:  # type: ignore
        request_id = getattr(request.state, "request_id", None)

        try:
            logger.info(f"[{request_id}] Received request with: {payload.model_dump()}")
            predictions = []
            for instance in payload.instances:
                prediction = predictor(payload={"inputs": instance, "parameters": payload.parameters})
                predictions.append(prediction)
            return output_schema(predictions=predictions)
        except (ValueError, ValidationError) as e:
            logger.error(f"[{request_id}] Failed validating I/O with: {str(e)}")
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"[{request_id}] Failed running inference with: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    return router
