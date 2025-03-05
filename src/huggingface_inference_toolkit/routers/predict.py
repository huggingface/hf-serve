from typing import Type, Union

from fastapi import APIRouter, Body, HTTPException
from pydantic import BaseModel

from huggingface_inference_toolkit.tasks.predictor import Predictor


def router(
    predictor: Predictor,
    input_schema: Union[Type[BaseModel], Type[Union[BaseModel, ...]]],  # type: ignore
    output_schema: Union[Type[BaseModel], Type[Union[BaseModel, ...]]],  # type: ignore
) -> APIRouter:
    router = APIRouter()

    # NOTE: for Inference Endpoints we also need to route to / for the /predict route, as
    # that's the endpoint being hit within the Inference API widgets
    @router.post("/", response_model=output_schema)
    @router.post("/predict", response_model=output_schema)
    async def predict(payload: input_schema = Body(...)) -> output_schema:  # type: ignore
        try:
            return predictor(payload)
        # TODO(alvarobartt): create better custom exceptions and handle those here with different
        # error codes for I/O validation errors, ser/de errors, or pipeline errors
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return router
