from typing import Type

from fastapi import APIRouter, Body, HTTPException
from pydantic import BaseModel

from huggingface_inference_toolkit.tasks.predictor import Predictor


def router(predictor: Predictor, input_schema: Type[BaseModel], output_schema: Type[BaseModel]) -> APIRouter:
    router = APIRouter()

    @router.post("/predict", response_model=output_schema)
    async def predict(input: input_schema = Body(...)) -> output_schema:  # type: ignore
        try:
            return predictor(input=input)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return router
