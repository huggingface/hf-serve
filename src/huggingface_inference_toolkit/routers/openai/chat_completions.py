from typing import Iterator, Type, Union

from fastapi import APIRouter, Body, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from huggingface_inference_toolkit.tasks.predictor import Predictor


def router(
    predictor: Predictor,
    input_schema: Union[Type[BaseModel], Type[Union[BaseModel, ...]]],  # type: ignore
    output_schema: Union[Type[BaseModel], Type[Union[BaseModel, Iterator[BaseModel], ...]]],  # type: ignore
) -> APIRouter:
    router = APIRouter()

    @router.post("/v1/chat/completions", response_model=output_schema)
    async def predict(payload: input_schema = Body(...)) -> output_schema:  # type: ignore
        try:
            if payload.stream:
                return StreamingResponse(predictor(payload=payload), media_type="application/json")
            return predictor(payload=payload)
        # TODO(alvarobartt): create better custom exceptions and handle those here with different
        # error codes for I/O validation errors, ser/de errors, or pipeline errors
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return router
