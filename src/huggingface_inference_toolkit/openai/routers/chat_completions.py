from collections.abc import AsyncIterator
from typing import Iterator, Type, Union

from fastapi import APIRouter, Body, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from huggingface_inference_toolkit.tasks.predictor import Predictor


def router(
    predictor: Predictor,
    input_schema: Union[Type[BaseModel], Type[Union[BaseModel, ...]]],  # type: ignore
    output_schema: Union[Type[BaseModel], Type[Union[BaseModel, ...]]],  # type: ignore
) -> APIRouter:
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
                        "created": "",
                        "owned_by": predictor.model_id.split("/")[0]
                        if predictor.model_id is not None and predictor.model_id.__contains__("/")
                        else None,
                    }
                ],
            }
        )

    @router.post("/v1/chat/completions", response_model=output_schema)
    async def predict(payload: input_schema = Body(...)) -> Union[output_schema, StreamingResponse]:  # type: ignore
        try:
            if payload.stream is True:
                return StreamingResponse(
                    iter_chunks(predictor(payload=payload)), media_type="text/event-stream"
                )

            try:
                output = next(predictor(payload=payload))
            except StopIteration as e:
                if e.value and isinstance(e.value, output_schema):
                    return e.value
                raise HTTPException(500, "Non-streaming chat completion couldn't be generated") from e

            return output

        # TODO(alvarobartt): create better custom exceptions and handle those here with different
        # error codes for I/O validation errors, ser/de errors, or pipeline errors
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return router


async def iter_chunks(chunks: Iterator[BaseModel]) -> AsyncIterator[bytes]:
    for chunk in chunks:
        yield b"data: " + chunk.model_dump_json().encode() + b"\n\n"
    yield b"data: [DONE]\n\n"
