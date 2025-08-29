from collections.abc import AsyncIterator
from typing import Iterator, Union

from fastapi import APIRouter, Body, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import ValidationError

from huggingface_inference_toolkit.logging import logger
from huggingface_inference_toolkit.openai.schemas.chat_completions import (
    ChatCompletionsInput,
    ChatCompletionsOutput,
    ChatCompletionsOutputChunk,
)
from huggingface_inference_toolkit.openai.tasks.chat_completions import ChatCompletions


def router(predictor: ChatCompletions) -> APIRouter:
    router = APIRouter()

    @router.post(
        "/v1/chat/completions",
        response_model=None,  # NOTE: `None` as `Union[ChatCompletionsOutput, StreamingResponse]` won't work
    )
    async def chat_completions(
        request: Request,
        payload: ChatCompletionsInput = Body(...),
    ) -> Union[ChatCompletionsOutput, StreamingResponse]:
        request_id = getattr(request.state, "request_id", None)

        try:
            logger.info(f"[{request_id}] Received chat completions request with: {payload.model_dump()}")

            if payload.stream is True:
                return StreamingResponse(
                    iter_chunks(predictor(payload=payload, request_id=request_id)),  # type: ignore
                    media_type="text/event-stream",
                )

            try:
                output = next(predictor(payload=payload, request_id=request_id))  # type: ignore
            except StopIteration as e:
                if e.value and isinstance(e.value, ChatCompletionsOutput):
                    return e.value
                raise HTTPException(500, "Non-streaming chat completion couldn't be generated") from e

            return output

        except ValidationError as e:
            logger.error(f"[{request_id}] Failed validating I/O with: {str(e)}")
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"[{request_id}] Failed running inference with: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    return router


async def iter_chunks(chunks: Iterator[ChatCompletionsOutputChunk]) -> AsyncIterator[bytes]:
    for chunk in chunks:
        yield b"data: " + chunk.model_dump_json().encode() + b"\n\n"
    yield b"data: [DONE]\n\n"
