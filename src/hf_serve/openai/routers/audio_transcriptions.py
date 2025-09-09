from collections.abc import AsyncIterator
from typing import Iterator, Union

from fastapi import APIRouter, Body, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import ValidationError

from hf_serve.logging import logger
from hf_serve.openai.schemas.audio_transcriptions import (
    AudioTranscriptionsInput,
    AudioTranscriptionsOutput,
    TranscriptTextDelta,
    TranscriptTextDone,
)
from hf_serve.openai.tasks.audio_transcriptions import AudioTranscriptions


def router(predictor: AudioTranscriptions) -> APIRouter:
    router = APIRouter()

    @router.post(
        "/v1/audio/transcriptions",
        response_model=None # NOTE: `None` as `Union[...]` of all possible outputs won't work
    )
    async def audio_transcriptions(request: Request,
                                   payload: AudioTranscriptionsInput = Body(...)
                                   ) -> Union[AudioTranscriptionsOutput, StreamingResponse]:
        request_id = getattr(request.state, "request_id", None)

        try:    
            logger.info(f"[{request_id}] Received audio transcriptions request with: {payload.model_dump()}")

            if payload.stream is True:
                return StreamingResponse(
                    iter_chunks(predictor(payload=payload, request_id=request_id)),  # type: ignore
                    media_type="text/event-stream",
                )

            try:
                output = next(predictor(payload=payload, request_id=request_id))  # type: ignore
            except StopIteration as e:
                if e.value and isinstance(e.value, AudioTranscriptionsOutput):
                    return e.value
                raise HTTPException(500, "Non-streaming audio transcription couldn't be generated") from e

            return output

        except (ValueError, ValidationError) as e:
            logger.error(f"[{request_id}] Failed validating I/O with: {str(e)}")
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"[{request_id}] Failed running inference with: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    return router


async def iter_chunks(chunks: Iterator[Union[TranscriptTextDelta, TranscriptTextDone]]) -> AsyncIterator[bytes]:
    for chunk in chunks:
        yield b"data: " + chunk.model_dump_json().encode() + b"\n\n"
    yield b"data: [DONE]\n\n"