from fastapi import APIRouter, Body, HTTPException, Request, Response
from pydantic import ValidationError

from hf_serve.logging import logger
from hf_serve.openai.schemas.speech import SpeechInput
from hf_serve.openai.tasks.speech import Speech


def router(predictor: Speech) -> APIRouter:
    router = APIRouter()

    @router.post("/v1/audio/speech")
    async def speech(request: Request, payload: SpeechInput = Body(...)) -> Response:
        request_id = getattr(request.state, "request_id", None)

        try:
            logger.info(
                f"[{request_id}] Received create speech request with: {payload.model_dump(by_alias=True)}"
            )

            if payload.stream_format == "sse":
                raise ValueError(
                    "`stream_format: 'sse'` is not yet supported in `hf-serve`, hence when calling this endpoint with `stream_format` set to `sse` instead of `audio` it will fail.\nCheck https://github.com/huggingface/transformers to track the progress on streaming support for audio pipelines."
                )

            output = predictor(payload=payload, request_id=request_id)
            return Response(content=output.root, media_type=f"audio/{payload.response_format}")
        except (ValueError, ValidationError) as e:
            logger.error(f"[{request_id}] Failed validating I/O with: {str(e)}")
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"[{request_id}] Failed running inference with: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    return router
