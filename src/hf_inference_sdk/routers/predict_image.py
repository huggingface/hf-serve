from io import BytesIO
from typing import Type, Union

from fastapi import APIRouter, Body, Request, Response
from fastapi.exceptions import HTTPException
from pydantic import BaseModel, ValidationError

from hf_inference_sdk import idle
from hf_inference_sdk.logging import logger
from hf_inference_sdk.tasks.diffusers.text_to_image import TextToImage


def router(
    predictor: TextToImage,
    input_schema: Union[Type[BaseModel], Type[Union[BaseModel, ...]]],  # type: ignore
) -> APIRouter:
    router = APIRouter()

    responses = {
        200: {
            "content": {"image/png": {}, "image/jpeg": {}, "image/webp": {}},
            "description": "Returns the image as PNG, JPEG, or WEBP, depending on Accept header.",
        }
    }

    # NOTE: As `text-to-image` returns the raw bytes of the image, there's no `response_model`
    @router.post("/", responses=responses)  # type: ignore
    @router.post("/predict", responses=responses)  # type: ignore
    async def predict(request: Request, payload: input_schema = Body(...)) -> Response:  # type: ignore
        request_id = getattr(request.state, "request_id", None)

        match media_type := request.headers.get("accept", ""):
            case "image/png" | "image/jpeg" | "image/webp":
                image_format = media_type.split("/")[-1].lower()
            case _:
                raise HTTPException(
                    status_code=406,
                    detail="Accept header must specify image/png, image/jpeg, or image/webp",
                )

        async with idle.request_tracker():
            try:
                logger.info(f"[{request_id}] Received request with: {payload.model_dump()}")

                output = predictor(payload=payload)

                # NOTE: We don't use the `Image.serialize` as we don't need to serialize it
                # as base64 but rather provide the file bytes instead
                buffer = BytesIO()
                output.root.save(buffer, **{"format": image_format})
                content = buffer.getvalue()

                return Response(content=content, media_type=media_type)
            except (ValueError, ValidationError) as e:
                logger.error(f"[{request_id}] Failed validating I/O with: {str(e)}")
                raise HTTPException(status_code=400, detail=str(e))
            except Exception as e:
                logger.error(f"[{request_id}] Failed running inference with: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))

    return router
