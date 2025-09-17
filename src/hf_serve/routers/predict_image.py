from typing import Type, Union

from fastapi import APIRouter, Body, Request, Response
from fastapi.exceptions import HTTPException
from pydantic import BaseModel, ValidationError

from hf_serve.logging import logger
from hf_serve.serde.image import Image
from hf_serve.tasks.diffusers.text_to_image import TextToImage


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

        try:
            logger.info(f"[{request_id}] Received request with: {payload.model_dump()}")

            image = predictor(payload=payload)
            return Response(
                content=Image.serialize(
                    image=image.root,
                    image_format=image_format,  # type: ignore
                ),
                media_type=media_type,
            )
        except (ValueError, ValidationError) as e:
            logger.error(f"[{request_id}] Failed validating I/O with: {str(e)}")
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"[{request_id}] Failed running inference with: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    return router
