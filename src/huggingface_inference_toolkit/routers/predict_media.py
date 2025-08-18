from pathlib import Path
from typing import Type, Union

from fastapi import APIRouter, Body, HTTPException, UploadFile, Request
import magic
from pydantic import BaseModel

from huggingface_inference_toolkit.tasks.predictor import Predictor


def media_router(
    predictor: Predictor,
    input_schema: Union[Type[BaseModel], Type[Union[BaseModel, ...]]],  # type: ignore
    output_schema: Union[Type[BaseModel], Type[Union[BaseModel, ...]]],  # type: ignore
    media_type: str = "all",  # "audio", "image", or "all"
) -> APIRouter:
    router = APIRouter()

    audio_extensions = ["flac", "mp3", "mp4", "mpeg", "mpga", "m4a", "ogg", "wav", "webm"]
    audio_mimetypes = [
        "audio/flac",
        "audio/xflac",
        "audio/mpeg",
        "audio/mp4",
        "audio/ogg",
        "audio/wav",
        "audio/webm",
    ]

    image_extensions = ["jpg", "jpeg", "png", "bmp", "gif"]
    image_mimetypes = [
        "image/jpeg",
        "image/png",
        "image/bmp",
        "image/gif",
    ]

    if media_type == "audio":
        accepted_extensions = audio_extensions
        accepted_mimetypes = audio_mimetypes + ["application/octet-stream"]
    elif media_type == "image":
        accepted_extensions = image_extensions
        accepted_mimetypes = image_mimetypes + ["application/octet-stream"]
    else:  # "all"
        accepted_extensions = audio_extensions + image_extensions
        accepted_mimetypes = audio_mimetypes + image_mimetypes + ["application/octet-stream"]

    @router.post("/predict-json", response_model=output_schema)
    async def predict_json(payload: input_schema = Body(...)) -> output_schema:  # type: ignore
        try:
            return predictor(payload=payload)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @router.post("/predict-file", response_model=output_schema)
    async def predict_file(file: UploadFile) -> output_schema:  # type: ignore
        try:
            extension = Path(file.filename).suffix.lower().lstrip(".")
            chunk = await file.read(2048)
            mime_type = magic.from_buffer(chunk, mime=True)

            if extension not in accepted_extensions and mime_type not in accepted_mimetypes:
                raise HTTPException(
                    status_code=400,
                    detail=f"Provided file with extension {extension} and MIME type {mime_type} is not supported. \
                                            Supported extensions are: "
                    + ", ".join(accepted_extensions)
                    + ". Supported MIME types are: "
                    + ", ".join(accepted_mimetypes),
                )
            await file.seek(0)
            content = await file.read()

            payload = input_schema(inputs=content, parameters=None)  # type: ignore

            return predictor(payload=payload)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @router.post("/", response_model=output_schema)
    @router.post("/predict", response_model=output_schema)
    async def predict(request: Request) -> output_schema:  # type: ignore
        ct = request.headers.get("content-type", "")
        if "application/json" in ct:
            payload = await request.json()
            try:
                payload = input_schema(**payload)  # type: ignore
            except Exception as e:
                raise HTTPException(status_code=422, detail=e.errors())

            return await predict_json(payload=payload)
        elif "multipart/form-data" in ct:
            form = await request.form()
            file = form.get("file", None)
            if not file:
                raise HTTPException(status_code=400, detail="File not found in the request.")

            return await predict_file(file=file)

    return router
