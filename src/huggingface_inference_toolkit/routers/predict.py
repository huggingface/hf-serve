from pathlib import Path
from typing import Type, Union

from fastapi import APIRouter, Body, HTTPException, UploadFile, Request
import magic
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
            return predictor(payload=payload)
        # TODO(alvarobartt): create better custom exceptions and handle those here with different
        # error codes for I/O validation errors, ser/de errors, or pipeline errors
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return router


def audio_router(
    predictor: Predictor,
    input_schema: Union[Type[BaseModel], Type[Union[BaseModel, ...]]],  # type: ignore
    output_schema: Union[Type[BaseModel], Type[Union[BaseModel, ...]]],  # type: ignore
    task_name: str = "automatic_speech_recognition",
) -> APIRouter:
    router = APIRouter()

    accepted_extensions = ["flac", "mp3", "mp4", "mpeg", "mpga", "m4a", "ogg", "wav", "webm"]
    # TODO (juanjucm): double check how MIME types are handeled when sending files as multipart/form-data
    # It seems like the MIME type is always application/octet-stream.
    accepted_mimetypes = [
        "audio/flac",
        "audio/xflac",
        "audio/mpeg",
        "audio/mp4",
        "audio/ogg",
        "audio/wav",
        "audio/webm",
        "application/octet-stream",
    ]

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
            else:
                content = await file.read()

            payload = input_schema(inputs=content, parameters=None)  # type: ignore

            return predictor(payload=payload)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @router.post("/", response_model=output_schema)
    @router.post("/predict", response_model=output_schema)
    async def predict(request: Request) -> output_schema:  # type: ignore
        ct = request.headers.get("content-type", "")
        match ct:
            case _ if "application/json" in ct:
                payload = await request.json()
                try:
                    payload = input_schema(**payload)  # type: ignore
                except Exception as e:
                    raise HTTPException(status_code=422, detail=e.errors())

                return await predict_json(payload=payload)
            case _ if "multipart/form-data" in ct:
                form = await request.form()
                file = form.get("file")

                if not file:
                    raise HTTPException(status_code=400, detail="File not found in the request.")

                return await predict_file(file=file)

    return router
