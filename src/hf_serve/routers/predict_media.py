from typing import Type, Union

from fastapi import APIRouter, Body, HTTPException, UploadFile, Request
import magic
from pydantic import BaseModel, ValidationError

from hf_serve.logging import logger
from hf_serve.tasks.predictor import Predictor


def media_router(
    predictor: Predictor,
    input_schema: Union[Type[BaseModel], Type[Union[BaseModel, ...]]],  # type: ignore
    output_schema: Union[Type[BaseModel], Type[Union[BaseModel, ...]]],  # type: ignore
    accepted_mimetypes: list[str],
) -> APIRouter:
    router = APIRouter()

    @router.post("/predict-json", response_model=output_schema)
    async def predict_json(request: Request, payload: input_schema = Body(...)) -> output_schema:  # type: ignore
        request_id = getattr(request.state, "request_id", None)

        try:
            logger.info(f"[{request_id}] Received request with: {payload.model_dump()}")
            return predictor(payload=payload)
        except (ValueError, ValidationError) as e:
            logger.error(f"[{request_id}] Failed validating I/O with: {str(e)}")
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"[{request_id}] Failed running inference with: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    @router.post("/predict-file", response_model=output_schema)
    async def predict_file(file: UploadFile) -> output_schema:  # type: ignore
        try:
            chunk = await file.read(2048)
            mime_type = magic.from_buffer(chunk, mime=True)

            if mime_type not in accepted_mimetypes:
                raise HTTPException(
                    status_code=400,
                    detail=f"Provided file with MIME type {mime_type} is not supported. "
                    + "Supported MIME types are: "
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
