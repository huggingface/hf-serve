from typing import Annotated, Type, Union

from fastapi import APIRouter, Body, HTTPException, UploadFile, Request

from pydantic import BaseModel, ValidationError

from hf_serve.logging import logger
from hf_serve.tasks.predictor import Predictor
from hf_serve.routers.routers_utils import DocumentValidator


def media_router(
    predictor: Predictor,
    input_schema: Union[Type[BaseModel], Type[Union[BaseModel, ...]]],  # type: ignore
    output_schema: Union[Type[BaseModel], Type[Union[BaseModel, ...]]],  # type: ignore
    accepted_mimetypes: list[str],
    max_document_size: int,
) -> APIRouter:
    router = APIRouter()

    doc_validator = DocumentValidator(accepted_mimetypes=accepted_mimetypes, max_size=max_document_size)

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

    @router.post("/predict-form-file", response_model=output_schema)
    async def predict_form_file(request: Request, file: UploadFile) -> output_schema:  # type: ignore
        request_id = getattr(request.state, "request_id", None)

        try:
            content = await file.read()

            res = await doc_validator.validate_file(content)
            if not res["valid"]:
                raise ValueError(f"Invalid file: {'\n'.join(res['errors'])}")

            payload = input_schema(inputs=content, parameters=None)

            return predictor(payload=payload)
        except (ValueError, ValidationError) as e:
            logger.error(f"[{request_id}] Failed validating I/O with: {str(e)}")
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"[{request_id}] Failed running inference with: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    @router.post("/predict-bytes-file", response_model=output_schema)
    async def predict_bytes_file(request: Request, file: Annotated[bytes, Body()]) -> output_schema:  # type: ignore
        request_id = getattr(request.state, "request_id", None)

        try:
            res = await doc_validator.validate_file(file)
            if not res["valid"]:
                raise ValueError(f"Invalid file: {'\n'.join(res['errors'])}")

            payload = input_schema(inputs=file, parameters=None)

            return predictor(payload=payload)
        except (ValueError, ValidationError) as e:
            logger.error(f"[{request_id}] Failed validating I/O with: {str(e)}")
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"[{request_id}] Failed running inference with: {str(e)}")
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

            return await predict_json(request=request, payload=payload)
        else:
            form = await request.form()
            file = form.get("file", None)
            if not file:
                body_file = await request.body()
                return await predict_bytes_file(request=request, file=body_file)

            return await predict_form_file(request=request, file=file)

    return router
