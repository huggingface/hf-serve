from typing import Annotated, List, Optional, Type, Union

from fastapi import APIRouter, Body, Form, HTTPException, Request
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, ValidationError

from hf_inference_sdk import idle
from hf_inference_sdk.file_validator import FileValidator
from hf_inference_sdk.logging import logger
from hf_inference_sdk.tasks.predictor import Predictor


def media_router(
    predictor: Predictor,
    input_schema: Union[Type[BaseModel], Type[Union[BaseModel, ...]]],  # type: ignore
    input_form_schema: Union[Type[BaseModel], Type[Union[BaseModel, ...]]],  # type: ignore
    output_schema: Union[Type[BaseModel], Type[Union[BaseModel, ...]]],  # type: ignore
    accepted_mimetypes: List[str],
    max_file_size: Optional[int] = None,
) -> APIRouter:
    router = APIRouter()

    file_validator = FileValidator(accepted_mimetypes=accepted_mimetypes, max_size=max_file_size)

    @router.post("/predict-json", response_model=output_schema)
    async def predict_json(request: Request, payload: input_schema = Body(...)) -> output_schema:  # type: ignore
        request_id = getattr(request.state, "request_id", None)

        async with idle.request_tracker():
            try:
                logger.info(f"[{request_id}] Received request with: {payload.model_dump()}")
                return predictor(payload=payload)
            except (ValueError, ValidationError) as e:
                logger.error(f"[{request_id}] Failed validating I/O with: {str(e)}")
                raise HTTPException(status_code=400, detail=str(e))
            except Exception as e:
                logger.error(f"[{request_id}] Failed running inference with: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))

    @router.post("/predict-form", response_model=output_schema)
    async def predict_form(request: Request, form: Annotated[input_form_schema, Form()]) -> output_schema:  # type: ignore
        request_id = getattr(request.state, "request_id", None)

        async with idle.request_tracker():
            try:
                file_validator(form.file)

                # dump form into input schema
                form = form.model_dump()
                payload = {
                    "inputs": form.pop("file"),
                    "parameters": {},
                }

                # NOTE: Identify 'parameters' add non-'parameters' args in the schema (e.g. 'candidate_labels' in ZeroShotAudioClassification)
                # and manage each case accordingly.
                for field in input_schema.model_fields.keys():
                    if field not in ("inputs", "parameters") and field in form:
                        payload[field] = form.pop(field)
                if form:
                    payload["parameters"] = form

                payload = input_schema(**payload)  # type: ignore

                return predictor(payload=payload)
            except (ValueError, ValidationError) as e:
                logger.error(f"[{request_id}] Failed validating I/O with: {str(e)}")
                raise HTTPException(status_code=400, detail=str(e))
            except Exception as e:
                logger.error(f"[{request_id}] Failed running inference with: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))

    @router.post("/predict-file", response_model=output_schema)
    async def predict_file(request: Request, file: Annotated[bytes, Body()]) -> output_schema:  # type: ignore
        request_id = getattr(request.state, "request_id", None)

        async with idle.request_tracker():
            try:
                file_validator(file)

                payload = input_schema(inputs=file, parameters=None)

                return predictor(payload=payload)
            except (ValueError, ValidationError) as e:
                logger.error(f"[{request_id}] Failed validating I/O with: {str(e)}")
                raise HTTPException(status_code=400, detail=str(e))
            except Exception as e:
                logger.error(f"[{request_id}] Failed running inference with: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))

    # NOTE: This endpoint schema is not a good practice. Endpoints should be independent per input type.
    @router.post("/", response_model=output_schema)
    @router.post("/predict", response_model=output_schema)
    async def predict(request: Request) -> output_schema:  # type: ignore
        ct = request.headers.get("content-type", "").split(";")[0]
        # TODO: Revisit this logic.
        match ct:
            case "application/json":
                try:
                    return RedirectResponse(url="/predict-json")
                except Exception as e:
                    raise HTTPException(status_code=422, detail=str(e))
            case "multipart/form-data":
                try:
                    return RedirectResponse(url="/predict-form")
                except Exception as e:
                    raise HTTPException(status_code=422, detail=str(e))

            # NOTE: All other content-types are managed as binary files.
            # MIME type checking will happen inside `predict-file()`.
            # TODO (juanjucm/alvarobartt): Revisit this logic.
            case _:
                try:
                    return RedirectResponse(url="/predict-file")
                except Exception as e:
                    raise HTTPException(status_code=422, detail=str(e))

    return router
