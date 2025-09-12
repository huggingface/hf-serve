from typing import Annotated, List, Optional, Type, Union

from fastapi import APIRouter, Body, Form, HTTPException, Request
from pydantic import BaseModel, ValidationError

from hf_serve.logging import logger
from hf_serve.tasks.predictor import Predictor
from hf_serve.routers.routers_utils import FileValidator


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

        try:
            errors = await file_validator.validate_file(form.file)
            if errors:
                raise ValueError(f"Invalid file: {'\n'.join(errors)}")

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

        try:
            errors = await file_validator.validate_file(file)
            if errors:
                raise ValueError(f"Invalid file: {'\n'.join(errors)}")

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
                payload = await request.json()
                try:
                    payload = input_schema(**payload)  # type: ignore
                except Exception as e:
                    raise HTTPException(status_code=422, detail=e.errors())

                return await predict_json(request=request, payload=payload)
            case "multipart/form-data":
                form = await request.form()
                try:
                    # NOTE: Since we are calling `predict_form()` manually (again, not a good practice),
                    # we need to parse the form and create a `input_form_schema`.
                    # File needs to be manually read and converted from `UploadFile` to expected bytes.
                    form = dict(form)

                    file = form.pop("file", None)
                    file = await file.read()

                    form_schema = input_form_schema(file=file, **form)  # type: ignore
                except Exception as e:
                    raise HTTPException(status_code=422, detail=str(e))

                return await predict_form(request=request, form=form_schema)
            case _ if (
                ct in accepted_mimetypes
            ):  # NOTE: Manage files sent direclty which any valid content-type.
                body_file = await request.body()
                return await predict_file(request=request, file=body_file)
            case _:
                raise HTTPException(
                    status_code=415,
                    detail=f"Unsupported Content-Type '{ct}'. Supported Content-Types are: 'application/json', 'multipart/form-data' or any of {accepted_mimetypes}",
                )

    return router
