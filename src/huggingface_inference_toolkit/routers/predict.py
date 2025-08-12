from typing import Type, Union

from fastapi import APIRouter, Body, File, HTTPException, UploadFile, Request
from pydantic import BaseModel

from huggingface_inference_toolkit.tasks.predictor import Predictor
from huggingface_inference_toolkit.tasks.transformers.audio_classification import AudioClassificationInput
from huggingface_inference_toolkit.tasks.transformers.automatic_speech_recognition import AutomaticSpeechRecognitionInput
from huggingface_inference_toolkit.tasks.transformers.zero_shot_audio_classification import ZeroShotAudioClassificationInput

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


    @router.post("/predict-json", response_model=output_schema)
    async def predict_json(payload: input_schema = Body(...)) -> output_schema:  # type: ignore
        try:
            return predictor(payload=payload)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        
    @router.post("/predict-file", response_model=output_schema)
    async def predict_file(file: UploadFile) -> output_schema:  # type: ignore
        try:
            if not file.filename.lower().endswith(('.flac', '.mp3', '.mp4', '.mpeg', '.mpga', '.m4a', '.ogg', '.wav', '.webm')):
                    raise HTTPException(status_code=400, detail="Unsupported audio file format.")

            content = await file.read()
            payload = input_schema(inputs=content, parameters=None)  # type: ignore

            try:
                res = predictor(payload=payload)
            except Exception as e:
                import traceback
                traceback.print_exc()
            return res
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @router.post("/", response_model=output_schema)
    @router.post("/predict", response_model=output_schema)
    async def predict(request: Request) -> output_schema:  # type: ignore
        ct = request.headers.get("content-type", "")
        match ct:
            case "application/json":
                payload = await request.json()
                audio_payload = {}
                try:
                    audio_payload = input_schema(**payload)  # type: ignore
                except Exception as e:
                    raise HTTPException(status_code=422, detail=e.errors())

                return await predict_json(payload=audio_payload)
            case "multipart/form-data":
                form = await request.form()
                file = form.get("file")

                # Checks audio
                if not file:
                    raise HTTPException(status_code=400, detail="File not found in the request.")

                return await predict_file(file=file)



    return router
