from typing import Annotated, List, Literal, Optional, Union

from fastapi import Form
from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from hf_serve.serde import Audio
from hf_serve.tasks.predictor import Predictor
from hf_serve.types import FileForm, IntForm


class AudioClassificationParameters(BaseModel):
    function_to_apply: Optional[Literal["sigmoid", "softmax", None]] = Field(default=None)
    top_k: Optional[int] = None


class AudioClassificationInput(BaseModel):
    inputs: Union[str, bytes] = Field(validation_alias=AliasChoices("inputs", "audio"))
    parameters: Optional[AudioClassificationParameters] = None

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "inputs": "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/1.flac",
                    "parameters": {
                        "top_k": 5,
                        "function_to_apply": "softmax",
                    },
                }
            ]
        }
    )


class AudioClassificationFormInput(BaseModel):
    file: FileForm
    function_to_apply: Optional[Annotated[Literal["sigmoid", "softmax", None], Form()]] = None
    top_k: Optional[IntForm] = None

    model_config = ConfigDict(extra="forbid")


class AudioClassificationOutputValue(BaseModel):
    label: str
    score: float


class AudioClassificationOutput(BaseModel):
    results: List[AudioClassificationOutputValue]


class AudioClassification(Predictor[AudioClassificationInput, AudioClassificationOutput]):
    def __init__(self, model_id: str, dtype: Optional[str] = None, device: str = "auto") -> None:
        super().__init__()

        import torch
        from transformers.pipelines import pipeline
        from transformers.pipelines.audio_classification import AudioClassificationPipeline

        # NOTE: Apparently some (not all) models don't support the `device_map=auto` so we should probably
        # either add a check or just default to CUDA instead
        if device == "auto":
            # e.g. DistilBertForSequenceClassification won't support it
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"

        self.pipeline: AudioClassificationPipeline = pipeline(
            task="audio-classification",
            model=model_id,
            dtype=getattr(torch, dtype) if dtype is not None else "auto",
            device=device if device not in {"auto"} else None,
            device_map=device if device in {"auto"} else None,
        )

        if torch.mps.is_available():
            torch.mps.empty_cache()
            torch.mps.set_per_process_memory_fraction(0.9)

    def __call__(self, payload: AudioClassificationInput) -> AudioClassificationOutput:
        parameters = {}
        if payload.parameters:
            parameters = payload.parameters.model_dump(exclude_none=True)

        # NOTE: Handle different input types: bytes, URL, file path, or base64; no need to handle others given that
        # `Pydantic` already validates that the `inputs` is either `bytes` or `str`
        if isinstance(payload.inputs, bytes):
            audio_bytes = payload.inputs
        elif isinstance(payload.inputs, str):
            audio_bytes = Audio.deserialize(payload.inputs)

        results = self.pipeline(audio_bytes, **parameters)  # type: ignore
        return AudioClassificationOutput(results=results)  # type: ignore
