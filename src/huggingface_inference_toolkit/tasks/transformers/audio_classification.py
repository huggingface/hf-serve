from typing import List, Optional, Union

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator
import torch
from transformers.pipelines import pipeline

from huggingface_inference_toolkit.serde import Audio
from huggingface_inference_toolkit.tasks.predictor import Predictor


class AudioClassificationParameters(BaseModel):
    function_to_apply: Optional[str] = None
    top_k: Optional[int] = None


class AudioClassificationInput(BaseModel):
    inputs: str = Field(validation_alias=AliasChoices("inputs", "audio"))
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

class AudioClassificationOutputValue(BaseModel):
    label: str
    score: float


class AudioClassificationOutput(BaseModel):
    results: List[AudioClassificationOutputValue]


class AudioClassification(Predictor[AudioClassificationInput, AudioClassificationOutput]):
    def __init__(self, model_id: str, dtype: str = "float16", device: str = "auto") -> None:
        super().__init__()

        # Handle device selection for models that don't support device_map
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"

        self.pipeline = pipeline(
            task="audio-classification",
            model=model_id,
            torch_dtype=getattr(torch, dtype),
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

        audio_input = payload.inputs
        if isinstance(audio_input, str):
            if not audio_input.startswith(("/", "http://", "https://")) and "." not in audio_input.split("/")[-1]:
                audio_input = Audio.deserialize(audio_input)

        results = self.pipeline(audio_input, **parameters)

        return AudioClassificationOutput(results=results)
