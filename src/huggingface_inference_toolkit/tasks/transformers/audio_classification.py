from typing import List, Optional, Union

import torch
from pydantic import BaseModel, ConfigDict, RootModel, field_validator

from huggingface_inference_toolkit.tasks.predictor import Predictor


class AudioClassificationParameters(BaseModel):
    function_to_apply: Optional[str] = None
    top_k: Optional[int] = None

    @field_validator("function_to_apply")
    def validate_function_to_apply(cls, v):
        if v in {'sigmoid', 'softmax', None}:
            return v
        else:
            raise ValueError("Parameter `function_to_apply` must be one of: sigmoid, softmax, none.")



class AudioClassificationInput(BaseModel):
    inputs: Union[str, bytes]  # Audio file path, audio in base64 encoding or raw audio bytes
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

    @field_validator("inputs")
    def validate_inputs(cls, v):
        if isinstance(v, str):
            # If it's a string, it should be a valid file path
            return v
        elif isinstance(v, bytes):
            # If it's bytes, return as is
            return v
        else:
            raise ValueError("inputs must be either a file path (str) or audio bytes")


class AudioClassificationOutputValue(BaseModel):
    label: str
    score: float


class AudioClassificationOutput(RootModel):
    root: List[AudioClassificationOutputValue]


class AudioClassification(Predictor[AudioClassificationInput, AudioClassificationOutput]):
    def __init__(self, model_id: str, dtype: str = "float16", device: str = "balanced") -> None:
        super().__init__()

        from transformers import pipeline as transformers_pipeline  # type: ignore

        # Handle device selection for models that don't support device_map
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"

        self.pipeline = transformers_pipeline(
            task="audio-classification",
            model=model_id,
            torch_dtype=getattr(torch, dtype),
            device=device if device not in {"auto"} else None,
            device_map=device if device in {"auto"} else None,
        )

        if torch.mps.is_available():
            torch.mps.empty_cache()
            torch.mps.set_per_process_memory_fraction(0.9)

        # Note: Warmup is skipped for audio classification as it requires actual audio data
        # which cannot be easily mocked from the example
        # TODO: include short audio for audio tasks warmup.

    def __call__(self, input: AudioClassificationInput) -> AudioClassificationOutput:
        payload = input.model_dump(exclude_none=True)

        # Extract inputs
        inputs = payload.pop("inputs")

        # The HF library expects parameters to be flattened
        if "parameters" in payload:
            parameters = payload.pop("parameters") or {}
            payload.update(parameters)

        pipeline_results = self.pipeline(inputs, **payload)  # type: ignore
        return AudioClassificationOutput(root=pipeline_results)

    @property
    def model_id(self) -> Union[str, None]:
        return self.pipeline.model.name_or_path if hasattr(self.pipeline, "model") else None