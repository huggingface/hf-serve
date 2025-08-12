from typing import List, Optional, Union

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator
import torch
from transformers.pipelines import pipeline

from huggingface_inference_toolkit.serde import Audio
from huggingface_inference_toolkit.tasks.predictor import Predictor


class ZeroShotAudioClassificationParameters(BaseModel):
    hypothesis_template: Optional[str] = None

    @field_validator("hypothesis_template")
    def validate_hypothesis_template(cls, v):
        if v is not None and "{}" not in v:
            raise ValueError("hypothesis_template must contain '{}' placeholder for label insertion")
        return v


class ZeroShotAudioClassificationInput(BaseModel):
    inputs: Union[str, bytes] = Field(validation_alias=AliasChoices("inputs", "audio"))
    candidate_labels: List[str] = Field(validation_alias=AliasChoices("candidate_labels", "labels"))
    parameters: Optional[ZeroShotAudioClassificationParameters] = None

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "inputs": "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/1.flac",
                    "candidate_labels": ["Sound of a dog", "Sound of vacuum cleaner", "Sound of a human voice"],
                    "parameters": {
                        "hypothesis_template": "This is a sound of {}",
                    },
                }
            ]
        }
    )

    @field_validator("inputs")
    def validate_inputs(cls, v):
        if isinstance(v, str):
            # If it's a string, it could be a file path or base64-encoded audio
            return v
        else:
            raise ValueError("inputs must be either a file path (str) or base64-encoded string")

    @field_validator("candidate_labels")
    def validate_candidate_labels(cls, v):
        if not v:
            raise ValueError("candidate_labels must contain at least one label")
        return v


class ZeroShotAudioClassificationOutputValue(BaseModel):
    label: str
    score: float


class ZeroShotAudioClassificationOutput(BaseModel):
    results: List[ZeroShotAudioClassificationOutputValue]


class ZeroShotAudioClassification(Predictor[ZeroShotAudioClassificationInput, ZeroShotAudioClassificationOutput]):
    def __init__(self, model_id: str, dtype: str = "float16", device: str = "auto") -> None:
        super().__init__()

        # Handle device selection for models that don't support device_map
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"

        self.pipeline = pipeline(
            task="zero-shot-audio-classification",
            model=model_id,
            torch_dtype=getattr(torch, dtype),
            device=device if device not in {"auto"} else None,
            device_map=device if device in {"auto"} else None,
        )

        if torch.mps.is_available():
            torch.mps.empty_cache()
            torch.mps.set_per_process_memory_fraction(0.9)

    def __call__(self, payload: ZeroShotAudioClassificationInput) -> ZeroShotAudioClassificationOutput:
        parameters = {}
        if payload.parameters:
            parameters = payload.parameters.model_dump(exclude_none=True)

        # Process the input based on its type
        audio_input = payload.inputs
        if isinstance(audio_input, str):
            # Check if it's a base64-encoded string (not a file path/URL)
            if not audio_input.startswith(('/', 'http://', 'https://')) and '.' not in audio_input.split('/')[-1]:
                audio_input = Audio.deserialize(audio_input)


        results = self.pipeline(audio_input,
                                candidate_labels=payload.candidate_labels,
                                **parameters
                                )

        return ZeroShotAudioClassificationOutput(results=results)