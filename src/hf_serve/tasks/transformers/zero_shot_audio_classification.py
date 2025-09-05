from typing import List, Optional, Union

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator

from hf_serve.serde import Audio
from hf_serve.tasks.predictor import Predictor


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


class ZeroShotAudioClassification(
    Predictor[ZeroShotAudioClassificationInput, ZeroShotAudioClassificationOutput]
):
    def __init__(self, model_id: str, dtype: str = "float16", device: str = "auto") -> None:
        super().__init__()

        import torch
        from transformers.pipelines import pipeline
        from transformers.pipelines.zero_shot_audio_classification import ZeroShotAudioClassificationPipeline

        # NOTE: Apparently some (not all) models don't support the `device_map=auto` so we should probably
        # either add a check or just default to CUDA instead
        if device == "auto":
            # e.g. DistilBertForSequenceClassification won't support it
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"

        self.pipeline: ZeroShotAudioClassificationPipeline = pipeline(
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

        audio = payload.inputs
        if isinstance(audio, str):
            # NOTE: Deserialize it into `bytes` if it's a base64-encoded `str`, or leave it as is if it's either
            # a URL or a filepath as it will be automatically handled by the `AutoPipeline.__call__`
            if not audio.startswith(("/", "http://", "https://")) and "." not in audio.split("/")[-1]:
                audio = Audio.deserialize(audio)

        results = self.pipeline(audio, candidate_labels=payload.candidate_labels, **parameters)
        return ZeroShotAudioClassificationOutput(results=results)  # type: ignore
