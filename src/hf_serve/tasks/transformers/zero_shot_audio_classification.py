from typing import Annotated, List, Optional, Union

from fastapi import Form
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, RootModel, field_validator

from hf_serve.serde import Audio
from hf_serve.tasks.predictor import Predictor
from hf_serve.types.form import FileForm, StrForm


class ZeroShotAudioClassificationParameters(BaseModel):
    candidate_labels: List[str] = Field(validation_alias=AliasChoices("candidate_labels", "labels"))
    hypothesis_template: Optional[str] = None

    @field_validator("candidate_labels")
    def validate_candidate_labels(cls, v):
        if not v:
            raise ValueError("candidate_labels must contain at least one label")
        return v

    @field_validator("hypothesis_template")
    def validate_hypothesis_template(cls, v):
        if v is not None and "{}" not in v:
            raise ValueError("hypothesis_template must contain '{}' placeholder for label insertion")
        return v


class ZeroShotAudioClassificationInput(BaseModel):
    inputs: Union[str, bytes] = Field(validation_alias=AliasChoices("inputs", "audio"))
    parameters: ZeroShotAudioClassificationParameters

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


class ZeroShotAudioClassificationFormInput(BaseModel):
    file: FileForm

    candidate_labels: Annotated[List[str], Form()]
    hypothesis_template: Optional[StrForm] = None

    model_config = ConfigDict(extra="forbid")


class ZeroShotAudioClassificationOutputValue(BaseModel):
    label: str
    score: float


class ZeroShotAudioClassificationOutput(RootModel):
    root: List[ZeroShotAudioClassificationOutputValue]


class ZeroShotAudioClassification(
    Predictor[ZeroShotAudioClassificationInput, ZeroShotAudioClassificationOutput]
):
    def __init__(
        self,
        model_id: str,
        revision: Optional[str] = None,
        dtype: Optional[str] = None,
        device: str = "auto",
        trust_remote_code: bool = False,
    ) -> None:
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
            revision=revision,
            dtype=getattr(torch, dtype) if dtype is not None else "auto",
            device=device,
            trust_remote_code=trust_remote_code,
        )

        if torch.mps.is_available():
            torch.mps.empty_cache()
            torch.mps.set_per_process_memory_fraction(0.9)

    def __call__(self, payload: ZeroShotAudioClassificationInput) -> ZeroShotAudioClassificationOutput:
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
        return ZeroShotAudioClassificationOutput(root=results)  # type: ignore
