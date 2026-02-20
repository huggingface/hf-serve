from typing import List, Optional

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, RootModel, field_validator

from hf_inference_sdk.tasks.predictor import Predictor


class ZeroShotClassificationParameters(BaseModel):
    candidate_labels: List[str] = Field(validation_alias=AliasChoices("candidate_labels", "labels"))
    hypothesis_template: Optional[str] = None
    multi_label: Optional[bool] = None

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


class ZeroShotClassificationInput(BaseModel):
    inputs: str = Field(
        validation_alias=AliasChoices("inputs", "sequences", "text"),
    )
    parameters: ZeroShotClassificationParameters

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "inputs": "I have a problem with my iphone that needs to be resolved ASAP!",
                    "parameters": {
                        "candidate_labels": ["urgent", "not urgent", "phone", "tablet", "computer"],
                    },
                }
            ]
        }
    )


class ZeroShotClassificationOutputValue(BaseModel):
    sequence: str
    labels: List[str]
    scores: List[float]


class ZeroShotClassificationOutput(RootModel):
    root: ZeroShotClassificationOutputValue


class ZeroShotClassification(Predictor[ZeroShotClassificationInput, ZeroShotClassificationOutput]):
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
        from transformers import pipeline
        from transformers.pipelines.zero_shot_classification import ZeroShotClassificationPipeline

        # NOTE: Apparently some (not all) models don't support the `device_map=auto` so we should probably
        # either add a check or just default to CUDA instead
        if device == "auto":
            # e.g. DistilBertForSequenceClassification won't support it
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"

        self.pipeline: ZeroShotClassificationPipeline = pipeline(
            task="zero-shot-classification",
            model=model_id,
            revision=revision,
            dtype=getattr(torch, dtype) if dtype is not None else "auto",
            device=device,
            trust_remote_code=trust_remote_code,
        )

        if torch.mps.is_available():
            torch.mps.empty_cache()
            torch.mps.set_per_process_memory_fraction(0.9)

    def __call__(self, payload: ZeroShotClassificationInput) -> ZeroShotClassificationOutput:
        parameters = {}
        if payload.parameters:
            parameters = payload.parameters.model_dump(exclude_none=True)

        output = self.pipeline(payload.inputs, **parameters)
        return ZeroShotClassificationOutput(root=output)  # type: ignore
