from typing import List, Optional, Union

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, RootModel, field_validator

from hf_inference_sdk.serde import Image
from hf_inference_sdk.tasks.predictor import Predictor


class ZeroShotImageClassificationParameters(BaseModel):
    candidate_labels: List[str] = Field(validation_alias=AliasChoices("candidate_labels", "labels"))
    hypothesis_template: Optional[str] = Field(default="This is a photo of {}")

    @field_validator("candidate_labels")
    def validate_candidate_labels(cls, v):
        if not v:
            raise ValueError("candidate_labels must contain at least one label")
        return v

    @field_validator("hypothesis_template")
    def validate_hypothesis_template(cls, v):
        if not v:
            return None
        if not v.__contains__("{}"):
            raise ValueError(
                f'The provided `hypothesis_template={v}` doesn\'t contain {{}} which is required to flag where the `candidate_labels` need to be replaced in, as e.g. the default value which is "This is a photo of {{}}"'
            )
        return v


class ZeroShotImageClassificationInput(BaseModel):
    inputs: Union[str, bytes] = Field(validation_alias=AliasChoices("inputs", "image"))
    parameters: ZeroShotImageClassificationParameters

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "inputs": "https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png",
                    "parameters": {
                        "candidate_labels": ["parrots", "car", "building"],
                        "hypothesis_template": "This is a photo of {}",
                    },
                }
            ]
        },
    )


class ZeroShotImageClassificationOutputValue(BaseModel):
    label: str
    score: float


class ZeroShotImageClassificationOutput(RootModel):
    root: List[ZeroShotImageClassificationOutputValue]


class ZeroShotImageClassification(
    Predictor[ZeroShotImageClassificationInput, ZeroShotImageClassificationOutput]
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
        from transformers import pipeline
        from transformers.pipelines.zero_shot_image_classification import ZeroShotImageClassificationPipeline

        # NOTE: Apparently some (not all) models don't support the `device_map=auto` so we should probably
        # either add a check or just default to CUDA instead
        if device == "auto":
            # e.g. DistilBertForSequenceClassification won't support it
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"

        self.pipeline: ZeroShotImageClassificationPipeline = pipeline(
            task="zero-shot-image-classification",
            model=model_id,
            revision=revision,
            dtype=getattr(torch, dtype) if dtype is not None else "auto",
            device=device if device not in {"auto"} else None,
            device_map=device if device in {"auto"} else None,
            trust_remote_code=trust_remote_code,
        )

        if torch.mps.is_available():
            torch.mps.empty_cache()
            torch.mps.set_per_process_memory_fraction(0.9)

    def __call__(self, payload: ZeroShotImageClassificationInput) -> ZeroShotImageClassificationOutput:
        parameters = {}
        if payload.parameters:
            parameters = payload.parameters.model_dump(exclude_none=True)

        output = self.pipeline(Image.deserialize(payload.inputs), **parameters)
        return ZeroShotImageClassificationOutput(root=output)  # type: ignore
