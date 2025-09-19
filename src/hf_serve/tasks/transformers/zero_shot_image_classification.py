from typing import List, Optional, Union

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, RootModel, field_validator

from hf_serve.serde import Image
from hf_serve.tasks.predictor import Predictor


class ZeroShotImageClassificationParameters(BaseModel):
    candidate_labels: List[str]
    hypothesis_template: Optional[str] = Field(default="This is a photo of {}")

    # TODO: Revisit the other `zero-shot-...` implementations to make sure those also include the validation
    # for both `candidate_labels` and `hypothesis_template`
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
    parameters: Optional[ZeroShotImageClassificationParameters] = Field(default=None)

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
    # TODO: Revisit the other `zero-shot-...` implementations to make sure this is a `root` rather than a key
    # as e.g. `results`
    root: List[ZeroShotImageClassificationOutputValue]


class ZeroShotImageClassification(
    Predictor[ZeroShotImageClassificationInput, ZeroShotImageClassificationOutput]
):
    def __init__(self, model_id: str, dtype: str = "float16", device: str = "auto") -> None:
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
            dtype=getattr(torch, dtype),
            device=device if device not in {"auto"} else None,
            device_map=device if device in {"auto"} else None,
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
