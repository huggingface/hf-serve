from typing import List, Optional, Union

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, RootModel, field_validator

from hf_serve.serde import Image
from hf_serve.tasks.predictor import Predictor
from hf_serve.tasks.transformers.object_detection import BoundingBox


class ZeroShotObjectDetectionParameters(BaseModel):
    candidate_labels: List[str] = Field(validation_alias=AliasChoices("candidate_labels", "labels"))
    threshold: Optional[float] = None
    top_k: Optional[int] = Field(default=None, gt=0)

    @field_validator("candidate_labels")
    def validate_candidate_labels(cls, v):
        if not v:
            raise ValueError("candidate_labels must contain at least one label")
        return v


class ZeroShotObjectDetectionInput(BaseModel):
    inputs: Union[str, bytes] = Field(validation_alias=AliasChoices("inputs", "image"))
    parameters: ZeroShotObjectDetectionParameters

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "inputs": "https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png",
                    "parameters": {
                        "candidate_labels": ["parrot", "branch"],
                        "threshold": 0.1,
                    },
                }
            ]
        },
    )


class ZeroShotObjectDetectionOutputValue(BaseModel):
    label: str
    score: float
    box: BoundingBox


class ZeroShotObjectDetectionOutput(RootModel):
    root: List[ZeroShotObjectDetectionOutputValue]


class ZeroShotObjectDetection(Predictor[ZeroShotObjectDetectionInput, ZeroShotObjectDetectionOutput]):
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
        from transformers.pipelines.zero_shot_object_detection import ZeroShotObjectDetectionPipeline

        # NOTE: Apparently some (not all) models don't support the `device_map=auto` so we should probably
        # either add a check or just default to CUDA instead
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"

        self.pipeline: ZeroShotObjectDetectionPipeline = pipeline(
            task="zero-shot-object-detection",
            model=model_id,
            revision=revision,
            dtype=getattr(torch, dtype) if dtype is not None else "auto",
            device=device,
            trust_remote_code=trust_remote_code,
        )

        if torch.mps.is_available():
            torch.mps.empty_cache()
            torch.mps.set_per_process_memory_fraction(0.9)

    def __call__(self, payload: ZeroShotObjectDetectionInput) -> ZeroShotObjectDetectionOutput:
        parameters = payload.parameters.model_dump(exclude_none=True)
        output = self.pipeline(Image.deserialize(payload.inputs), **parameters)
        return ZeroShotObjectDetectionOutput(root=output)  # type: ignore
