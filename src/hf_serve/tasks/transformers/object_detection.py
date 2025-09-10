from typing import List, Optional, Union

from PIL.Image import Image as ImageType
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_serializer

from hf_serve.serde import Image
from hf_serve.tasks.predictor import Predictor


class ObjectDetectionParameters(BaseModel):
    threshold: Optional[float] = None


class ObjectDetectionInput(BaseModel):
    inputs: Union[str, bytes] = Field(validation_alias=AliasChoices("inputs", "image"))
    parameters: Optional[ObjectDetectionParameters] = None

    @field_serializer("inputs")
    @classmethod
    def deserialize_inputs(cls, v: Union[str, bytes]) -> ImageType:
        return Image.deserialize(v)

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        json_schema_extra={
            "examples": [
                {
                    "inputs": "https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png",
                    "parameters": {
                        "threshold": 0.9,
                    },
                }
            ]
        },
    )


class BoundingBox(BaseModel):
    xmin: float
    ymin: float
    xmax: float
    ymax: float


class ObjectDetectionOutputValue(BaseModel):
    label: str
    score: float
    box: BoundingBox


class ObjectDetectionOutput(BaseModel):
    results: List[ObjectDetectionOutputValue]


class ObjectDetection(Predictor[ObjectDetectionInput, ObjectDetectionOutput]):
    def __init__(self, model_id: str, dtype: str = "float16", device: str = "auto") -> None:
        super().__init__()

        import torch
        from transformers import pipeline
        from transformers.pipelines.object_detection import ObjectDetectionPipeline

        # NOTE: Apparently some (not all) models don't support the `device_map=auto` so we should probably
        # either add a check or just default to CUDA instead
        if device == "auto":
            # e.g. DistilBertForSequenceClassification won't support it
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"

        self.pipeline: ObjectDetectionPipeline = pipeline(
            task="object-detection",
            model=model_id,
            torch_dtype=getattr(torch, dtype),
            device=device if device not in {"auto"} else None,
            device_map=device if device in {"auto"} else None,
        )

        if torch.mps.is_available():
            torch.mps.empty_cache()
            torch.mps.set_per_process_memory_fraction(0.9)

    def __call__(self, payload: ObjectDetectionInput) -> ObjectDetectionOutput:
        payload = payload.model_dump(exclude_none=True)  # dumping model to trigger @field_serializer.
        parameters = payload.get("parameters", {})
        image_input = payload.get("inputs")

        results = self.pipeline(image_input, **parameters)  # type: ignore
        return ObjectDetectionOutput(results=results)
