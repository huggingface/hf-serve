from typing import List, Optional, Union

from pydantic import AliasChoices, BaseModel, ConfigDict, Field
import torch
from transformers.pipelines import pipeline

from huggingface_inference_toolkit.serde import Image
from huggingface_inference_toolkit.tasks.predictor import Predictor


class ObjectDetectionParameters(BaseModel):
    threshold: Optional[float] = None


class ObjectDetectionInput(BaseModel):
    inputs: Union[str, bytes] = Field(validation_alias=AliasChoices("inputs", "image"))
    parameters: Optional[ObjectDetectionParameters] = None

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "inputs": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/cats.jpg",
                    "parameters": {
                        "threshold": 0.9,
                    },
                }
            ]
        }
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

        # Handle device selection for models that don't support device_map
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"

        self.pipeline = pipeline(
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
        parameters = {}
        if payload.parameters:
            parameters = payload.parameters.model_dump(exclude_none=True)

        image_input = payload.inputs

        # Deserialize if the input is bytes or a base64 string
        is_bytes = isinstance(image_input, bytes)
        is_base64_string = (
            isinstance(image_input, str)
            and not image_input.startswith(("/", "http://", "https://"))
            and "." not in image_input.split("/")[-1]
        )
        if is_bytes or is_base64_string:
            image_input = Image.deserialize(image_input)

        results = self.pipeline(image_input, **parameters)

        return ObjectDetectionOutput(results=results)
