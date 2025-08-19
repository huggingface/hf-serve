from typing import List, Literal, Optional, Union

from pydantic import AliasChoices, BaseModel, ConfigDict, Field
import torch
from transformers.pipelines import pipeline

from huggingface_inference_toolkit.serde import ImageInput
from huggingface_inference_toolkit.tasks.predictor import Predictor


class ImageClassificationParameters(BaseModel):
    function_to_apply: Optional[Literal["sigmoid", "softmax", "none"]] = None
    top_k: Optional[int] = None


class ImageClassificationInput(ImageInput):
    inputs: Union[str, bytes] = Field(validation_alias=AliasChoices("inputs", "image"))
    parameters: Optional[ImageClassificationParameters] = None

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "inputs": "https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png",
                    "parameters": {
                        "function_to_apply": "softmax",
                        "top_k": 5,
                    },
                }
            ]
        }
    )


class ImageClassificationOutputValue(BaseModel):
    label: str
    score: float


class ImageClassificationOutput(BaseModel):
    results: List[ImageClassificationOutputValue]


class ImageClassification(Predictor[ImageClassificationInput, ImageClassificationOutput]):
    def __init__(self, model_id: str, dtype: str = "float16", device: str = "auto") -> None:
        super().__init__()

        # Handle device selection for models that don't support device_map
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"

        self.pipeline = pipeline(
            task="image-classification",
            model=model_id,
            torch_dtype=getattr(torch, dtype),
            device=device if device not in {"auto"} else None,
            device_map=device if device in {"auto"} else None,
        )

        if torch.mps.is_available():
            torch.mps.empty_cache()
            torch.mps.set_per_process_memory_fraction(0.9)

    def __call__(self, payload: ImageClassificationInput) -> ImageClassificationOutput:
        parameters = {}
        if payload.parameters:
            parameters = payload.parameters.model_dump(exclude_none=True)

        image_input = payload.inputs

        results = self.pipeline(image_input, **parameters)

        return ImageClassificationOutput(results=results)
