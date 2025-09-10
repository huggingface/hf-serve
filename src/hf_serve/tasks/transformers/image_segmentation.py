from typing import List, Literal, Optional, Union

import PIL
from PIL.Image import Image as ImageType
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_serializer

from hf_serve.serde import Image
from hf_serve.tasks.predictor import Predictor


class ImageSegmentationParameters(BaseModel):
    mask_threshold: Optional[float] = None
    overlap_mask_area_threshold: Optional[float] = None
    subtask: Optional[Literal["instance", "panoptic", "semantic"]] = None
    threshold: Optional[float] = None


class ImageSegmentationInput(BaseModel):
    inputs: Union[str, bytes] = Field(validation_alias=AliasChoices("inputs", "image"))
    parameters: Optional[ImageSegmentationParameters] = None

    @field_serializer("inputs")
    @classmethod
    def deserialize_inputs(cls, v: Union[str, bytes]) -> ImageType:
        return Image.deserialize(v)

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        json_schema_extra={
            "examples": [
                {
                    "inputs": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/segmentation_input.jpg",
                    "parameters": {
                        "mask_threshold": 0.5,
                        "overlap_mask_area_threshold": 0.5,
                        "subtask": "semantic",
                        "threshold": 0.9,
                    },
                }
            ]
        },
    )


class ImageSegmentationOutputValue(BaseModel):
    label: str
    mask: str
    score: Optional[float] = None


class ImageSegmentationOutput(BaseModel):
    results: List[ImageSegmentationOutputValue]


class ImageSegmentation(Predictor[ImageSegmentationInput, ImageSegmentationOutput]):
    def __init__(self, model_id: str, dtype: str = "float16", device: str = "auto") -> None:
        super().__init__()

        import torch
        from transformers import pipeline
        from transformers.pipelines.image_segmentation import ImageSegmentationPipeline

        # NOTE: Apparently some (not all) models don't support the `device_map=auto` so we should probably
        # either add a check or just default to CUDA instead
        if device == "auto":
            # e.g. DistilBertForSequenceClassification won't support it
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"

        self.pipeline: ImageSegmentationPipeline = pipeline(
            task="image-segmentation",
            model=model_id,
            torch_dtype=getattr(torch, dtype),
            device=device if device not in {"auto"} else None,
            device_map=device if device in {"auto"} else None,
        )

        if torch.mps.is_available():
            torch.mps.empty_cache()
            torch.mps.set_per_process_memory_fraction(0.9)

    def __call__(self, payload: ImageSegmentationInput) -> ImageSegmentationOutput:
        payload = payload.model_dump(exclude_none=True)  # dumping model to trigger @field_serializer.
        parameters = payload.get("parameters", {})
        image_input = payload.get("inputs")

        results = self.pipeline(image_input, **parameters)

        # Convert masks to base64 strings if they are PIL images
        for result in results:
            if "mask" in result:
                # If mask is a PIL Image, serialize it to base64
                if isinstance(result["mask"], PIL.Image.Image):
                    result["mask"] = Image.serialize(result["mask"])

        return ImageSegmentationOutput(results=results)
