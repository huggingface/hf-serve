from typing import Annotated, List, Literal, Optional, Union

from fastapi import Form
from PIL.Image import Image as ImageType
from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from hf_inference_sdk.serde import Image
from hf_inference_sdk.tasks.predictor import Predictor
from hf_inference_sdk.types.form import FileForm, FloatForm


class ImageSegmentationParameters(BaseModel):
    mask_threshold: Optional[float] = None
    overlap_mask_area_threshold: Optional[float] = None
    subtask: Optional[Literal["instance", "panoptic", "semantic"]] = None
    threshold: Optional[float] = None


class ImageSegmentationInput(BaseModel):
    inputs: Union[str, bytes] = Field(validation_alias=AliasChoices("inputs", "image"))
    parameters: Optional[ImageSegmentationParameters] = None

    model_config = ConfigDict(
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


class ImageSegmentationFormInput(BaseModel):
    file: FileForm

    mask_threshold: Optional[FloatForm] = None
    overlap_mask_area_threshold: Optional[FloatForm] = None
    subtask: Optional[Annotated[Literal["instance", "panoptic", "semantic"], Form()]] = None
    threshold: Optional[FloatForm] = None

    model_config = ConfigDict(extra="forbid")


class ImageSegmentationOutputValue(BaseModel):
    label: str
    mask: str
    score: Optional[float] = None


class ImageSegmentationOutput(BaseModel):
    results: List[ImageSegmentationOutputValue]


class ImageSegmentation(Predictor[ImageSegmentationInput, ImageSegmentationOutput]):
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
        from transformers.pipelines.image_segmentation import ImageSegmentationPipeline

        # NOTE: Apparently some (not all) models don't support the `device_map=auto` so we should probably
        # either add a check or just default to CUDA instead
        if device == "auto":
            # e.g. DistilBertForSequenceClassification won't support it
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"

        self.pipeline: ImageSegmentationPipeline = pipeline(
            task="image-segmentation",
            model=model_id,
            revision=revision,
            dtype=getattr(torch, dtype) if dtype is not None else "auto",
            device=device,
            trust_remote_code=trust_remote_code,
        )

        if torch.mps.is_available():
            torch.mps.empty_cache()
            torch.mps.set_per_process_memory_fraction(0.9)

    def __call__(self, payload: ImageSegmentationInput) -> ImageSegmentationOutput:
        parameters = {}
        if payload.parameters:
            parameters = payload.parameters.model_dump(exclude_none=True)

        results = self.pipeline(Image.deserialize(payload.inputs), **parameters)

        # Convert masks to base64 strings if they are PIL images
        for result in results:
            if "mask" in result:
                # If mask is a PIL Image, serialize it to base64
                if isinstance(result["mask"], ImageType):
                    result["mask"] = Image.serialize(image=result["mask"])

        return ImageSegmentationOutput(results=results)  # type: ignore
