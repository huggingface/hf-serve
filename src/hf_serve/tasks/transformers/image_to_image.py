from typing import TYPE_CHECKING, Optional, Union

from PIL.Image import Image as ImageType
from pydantic import AliasChoices, AliasPath, BaseModel, ConfigDict, Field, RootModel

from hf_serve.serde import Image
from hf_serve.tasks.predictor import Predictor
from hf_serve.types.form import FileForm, FloatForm, IntForm, StrForm


class ImageToImageParameters(BaseModel):
    prompt: Optional[str] = Field(default=None)
    negative_prompt: Optional[str] = Field(default=None)
    num_inference_steps: Optional[int] = Field(default=None)
    guidance_scale: Optional[float] = Field(default=None)

    width: Optional[int] = Field(
        default=None,
        validation_alias=AliasChoices("width", AliasPath("target_size", "width")),
    )
    height: Optional[int] = Field(
        default=None,
        validation_alias=AliasChoices("height", AliasPath("target_size", "height")),
    )


class ImageToImageInput(BaseModel):
    inputs: Union[str, bytes] = Field(validation_alias=AliasChoices("inputs", "image"))
    parameters: Optional[ImageToImageParameters] = None

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "inputs": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/segmentation_input.jpg",
                    "parameters": {
                        "target_size": {"heigth": 768, "width": 768},
                    },
                }
            ]
        },
    )


class ImageToImageFormInput(BaseModel):
    file: FileForm

    prompt: Optional[StrForm] = Field(default=None)
    negative_prompt: Optional[StrForm] = Field(default=None)
    num_inference_steps: Optional[IntForm] = Field(default=None)
    guidance_scale: Optional[FloatForm] = Field(default=None)

    width: Optional[IntForm] = Field(default=None)
    height: Optional[IntForm] = Field(default=None)

    model_config = ConfigDict(extra="forbid")


class ImageToImageOutput(RootModel):
    root: ImageType

    # NOTE: No serialization to bytes here, given that the Hugging Face API expects the `Accept` header to reply
    # with the image rather than the bytes
    model_config = ConfigDict(arbitrary_types_allowed=True)


class ImageToImage(Predictor[ImageToImageInput, ImageToImageOutput]):
    def __init__(self, model_id: str, dtype: Optional[str] = None, device: str = "auto") -> None:
        super().__init__()

        import torch
        from transformers import pipeline

        if TYPE_CHECKING:
            from transformers.pipelines.image_to_image import ImageToImagePipeline

        # NOTE: Apparently some (not all) models don't support the `device_map=auto` so we should probably
        # either add a check or just default to CUDA instead
        if device == "auto":
            # e.g. DistilBertForSequenceClassification won't support it
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"

        self.pipeline: ImageToImagePipeline = pipeline(
            task="image-to-image",
            model=model_id,
            dtype=getattr(torch, dtype) if dtype is not None else "auto",
            device=device,
        )

        if torch.mps.is_available():
            torch.mps.empty_cache()
            torch.mps.set_per_process_memory_fraction(0.9)

    def __call__(self, payload: ImageToImageInput) -> ImageToImageOutput:
        parameters = {}
        if payload.parameters:
            parameters = payload.parameters.model_dump(exclude_none=True)

        output = self.pipeline(Image.deserialize(payload.inputs), **parameters)
        return ImageToImageOutput(root=output)  # type: ignore
