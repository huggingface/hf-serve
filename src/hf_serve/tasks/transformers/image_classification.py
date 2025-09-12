from typing import Annotated, List, Literal, Optional, Union

from fastapi import Form
from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from hf_serve.serde import Image
from hf_serve.tasks.predictor import Predictor
from hf_serve.types import FileForm, IntForm


class ImageClassificationParameters(BaseModel):
    function_to_apply: Optional[Literal["sigmoid", "softmax", None]] = Field(default=None)
    top_k: Optional[int] = Field(default=None)


class ImageClassificationInput(BaseModel):
    inputs: Union[str, bytes] = Field(validation_alias=AliasChoices("inputs", "image"))
    parameters: Optional[ImageClassificationParameters] = Field(default=None)

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
        },
    )


class ImageClassificationFormInput(BaseModel):
    file: FileForm
    function_to_apply: Optional[Annotated[Literal["sigmoid", "softmax", None], Form()]] = None
    top_k: Optional[IntForm] = None

    model_config = ConfigDict(extra="forbid")


class ImageClassificationOutputValue(BaseModel):
    label: str
    score: float


class ImageClassificationOutput(BaseModel):
    results: List[ImageClassificationOutputValue]


class ImageClassification(Predictor[ImageClassificationInput, ImageClassificationOutput]):
    def __init__(self, model_id: str, dtype: str = "float16", device: str = "auto") -> None:
        super().__init__()

        import torch
        from transformers import pipeline
        from transformers.pipelines.image_classification import ImageClassificationPipeline

        # NOTE: Apparently some (not all) models don't support the `device_map=auto` so we should probably
        # either add a check or just default to CUDA instead
        if device == "auto":
            # e.g. DistilBertForSequenceClassification won't support it
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"

        self.pipeline: ImageClassificationPipeline = pipeline(
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

        image_input = Image.deserialize(payload.inputs)

        results = self.pipeline(image_input, **parameters)

        return ImageClassificationOutput(results=results)
