from typing import List

import torch
from pydantic import AliasChoices, AliasPath, BaseModel, Field, RootModel

from huggingface_inference_toolkit.tasks.predictor import Predictor


class TextClassificationInput(BaseModel):
    inputs: str = Field(
        validation_alias=AliasChoices("inputs", AliasPath("text"), AliasPath("inputs", "text")),
    )


class TextClassificationOutputValue(BaseModel):
    label: str
    score: float


class TextClassificationOutput(RootModel):
    root: List[TextClassificationOutputValue]


# TODO: missing AIP_MODE handling i.e. input contains `instances` and output contains `predictions`
class TextClassification(Predictor[TextClassificationInput, TextClassificationOutput]):
    def __init__(self, model_id: str, dtype: str = "float16", device: str = "balanced") -> None:
        super().__init__()

        from transformers import pipeline as transformers_pipeline  # type: ignore

        # apparently some (not all) the models do not support the `device_map=auto` so we should probably
        # either add a check or just default to CUDA instead
        if device == "auto":
            # e.g. DistilBertForSequenceClassification won't support it
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"

        self.pipeline = transformers_pipeline(
            task="text-classification",
            model=model_id,
            torch_dtype=getattr(torch, dtype),
            device=device if device not in {"auto"} else None,
            device_map=device if device in {"auto"} else None,
        )

        if torch.mps.is_available():
            torch.mps.empty_cache()
            torch.mps.set_per_process_memory_fraction(0.9)

        # first-time "warmup" pass to ensure that the model is ready to start serving requets
        _ = self.pipeline(
            "This was a masterpiece. Not completely faithful to the books, but enthralling from beginning to end. Might be my favorite of the three."
        )  # type: ignore

    def __call__(self, input: TextClassificationInput) -> TextClassificationOutput:
        return TextClassificationOutput(root=self.pipeline(**input.model_dump()))  # type: ignore
