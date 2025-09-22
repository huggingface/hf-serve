from typing import List, Optional

from pydantic import AliasChoices, AliasPath, BaseModel, Field, RootModel

from hf_serve.tasks.predictor import Predictor


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
    def __init__(self, model_id: str, dtype: Optional[str] = None, device: str = "balanced") -> None:
        super().__init__()

        import torch
        from transformers import pipeline
        from transformers.pipelines.text_classification import TextClassificationPipeline

        # NOTE: Apparently some (not all) models don't support the `device_map=auto` so we should probably
        # either add a check or just default to CUDA instead
        if device == "auto":
            # e.g. DistilBertForSequenceClassification won't support it
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"

        self.pipeline: TextClassificationPipeline = pipeline(
            task="text-classification",
            model=model_id,
            dtype=getattr(torch, dtype) if dtype is not None else "auto",
            device=device if device != "auto" else None,
            device_map=device if device == "auto" else None,
        )

        if torch.mps.is_available():
            torch.mps.empty_cache()
            torch.mps.set_per_process_memory_fraction(0.9)

    def __call__(self, payload: TextClassificationInput) -> TextClassificationOutput:
        return TextClassificationOutput(root=self.pipeline(**payload.model_dump()))  # type: ignore
