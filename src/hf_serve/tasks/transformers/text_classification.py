from typing import List, Literal, Optional

from pydantic import AliasChoices, AliasPath, BaseModel, ConfigDict, Field, RootModel

from hf_serve.tasks.predictor import Predictor


class TextClassificationParameters(BaseModel):
    top_k: Optional[int] = None
    function_to_apply: Optional[Literal["sigmoid", "softmax", "none"]] = None


class TextClassificationInput(BaseModel):
    inputs: str = Field(
        validation_alias=AliasChoices("inputs", AliasPath("text"), AliasPath("inputs", "text")),
    )
    parameters: Optional[TextClassificationParameters] = None

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "inputs": "What is the capital of France? Paris is the capital of France.",
                    "parameters": {
                        "top_k": 2,
                        "function_to_apply": "softmax",
                    },
                }
            ]
        }
    )


class TextClassificationOutputValue(BaseModel):
    label: str
    score: float


class TextClassificationOutput(RootModel):
    root: List[TextClassificationOutputValue]


class TextClassification(Predictor[TextClassificationInput, TextClassificationOutput]):
    def __init__(self, model_id: str, dtype: Optional[str] = None, device: str = "auto") -> None:
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
            device=device,
        )

        if torch.mps.is_available():
            torch.mps.empty_cache()
            torch.mps.set_per_process_memory_fraction(0.9)

    def __call__(self, payload: TextClassificationInput) -> TextClassificationOutput:
        parameters = {}
        if payload.parameters:
            parameters = payload.parameters.model_dump(exclude_none=True)

        output = self.pipeline(payload.inputs, **parameters)
        return TextClassificationOutput(root=output)  # type: ignore
