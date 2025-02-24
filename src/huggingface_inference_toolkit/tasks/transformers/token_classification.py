from typing import List, Literal, Optional

import torch
from pydantic import AliasChoices, AliasPath, BaseModel, ConfigDict, Field, RootModel

from huggingface_inference_toolkit.tasks.predictor import Predictor

TokenClassificationAggregationStrategy = Literal["none", "simple", "first", "average", "max"]

class TokenClassificationParameters(BaseModel):
    aggregation_strategy: Optional["TokenClassificationAggregationStrategy"] = None
    ignore_labels: Optional[List[str]] = None
    stride: Optional[int] = None


class TokenClassificationInput(BaseModel):
    inputs: str = Field(
        validation_alias=AliasChoices("inputs", AliasPath("text"), AliasPath("inputs", "text")),
    )
    parameters: Optional[TokenClassificationParameters] = None

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "inputs": "Je m'appelle jean-baptiste et je vis à montréal",
                    "parameters": {
                        "aggregation_strategy": "simple",
                    },
                }
            ]
        }
    )


class TokenClassificationOutputValue(BaseModel):
    end: int
    score: float
    start: int
    word: str
    entity: Optional[str] = None
    entity_group: Optional[str] = None

class TokenClassificationOutput(RootModel):
    root: List[TokenClassificationOutputValue]


class TokenClassification(Predictor[TokenClassificationInput, TokenClassificationOutput]):
    def __init__(self, model_id: str, dtype: str = "float16", device: str = "balanced") -> None:
        super().__init__()

        from transformers import pipeline as transformers_pipeline  # type: ignore

        # apparently some (not all) the models do not support the `device_map=auto` so we should probably
        # either add a check or just default to CUDA instead
        if device == "auto":
            # e.g. DistilBertForSequenceClassification won't support it
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"

        self.pipeline = transformers_pipeline(
            task="token-classification",
            model=model_id,
            torch_dtype=getattr(torch, dtype),
            device=device if device not in {"auto"} else None,
            device_map=device if device in {"auto"} else None,
        )

        if torch.mps.is_available():
            torch.mps.empty_cache()
            torch.mps.set_per_process_memory_fraction(0.9)

        # first-time "warmup" pass to ensure that the model is ready to start serving requets
        warmup_input = TokenClassificationInput(**TokenClassificationInput.model_json_schema().get("examples")[0])
        _ = self(warmup_input)

    def __call__(self, input: TokenClassificationInput) -> TokenClassificationOutput:
        payload = input.model_dump(exclude_none=True)

        # The HF library has top_k and targets nested in parameters whereas the pipeline expects them flattened
        if "parameters" in payload:
            parameters = payload.pop("parameters") or {}
            payload.update(parameters)

        pipeline_results = self.pipeline(**payload)  # type: ignore
        return TokenClassificationOutput(root=pipeline_results)
