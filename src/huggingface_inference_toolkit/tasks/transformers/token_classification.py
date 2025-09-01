from typing import List, Literal, Optional

from pydantic import AliasChoices, AliasPath, BaseModel, ConfigDict, Field, RootModel

from huggingface_inference_toolkit.tasks.predictor import Predictor


class TokenClassificationInput(BaseModel):
    inputs: str = Field(
        validation_alias=AliasChoices("inputs", AliasPath("text"), AliasPath("inputs", "text")),
    )
    aggregation_strategy: Optional[Literal["none", "simple", "first", "average", "max"]] = Field(
        None,
        validation_alias=AliasChoices("aggregation_strategy", AliasPath("parameters", "aggregation_strategy")),
    )
    ignore_labels: Optional[List[str]] = Field(
        None, validation_alias=AliasChoices("ignore_labels", AliasPath("parameters", "ignore_labels"))
    )
    stride: Optional[int] = Field(
        None, validation_alias=AliasChoices("stride", AliasPath("parameters", "stride"))
    )

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

        import torch
        from transformers import pipeline
        from transformers.pipelines.token_classification import TokenClassificationPipeline

        # NOTE: Apparently some (not all) models don't support the `device_map=auto` so we should probably
        # either add a check or just default to CUDA instead
        if device == "auto":
            # e.g. DistilBertForSequenceClassification won't support it
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"

        self.pipeline: TokenClassificationPipeline = pipeline(
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
        warmup_input = TokenClassificationInput(
            **TokenClassificationInput.model_json_schema().get("examples")[0]
        )
        self(warmup_input)

    def __call__(self, payload: TokenClassificationInput) -> TokenClassificationOutput:
        results = self.pipeline(**payload.model_dump(exclude_none=True))
        return TokenClassificationOutput(root=results)
