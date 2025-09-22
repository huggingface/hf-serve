from typing import Any, Dict, List, Literal, Optional

from pydantic import AliasChoices, AliasPath, BaseModel, ConfigDict, Field, RootModel

from hf_serve.tasks.predictor import Predictor


class SummarizationParameters(BaseModel):
    clean_up_tokenization_spaces: Optional[bool] = None
    truncation: Optional[Literal["do_not_truncate", "longest_first", "only_first", "only_second"]] = None

    generate_parameters: Optional[Dict[str, Any]] = None


class SummarizationInput(BaseModel):
    inputs: str = Field(
        validation_alias=AliasChoices("inputs", AliasPath("text"), AliasPath("inputs", "text")),
    )
    parameters: Optional[SummarizationParameters] = None

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "inputs": "An apple a day, keeps the doctor away",
                    "parameters": {
                        "generate_parameters": {
                            "min_length": 5,
                            "max_length": 20,
                        },
                    },
                }
            ]
        }
    )


class SummarizationOutputValue(BaseModel):
    summary_text: str


class SummarizationOutput(RootModel):
    root: List[SummarizationOutputValue]


class Summarization(Predictor[SummarizationInput, SummarizationOutput]):
    def __init__(self, model_id: str, dtype: Optional[str] = None, device: str = "auto") -> None:
        super().__init__()

        import torch
        from transformers import pipeline
        from transformers.pipelines.text2text_generation import SummarizationPipeline

        # NOTE: Apparently some (not all) models don't support the `device_map=auto` so we should probably
        # either add a check or just default to CUDA instead
        if device == "auto":
            # e.g. DistilBertForSequenceClassification won't support it
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"

        self.pipeline: SummarizationPipeline = pipeline(
            task="summarization",
            model=model_id,
            dtype=getattr(torch, dtype) if dtype is not None else "auto",
            device=device,
        )

        if torch.mps.is_available():
            torch.mps.empty_cache()
            torch.mps.set_per_process_memory_fraction(0.9)

    def __call__(self, payload: SummarizationInput) -> SummarizationOutput:
        parameters = {}
        if payload.parameters:
            parameters = payload.parameters.model_dump(exclude_none=True)
            # NOTE: Given that `generate_parameters` can be provided as a nested dict, then we should flatten
            # that too if applicable
            if "generate_parameters" in parameters and isinstance(
                parameters.get("generate_parameters", None), dict
            ):
                parameters.update(parameters.pop("generate_parameters"))

        output = self.pipeline(payload.inputs, **parameters)
        return SummarizationOutput(root=output)  # type: ignore
