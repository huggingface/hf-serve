from typing import Any, Dict, List, Literal, Optional

from pydantic import AliasChoices, AliasPath, BaseModel, ConfigDict, Field, RootModel

from huggingface_inference_toolkit.tasks.predictor import Predictor

SummarizationTruncationStrategy = Literal["do_not_truncate", "longest_first", "only_first", "only_second"]


class SummarizationParameters(BaseModel):
    clean_up_tokenization_spaces: Optional[bool] = None
    generate_parameters: Optional[Dict[str, Any]] = None
    truncation: Optional["SummarizationTruncationStrategy"] = None


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
    def __init__(self, model_id: str, dtype: str = "float16", device: str = "balanced") -> None:
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
            torch_dtype=getattr(torch, dtype),
            device=device if device not in {"auto"} else None,
            device_map=device if device in {"auto"} else None,
        )

        if torch.mps.is_available():
            torch.mps.empty_cache()
            torch.mps.set_per_process_memory_fraction(0.9)

        # first-time "warmup" pass to ensure that the model is ready to start serving requets
        warmup_input = SummarizationInput(**SummarizationInput.model_json_schema().get("examples")[0])
        _ = self(warmup_input)

    def __call__(self, payload: SummarizationInput) -> SummarizationOutput:
        payload = payload.model_dump(exclude_none=True)  # type: ignore

        if "parameters" in payload:
            parameters = payload.pop("parameters") or {}
            # Extract nested generate_parameters.
            generate_params = parameters.pop("generate_parameters", None)
            # Update payload with other parameters.
            payload.update(parameters)
            # Merge generation parameters directly into the payload instead
            # of nested under 'generate_kwargs'.
            if generate_params:
                payload.update(generate_params)

        inputs = payload.pop("inputs")

        pipeline_results = self.pipeline(inputs, **payload)  # type: ignore
        return SummarizationOutput(root=pipeline_results)
