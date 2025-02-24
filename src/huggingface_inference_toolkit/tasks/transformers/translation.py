from typing import Any, Dict, List, Literal, Optional

import torch
from pydantic import AliasChoices, AliasPath, BaseModel, ConfigDict, Field, RootModel

from huggingface_inference_toolkit.tasks.predictor import Predictor

TranslationTruncationStrategy = Literal["do_not_truncate", "longest_first", "only_first", "only_second"]


class TranslationParameters(BaseModel):
    clean_up_tokenization_spaces: Optional[bool] = None
    generate_parameters: Optional[Dict[str, Any]] = None
    src_lang: Optional[str] = None
    tgt_lang: Optional[str] = None
    truncation: Optional["TranslationTruncationStrategy"] = None


class TranslationInput(BaseModel):
    inputs: str = Field(
        validation_alias=AliasChoices("inputs", AliasPath("text"), AliasPath("inputs", "text")),
    )
    parameters: Optional[TranslationParameters] = None

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "inputs": "Mona Lisa is located in the [MASK], which is where I was it for the first time",
                    "parameters": {
                        "top_k": 3,
                    },
                }
            ]
        }
    )


class TranslationOutputValue(BaseModel):
    translation_text: str


class TranslationOutput(RootModel):
    root: List[TranslationOutputValue]


class Translation(Predictor[TranslationInput, TranslationOutput]):
    def __init__(self, model_id: str, dtype: str = "float16", device: str = "balanced") -> None:
        super().__init__()

        from transformers import pipeline as transformers_pipeline  # type: ignore

        # apparently some (not all) the models do not support the `device_map=auto` so we should probably
        # either add a check or just default to CUDA instead
        if device == "auto":
            # e.g. DistilBertForSequenceClassification won't support it
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"

        self.pipeline = transformers_pipeline(
            task="translation",
            model=model_id,
            torch_dtype=getattr(torch, dtype),
            device=device if device not in {"auto"} else None,
            device_map=device if device in {"auto"} else None,
        )

        if torch.mps.is_available():
            torch.mps.empty_cache()
            torch.mps.set_per_process_memory_fraction(0.9)

        # first-time "warmup" pass to ensure that the model is ready to start serving requets
        warmup_input = TranslationInput(**TranslationInput.model_json_schema().get("examples")[0])
        _ = self(warmup_input)

    def __call__(self, input: TranslationInput) -> TranslationOutput:
        payload = input.model_dump(exclude_none=True)

        # The HF library has top_k and targets nested in parameters whereas the pipeline expects them flattened
        if "parameters" in payload:
            parameters = payload.pop("parameters") or {}
            payload.update(parameters)

        pipeline_results = self.pipeline(**payload)  # type: ignore
        return TranslationOutput(root=pipeline_results)
