from typing import Any, Dict, List, Literal, Optional
from huggingface_inference_toolkit.logging import logger

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
                    "inputs": "How old are you",
                    "parameters": {
                        "src_lang": "en",
                        "tgt_lang": "fr",
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

        from transformers import pipeline as transformers_pipeline, AutoModelForSeq2SeqLM, AutoTokenizer  # type: ignore

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"

        # Load model and tokenizer once
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_id,
            torch_dtype=getattr(torch, dtype),
            device_map=device if device in {"auto"} else None,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        available_languages = self.model.config.task_specific_params
        self.translation_pairs = {
            key.replace("translation_", ""): params
            for key, params in available_languages.items()
            if key.startswith("translation_")
        }
        logger.info(f"Available translation pairs: {list(self.translation_pairs.keys())}")

        # Initialize pipelines with shared model and tokenizer
        self.pipelines = {}
        for lang_pair in self.translation_pairs.keys():
            self.pipelines[lang_pair] = transformers_pipeline(
                task=f"translation_{lang_pair}",
                model=self.model,
                tokenizer=self.tokenizer,
                device=device if device not in {"auto"} else None,
            )

        if torch.mps.is_available():
            torch.mps.empty_cache()
            torch.mps.set_per_process_memory_fraction(0.9)

        # Warmup each pipeline
        warmup_input = TranslationInput(**TranslationInput.model_json_schema().get("examples")[0])
        for pipeline in self.pipelines.values():
            _ = pipeline(warmup_input.inputs)

    def __call__(self, input: TranslationInput) -> TranslationOutput:
        parameters = input.parameters

        if parameters and parameters.src_lang and parameters.tgt_lang:
            lang_pair = f"{parameters.src_lang}_to_{parameters.tgt_lang}"
        else:
            lang_pair = next(iter(self.pipelines.keys()))
            logger.info(f"No language pair specified, defaulting to {lang_pair}")

        if lang_pair not in self.pipelines:
            raise ValueError(
                f"Unsupported language pair: {lang_pair}. Available pairs are: {list(self.pipelines.keys())}"
            )

        pipeline_results = self.pipelines[lang_pair](input.inputs)  # type: ignore

        return TranslationOutput(root=pipeline_results)
