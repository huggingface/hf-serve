from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, RootModel

from hf_serve.logging import logger
from hf_serve.tasks.predictor import Predictor


class TranslationParameters(BaseModel):
    clean_up_tokenization_spaces: Optional[bool] = Field(default=None)
    src_lang: Optional[str] = Field(default=None)
    tgt_lang: Optional[str] = Field(default=None)
    truncation: Optional[Literal["do_not_truncate", "longest_first", "only_first", "only_second"]] = Field(
        default=None
    )

    generate_parameters: Optional[Dict[str, Any]] = Field(default=None)


class TranslationInput(BaseModel):
    inputs: str
    parameters: Optional[TranslationParameters] = Field(default=None)

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
    def __init__(self, model_id: str, dtype: Optional[str] = None, device: str = "auto") -> None:
        super().__init__()

        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
        from transformers.pipelines.text2text_generation import TranslationPipeline

        # NOTE: Apparently some (not all) models don't support the `device_map=auto` so we should probably
        # either add a check or just default to CUDA instead
        if device == "auto":
            # e.g. DistilBertForSequenceClassification won't support it
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"

        try:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_id,
                dtype=getattr(torch, dtype) if dtype is not None else "auto",
                device=device,
            )
        except TypeError as e:
            # NOTE: Some models won't support the `device` argument as e.g. `facebook/nllb-200-3.3B`, which will
            # fail with the following exception:
            # TypeError: M2M100ForConditionalGeneration.__init__() got an unexpected keyword argument 'device'
            if str(e).__contains__(".__init__() got an unexpected keyword argument 'device'"):
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_id,
                    dtype=getattr(torch, dtype) if dtype is not None else "auto",
                )

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        # TODO(alvarobartt): Given that I didn't implement this myself, I'll check to make sure that this is the
        # only way of pulling the available language pairs as it feels odd
        languages = self.model.config.task_specific_params
        if languages is None or not isinstance(languages, dict):
            raise ValueError(
                "`task_specific_params` with the available `translation_` pairs is not defined in the `config.json`, hence the available languages cannot be inferred."
            )

        self.translation_pairs = {
            key.replace("translation_", ""): params
            for key, params in languages.items()
            if key.startswith("translation_")
        }
        logger.info(f"Available translation pairs: {list(self.translation_pairs.keys())}")

        self.pipelines: Dict[str, TranslationPipeline] = {}
        for lang_pair in self.translation_pairs.keys():
            self.pipelines[lang_pair] = pipeline(
                task=f"translation_{lang_pair}",  # type: ignore
                model=self.model,
                tokenizer=self.tokenizer,
                device=device,
            )

        if torch.mps.is_available():
            torch.mps.empty_cache()
            torch.mps.set_per_process_memory_fraction(0.9)

    def __call__(self, payload: TranslationInput) -> TranslationOutput:
        parameters = {}
        if payload.parameters:
            parameters = payload.parameters.model_dump(exclude_none=True)

        if (src_lang := parameters.get("src_lang", None)) and (tgt_lang := parameters.get("tgt_lang", None)):
            lang_pair = f"{src_lang}_to_{tgt_lang}"
            logger.info(f"language pair specified {lang_pair}")

            if lang_pair not in self.pipelines:
                raise ValueError(
                    f"Unsupported language pair: {lang_pair}. Available pairs are: {list(self.pipelines.keys())}"
                )
        else:
            lang_pair = next(iter(self.pipelines.keys()))
            logger.info(f"No language pair specified, defaulting to {lang_pair}")

        output = self.pipelines[lang_pair](payload.inputs, **parameters)
        return TranslationOutput(root=output)  # type: ignore
