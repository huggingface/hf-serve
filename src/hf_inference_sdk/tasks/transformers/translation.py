from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, RootModel

from hf_inference_sdk.logging import logger
from hf_inference_sdk.tasks.predictor import Predictor


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
    def __init__(
        self,
        model_id: str,
        revision: Optional[str] = None,
        dtype: Optional[str] = None,
        device: str = "auto",
        trust_remote_code: bool = False,
    ) -> None:
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
                revision=revision,
                dtype=getattr(torch, dtype) if dtype is not None else "auto",
                device=device,
                trust_remote_code=trust_remote_code,
            )
        except TypeError as e:
            # NOTE: Some models won't support the `device` argument as e.g. `facebook/nllb-200-3.3B`, which will
            # fail with the following exception:
            # TypeError: M2M100ForConditionalGeneration.__init__() got an unexpected keyword argument 'device'
            if str(e).__contains__(".__init__() got an unexpected keyword argument 'device'"):
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_id,
                    revision=revision,
                    dtype=getattr(torch, dtype) if dtype is not None else "auto",
                    trust_remote_code=trust_remote_code,
                )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, revision=revision, trust_remote_code=trust_remote_code
        )
        self.pairs = {}

        # NOTE: Single pair i.e. the model is fine-tuned for a single translation pair
        if (target_lang := self.tokenizer.init_kwargs.get("target_lang", None)) and (
            source_lang := self.tokenizer.init_kwargs.get("source_lang", None)
        ):
            self.pairs = {f"{source_lang}_to_{target_lang}"}

        # NOTE: The model is fine-tuned for multiple translation pairs
        if languages := getattr(self.model.config, "task_specific_params", None):
            if languages is None or not isinstance(languages, dict):
                raise ValueError(
                    "`task_specific_params` with the available `translation_` pairs is not defined in the `config.json`, hence the available languages cannot be inferred."
                )

            self.pairs = {
                key.replace("translation_", "")
                for key, _ in languages.items()
                if key.startswith("translation_")
            }

        if not self.pairs or self.pairs == {}:
            raise ValueError(
                "Translation pairs were not found neither on the `tokenizer_config.json` nor on the `config.json`, meaning that the pipeline might not be suited for `translation` tasks, but rather a model fine-tuned for another task but capable of performing `translation`."
            )

        logger.info(f"Translation pairs (formatted as `<source>_to_<target>`): {self.pairs}")

        self.pipelines: Dict[str, TranslationPipeline] = {}
        for pair in self.pairs:
            self.pipelines[pair] = pipeline(
                task=f"translation_{pair}",  # type: ignore
                model=self.model,
                tokenizer=self.tokenizer,
                device=device,
                trust_remote_code=trust_remote_code,
            )

        if torch.mps.is_available():
            torch.mps.empty_cache()
            torch.mps.set_per_process_memory_fraction(0.9)

    def __call__(self, payload: TranslationInput) -> TranslationOutput:
        parameters = {}
        if payload.parameters:
            parameters = payload.parameters.model_dump(exclude_none=True)

        pair = None
        if (src_lang := parameters.get("src_lang", None)) and (tgt_lang := parameters.get("tgt_lang", None)):
            pair = f"{src_lang}_to_{tgt_lang}"

            if pair not in self.pipelines:
                raise ValueError(
                    f"The translation pair {pair} is not supported. Available translation pairs are (formatted as `<source>_to_<target>`): {self.pairs}"
                )

        if pair is None:
            pair = next(iter(self.pairs))
            logger.warning(
                f"Given that no translation pair has been specified, {pair} (formatted as `<source>_to_<target>`) will be used."
            )

        # NOTE: In this point we already know that the `pair` does indeed exist within the `pipelines` meaning
        # that there's no need to explore the `None` path
        pipeline = self.pipelines.get(pair, None)
        output = pipeline(payload.inputs, **parameters)  # type: ignore
        return TranslationOutput(root=output)  # type: ignore
