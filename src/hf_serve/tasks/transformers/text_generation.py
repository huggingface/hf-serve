from typing import Optional, Self

from pydantic import AliasChoices, AliasPath, BaseModel, Field, model_validator

from hf_serve.logging import logger
from hf_serve.tasks.predictor import Predictor


class TextGenerationParameters(BaseModel):
    do_sample: Optional[bool] = Field(default=True)
    max_new_tokens: Optional[int] = Field(default=20)
    repetition_penalty: Optional[float] = Field(default=1.0)
    return_full_text: Optional[bool] = Field(default=False)
    seed: Optional[int] = Field(default=None)
    temperature: Optional[float] = Field(default=1.0)
    top_k: Optional[int] = Field(default=None)
    top_p: Optional[float] = Field(default=1.0)
    typical_p: Optional[float] = Field(default=1.0)

    # NOTE: All the parameters below are defined within the Inference API Specification but not supported within
    # `hf-serve`, hence allowing those but raising a warning if those are provided
    adapter_id: Optional[str] = Field(default=None)
    grammar: Optional[str] = Field(default=None)
    stop: Optional[list[str]] = Field(default=None)
    top_n_tokens: Optional[int] = Field(default=None)
    truncate: Optional[int] = Field(default=None)

    @model_validator(mode="after")
    def validate_unsupported_params(self: Self) -> Self:
        if any(getattr(self, p) for p in {"adapter_id", "grammar", "top_n_tokens", "truncate"}):
            logger.warning(
                "Unsupported parameters will be ignored: adapter_id, grammar, top_n_tokens, truncate"
            )
        return self


class TextGenerationInput(BaseModel):
    inputs: str = Field(validation_alias=AliasChoices("inputs", AliasPath("text")))
    parameters: Optional[TextGenerationParameters] = Field(default=None)


class TextGenerationOutput(BaseModel):
    generated_text: str


class TextGeneration(Predictor[TextGenerationInput, TextGenerationOutput]):
    def __init__(self, model_id: str, dtype: Optional[str] = None, device: str = "auto") -> None:
        super().__init__()

        import torch
        from transformers import pipeline
        from transformers.pipelines.text_generation import TextGenerationPipeline

        self.pipeline: TextGenerationPipeline = pipeline(
            task="text-generation",
            model=model_id,
            dtype=getattr(torch, dtype) if dtype is not None else "auto",
            device=device if device != "auto" else None,
            device_map=device if device == "auto" else None,
        )

        if torch.mps.is_available():
            torch.mps.empty_cache()
            torch.mps.set_per_process_memory_fraction(0.9)

    def __call__(self, payload: TextGenerationInput) -> TextGenerationOutput:
        parameters = {}
        if payload.parameters:
            parameters = payload.parameters.model_dump(exclude_none=True)

        if seed := parameters.pop("seed", None):
            from transformers import set_seed

            set_seed(seed)

        # NOTE: We need to capture the `generate_kwargs` manually, given that the I/O interface for `text-generation`
        # on the Inference API is aligned with Text Generation Inference (TGI) not with Transformers, meaning that
        # there might be some subtle differences on how the parameters are handled. In any case, this should be a
        # discussion point anytime soon, given that Transformers offers much more paremeters during the forward pass
        # than the ones captured in the input schemas, meaning that we're "losing" some capabilities as of today
        # when using `hf-serve` for `text-generation` via the Inference API.
        generate_kwargs = {}

        # NOTE: https://huggingface.co/docs/transformers/v4.57.3/en/main_classes/text_generation#transformers.GenerationConfig.stop_strings
        if stop_strings := parameters.pop("stop", None):
            generate_kwargs["stop_strings"] = stop_strings

        # NOTE: Removing these here instead of within the schema as otherwise when logging the schema the user
        # might be confused if they see that the schema is different to what they provided despite the warning
        for p in {"adapter_id", "grammar", "top_n_tokens", "truncate"}:
            logger.warning(
                f"`{p}={parameters.get(p)}` was provided, which is valid as per the Inference Endpoints API Specification, but given that's a parameter only supported on Text Generation Inference (TGI), it will be skipped!"
            )
            parameters.pop(p, None)

        output = self.pipeline(text_inputs=payload.inputs, generate_kwargs=generate_kwargs, **parameters)
        generated_text = output[0]["generated_text"]
        return TextGenerationOutput(generated_text=generated_text)
