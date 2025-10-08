from typing import Optional, Self

from pydantic import BaseModel, Field, model_validator

from hf_serve.logging import logger
from hf_serve.tasks.predictor import Predictor


class ImageTextToTextParameters(BaseModel):
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
        if any(getattr(self, p) for p in {"adapter_id", "grammar", "stop", "top_n_tokens", "truncate"}):
            logger.warning(
                "Unsupported parameters will be ignored: adapter_id, grammar, stop, top_n_tokens, truncate"
            )
        return self


class ImageTextToTextInputs(BaseModel):
    text: str
    image: str


class ImageTextToTextInput(BaseModel):
    inputs: ImageTextToTextInputs
    parameters: Optional[ImageTextToTextParameters] = Field(default=None)


class ImageTextToTextOutput(BaseModel):
    generated_text: str


class ImageTextToText(Predictor[ImageTextToTextInput, ImageTextToTextOutput]):
    def __init__(self, model_id: str, dtype: Optional[str] = None, device: str = "auto") -> None:
        super().__init__()

        import torch
        from transformers import pipeline
        from transformers.pipelines.image_text_to_text import ImageTextToTextPipeline

        self.pipeline: ImageTextToTextPipeline = pipeline(
            task="image-text-to-text",
            model=model_id,
            dtype=getattr(torch, dtype) if dtype is not None else "auto",
            device=device if device != "auto" else None,
            device_map=device if device == "auto" else None,
        )

        if torch.mps.is_available():
            torch.mps.empty_cache()
            torch.mps.set_per_process_memory_fraction(0.9)

    def __call__(self, payload: ImageTextToTextInput) -> ImageTextToTextOutput:
        parameters = {}
        if payload.parameters:
            parameters = payload.parameters.model_dump(exclude_none=True)

        if seed := parameters.pop("seed", None):
            from transformers import set_seed

            set_seed(seed)

        # NOTE: Removing these here intead of within the schema as otherwise when logging the schema the user
        # might be confused if they see that the schema is different to what they provided despite the warning
        for p in {"adapter_id", "grammar", "stop", "top_n_tokens", "truncate"}:
            parameters.pop(p, None)

        output = self.pipeline(image=payload.inputs.image, text=payload.inputs.text, **parameters)
        generated_text = output[0]["generated_text"]
        return ImageTextToTextOutput(generated_text=generated_text)
