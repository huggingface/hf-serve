from typing import Optional

from pydantic import AliasChoices, AliasPath, BaseModel, Field

from hf_serve.tasks.predictor import Predictor


class TextGenerationParameters(BaseModel):
    adapter_id: Optional[str] = Field(default=None)
    do_sample: Optional[bool] = Field(default=True)
    grammar: Optional[str] = Field(default=None)
    max_new_tokens: Optional[int] = Field(default=20)
    repetition_penalty: Optional[float] = Field(default=1.0)
    return_full_text: Optional[bool] = Field(default=False)
    seed: Optional[int] = Field(default=None)
    stop: Optional[list[str]] = Field(default=None)
    temperature: Optional[float] = Field(default=1.0)
    top_k: Optional[int] = Field(default=None)
    top_n_tokens: Optional[int] = Field(default=None)
    top_p: Optional[float] = Field(default=1.0)
    truncate: Optional[int] = Field(default=None)
    typical_p: Optional[float] = Field(default=1.0)


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

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"

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

        # NOTE: `generate_parameters` should be passed as separate argument to the pipeline
        # according to the Transformers documentation
        # Reference: https://huggingface.co/docs/transformers/en/main_classes/pipelines#transformers.TextGenerationPipeline
        # Reference: https://huggingface.co/docs/transformers/en/main_classes/text_generation#transformers.GenerationConfig
        generated_text = self.pipeline(payload.inputs, **parameters)[0]["generated_text"]
        return TextGenerationOutput(generated_text=generated_text)
