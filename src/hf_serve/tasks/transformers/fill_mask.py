from typing import List, Optional

from pydantic import AliasChoices, AliasPath, BaseModel, ConfigDict, Field, RootModel

from hf_serve.tasks.predictor import Predictor


class FillMaskParameters(BaseModel):
    targets: Optional[List[str]] = None
    top_k: Optional[int] = None


class FillMaskInput(BaseModel):
    inputs: str = Field(
        validation_alias=AliasChoices("inputs", AliasPath("text"), AliasPath("inputs", "text")),
    )
    parameters: Optional[FillMaskParameters] = None

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


class FillMaskOutputValue(BaseModel):
    score: float
    sequence: str
    token: int
    token_str: str  # NOTE: `token_str` was flagged as `Any`, but it should be `str` instead
    fill_mask_output_token_str: Optional[str] = None


class FillMaskOutput(RootModel):
    root: List[FillMaskOutputValue]


class FillMask(Predictor[FillMaskInput, FillMaskOutput]):
    def __init__(
        self, model_id: str, dtype: Optional[str] = None, device: str = "auto", trust_remote_code: bool = False
    ) -> None:
        super().__init__()

        import torch
        from transformers import pipeline
        from transformers.pipelines.fill_mask import FillMaskPipeline

        # NOTE: Apparently some (not all) models don't support the `device_map=auto` so we should probably
        # either add a check or just default to CUDA instead
        if device == "auto":
            # e.g. DistilBertForSequenceClassification won't support it
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"

        self.pipeline: FillMaskPipeline = pipeline(
            task="fill-mask",
            model=model_id,
            dtype=getattr(torch, dtype) if dtype is not None else "auto",
            device=device,
            trust_remote_code=trust_remote_code,
        )

        if torch.mps.is_available():
            torch.mps.empty_cache()
            torch.mps.set_per_process_memory_fraction(0.9)

    def __call__(self, payload: FillMaskInput) -> FillMaskOutput:
        parameters = {}
        if payload.parameters:
            parameters = payload.parameters.model_dump(exclude_none=True)

        output = self.pipeline(payload.inputs, **parameters)
        return FillMaskOutput(root=output)  # type: ignore
