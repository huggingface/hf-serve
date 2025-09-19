from typing import Annotated, List, Optional, Union

from fastapi import Form
from pydantic import BaseModel, ConfigDict, Field, RootModel, conint

from hf_serve.serde import Image
from hf_serve.tasks.predictor import Predictor
from hf_serve.types import FileForm


class VisualQuestionAnsweringInputs(BaseModel):
    image: Union[str, bytes]
    question: str


class VisualQuestionAnsweringParameters(BaseModel):
    top_k: Optional[Annotated[int, conint(ge=0)]] = Field(default=1)


class VisualQuestionAnsweringInput(BaseModel):
    inputs: VisualQuestionAnsweringInputs
    parameters: Optional[VisualQuestionAnsweringParameters] = Field(default=None)

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "inputs": {
                        "image": "https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png",
                        "question": "What is in the image?",
                    },
                    "parameters": {
                        "top_k": 3,
                    },
                }
            ]
        },
    )


class VisualQuestionAnsweringFormInput(BaseModel):
    file: FileForm
    question: Annotated[str, Form()]  # NOTE: The `predict_media` router will include this as a parameter

    top_k: Optional[Annotated[int, conint(ge=0), Form()]] = Field(default=1)

    model_config = ConfigDict(extra="forbid")


class VisualQuestionAnsweringOutputValue(BaseModel):
    label: str
    score: float


class VisualQuestionAnsweringOutput(RootModel):
    root: List[VisualQuestionAnsweringOutputValue]


class VisualQuestionAnswering(Predictor[VisualQuestionAnsweringInput, VisualQuestionAnsweringOutput]):
    def __init__(self, model_id: str, dtype: str = "float16", device: str = "auto") -> None:
        super().__init__()

        import torch
        from transformers import pipeline
        from transformers.pipelines.visual_question_answering import VisualQuestionAnsweringPipeline

        # NOTE: Apparently some (not all) models don't support the `device_map=auto` so we should probably
        # either add a check or just default to CUDA instead
        if device == "auto":
            # e.g. DistilBertForSequenceClassification won't support it
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"

        self.pipeline: VisualQuestionAnsweringPipeline = pipeline(
            task="visual-question-answering",
            model=model_id,
            dtype=getattr(torch, dtype),
            device=device if device not in {"auto"} else None,
            device_map=device if device in {"auto"} else None,
        )

        if torch.mps.is_available():
            torch.mps.empty_cache()
            torch.mps.set_per_process_memory_fraction(0.9)

    def __call__(self, payload: VisualQuestionAnsweringInput) -> VisualQuestionAnsweringOutput:
        parameters = {}
        if payload.parameters:
            parameters = payload.parameters.model_dump(exclude_none=True)

        output = self.pipeline(
            image=Image.deserialize(payload.inputs.image), question=payload.inputs.question, **parameters
        )
        return VisualQuestionAnsweringOutput(root=output)  # type: ignore
