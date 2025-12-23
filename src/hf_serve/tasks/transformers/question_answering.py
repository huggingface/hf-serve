from typing import List, Optional

import torch
from pydantic import BaseModel, ConfigDict, RootModel

from hf_serve.tasks.predictor import Predictor


class QuestionAnsweringInputs(BaseModel):
    context: str
    question: str


class QuestionAnsweringParameters(BaseModel):
    align_to_words: Optional[bool] = None
    doc_stride: Optional[int] = None
    handle_impossible_answer: Optional[bool] = None
    max_answer_len: Optional[int] = None
    max_question_len: Optional[int] = None
    max_seq_len: Optional[int] = None
    top_k: Optional[int] = None


class QuestionAnsweringInput(BaseModel):
    inputs: QuestionAnsweringInputs
    parameters: Optional[QuestionAnsweringParameters] = None

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "inputs": {
                        "question": "Where do I live",
                        "context": "My name is Jan and I live in Paris",
                    },
                    "parameters": {
                        "top_k": 2,
                        "max_seq_len": 124,
                    },
                }
            ]
        }
    )


class QuestionAnsweringOutputValue(BaseModel):
    answer: str
    end: int
    score: float
    start: int


class QuestionAnsweringOutput(RootModel):
    root: QuestionAnsweringOutputValue


class QuestionAnswering(Predictor[QuestionAnsweringInput, QuestionAnsweringOutput]):
    def __init__(
        self,
        model_id: str,
        revision: Optional[str] = None,
        dtype: Optional[str] = None,
        device: str = "auto",
        trust_remote_code: bool = False,
    ) -> None:
        super().__init__()

        from transformers import pipeline
        from transformers.pipelines.question_answering import QuestionAnsweringPipeline

        # NOTE: Apparently some (not all) models don't support the `device_map=auto` so we should probably
        # either add a check or just default to CUDA instead
        if device == "auto":
            # e.g. DistilBertForSequenceClassification won't support it
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"

        self.pipeline: QuestionAnsweringPipeline = pipeline(
            task="question-answering",
            model=model_id,
            revision=revision,
            dtype=getattr(torch, dtype) if dtype is not None else "auto",
            device=device,
            trust_remote_code=trust_remote_code,
        )

        if torch.mps.is_available():
            torch.mps.empty_cache()
            torch.mps.set_per_process_memory_fraction(0.9)

    def __call__(self, payload: QuestionAnsweringInput) -> QuestionAnsweringOutput:
        parameters = {}
        if payload.parameters:
            parameters = payload.parameters.model_dump(exclude_none=True)

        output = self.pipeline(question=payload.inputs.question, context=payload.inputs.context, **parameters)
        return QuestionAnsweringOutput(root=output)  # type: ignore
