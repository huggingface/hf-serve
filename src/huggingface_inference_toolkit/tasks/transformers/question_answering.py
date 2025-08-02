import os
from pathlib import Path
from typing import List, Optional, Union

import torch
from pydantic import BaseModel, ConfigDict, RootModel

from huggingface_inference_toolkit.tasks.predictor import Predictor


class QuestionAnsweringInputData(BaseModel):
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
    inputs: QuestionAnsweringInputData
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
    root: List[QuestionAnsweringOutputValue]


class QuestionAnswering(Predictor[QuestionAnsweringInput, QuestionAnsweringOutput]):
    def __init__(self, model_id: str, dtype: str = "float16", device: str = "balanced") -> None:
        super().__init__()

        from transformers import pipeline as transformers_pipeline  # type: ignore

        # apparently some (not all) the models do not support the `device_map=auto` so we should probably
        # either add a check or just default to CUDA instead
        if device == "auto":
            # e.g. DistilBertForSequenceClassification won't support it
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"

        self.pipeline = transformers_pipeline(
            task="question-answering",
            model=model_id,
            torch_dtype=getattr(torch, dtype),
            device=device if device not in {"auto"} else None,
            device_map=device if device in {"auto"} else None,
        )

        if torch.mps.is_available():
            torch.mps.empty_cache()
            torch.mps.set_per_process_memory_fraction(0.9)

        # first-time "warmup" pass to ensure that the model is ready to start serving requets
        warmup_input = QuestionAnsweringInput(**QuestionAnsweringInput.model_json_schema().get("examples")[0])
        _ = self(warmup_input)

    @property
    def model_id(self) -> Union[str, None]:
        return (
            self.pipeline.model.config._name_or_path
            if not Path(self.pipeline.model.config._name_or_path).exists()
            else os.getenv("MODEL_ID")
        )

    def __call__(self, input: QuestionAnsweringInput) -> QuestionAnsweringOutput:
        payload = input.model_dump(exclude_none=True)

        # Flatten the inputs dictionary into the payload
        if "inputs" in payload:
            inputs = payload.pop("inputs") or {}
            payload.update(inputs)

        # The HF library has top_k and other params nested in parameters whereas the pipeline expects them flattened
        if "parameters" in payload:
            parameters = payload.pop("parameters") or {}
            payload.update(parameters)

        pipeline_results = self.pipeline(**payload)  # type: ignore
        return QuestionAnsweringOutput(root=pipeline_results)
