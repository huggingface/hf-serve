from typing import Dict, List, Literal, Optional

import torch
from pydantic import AliasChoices, AliasPath, BaseModel, ConfigDict, Field, RootModel
import pandas as pd

from huggingface_inference_toolkit.tasks.predictor import Predictor

Padding = Literal["do_not_pad", "longest", "max_length"]


class TableQuestionAnsweringInputData(BaseModel):
    question: str = Field(
        validation_alias=AliasChoices("question", AliasPath("query"), AliasPath("question", "query")),
    )
    table: Dict[str, List[str]]


class QuestionAnsweringParameters(BaseModel):
    padding: Optional["Padding"] = None
    sequential: Optional[bool] = None
    truncation: Optional[bool] = None


class TableQuestionAnsweringInput(BaseModel):
    inputs: TableQuestionAnsweringInputData
    parameters: Optional[QuestionAnsweringParameters] = None

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "inputs": {
                        "table": {
                            "Repository": ["Transformers", "Datasets", "Tokenizers"],
                            "Stars": ["36542", "4512", "3934"],
                            "Contributors": ["651", "77", "34"],
                            "Programming language": ["Python", "Python", "Rust, Python and NodeJS"],
                        },
                        "query": "Which repo has the most contributors?",
                    },
                    "parameters": {"padding": "longest"},
                }
            ]
        }
    )


class QuestionAnsweringOutputValue(BaseModel):
    answer: str
    cells: List[str]
    coordinates: List[List[int]]
    aggregator: Optional[str] = None


class TableQuestionAnsweringOutput(RootModel):
    root: List[QuestionAnsweringOutputValue]


class TableQuestionAnswering(Predictor[TableQuestionAnsweringInput, TableQuestionAnsweringOutput]):
    def __init__(self, model_id: str, dtype: str = "float16", device: str = "balanced") -> None:
        super().__init__()

        from transformers import pipeline as transformers_pipeline  # type: ignore

        # apparently some (not all) the models do not support the `device_map=auto` so we should probably
        # either add a check or just default to CUDA instead
        if device == "auto":
            # Many tqa models don't support MPS without CPU fallback
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.pipeline = transformers_pipeline(
            task="table-question-answering",
            model=model_id,
            torch_dtype=getattr(torch, dtype),
            device=device if device not in {"auto"} else None,
            device_map=device if device in {"auto"} else None,
        )

        # first-time "warmup" pass to ensure that the model is ready to start serving requets
        warmup_input = TableQuestionAnsweringInput(
            **TableQuestionAnsweringInput.model_json_schema().get("examples")[0]
        )

        _ = self(warmup_input)

    def __call__(self, input: TableQuestionAnsweringInput) -> TableQuestionAnsweringOutput:
        payload = input.model_dump(exclude_none=True)

        # Flatten the inputs dictionary into the payload
        if "inputs" in payload:
            inputs = payload.pop("inputs") or {}

            # Convert table format using pandas and create the expected format
            if "table" in inputs:
                table_data = inputs.pop("table")
                if isinstance(table_data, dict):
                    df = pd.DataFrame(table_data)
                    table_data = df.to_dict("records")

                # Get the question/query
                question = inputs.get("question", "")

                # Create the expected format - a list with a single dict containing table and query
                payload = [{"table": table_data, "query": question}]

        # The parameters should be passed separately
        if "parameters" in payload:
            parameters = payload.pop("parameters") or {}
            pipeline_results = self.pipeline(payload, **parameters)  # type: ignore
        else:
            pipeline_results = self.pipeline(payload)  # type: ignore

        # Make to a list if only outputs one QuestionAnsweringOutputValue
        if not isinstance(pipeline_results, list):
            pipeline_results = [pipeline_results]

        return TableQuestionAnsweringOutput(root=pipeline_results)
