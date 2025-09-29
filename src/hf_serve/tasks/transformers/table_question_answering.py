from typing import Annotated, Dict, List, Literal, Optional

from pydantic import AliasChoices, BaseModel, BeforeValidator, ConfigDict, Field, RootModel

from hf_serve.tasks.predictor import Predictor


class TableQuestionAnsweringParameters(BaseModel):
    padding: Optional[Literal["do_not_pad", "longest", "max_length"]] = Field(default=None)
    sequential: Optional[bool] = Field(default=None)
    truncation: Optional[bool] = Field(default=None)


class TableQuestionAnsweringInputs(BaseModel):
    query: str = Field(validation_alias=AliasChoices("query", "question"))
    table: Dict[str, List[str]]


class TableQuestionAnsweringInput(BaseModel):
    inputs: TableQuestionAnsweringInputs
    parameters: Optional[TableQuestionAnsweringParameters] = Field(default=None)

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "inputs": {
                        "table": {
                            "Repository": ["Transformers", "Datasets", "Tokenizers"],
                            "Stars": ["36542", "4512", "3934"],
                            "Contributors": ["651", "77", "34"],
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
    root: Annotated[
        List[QuestionAnsweringOutputValue],
        BeforeValidator(lambda value: [value] if not isinstance(value, list) else value),
    ]


class TableQuestionAnswering(Predictor[TableQuestionAnsweringInput, TableQuestionAnsweringOutput]):
    def __init__(self, model_id: str, dtype: Optional[str] = None, device: str = "auto") -> None:
        super().__init__()

        import torch
        from transformers import pipeline
        from transformers.pipelines.table_question_answering import TableQuestionAnsweringPipeline

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"

        # NOTE: Many `table-question-answering` models don't support MPS without CPU fallback
        if device == "mps":
            device = "cpu"

        self.pipeline: TableQuestionAnsweringPipeline = pipeline(
            task="table-question-answering",
            model=model_id,
            dtype=getattr(torch, dtype) if dtype is not None else "auto",
            device=device,
        )

    def __call__(self, payload: TableQuestionAnsweringInput) -> TableQuestionAnsweringOutput:
        parameters = {}
        if payload.parameters:
            parameters = payload.parameters.model_dump(exclude_none=True)

        output = self.pipeline(table=payload.inputs.table, query=payload.inputs.query, **parameters)
        return TableQuestionAnsweringOutput(root=output)  # type: ignore
