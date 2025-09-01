from typing import Annotated, Dict, List, Literal, Optional

from pydantic import AliasChoices, AliasPath, BaseModel, BeforeValidator, ConfigDict, Field, RootModel

from huggingface_inference_toolkit.tasks.predictor import Predictor


class TableQuestionAnsweringInput(BaseModel):
    question: str = Field(
        validation_alias=AliasChoices("question", AliasPath("inputs", "question"), AliasPath("inputs", "query"))
    )
    table: Dict[str, List[str]] = Field(validation_alias=AliasChoices("table", AliasPath("inputs", "table")))
    padding: Optional[Literal["do_not_pad", "longest", "max_length"]] = Field(
        None, validation_alias=AliasChoices("padding", AliasPath("parameters", "padding"))
    )
    sequential: Optional[bool] = Field(
        None, validation_alias=AliasChoices("sequential", AliasPath("parameters", "sequential"))
    )
    truncation: Optional[bool] = Field(
        None, validation_alias=AliasChoices("truncation", AliasPath("parameters", "truncation"))
    )

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
    def __init__(self, model_id: str, dtype: str = "float16", device: str = "balanced") -> None:
        super().__init__()

        import torch
        from transformers import pipeline
        from transformers.pipelines.table_question_answering import TableQuestionAnsweringPipeline

        # NOTE: Apparently some (not all) models don't support the `device_map=auto` so we should probably
        # either add a check or just default to CUDA instead
        if device == "auto":
            # Many tqa models don't support MPS without CPU fallback
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.pipeline: TableQuestionAnsweringPipeline = pipeline(
            task="table-question-answering",
            model=model_id,
            torch_dtype=getattr(torch, dtype),
            device=device if device not in {"auto"} else None,
            device_map=device if device in {"auto"} else None,
        )

        # first-time "warmup" pass to ensure that the model is ready to start serving requests
        warmup_input = TableQuestionAnsweringInput(
            **TableQuestionAnsweringInput.model_json_schema().get("examples")[0]
        )

        self(warmup_input)

    # TODO: update
    def __call__(self, payload: TableQuestionAnsweringInput) -> TableQuestionAnsweringOutput:
        optional_params = {
            k: v
            for k, v in payload.model_dump().items()
            if k in ["padding", "sequential", "truncation"] and v is not None
        }

        results = self.pipeline(
            **payload.inputs,
            **optional_params,
        )

        return TableQuestionAnsweringOutput(root=results)
