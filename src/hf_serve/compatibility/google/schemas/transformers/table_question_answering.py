from typing import Annotated, List, Optional

from annotated_types import Len
from pydantic import BaseModel, Field

from hf_serve.tasks.transformers.table_question_answering import (
    TableQuestionAnsweringInputs,
    TableQuestionAnsweringOutput,
    TableQuestionAnsweringParameters,
)


class TableQuestionAnsweringInputForGoogle(BaseModel):
    instances: Annotated[List[TableQuestionAnsweringInputs], Len(min_length=1)]
    parameters: Optional[TableQuestionAnsweringParameters] = Field(default=None)


class TableQuestionAnsweringOutputForGoogle(BaseModel):
    predictions: List[TableQuestionAnsweringOutput]
