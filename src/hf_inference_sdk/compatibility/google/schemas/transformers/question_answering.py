from typing import Annotated, List, Optional

from annotated_types import Len
from pydantic import BaseModel, Field

from hf_inference_sdk.tasks.transformers.question_answering import (
    QuestionAnsweringInputs,
    QuestionAnsweringOutput,
    QuestionAnsweringParameters,
)


class QuestionAnsweringInputForGoogle(BaseModel):
    instances: Annotated[List[QuestionAnsweringInputs], Len(min_length=1)]
    parameters: Optional[QuestionAnsweringParameters] = Field(default=None)


class QuestionAnsweringOutputForGoogle(BaseModel):
    predictions: List[QuestionAnsweringOutput]
