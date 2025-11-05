from typing import Annotated, List, Optional

from annotated_types import Len
from pydantic import BaseModel, Field

from hf_serve.tasks.transformers.visual_question_answering import (
    VisualQuestionAnsweringInputs,
    VisualQuestionAnsweringOutput,
    VisualQuestionAnsweringParameters,
)


class VisualQuestionAnsweringInputForGoogle(BaseModel):
    instances: Annotated[List[VisualQuestionAnsweringInputs], Len(min_length=1)]
    parameters: Optional[VisualQuestionAnsweringParameters] = Field(default=None)


class VisualQuestionAnsweringOutputForGoogle(BaseModel):
    predictions: List[VisualQuestionAnsweringOutput]
