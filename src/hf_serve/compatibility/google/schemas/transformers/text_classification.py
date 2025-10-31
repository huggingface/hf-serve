from typing import Annotated, List, Optional

from annotated_types import Len
from pydantic import BaseModel, Field

from hf_serve.tasks.transformers.text_classification import TextClassificationOutput


class TextClassificationInputForGoogle(BaseModel):
    instances: Annotated[List[str], Len(min_length=1)]
    parameters: Optional[None] = Field(default=None)


class TextClassificationOutputForGoogle(BaseModel):
    predictions: List[TextClassificationOutput]
