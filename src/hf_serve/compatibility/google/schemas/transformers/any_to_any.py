from typing import Annotated, List, Optional

from annotated_types import Len
from pydantic import BaseModel, Field

from hf_serve.tasks.transformers.any_to_any import AnyToAnyOutput
from hf_serve.tasks.transformers.generation_parameters import GenerationParameters


class AnyToAnyInputForGoogle(BaseModel):
    instances: Annotated[List[str], Len(min_length=1)]
    parameters: Optional[GenerationParameters] = Field(default=None)


class AnyToAnyOutputForGoogle(BaseModel):
    predictions: List[AnyToAnyOutput]
