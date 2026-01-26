from typing import Annotated, List, Optional

from annotated_types import Len
from pydantic import BaseModel, Field

from hf_inference_sdk.tasks.transformers.text_generation import TextGenerationOutput, TextGenerationParameters


class TextGenerationInputForGoogle(BaseModel):
    instances: Annotated[List[str], Len(min_length=1)]
    parameters: Optional[TextGenerationParameters] = Field(default=None)


class TextGenerationOutputForGoogle(BaseModel):
    predictions: List[TextGenerationOutput]
