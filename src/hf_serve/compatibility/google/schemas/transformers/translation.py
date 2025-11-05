from typing import Annotated, List, Optional

from annotated_types import Len
from pydantic import BaseModel, Field

from hf_serve.tasks.transformers.translation import TranslationOutput, TranslationParameters


class TranslationInputForGoogle(BaseModel):
    instances: Annotated[List[str], Len(min_length=1)]
    parameters: Optional[TranslationParameters] = Field(default=None)


class TranslationOutputForGoogle(BaseModel):
    predictions: List[TranslationOutput]
