from typing import Annotated, List, Optional, Union

from annotated_types import Len
from pydantic import BaseModel, Field

from hf_serve.tasks.transformers.speaker_diarization import (
    SpeakerDiarizationOutput,
    SpeakerDiarizationParameters,
)


class SpeakerDiarizationInputForGoogle(BaseModel):
    instances: Annotated[List[Union[str, bytes]], Len(min_length=1)]
    parameters: Optional[SpeakerDiarizationParameters] = Field(default=None)


class SpeakerDiarizationOutputForGoogle(BaseModel):
    predictions: List[SpeakerDiarizationOutput]
