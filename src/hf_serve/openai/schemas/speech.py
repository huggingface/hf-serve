from typing import Literal, Optional

from pydantic import BaseModel, Field, RootModel


class SpeechInput(BaseModel):
    # NOTE: Using `alias="input"` because `input` is a reserved keyword. Note that when dumping the schema into
    # a JSON, you should use `.model_dump(by_alias=True)` so that the dump uses the alias rather than `input_`
    input_: str = Field(..., alias="input", min_length=1, max_length=4096)
    model: str
    voice: str = Field(default="")
    instructions: Optional[str] = Field(default=None)
    response_format: Optional[Literal["mp3", "opus", "aac", "flac", "wav", "pcm"]] = Field(default="mp3")
    speed: Optional[float] = Field(default=1.0, ge=0.25, le=4.0)
    stream_format: Optional[Literal["audio", "sse"]] = Field(default="audio")


class SpeechOutput(RootModel):
    root: bytes
