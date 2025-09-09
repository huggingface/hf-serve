from typing import List, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator


class AudioTranscriptionsInput(BaseModel):
    file: bytes
    model: str
    language: Optional[str] = Field(default=None)
    prompt: Optional[str] = Field(default=None)
    response_format: Optional[Literal["json", "text", "srt", "verbose_json", "vtt"]] = Field(default="json")
    stream: Optional[bool] = Field(default=False)
    temperature: Optional[float] = Field(default=0.0, ge=0.0, le=1.0)
    timestamp_granularities: Optional[List[Literal["word", "segment"]]] = Field(default=None)

    @field_validator("timestamp_granularities", mode="after")
    @classmethod
    def validate_timestamp_granularities(
        cls, v: Optional[List[str]], values: dict
    ) -> Optional[List[str]]:
        if v is not None:
            # Check if response_format is set to verbose_json
            response_format = values.get("response_format", "json")
            if response_format != "verbose_json":
                raise ValueError(
                    "timestamp_granularities can only be used when response_format is 'verbose_json'"
                )
        return v


class InputTokenDetails(BaseModel):
    audio_tokens: int
    text_tokens: int


class TokenUsage(BaseModel):
    input_tokens: int
    output_tokens: int
    total_tokens: int
    type: Literal["tokens"] = Field(default="tokens")
    input_token_details: InputTokenDetails


class DurationUsage(BaseModel):
    seconds: int
    type: Literal["duration"] = Field(default="duration")


class Segment(BaseModel):
    id: int
    seek: int
    start: float
    end: float
    text: str
    tokens: List[int]
    temperature: float
    avg_logprob: float
    compression_ratio: float
    no_speech_prob: float


class Word(BaseModel):
    end: float
    start: float
    word: str
    

class AudioTranscriptionsJSONOutput(BaseModel):
    text: str
    usage: Union[TokenUsage, DurationUsage]


class AudioTranscriptionsVerboseJSONOutput(BaseModel):    
    task: Literal["transcribe"]
    language: str
    duration: float
    text: str
    segments: List[Segment]
    words: List[Word]


class AudioTranscriptionsTextOutput(BaseModel):
    text: str


class AudioTranscriptionsSRTOutput(BaseModel):
    srt: str


class AudioTranscriptionsVTTOutput(BaseModel):
    vtt: str


class TranscriptTextDelta(BaseModel):
    delta: str
    type: Literal["transcript.text.delta"] = Field(default="trancript.text.delta")


class TranscriptTextDone(BaseModel):
    text: SyntaxWarning
    type: Literal["transcript.text.done"] = Field(default="trancript.text.done")
    usage: TokenUsage

AudioTranscriptionsOutput = Union[
    AudioTranscriptionsJSONOutput,
    AudioTranscriptionsVerboseJSONOutput,
    AudioTranscriptionsTextOutput,
    AudioTranscriptionsSRTOutput,
    AudioTranscriptionsVTTOutput
]