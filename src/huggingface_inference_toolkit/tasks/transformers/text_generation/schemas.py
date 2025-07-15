from typing import Annotated, Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, conint, field_validator, model_validator
from pydantic.aliases import AliasChoices, AliasPath

from huggingface_inference_toolkit.logging import logger


class ContentPartText(BaseModel):
    text: str
    type: Literal["text"] = Field(default="text")


class ImageData(BaseModel):
    url: str
    detail: Optional[Literal["auto"]] = Field(default="auto")


class ContentPartImage(BaseModel):
    image_url: ImageData
    type: Literal["image_url"] = Field(default="image_url")


class AudioData(BaseModel):
    data: str
    format: Literal["wav", "mp3"]


class ContentPartAudio(BaseModel):
    input_audio: AudioData
    type: Literal["input_audio"] = Field(default="input_audio")


class FileData(BaseModel):
    file_data: Optional[str] = Field(default=None)
    file_id: Optional[str] = Field(default=None)
    filename: Optional[str] = Field(default=None)


class ContentPartFile(BaseModel):
    file: FileData
    type: Literal["file"] = Field(default="file")


class DeveloperMessage(BaseModel):
    role: Literal["developer"] = Field(default="developer")
    content: Union[str, List[ContentPartText]]
    name: Optional[str] = Field(default=None)


class SystemMessage(BaseModel):
    role: Literal["system"] = Field(default="system")
    content: Union[str, List[ContentPartText]]
    name: Optional[str] = Field(default=None)


class UserMessage(BaseModel):
    role: Literal["user"] = Field(default="user")
    content: Union[str, List[Union[ContentPartText, ContentPartImage, ContentPartAudio, ContentPartFile]]]
    name: Optional[str] = Field(default=None)


class FunctionCall(BaseModel):
    arguments: str
    name: str


class ToolCall(BaseModel):
    function: FunctionCall
    id: str
    type: Literal["function"] = Field(default="function")


class ContentPartRefusal(BaseModel):
    refusal: str
    type: Literal["refusal"] = Field(default="refusal")


class AssistantMessage(BaseModel):
    role: Literal["assistant"] = Field(default="assistant")
    audio: Optional[Dict[Literal["id"], str]] = Field(default=None)
    content: Optional[Union[str, List[Union[ContentPartText, ContentPartRefusal]]]] = Field(default=None)
    function_call: Optional[Dict[str, Any]] = Field(default=None, deprecated=True)
    name: Optional[str] = Field(default=None)
    refusal: Optional[str] = Field(default=None)
    tool_calls: Optional[List[ToolCall]] = Field(default=None)


class ToolMessage(BaseModel):
    role: Literal["tool"] = Field(default="tool")
    content: Union[str, List[ContentPartText]]
    tool_call_id: str


class FunctionMessage(BaseModel):
    role: Literal["function"] = Field(default="function")
    content: Optional[str] = Field(default=None)
    name: str

    def __init__(self, **data):
        logger.warning("FunctionMessage is deprecated in favor of ToolMessage. Please use ToolMessage instead.")
        super().__init__(**data)


InputMessage = Union[
    DeveloperMessage, SystemMessage, UserMessage, AssistantMessage, ToolMessage, FunctionMessage
]


class AudioInput(BaseModel):
    format: Literal["wav", "mp3", "flac", "opus", "pcm16"]
    voice: Literal["alloy", "ash", "ballad", "coral", "echo", "fable", "nova", "onyx", "sage", "shimmer"]


class Prediction(BaseModel):
    content: Union[str, List[ContentPartText]]
    type: str


class ResponseFormatText(BaseModel):
    type: Literal["text"] = Field(default="text")


class JsonSchema(BaseModel):
    name: str
    description: Optional[str] = Field(default=None)
    json_schema: Optional[Dict[str, Any]] = Field(default=None, serialization_alias="schema", alias="schema")
    strict: Optional[bool] = Field(default=False)


class ResponseFormatJsonSchema(BaseModel):
    type: Literal["json_schema"] = Field(default="json_schema")
    json_schema: JsonSchema


class ResponseFormatJsonObject(BaseModel):
    type: Literal["json_object"] = Field(default="json_object")


class StreamOptions(BaseModel):
    include_usage: Optional[bool] = Field(default=None)


class ToolChoice(BaseModel):
    function: Dict[Literal["name"], str]
    type: str


class ToolFunction(BaseModel):
    name: str
    description: Optional[str] = Field(default=None)
    parameters: Optional[Dict[str, Any]] = Field(default=None)
    strict: Optional[bool] = Field(default=False)


class Tool(BaseModel):
    function: ToolFunction
    type: str


class ApproximateLocation(BaseModel):
    city: Optional[str] = Field(default=None)
    country: Optional[str] = Field(default=None)
    region: Optional[str] = Field(default=None)
    timezone: Optional[str] = Field(default=None)


class UserLocation(BaseModel):
    approximate: ApproximateLocation
    type: str


class WebSearchOptions(BaseModel):
    search_context_size: Optional[Literal["low", "medium", "high"]] = Field(default="medium")


class TextGenerationInput(BaseModel):
    messages: List[Annotated[InputMessage, Field(discriminator="role")]]
    model: str
    audio: Optional[AudioInput] = Field(default=None)
    frequency_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
    function_call: Optional[Union[Literal["none", "auto"], str]] = Field(default=None, deprecated=True)
    functions: Optional[Dict[str, Any]] = Field(default=None, deprecated=True)
    logit_bias: Optional[Dict[int, Annotated[int, conint(ge=-100, le=100)]]] = Field(default=None)
    logprobs: Optional[bool] = Field(default=False)
    max_completion_tokens: Optional[int] = Field(
        default=None,
        validation_alias=AliasChoices("max_completion_tokens", AliasPath("max_tokens")),
    )
    metadata: Optional[Dict[str, Any]] = Field(default=None)
    modalities: List[Literal["text", "audio"]] = Field(default=["text"])
    n: Optional[int] = Field(default=1)
    parallel_tool_calls: Optional[bool] = Field(default=True)
    prediction: Optional[Prediction] = Field(default=None)
    presence_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
    reasoning_effort: Optional[Literal["low", "medium", "high"]] = Field(default="medium")
    response_format: Optional[Union[ResponseFormatText, ResponseFormatJsonSchema, ResponseFormatJsonObject]] = (
        Field(default=ResponseFormatText())
    )
    seed: Optional[int] = Field(default=None)
    service_tier: Optional[Literal["auto", "default", "flex"]] = Field(default="auto")
    stop: Optional[Union[str, List[str]]] = Field(default=None)
    store: Optional[bool] = Field(default=False)
    stream: Optional[bool] = Field(default=False)
    stream_options: Optional[StreamOptions] = Field(default=None)
    temperature: Optional[float] = Field(default=1.0, ge=0.0, le=2.0)
    tool_choice: Optional[Union[Literal["none", "auto", "required"], ToolChoice]] = Field(default="none")
    tools: Optional[List[Tool]] = Field(default=None)
    top_logprobs: Optional[int] = Field(default=1, ge=0, le=20)
    top_p: Optional[float] = Field(default=1.0, ge=0.0, le=1.0)
    user: Optional[str] = Field(default=None)
    web_search_options: Optional[WebSearchOptions] = Field(default=None)

    @field_validator("metadata")
    @classmethod
    def validate_metadata(cls, v: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if v is None:
            return v
        if len(v) > 16:
            raise ValueError("`metadata` can have at most 16 key-value pairs")
        for key, value in v.items():
            if len(key) > 64:
                raise ValueError("`metadata` keys must be at most 64 characters")
            if len(str(value)) > 512:
                raise ValueError("`metadata` values must be at most 512 characters")
        return v

    @model_validator(mode="after")
    def validate_compatibility(self) -> "TextGenerationInput":
        if self.function_call is not None:
            logger.warning("`function_call` is deprecated in favor of `tool_choice`. Mapping to `tool_choice`.")
            if self.function_call == "none":
                self.tool_choice = "none"
            elif self.function_call == "auto":
                self.tool_choice = "auto"
            else:
                self.tool_choice = ToolChoice(function={"name": self.function_call}, type="function")

        if self.functions is not None:
            logger.warning("`functions` is deprecated in favor of `tools`. Mapping to `tools`.")
            if self.tools is None:
                self.tools = []
            for func_name, func_def in self.functions.items():
                tool = Tool(
                    function=ToolFunction(
                        name=func_name,
                        description=func_def.get("description"),
                        parameters=func_def.get("parameters"),
                    ),
                    type="function",
                )
                self.tools.append(tool)

        if self.tools is not None and self.tool_choice == "none":
            logger.warning(
                "`tool_choice` is set to `none` but `tools` are provided. Setting `tool_choice` to `auto`."
            )
            self.tool_choice = "auto"
        elif self.tools is None and self.tool_choice not in ["none", None]:
            logger.warning("`tool_choice` is set but no `tools` are provided. Setting `tool_choice` to `none`.")
            self.tool_choice = "none"

        if self.stream_options is not None and not self.stream:
            logger.warning("`stream_options` provided but `stream` is False. `stream_options` will be ignored.")

        if self.top_logprobs is not None and self.top_logprobs > 0 and not self.logprobs:
            logger.warning("`top_logprobs` provided but `logprobs` is False. `top_logprobs` will be ignored.")

        return self


class TopLogProb(BaseModel):
    bytes: Optional[List[int]] = Field(default=None)
    logprob: float
    token: str


class LogProb(BaseModel):
    bytes: Optional[List[int]] = Field(default=None)
    logprob: float
    token: str
    top_logprobs: List[TopLogProb]


class LogProbs(BaseModel):
    content: Optional[List[LogProb]] = Field(default=None)
    refusal: Optional[List[LogProb]] = Field(default=None)


class OutputAudio(BaseModel):
    data: str
    expires_at: int
    id: str
    transcript: str


class UrlCitation(BaseModel):
    end_index: int
    start_index: int
    title: str
    url: str


class Annotation(BaseModel):
    type: Literal["url_citation"] = Field(default="url_citation")
    url_citation: UrlCitation


class OutputMessage(BaseModel):
    content: Optional[str] = Field(default=None)
    refusal: Optional[str] = Field(default=None)
    role: str
    annotations: List[Annotation]
    audio: Optional[OutputAudio] = Field(default=None)
    function_call: Annotated[FunctionCall, Field(deprecated=True)]
    tool_calls: List[ToolCall]


class Choice(BaseModel):
    index: int
    message: OutputMessage
    logprobs: Optional[LogProbs] = Field(default=None)
    finish_reason: Literal["stop", "length", "content_filter", "tool_calls", "function_call"]


class Delta(BaseModel):
    content: Optional[str] = Field(default=None)
    function_call: Optional[Dict[str, Any]] = Field(default=None, deprecated=True)
    refusal: Optional[str] = Field(default=None)
    role: Literal["developer", "system", "user", "assistant", "function", "tool"]
    tool_calls: Optional[List[ToolCall]] = Field(default=None)


class ChoiceChunk(BaseModel):
    index: int
    delta: Delta
    logprobs: Optional[LogProbs] = Field(default=None)
    finish_reason: Optional[Literal["stop", "length", "content_filter", "tool_calls", "function_call"]] = Field(
        default=None
    )


class CompletionTokensDetails(BaseModel):
    accepted_prediction_tokens: int
    audio_tokens: int
    reasoning_tokens: int
    rejected_prediction_tokens: int


class PromptTokensDetails(BaseModel):
    audio_tokens: int
    cached_tokens: int


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    reasoning_tokens: int
    total_tokens: int
    completion_tokens_details: CompletionTokensDetails
    prompt_tokens_details: PromptTokensDetails


class TextGenerationOutput(BaseModel):
    id: str
    object: Literal["chat.completion"] = Field(default="chat.completion")
    created: int
    model: str
    choices: List[Choice]
    usage: Usage
    service_tier: Literal["auto", "default", "flex"] = Field(default="auto")
    system_fingerprint: str


class TextGenerationOutputChunk(BaseModel):
    id: str
    object: Literal["chat.completion.chunk"] = Field(default="chat.completion.chunk")
    created: int
    model: str
    choices: List[ChoiceChunk]
    usage: Usage
    service_tier: Literal["auto", "default", "flex"] = Field(default="auto")
    system_fingerprint: str
