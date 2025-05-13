import os
from time import time
from typing import Annotated, Any, Dict, List, Literal, Optional, Union
from uuid import uuid4

import torch
from pydantic import BaseModel, Field, conint
from transformers import AutoModelForCausalLM, AutoProcessor
from transformers.image_utils import load_image

from huggingface_inference_toolkit.logging import logger
from huggingface_inference_toolkit.tasks.predictor import Predictor


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


# NOTE: deprecated in favor of `ToolMessage`
class FunctionMessage(BaseModel):
    role: Literal["function"] = Field(default="function")
    content: Optional[str] = Field(default=None)
    name: str


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
    # NOTE: could be a separate schema but since it just contains a single field is left as is
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


class ImageTextToTextInput(BaseModel):
    messages: List[Annotated[InputMessage, Field(discriminator="role")]]
    model: str
    audio: Optional[AudioInput] = Field(default=None)
    frequency_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
    function_call: Optional[Union[Literal["none", "auto"], str]] = Field(default=None, deprecated=True)
    functions: Optional[Dict[str, Any]] = Field(default=None, deprecated=True)
    logit_bias: Optional[Dict[int, Annotated[int, conint(ge=-100, le=100)]]] = Field(default=None)
    logprobs: Optional[bool] = Field(default=False)
    max_completion_tokens: Optional[int] = Field(alias="max_tokens")
    metadata: Optional[Dict[str, Any]] = Field(
        default=None
    )  # NOTE: up-to 16 kv pairs, key 64 chars, value 512 chars
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
    stream_options: Optional[StreamOptions] = Field(default=None)  # NOTE: ignore if `stream=False`
    temperature: Optional[float] = Field(default=1.0, ge=0.0, le=2.0)
    tool_choice: Optional[Union[Literal["none", "auto", "required"], ToolChoice]] = Field(
        default="none"
    )  # NOTE: set to "auto" if `tools` is not None, and to None if `tools` is None
    tools: Optional[List[Tool]] = Field(default=None)
    top_logprobs: Optional[int] = Field(default=1, ge=0, le=20)  # NOTE: ignore field if `logprobs=False`
    top_p: Optional[float] = Field(default=1.0, ge=0.0, le=1.0)
    user: Optional[str] = Field(default=None)
    web_search_options: Optional[WebSearchOptions] = Field(default=None)


class TopLogProb(BaseModel):
    bytes: Optional[List[int]] = Field(default=None)
    logprob: float
    token: str


class Refusal(BaseModel):
    bytes: Optional[List[int]] = Field(default=None)
    logprob: float
    token: str
    top_logprobs: List[TopLogProb]


class LogProbs(BaseModel):
    content: str
    refusal: Refusal


class OutputAudio(BaseModel):
    data: str
    expires_at: int
    id: str
    transcript: str


class Annotation(BaseModel):
    audio: Optional[OutputAudio] = Field(default=None)
    function_call: Annotated[FunctionCall, Field(deprecated=True)]
    tool_calls: List[ToolCall]


class OutputMessage(BaseModel):
    content: Optional[str] = Field(default=None)
    refusal: Optional[str] = Field(default=None)
    role: str
    annotations: List[Annotation]


class Choice(BaseModel):
    index: int
    message: OutputMessage
    logprobs: Optional[LogProbs] = Field(default=None)
    finish_reason: Literal["stop", "length", "content_filter", "tool_calls", "function_call"]


class CompletionTokensDetails(BaseModel):
    accepted_prediction_tokens: int
    audio_tokens: int
    reasoning_tokens: int
    rejected_prediction_tokens: int


class PromptTokensDetails(BaseModel):
    audio_tokens: int
    cached_tokens: int


class Usage(BaseModel):
    completion_tokens: int
    reasoning_tokens: int
    total_tokens: int
    completion_tokens_details: CompletionTokensDetails
    prompt_tokens_details: PromptTokensDetails


class ImageTextToTextOutput(BaseModel):
    id: str
    object: Literal["chat.completion"] = Field(default="chat.completion")
    created: int
    model: str
    choices: List[Choice]
    usage: Usage
    service_tier: Literal["auto", "default", "flex"] = Field(default="auto")
    system_fingerprint: str


class ImageTextToText(Predictor[ImageTextToTextInput, ImageTextToTextOutput]):
    def __init__(self, model_id: str, dtype: str = "float16", device: str = "auto") -> None:
        super().__init__()

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True
            if os.getenv("TRUST_REMOTE_CODE", None) not in {None, 0, "false", "False"}
            else False,
            torch_dtype=getattr(torch, dtype),
        )

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"

        self.model = self.model.to(device)

        self.processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True
            if os.getenv("TRUST_REMOTE_CODE", None) not in {None, 0, "false", "False"}
            else False,
        )

        if torch.mps.is_available():
            torch.mps.empty_cache()
            torch.mps.set_per_process_memory_fraction(0.9)

    def __call__(self, payload: ImageTextToTextInput) -> ImageTextToTextOutput:
        logger.info(f"Received input {payload}")

        messages, images = [], []
        for message in payload.messages:
            match message.role:
                case "user":
                    formatted_message = {"role": message.role}
                    if isinstance(message.content, str):
                        formatted_message["content"] = message.content
                    elif isinstance(message.content, list):
                        formatted_message["content"] = []  # type: ignore
                        for content in message.content:
                            if isinstance(content, ContentPartText):
                                formatted_message["content"].append(content)
                            elif isinstance(content, ContentPartImage):
                                images.append(load_image(content.image_url.url))
                                formatted_message["content"].append({"type": "image"})
                    messages.append(formatted_message)
                case "system":
                    formatted_message = {"role": message.role}
                    if isinstance(message.content, str):
                        formatted_message["content"] = message.content
                    elif isinstance(message.content, ContentPartText):
                        formatted_message["content"] = message.content.text
                    messages.append(formatted_message)
                case "assistant" | "developer" | "tool" | "function":
                    pass

        prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.processor(texts=prompt, images=images, return_tensors="pt")
        inputs = inputs.to(self.model.device)

        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=payload.max_completion_tokens or 128,
                temperature=payload.temperature,
                top_p=payload.top_p,
            )

        logger.info(f"output contains {output=}")
        logger.info(f"output has shape {output.shape()=}")

        return ImageTextToTextOutput(
            id=f"chatcmpl-{uuid4().hex[:10]}",
            object="chat.completion",
            created=int(time()),
            model=payload.model,
            choices=[
                Choice(
                    index=0,
                    message=OutputMessage(
                        content=self.processor.batch_decode(output, skip_special_tokens=True),
                        refusal=None,
                        role="assistant",
                        annotations=[],
                    ),
                    logprobs=None,
                    finish_reason="stop" if output.size(1) < payload.max_completion_tokens or 128 else "length",
                )
            ],
            usage=Usage(
                completion_tokens=output.size(1),
                reasoning_tokens=0,
                total_tokens=inputs["input_ids"].size(1),
                completion_tokens_details=CompletionTokensDetails(
                    accepted_prediction_tokens=0,
                    audio_tokens=0,
                    reasoning_tokens=0,
                    rejected_prediction_tokens=0,
                ),
                prompt_tokens_details=PromptTokensDetails(audio_tokens=0, cached_tokens=0),
            ),
            service_tier="default",
            system_fingerprint=str(uuid4()),
        )
