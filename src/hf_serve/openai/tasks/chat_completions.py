import json
import os
import re
from functools import lru_cache
from pathlib import Path
from threading import Thread
from time import time
from typing import Iterator, List, Optional, Union
from uuid import uuid4

import torch
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
from transformers.generation.streamers import TextIteratorStreamer
from transformers.image_utils import load_image

from hf_serve.openai.schemas.chat_completions import (
    ChatCompletionsInput,
    ChatCompletionsOutput,
    ChatCompletionsOutputChunk,
    Choice,
    ChoiceChunk,
    CompletionTokensDetails,
    ContentPartAudio,
    ContentPartFile,
    ContentPartImage,
    ContentPartRefusal,
    ContentPartText,
    Delta,
    FunctionCall,
    OutputMessage,
    PromptTokensDetails,
    ToolCall,
    Usage,
)


def extract_tool_calls(text: str) -> List[ToolCall]:
    """Extract tool calls from generated text."""
    tool_calls = []

    patterns = [
        # Pattern for <tool_call> format
        r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
        # Pattern for function calls in JSON format
        r"<function_call>\s*(\{.*?\})\s*</function_call>",
        # Pattern for direct JSON tool calls
        r'```json\s*(\{[^}]*"name"[^}]*"arguments"[^}]*\})\s*```',
        # Pattern for tool_response format
        r"<\|tool_response_start\|>\s*(\{.*?\})\s*<\|tool_response_end\|>",
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            try:
                tool_data = json.loads(match)
                if "name" in tool_data and "arguments" in tool_data:
                    tool_call = ToolCall(
                        id=f"call-{uuid4().hex[:8]}",
                        type="function",
                        function=FunctionCall(
                            name=tool_data["name"],
                            arguments=json.dumps(tool_data["arguments"])
                            if isinstance(tool_data["arguments"], dict)
                            else tool_data["arguments"],
                        ),
                    )
                    tool_calls.append(tool_call)
            except json.JSONDecodeError:
                continue

    return tool_calls


class ChatCompletions:
    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: Optional[AutoTokenizer] = None,
        processor: Optional[AutoProcessor] = None,
    ) -> None:
        super().__init__()

        if tokenizer is None and processor is None:
            raise RuntimeError(
                "Any of `tokenizer` or `processor` should be provided to the `ChatCompletions` handler. For context, the `tokenizer` should be provided when working with LLMs, and the `processor` when working with VLMs."
            )

        self.model = model
        self.tokenizer = tokenizer if tokenizer is not None else self.processor.tokenizer  # type: ignore
        self.processor = processor
        self.streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True)

    @property
    @lru_cache(maxsize=1)
    def model_id(self) -> Union[str, None]:
        return (
            self.model.config._name_or_path  # type: ignore
            if self.model.config is not None and not Path(self.model.config._name_or_path).exists()  # type: ignore
            else os.getenv("MODEL_ID", os.getenv("MODEL_DIR"))
        )

    def __call__(
        self, payload: ChatCompletionsInput, request_id: Optional[str] = None
    ) -> Union[Iterator[ChatCompletionsOutputChunk], ChatCompletionsOutput]:
        messages = []
        images = []  # NOTE: only required when the model is a VLM and the `processor` is provided
        for message in payload.messages:
            match message.role:
                case "system" | "developer":
                    formatted_message = {"role": message.role}
                    if isinstance(message.content, str):
                        formatted_message["content"] = message.content
                    elif isinstance(message.content, ContentPartText):
                        formatted_message["content"] = message.content.text
                    messages.append(formatted_message)
                case "user":
                    if self.processor is None:
                        formatted_message = {"role": message.role}
                        if isinstance(message.content, str):
                            formatted_message["content"] = message.content
                        elif isinstance(message.content, list):
                            formatted_message["content"] = ""
                            for content in message.content:
                                if isinstance(content, ContentPartText):
                                    formatted_message["content"] += content.text
                                elif isinstance(content, ContentPartRefusal):
                                    formatted_message["content"] += content.refusal
                                elif isinstance(
                                    content,
                                    (ContentPartImage, ContentPartAudio, ContentPartFile),
                                ):
                                    raise ValueError(
                                        f"Provided {payload.messages=} contains an input that's either an image, audio, or file, which is either not supported or not compatible yet."
                                    )
                        messages.append(formatted_message)
                    # NOTE: when the `processor` is provided, it currently means that it's a VLM
                    else:
                        formatted_message = {"role": message.role}
                        if isinstance(message.content, str):
                            formatted_message["content"] = message.content
                        elif isinstance(message.content, list):
                            formatted_message["content"] = []  # type: ignore
                            for content in message.content:
                                if isinstance(content, ContentPartText):
                                    formatted_message["content"].append({"type": "text", "text": content.text})
                                elif isinstance(content, ContentPartRefusal):
                                    formatted_message["content"].append(
                                        {"type": "text", "text": content.refusal}
                                    )
                                elif isinstance(content, ContentPartImage):
                                    images.append(load_image(content.image_url.url))
                                    formatted_message["content"].append({"type": "image"})
                                elif isinstance(
                                    content,
                                    (ContentPartAudio, ContentPartFile),
                                ):
                                    raise ValueError(
                                        f"Provided {payload.messages=} contains an input that's either audio or file, which is either not supported or not compatible yet."
                                    )
                        messages.append(formatted_message)
                case "assistant":
                    formatted_message = {"role": message.role}
                    if message.content is not None:
                        if isinstance(message.content, str):
                            formatted_message["content"] = message.content
                        elif isinstance(message.content, list):
                            formatted_message["content"] = ""
                            for content in message.content:
                                if isinstance(content, ContentPartText):
                                    formatted_message["content"] += content.text
                                elif isinstance(content, ContentPartRefusal):
                                    formatted_message["content"] += content.refusal
                    if message.tool_calls is not None:
                        formatted_message["tool_calls"] = [  # type: ignore
                            {
                                "id": tool_call.id,
                                "type": tool_call.type,
                                "function": {
                                    "name": tool_call.function.name,
                                    "arguments": tool_call.function.arguments,
                                },
                            }
                            for tool_call in message.tool_calls
                        ]
                    if message.function_call is not None:
                        formatted_message["function_call"] = message.function_call  # type: ignore
                    messages.append(formatted_message)
                case "tool":
                    formatted_message = {"role": message.role, "tool_call_id": message.tool_call_id}
                    if isinstance(message.content, str):
                        formatted_message["content"] = message.content
                    elif isinstance(message.content, list):
                        formatted_message["content"] = ""
                        for content in message.content:
                            if isinstance(content, ContentPartText):
                                formatted_message["content"] += content.text
                    messages.append(formatted_message)
                case "function":
                    formatted_message = {"role": message.role, "name": message.name}
                    if message.content is not None:
                        formatted_message["content"] = message.content
                    messages.append(formatted_message)

        tools = None
        if payload.tools is not None:
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.function.name,
                        "description": tool.function.description,
                        "parameters": tool.function.parameters,
                    },
                }
                for tool in payload.tools
            ]

        prompt = self.tokenizer.apply_chat_template(  # type: ignore
            messages,
            tools=tools,
            tokenize=False,
            add_generation_prompt=True,
        )

        if images and len(images) > 0:
            inputs = self.processor(texts=prompt, images=images, return_tensors="pt")  # type: ignore
            inputs["pixel_values"] = inputs["pixel_values"].unsqueeze(0)
            inputs["image_sizes"] = inputs["image_sizes"].unsqueeze(0)
        else:
            inputs = self.tokenizer(prompt, return_tensors="pt")  # type: ignore

        # NOTE: no need to cast to `self.model.dtype` as per `Attempting to cast a BatchEncoding to type torch.float16. This is not supported.`
        inputs = inputs.to(self.model.device)  # type: ignore

        generation_kwargs = dict(
            inputs,
            max_new_tokens=payload.max_completion_tokens or 256,
            do_sample=True if (payload.temperature is not None and payload.temperature != 1.0) else False,
            temperature=payload.temperature if payload.temperature is not None else 1.0,
            top_p=payload.top_p if payload.top_p is not None else 1.0,
        )

        if payload.seed:
            from transformers import set_seed

            set_seed(payload.seed)

        if payload.stream is True:
            generation_kwargs["streamer"] = self.streamer  # type: ignore

            id = f"chatcmpl-{request_id or uuid4().hex}"
            system_fingerprint = str(uuid4())

            with torch.no_grad():
                thread = Thread(target=self.model.generate, kwargs=generation_kwargs)  # type: ignore
                thread.start()

            completion_tokens = -1
            accumulated_text = ""

            for stream in self.streamer:
                completion_tokens += 1
                accumulated_text += stream

                # TODO: handle within_tool to capture whether it makes sense to capture the tool_call or not, i.e., ensure only when tool_call is done as per the last token (might not be super robust so think a bit carefully about it)
                tool_calls = extract_tool_calls(accumulated_text) if payload.tools else None

                finish_reason = None
                if stream in {self.tokenizer.eos_token, self.tokenizer.pad_token}:  # type: ignore
                    finish_reason = "stop"
                elif completion_tokens >= (payload.max_completion_tokens or 256):
                    finish_reason = "length"
                elif tool_calls:
                    finish_reason = "tool_calls"

                delta = Delta(role="assistant", content=stream)
                if tool_calls:
                    delta.tool_calls = tool_calls

                yield ChatCompletionsOutputChunk(
                    id=id,
                    object="chat.completion.chunk",
                    created=int(time()),
                    model=payload.model,
                    choices=[
                        ChoiceChunk(
                            index=0,
                            delta=delta,
                            logprobs=None,
                            finish_reason=finish_reason,
                        ),
                    ],
                    usage=Usage(
                        prompt_tokens=inputs["input_ids"].size(1),
                        completion_tokens=completion_tokens,
                        reasoning_tokens=0,
                        total_tokens=completion_tokens + inputs["input_ids"].size(1),
                        completion_tokens_details=CompletionTokensDetails(
                            accepted_prediction_tokens=0,
                            audio_tokens=0,
                            reasoning_tokens=0,
                            rejected_prediction_tokens=0,
                        ),
                        prompt_tokens_details=PromptTokensDetails(audio_tokens=0, cached_tokens=0),
                    ),
                    service_tier="default",
                    system_fingerprint=system_fingerprint,
                )
        else:
            with torch.no_grad():
                output = self.model.generate(**generation_kwargs)  # type: ignore
            output = output[:, inputs["input_ids"].shape[-1] :][0]

            decoded_output = self.tokenizer.decode(output, skip_special_tokens=True)  # type: ignore

            tool_calls = extract_tool_calls(decoded_output) if payload.tools else None

            finish_reason = "stop"
            if output.shape[0] >= (payload.max_completion_tokens or 256):
                finish_reason = "length"
            elif tool_calls:
                finish_reason = "tool_calls"

            return ChatCompletionsOutput(
                id=f"chatcmpl-{request_id or uuid4().hex}",
                object="chat.completion",
                created=int(time()),
                model=payload.model,
                choices=[
                    Choice(
                        index=0,
                        message=OutputMessage(
                            content=decoded_output,
                            refusal=None,
                            role="assistant",
                            annotations=[],
                            tool_calls=tool_calls,
                        ),
                        logprobs=None,
                        finish_reason=finish_reason,
                    )
                ],
                usage=Usage(
                    prompt_tokens=inputs["input_ids"].size(1),
                    completion_tokens=output.shape[0],
                    reasoning_tokens=0,
                    total_tokens=output.shape[0] + inputs["input_ids"].size(1),
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
