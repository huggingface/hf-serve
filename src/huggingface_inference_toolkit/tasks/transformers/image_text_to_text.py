import json
import os
import re
from threading import Thread
from time import time
from typing import Iterator, List, Union
from uuid import uuid4

import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from transformers.generation.streamers import TextIteratorStreamer
from transformers.image_utils import load_image

from huggingface_inference_toolkit.logging import logger
from huggingface_inference_toolkit.openai.schemas.chat_completions import (
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
from huggingface_inference_toolkit.tasks.predictor import Predictor

ImageTextToTextInput = ChatCompletionsInput
ImageTextToTextOutput = ChatCompletionsOutput
ImageTextToTextOutputChunk = ChatCompletionsOutputChunk


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


class ImageTextToText(
    Predictor[ImageTextToTextInput, Union[Iterator[ImageTextToTextOutputChunk], ImageTextToTextOutput]]
):
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

        if torch.mps.is_available():
            torch.mps.empty_cache()
            torch.mps.set_per_process_memory_fraction(0.9)

        self.processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True
            if os.getenv("TRUST_REMOTE_CODE", None) not in {None, 0, "false", "False"}
            else False,
        )

        self.streamer = TextIteratorStreamer(self.processor.tokenizer, skip_prompt=True)

    def __call__(
        self, payload: ImageTextToTextInput
    ) -> Union[Iterator[ImageTextToTextOutputChunk], ImageTextToTextOutput]:
        logger.info(f"Received input {payload=}")

        messages, images = [], []
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
                            elif isinstance(content, ContentPartImage):
                                images.append(load_image(content.image_url.url))
                                # NOTE: ideally the image tokens would be included when applying the chat template if including the following
                                # formatted_message["content"].append({"type": "image"})
                                # Since that won't work for `microsoft/Magma-8B`, we'll just add it as it follows
                                while formatted_message["content"].count(
                                    "<image_start><image><image_end>\n"
                                ) < len(images):
                                    formatted_message["content"] = (
                                        "<image_start><image><image_end>\n" + formatted_message["content"]
                                    )
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

        prompt = self.processor.apply_chat_template(
            messages,
            tools=tools,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.processor(texts=prompt, images=images or None, return_tensors="pt")
        if images:
            inputs["pixel_values"] = inputs["pixel_values"].unsqueeze(0)
            inputs["image_sizes"] = inputs["image_sizes"].unsqueeze(0)
        inputs = inputs.to(self.model.device).to(self.model.dtype)

        generation_kwargs = dict(
            inputs,
            max_new_tokens=payload.max_completion_tokens or 256,
            do_sample=True if (payload.temperature is not None and payload.temperature != 1.0) else False,
            temperature=payload.temperature if payload.temperature is not None else 1.0,
            top_p=payload.top_p if payload.top_p is not None else 1.0,
        )

        if payload.stream is True:
            generation_kwargs["streamer"] = self.streamer  # type: ignore

            id = f"chatcmpl-{uuid4().hex[:10]}"
            system_fingerprint = str(uuid4())

            with torch.no_grad():
                thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
                thread.start()

            completion_tokens = -1
            accumulated_text = ""

            for stream in self.streamer:
                completion_tokens += 1
                accumulated_text += stream

                # TODO: handle within_tool to capture whether it makes sense to capture the tool_call or not, i.e., ensure only when tool_call is done as per the last token (might not be super robust so think a bit carefully about it)
                tool_calls = extract_tool_calls(accumulated_text) if payload.tools else None

                finish_reason = None
                if stream in {self.processor.tokenizer.eos_token, self.processor.tokenizer.pad_token}:
                    finish_reason = "stop"
                elif completion_tokens >= (payload.max_completion_tokens or 256):
                    finish_reason = "length"
                elif tool_calls:
                    finish_reason = "tool_calls"

                delta = Delta(role="assistant", content=stream)
                if tool_calls:
                    delta.tool_calls = tool_calls

                yield ImageTextToTextOutputChunk(
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
                output = self.model.generate(**generation_kwargs)
            output = output[:, inputs["input_ids"].shape[-1] :][0]

            decoded_output = self.processor.decode(output, skip_special_tokens=True)

            tool_calls = extract_tool_calls(decoded_output) if payload.tools else None

            finish_reason = "stop"
            if output.shape[0] >= (payload.max_completion_tokens or 256):
                finish_reason = "length"
            elif tool_calls:
                finish_reason = "tool_calls"

            return ImageTextToTextOutput(
                id=f"chatcmpl-{uuid4().hex[:10]}",
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
