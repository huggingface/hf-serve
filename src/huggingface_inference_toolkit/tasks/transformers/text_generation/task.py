import os
from pathlib import Path
from threading import Thread
from time import time
from typing import Iterator, Union
from uuid import uuid4

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.streamers import TextIteratorStreamer

from huggingface_inference_toolkit.logging import logger
from huggingface_inference_toolkit.tasks.predictor import Predictor
from huggingface_inference_toolkit.tasks.transformers.image_text_to_text import ContentPartAudio
from huggingface_inference_toolkit.tasks.transformers.text_generation.schemas import (
    Choice,
    ChoiceChunk,
    CompletionTokensDetails,
    ContentPartFile,
    ContentPartImage,
    ContentPartRefusal,
    ContentPartText,
    Delta,
    OutputMessage,
    PromptTokensDetails,
    TextGenerationInput,
    TextGenerationOutput,
    TextGenerationOutputChunk,
    Usage,
)


class TextGeneration(
    Predictor[TextGenerationInput, Union[Iterator[TextGenerationOutputChunk], TextGenerationOutput]]
):
    def __init__(self, model_id: str, dtype: str = "float16", device: str = "auto") -> None:
        super().__init__()

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=getattr(torch, dtype),
        )

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"

        self.model = self.model.to(device)

        if torch.mps.is_available():
            torch.mps.empty_cache()
            torch.mps.set_per_process_memory_fraction(0.9)

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        self.streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True)

    @property
    def model_id(self) -> Union[str, None]:
        return (
            self.model.config._name_or_path
            if not Path(self.model.config._name_or_path).exists()
            else os.getenv("MODEL_ID")
        )

    def __call__(
        self, payload: TextGenerationInput
    ) -> Union[Iterator[TextGenerationOutputChunk], TextGenerationOutput]:
        logger.info(f"Received input {payload=}")

        messages = []
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
                            elif isinstance(
                                content,
                                (ContentPartImage, ContentPartAudio, ContentPartFile),
                            ):
                                raise ValueError(
                                    f"Provided {payload.messages=} contains an input that's either an image, audio, or file, which is either not supported or not compatible yet."
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

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = inputs.to(self.model.device)  # .to(self.model.dtype)

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

            # NOTE: since the first token is always ""
            completion_tokens = -1
            for stream in self.streamer:
                completion_tokens += 1
                yield TextGenerationOutputChunk(
                    id=id,
                    object="chat.completion.chunk",
                    created=int(time()),
                    model=payload.model,
                    choices=[
                        ChoiceChunk(
                            index=0,
                            delta=Delta(role="assistant", content=stream),
                            logprobs=None,
                            finish_reason="length"
                            if stream not in {self.tokenizer.eos_token, self.tokenizer.pad_token}
                            and completion_tokens >= (payload.max_completion_tokens or 256)
                            else "stop"
                            if stream in {self.tokenizer.eos_token, self.tokenizer.pad_token}
                            else None,
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

            return TextGenerationOutput(
                id=f"chatcmpl-{uuid4().hex[:10]}",
                object="chat.completion",
                created=int(time()),
                model=payload.model,
                choices=[
                    Choice(
                        index=0,
                        message=OutputMessage(
                            content=self.tokenizer.decode(output, skip_special_tokens=True),
                            refusal=None,
                            role="assistant",
                            annotations=[],
                        ),
                        logprobs=None,
                        finish_reason="length"
                        if output.shape[0] >= (payload.max_completion_tokens or 256)
                        else "stop",
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
