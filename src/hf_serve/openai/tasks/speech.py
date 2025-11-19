import os
import random
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional, Union

import soundfile as sf

from hf_serve.logging import logger
from hf_serve.openai.schemas.speech import SpeechInput, SpeechOutput

if TYPE_CHECKING:
    from diffusers.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler
    from transformers import TextToAudioPipeline


class Speech:
    def __init__(
        self,
        pipeline: "TextToAudioPipeline",
        voices: Dict[str, Path],
        noise_scheduler: Optional["DPMSolverMultistepScheduler"] = None,
    ) -> None:
        super().__init__()

        self.pipeline = pipeline
        self.noise_scheduler = noise_scheduler

        self.voices = voices

        logger.info(
            f'The `voice` parameter in `v1/audio/speech` can be any of the following values: "'
            + '", "'.join(list(self.voices.keys()))
            + '".'
        )

    @property
    @lru_cache(maxsize=1)
    def model_id(self) -> Union[str, None]:
        return (
            self.pipeline.model.config._name_or_path  # type: ignore
            if self.pipeline.model.config is not None
            and not Path(self.pipeline.model.config._name_or_path).exists()  # type: ignore
            else os.getenv("MODEL_ID", os.getenv("MODEL_DIR"))
        )

    def __call__(self, payload: SpeechInput, request_id: Optional[str] = None) -> SpeechOutput:
        # NOTE: `instructions` is not supported
        # instructions: Optional[str] = Field(default=None)
        # NOTE: `response_format` shouldn't be a parameter but rather provided within the `Accept` header
        # response_format: Optional[Literal["mp3", "opus", "aac", "flac", "wav", "pcm"]] = Field(default="mp3")
        # NOTE: `speed` is apparently not supported either
        # speed: Optional[float] = Field(default=1.0, ge=0.25, le=4.0)
        # NOTE: `stream_format` is supported, but only for "audio" given that audio streaming is not yet in `transformers`
        # stream_format: Optional[Literal["audio", "sse"]] = Field(default="audio")

        if payload.voice not in self.voices:
            raise RuntimeError(
                f'[{request_id}] The provided `voice={payload.voice}` is not listed among the available voices within the provided `AUDIO_PATH={os.getenv("AUDIO_PATH")}`. Please use any of the following voices instead: "'
                + '", "'.join(list(self.voices.keys()))
                + '"'
            )

        messages = [
            {
                "role": "0",
                "content": [
                    {"type": "text", "text": payload.input_},
                    {"type": "audio", "path": self.voices[payload.voice]},
                ],
            }
        ]

        if self.pipeline.tokenizer is None:
            raise RuntimeError(
                f"[{request_id}] The provided `pipeline` contains an invalid `tokenizer`. Please raise an issue at https://github.com/huggingface/hf-serve to help us debug and fix the issue."
            )

        inputs = self.pipeline.tokenizer.apply_chat_template(messages, tokenize=False)

        output = self.pipeline.__call__(
            inputs,
            generate_kwargs={
                "noise_scheduler": self.noise_scheduler,
                "max_new_tokens": self.pipeline.model.generation_config.max_new_tokens
                if self.pipeline.model.generation_config is not None
                and hasattr(self.pipeline.model.generation_config, "max_new_tokens")
                else None,
            },
        )

        audio = output["audio"][0]
        if audio.ndim > 1:
            audio = audio.squeeze()
        sampling_rate = int(sr[0]) if (sr := output.get("sampling_rate", None)) else 24000

        buf = BytesIO()
        buf.name = f"file.{payload.response_format}"
        sf.write(buf, audio, sampling_rate, format=payload.response_format)
        buf.seek(0)

        return SpeechOutput(root=buf.read())
