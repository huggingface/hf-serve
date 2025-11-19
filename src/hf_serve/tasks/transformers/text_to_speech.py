import os
import random
from io import BytesIO
from pathlib import Path
from typing import Literal, Optional, Union

import soundfile as sf
from pydantic import AliasChoices, AliasPath, BaseModel, ConfigDict, Field

from hf_serve.logging import logger
from hf_serve.serde.audio import Audio
from hf_serve.tasks.predictor import Predictor


# NOTE: Not sure about this one, but feels redundant to have to define the nested `generation_parameters`
# Reference: https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/inference/_generated/types/text_to_speech.py
class TextToSpeechParameters(BaseModel):
    do_sample: Optional[bool] = Field(
        default=None,
        validation_alias=AliasChoices("do_sample", AliasPath("generation_parameters", "do_sample")),
    )
    early_stopping: Optional[Union[bool, Literal["never"]]] = Field(
        default=None,
        validation_alias=AliasChoices("early_stopping", AliasPath("generation_parameters", "early_stopping")),
    )
    epsilon_cutoff: Optional[float]
    eta_cutoff: Optional[float] = Field(
        default=None,
        validation_alias=AliasChoices("eta_cutoff", AliasPath("generation_parameters", "eta_cutoff")),
    )
    max_length: Optional[int] = Field(
        default=None,
        validation_alias=AliasChoices("max_length", AliasPath("generation_parameters", "max_length")),
    )
    max_new_tokens: Optional[int] = Field(
        default=None,
        validation_alias=AliasChoices("max_new_tokens", AliasPath("generation_parameters", "max_new_tokens")),
    )
    min_length: Optional[int] = Field(
        default=None,
        validation_alias=AliasChoices("min_length", AliasPath("generation_parameters", "min_length")),
    )
    min_new_tokens: Optional[int] = Field(
        default=None,
        validation_alias=AliasChoices("min_new_tokens", AliasPath("generation_parameters", "min_new_tokens")),
    )
    num_beam_groups: Optional[int] = Field(
        default=None,
        validation_alias=AliasChoices("num_beam_groups", AliasPath("generation_parameters", "num_beam_groups")),
    )
    num_beams: Optional[int] = Field(
        default=None,
        validation_alias=AliasChoices("num_beams", AliasPath("generation_parameters", "num_beams")),
    )
    penalty_alpha: Optional[float] = Field(
        default=None,
        validation_alias=AliasChoices("penalty_alpha", AliasPath("generation_parameters", "penalty_alpha")),
    )
    temperature: Optional[float] = Field(
        default=None,
        validation_alias=AliasChoices("temperature", AliasPath("generation_parameters", "temperature")),
    )
    top_k: Optional[int] = Field(
        default=None, validation_alias=AliasChoices("top_k", AliasPath("generation_parameters", "top_k"))
    )
    top_p: Optional[float] = Field(
        default=None, validation_alias=AliasChoices("top_p", AliasPath("generation_parameters", "top_p"))
    )
    typical_p: Optional[float] = Field(
        default=None,
        validation_alias=AliasChoices("typical_p", AliasPath("generation_parameters", "typical_p")),
    )
    use_cache: Optional[bool] = Field(
        default=None,
        validation_alias=AliasChoices("use_cache", AliasPath("generation_parameters", "use_cache")),
    )

    # NOTE: The `voice` parameter has been manually included given that otherwise there's no way for the users
    # to specify which `voice` to use as of today
    voice: Optional[str] = Field(default=None)


class TextToSpeechInput(BaseModel):
    inputs: str
    parameters: Optional[TextToSpeechParameters] = None

    # NOTE: The example has been temporarily excluded to prevent long start up times
    # model_config = ConfigDict(
    #     json_schema_extra={
    #         "examples": [
    #             {"inputs": "What is the capital of France? Paris is the capital of France.", "parameters": None}
    #         ]
    #     }
    # )


class TextToSpeechOutput(BaseModel):
    audio: bytes
    sampling_rate: Optional[float] = None

    model_config = ConfigDict(
        json_encoders={bytes: Audio.serialize},
        arbitrary_types_allowed=True,
    )


class TextToSpeech(Predictor[TextToSpeechInput, TextToSpeechOutput]):
    def __init__(self, model_id: str, dtype: Optional[str] = None, device: str = "auto") -> None:
        super().__init__()

        if audio_path := os.getenv("AUDIO_PATH", None):
            self.audio_path = Path(audio_path)
        else:
            raise RuntimeError(
                "To run `text-to-speech` / `tts` pipelines you need to provide the path to the audio (wav) files that you'd like to use for the voices via the environment variable `AUDIO_PATH`. Note that the path must be absolute."
            )

        if not self.audio_path.exists():
            raise RuntimeError(
                f"The provided `AUDIO_PATH={audio_path}` doesn't exist. Please make sure you provide an audio path that exists and contains at least one wav file inside for the default voice of the `text-to-speech` model."
            )

        if len([file for file in self.audio_path.glob("*.wav")]) < 1:
            raise RuntimeError(
                f"The provided `AUDIO_PATH={audio_path}` doesn't contain any valid audio (wav) file, required for the `text-to-speech` model to generate the audio."
            )

        self.voices = {file.stem: audio_path / file for file in self.audio_path.glob("*.wav")}
        if len(self.voices) < 1:
            raise RuntimeError(
                f"The provided `AUDIO_PATH={audio_path}` does not contain any audio (wav) file, hence it's not valid as it doesn't contain the required audio files for the voices."
            )

        self.default_voice = os.getenv("DEFAULT_VOICE", None)
        if self.default_voice and self.default_voice not in self.voices:
            raise ValueError(
                f'The provided `DEFAULT_VOICE={self.default_voice}` is not listed among the available voices within the provided `AUDIO_PATH={os.getenv("AUDIO_PATH")}`. Please make sure to unset the `DEFAULT_VOICE` environment variable or rather set it to any of the following values instead: "'
                + '", "'.join(list(self.voices.keys()))
                + '".'
            )

        import torch
        from transformers import pipeline
        from transformers.pipelines.text_to_audio import TextToAudioPipeline

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"

        self.pipeline: TextToAudioPipeline = pipeline(
            task="text-to-speech",  # type: ignore OR `text-to-audio`
            model=model_id,
            dtype=getattr(torch, dtype) if dtype is not None else "auto",
            device=device,
        )

        if torch.mps.is_available():
            torch.mps.empty_cache()
            torch.mps.set_per_process_memory_fraction(0.9)

        from diffusers.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler

        self.noise_scheduler = DPMSolverMultistepScheduler(
            beta_schedule="squaredcos_cap_v2", num_train_timesteps=1000, prediction_type="v_prediction"
        )

    def __call__(self, payload: TextToSpeechInput) -> TextToSpeechOutput:
        messages = [{"role": "0", "content": [{"type": "text", "text": payload.inputs}]}]

        if payload.parameters is not None and payload.parameters.voice is not None:
            if payload.parameters.voice not in self.voices:
                raise ValueError(
                    f'The provided `voice={payload.parameters.voice}` is not listed among the available voices within the provided `AUDIO_PATH={os.getenv("AUDIO_PATH")}`. Please use any of the following voices instead: "'
                    + '", "'.join(list(self.voices.keys()))
                    + '".'
                )

            path = self.voices[payload.parameters.voice]
        elif self.default_voice is not None:
            logger.info(
                f"The `voice` parameter inside `parameters` hasn't been provided but the `DEFAULT_VOICE` is set to `{self.default_voice}`, meaning that it will be used unless the `voice` in `parameters` is set to any of the following values instead: \""
                + '", "'.join(list(self.voices.keys()))
                + '".'
            )
            path = self.voices[self.default_voice]
        else:
            voice = random.choice(list(self.voices.keys()))
            path = self.voices[voice]

            logger.warning(
                f"Given that the `voice` in `parameters` is either not provided or empty, the default `voice` will be set to `{voice}` (random selection). It's recommended that the `voice` parameter is provided as `{{'inputs':'...','parameters':{{'voice':'{voice}',...}}}}`, with any of the following values: \""
                + '", "'.join(list(self.voices.keys()))
                + '".'
            )

        messages[0]["content"].append({"type": "audio", "path": path})

        inputs = self.pipeline.tokenizer.apply_chat_template(messages, tokenize=False)  # type: ignore

        parameters = {}
        if payload.parameters:
            parameters = payload.parameters.model_dump(exclude={"voice"}, exclude_none=True)

        if "max_new_tokens" not in parameters:
            parameters["max_new_tokens"] = (
                self.pipeline.model.generation_config.max_new_tokens
                if self.pipeline.model.generation_config is not None
                and hasattr(self.pipeline.model.generation_config, "max_new_tokens")
                else None
            )

        output = self.pipeline(inputs, generate_kwargs={"noise_scheduler": self.noise_scheduler, **parameters})

        audio = output["audio"][0]
        if audio.ndim > 1:
            audio = audio.squeeze()
        sampling_rate = int(sr[0]) if (sr := output.get("sampling_rate", None)) else 24000

        buf = BytesIO()
        buf.name = "file.wav"
        sf.write(buf, audio, sampling_rate, format="wav")
        buf.seek(0)

        return TextToSpeechOutput(audio=buf.read(), sampling_rate=float(sampling_rate))
