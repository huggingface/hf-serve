import os
import random
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

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
    # NOTE: `inputs` as per the Hugging Face API Specification should only be a string, but given that one interesting
    # use case for some `text-to-speech` models is generating conversations, it also allows a conversation-like input
    inputs: Union[str, List[Dict[str, Any]]]
    parameters: Optional[TextToSpeechParameters] = None

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


# TODO: Will the `AUDIO_PATH` be required for every `text-to-speech` model, or rather just for a handful collection
# of those? In such case, should `AUDIO_PATH` be optional? If so, what should we do if `AUDIO_PATH` not provided
# but *required*, given that we don't know that in advance?
# NOTE: This pipeline has only been extensively tested for VibeVoice, this being said, it should still be considered
# experimental
# TODO: Add a `decorator` as `@experimental` to flag the experimental pipelines given that now that all the standard
# `huggingface-inference-toolkit` tasks are covered, we'll start adding support for other tasks as e.g. `text-to-speech`
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

        audios = {file.stem: audio_path / file for file in self.audio_path.glob("*.wav")}
        if len(audios) < 1:
            raise RuntimeError(
                f"The provided `AUDIO_PATH={audio_path}` does not contain any audio (wav) file, hence it's not valid as it doesn't contain the required audio files for the voices."
            )

        from transformers.audio_utils import load_audio_librosa

        self.audios = {k: (v, load_audio_librosa(v.as_posix(), sampling_rate=24000)) for k, v in audios.items()}

        self.default_audio = os.getenv("DEFAULT_AUDIO", None)
        if self.default_audio and self.default_audio not in self.audios:
            raise ValueError(
                f'The provided `DEFAULT_AUDIO={self.default_audio}` is not listed among the available voices within the provided `AUDIO_PATH={os.getenv("AUDIO_PATH")}`. Please make sure to unset the `DEFAULT_AUDIO` environment variable or rather set it to any of the following values instead: "'
                + '", "'.join(list(self.audios.keys()))
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
            no_processor=False,
        )

        if torch.mps.is_available():
            torch.mps.empty_cache()
            torch.mps.set_per_process_memory_fraction(0.9)

        from diffusers.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler

        self.noise_scheduler = DPMSolverMultistepScheduler(
            beta_schedule="squaredcos_cap_v2", num_train_timesteps=1000, prediction_type="v_prediction"
        )

    def __call__(self, payload: TextToSpeechInput) -> TextToSpeechOutput:
        if isinstance(payload.inputs, str):
            if payload.parameters is not None and payload.parameters.voice is not None:
                if payload.parameters.voice not in self.audios:
                    raise ValueError(
                        f'The provided `voice={payload.parameters.voice}` is not listed among the available voices within the provided `AUDIO_PATH={os.getenv("AUDIO_PATH")}`. Please use any of the following voices instead: "'
                        + '", "'.join(list(self.audios.keys()))
                        + '".'
                    )

                path, audio = self.audios[payload.parameters.voice]
            elif self.default_audio is not None:
                logger.info(
                    f"The `voice` parameter inside `parameters` hasn't been provided but the `DEFAULT_AUDIO` is set to `{self.default_audio}`, meaning that it will be used unless the `voice` in `parameters` is set to any of the following values instead: \""
                    + '", "'.join(list(self.audios.keys()))
                    + '".'
                )
                path, audio = self.audios[self.default_audio]
            else:
                voice = random.choice(list(self.audios.keys()))
                path, audio = self.audios[voice]

                logger.warning(
                    f"Given that the `voice` in `parameters` is either not provided or empty, the default `voice` will be set to `{voice}` (random selection). It's recommended that the `voice` parameter is provided as `{{'inputs':'...','parameters':{{'voice':'{voice}',...}}}}`, with any of the following values: \""
                    + '", "'.join(list(self.audios.keys()))
                    + '".'
                )

            messages = [
                {
                    "role": "0",
                    "content": [
                        {"type": "text", "text": payload.inputs},
                        {"type": "audio", "path": path.as_posix()},
                    ],
                }
            ]

            preprocess_params = {"audio": audio}
        # NOTE: Non-compliant with the current Hugging Face API, but supports conversation-like inputs too
        elif isinstance(payload.inputs, list):
            messages = payload.inputs

            paths, audios = [], []
            for message in messages:
                for content in message["content"]:
                    if content.get("type") == "audio" and "path" in content:
                        # NOTE: If the `path` (provided as *only* the filename e.g. `en-Frank_man` for VibeVoice)
                        # is not in the "processed" paths, then add it to the list to prevent from loading the
                        # audio more than once (not loading but rather creating a list of those, but given that
                        # it needs to be a set, we skip the duplicates). Then we add the `audio` to `audios` if
                        # not there already, and update the `content["path"]` to point to the file path rather than
                        # the filename.
                        path, audio = self.audios[content["path"]]
                        if content["path"] not in paths:
                            paths.append(content["path"])
                            audios.append(audio)
                        content["path"] = path.as_posix()

            preprocess_params = {"audio": audios}

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

        output = self.pipeline(
            inputs,
            preprocess_params=preprocess_params,
            generate_kwargs={"noise_scheduler": self.noise_scheduler, **parameters},
        )

        audio = output["audio"][0]
        if audio.ndim > 1:
            audio = audio.squeeze()
        sampling_rate = sr if (sr := output.get("sampling_rate", None)) else 24000

        buf = BytesIO()
        buf.name = "file.wav"
        sf.write(buf, audio, sampling_rate, format="wav")
        buf.seek(0)

        return TextToSpeechOutput(audio=buf.read(), sampling_rate=float(sampling_rate))
