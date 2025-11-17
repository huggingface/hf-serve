import os
from io import BytesIO
from pathlib import Path
from typing import Literal, Optional, Union

import soundfile as sf
from pydantic import AliasChoices, AliasPath, BaseModel, ConfigDict, Field

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


class TextToSpeechInput(BaseModel):
    inputs: str
    parameters: Optional[TextToSpeechParameters] = None

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {"inputs": "What is the capital of France? Paris is the capital of France.", "parameters": None}
            ]
        }
    )


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
                "The provided `AUDIO_PATH` doesn't exist. Please make sure you provide an audio path that exists and contains at least one wav file inside for the default voice of the `text-to-speech` / `tts` model."
            )

        if len([file for file in self.audio_path.glob("*.wav")]) < 1:
            raise RuntimeError(
                "The provided `AUDIO_PATH` doesn't contain any valid audio (wav) file, required for the `text-to-speech` / `tts` model to generate the audio."
            )

        self.voices = {file.stem: audio_path / file for file in self.audio_path.glob("*.wav")}
        if len(self.voices) < 1:
            raise RuntimeError(
                "The provided `AUDIO_PATH` does not contain any audio (wav) file, hence it's not valid as it doesn't contain the required audio files for the voices."
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
        parameters = {}
        if payload.parameters:
            parameters = payload.parameters.model_dump(exclude_none=True)

        output = self.pipeline(
            payload.inputs, generate_kwargs={"noise_scheduler": self.noise_scheduler, **parameters}
        )
        audio = output["audio"][0].squeeze()

        buf = BytesIO()
        buf.name = "file.wav"
        sf.write(buf, audio, 24000, format="wav")
        buf.seek(0)

        return TextToSpeechOutput(audio=buf.read(), sampling_rate=24000.0)
