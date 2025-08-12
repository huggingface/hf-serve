from io import BytesIO
import logging
import requests
from typing import List, Optional, Union

import torch
from pydantic import AliasChoices, BaseModel, ConfigDict, Field
from pydub import AudioSegment
from transformers.pipelines import pipeline

from huggingface_inference_toolkit.serde import Audio
from huggingface_inference_toolkit.tasks.predictor import Predictor


class AutomaticSpeechRecognitionGenerationParameters(BaseModel):
    temperature: Optional[float] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    typical_p: Optional[float] = None
    epsilon_cutoff: Optional[float] = None
    eta_cutoff: Optional[float] = None
    max_length: Optional[int] = None
    max_new_tokens: Optional[int] = None
    min_length: Optional[int] = None
    min_new_tokens: Optional[int] = None
    do_sample: Optional[bool] = None
    early_stopping: Optional[Union[str, bool]] = None
    num_beams: Optional[int] = None
    num_beam_groups: Optional[int] = None
    penalty_alpha: Optional[float] = None
    use_cache: Optional[bool] = None


class AutomaticSpeechRecognitionParameters(BaseModel):
    return_timestamps: Optional[Union[bool, str]] = Field(default=None)
    generation_parameters: Optional[AutomaticSpeechRecognitionGenerationParameters] = Field(default=None)


class AutomaticSpeechRecognitionInput(BaseModel):
    inputs: Union[str, bytes] = Field(validation_alias=AliasChoices("inputs", "audio"))
    parameters: Optional[AutomaticSpeechRecognitionParameters] = Field(default=None)

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "inputs": "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/1.flac",
                    "parameters": {
                        "return_timestamps": True,
                        "generation_parameters": {"temperature": 0.1, "top_k": 50},
                    },
                }
            ]
        }
    )


class Chunk(BaseModel):
    text: str
    timestamp: List[float]


class AutomaticSpeechRecognitionOutput(BaseModel):
    text: str
    chunks: Optional[List[Chunk]] = None


class AutomaticSpeechRecognition(Predictor[AutomaticSpeechRecognitionInput, AutomaticSpeechRecognitionOutput]):
    def __init__(self, model_id: str, dtype: str = "float16", device: str = "auto") -> None:
        super().__init__()

        # Handle device selection for models that don't support device_map
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"

        self.pipeline = pipeline(
            task="automatic-speech-recognition",
            model=model_id,
            torch_dtype=getattr(torch, dtype),
            device=device if device not in {"auto"} else None,
            device_map=device if device in {"auto"} else None,
        )

        if torch.mps.is_available():
            torch.mps.empty_cache()
            torch.mps.set_per_process_memory_fraction(0.9)

        # Set default chunk size to 30 seconds.
        # Compliance with whisper models max audio length.
        # Wav2Vec2 models can handle longer audio, but chunking needed for performance and for avoiding OOM errors.
        # NOTE: more chunks (lower chunk_length_s) can lead to higher error rates when assembling the final transcription.
        # TODO: came up with a dinamyc way to set chunk_length_s based on model capabilities and hardware.
        self.chunk_length_s = 30

    def __call__(self, payload: AutomaticSpeechRecognitionInput) -> AutomaticSpeechRecognitionOutput:
        parameters = {}
        if payload.parameters:
            parameters = payload.parameters.model_dump(exclude_none=True)

        # TODO (@juanjucm): Check if maybe its better to standarize how the audio is passed to the pipeline (e.g. always as a bytes, instead of bytes or url).
        audio_input = payload.inputs
        if isinstance(audio_input, str):
            if audio_input.startswith(("http://", "https://")):
                res = requests.get(audio_input)
                audio_enc = BytesIO(res.content)
            elif "." in audio_input.split("/")[-1]:
                audio_enc = audio_input
            else:
                # audio as base64 encoded string, input has to be deserialized.
                audio_input = Audio.deserialize(audio_input)
                audio_enc = BytesIO(audio_input)
        else:  # audio as bytes
            audio_enc = BytesIO(audio_input)

        audio_length = AudioSegment.from_file(audio_enc).duration_seconds
        logging.info(
            f"Audio length: {audio_length} seconds. batch size set to {int(audio_length // self.chunk_length_s + 1)}"
        )

        result = self.pipeline(
            audio_input,
            return_timestamps=parameters.get("return_timestamps", None),
            chunk_length_s=self.chunk_length_s,
            batch_size=int(audio_length // self.chunk_length_s + 1),
            generate_kwargs=parameters.get("generation_parameters", None),
        )

        return AutomaticSpeechRecognitionOutput(
            # TODO (@juanjucm): check if an empty audio should return empty text or not.
            # If not needed, remove this fallback and crash.
            text=result.get("text", ""),
            chunks=[Chunk(text=chunk["text"], timestamp=chunk["timestamp"]) for chunk in result["chunks"]]
            if result.get("chunks")
            else None,
        )
