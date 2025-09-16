import logging
import warnings
from io import BytesIO
from typing import Annotated, Any, Dict, List, Literal, Optional, Union

from fastapi import Form
from pydantic import AliasChoices, BaseModel, ConfigDict, Field

warnings.filterwarnings("ignore", module="pydub")
from pydub import AudioSegment

from hf_serve.serde.audio import Audio
from hf_serve.tasks.predictor import Predictor
from hf_serve.types import BoolForm, FileForm, FloatForm, IntForm


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
    early_stopping: Optional[Literal["never", True, False]] = None
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


class AutomaticSpeechRecognitionFormInput(BaseModel):
    file: FileForm
    temperature: Optional[FloatForm] = None
    top_k: Optional[IntForm] = None
    top_p: Optional[FloatForm] = None
    typical_p: Optional[FloatForm] = None
    epsilon_cutoff: Optional[FloatForm] = None
    eta_cutoff: Optional[FloatForm] = None
    max_length: Optional[IntForm] = None
    max_new_tokens: Optional[IntForm] = None
    min_length: Optional[IntForm] = None
    min_new_tokens: Optional[IntForm] = None
    do_sample: Optional[BoolForm] = None
    early_stopping: Optional[Annotated[Literal["never", True, False], Form()]] = None
    num_beams: Optional[IntForm] = None
    num_beam_groups: Optional[IntForm] = None
    penalty_alpha: Optional[FloatForm] = None
    use_cache: Optional[BoolForm] = None

    model_config = ConfigDict(extra="forbid")


class Chunk(BaseModel):
    text: str
    timestamp: List[float]


class AutomaticSpeechRecognitionOutput(BaseModel):
    text: str
    chunks: Optional[List[Chunk]] = None


# TODO: given that for audio we need quite some specific stuff, should we install those
# libraries on the fly instead? e.g., `apt-get install espeak-ng`, `uv pip install phonemizers`, ...
class AutomaticSpeechRecognition(Predictor[AutomaticSpeechRecognitionInput, AutomaticSpeechRecognitionOutput]):
    def __init__(self, model_id: str, dtype: str = "float16", device: str = "auto") -> None:
        super().__init__()

        import torch
        from transformers import pipeline
        from transformers.pipelines.automatic_speech_recognition import AutomaticSpeechRecognitionPipeline

        # NOTE: Apparently some (not all) models don't support the `device_map=auto` so we should probably
        # either add a check or just default to CUDA instead
        if device == "auto":
            # e.g. DistilBertForSequenceClassification won't support it
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"

        self.pipeline: AutomaticSpeechRecognitionPipeline = pipeline(
            task="automatic-speech-recognition",
            model=model_id,
            dtype=getattr(torch, dtype),
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

        # NOTE: Handle different input types: bytes, URL, file path, or base64; no need to handle others given that
        # `Pydantic` already validates that the `inputs` is either `bytes` or `str`
        if isinstance(payload.inputs, bytes):
            audio_bytes = payload.inputs
        elif isinstance(payload.inputs, str):
            audio_bytes = Audio.deserialize(payload.inputs)

        audio_length = AudioSegment.from_file(BytesIO(audio_bytes)).duration_seconds  # type: ignore

        # TODO: eventually look into a context logger so that everything logged within the `__call__` method
        # of the task also containts the request ID to identify each request
        logging.info(
            f"Audio length: {audio_length} seconds. batch size set to {int(audio_length // self.chunk_length_s + 1)}"
        )

        result: Dict[str, Any] = self.pipeline(
            audio_bytes,  # type: ignore
            return_timestamps=parameters.get("return_timestamps", None),
            chunk_length_s=self.chunk_length_s,
            batch_size=int(audio_length // self.chunk_length_s + 1),
            generate_kwargs=parameters.get("generation_parameters", {}),
        )

        return AutomaticSpeechRecognitionOutput(
            # TODO (@juanjucm): check if an empty audio should return empty text or crash. For now, it always returns something.
            text=result["text"],  # type: ignore
            chunks=[Chunk(text=chunk["text"], timestamp=chunk["timestamp"]) for chunk in result["chunks"]]
            if result.get("chunks")
            else None,
        )
