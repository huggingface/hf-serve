import base64
from threading import Lock
from typing import Annotated, List, Literal, Optional, Union

from fastapi import Form
from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from hf_serve.serde import Audio
from hf_serve.tasks.predictor import Predictor
from hf_serve.types import FileForm

StreamingMode = Literal["low_latency", "very_low_latency", "ultra_low_latency"]


class VoiceActivityDetectionParameters(BaseModel):
    streaming_mode: Optional[StreamingMode] = None


class VoiceActivityDetectionInput(BaseModel):
    inputs: Union[str, bytes] = Field(validation_alias=AliasChoices("inputs", "audio"))
    parameters: Optional[VoiceActivityDetectionParameters] = None

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "inputs": (
                        "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/"
                        "resolve/main/diarization_example.mp3"
                    ),
                }
            ]
        }
    )


class VoiceActivityDetectionFormInput(BaseModel):
    file: FileForm
    streaming_mode: Optional[Annotated[StreamingMode, Form()]] = None

    model_config = ConfigDict(extra="forbid")


class SpeakerSegment(BaseModel):
    start: float = Field(validation_alias=AliasChoices("Start", "start"))
    end: float = Field(validation_alias=AliasChoices("End", "end"))
    speaker: int = Field(validation_alias=AliasChoices("Speaker", "speaker"))


class VoiceActivityDetectionOutput(BaseModel):
    segments: List[SpeakerSegment]


class VoiceActivityDetection(Predictor[VoiceActivityDetectionInput, VoiceActivityDetectionOutput]):
    def __init__(
        self,
        model_id: str,
        revision: Optional[str] = None,
        dtype: Optional[str] = None,
        device: str = "auto",
        trust_remote_code: bool = False,
    ) -> None:
        super().__init__()

        import torch
        from transformers import AutoModelForAudioFrameClassification, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(
            model_id, revision=revision, trust_remote_code=trust_remote_code
        )
        self.model = AutoModelForAudioFrameClassification.from_pretrained(
            model_id,
            revision=revision,
            device_map=device,
            dtype=getattr(torch, dtype) if dtype is not None else "auto",
            trust_remote_code=trust_remote_code,
        )
        # set_streaming_mode changes processor state, so requests must not interleave.
        self._lock = Lock()

    def _streaming_inputs(self, audio, sampling_rate):
        first_chunk_size = self.processor.num_samples_first_audio_chunk
        if len(audio) <= first_chunk_size:
            yield self.processor(
                audio,
                sampling_rate=sampling_rate,
                is_streaming=True,
                is_first_audio_chunk=True,
                is_last_audio_chunk=True,
            )
            return

        yield self.processor(
            audio[:first_chunk_size],
            sampling_rate=sampling_rate,
            is_streaming=True,
            is_first_audio_chunk=True,
        )

        mel_frame_idx = self.processor.num_mel_frames_per_step
        start_idx = self.processor.audio_chunk_start(mel_frame_idx)
        while (end_idx := start_idx + self.processor.num_samples_per_audio_chunk) <= len(audio):
            yield self.processor(
                audio[start_idx:end_idx],
                sampling_rate=sampling_rate,
                is_streaming=True,
                is_first_audio_chunk=False,
            )
            mel_frame_idx += self.processor.num_mel_frames_per_step
            start_idx = self.processor.audio_chunk_start(mel_frame_idx)

        yield self.processor(
            audio[start_idx:],
            sampling_rate=sampling_rate,
            is_streaming=True,
            is_first_audio_chunk=False,
            is_last_audio_chunk=True,
        )

    def __call__(self, payload: VoiceActivityDetectionInput) -> VoiceActivityDetectionOutput:
        import torch
        from transformers.audio_utils import load_audio

        audio_bytes = payload.inputs if isinstance(payload.inputs, bytes) else Audio.deserialize(payload.inputs)
        sampling_rate = self.processor.feature_extractor.sampling_rate
        # load_audio accepts base64 strings and resamples to the processor's rate.
        audio = load_audio(base64.b64encode(audio_bytes).decode("ascii"), sampling_rate=sampling_rate)
        if len(audio) == 0:
            raise ValueError("Audio input is empty")

        streaming_mode = payload.parameters.streaming_mode if payload.parameters else None
        with self._lock, torch.inference_mode():
            if streaming_mode is None:
                inputs = self.processor(audio, sampling_rate=sampling_rate).to(
                    self.model.device, dtype=self.model.dtype
                )
                logits = self.model(**inputs).logits
                segments = self.processor.extract_speaker_dict(logits, inputs.attention_mask)[0]
            else:
                self.processor.set_streaming_mode(streaming_mode)
                speaker_cache = None
                logits_chunks = []
                for inputs in self._streaming_inputs(audio, sampling_rate):
                    inputs = inputs.to(self.model.device, dtype=self.model.dtype)
                    outputs = self.model(**inputs, speaker_cache=speaker_cache)
                    logits_chunks.append(outputs.logits)
                    speaker_cache = outputs.speaker_cache
                logits = torch.cat(logits_chunks, dim=1)
                segments = self.processor.extract_speaker_dict(logits)[0]

        return VoiceActivityDetectionOutput(
            segments=[SpeakerSegment.model_validate(segment) for segment in segments]
        )
