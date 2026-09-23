import unittest
from threading import Lock
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from hf_serve.tasks.transformers.speaker_diarization import (
    SpeakerDiarization,
    SpeakerDiarizationInput,
    SpeakerDiarizationParameters,
)


class FakeInputs(dict):
    def __getattr__(self, name):
        return self[name]

    def to(self, device, dtype):
        return self


class FakeProcessor:
    feature_extractor = SimpleNamespace(sampling_rate=16000)
    num_samples_first_audio_chunk = 4
    num_samples_per_audio_chunk = 4
    num_mel_frames_per_step = 2

    def __init__(self):
        self.calls = []
        self.mode = None
        self.extraction_mask = None

    def set_streaming_mode(self, mode):
        self.mode = mode

    def audio_chunk_start(self, mel_frame_idx):
        return mel_frame_idx

    def __call__(self, audio, **kwargs):
        self.calls.append((len(audio), kwargs))
        return FakeInputs(attention_mask=torch.ones(1, 1, dtype=torch.bool))

    def extract_speaker_dict(self, logits, attention_mask=None):
        self.extraction_mask = attention_mask
        return [[{"Start": 0.0, "End": logits.shape[1] * 0.01, "Speaker": 0}]]


class FakeModel:
    device = "cpu"
    dtype = torch.float32

    def __init__(self):
        self.caches = []

    def __call__(self, **kwargs):
        self.caches.append(kwargs.get("speaker_cache"))
        return SimpleNamespace(
            logits=torch.ones(1, 2, 8),
            speaker_cache=len(self.caches),
        )


class SpeakerDiarizationTests(unittest.TestCase):
    def setUp(self):
        self.predictor = SpeakerDiarization.__new__(SpeakerDiarization)
        self.predictor.processor = FakeProcessor()
        self.predictor.model = FakeModel()
        self.predictor._lock = Lock()

    def predict(self, audio_length, streaming_mode=None):
        payload = SpeakerDiarizationInput(
            inputs=b"audio bytes",
            parameters=SpeakerDiarizationParameters(streaming_mode=streaming_mode),
        )
        with patch("transformers.audio_utils.load_audio", return_value=np.zeros(audio_length)) as decode:
            result = self.predictor(payload)
        decode.assert_called_once()
        self.assertEqual(decode.call_args.kwargs["sampling_rate"], 16000)
        return result

    def test_offline_uses_attention_mask_and_returns_segments(self):
        result = self.predict(11)
        self.assertEqual(result.model_dump(), {"segments": [{"start": 0.0, "end": 0.02, "speaker": 0}]})
        self.assertIsNotNone(self.predictor.processor.extraction_mask)
        self.assertEqual(self.predictor.model.caches, [None])

    def test_short_streaming_audio_is_first_and_last_chunk(self):
        result = self.predict(3, "low_latency")
        self.assertEqual(result.segments[0].end, 0.02)
        self.assertEqual(len(self.predictor.processor.calls), 1)
        self.assertTrue(self.predictor.processor.calls[0][1]["is_first_audio_chunk"])
        self.assertTrue(self.predictor.processor.calls[0][1]["is_last_audio_chunk"])

    def test_exact_first_chunk_is_also_last_chunk(self):
        self.predict(4, "very_low_latency")
        self.assertEqual(len(self.predictor.processor.calls), 1)
        self.assertTrue(self.predictor.processor.calls[0][1]["is_last_audio_chunk"])

    def test_long_streaming_audio_carries_cache_and_flushes_tail(self):
        result = self.predict(11, "ultra_low_latency")
        calls = self.predictor.processor.calls
        self.assertEqual([length for length, _ in calls], [4, 4, 4, 4, 3])
        self.assertEqual(self.predictor.model.caches, [None, 1, 2, 3, 4])
        self.assertTrue(calls[-1][1]["is_last_audio_chunk"])
        self.assertEqual(self.predictor.processor.mode, "ultra_low_latency")
        self.assertIsNone(self.predictor.processor.extraction_mask)
        self.assertEqual(result.segments[0].end, 0.1)

    def test_empty_audio_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Audio input is empty"):
            self.predict(0)


if __name__ == "__main__":
    unittest.main()
