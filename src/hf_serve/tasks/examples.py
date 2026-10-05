import base64
import math
import struct
import wave
from io import BytesIO

from PIL import Image as ImageModule


def _image_base64(size: int = 64) -> str:
    # NOTE: a gradient rather than a solid color, so that the image has some content to process
    image = ImageModule.new("RGB", (size, size))
    image.putdata(
        [(x * 255 // (size - 1), y * 255 // (size - 1), 128) for y in range(size) for x in range(size)]
    )
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def _audio_base64(duration_s: float = 1.0, sampling_rate: int = 8000, frequency: float = 440.0) -> str:
    # NOTE: a low-volume sine tone encoded as a mono 16-bit PCM WAV, which can be decoded by `ffmpeg`, `pydub`
    # and `transformers.audio_utils.load_audio` alike; the audio pipelines resample it to the model's sampling rate
    num_samples = int(duration_s * sampling_rate)
    frames = b"".join(
        struct.pack("<h", int(0.1 * 32767 * math.sin(2 * math.pi * frequency * i / sampling_rate)))
        for i in range(num_samples)
    )
    buffered = BytesIO()
    with wave.open(buffered, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sampling_rate)
        wav.writeframes(frames)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


IMAGE_BASE64 = _image_base64()
AUDIO_BASE64 = _audio_base64()
