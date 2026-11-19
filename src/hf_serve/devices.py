"""Device helpers shared across the tasks.

Besides CUDA, MPS and CPU, `hf-serve` supports AWS Trainium / Inferentia through Torch NeuronX 2.13+,
the PyTorch-native Neuron backend which exposes a regular PyTorch device named `neuron` (registered
through the `PrivateUse1` mechanism, and loaded automatically when importing `torch`).
"""

from typing import Literal

DeviceTypes = Literal["auto", "balanced", "cuda", "cpu", "mps", "neuron"]
DEVICES = ("auto", "balanced", "cuda", "cpu", "mps", "neuron")

# NOTE: Hub kernel loaded via `transformers[kernels]`, only available for CUDA
CUDA_ATTN_IMPLEMENTATION = "kernels-community/flash-attn2@v2"


def is_neuron_available() -> bool:
    import torch

    neuron = getattr(torch, "neuron", None)
    try:
        return neuron is not None and bool(neuron.is_available())
    except Exception:
        return False


def resolve_device(device: str) -> str:
    """Resolves `auto` to `neuron` when a Neuron device is the only accelerator available.

    CUDA and MPS keep `auto` (and `balanced`) as is, since the tasks rely on `device_map="auto"` there, which
    the Neuron backend does not support, so `neuron` needs to be explicitly provided to the tasks.
    """
    if device != "auto":
        return device

    import torch

    if torch.cuda.is_available() or torch.backends.mps.is_available():
        return device
    return "neuron" if is_neuron_available() else device


def default_attn_implementation(device: str) -> str:
    """Selects the attention implementation for `transformers` models loaded on the given device."""
    import torch

    if device == "mps" or (device == "auto" and torch.backends.mps.is_available()):
        return "eager"
    if device in {"auto", "balanced", "cuda"} and torch.cuda.is_available():
        return CUDA_ATTN_IMPLEMENTATION
    return "sdpa"
