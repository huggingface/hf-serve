# `hf-serve`

> [!WARNING]
> This project is still experimental, meant to replace the former
> [`huggingface-inference-toolkit`](https://github.com/huggingface/huggingface-inference-toolkit).

## Installation

First you need to setup your environment with [`uv`](https://github.com/astral-sh/uv) (or with your preferred Python environment manager).

```bash
uv venv --python 3.12
source .venv/bin/activate
```

> [!NOTE]
> Due to the need of `--preview-features extra-build-dependencies` to install
> [`flash-attn`](https://github.com/Dao-AILab/flash-attention) with `uv` without
> compiling it, but rather relying on the pre-built binaries, you need to use
> `uv` v0.8.13 (or higher, but beware on major updates since the feature is
> still experimental, so < v0.9.0 is recommended until stable).
>
> Reference: https://docs.astral.sh/uv/concepts/projects/config/#augmenting-build-dependencies
>
> To update `uv` once installed if `uv version` is lower than v0.8.13, simply
> `uv self update`.

Install it from the `uv.lock` file for CPU / MPS as follows:

```bash
uv sync --active --frozen --extra cpu
```

Alternatively, install it on NVIDIA CUDA 12.6 as follows:

```bash
uv sync --active --frozen --extra cuda --extra flash-attn --preview-features extra-build-dependencies
```

Or if you're on CUDA 12.8 then:

```bash
uv sync --active --frozen --extra cuda-128 --extra flash-attn --preview-features extra-build-dependencies
```

> [!NOTE]
> There's no `cuda-130` (CUDA 13.0) extra for now, as `flash-attn==2.8.3`'s
> wheel-detection logic predates CUDA 13 and would silently install a mismatched
> `cu12` wheel that fails at import time. It'll be added back once upstream fixes this.

> [!WARNING]
> The default registry for the NVIDIA CUDA wheels for PyTorch is set to CUDA 12.6. If
> you want to install another PyTorch version as per the CUDA compatibility, then
> run e.g. `uv pip install -e . --torch-backend cu128`, but note it won't be relying
> on the `uv.lock` so some dependencies might mismatch.
>
> Reference: https://docs.astral.sh/uv/guides/integration/pytorch/#automatic-backend-selection

```console
$ uv run hf-serve --help
```

## Examples

> [!NOTE]
> On the examples below, given the recently introduced `extra-build-dependencies`
> for `flash-attn` on CUDA as per https://docs.astral.sh/uv/concepts/projects/config/#build-isolation,
> it means that you'll need to run the examples as `uv run --preview-features extra-build-dependencies ...`
> to disable the warning:
> ```console
> warning: The `extra-build-dependencies` option is experimental and may change without warning. Pass `--preview-features extra-build-dependencies` to disable this warning.
> ```

### 🤏 Run `HuggingFaceTB/SmolLM3-3B` with an OpenAI API

```bash
uv run hf-serve --model-id HuggingFaceTB/SmolLM3-3B --task text-generation --dtype float16
```

> [!NOTE]
> If you are running on an instance with NVIDIA GPU, it's recommended to install `hf-serve`
> with `flash-attn` extra in order to benefit from accelerated inference:
> ```bash
> uv sync --active --frozen --extra cuda --extra flash-attn --preview-features extra-build-dependencies
> ```

### 🔵 Run `sentence-transformers/all-MiniLM-L6-v2` on Azure AI

```bash
uv run hf-serve --model-id sentence-transformers/all-MiniLM-L6-v2 --task sentence-similarity --dtype float32 --cloud azure
```

> [!WARNING]
> Given that Azure AI Foundry and Azure ML expect the inference route to be `/score`
> rather than `/predict`, which is the standard for Inference Endpoints API, and since
> `/score` is a redirect to `/predict`, then we need to send the `curl` request with the
> `-L/--location` flag so that it follows the redirect, otherwise we get an HTTP 307.
>
> ```bash
> curl -L http://localhost:8080/score -H "Content-Type: application/json" -d '{"inputs":{"source_sentence":"What is Deep Learning?","sentences":["Deep Learning is...","Deep Learning is not..."]}}'
> ```

### 👂 Run `facebook/wav2vec2-base-960h` an `automatic-speech-recognition` model

> [!NOTE]
> Before running `automatic-speech-recognition` or really any of `audio-classification`
> or `zero-shot-audio-classification` you will need to install some system dependencies
> in advance for those to work as `ffmpeg` and `libmagic-dev`.

```bash
uv run hf-serve --model-id facebook/wav2vec2-large-960h --task automatic-speech-recognition --dtype float16
```

> [!WARNING]
> On MacOS, if you installed `ffmpeg` via `brew`, you will need to set the following
> environment variable in advance `DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib`
>
> Reference: https://github.com/pytorch/torchcodec/issues/570#issuecomment-2913609176

And, then you can send a sample request as:

```bash
curl -L http://localhost:8080/predict \
    -H "Content-Type: application/json" \
    -d '{"inputs":"https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/1.flac"}'
```

> [!WARNING]
> Given the nature of some tasks that need to support JSON, forms, and files,
> the `/predict` method for those is a redirect to the respective inner endpoint:
> `/predict-json`, `/predict-form`, and `/predict-file`. Those are non-standard
> but required to keep full compatibility with the current Hugging Face Inference
> Endpoints API Specification, but in reality the redirect response (HTTP 307)
> shouldn't be used as an routing route, but rather dedicated routes for those.

> [!NOTE]
> The OpenAI Audio Transcriptions API is still not yet part of `hf-serve` but it's
> on the roadmap and it will be released soon, stay tuned!

### 🔈 Run `facebook/wav2vec2-lv-60-espeak-cv-ft` (with `phonemizer` and `espeak`)

> [!NOTE]
> Some models as e.g. `facebook/wav2vec2-lv-60-espeak-cv-ft`, rely on `phonemizer` for
> the "phonemization" of words and texts in many languages, based at the same time on
> different Text-To-Speech (TTS) backends as e.g. `espeak-ng` which is supports a lot
> of languages and IPA (International Phonetic Alphabet). This being said, such models
> require custom dependencies that need to be installed beforehand as those don't come
> as default `hf-serve` dependencies; whilst those can be installed as e.g. on MacOS:
>
> ```bash
> brew install ffmpeg
> brew install espeak
> ```

> [!WARNING]
> Beware that when installing `ffmpeg` with `brew` on a specific
> version as e.g. `brew install ffmpeg@7` as it will be installed as "keg-only",
> meaning that it won't be symlinked into `/opt/homebrew`, meaning that the path
> to the library won't be `/opt/homebrew/lib` but rather `/opt/homebrew/opt/ffmpeg/lib`
> instead, meaning that on MacOS you'll need to set `DYLD_FALLBACK_LIBRARY_PATH`
> to wherever the `ffmpeg` library is installed in.

To run `facebook/wav2vec2-lv-60-espeak-cv-ft` on e.g. MacOS, you need to run the following:

```bash
DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib uv run hf-serve --model-id facebook/wav2vec2-lv-60-espeak-cv-ft --task automatic-speech-recognition --dtype float16 --device mps
```

Note that if you have installed another version of `ffmpeg` with `brew` as e.g. `brew install ffmpeg@7`, you should use the following command instead:

```bash
DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/opt/ffmpeg@7/lib uv run hf-serve --model-id facebook/wav2vec2-lv-60-espeak-cv-ft --task automatic-speech-recognition --dtype float16 --device mps
```

The main difference relies on the path used for `DYLD_FALLBACK_LIBRARY_PATH` which is now pointing to the exact `brew`-installed version of `ffmpeg` instead. More information on the compatibility issues with `ffmpeg`, `torchcodec` and `torch` at https://github.com/meta-pytorch/torchcodec?tab=readme-ov-file#installing-torchcodec.
