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

Or for CUDA 13.0, including NVIDIA GB10 / DGX Spark:

```bash
uv sync --active --frozen --extra cuda-130
```

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

## Container images

On each push to `main`, CI publishes images to `ghcr.io/<owner>/hf-serve` using
the first seven characters of the commit SHA as the tag. Images are built on
native runners for the architectures listed below:

| Tag | Backend | Architectures |
| --- | --- | --- |
| `<sha>` | CUDA 12.6 | `linux/amd64`, `linux/arm64` |
| `<sha>-cu130-arm64` | CUDA 13.0, tested on NVIDIA GB10 / DGX Spark | `linux/arm64` |
| `<sha>-cpu` | CPU | `linux/amd64`, `linux/arm64` |

For example, to pull the CUDA 13.0 image published by the upstream repository:

```bash
docker pull ghcr.io/huggingface/hf-serve:<sha>-cu130-arm64
```

Use a commit published after multi-platform support was added; older tags may
only contain AMD64 images. Docker selects the architecture for the host.

To build the CUDA 13.0 variant locally from this checkout:

```bash
docker buildx build --platform linux/arm64 \
  --build-arg BASE_IMAGE=nvidia/cuda:13.0.2-devel-ubuntu24.04 \
  --build-arg TORCH_EXTRA=cuda-130 \
  --tag hf-serve:gb10 --load .
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

### 📏 Run `google/embeddinggemma-2`

[EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2) supports text,
images, audio, video, and interleaved inputs through Sentence Transformers.
It requires Transformers 5.19+ and Sentence Transformers 6.1+.

```bash
uv run hf-serve --model-id google/embeddinggemma-2 --task feature-extraction
```

The default precision is bfloat16 on CUDA devices that support it and float32
elsewhere. Use `--dtype float32` or `--dtype bfloat16` to override it; float16
is rejected because it produces invalid embeddings for this model.

Choose `SearchQuery` for search queries and `Document` for untitled documents.
Other prompt names, including `CodeRetrieval` and `SentenceSimilarity`, are
listed in the model card. No task prefix is added unless a prompt is selected.

```bash
curl http://localhost:8080/predict -H 'Content-Type: application/json' \
  -d '{"inputs":"What causes the northern lights?","prompt_name":"SearchQuery","dimensions":256}'

curl http://localhost:8080/v1/embeddings -H 'Content-Type: application/json' \
  -d '{"model":"google/embeddinggemma-2","input":"The northern lights are caused by charged particles from the sun.","prompt_name":"Document","dimensions":256}'
```

Both routes accept a single modality dictionary or a batch of dictionaries as
well as the existing text strings. Media values can be URLs or server-local
paths; images also support base64. A list of media values inside one dictionary
produces one shared embedding. Use the model's placeholder tokens to position
media within text:

```bash
curl http://localhost:8080/predict -H 'Content-Type: application/json' \
  -d '{"inputs":{"text":"Waterproof running shoes. <|image|> Grip test: <|video|>","image":"https://example.com/shoe.jpg","video":"/data/grip-test.mp4"},"dimensions":512}'
```

Use `image`, `audio`, or `video` dictionaries without a prompt for media-only
inputs. On Google Cloud, the same inputs can be supplied in `instances`.
The native vector has 768 dimensions; the model supports shortening it to 512,
256, or 128. Outputs are L2-normalized after truncation by default. `/predict`
also accepts `normalize: false` to disable that final normalization.

Media sampling and truncation follow the model's processor configuration. In
particular, video defaults to 1 FPS with at most 32 sampled frames per video;
accepting a long audio file does not guarantee that the whole recording is
represented. This API currently adds no separate media-count or context-budget
validation.

To run the CPU integration tests against the published weights in an activated
environment:

```bash
hf download google/embeddinggemma-2 --local-dir /tmp/embeddinggemma-2
HF_SERVE_EMBEDDINGGEMMA2_MODEL=/tmp/embeddinggemma-2 python -m unittest discover -s tests -v
```

Without that environment variable, the published-weight integration tests are
skipped and the regression tests run without downloading a model.

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

### 🟠 Run `sentence-transformers/all-MiniLM-L6-v2` on Amazon SageMaker

```bash
uv run hf-serve --model-id sentence-transformers/all-MiniLM-L6-v2 --task sentence-similarity --dtype float32 --cloud sagemaker
```

> [!NOTE]
> When `--cloud sagemaker` is set, middleware internally routes `GET /ping` to
> `/health` and `POST /invocations` to `/predict`. To invoke another POST route,
> set `route` in `X-Amzn-SageMaker-Custom-Attributes`, for example
> `route=/v1/chat/completions`. Unknown routes are rejected with HTTP 400.
> When deploying `Dockerfile.sagemaker`, set `TASK` in the SageMaker model's
> container environment. Set `MODEL_ID` to load from the Hub. If neither
> `MODEL_ID` nor `MODEL_DIR` is provided, model artifacts extracted by SageMaker
> into `/opt/ml/model` are selected automatically.
>
> ```bash
> curl http://localhost:8080/ping
> curl http://localhost:8080/invocations -H "Content-Type: application/json" -d '{"inputs":{"source_sentence":"What is Deep Learning?","sentences":["Deep Learning is...","Deep Learning is not..."]}}'
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

### Run voice activity detection

The `voice-activity-detection` task supports diarization-capable audio frame
classification models whose processor exposes `extract_speaker_dict`. It returns
speaker segments with `start`, `end` (seconds), and `speaker` (arrival-order index):

```bash
uv run hf-serve --model-id your-org/your-model --task voice-activity-detection
```

Send an audio URL, base64-encoded audio, or an audio file through the same JSON,
form, or binary endpoints used by ASR:

```bash
curl -L http://localhost:8080/predict-json \
    -H "Content-Type: application/json" \
    -d '{"inputs":"https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/diarization_example.mp3"}'
```

Offline inference is the default. To process the recording in streaming chunks
within one request, add `"parameters":{"streaming_mode":"low_latency"}` to the
JSON body, or send `streaming_mode=low_latency` with a form upload. The other
supported modes are `very_low_latency` and `ultra_low_latency`. Each request has
its own speaker cache; the API does not retain a streaming session across requests.

On Google Cloud, run with `--cloud google` and send audio strings in `instances`.
The same `parameters` apply to every instance, and the response contains one
`predictions` entry per instance:

```json
{"instances": ["<base64-encoded audio>"], "parameters": {"streaming_mode": "low_latency"}}
```

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
