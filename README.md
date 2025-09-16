# `hf-serve`

> [!WARNING]
> This project is still an experimental and early attempt of refactoring the former
> [`huggingface-inference-toolkit`](https://github.com/huggingface/huggingface-inference-toolkit),
> and it might ship breaking changes until stable.

## 🛠️ Installation

First you need to setup your environment with [`uv`](https://github.com/astral-sh/uv),
or with your preferred Python environment manager.

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

Alternatively, install it on NVIDIA CUDA as follows:

```bash
uv sync --active --frozen --extra cuda --extra flash-attn --preview-features extra-build-dependencies
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
usage: hf-serve [-h] [--host HOST] [--port PORT] [--model-id MODEL_ID] [--model-dir MODEL_DIR]
                [--task {image-text-to-text,text-generation,text2text-generation,conversational,chat-completion,text-to-image,sentence-similarity,feature-extraction,sentence-embeddings,embeddings,text-ranking,sentence-ranking,text-classification,fill-mask,question-answering,summarization,zero-shot-classification,token-classification,table-question-answering,translation,translation_xx_to_yy,zero-shot-audio-classification,audio-classification,automatic-speech-recognition,image-classification,image-segmentation,object-detection,custom}]
                [--device {auto,balanced,cuda,cpu,mps}] [--dtype {float32,float16,bfloat16,float8,int8,int4}]

Hugging Face Serve API

options:
  -h, --help            show this help message and exit
  --host HOST           The host into which the FastAPI API will be deployed to, defaults to 0.0.0.0, can also be set via the environment variable `HOST`
  --port PORT           The port in which the FastAPI API will listen to, defaults to 8080, can also be set via the environment variable `PORT`
  --model-id MODEL_ID   The model ID on the Hugging Face Hub, can also be set via the environment variable `MODEL_ID`
  --model-dir MODEL_DIR
                        A local directory that contains a Hugging Face compatible model, can also be set via the environment variable `MODEL_DIR`
  --task {image-text-to-text,text-generation,text2text-generation,conversational,chat-completion,text-to-image,sentence-similarity,feature-extraction,sentence-embeddings,embeddings,text-ranking,sentence-ranking,text-classification,fill-mask,question-answering,summarization,zero-shot-classification,token-classification,table-question-answering,translation,translation_xx_to_yy,zero-shot-audio-classification,audio-classification,automatic-speech-recognition,image-classification,image-segmentation,object-detection,custom}
                        Any of the supported tasks for either Transformers, Diffusers, or Sentence Transformers, can also be set via the environment variable `TASK`
  --device {auto,balanced,cuda,cpu,mps}
                        The device on which the model weights will be loaded into, defaults to auto that selects an accelerator if available, otherwise it falls back to the CPU, can also be set via the
                        environment variable `DEVICE`
  --dtype {float32,float16,bfloat16,float8,int8,int4}
                        The PyTorch dtype in which the model weights will be loaded, defaults to `float16`, can also be set via the environment variable `DTYPE`
```

## 💻 Examples

> [!NOTE]
> On the examples below, given the recently introduced `extra-build-dependencies`
> for `flash-attn` on CUDA as per https://docs.astral.sh/uv/concepts/projects/config/#build-isolation,
> it means that you'll need to run the examples as `uv run --preview-features extra-build-dependencies`
> to disable the warning:
> ```console
> warning: The `extra-build-dependencies` option is experimental and may change without warning. Pass `--preview-features extra-build-dependencies` to disable this warning.
> ```

### 🤏 Run `HuggingFaceTB/SmolLM3-3B` with an OpenAI API

```bash
uv run hf-serve --model-id HuggingFaceTB/SmolLM3-3B --task text-generation --dtype float16
```

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
> uv pip install phonemizer --upgrade
> ```

To run `facebook/wav2vec2-lv-60-espeak-cv-ft` on e.g. MacOS, you need to run the following:

```bash
DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib uv run hf-serve --model-id facebook/wav2vec2-lv-60-espeak-cv-ft --task automatic-speech-recognition --dtype float16 --device mps
```

## 🔮 Upcoming

- [ ] Rewrite the CLI to support task-specific arguments e.g. `hf-serve sentence-similarity --model-id sentence-transformers/all-MiniLM-L6-v2 --similarity-fn-name cosine ...`

- [ ] Add support for OpenAI Responses API for `text-generation`

- [ ] Add a memory estimation tool prior loading the model to identify whether the model will fit or not and provide the user with meaningful feedback on the requirements for the given model

- [ ] Improve error messages when input data validation fails, to make those more readable as default Pydantic errors are not so easy to read and there's not a clear action for the user

- [ ] Validate that the given model can be loaded with the provided `--task` otherwise fail and suggest the `--task` value
