# `hf-serve`

> [!WARNING]
> This project is still an experimental and early attempt of refactoring the former
> [`huggingface-inference-toolkit`](https://github.com/huggingface/huggingface-inference-toolkit),
> and it might ship breaking changes until stable.

## 🛠️ Installation

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e .
```

## 💻 Example

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

## 🔮 Upcoming

- [ ] Rewrite the CLI to support task-specific arguments e.g. `hf-serve sentence-similarity --model-id sentence-transformers/all-MiniLM-L6-v2 --similarity-fn-name cosine ...`

- [ ] Add support for OpenAI Responses API for `text-generation`

- [ ] Add a memory estimation tool prior loading the model to identify whether the model will fit or not and provide the user with meaningful feedback on the requirements for the given model

- [ ] Improve error messages when input data validation fails, to make those more readable as default Pydantic errors are not so easy to read and there's not a clear action for the user
