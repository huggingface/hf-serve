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

### 🤏🏻 Run SmolLM3 with an OpenAI API

```bash
uv run hf-serve --model-id HuggingFaceTB/SmolLM3-3B --task text-generation --dtype float16
```

## 🔮 Upcoming

- [ ] Rewrite the CLI to support task-specific arguments e.g. `hf-serve sentence-similarity --model-id sentence-transformers/all-MiniLM-L6-v2 --similarity-fn-name cosine ...`

- [ ] Add support for OpenAI Responses API for `text-generation`

- [ ] Add a memory estimation tool prior loading the model to identify whether the model will fit or not and provide the user with meaningful feedback on the requirements for the given model
