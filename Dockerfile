FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04
LABEL maintainer="Hugging Face"

SHELL ["/bin/bash", "-c"]

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:/root/.local/bin/:$PATH"

RUN uv python install 3.12

RUN uv venv /venv --python 3.12
ENV VIRTUAL_ENV=/venv \
    PATH="/venv/bin:$PATH"

WORKDIR /app
COPY . .

RUN uv sync --frozen && uv pip install .[cuda]
ENV HF_HUB_USER_AGENT_ORIGIN=azure:foundry:gpu-cuda:inference:huggingface-inference-toolkit

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
