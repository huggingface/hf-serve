import json
from pathlib import Path
from typing import Optional


def resolve_embedding_dtype(
    model_id: str, revision: Optional[str], dtype: Optional[str], device: Optional[str]
) -> str:
    import torch
    from transformers import PretrainedConfig

    model_dir = Path(model_id)
    config_ids = [model_id]
    modules_file = model_dir / "modules.json"
    if model_dir.is_dir() and modules_file.exists():
        # Saved pipelines may keep the backbone alongside Pooling/Dense configs.
        config_ids.extend(
            str(model_dir / module["path"]) for module in json.loads(modules_file.read_text()) if module["path"]
        )

    for config_id in config_ids:
        try:
            # Read metadata only: a valid Sentence Transformers module need not have
            # a Transformers config class (or even a model_type field).
            config, _ = PretrainedConfig.get_config_dict(config_id, revision=revision)
        except OSError:
            continue
        if config.get("model_type") in {"embedding_gemma2", "embedding_gemma2_text"}:
            break
    else:
        return dtype or "auto"

    if dtype == "float16":
        raise ValueError("EmbeddingGemma 2 does not support float16; use float32 or bfloat16")
    if dtype is not None:
        return dtype

    # CPU and MPS use float32; supported CUDA devices can halve memory with bfloat16.
    if device not in {"cpu", "mps"} and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return "bfloat16"
    return "float32"
