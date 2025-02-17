import os
from pathlib import Path

from huggingface_inference_toolkit.logging import logger


def check_backwards_compatibility() -> None:
    mappings = {
        "HF_MODEL_DIR": "MODEL_DIR",
        "HF_MODEL_ID": "MODEL_ID",
        "HF_TASK": "TASK",
        "HF_REVISION": "REVISION",
        "HF_HUB_TOKEN": "HF_TOKEN",
        # TODO: this one most likely should be deprecated, IIRC Google needed it, but it
        # shouldn't be allowed publicly in Inference Endpoints
        # "HF_TRUST_REMOTE_CODE": "TRUST_REMOTE_CODE",
    }

    for old_key, new_key in mappings.items():
        if old_key in os.environ:
            logger.warning(
                f"Environment variable '{old_key}' is deprecated. Please use '{new_key}' instead. Remapping '{old_key}' to '{new_key}'."
            )
            if new_key not in os.environ:
                os.environ[new_key] = os.environ[old_key]

    if "HF_FRAMEWORK" in os.environ:
        logger.warning(
            "Environment variable 'HF_FRAMEWORK' is deprecated and now defaults to 'torch'. The provided value is ignored."
        )
        os.environ["FRAMEWORK"] = "torch"

    # TODO: "HF_DEFAULT_PIPELINE_NAME"
    if "CUSTOM_HANDLER" not in os.environ:
        handler_file = os.environ.get("CUSTOM_HANDLER_FILE", "handler.py")
        os.environ["CUSTOM_HANDLER"] = f"{Path(handler_file).stem}.EndpointHandler"
