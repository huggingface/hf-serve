import os

from huggingface_inference_toolkit.logging import logger


def check_backwards_compatibility() -> None:
    mappings = {
        "HF_MODEL_DIR": "MODEL_DIR",
        "HF_MODEL_ID": "MODEL_ID",
        "HF_TASK": "TASK",
        "HF_REVISION": "REVISION",
        "HF_HUB_TOKEN": "HF_TOKEN",
        "HF_MODULE_NAME": "CUSTOM_HANDLER",
        "HF_DEFAULT_PIPELINE_NAME": "CUSTOM_HANDLER_FILE",
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

    # NOTE: for the custom handler we need to set those values by default if not provided, otherwise,
    # the provided values will be used
    if "CUSTOM_HANDLER_FILE" not in os.environ:
        os.environ["CUSTOM_HANDLER_FILE"] = "handler.py"

    if "CUSTOM_HANDLER" not in os.environ:
        # NOTE: this value used to be provided as the full path to the `EndpointHandler` or whatever name it
        # had, but since the `CUSTOM_HANDLER_FILE` is a single file, it doesn't make much sense to provided the
        # stem path i.e. `handler.EndpointHandler`, as that will be automatically defined internally already
        os.environ["CUSTOM_HANDLER"] = "EndpointHandler"
