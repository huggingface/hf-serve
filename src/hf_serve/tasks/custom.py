import importlib.util
import os
from pathlib import Path
from typing import Any


class Custom:
    @staticmethod
    def load(model_dir: str = "/repository") -> Any:
        """Loads a custom handler from the given repository if it exists. The custom handler file should follow
        the following signature (at least).

        ```python
        from typing import Any, Dict

        class EndpointHandler:
            def __init__(self, model_dir: str, **kwargs: Any) -> None:
                ...

            def __call__(self, data: Dict[str, Any]) -> Any:
                ...
        ```

        Note that the implementation above, and so on the default custom implementation supported within the
        `Custom` class is not fully aligned with the current `hf-serve` leveraging all the features and
        improvements included here, but rather based on the former `huggingface-inference-toolkit`.

        To see some of the former `handler.py` models created on the Hub you can check the following:
        https://huggingface.co/models?other=endpoints-template&sort=trending. Note that those are not ALL the
        models with a custom handler on the Hugging Face Hub, but just the ones with the `endpoints-template`
        tag set.
        """
        model_dir = Path(model_dir)  # type: ignore

        # NOTE: `CUSTOM_HANDLER_FILE` will always have a value as handled within `hf_serve.backwards`
        handler_file = os.getenv("CUSTOM_HANDLER_FILE")
        if not handler_file.endswith(".py"):  # type: ignore
            raise ValueError(f"The provided `{handler_file=}` is not valid since it's not a Python file.")

        handler_path = model_dir / handler_file  # type: ignore
        if not handler_path.is_file():
            raise FileNotFoundError(f"`{handler_file}` file not found in the directory `{model_dir}`")

        spec = importlib.util.spec_from_file_location(handler_file.removesuffix(".py"), handler_path)  # type: ignore
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load module from `{handler_path}`")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        handler_name = os.getenv("CUSTOM_HANDLER")
        if not hasattr(module, handler_name):  # type: ignore
            raise AttributeError(f"The class `{handler_name}` was not found in `{handler_file}`")

        EndpointHandler = getattr(module, handler_name)  # type: ignore
        # NOTE: AFAIK until the moment we were just ignoring named kwargs meaning that we were supporting "wrong"
        # implementations for the custom handler (not sure if there's a lot of those out there), but in this case
        # the ideal next statement should be something like `return EndpointHandler(model_dir=str(model_dir))`, i.e.
        # including the named kwarg rather than any arbitrary positional argument
        # e.g. https://huggingface.co/philschmid/flan-t5-xxl-sharded-fp16/blob/7edc82dd78b8f084526109f2aafa1126992519e7/handler.py#L6
        return EndpointHandler(str(model_dir))
