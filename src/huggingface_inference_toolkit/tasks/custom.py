import importlib.util
import os
from pathlib import Path
from typing import Any


class Custom:
    @staticmethod
    def load(model_dir: str = "/repository") -> Any:
        model_dir = Path(model_dir)  # type: ignore

        # NOTE: `CUSTOM_HANDLER_FILE` will always have a value as handled within `huggingface_inference_toolkit.backwards`
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
        return EndpointHandler(model_dir=str(model_dir))
