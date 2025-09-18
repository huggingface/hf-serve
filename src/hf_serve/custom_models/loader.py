import importlib.util
from pathlib import Path
from typing import Tuple, Type

from pydantic import BaseModel

from hf_serve.tasks.predictor import Predictor

MAPPING = {
    "Qwen/Qwen3-Reranker-0.6B": "qwen3_reranker",
    "Qwen/Qwen3-Reranker-4B": "qwen3_reranker",
    "Qwen/Qwen3-Reranker-8B": "qwen3_reranker",
}


def load_custom_predictor(
    model_id: str,
) -> Tuple[Type[Predictor], Type[BaseModel], Type[BaseModel]]:
    if model_id not in MAPPING:
        raise ValueError(f"Model `{model_id}` doesn't have a custom implementation available in `hf-serve`.")

    model_path = MAPPING[model_id]
    predictor_dir = Path(__file__).parent / model_path
    predictor_file = predictor_dir / "predictor.py"

    if not predictor_file.exists():
        raise ValueError(
            f"Model `{model_id}` mapped to `{model_path}` but predictor.py file not found at {predictor_file}"
        )

    try:
        spec = importlib.util.spec_from_file_location(f"custom_models.{model_path}.predictor", predictor_file)
        if spec is None or spec.loader is None:
            raise RuntimeError(
                f"Failed to create module spec for `{model_id}` custom predictor at {predictor_file}. The file may be corrupted or have syntax errors."
            )

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        custom_predictor = getattr(module, "CustomPredictor", None)
        input_cls = getattr(module, "Input", None)
        output_cls = getattr(module, "Output", None)

        if custom_predictor and input_cls and output_cls:
            return custom_predictor, input_cls, output_cls

        raise RuntimeError(
            f"Model `{model_id}` predictor module is missing required classes: {', '.join([cls for cls in zip([custom_predictor, input_cls, output_cls], ['CustomPredictor', 'Input', 'Output']) if cls is None])}. "
            f"File: {predictor_file}"
        )
    except (ImportError, AttributeError, SyntaxError) as e:
        raise RuntimeError(
            f"Failed to import custom predictor for model `{model_id}` from {predictor_file}: {str(e)}"
        ) from e
    except Exception as e:
        raise RuntimeError(
            f"Unexpected error loading custom predictor for model `{model_id}` from {predictor_file}: {str(e)}"
        ) from e
