import importlib.util
from pathlib import Path
from typing import Optional, Tuple, Type

from pydantic import BaseModel

from hf_serve.tasks.predictor import Predictor
from hf_serve.types import TaskTypes


def load_custom_predictor(
    task: TaskTypes,
    model_id: Optional[str] = None,
    model_dir: Optional[str] = None,
) -> Tuple[Type[Predictor], Type[BaseModel], Type[BaseModel]]:
    model_path = None
    match task:
        # TODO: Given the complexity of defining which models contain a custom implementation and which ones don't
        # we are creating this on a per-custom-model implementation basis at the moment, is obviously not the best
        # and won't scale, but it's a decent starting point given that we don't have much custom models that we'd
        # like to natively support at the moment
        case "text-ranking" | "sentence-ranking":
            # NOTE: Rename to `text-ranking` given that the `task` is used to build the path, and the path is
            # formatted as e.g. `src/hf_serve/custom_models/text_ranking/qwen3/predictor.py`
            task = "text-ranking"

            if model_id is not None:
                if model_id in {
                    "Qwen/Qwen3-Reranker-0.6B",
                    "Qwen/Qwen3-Reranker-4B",
                    "Qwen/Qwen3-Reranker-8B",
                }:
                    model_path = "qwen3"
                else:
                    raise ValueError(
                        f"Model `{model_id}` doesn't have a custom implementation available in `hf-serve`."
                    )

            if model_dir is not None:
                config_file = Path(model_dir) / "config.json"
                if config_file.exists():
                    from transformers import AutoConfig

                    config = AutoConfig.from_pretrained(config_file)
                    if (
                        config.architectures is not None
                        and isinstance(config.architectures, list)
                        and any(arch.__contains__("Qwen3ForCausalLM") for arch in config.architectures)
                        and config.model_type == "qwen3"
                    ):
                        model_path = "qwen3"
                else:
                    raise ValueError(
                        "Model in dir `{model_dir}` doesn't contain a `config.json` file used to identify whether the provided model contains a custom implementation in `hf-serve`."
                    )

        case _:
            raise ValueError(
                "Custom modelling is only implemented for `text-ranking` (or `sentence-ranking`) tasks at the moment, if you are willing to add support for a new model and/or task with custom modelling code, feel free to open an issue or create a PR at https://github.com/huggingface/hf-serve."
            )

    # NOTE: We replace `-` with `_` given that the `task` arg in the CLI is formatted as e.g. `text-classification`,
    # but the actual paths in Python are formatted with underscore instead as e.g. `text_classification`
    task_path = task.replace("-", "_")

    predictor_file = Path(__file__).parent / task_path / model_path / "predictor.py"  # type: ignore
    if predictor_file is None:
        raise ValueError(
            "No custom implementation found for `{model_id or model_dir}` in `hf-serve`, hence using the default `hf-serve` implementation for `{task}`."
        )

    if not predictor_file.exists():
        raise ValueError(
            f"Model `{model_id}` mapped to `{predictor_file}` but predictor.py file not found at {predictor_file}"
        )

    try:
        spec = importlib.util.spec_from_file_location(
            f"custom_models.{task}.{model_path}.predictor", predictor_file
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(
                f"Failed to create module spec for `{model_id or model_dir}` custom predictor at {predictor_file}. The file may be corrupted or have syntax errors."
            )

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        custom_predictor = getattr(module, "CustomPredictor", None)
        input_cls = getattr(module, "Input", None)
        output_cls = getattr(module, "Output", None)

        if custom_predictor and input_cls and output_cls:
            return custom_predictor, input_cls, output_cls

        raise RuntimeError(
            f"Model `{model_id or model_dir}` predictor module is missing required classes: {', '.join([cls for cls in zip([custom_predictor, input_cls, output_cls], ['CustomPredictor', 'Input', 'Output']) if cls is None])}. "
            f"File: {predictor_file}"
        )
    except (ImportError, AttributeError, SyntaxError) as e:
        raise RuntimeError(
            f"Failed to import custom predictor for model `{model_id or model_dir}` from {predictor_file}: {str(e)}"
        ) from e
    except Exception as e:
        raise RuntimeError(
            f"Unexpected error loading custom predictor for model `{model_id or model_dir}` from {predictor_file}: {str(e)}"
        ) from e
