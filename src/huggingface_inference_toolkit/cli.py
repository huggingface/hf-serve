import argparse
import os
from pathlib import Path

from huggingface_inference_toolkit.backwards import check_backwards_compatibility
from huggingface_inference_toolkit.server import launch
from huggingface_inference_toolkit.utils import get_available_tasks

# NOTE: required in order for the actual values for the environment variables
# to be set before the defaults are provided to the `argparse` arguments
check_backwards_compatibility()

def get_available_tasks():
    """
    Small hack to retrieve available tasks by scanning the tasks directory.
    This function assumes that each task is defined in a .py file under
    the 'tasks/[library]' directory (ignoring '__init__.py').
    """
    tasks_dir = Path(__file__).parent / "tasks"
    
    available_tasks = []
    if tasks_dir.exists() and tasks_dir.is_dir():
        for library_dir in tasks_dir.iterdir():
            if library_dir.is_dir():
                # List all .py files (excluding __init__.py) in the library directory.
                # Replace "_" with "-" in the final strings.
                available_tasks.extend(
                    [p.stem.replace("_", "-") for p in library_dir.glob("*.py") if p.name != "__init__.py"]
                )
    return available_tasks


parser = argparse.ArgumentParser(description="Hugging Face Inference Toolkit")
parser.add_argument("--host", type=str, default=os.getenv("HOST", "0.0.0.0"), required=False)
parser.add_argument("--port", type=int, default=os.getenv("PORT", 8080), required=False)
parser.add_argument("--model-id", type=str, default=os.getenv("MODEL_ID", None))
parser.add_argument(
    "--task",
    type=str,
    default=os.getenv("TASK", None),
    choices=get_available_tasks(),
)
parser.add_argument(
    "--device",
    type=str,
    default=os.getenv("DEVICE", "auto"),
    choices=["auto", "balanced", "cuda", "cpu", "mps"],
    required=False,
)
parser.add_argument(
    "--dtype",
    type=str,
    default=os.getenv("DTYPE", "float16"),
    choices=["float32", "float16", "bfloat16", "float8", "int8", "int4"],
    required=False,
)


def main() -> None:
    args = parser.parse_args()

    launch(
        host=args.host,
        port=args.port,
        model_id=args.model_id,
        task=args.task,
        device=args.device,
        dtype=args.dtype,
    )
