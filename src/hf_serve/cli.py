import argparse
import os
import typing

from hf_serve.backwards import check_backwards_compatibility
from hf_serve.server import launch
from hf_serve.types import TaskTypes

# NOTE: required in order for the actual values for the environment variables
# to be set before the defaults are provided to the `argparse` arguments
check_backwards_compatibility()

parser = argparse.ArgumentParser(description="Hugging Face Serve API")
parser.add_argument(
    "--host",
    type=str,
    default=os.getenv("HOST", "0.0.0.0"),
    required=False,
    help="The host into which the FastAPI API will be deployed to, defaults to 0.0.0.0, can also be set via the environment variable `HOST`",
)
parser.add_argument(
    "--port",
    type=int,
    default=os.getenv("PORT", 8080),
    required=False,
    help="The port in which the FastAPI API will listen to, defaults to 8080, can also be set via the environment variable `PORT`",
)
# NOTE: only one of `--model-id` or `--model-dir` should be provided
parser.add_argument(
    "--model-id",
    type=str,
    default=os.getenv("MODEL_ID", None),
    help="The model ID on the Hugging Face Hub, can also be set via the environment variable `MODEL_ID`",
)
parser.add_argument(
    "--model-dir",
    type=str,
    default=os.getenv("MODEL_DIR", None),
    help="A local directory that contains a Hugging Face compatible model, can also be set via the environment variable `MODEL_DIR`",
)
parser.add_argument(
    "--task",
    type=str,
    default=os.getenv("TASK", None),
    choices=typing.get_args(TaskTypes),
    help="Any of the supported tasks for either Transformers, Diffusers, or Sentence Transformers, can also be set via the environment variable `TASK`",
)
parser.add_argument(
    "--device",
    type=str,
    default=os.getenv("DEVICE", "auto"),
    choices=["auto", "balanced", "cuda", "cpu", "mps"],
    required=False,
    help="The device on which the model weights will be loaded into, defaults to auto that selects an accelerator if available, otherwise it falls back to the CPU, can also be set via the environment variable `DEVICE`",
)
parser.add_argument(
    "--dtype",
    type=str,
    default=os.getenv("DTYPE", "float16"),
    choices=["float32", "float16", "bfloat16", "float8", "int8", "int4"],
    required=False,
    help="The PyTorch dtype in which the model weights will be loaded, defaults to `float16`, can also be set via the environment variable `DTYPE`",
)


def main() -> None:
    args = parser.parse_args()

    launch(
        host=args.host,
        port=args.port,
        model_id=args.model_id,
        model_dir=args.model_dir,
        task=args.task,
        device=args.device,
        dtype=args.dtype,
    )
