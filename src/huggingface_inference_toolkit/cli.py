import argparse
import os

from huggingface_inference_toolkit.backwards import check_backwards_compatibility
from huggingface_inference_toolkit.server import launch
from huggingface_inference_toolkit.utils import get_available_tasks

# NOTE: required in order for the actual values for the environment variables
# to be set before the defaults are provided to the `argparse` arguments
check_backwards_compatibility()

parser = argparse.ArgumentParser(description="Hugging Face Inference Toolkit")
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
    choices=get_available_tasks(),
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
    from huggingface_inference_toolkit.logging import logger

    args = parser.parse_args()

    # NOTE: tried `group = parser.add_mutually_exclusive_group(required=True)`, but it's not working fine because
    # it won't capture the values from the environment variables values
    if args.model_id and args.model_dir:
        logger.warning(
            f"Both {args.model_id=} and {args.model_dir=} have been provided but those are mutually exclusive, if both are provided then `--model-dir` has preference over `--model-id`"
        )

        args.model_id = None

    if not args.model_id and not args.model_dir:
        raise ValueError(
            "Any of `--model-id` or `--model-dir` should be provided but both cannot be None (alternatively those can be provided via the environment variables `MODEL_ID` or `MODEL_DIR`, respectively."
        )

    launch(
        host=args.host,
        port=args.port,
        model_id=args.model_id,
        model_dir=args.model_dir,
        task=args.task,
        device=args.device,
        dtype=args.dtype,
    )
