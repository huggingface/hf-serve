import argparse
import os
from typing import get_args

from hf_serve.compatibility.backwards import ensure_backwards_compatibility
from hf_serve.server import launch
from hf_serve.types import TaskTypes

# NOTE: required in order for the actual values for the environment variables
# to be set before the defaults are provided to the `argparse` arguments
ensure_backwards_compatibility()

parser = argparse.ArgumentParser(description="Hugging Face Serve API")

parser.add_argument(
    "--host",
    type=str,
    default=os.getenv("HOST", None) or "0.0.0.0",
    required=False,
    help="The host into which the FastAPI API will be deployed to, defaults to 0.0.0.0. It can also be set via the environment variable `HOST`.",
)

parser.add_argument(
    "--port",
    type=int,
    default=os.getenv("PORT", None) or 8080,
    required=False,
    help="The port in which the FastAPI API will listen to, defaults to 8080. It can also be set via the environment variable `PORT`.",
)

# NOTE: only one of `--model-id` or `--model-dir` should be provided
parser.add_argument(
    "--model-id",
    type=str,
    default=os.getenv("MODEL_ID", None),
    help="The model ID on the Hugging Face Hub. It can also be set via the environment variable `MODEL_ID`.",
)

parser.add_argument(
    "--model-dir",
    type=str,
    default=os.getenv("MODEL_DIR", None),
    help="A local directory that contains a Hugging Face compatible model. It can also be set via the environment variable `MODEL_DIR`.",
)

parser.add_argument(
    "--task",
    type=str,
    default=os.getenv("TASK", None),
    choices=get_args(TaskTypes),
    help="Any of the supported tasks for either Transformers, Diffusers, or Sentence Transformers. It can also be set via the environment variable `TASK`.",
)

parser.add_argument(
    "--device",
    type=str,
    default=os.getenv("DEVICE", None) or "auto",
    choices=["auto", "balanced", "cuda", "cpu", "mps"],
    required=False,
    help="The device on which the model weights will be loaded into, defaults to auto that selects an accelerator if available, otherwise it falls back to the CPU. It can also be set via the environment variable `DEVICE`.",
)

parser.add_argument(
    "--dtype",
    type=str,
    # NOTE: This might seem weird, but if `DTYPE=""` then it won't be None,
    # hence its value will be `""` but since we don't want that, we need to add
    # the check `or None` to make sure that if `""` is set, then we default to
    # `None`
    default=os.getenv("DTYPE", None) or None,
    choices=["float32", "float16", "bfloat16", "float8", "int8", "int4"],
    required=False,
    help="The PyTorch dtype in which the model weights will be loaded, defaults to None meaning that the default dtype for the given model will be used i.e., the dtype in which the model weights are available. It can also be set via the environment variable `DTYPE`.",
)

# TODO(juanjucm): validate accepted_mimetypes values based on the task.
# Check processor's accepted file formats (e.g. ffmpeg for audio (https://www.ffmpeg.org/general.html#File-Formats))
parser.add_argument(
    "--accepted-mimetypes",
    type=str,
    default=os.getenv("ACCEPTED_MIMETYPES", None),
    required=False,
    help="A comma-separated list of accepted MIME types for file uploads. By default, each task will have all valid MIME types (e.g. audio/* for audio tasks, image/* for image tasks). It can also be set via the environment variable `ACCEPTED_MIMETYPES`.",
)

parser.add_argument(
    "--max-file-size",
    type=int,
    # TODO: Maybe remove the `or None` and instead of checking that `if max_file_size is not None`
    # we could just `if not max_file_size` (which handles `""`)
    default=os.getenv("MAX_FILE_SIZE", None) or None,
    required=False,
    help="The maximum file size in bytes for file uploads (e.g 10485760 for 10MB). By default, no file size limit is considered. It can also be set via the environment variable `MAX_FILE_SIZE`.",
)

parser.add_argument(
    "--cloud",
    type=str,
    default=os.getenv("CLOUD", None) or None,
    choices=["azure", "google"],
    required=False,
    help="To be defined when deploying on a cloud provider to ensure that it's compatible with the provider expectations e.g. `/score` route needs to be exposed for Azure AI Foundry and Azure ML deployments (among others); or e.g. `instances` needs to be a list of inputs for Vertex AI (among others).",
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
        # TODO(juanjucm): This can most likely be a list, and it will automatically be formatted this way
        # without having to handle that here
        accepted_mimetypes=args.accepted_mimetypes.split(",") if args.accepted_mimetypes else None,
        max_file_size=args.max_file_size,
        cloud=args.cloud,
    )
