import argparse
import os

from huggingface_inference_toolkit.backwards import check_backwards_compatibility
from huggingface_inference_toolkit.server import launch

# NOTE: required in order for the actual values for the environment variables
# to be set before the defaults are provided to the `argparse` arguments
check_backwards_compatibility()

parser = argparse.ArgumentParser(description="Hugging Face Inference Toolkit")
parser.add_argument("--host", type=str, default=os.getenv("HOST", "0.0.0.0"), required=False)
parser.add_argument("--port", type=int, default=os.getenv("PORT", 8080), required=False)
parser.add_argument("--model-id", type=str, default=os.getenv("MODEL_ID", None))
parser.add_argument(
    "--task",
    type=str,
    default=os.getenv("TASK", None),
    choices=[
        # diffusers
        "text-to-image",
        # sentence-transformers
        "sentence-similarity",
        "sentence-embeddings",
        "sentence-ranking",
        # transformers
        "text-classification",
    ],
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
