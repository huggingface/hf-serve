import base64
import os
from contextvars import ContextVar
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

from hf_serve.logging import logger
from hf_serve.openai.schemas.embeddings import EmbeddingOutput, EmbeddingsInput, EmbeddingsOutput, Usage

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


_usage_request: ContextVar[Optional[object]] = ContextVar("embedding_usage_request", default=None)


class Embeddings:
    def __init__(self, pipeline: "SentenceTransformer") -> None:
        super().__init__()

        self.pipeline = pipeline

    @property
    @lru_cache(maxsize=1)
    def model_id(self) -> Union[str, None]:
        if self.pipeline.model_card_data is not None:
            model_id = self.pipeline.model_card_data.model_id
            if model_id is not None and not Path(model_id).exists():
                return model_id
            base_model = self.pipeline.model_card_data.base_model
            if base_model is not None and not Path(base_model).exists():
                return base_model
        return os.getenv("MODEL_ID", os.getenv("MODEL_DIR"))

    def __call__(self, payload: EmbeddingsInput, request_id: Optional[str] = None) -> EmbeddingsOutput:
        # NOTE: The `user` parameter is only supported within OpenAI, so reporting those to let the user know
        # that those will be ignored as those can't and won't be used
        # NOTE: It's placed here instead of as a model-validator under `EmbeddingsInputs` as otherwise we
        # cannot propagate the `request_id` meaning the logging message is meaningless
        if payload.user is not None:
            message = f"[{request_id}] `user={payload.user}` was provided, but it's not handled as it's OpenAI-specific, so it will be ignored."
            logger.warning(message)

        payload_json: Dict[str, Any] = {
            "convert_to_numpy": True,
            "normalize_embeddings": True,
        }
        if payload.dimensions is not None:
            payload_json["truncate_dim"] = payload.dimensions
        if payload.prompt_name is not None:
            payload_json["prompt_name"] = payload.prompt_name

        prompt_tokens = 0
        request_marker = object()

        def count_tokens(_module: Any, args: Any) -> None:
            nonlocal prompt_tokens
            # Hooks belong to the shared pipeline; only count this request's forwards.
            if _usage_request.get() is not request_marker:
                return
            features = args[0]
            if (mask := features.get("attention_mask")) is not None:
                prompt_tokens += int(mask.sum().item())
            elif (input_ids := features.get("input_ids")) is not None:
                prompt_tokens += input_ids.numel()

        # Observe the exact batch used by encode(), including prompts and media tokens,
        # without downloading/decoding media a second time. Ignore batch padding.
        # SentenceTransformer.encode passes features positionally. Avoid with_kwargs:
        # PyTorch tracks that flag separately, so removing a hook during another
        # request's forward can otherwise change its calling convention mid-iteration.
        handle = self.pipeline.register_forward_pre_hook(count_tokens)
        context_token = _usage_request.set(request_marker)
        try:
            outputs = self.pipeline.encode(payload.input_, **payload_json)
        finally:
            handle.remove()
            _usage_request.reset(context_token)

        usage = Usage(prompt_tokens=prompt_tokens, total_tokens=prompt_tokens)

        # NOTE: Add the batch dimension if `ndim` is 1 i.e., a 1D-vector so that we can unify the output
        if outputs.ndim == 1:
            outputs = outputs.reshape(1, -1)

        return EmbeddingsOutput(
            data=[
                EmbeddingOutput(
                    embedding=output.tolist()
                    if payload.encoding_format == "float"
                    else base64.b64encode(output.tobytes()).decode(),
                    index=index,
                )
                for (index, output) in enumerate(outputs)
            ],
            model=payload.model,
            usage=usage,
        )
