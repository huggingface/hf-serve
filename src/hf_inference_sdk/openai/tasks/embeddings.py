import base64
import os
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

from hf_inference_sdk.logging import logger
from hf_inference_sdk.openai.schemas.embeddings import EmbeddingOutput, EmbeddingsInput, EmbeddingsOutput, Usage

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


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
            "sentences": payload.input_,
            "convert_to_numpy": True,
            "normalize_embeddings": True,
        }
        if payload.dimensions is not None:
            payload_json["truncate_dim"] = payload.dimensions

        prompt_tokens = 0
        if isinstance(payload.input_, str):
            inputs = self.pipeline.tokenize([payload.input_])
            prompt_tokens = inputs["input_ids"].size(1)
            del inputs
        else:
            # NOTE: `if` already covers `str`, so if falling into `elif` it's a `List[str]` given that otherwise
            # the `pydantic` validation would've failed before
            for input_ in payload.input_:
                inputs = self.pipeline.tokenize([input_])
                prompt_tokens += inputs["input_ids"].size(1)
                del inputs

        usage = Usage(prompt_tokens=prompt_tokens, total_tokens=prompt_tokens)

        outputs = self.pipeline.encode(**payload_json)

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
