"""`CustomPredictor` for `Qwen/Qwen3-Reranker-*` models, given that despite those
are `text-ranking` models, as those as LLM-based rerankers, there's no native support
for those within `sentence-transformers` yet, but rather only via `transformers` with
a custom inference loop that captures the `yes` and `no` token logits to determine
the ranks.

Notes:
    - All the models with `text-ranking` as the task and `Qwen3ForCausalLM` as the
    `architecture` in `config.json` should fall into this implementation
    - Only `Qwen/Qwen3-Reranker-0.6B`, `Qwen/Qwen3-Reranker-4B`, `Qwen/Qwen3-Reranker-8B`
    and any fine-tune or re-upload from those is supported at the moment

References:
    - https://huggingface.co/Qwen/Qwen3-Reranker-0.6B#transformers-usage
    - https://huggingface.co/Qwen/Qwen3-Reranker-4B#transformers-usage
    - https://huggingface.co/Qwen/Qwen3-Reranker-8B#transformers-usage
"""

from typing import Literal, Optional

import torch  # NOTE: `torch` import cannot be lazy since it's used on both `__init__` and `__call__`

from hf_serve.logging import logger
from hf_serve.tasks.predictor import Predictor
from hf_serve.tasks.sentence_transformers.text_ranking import RankInput, RankOutput

Input = RankInput
Output = RankOutput


class CustomPredictor(Predictor[Input, Output]):
    def __init__(
        self,
        model_id: str,
        dtype: Optional[Literal["float32", "float16", "bfloat16"]] = None,
        device: Optional[Literal["auto", "cpu", "cuda", "mps"]] = None,
        backend: Literal["torch", "onnx", "openvino"] = "torch",
        attn_implementation: Optional[Literal["eager", "sdpa", "flash_attention_2"]] = None,
    ) -> None:
        super().__init__()

        # TODO: Given that we know that the minimum supported Transformers version is 4.51.0, it might make sense
        # to include a function to check that given packages are installed and in the correct version e.g. check
        # that whatever `transformers` version is installed is compliant with the minimum required version, otherwise
        # eventually leverage the `DynamicInstaller` if applicable

        if backend is not None and backend != "torch":
            raise RuntimeError(
                f"Given that {model_id=} runs with custom code, the custom Transformers-based implementation won't support neither `onnx` nor `openvino` as contrary as other `text-ranking` models that are exposed via `sentence_transformers` instead."
            )

        match device:
            case "cuda":
                if attn_implementation is not None and attn_implementation != "flash_attention_2":
                    logger.warning(
                        f"{model_id=} is recommended to run with `flash_attention_2`, which you can install as `uv pip install flash-attn --no-build-isolation` or rather from `hf-serve` as `uv sync --extra cuda --extra flash-attn --preview-features extra-build-dependencies`"
                    )
                init_kwargs = {
                    "pretrained_model_name_or_path": model_id,
                    # NOTE: We enforce the default `dtype` to FP16 as specified within
                    # the `Qwen/Qwen3-Reranker-*` model cards.
                    "dtype": getattr(torch, dtype) if dtype is not None else torch.float16,
                    "attn_implementation": attn_implementation or "flash_attention_2",
                    "device": device,
                }
            case "mps":
                if attn_implementation is not None and attn_implementation != "sdpa":
                    logger.warning(
                        f"{attn_implementation=} support on MPS is known to have some flaws, hence `eager` is recommended instead."
                    )
                init_kwargs = {
                    "pretrained_model_name_or_path": model_id,
                    # NOTE: We enforce the default `dtype` to FP32 as specified within
                    # the `Qwen/Qwen3-Reranker-*` model cards.
                    "dtype": getattr(torch, dtype) if dtype is not None else torch.float32,
                    # NOTE: use `eager` as default given that `sdpa` in MPS is known to have flaws
                    "attn_implementation": attn_implementation or "eager",
                    "device_map": device,
                }
            case "cpu":
                init_kwargs = {
                    "pretrained_model_name_or_path": model_id,
                    # NOTE: We enforce the default `dtype` to FP32 as specified within
                    # the `Qwen/Qwen3-Reranker-*` model cards.
                    "dtype": getattr(torch, dtype) if dtype is not None else torch.float32,
                    "attn_implementation": attn_implementation or "sdpa",
                    "device_map": device,
                }
            case _:
                init_kwargs = {
                    "pretrained_model_name_or_path": model_id,
                    "device_map": "auto",
                }
                if dtype:
                    init_kwargs["dtype"] = getattr(torch, dtype)
                if attn_implementation:
                    init_kwargs["attn_implementation"] = attn_implementation

        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(**init_kwargs)

        self.no_token_id = self.tokenizer.convert_tokens_to_ids("no")
        self.yes_token_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.max_length = 8192

    def __call__(self, payload: Input) -> Output:
        pairs = [
            f"<Instruct>: Given a web search query, retrieve relevant passages that answer the query\n<Query>: {payload.query}\n<Document>: {text}"
            for text in payload.texts
        ]

        # NOTE: Both the `prefix` and the `suffix` are tokenized separately to make sure that if we need to
        # truncate the input, then `suffix` tokens are still there, otherwise the generation would be gibberish
        prefix_tokens = self.tokenizer.encode(
            '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n',
            add_special_tokens=False,
        )
        suffix_tokens = self.tokenizer.encode(
            "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", add_special_tokens=False
        )

        if payload.truncate in {True, False} or payload.truncation_direction in {"right", "left"}:
            logger.warning(
                f"Neither `{payload.truncate=}` nor `{payload.truncation_direction=}` values will be used, given that `Qwen/Qwen3-Reranker-*` models use a pre-defined truncation strategy to prevent from trimming the chat template and/or the special tokens which might hurt the performance."
            )

        # NOTE: For every `pair` in `pairs`, we need to make sure that the `prefix_tokens` are prepended and that
        # the `suffix_tokens` are appended to each of the pairs
        inputs = self.tokenizer(
            pairs,
            padding=False,
            truncation="longest_first",
            max_length=self.max_length - len(prefix_tokens) - len(suffix_tokens),
            return_attention_mask=False,
        )
        for idx, input_ids in enumerate(inputs["input_ids"]):
            inputs["input_ids"][idx] = prefix_tokens + input_ids + suffix_tokens

        inputs = self.tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=self.max_length)

        # NOTE: Moving to the `device` all the inputs as it's a `BatchEncoding` and calling `to(dtype)` over it
        # won't work as it needs to be done per each element in the batch instead
        for key in inputs:
            inputs[key] = inputs[key].to(self.model.device)

        batch_scores = self.model(**inputs).logits[:, -1, :]

        yes_vector = batch_scores[:, self.yes_token_id]
        no_vector = batch_scores[:, self.no_token_id]

        batch_scores = torch.stack([no_vector, yes_vector], dim=1)
        if payload.raw_scores is False:
            batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)

        return Output(
            scores=[
                {
                    "index": index,
                    "score": score,  # type: ignore
                    "text": payload.texts[index] if payload.return_documents else None,
                }
                for index, score in enumerate(batch_scores[:, 1].exp().tolist())
            ]
        )
