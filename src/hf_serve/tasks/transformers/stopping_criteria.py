"""Stop strings handling shared across the `transformers` generation tasks."""

from functools import cache
from typing import Any


@cache
def cpu_stop_string_criteria_cls():
    # NOTE: built lazily so that importing this module does not import `transformers`
    from transformers import StopStringCriteria

    class CPUStopStringCriteria(StopStringCriteria):
        """`StopStringCriteria` evaluated on CPU, returning the per-sample result on the device of `input_ids`.

        Only the trailing `maximum_token_len` tokens are needed to match the stop strings, so that is all that is
        copied to CPU on each decoding step.
        """

        def __call__(self, input_ids, scores, **kwargs):
            tail = input_ids[:, -self.maximum_token_len :].cpu()
            return super().__call__(tail, scores, **kwargs).to(input_ids.device)

    return CPUStopStringCriteria


def stop_strings_generate_kwargs(stop_strings: list[str], tokenizer: Any, device_type: str) -> dict[str, Any]:
    """Builds the `generate` kwargs to stop the generation on any of the given stop strings."""
    if device_type == "neuron":
        # NOTE: `StopStringCriteria` relies on ops that Torch NeuronX cannot lower (`amax` over non-adjacent dims,
        # see aws-neuron/torch-neuronx#1045), so the stop strings are matched on CPU instead
        from transformers import StoppingCriteriaList

        criteria = cpu_stop_string_criteria_cls()(tokenizer=tokenizer, stop_strings=stop_strings)
        return {"stopping_criteria": StoppingCriteriaList([criteria])}
    # NOTE: `generate` needs the tokenizer to match the stop strings against the generated tokens
    return {"stop_strings": stop_strings, "tokenizer": tokenizer}
