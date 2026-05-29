import time
from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Union, get_args, get_origin

from pydantic import BaseModel

from hf_serve.logging import logger

# NOTE: here to handle both the standard type and the `anyOf` syntax if multiple I/O schemas are valid
InputType = TypeVar("InputType", bound=Union[BaseModel, Union[BaseModel, ...]])  # type: ignore
OutputType = TypeVar("OutputType", bound=Union[BaseModel, Union[BaseModel, ...]])  # type: ignore


class Predictor(ABC, Generic[InputType, OutputType]):
    def __init__(self) -> None: ...

    @abstractmethod
    def __call__(self, payload: InputType) -> OutputType: ...

    # TODO: works fine, but the code is not so clean, so we might need to redesign this
    # whilst adding comments all the way through
    def warmup(self) -> None:
        start_time = time.perf_counter()
        logger.info("Warming up model...")

        input_type = None
        for base in getattr(self.__class__, "__orig_bases__", []):
            if get_origin(base) is not None and issubclass(get_origin(base), Predictor):
                args = get_args(base)
                if args:
                    input_type = args[0]
                    break

        if input_type is None:
            logger.warning(
                "Could not determine input type for warmup.\nNote that the first inference request/s may take longer as models need to be initialized."
            )
            return

        examples = []
        types_to_check = [input_type]
        if get_origin(input_type) is Union:
            types_to_check = [arg for arg in get_args(input_type) if arg is not type(None)]

        for model_type in types_to_check:
            if isinstance(model_type, type) and issubclass(model_type, BaseModel):
                if hasattr(model_type, "model_config") and isinstance(model_type.model_config, dict):
                    json_schema_extra = model_type.model_config.get("json_schema_extra", {})
                    if isinstance(json_schema_extra, dict):
                        for example_data in json_schema_extra.get("examples", []):  # type: ignore
                            try:
                                examples.append(model_type(**example_data))
                            except Exception as e:
                                logger.warning(f"Failed to create example from schema data: {e}")

        if not examples:
            logger.warning(
                "Could not find any warmup examples within the input schema/s.\nNote that the first inference request/s may take longer as models need to be initialized."
            )
            return

        warmup_successful = False
        for i, example in enumerate(examples):
            try:
                logger.info(f"Running warmup with example {i + 1}/{len(examples)}: {example}")
                self(example)
                warmup_successful = True
                break
            except Exception as e:
                logger.warning(f"Warmup {i + 1}/{len(examples)} failed: {e}")
                if i == len(examples) - 1:
                    logger.error(
                        "All warmup attempts failed. This indicates issues with the model or input validation."
                    )
                    raise RuntimeError(
                        "Warmup failed: all example attempts failed. Cannot initialize API."
                    ) from e

        if warmup_successful:
            elapsed_time = time.perf_counter() - start_time
            logger.info(f"Model warmup completed successfully in {elapsed_time:.2f} seconds")
