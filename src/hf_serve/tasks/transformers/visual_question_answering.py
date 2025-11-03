from typing import Annotated, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, RootModel, conint

from hf_serve.serde import Image
from hf_serve.tasks.predictor import Predictor


class VisualQuestionAnsweringInputs(BaseModel):
    image: Union[str, bytes]
    question: str


class VisualQuestionAnsweringParameters(BaseModel):
    top_k: Optional[Annotated[int, conint(ge=0)]] = Field(default=1)


class VisualQuestionAnsweringInput(BaseModel):
    inputs: VisualQuestionAnsweringInputs
    parameters: Optional[VisualQuestionAnsweringParameters] = Field(default=None)

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "inputs": {
                        "image": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/rabbit.png",
                        "question": "What is in the image?",
                    },
                    "parameters": {
                        "top_k": 3,
                    },
                }
            ]
        },
    )


class VisualQuestionAnsweringOutputValue(BaseModel):
    # NOTE: Transformers documentation says this should be `label` but apparently it's `answer` instead
    answer: str
    score: Optional[float] = Field(default=None)


class VisualQuestionAnsweringOutput(RootModel):
    root: List[VisualQuestionAnsweringOutputValue]


class VisualQuestionAnswering(Predictor[VisualQuestionAnsweringInput, VisualQuestionAnsweringOutput]):
    def __init__(self, model_id: str, dtype: Optional[str] = None, device: str = "auto") -> None:
        super().__init__()

        import torch
        from transformers import pipeline
        from transformers.pipelines.visual_question_answering import VisualQuestionAnsweringPipeline

        # NOTE: Apparently some (not all) models don't support the `device_map=auto` so we should probably
        # either add a check or just default to CUDA instead
        if device == "auto":
            # e.g. DistilBertForSequenceClassification won't support it
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"

        self.pipeline: VisualQuestionAnsweringPipeline = pipeline(
            task="visual-question-answering",
            model=model_id,
            dtype=getattr(torch, dtype) if dtype is not None else "auto",
            device=device if device not in {"auto"} else None,
            device_map=device if device in {"auto"} else None,
        )

        if (
            hasattr(self.pipeline.image_processor, "is_vqa")
            and getattr(self.pipeline.image_processor, "is_vqa") is True
        ):
            raise RuntimeError(
                f"{model_id=} is unsupported with the `visual-question-answering` pipeline. Feel free to open an issue describing the error on either https://github.com/huggingface/transformers/issues/new or rather in https://github.com/huggingface/hf-serve/issues/new instead."
            )

        if torch.mps.is_available():
            torch.mps.empty_cache()
            torch.mps.set_per_process_memory_fraction(0.9)

    def __call__(self, payload: VisualQuestionAnsweringInput) -> VisualQuestionAnsweringOutput:
        parameters = {}
        if payload.parameters:
            parameters = payload.parameters.model_dump(exclude_none=True)

        # NOTE: For models like e.g. `google/deplot` the default pipeline for `visual-question-answering` won't
        # work, then the following patch is required on top of `transformers`:
        #
        # ```diff
        # diff --git a/transformers/pipelines/visual_question_answering.py b/transformers/pipelines/visual_question_answering_new.py
        # index 609eaf2..0b9598d 100644
        # --- a/transformers/pipelines/visual_question_answering.py
        # +++ b/transformers/pipelines/visual_question_answering_new.py
        # @@ -178,7 +178,10 @@ class VisualQuestionAnsweringPipeline(Pipeline):
        #              padding=padding,
        #              truncation=truncation,
        #          )
        # -        image_features = self.image_processor(images=image, return_tensors=self.framework)
        # +        if self.image_processor.is_vqa:
        # +            image_features = self.image_processor(images=image, header_text=inputs["question"], return_tensors=self.framework)
        # +        else:
        # +            image_features = self.image_processor(images=image, return_tensors=self.framework)
        #          if self.framework == "pt":
        #          image_features = image_features.to(self.dtype)
        #          model_inputs.update(image_features)
        # ```
        # NOTE: Besides that, the following snippet should be defined here to capture whether the image processor
        # has the `is_vqa` flag set, in which case the expected input is `text_header`. But note that given
        # how the inputs are defined for `visual-question-answering` then the `question` also need to be provided
        # otherwise the validation will lead to the inputs on the `preprocess` method to break
        #
        # ```python
        # if (
        #     hasattr(self.pipeline.image_processor, "is_vqa")
        #     and getattr(self.pipeline.image_processor, "is_vqa") is True
        # ):
        #     output = self.pipeline(
        #         image=Image.deserialize(payload.inputs.image),
        #         question=payload.inputs.question,
        #         header_text=payload.inputs.question,
        #         **parameters,
        #     )
        # else:
        #     output = self.pipeline(
        #         image=Image.deserialize(payload.inputs.image), question=payload.inputs.question, **parameters
        #     )
        # ```
        output = self.pipeline(
            image=Image.deserialize(payload.inputs.image), question=payload.inputs.question, **parameters
        )
        return VisualQuestionAnsweringOutput(root=output)  # type: ignore
