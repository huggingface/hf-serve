from typing import Annotated, List, Optional

from annotated_types import Len
from pydantic import BaseModel, ConfigDict, Field

from hf_inference_sdk.tasks.transformers.zero_shot_classification import (
    ZeroShotClassificationOutput,
    ZeroShotClassificationParameters,
)


class ZeroShotClassificationInputForGoogle(BaseModel):
    instances: Annotated[List[str], Len(min_length=1)]
    parameters: Optional[ZeroShotClassificationParameters] = Field(default=None)

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "instances": ["I have a problem with my iphone that needs to be resolved ASAP!"],
                    "parameters": {
                        "candidate_labels": ["urgent", "not urgent", "phone", "tablet", "computer"],
                    },
                }
            ]
        }
    )


class ZeroShotClassificationOutputForGoogle(BaseModel):
    predictions: List[ZeroShotClassificationOutput]
