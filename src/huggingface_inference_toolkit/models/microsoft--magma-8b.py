from typing import List, Literal, Optional

import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel, Field, model_validator
from transformers import AutoTokenizer
from typing_extensions import Self


class NormalizationStats(BaseModel):
    mask: List[bool]
    q01: List[float]
    q99: List[float]


class ActionCompletionsRequest(BaseModel):
    text: str
    robot_type: Optional[Literal["bridge_orig", "google_robot"]] = Field(default=None)
    normalization_stats: Optional[NormalizationStats] = Field(default=None)

    @model_validator(mode="after")
    def check_mutually_exclusive(cls, values) -> Self:
        robot_type = values.robot_type
        normalization_stats = values.normalization_stats
        if robot_type is not None and normalization_stats is not None:
            raise ValueError("`robot_type` and `normalization_stats` are mutually exclusive")
        if robot_type is None and normalization_stats is None:
            values.robot_type = "bridge_orig"
        return values


class ActionCompletionsResponse(BaseModel):
    x: float
    y: float
    z: float
    roll: float
    pitch: float
    yaw: float
    gripper: float


# Known robot normalization stats
# Reference: https://github.com/microsoft/Magma/blob/064762dca5e4ca089d84af016d9297ffc3c8f456/tools/simplerenv-magma/simpler_env/policies/magma/magma_model.py#L16
NORMALIZATION_STATS = {
    "bridge_orig": {
        "mask": [True, True, True, True, True, True, False],
        "q01": [
            -0.02872725307941437,
            -0.04170349963009357,
            -0.026093858778476715,
            -0.08092105075716972,
            -0.09288699507713317,
            -0.20718276381492615,
            0.0,
        ],
        "q99": [
            0.028309678435325586,
            0.040855254605412394,
            0.040161586627364146,
            0.08192047759890528,
            0.07792850524187081,
            0.20382574498653397,
            1.0,
        ],
    },
    "google_robot": {
        "mask": [True, True, True, True, True, True, False],
        "q01": [
            -0.22453527510166169,
            -0.14820013284683228,
            -0.231589707583189,
            -0.3517994859814644,
            -0.4193011274933815,
            -0.43643461108207704,
            0.0,
        ],
        "q99": [
            0.17824687153100965,
            0.14938379630446405,
            0.21842354819178575,
            0.5892666035890578,
            0.35272657424211445,
            0.44796681255102094,
            1.0,
        ],
    },
}

router = APIRouter()
tokenizer = AutoTokenizer.from_pretrained("microsoft/Magma-8B", trust_remote_code=True)


@router.post("/v1/action-tokens", response_model=ActionCompletionsResponse)
async def action_completions(request: ActionCompletionsRequest) -> ActionCompletionsResponse:
    try:
        if request.normalization_stats is not None:
            norm_stats = dict(request.normalization_stats)
        else:
            norm_stats = NORMALIZATION_STATS[request.robot_type]  # type: ignore

        # Extract action tokens from the text
        input_ids = tokenizer.encode(request.text)

        # Extract the last 7 tokens (6 DOF + gripper)
        action_ids = input_ids[-7:]

        # Convert to discretized actions
        n_action_bins = 256
        bins = np.linspace(-1, 1, n_action_bins)
        bin_centers = (bins[:-1] + bins[1:]) / 2.0
        discretized_actions = tokenizer.vocab_size - np.array(action_ids).astype(np.int64)
        discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=bin_centers.shape[0] - 1)

        # Get normalized actions (-1 to 1 range)
        normalized_actions = bin_centers[discretized_actions]

        # Unnormalize actions using the selected stats
        mask = norm_stats.get("mask", np.ones_like(normalized_actions, dtype=bool))
        action_high, action_low = (
            np.array(norm_stats["q99"]),
            np.array(norm_stats["q01"]),
        )

        raw_actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )

        return ActionCompletionsResponse(
            x=float(raw_actions[0]),
            y=float(raw_actions[1]),
            z=float(raw_actions[2]),
            roll=float(raw_actions[3]),
            pitch=float(raw_actions[4]),
            yaw=float(raw_actions[5]),
            gripper=float(raw_actions[6]),
        )
    except Exception as e:
        raise ValueError(f"Failed to extract actions: {str(e)}")
