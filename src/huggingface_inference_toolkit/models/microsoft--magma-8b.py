from typing import Literal

import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel, Field
from transformers import AutoTokenizer


class ActionCompletionsRequest(BaseModel):
    text: str
    robot_type: Literal["bridge_orig", "google_robot"] = Field(default="bridge_orig")


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
        "max": [
            0.41691166162490845,
            0.25864794850349426,
            0.21218234300613403,
            3.122201919555664,
            1.8618112802505493,
            6.280478477478027,
            1.0,
        ],
        "mean": [
            0.0002334194869035855,
            0.00013004911306779832,
            -0.00012762474943883717,
            -0.0001556558854645118,
            -0.0004039328487124294,
            0.00023557482927571982,
            0.5764579176902771,
        ],
        "min": [
            -0.4007510244846344,
            -0.13874775171279907,
            -0.22553899884223938,
            -3.2010786533355713,
            -1.8618112802505493,
            -6.279075622558594,
            0.0,
        ],
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
        "std": [
            0.009765930473804474,
            0.013689135201275349,
            0.012667362578213215,
            0.028534092009067535,
            0.030637972056865692,
            0.07691419124603271,
            0.4973701536655426,
        ],
    },
    "google_robot": {
        "mask": [True, True, True, True, True, True, False],
        "max": [
            2.9984593391418457,
            22.09052848815918,
            2.7507524490356445,
            1.570636510848999,
            1.5321086645126343,
            1.5691522359848022,
            1.0,
        ],
        "mean": [
            0.006987582892179489,
            0.006265917327255011,
            -0.01262515690177679,
            0.04333311319351196,
            -0.005756212864071131,
            0.0009130256366916001,
            0.5354204773902893,
        ],
        "min": [
            -2.0204520225524902,
            -5.497899532318115,
            -2.031663417816162,
            -1.569917917251587,
            -1.569892168045044,
            -1.570419430732727,
            0.0,
        ],
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
        "std": [
            0.0692116990685463,
            0.05970962345600128,
            0.07353084534406662,
            0.15610496699810028,
            0.13164450228214264,
            0.14593800902366638,
            0.497110515832901,
        ],
    },
}

router = APIRouter()
tokenizer = AutoTokenizer.from_pretrained("microsoft/Magma-8B", trust_remote_code=True)


@router.post("/v1/action-tokens", response_model=ActionCompletionsResponse)
async def action_completions(request: ActionCompletionsRequest) -> ActionCompletionsResponse:
    try:
        norm_stats = NORMALIZATION_STATS[request.robot_type]

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
