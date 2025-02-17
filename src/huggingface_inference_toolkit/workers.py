import os

import torch


def num_workers() -> int:
    if torch.cuda.is_available():
        # one worker per gpu for cuda devices
        return torch.cuda.device_count()
    elif torch.mps.is_available():
        # mps requires single process due to metal framework limitations
        return 1
    else:
        cpu_count = os.cpu_count() or 1
        return max(1, cpu_count // 2)
