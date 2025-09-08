import torch
from torch import nn


def cosine_with_warmup(total_steps: int, warmup_steps: int = 1000):
    import math

    def _lambda(step):
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return _lambda
