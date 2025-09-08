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


class EMA:
    """Exponential Moving Average of model weights (eval-time smoothing)."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {
            k: v.detach().clone()
            for k, v in model.state_dict().items()
            if v.dtype.is_floating_point
        }

    def update(self, model: nn.Module):
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if k in self.shadow and v.dtype.is_floating_point:
                    self.shadow[k].mul_(self.decay).add_(
                        v.detach(), alpha=1.0 - self.decay
                    )

    def apply_to(self, model: nn.Module):
        model.load_state_dict({**model.state_dict(), **self.shadow}, strict=False)


lambda_func = LambdaLR(
    optim,
    cosine_with_warmup(total_steps, warmup_steps=min(1000, total_steps // 10)),
)
