import torch


class EMA:
    """Exponential Moving Average of model weights (eval-time smoothing)."""

    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {
            k: v.detach().clone()
            for k, v in model.state_dict().items()
            if v.dtype.is_floating_point
        }

    def update(self, model: torch.nn.Module):
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if k in self.shadow and v.dtype.is_floating_point:
                    self.shadow[k].mul_(self.decay).add_(
                        v.detach(), alpha=1.0 - self.decay
                    )

    def apply_to(self, model: torch.nn.Module):
        model.load_state_dict({**model.state_dict(), **self.shadow}, strict=False)
