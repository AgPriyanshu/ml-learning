import torch


def bce_loss( pos_weight: float = 2.0, device: str = "cuda"):
    bce = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight, device=device)
    )

    def criterion(logits, target, keep_mask):
        # logits, target: (B,1,H,W); keep_mask: (B,1,H,W) True=use
        if keep_mask is not None:
            logits = logits[keep_mask]
            target = target[keep_mask]
        loss_bce = bce(logits, target)
        probs = torch.sigmoid(logits)
        inter = (probs * target).sum()
        denom = probs.sum() + target.sum()
        loss_dice = 1.0 - (2.0 * inter + 1.0) / (denom + 1.0)
        return 0.5 * loss_bce + 0.5 * loss_dice

    return criterion
