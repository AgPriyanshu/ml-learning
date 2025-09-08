def make_optimizer(lr, model, weight_decay, backbone_lr_mult):
    # param groups: smaller LR for encoder, larger for decoder/seghead
    enc_params = list(model.encoder.parameters())
    dec_params = [
        p for n, p in model.named_parameters() if not n.startswith("encoder.")
    ]
    param_groups = [
        {"params": enc_params, "lr": lr * backbone_lr_mult},
        {"params": dec_params, "lr": lr},
    ]
    try:
        return AdamW(param_groups, lr=lr, weight_decay=weight_decay, fused=True)
    except TypeError:
        return AdamW(param_groups, lr=lr, weight_decay=weight_decay)
