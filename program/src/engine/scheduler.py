from __future__ import annotations
from torch.optim.lr_scheduler import LambdaLR

def cosine_warmup(optimizer, warmup_epochs: int, total_epochs: int, base_lr: float):
    import math
    def lr_lambda(epoch):
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        # cosine from warmup to end
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda=lr_lambda)
