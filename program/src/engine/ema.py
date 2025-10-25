from __future__ import annotations
import copy
from typing import Optional
import torch
from torch import nn

class ModelEma:
    def __init__(self, model: nn.Module, decay: float = 0.9998, device: Optional[torch.device] = None):
        self.ema: nn.Module = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay = float(decay)
        if device is not None:
            self.ema.to(device=device)

    @torch.no_grad()
    def update(self, model: nn.Module):
        msd, esd = model.state_dict(), self.ema.state_dict()
        for k, v in esd.items():
            mv = msd[k]
            if v.dtype.is_floating_point:
                v.copy_(v * self.decay + mv * (1.0 - self.decay))
            else:
                v.copy_(mv)
