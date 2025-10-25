from __future__ import annotations
import torch
import torch.nn.functional as F

class MixupCutmix:
    def __init__(self, num_classes: int, mixup_alpha=0.2, cutmix_alpha=1.0, prob=1.0):
        self.num_classes = num_classes
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.prob = prob

    def _one_hot(self, y: torch.Tensor):
        return F.one_hot(y, num_classes=self.num_classes).float()

    def _rand_bbox(self, W, H, lam):
        import math, random
        cut_rat = (1. - lam) ** 0.5
        cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
        cx, cy = random.randint(0, W), random.randint(0, H)
        x1, y1 = max(cx - cut_w // 2, 0), max(cy - cut_h // 2, 0)
        x2, y2 = min(cx + cut_w // 2, W), min(cy + cut_h // 2, H)
        return x1, y1, x2, y2

    def __call__(self, x: torch.Tensor, y: torch.Tensor):
        import random
        if random.random() > self.prob:
            return x, self._one_hot(y), 1.0, (False, False)
        B, C, H, W = x.size()
        perm = torch.randperm(B, device=x.device)
        x2, y2 = x[perm], y[perm]
        use_cutmix = (self.cutmix_alpha is not None and self.cutmix_alpha > 0.0)
        use_mixup  = (self.mixup_alpha  is not None and self.mixup_alpha  > 0.0)
        if use_cutmix:
            import numpy as np
            lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
            x1, y1, x2b, y2b = self._rand_bbox(W, H, lam)
            x[:, :, y1:y2b, x1:x2b] = x2[:, :, y1:y2b, x1:x2b]
            lam = 1. - ((x2b - x1) * (y2b - y1) / float(W * H))
            y_one = self._one_hot(y); y2_one = self._one_hot(y2)
            y_m = y_one * lam + y2_one * (1. - lam)
            return x, y_m, lam, (True, False)
        if use_mixup:
            import numpy as np
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            x_m = x * lam + x2 * (1. - lam)
            y_one = self._one_hot(y); y2_one = self._one_hot(y2)
            y_m = y_one * lam + y2_one * (1. - lam)
            return x_m, y_m, lam, (False, True)
        return x, self._one_hot(y), 1.0, (False, False)
