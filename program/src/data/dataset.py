from __future__ import annotations
from pathlib import Path
from typing import List, Tuple
import json
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from src.engine.utils import load_json

def _imagenet_stats():
    return (0.485,0.456,0.406), (0.229,0.224,0.225)

def _dataset_stats(stats_file: str|Path):
    info = load_json(stats_file, default=None)
    if not info or "mean" not in info or "std" not in info:
        return _imagenet_stats()
    mean = tuple(float(x) for x in info["mean"]); std = tuple(float(x) for x in info["std"])
    return mean, std

def _build_transform(img_size: int, aug_level: str, norm: str, stats_file: str|Path):
    if norm == "dataset": mean,std = _dataset_stats(stats_file)
    else: mean,std = _imagenet_stats()
    if aug_level == "strong":
        return T.Compose([
            T.Resize(int(img_size*1.14)), T.RandomResizedCrop(img_size, scale=(0.7,1.0)),
            T.RandomHorizontalFlip(), T.AutoAugment(),
            T.ToTensor(), T.Normalize(mean,std), T.RandomErasing(p=0.25)
        ])
    else:
        return T.Compose([T.Resize(int(img_size*1.14)), T.CenterCrop(img_size), T.ToTensor(), T.Normalize(mean,std)])

class TxtDataset(Dataset):
    def __init__(self, data_root: str|Path, split: str, img_size: int, aug_level="baseline", norm="imagenet", stats_file="data/stats.json"):
        self.root = Path(data_root)
        txt = "Train.txt" if split.lower().startswith("train") else "Test.txt"
        self.paths: List[Path] = []; self.labels: List[int] = []
        with open(self.root/"../data"/txt if (self.root/"../data"/txt).exists() else self.root/txt, "r", encoding="utf-8") as f:
            for line in f:
                line=line.strip()
                if not line: continue
                im, lab = line.rsplit(" ",1); lab=int(lab)
                p = Path(im)
                if not p.is_absolute():
                    # if only filename provided, try Train/ or Test/ under data_root
                    cands = [self.root/im, self.root/"Train"/im, self.root/"Test"/im]
                    for c in cands:
                        if c.exists(): p = c; break
                self.paths.append(p); self.labels.append(lab)
        self.num_classes = max(self.labels) + 1
        self.tx = _build_transform(img_size, aug_level, norm, stats_file)

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor,int]:
        p = self.paths[idx]
        with Image.open(p).convert("RGB") as img:
            x = self.tx(img)
        return x, self.labels[idx]
