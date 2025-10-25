from __future__ import annotations
from pathlib import Path
import json, random, os
import numpy as np
import torch

def save_json(obj, path: str | Path):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def load_json(path: str | Path, default=None):
    p = Path(path)
    if not p.exists(): return default
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def seed_everything(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
