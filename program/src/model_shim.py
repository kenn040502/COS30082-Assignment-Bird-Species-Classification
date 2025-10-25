# src/model_shim.py
from __future__ import annotations
import sys
from pathlib import Path

# Ensure project root (the folder that contains "model/") is on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# Import the model factory from model/build.py
from model.build import create_model  # <- this is the only import you need

__all__ = ["create_model"]
