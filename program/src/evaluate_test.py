# src/evaluate_test.py  (replace the whole file)
from __future__ import annotations
from pathlib import Path
import sys, argparse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.engine.eval import evaluate

def main():
    ap = argparse.ArgumentParser("Evaluate a trained checkpoint on Test")
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--model", required=True,
                    choices=["resnet50", "efficientnet_b3"])  # ViT removed
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--img_size", type=int, required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--norm", choices=["imagenet","dataset"], default="imagenet")
    ap.add_argument("--batch", type=int, default=16)   # keep OOM-friendly options
    ap.add_argument("--amp", action="store_true")
    args = ap.parse_args()

    evaluate(
        model_name=args.model,
        ckpt=args.ckpt,
        data_root=args.data_root,
        img_size=args.img_size,
        out_dir=args.out_dir,
        tta=args.tta,
        norm=args.norm,
        batch_size=args.batch,
        use_amp=args.amp,
    )

if __name__ == "__main__":
    main()
