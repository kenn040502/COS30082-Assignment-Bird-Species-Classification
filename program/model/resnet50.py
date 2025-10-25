from __future__ import annotations
import argparse, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (ROOT, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.append(str(p))

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from src.data.dataset import TxtDataset
from src.engine.train import train_loop
from src.engine.scheduler import cosine_warmup
from src.engine.utils import seed_everything
from src.engine.ema import ModelEma
from src.engine.mixup import MixupCutmix
from src.engine.run_namer import next_ordinal_run_name
from model.build import create_model

MODEL_NAME = "resnet50"

TUNE = {
    "variant": "baseline",
    "img_size": 256,
    "batch": 32,
    "epochs": 100,
    "lr": 5e-4,
    "wd": 2e-4,
    "run_name": "auto",
    "seed": 42,
    "norm": "dataset",
    "stats_file": "data/stats.json",
    "smooth": 0.1,
    "warmup": 5,
    "dropout": 0.0,     
    "mixup": 0.1,
    "cutmix": 0.5,
    "mixprob": 0.5,
    "ema": 0.9995,
}

def apply_defaults(args, defaults):
    for k, v in defaults.items():
        if getattr(args, k, None) is None:
            setattr(args, k, v)

def main():
    ap = argparse.ArgumentParser("ResNet-50 trainer")
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--variant", choices=["baseline", "opt"], default=None)
    ap.add_argument("--img_size", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--wd", type=float, default=None)
    ap.add_argument("--run_name", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--norm", choices=["imagenet", "dataset"], default=None)
    ap.add_argument("--stats_file", default=None)
    ap.add_argument("--smooth", type=float, default=None)
    ap.add_argument("--warmup", type=int, default=None)
    ap.add_argument("--dropout", type=float, default=None)
    ap.add_argument("--mixup", type=float, default=None)
    ap.add_argument("--cutmix", type=float, default=None)
    ap.add_argument("--mixprob", type=float, default=None)
    ap.add_argument("--ema", type=float, default=None)
    args = ap.parse_args()
    apply_defaults(args, TUNE)

    is_opt = (args.variant == "opt")
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Strong aug only in optimized mode
    train_ds = TxtDataset(
        args.data_root, "train", args.img_size,
        aug_level=("strong" if is_opt else "baseline"),
        norm=args.norm, stats_file=args.stats_file
    )
    val_ds = TxtDataset(
        args.data_root, "test", args.img_size,
        aug_level="baseline",
        norm=args.norm, stats_file=args.stats_file
    )
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=2, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=max(1, args.batch * 2), shuffle=False, num_workers=2, pin_memory=True)

    model = create_model(MODEL_NAME, num_classes=train_ds.num_classes, pretrained=True, dropout=(args.dropout or 0.0))
    model.to(device)

    if is_opt:
        ema = ModelEma(model, decay=args.ema, device=device)
        mixup_fn = MixupCutmix(train_ds.num_classes, mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, prob=args.mixprob)
        smooth, warmup = args.smooth, args.warmup
        if args.lr is None: args.lr = 5e-4
        if args.wd is None: args.wd = 2e-4
    else:
        ema, mixup_fn = None, None
        smooth, warmup = 0.0, 0
        if args.lr is None: args.lr = 1e-3
        if args.wd is None: args.wd = 1e-4

    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sch = cosine_warmup(opt, warmup_epochs=warmup, total_epochs=args.epochs, base_lr=args.lr)

    mode_dir = "optimized" if is_opt else "baseline"
    model_root = ROOT / f"outputs/{MODEL_NAME}/{mode_dir}"
    rn = (args.run_name or "").strip()
    if rn.lower() in ("", "auto", "next"):
        rn = next_ordinal_run_name(model_root)
    out_dir = model_root / rn

    train_loop(
        model, opt, sch, train_dl, val_dl, device,
        epochs=args.epochs, out_dir=out_dir,
        label_smoothing=smooth, ema=ema, mixup_fn=mixup_fn,
        run_tag=mode_dir,
        extra_hparams={"img_size": str(args.img_size), "batch": str(args.batch), "norm": args.norm}
    )

if __name__ == "__main__":
    main()
