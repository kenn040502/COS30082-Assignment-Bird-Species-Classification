# src/engine/train.py
from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict, List
import time, os, sys, csv

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .utils import save_json

def _format_hms(seconds: float) -> str:
    m, s = divmod(int(seconds), 60); h, m = divmod(m, 60); return f"{h:d}:{m:02d}:{s:02d}"

def _make_plots(log_rows: List[Dict], out_dir: Path):
    """Return paths (acc_png, loss_png, lr_png)."""
    try:
        import pandas as pd, matplotlib.pyplot as plt
    except Exception:
        return (None, None, None)
    try:
        df = pd.DataFrame(log_rows)
    except Exception:
        return (None, None, None)

    acc_png  = out_dir / "accuracy_curve.png"
    loss_png = out_dir / "loss_curve.png"
    lr_png   = out_dir / "learning_rate_curve.png"

    # Accuracy
    try:
        plt.figure(figsize=(11,6))
        if "train_acc" in df: plt.plot(df["epoch"], df["train_acc"], label="Train Acc")
        if "val_acc"   in df: plt.plot(df["epoch"], df["val_acc"],   label="Val Acc")
        plt.title("Accuracy vs. Epoch"); plt.xlabel("Epoch"); plt.ylabel("Accuracy")
        plt.legend(); plt.tight_layout(); plt.savefig(acc_png, dpi=150); plt.close()
    except Exception:
        acc_png = None

    # Loss
    try:
        plt.figure(figsize=(11,6))
        if "train_loss" in df: plt.plot(df["epoch"], df["train_loss"], label="Train Loss")
        if "val_loss"   in df: plt.plot(df["epoch"], df["val_loss"],   label="Val Loss")
        plt.title("Loss vs. Epoch"); plt.xlabel("Epoch"); plt.ylabel("Loss")
        plt.legend(); plt.tight_layout(); plt.savefig(loss_png, dpi=150); plt.close()
    except Exception:
        loss_png = None

    # Learning rate
    try:
        if "lr" in df:
            plt.figure(figsize=(11,4.5))
            plt.plot(df["epoch"], df["lr"], label="Learning Rate")
            plt.title("Learning Rate Schedule"); plt.xlabel("Epoch"); plt.ylabel("LR")
            plt.tight_layout(); plt.savefig(lr_png, dpi=150); plt.close()
        else:
            lr_png = None
    except Exception:
        lr_png = None

    return (acc_png, loss_png, lr_png)

def _topk_accuracy(logits: torch.Tensor, targets: torch.Tensor, ks=(1,5)):
    maxk = max(ks)
    _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t(); correct = pred.eq(targets.view(1,-1))
    B = targets.size(0); return {k: correct[:k].any(dim=0).float().sum().item()/max(1,B) for k in ks}

def _optim_desc(optimizer: torch.optim.Optimizer) -> str:
    name = optimizer.__class__.__name__; g = optimizer.param_groups[0]
    lr = g.get("lr", None); wd = g.get("weight_decay", 0.0)
    lr_s = f"{lr:.4g}" if isinstance(lr,(int,float)) else "?"
    wd_s = f"{wd:.4g}" if isinstance(wd,(int,float)) else str(wd)
    return f"{name}(wd={wd_s}), LR: {lr_s}"

def _write_training_summary(out_dir: Path, mode: str, device_name: str, gpu_name: str,
                            epochs_req: int, best_acc: float, best_epoch: int,
                            optimizer_desc: str, scheduler_name: str,
                            label_smoothing: float, use_amp: bool,
                            log_rows: List[Dict], elapsed_hms: str,
                            extra_hparams: Dict[str,str]|None):
    """Create summary.txt with header + full epoch table (rounded)."""
    lines = []
    lines.append(f"Mode              : {mode}")
    lines.append(f"Run folder        : {out_dir}")
    lines.append(f"Device/GPU        : {device_name} / {gpu_name}")
    lines.append(f"Epochs            : requested={epochs_req}, best_epoch={best_epoch}")
    lines.append(f"Optimizer/Sched   : {optimizer_desc} | {scheduler_name}")
    lines.append(f"Label smoothing   : {label_smoothing}")
    lines.append(f"AMP               : {use_amp}")
    if extra_hparams:
        pairs = ", ".join(f"{k}={v}" for k,v in extra_hparams.items())
        lines.append(f"Hparams           : {pairs}")
    lines.append(f"Best val top1     : {best_acc:.3f}")
    lines.append("Artifacts         :")
    lines.append(f"  - CSV           : {out_dir/'training_log.csv'}")
    lines.append(f"  - Curves        : {out_dir/'accuracy_curve.png'} , {out_dir/'loss_curve.png'} , {out_dir/'learning_rate_curve.png'}")
    lines.append(f"  - Checkpoint    : {out_dir/'checkpoints'/'best.pt'}")
    lines.append(f"  - Best meta     : {out_dir/'best.json'}")
    lines.append("")
    lines.append("Epoch records (rounded):")
    lines.append("epoch  train_acc  train_loss  val_acc  val_top5  val_loss      lr")

    for r in log_rows:
        e  = int(r.get("epoch", 0))
        ta = float(r.get("train_acc", 0.0))
        tl = float(r.get("train_loss", 0.0))
        va = float(r.get("val_acc", 0.0))
        v5 = float(r.get("val_top5", 0.0))
        vl = float(r.get("val_loss", 0.0))
        lr = float(r.get("lr", 0.0))
        lines.append(f"{e:<5d}  {ta:>9.3f}  {tl:>10.3f}  {va:>7.3f}  {v5:>8.3f}  {vl:>8.3f}  {lr:>8.6f}")

    lines.append("")
    lines.append(f"Total time        : {elapsed_hms}")

    (out_dir / "summary.txt").write_text("\n".join(lines), encoding="utf-8")

def train_loop(model: nn.Module, optimizer, scheduler, train_loader: DataLoader, val_loader: DataLoader,
               device: torch.device, epochs: int, out_dir: str|Path, label_smoothing=0.0, ema=None, mixup_fn=None,
               max_norm: Optional[float]=1.0, auto_show=True, decimals: int=3, run_tag: str="baseline",
               extra_hparams: Dict[str,str]|None=None):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir/"checkpoints"; ckpt_dir.mkdir(exist_ok=True)

    use_amp = True
    scaler = torch.amp.GradScaler('cuda' if device.type=='cuda' else 'cpu')
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    device_name = "cuda" if device.type=="cuda" else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if device.type=="cuda" else "CPU"
    sched_name = scheduler.__class__.__name__ if scheduler is not None else "None"

    best_acc, best_epoch, t0 = 0.0, 0, time.time()

    print(f"Run folder: {out_dir}")
    print(f"Mode: {run_tag}")
    print(f"Device: {device_name}")
    print(f"GPU: {gpu_name}")
    print(f"Requested epochs: {epochs} | Stopped at: (running…)")
    print(f"{_optim_desc(optimizer)} | Scheduler: {sched_name}")
    print(f"Label smoothing: {label_smoothing} | AMP: {use_amp}")
    if extra_hparams: print("Hparams:", ", ".join(f"{k}={v}" for k,v in extra_hparams.items()))
    print(); print(f"{'epoch':<4} {'train_acc':>9} {'train_loss':>11} {'val_acc':>8} {'val_top5':>10} {'val_loss':>9} {'lr':>8}")
    print("-"*75)

    log_rows: List[Dict] = []

    for epoch in range(1, epochs+1):
        model.train(); total=0; correct_hard=0; loss_sum=0.0
        for x,y in train_loader:
            x,y = x.to(device), y.to(device)
            if mixup_fn is not None:
                x_used, y_mix, _, _ = mixup_fn(x,y); targets = y_mix; criterion_ = nn.BCEWithLogitsLoss()
            else:
                x_used, targets, criterion_ = x, y, criterion
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits = model(x_used); loss = criterion_(logits, targets)
            scaler.scale(loss).backward()
            if max_norm is not None:
                scaler.unscale_(optimizer); nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer); scaler.update()
            if ema is not None: ema.update(model)
            B = x.size(0); loss_sum += loss.item()*B; total += B
            with torch.no_grad():
                preds = logits.argmax(1); correct_hard += (preds==y).sum().item()
        train_loss = loss_sum/max(1,total); train_acc = correct_hard/max(1,total)

        # validation
        model.eval()
        with torch.no_grad():
            total=0; c1=0.0; c5=0.0; vloss=0.0
            for x,y in val_loader:
                x,y = x.to(device), y.to(device)
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    logits = (ema.ema if ema is not None else model)(x)
                    loss = criterion(logits, y)
                vloss += loss.item()*x.size(0)
                tk = _topk_accuracy(logits,y,ks=(1,5)); c1 += tk[1]*x.size(0); c5 += tk[5]*x.size(0); total += x.size(0)
            val_loss = vloss/max(1,total); val_acc = c1/max(1,total); val_top5 = c5/max(1,total)

        if scheduler is not None: scheduler.step()
        current_lr = float(optimizer.param_groups[0].get("lr", 0.0))

        row = {"epoch":epoch, "train_loss":float(train_loss), "val_loss":float(val_loss),
               "train_acc":float(train_acc), "val_acc":float(val_acc), "val_top5":float(val_top5),
               "lr": current_lr}
        log_rows.append(row)

        fmt = lambda v: f"{v:.{decimals}f}"
        print(f"{epoch:02d}   {fmt(train_acc):>7}   {fmt(train_loss):>9}   {fmt(val_acc):>7}  {fmt(val_top5):>8}  {fmt(val_loss):>8}  {fmt(current_lr):>7}")

        if val_acc > best_acc:
            best_acc, best_epoch = val_acc, epoch
            torch.save((ema.ema if ema is not None else model).state_dict(), ckpt_dir/"best.pt")
            save_json({"best_val_acc": float(best_acc), "epoch": best_epoch}, out_dir/"best.json")

    # CSV
    try:
        import pandas as pd
        pd.DataFrame(log_rows).to_csv(out_dir/"training_log.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    except Exception:
        pass

    # footer
    elapsed = _format_hms(time.time()-t0)
    print("-"*75); print(f"Best val acc: {best_acc:.{decimals}f} (epoch {best_epoch})")
    print(f"Total time: {elapsed}"); print(f"CSV: {out_dir/'training_log.csv'}"); print()

    # plots + auto-open
    acc_png, loss_png, lr_png = _make_plots(log_rows, out_dir)
    if acc_png is not None:
        print(f"Saved plots: {acc_png} , {loss_png} , {lr_png}")
    if auto_show:
        for p in [acc_png, loss_png, lr_png]:
            if p is None: continue
            try:
                if os.name=="nt": os.startfile(str(p))
                elif sys.platform=="darwin": os.system(f'open \"{p}\"')
                else: os.system(f'xdg-open \"{p}\" 2>/dev/null || true')
            except Exception:
                pass

    # write training summary
    _write_training_summary(
        out_dir=out_dir, mode=run_tag,
        device_name=device_name, gpu_name=gpu_name,
        epochs_req=epochs, best_acc=best_acc, best_epoch=best_epoch,
        optimizer_desc=_optim_desc(optimizer), scheduler_name=sched_name,
        label_smoothing=label_smoothing, use_amp=use_amp,
        log_rows=log_rows, elapsed_hms=elapsed, extra_hparams=extra_hparams
    )
