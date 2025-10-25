# src/tools/plot_training_curves.py
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser("Plot training curves from a run folder")
    ap.add_argument("--run_dir", required=True, help="e.g. outputs\\efficientnet_b3\\baseline\\3rd")
    ap.add_argument("--pop", action="store_true", help="also display the figures after saving")
    args = ap.parse_args()

    run = Path(args.run_dir)
    csv = run / "training_log.csv"
    if not csv.exists():
        raise SystemExit(f"Missing: {csv}")

    df = pd.read_csv(csv)

    # ---- Accuracy ----
    plt.figure(figsize=(12,6))
    if "train_acc" in df.columns:
        plt.plot(df["epoch"], df["train_acc"], label="Train Acc", linewidth=2)
    if "val_acc" in df.columns:
        plt.plot(df["epoch"], df["val_acc"], label="Val Acc", linewidth=2)
    plt.title("Accuracy vs Epoch"); plt.xlabel("Epoch"); plt.ylabel("Accuracy")
    plt.legend(); plt.tight_layout()
    acc_path = run / "accuracy_curve.png"
    plt.savefig(acc_path, dpi=180)
    if args.pop: plt.show()
    plt.close()

    # ---- Loss ----
    plt.figure(figsize=(12,6))
    if "train_loss" in df.columns:
        plt.plot(df["epoch"], df["train_loss"], label="Train Loss", linewidth=2)
    if "val_loss" in df.columns:
        plt.plot(df["epoch"], df["val_loss"], label="Val Loss", linewidth=2)
    plt.title("Loss vs Epoch"); plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.legend(); plt.tight_layout()
    loss_path = run / "loss_curve.png"
    plt.savefig(loss_path, dpi=180)
    if args.pop: plt.show()
    plt.close()

    # ---- Learning Rate (if present) ----
    lr_col = None
    for c in ["lr", "learning_rate", "LR"]:
        if c in df.columns:
            lr_col = c; break

    if lr_col is not None:
        plt.figure(figsize=(12,4.5))
        plt.plot(df["epoch"], df[lr_col], linewidth=2)
        plt.title("Learning Rate Schedule"); plt.xlabel("Epoch"); plt.ylabel("LR")
        plt.tight_layout()
        lr_path = run / "learning_rate_curve.png"
        plt.savefig(lr_path, dpi=180)
        if args.pop: plt.show()
        plt.close()

    print("Saved:", acc_path)
    print("Saved:", loss_path)
    if lr_col is not None:
        print("Saved:", lr_path)
    else:
        print("No LR column found; skipped LR plot.")

if __name__ == "__main__":
    main()
