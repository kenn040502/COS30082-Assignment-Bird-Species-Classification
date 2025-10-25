from __future__ import annotations
from pathlib import Path
import argparse, json, pandas as pd

def main():
    ap = argparse.ArgumentParser("Re-create training summary.txt from a run folder")
    ap.add_argument("--run_dir", required=True, help="e.g. outputs/resnet50/optimized/1st")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--gpu_name", default="NVIDIA GPU")
    ap.add_argument("--mode", default=None, help="baseline or optimized (auto if folder name contains it)")
    args = ap.parse_args()

    run = Path(args.run_dir)
    mode = args.mode or ("optimized" if "optimized" in str(run) else "baseline")

    # Load CSV and best.json
    log_csv = run / "training_log.csv"
    best_json = run / "best.json"
    if not log_csv.exists():
        raise SystemExit(f"Missing: {log_csv}")
    df = pd.read_csv(log_csv)

    best_epoch =  int(json.loads(best_json.read_text())["epoch"]) if best_json.exists() else int(df["val_acc"].idxmax())+1
    best_acc   =  float(df.loc[best_epoch-1, "val_acc"])

    # Build summary text
    lines = []
    lines.append(f"Mode              : {mode}")
    lines.append(f"Run folder        : {run.resolve()}")
    lines.append(f"Device/GPU        : {args.device} / {args.gpu_name}")
    lines.append(f"Epochs            : requested={int(df['epoch'].iloc[-1])}, best_epoch={best_epoch}")
    lines.append(f"Optimizer/Sched   : (see console) | (see console)")
    lines.append(f"Label smoothing   : (see console)")
    lines.append(f"AMP               : True")
    lines.append(f"Best val top1     : {best_acc:.3f}")
    lines.append("Artifacts         :")
    lines.append(f"  - CSV           : {log_csv}")
    lines.append(f"  - Curves        : {run/'accuracy_curve.png'} , {run/'loss_curve.png'} , {run/'learning_rate_curve.png'}")
    lines.append(f"  - Checkpoint    : {run/'checkpoints'/'best.pt'}")
    lines.append(f"  - Best meta     : {best_json}")
    lines.append("")
    lines.append("Epoch records (rounded):")
    lines.append("epoch  train_acc  train_loss  val_acc  val_top5  val_loss      lr")
    for _, r in df.iterrows():
        lines.append(f"{int(r.epoch):<5d}  {r.train_acc:>9.3f}  {r.train_loss:>10.3f}  {r.val_acc:>7.3f}  {r.val_top5:>8.3f}  {r.val_loss:>8.3f}  {float(r.lr):>8.6f}")

    (run / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("Wrote:", run / "summary.txt")

if __name__ == "__main__":
    main()
