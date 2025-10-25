# src/tools/compare_models.py  (replace the whole file)
from __future__ import annotations
from pathlib import Path
import argparse, json
import pandas as pd
import matplotlib.pyplot as plt

def load_metrics(run_dir: Path) -> dict:
    m = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    return {
        "top1": float(m.get("top1", 0.0)),
        "macro_acc": float(m.get("macro_acc", 0.0)),
    }

def barplot(df: pd.DataFrame, y: str, title: str, out_png: Path):
    plt.figure(figsize=(8, 5))
    plt.bar(df["model"], df[y])
    plt.title(title)
    plt.ylabel(y.replace("_", " ").title())
    plt.ylim(0, 1.0)
    for i, v in enumerate(df[y].values):
        plt.text(i, min(0.98, v + 0.015), f"{v:.3f}", ha="center", va="bottom")
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=160)
    plt.close()

def main():
    ap = argparse.ArgumentParser("Compare model results (Top-1 & Macro) and plot charts")
    ap.add_argument("--mode", required=True, choices=["baseline", "optimized"])
    ap.add_argument("--resnet_test", help="outputs/resnet50/<mode>/<N>/test", required=True)
    ap.add_argument("--effnet_test", help="outputs/efficientnet_b3/<mode>/<N>/test", required=True)
    ap.add_argument("--vit_test", help="(optional) outputs/vit_b16/<mode>/<N>/test", default=None)
    ap.add_argument("--out_csv", default=None)
    args = ap.parse_args()

    rows = []
    for name, p in [
        ("ResNet-50", args.resnet_test),
        ("EffNet-B3", args.effnet_test),
        ("ViT-B/16", args.vit_test),
    ]:
        if not p:
            continue
        run = Path(p)
        mets = load_metrics(run)
        rows.append({"model": name, "top1": mets["top1"], "macro_acc": mets["macro_acc"], "dir": str(run)})

    if len(rows) < 2:
        raise SystemExit("Need at least two models to compare. Provide --resnet_test and --effnet_test.")

    df = pd.DataFrame(rows)
    out_root = Path(f"outputs/compare/{args.mode}")
    out_root.mkdir(parents=True, exist_ok=True)
    out_csv = Path(args.out_csv) if args.out_csv else (out_root / "summary.csv")
    df.to_csv(out_csv, index=False)

    barplot(df, "top1",      f"Top-1 Accuracy ({args.mode})", out_root / "top1_bar.png")
    barplot(df, "macro_acc", f"Average Accuracy per Class ({args.mode})", out_root / "macro_acc_bar.png")

    print(f"Saved: {out_csv}")
    print(f"Saved: {out_root / 'top1_bar.png'}")
    print(f"Saved: {out_root / 'macro_acc_bar.png'}")

if __name__ == "__main__":
    main()
