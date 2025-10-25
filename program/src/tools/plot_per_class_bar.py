from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Optional, Dict
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load_names(names_file: Optional[str]) -> Dict[int, str]:
    """
    Optional class-name mapping: one name per line, index = class id.
    Line 0 -> class 0, line 1 -> class 1, ...
    """
    if not names_file:
        return {}
    names = {}
    with open(names_file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            name = line.strip()
            if name:
                names[i] = name
    return names

def load_csv(p: str | Path) -> pd.DataFrame:
    df = pd.read_csv(p)
    # expected columns: class, acc
    if "class" not in df.columns or "acc" not in df.columns:
        raise SystemExit(f"{p} must contain columns: class, acc")
    df = df.copy()
    df["class"] = df["class"].astype(int)
    df["acc"] = df["acc"].astype(float)
    return df

def main():
    ap = argparse.ArgumentParser("Plot per-class accuracy bar chart (no confusion matrix)")
    ap.add_argument("--csv", nargs="+", required=True,
                    help="One or more per_class_accuracy.csv files to plot (for comparison).")
    ap.add_argument("--labels", nargs="*", default=None,
                    help="Optional short labels for the legend (same order as --csv).")
    ap.add_argument("--names", default=None,
                    help="Optional class-name file (one name per line; index is class id).")
    ap.add_argument("--out", default=None,
                    help="Output PNG path. Default: <first_csv_parent>/per_class_accuracy_bar.png")
    ap.add_argument("--title", default="Per-Class Accuracy (Test)")
    ap.add_argument("--sort", choices=["asc", "desc", "none"], default="asc",
                    help="Sort classes by accuracy (asc=hardest first).")
    ap.add_argument("--topk", type=int, default=None,
                    help="If set, only plot the worst/best K classes after sorting.")
    ap.add_argument("--dpi", type=int, default=180)
    ap.add_argument("--width_per_10", type=float, default=3.0,
                    help="Figure width per 10 classes (for auto sizing).")
    args = ap.parse_args()

    # Load dataframes
    dfs = [load_csv(p) for p in args.csv]
    num_classes = dfs[0]["class"].nunique()
    for i, df in enumerate(dfs[1:], start=2):
        if df["class"].nunique() != num_classes:
            raise SystemExit(f"--csv #{i} has {df['class'].nunique()} classes, expected {num_classes}.")

    # Merge for ordering
    base = dfs[0][["class", "acc"]].rename(columns={"acc": "acc_0"})
    for i, df in enumerate(dfs[1:], start=1):
        base = base.merge(df[["class", "acc"]].rename(columns={"acc": f"acc_{i}"}), on="class", how="inner")

    # Sorting
    if args.sort != "none":
        key_col = "acc_0"  # sort by the first CSV's accuracy
        ascending = (args.sort == "asc")
        base = base.sort_values(key_col, ascending=ascending).reset_index(drop=True)

    # TopK filter
    if args.topk is not None and args.topk < len(base):
        base = base.iloc[:args.topk].reset_index(drop=True)

    # Class labels
    id2name = load_names(args.names)
    def class_label(cid: int) -> str:
        return id2name.get(cid, str(cid))

    x_labels = [class_label(int(c)) for c in base["class"].tolist()]
    x = np.arange(len(base), dtype=float)

    # Figure size auto: width grows with number of classes
    width_in = max(10.0, (len(base) / 10.0) * args.width_per_10)
    height_in = 6.0 if len(dfs) == 1 else 7.0

    plt.figure(figsize=(width_in, height_in))
    n = len(dfs)
    bar_w = min(0.8 / n, 0.35)  # keep bars readable

    # Legend labels
    if args.labels and len(args.labels) == n:
        lbls = args.labels
    else:
        # Default legend labels from filenames
        def short(p):
            p = Path(p)
            # e.g., ".../outputs/resnet50/optimized/1st/test/per_class_accuracy.csv" -> "resnet50/optimized/1st"
            try:
                parts = p.parts
                # find "outputs" and take next 3 parts if possible
                if "outputs" in parts:
                    i = parts.index("outputs")
                    return "/".join(parts[i+1:i+4])
            except Exception:
                pass
            return p.parent.name
        lbls = [short(p) for p in args.csv]

    # Draw bars
    for i in range(n):
        y = base[f"acc_{i if i>0 else 0}"].values if i>0 else base["acc_0"].values
        plt.bar(x + (i - (n-1)/2) * bar_w, y, width=bar_w, label=lbls[i])

    # Aesthetics
    plt.title(args.title)
    plt.ylabel("Accuracy")
    plt.ylim(0.0, 1.0)
    plt.xticks(x, x_labels, rotation=90 if len(base) > 30 else 45, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    if n > 1:
        plt.legend()

    plt.tight_layout()

    # Output path
    out_png = Path(args.out) if args.out else (Path(args.csv[0]).parent / "per_class_accuracy_bar.png")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=args.dpi)
    print("Saved:", out_png)

if __name__ == "__main__":
    main()
