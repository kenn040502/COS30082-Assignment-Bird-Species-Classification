from __future__ import annotations
from pathlib import Path
from typing import Dict
import numpy as np, pandas as pd, torch, torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

from src.data.dataset import TxtDataset
from src.engine.utils import save_json
from src.model_shim import create_model

@torch.no_grad()
def _topk_hits(probs: torch.Tensor, targets: torch.Tensor, k: int = 5) -> float:
    _, pred = probs.topk(k, dim=1, largest=True, sorted=True)
    return float((pred == targets.view(-1,1)).any(dim=1).float().mean().item())

def evaluate(model_name: str,
             ckpt: str | Path,
             data_root: str | Path,
             img_size: int,
             out_dir: str | Path,
             tta: bool = False,
             norm: str = "imagenet",
             batch_size: int = 16,
             use_amp: bool = True
             ) -> Dict[str, float]:

    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    ds = TxtDataset(data_root, split="test", img_size=img_size, aug_level="baseline", norm=norm)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(model_name, num_classes=ds.num_classes, pretrained=False)

    # Load weights (prefer safe mode if available)
    try:
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(ckpt, map_location="cpu") 
    model.load_state_dict(state, strict=True)
    model.to(device).eval()

    y_true_all, y_pred_all = [], []
    top1_sum=0.0; top5_sum=0.0; n_total=0

    amp_ctx = torch.amp.autocast(device_type="cuda", enabled=(use_amp and device.type=="cuda"))

    with torch.no_grad():
        for x,y in dl:
            x,y = x.to(device), y.to(device)
            with amp_ctx:
                logits = model(x)
                if tta:
                    logits = (logits + model(torch.flip(x, dims=[3]))) / 2.0
                probs = F.softmax(logits, dim=1)
                pred1 = probs.argmax(1)

            b = y.size(0)
            top1_sum += (pred1==y).float().sum().item()
            top5_sum += _topk_hits(probs,y,5)*b
            n_total  += b

            y_true_all.append(y.cpu().numpy())
            y_pred_all.append(pred1.cpu().numpy())

    y_true = np.concatenate(y_true_all); y_pred = np.concatenate(y_pred_all)
    top1 = float(top1_sum/max(1,n_total)); top5 = float(top5_sum/max(1,n_total))

    # per-class + macro
    rows=[]; num_classes=ds.num_classes
    for c in range(num_classes):
        m = (y_true==c); acc_c = float((y_pred[m]==y_true[m]).mean()) if m.any() else float("nan")
        rows.append({"class": c, "acc": acc_c})
    per_df = pd.DataFrame(rows); per_df.to_csv(out_dir/"per_class_accuracy.csv", index=False)
    macro_acc = float(np.nanmean(per_df["acc"].values))

    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    pd.DataFrame(cm).to_csv(out_dir/"confusion_matrix.csv", index=False)
    try:
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(8,8)); plt.imshow(cm, interpolation="nearest"); plt.title("Confusion Matrix"); plt.colorbar(); plt.tight_layout()
        fig.savefig(out_dir/"confusion_matrix.png", dpi=200); plt.close(fig)
    except Exception:
        pass

    metrics = {"top1": top1, "top5": top5, "macro_acc": macro_acc}
    save_json(metrics, out_dir/"metrics.json")
    print(f"Test — Top1: {top1:.3f} | Top5: {top5:.3f} | Macro: {macro_acc:.3f}")
    try:
        lines = [
            "Test Summary",
            f"Checkpoint   : {Path(ckpt).resolve()}",
            f"Out dir      : {out_dir.resolve()}",
            f"Batch / AMP  : {batch_size} / {use_amp}",
            f"Top-1        : {top1:.3f}",
            f"Top-5        : {top5:.3f}",
            f"Macro Acc    : {macro_acc:.3f}",
            "Artifacts    :",
            f"  - metrics  : {out_dir/'metrics.json'}",
            f"  - per-class: {out_dir/'per_class_accuracy.csv'}",
            f"  - conf mat : {out_dir/'confusion_matrix.csv'}",
        ]
        (out_dir/"summary.txt").write_text("\n".join(lines), encoding="utf-8")
    except Exception:
        pass

    return metrics
