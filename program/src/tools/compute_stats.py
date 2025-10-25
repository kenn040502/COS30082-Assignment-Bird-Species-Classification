from __future__ import annotations
from pathlib import Path
import argparse, json
from PIL import Image
import numpy as np

def main():
    ap = argparse.ArgumentParser("Compute dataset mean/std (RGB) from Train subset")
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--subset", default="Train")
    ap.add_argument("--max_images", type=int, default=10000)
    ap.add_argument("--out", default="data/stats.json")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    txt = "Train.txt"
    mean = np.zeros(3); var = np.zeros(3); npx=0
    count=0
    with open(data_root/txt, "r", encoding="utf-8") as f:
        for line in f:
            if count >= args.max_images: break
            line=line.strip(); 
            if not line: continue
            im,_ = line.rsplit(" ",1)
            p = Path(im)
            if not p.is_absolute():
                cands = [data_root/im, data_root/"Train"/im]
                for c in cands:
                    if c.exists(): p=c; break
            img = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32)/255.0
            h,w,_ = img.shape
            mean += img.reshape(-1,3).sum(axis=0)
            var  += (img.reshape(-1,3)**2).sum(axis=0)
            npx  += h*w
            count+=1

    mean /= npx; var = var/npx - mean**2
    std = np.sqrt(np.maximum(var, 1e-8))
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"mean": mean.tolist(), "std": std.tolist()}, indent=2), encoding="utf-8")
    print("Saved", out)

if __name__=="__main__": main()
