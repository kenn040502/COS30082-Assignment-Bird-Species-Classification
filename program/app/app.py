# app.py
from __future__ import annotations
import json, sys, time, re
from pathlib import Path
from typing import List, Dict, Tuple
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import gradio as gr

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))
from build import create_model  # build.py is at repo root

# ---- files in your Space (root-level) ----
MODEL_FILES = {
    "efficientnet_b3": ROOT / "best_eff.pt",
    "resnet50":        ROOT / "best_res.pt",
}
DEFAULTS = {
    "efficientnet_b3": {"img_size": 320, "dropout": 0.4},
    "resnet50":        {"img_size": 256, "dropout": 0.0},
}
IMAGENET_STATS = {"mean":[0.485,0.456,0.406], "std":[0.229,0.224,0.225]}
DATA_STATS = ROOT / "stats.json"               # you uploaded this at root
CLASS_NAMES_TXT = ROOT / "class_names.txt"     # optional; we’ll auto-build if missing
TRAIN_TXT = ROOT / "train.txt"                 # you uploaded at root
TEST_TXT  = ROOT / "test.txt"                  # optional

TOPK_MAX = 10

# ---------- class names ----------
_SEPS = re.compile(r"[\\/]+")
_TRAIL_NUM = re.compile(r"[_-]?\d+([_-]\d+)*$")

def _clean_name(s: str) -> str:
    s = Path(s).stem
    s = _TRAIL_NUM.sub("", s)
    s = re.sub(r"[_\-.]+", " ", s).strip()
    s = re.sub(r"\s{2,}", " ", s)
    return " ".join(w if w.isupper() else w.capitalize() for w in s.split())

def _guess_from_path(p: str) -> str:
    parts = _SEPS.split(p)
    if len(parts) >= 2:
        parent = parts[-2]
        if parent.lower() not in {"train","test","images","img","data"}:
            return _clean_name(parent)
    return _clean_name(parts[-1])

def _read_list(txt: Path):
    if not txt.exists(): return []
    out=[]
    for line in txt.read_text(encoding="utf-8").splitlines():
        line=line.strip()
        if not line: continue
        try:
            path, lbl = line.rsplit(" ", 1)
            out.append((path, int(lbl)))
        except Exception:
            parts = line.split()
            if len(parts)>=2 and parts[-1].isdigit():
                out.append((" ".join(parts[:-1]), int(parts[-1])))
    return out

def _auto_build_names() -> List[str]:
    from collections import defaultdict, Counter
    votes = defaultdict(Counter)
    for p,l in _read_list(TRAIN_TXT): votes[l][_guess_from_path(p)] += 1
    for p,l in _read_list(TEST_TXT):  votes[l][_guess_from_path(p)] += 1
    if not votes:  # fallback 200 generic
        return [f"Class {i}" for i in range(200)]
    max_lbl = max(votes.keys())
    names=[]
    for i in range(max_lbl+1):
        if i in votes and votes[i]:
            names.append(votes[i].most_common(1)[0][0])
        else:
            names.append(f"Class {i}")
    # also write a file so it’s cached in Space storage
    try:
        CLASS_NAMES_TXT.write_text("\n".join(names)+"\n", encoding="utf-8")
    except Exception:
        pass
    return names

def get_class_names() -> List[str]:
    if CLASS_NAMES_TXT.exists():
        return [ln.strip() for ln in CLASS_NAMES_TXT.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return _auto_build_names()

# ---------- normalization ----------
def get_stats(norm: str) -> Dict[str, List[float]]:
    if norm=="dataset" and DATA_STATS.exists():
        try:
            return json.loads(DATA_STATS.read_text(encoding="utf-8"))
        except Exception:
            pass
    return IMAGENET_STATS

# ---------- preprocessing ----------
def preprocess(pil: Image.Image, img_size: int, norm: str) -> torch.Tensor:
    img = pil.convert("RGB")
    w,h = img.size
    s = min(w,h); left=(w-s)//2; top=(h-s)//2
    img = img.crop((left, top, left+s, top+s)).resize((img_size,img_size), Image.BICUBIC)
    x = np.asarray(img).astype(np.float32)/255.0
    stats = get_stats(norm)
    mean = np.array(stats["mean"], dtype=np.float32); std = np.array(stats["std"], dtype=np.float32)
    x = (x-mean)/std
    x = np.transpose(x, (2,0,1))
    return torch.from_numpy(x).unsqueeze(0)

# ---------- load / cache models ----------
_MODEL_CACHE: Dict[Tuple[str,int], torch.nn.Module] = {}

@torch.inference_mode()
def load_model(model_name: str, num_classes: int) -> torch.nn.Module:
    key=(model_name,num_classes)
    if key in _MODEL_CACHE: return _MODEL_CACHE[key]
    m = create_model(model_name, num_classes=num_classes, pretrained=False, dropout=DEFAULTS[model_name]["dropout"])
    ckpt = MODEL_FILES[model_name]
    if ckpt.exists():
        state = torch.load(ckpt, map_location="cpu")
        m.load_state_dict(state, strict=True)
    m.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m.to(device)
    _MODEL_CACHE[key]=m
    return m

# ---------- inference ----------
@torch.inference_mode()
def run_inference(image: Image.Image, model_name: str, img_size: int, norm: str, topk: int):
    if image is None:
        return None, None, "Please upload an image."
    names = get_class_names()
    num_classes = len(names)
    model = load_model(model_name, num_classes=num_classes)
    device = next(model.parameters()).device

    x = preprocess(image, img_size, norm).to(device)

    t0=time.time()
    with torch.autocast(device_type=device.type, enabled=(device.type=="cuda")):
        logits = model(x)
        probs  = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
    dt=time.time()-t0

    idx = np.argsort(-probs)[:topk]
    labels=[names[i] for i in idx]
    scores=[float(probs[i]) for i in idx]

    rows = [[i + 1, labels[i], round(scores[i], 4)] for i in range(len(labels))]

    def bar(labels, scores, title):
        import matplotlib.pyplot as plt
        fig=plt.figure(figsize=(6.4,3.6)); ax=fig.add_subplot(111)
        ax.bar(labels, scores); ax.set_ylim(0,1); ax.set_ylabel("Probability"); ax.set_title(title)
        for i,v in enumerate(scores): ax.text(i, min(0.98, v+0.02), f"{v:.2f}", ha="center", va="bottom", fontsize=9)
        plt.xticks(rotation=30, ha="right"); plt.tight_layout(); return fig

    fig = bar(labels, scores, f"{model_name} — Top-{topk}")
    info=f"Model: {model_name} | ckpt: {MODEL_FILES[model_name].name if MODEL_FILES[model_name].exists() else 'N/A'} | size={img_size} | norm={norm} | time={dt*1000:.1f} ms"
    return rows, fig, info

# ---------- UI ----------
def build_ui():
    theme = gr.themes.Soft(primary_hue="blue", neutral_hue="slate")
    with gr.Blocks(title="Bird Classifier — ResNet-50 & EfficientNet-B3", theme=theme) as demo:
        gr.Markdown("# Bird Classifier")
        gr.Markdown("Upload an image. Choose a model. Top-5 predictions are shown as a table and bar chart. "
                    "If `class_names.txt` is missing, names are inferred from `train.txt`/`test.txt`.")

        with gr.Tab("Predict"):
            with gr.Row():
                img_in = gr.Image(type="pil", label="Upload image", height=360)
                with gr.Column():
                    choices = [k for k,v in MODEL_FILES.items() if v.exists()]
                    if not choices: choices = list(MODEL_FILES.keys())
                    model_dd = gr.Dropdown(choices=choices, value=choices[0], label="Model")
                    size_in  = gr.Slider(128, 384, step=16, value=DEFAULTS[choices[0]]["img_size"], label="Image size")
                    norm_dd  = gr.Dropdown(choices=["dataset","imagenet"], value="dataset", label="Normalization")
                    topk_in  = gr.Slider(1, TOPK_MAX, step=1, value=5, label="Top-K")
                    btn      = gr.Button("Predict", variant="primary")

            table_out = gr.Dataframe(
                headers=["rank", "label", "prob"],
                datatype=["number", "str", "number"],
                row_count=(0, "dynamic"),
                col_count=(3, "fixed"),
                label="Top-K table",
                interactive=False,
)
            plot_out  = gr.Plot(label="Top-K bar chart")
            info_out  = gr.Markdown("")

            def _auto_size(m): return DEFAULTS.get(m, {"img_size":224})["img_size"]
            model_dd.change(fn=_auto_size, inputs=model_dd, outputs=size_in)
            btn.click(run_inference, inputs=[img_in, model_dd, size_in, norm_dd, topk_in],
                      outputs=[table_out, plot_out, info_out])

        with gr.Tab("About"):
            gr.Markdown(
                "- Models: **EfficientNet-B3** & **ResNet-50** (optimized fine-tuned checkpoints).\n"
                "- Preprocess: center-crop → resize; normalization = dataset stats (if `stats.json`) else ImageNet.\n"
                "- Class names: from `class_names.txt` or auto-generated from `train.txt`/`test.txt`.\n"
                "- Inference: AMP on GPU Spaces, CPU on CPU Spaces."
            )
    return demo

if __name__ == "__main__":
    build_ui().launch()
