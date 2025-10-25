Bird Species Classification (200 classes)

ResNet-50 & EfficientNet-B3 — Baseline vs Optimized + Gradio Demo

Overview

This project trains and evaluates two CNN backbones (ResNet-50, EfficientNet-B3) on a 200-class bird dataset. Each backbone is run in a baseline configuration and an optimized configuration (label smoothing, warm-up + cosine LR; EfficientNet head dropout). The repo produces:

Per-epoch logs (training_log.csv) and plots (accuracy, loss, learning rate)

Saved checkpoints (best validation Top-1)

Test metrics (Top-1, Top-5, macro) and per-class accuracy CSVs

Comparison charts (baseline vs optimized; model vs model)

A Gradio web app for interactive inference (local or Hugging Face Space)

Requirements

Python 3.9–3.11

CUDA-capable GPU (8 GB VRAM works), or CPU (slower)

Recommended: Windows/PowerShell (commands below). Linux/macOS OK.

Install:

# from the 'program' folder
python -m pip install -r requirements.txt

Data setup

Set your data root (PowerShell):

$env:DATA_ROOT = "C:\Users\User\Desktop\COS30082\Assignment\program\data"
echo $env:DATA_ROOT


(Optional) compute dataset mean/std (saves data\stats.json):

python src\tools\compute_stats.py --data_root "$env:DATA_ROOT" --subset Train --out data\stats.json


Note (PowerShell quoting): always wrap $env:DATA_ROOT in quotes when passing to --data_root.

Train (baseline)
# ResNet-50 (224)
python model\resnet50.py --data_root "$env:DATA_ROOT" --variant baseline --run_name auto

# EfficientNet-B3 (300)
python model\efficientnet_b3.py --data_root "$env:DATA_ROOT" --variant baseline --run_name auto


What you get (per run):

outputs/<model>/baseline/<ordinal>/
  training_log.csv
  accuracy_curve.png
  loss_curve.png
  learning_rate_curve.png
  summary.txt
  checkpoints/best.pt

Test (baseline)

Replace <N> with your run folder (e.g., 1st, 2nd, 3rd):

# ResNet-50
python -m src.evaluate_test --data_root "$env:DATA_ROOT" --model resnet50 ^
  --ckpt outputs\resnet50\baseline\<N>\checkpoints\best.pt ^
  --img_size 224 --out_dir outputs\resnet50\baseline\<N>\test --tta --batch 8 --amp

# EfficientNet-B3
python -m src.evaluate_test --data_root "$env:DATA_ROOT" --model efficientnet_b3 ^
  --ckpt outputs\efficientnet_b3\baseline\<N>\checkpoints\best.pt ^
  --img_size 300 --out_dir outputs\efficientnet_b3\baseline\<N>\test --batch 8 --amp


Artifacts:

<...>\test\
  metrics.json        # {"top1": ..., "top5": ..., "macro": ...}
  per_class_accuracy.csv
  per_class_accuracy_bar.png

Train (optimized)
# ResNet-50 (opt): label smoothing, warm-up + cosine
python model\resnet50.py --data_root "$env:DATA_ROOT" --variant opt --run_name auto

# EfficientNet-B3 (opt): + classifier head dropout
python model\efficientnet_b3.py --data_root "$env:DATA_ROOT" --variant opt --run_name auto

Test (optimized)
# ResNet-50 (opt)
python -m src.evaluate_test --data_root "$env:DATA_ROOT" --model resnet50 ^
  --ckpt outputs\resnet50\optimized\<N>\checkpoints\best.pt ^
  --img_size 256 --out_dir outputs\resnet50\optimized\<N>\test --tta --batch 8 --amp

# EfficientNet-B3 (opt)
python -m src.evaluate_test --data_root "$env:DATA_ROOT" --model efficientnet_b3 ^
  --ckpt outputs\efficientnet_b3\optimized\<N>\checkpoints\best.pt ^
  --img_size 320 --out_dir outputs\efficientnet_b3\optimized\<N>\test --batch 8 --amp

Compare models

Baseline compare

python src\tools\compare_models.py --mode baseline `
  --resnet_test outputs\resnet50\baseline\<RN_RUN>\test `
  --effnet_test outputs\efficientnet_b3\baseline\<B3_RUN>\test


Optimized compare

python src\tools\compare_models.py --mode optimized `
  --resnet_test outputs\resnet50\optimized\<RN_RUN>\test `
  --effnet_test outputs\efficientnet_b3\optimized\<B3_RUN>\test


Outputs:

outputs/compare/<mode>/
  summary.csv
  top1_bar.png
  macro_acc_bar.png

Plot training curves (accuracy/loss/LR)
python src\tools\plot_training_curves.py --run_dir outputs\efficientnet_b3\baseline\<N>
# (use any run dir; creates/overwrites accuracy_curve.png, loss_curve.png, learning_rate_curve.png)

Per-class bar charts

Single model, all 200 classes (sorted ascending):

python src\tools\plot_per_class_bar.py `
  --csv outputs\resnet50\optimized\<N>\test\per_class_accuracy.csv `
  --title "ResNet-50 (Optimized) — All 200 Classes" `
  --sort asc --topk 200 --out outputs\compare\optimized\rn50_opt_all200.png


Two-model side-by-side (hardest 100 classes):

python src\tools\plot_per_class_bar.py `
  --csv outputs\resnet50\optimized\<RN_RUN>\test\per_class_accuracy.csv `
       outputs\efficientnet_b3\optimized\<B3_RUN>\test\per_class_accuracy.csv `
  --labels "ResNet-50 (opt)" "EffNet-B3 (opt)" `
  --title "Worst 100 Classes — Optimized Models" `
  --sort asc --topk 100 --out outputs\compare\optimized\per_class_worst100.png


(Optional) class names:

# create a names file: data\class_names.txt
0..199 | ForEach-Object { "Class $_" } | Set-Content -Encoding utf8 data\class_names.txt
# then add: --names data\class_names.txt

Gradio app (local)
python app\app.py
# open the local URL printed by Gradio


Features:

Choose model (ResNet-50 / EffNet-B3), choose recipe (baseline / optimized)

Upload an image → Top-K predictions table + probability bar chart

Uses the same preprocessing & normalization as training

Deploy on Hugging Face Spaces (brief)

Create a new Space → “Gradio” → Public/Private.

Push these files at minimum:

app/app.py, requirements.txt

model_shim.py, src/data/dataset.py (if your app loads class names)

checkpoints/ (or download from a release link at runtime)

Any helper code your app.py imports

In app.py, ensure imports use relative modules available in the Space (e.g., from src.data.dataset import ...) and checkpoints are loaded from ./checkpoints/....

Set Hardware to include GPU if available; otherwise CPU is fine but slower.

Tips & Troubleshooting

PowerShell: --data_root: expected one argument
→ You didn’t quote the env variable. Use --data_root "$env:DATA_ROOT".

Pylance “Import could not be resolved (src.*)”
→ Open VS Code at the project root and add it to Python path:
Settings → Python › Analysis: Auto Import Completions on, or run from the root so src is importable. Alternatively:

$env:PYTHONPATH = "$PWD"


CUDA OOM (out of memory)

Reduce --batch, reduce --img_size (e.g., 224 instead of 256/320).

Keep --amp on.

Close other GPU apps; clear cache between runs: torch.cuda.empty_cache() (already handled on epoch boundaries in most cases).

Train accuracy shows n/a on optimized runs
→ MixUp/CutMix style augmentation can make “hard” epochwise accuracy undefined. The code already suppresses train accuracy printing in that case. Validation metrics are correct.

Reproducibility

Seeds are fixed for model/init and dataloaders where supported.

Best-model selection is strictly by validation Top-1.

All hyperparameters used in your latest runs are baked into each model script and can be adjusted in-code.

License

Academic/educational use. Replace with your course’s policy if needed.

Acknowledgments

PyTorch & timm

Gradio

Course staff & dataset maintainers
