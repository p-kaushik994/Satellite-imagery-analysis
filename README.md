# Satellite Image Analyzer + Mission Planner

ResNet18 (EuroSAT, transfer learning) + Grad-CAM + change detection + greedy mission planner, in one Streamlit app.

| File | What it does |
|---|---|
| `train.py` | Trains the model with a **seeded, reproducible split**; saves weights, `metrics.json`, `confusion_matrix.png`, `gradcam.png` |
| `make_samples.py` | Builds demo tiles (`Class_lat_lon.png`) and a before/after scene with known changes |
| `core.py` | Model, Grad-CAM, cloud estimate, change detection, planner |
| `app.py` | Streamlit app: Analyze / Change detection / Mission planner |
| `REPORT.md` | Report + slides structure, demo script, likely faculty questions |

## Run on your Windows laptop

Python 3.14 is very new and PyTorch may not have wheels for it. If `pip install` fails on torch, install **Python 3.12** from python.org (tick "Add to PATH") and use it below.

```powershell
cd sat_project
py -3.12 -m venv .venv          # or: python -m venv .venv
.\.venv\Scripts\Activate.ps1    # if blocked: Set-ExecutionPolicy -Scope Process Bypass
pip install -r requirements.txt
python train.py --epochs 5      # CPU: slower (maybe 10-30 min/epoch); use Colab for training if so
python make_samples.py
python -m streamlit run app.py
```

Already have `eurosat_resnet18.pth` from Colab? Copy it into this folder and skip `train.py`.
`make_samples.py` then picks random tiles from the whole dataset, so some may have been in
training. For the report, use the **96.07%** from your Colab run, or retrain once with `train.py`
for a clean, reproducible test accuracy.

## Train on Colab (GPU, fast)

1. Runtime > Change runtime type > T4 GPU.
2. Upload `core.py`, `train.py`, `make_samples.py` (folder icon on the left).
3. New cells:
   ```
   !python train.py --epochs 5
   !python make_samples.py
   ```
4. Download `eurosat_resnet18.pth`, `samples/` (zip it with `!zip -r samples.zip samples`),
   `metrics.json`, `confusion_matrix.png`, `gradcam.png` and put them in this folder.
   Colab deletes files when the session ends, so download right away.

## Demo script (5 min)
1. **Analyze**: upload 2-3 files from `samples/` (a River, a Forest). Show prediction + heatmap.
2. **Change detection**: upload `scene_before.png` and `scene_after.png`. Show red-boxed cells. Move the grid/threshold sliders.
3. **Mission planner**: upload several `samples/*.png` (they carry coordinates in the filename). Move the time budget, set River priority to 0, show the plan change. Point to "vs random plan".
4. Show `confusion_matrix.png` and `gradcam.png` (including a misclassified example).

## What is real vs simulated (say this to faculty)
- Real: classifier, Grad-CAM, change detection, planner algorithm, pixel-based cloud estimate.
- Simulated: locations unless given in the filename; the "simulated" background targets. The table's `src` column labels every row.
- Cloud estimate is a brightness heuristic, not a trained detector; EuroSAT tiles are mostly cloud-free, so values are near 0.
- EuroSAT tiles are 64x64 px single patches; the change-detection grid cuts a scene into such patches.
