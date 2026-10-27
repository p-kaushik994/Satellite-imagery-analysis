"""Shared logic: model loading, classification, Grad-CAM, cloud estimate,
change detection and the mission planner. Used by train.py, make_samples.py, app.py."""
import re
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

CLASSES = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial',
           'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']
DEFAULT_PRIORITY = {"River": 1.0, "SeaLake": 0.9, "Forest": 0.8, "Industrial": 0.8,
                    "Residential": 0.6, "AnnualCrop": 0.6, "PermanentCrop": 0.5,
                    "Pasture": 0.4, "HerbaceousVegetation": 0.4, "Highway": 0.5}
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TF = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor(),
                         transforms.Normalize(MEAN, STD)])
START = (21.0, 72.0)  # where the satellite footprint starts (lat, lon)


# ---------------------------------------------------------------- model
def build_model(pretrained=False):
    w = models.ResNet18_Weights.DEFAULT if pretrained else None
    m = models.resnet18(weights=w)
    m.fc = nn.Linear(m.fc.in_features, len(CLASSES))
    return m


def load_model(path="eurosat_resnet18.pth"):
    m = build_model()
    m.load_state_dict(torch.load(path, map_location=DEVICE))
    return m.to(DEVICE).eval()


def classify(model, img):
    """PIL image -> numpy probabilities (10,)"""
    with torch.no_grad():
        x = TF(img.convert("RGB")).unsqueeze(0).to(DEVICE)
        return F.softmax(model(x), 1)[0].cpu().numpy()


def classify_batch(model, imgs):
    with torch.no_grad():
        x = torch.stack([TF(i.convert("RGB")) for i in imgs]).to(DEVICE)
        return F.softmax(model(x), 1).cpu().numpy()


# ---------------------------------------------------------------- Grad-CAM
def gradcam(model, img, class_idx=None):
    """PIL image -> (heatmap 128x128 in [0,1], class index, probability)."""
    store = {}
    layer = model.layer4[-1]

    def fwd(module, inp, out):
        store["a"] = out
        if out.requires_grad:
            out.register_hook(lambda g: store.__setitem__("g", g))

    h = layer.register_forward_hook(fwd)
    try:
        x = TF(img.convert("RGB")).unsqueeze(0).to(DEVICE)
        model.eval()
        with torch.enable_grad():
            out = model(x)
            c = int(out.argmax(1)) if class_idx is None else class_idx
            model.zero_grad()
            out[0, c].backward()
        prob = F.softmax(out, 1)[0, c].item()
    finally:
        h.remove()
    w = store["g"].mean(dim=(2, 3), keepdim=True)
    cam = F.relu((w * store["a"]).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=(128, 128), mode="bilinear", align_corners=False)
    cam = cam[0, 0].detach().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam, c, prob


# ---------------------------------------------------------------- image helpers
def est_cloud(img):
    """Crude cloud fraction: bright, low-saturation pixels. A heuristic, not a cloud detector."""
    a = np.asarray(img.convert("RGB").resize((64, 64))).astype(float) / 255
    mx, mn = a.max(2), a.min(2)
    return float(((mn > 0.65) & (mx - mn < 0.12)).mean())


def coords_from_name(name):
    """'River_19.5_74.2.png' -> (19.5, 74.2); None if the name has no coordinates."""
    m = re.search(r"_(-?\d+\.?\d*)_(-?\d+\.?\d*)\.\w+$", name)
    return (float(m.group(1)), float(m.group(2))) if m else None


# ---------------------------------------------------------------- change detection
def pixel_change(img_a, img_b, size=256, blur=5):
    """Per-band standardised difference (reduces brightness/season offsets).
    Returns a change-intensity map in [0,1], shape (size, size)."""
    def prep(im):
        a = np.asarray(im.convert("RGB").resize((size, size)), dtype=np.float32) / 255
        return (a - a.mean((0, 1))) / (a.std((0, 1)) + 1e-6)
    d = np.sqrt(((prep(img_a) - prep(img_b)) ** 2).sum(2))
    t = torch.from_numpy(d)[None, None]
    t = F.avg_pool2d(t, blur, stride=1, padding=blur // 2)[0, 0].numpy()
    return np.clip(t / (np.percentile(t, 99) + 1e-8), 0, 1)


def grid_classes(model, img, cells=4, tile=64):
    """Cut a scene into cells x cells tiles, classify each. Returns probs (cells*cells, 10)."""
    img = img.convert("RGB").resize((cells * tile, cells * tile))
    patches = [img.crop((c * tile, r * tile, (c + 1) * tile, (r + 1) * tile))
               for r in range(cells) for c in range(cells)]
    return classify_batch(model, patches)


# ---------------------------------------------------------------- mission planner
def dist_km(a, b):
    dy = (a[0] - b[0]) * 111
    dx = (a[1] - b[1]) * 111 * np.cos(np.radians((a[0] + b[0]) / 2))
    return float(np.hypot(dx, dy))


def plan(T, strategy="greedy", budget=12.0, img_min=1.0, slew=100.0, start=START, seed=0):
    """T: DataFrame with lat, lon, value. Returns (route as row positions, total value, time used)."""
    rng = np.random.default_rng(seed)
    lat, lon, val = T["lat"].to_numpy(), T["lon"].to_numpy(), T["value"].to_numpy()
    left, pos, used, route = list(range(len(T))), start, 0.0, []
    while left:
        cost = {i: img_min + dist_km(pos, (lat[i], lon[i])) / slew for i in left}
        feas = [i for i in left if used + cost[i] <= budget]
        if not feas:
            break
        i = max(feas, key=lambda j: val[j] / cost[j]) if strategy == "greedy" \
            else feas[int(rng.integers(len(feas)))]
        used += cost[i]
        pos = (lat[i], lon[i])
        route.append(i)
        left.remove(i)
    total = float(val[route].sum()) if route else 0.0
    return route, total, used


def random_baseline(T, runs=100, **kw):
    return float(np.mean([plan(T, "random", seed=s, **kw)[1] for s in range(runs)]))
