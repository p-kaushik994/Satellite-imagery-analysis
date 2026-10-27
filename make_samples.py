"""Create demo material in ./samples:
 - tiles named Class_lat_lon.png (for the Analyze and Mission Planner tabs)
 - scene_before.png / scene_after.png (4x4 grid, known changes) for the Change Detection tab
Usage: python make_samples.py"""
import os, json, numpy as np
from PIL import Image
from torchvision import datasets
import core

rng = np.random.default_rng(7)
raw = datasets.EuroSAT("data", download=True, transform=None)
idxs = json.load(open("split.json"))["test"] if os.path.exists("split.json") else range(len(raw))
by_class = {c: [] for c in core.CLASSES}
for i in idxs:
    by_class[core.CLASSES[raw.targets[i] if hasattr(raw, "targets") else raw[i][1]]].append(i)
pick = lambda c: raw[int(rng.choice(by_class[c]))][0]

os.makedirs("samples", exist_ok=True)
for c in core.CLASSES:
    for k in range(2):
        lat, lon = round(rng.uniform(15, 21), 2), round(rng.uniform(72, 78), 2)
        pick(c).resize((128, 128)).save(f"samples/{c}_{lat}_{lon}.png")

# change-detection scene: forest + crops before; some cells become Residential / SeaLake after
# unchanged cells reuse the exact same tile, so only the real changes differ
before_cls = [str(rng.choice(["Forest", "AnnualCrop"])) for _ in range(16)]
before_tiles = [pick(c) for c in before_cls]
after_tiles = list(before_tiles)
changed = rng.choice(16, 7, replace=False)
for n, cell in enumerate(changed):
    after_tiles[cell] = pick("Residential" if n < 4 else "SeaLake")
def scene(tiles):
    im = Image.new("RGB", (256, 256))
    for k, t in enumerate(tiles):
        im.paste(t, ((k % 4) * 64, (k // 4) * 64))
    return im
scene(before_tiles).save("samples/scene_before.png")
scene(after_tiles).save("samples/scene_after.png")
print("Changed cells (row-major, 0-15):", sorted(int(c) for c in changed))
print("Saved samples to ./samples")
