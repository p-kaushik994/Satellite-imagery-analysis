"""Train ResNet18 on EuroSAT with a reproducible split, then save weights, metrics,
confusion matrix and Grad-CAM figures.   Usage:  python train.py --epochs 5"""
import argparse, json
import numpy as np, torch, torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import random_split, DataLoader
from torchvision import datasets
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import core

ap = argparse.ArgumentParser()
ap.add_argument("--epochs", type=int, default=5)
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()
dev = core.DEVICE
print("Using:", dev)

data = datasets.EuroSAT("data", download=True, transform=core.TF)
raw = datasets.EuroSAT("data", transform=None)  # same order, untransformed PIL images
n = len(data)
n_tr, n_val = int(0.7 * n), int(0.15 * n)
train, val, test = random_split(data, [n_tr, n_val, n - n_tr - n_val],
                                generator=torch.Generator().manual_seed(args.seed))
json.dump({"test": list(test.indices), "val": list(val.indices), "train": list(train.indices)},
          open("split.json", "w"))
print(f"train {len(train)}  val {len(val)}  test {len(test)}  classes {data.classes}")
assert list(data.classes) == core.CLASSES

tl = DataLoader(train, batch_size=args.batch, shuffle=True)
vl = DataLoader(val, batch_size=args.batch)
sl = DataLoader(test, batch_size=args.batch)

model = core.build_model(pretrained=True).to(dev)
crit = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr=args.lr)

def evaluate(loader):
    model.eval()
    ok = tot = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(dev), y.to(dev)
            ok += (model(x).argmax(1) == y).sum().item()
            tot += y.size(0)
    return ok / tot

best = 0
for ep in range(args.epochs):
    model.train()
    run = 0
    for x, y in tl:
        x, y = x.to(dev), y.to(dev)
        opt.zero_grad()
        loss = crit(model(x), y)
        loss.backward()
        opt.step()
        run += loss.item()
    va = evaluate(vl)
    print(f"Epoch {ep+1}/{args.epochs}: loss {run/len(tl):.3f}  val acc {va:.3f}", flush=True)
    if va > best:
        best = va
        torch.save(model.state_dict(), "eurosat_resnet18.pth")

# evaluate the best checkpoint on the untouched test split
model = core.load_model("eurosat_resnet18.pth")
preds, labels = [], []
with torch.no_grad():
    for x, y in sl:
        preds += model(x.to(dev)).argmax(1).cpu().tolist()
        labels += y.tolist()
acc = float(np.mean(np.array(preds) == np.array(labels)))
print("\nTEST ACCURACY:", round(acc, 4))
rep = classification_report(labels, preds, target_names=core.CLASSES, digits=3, output_dict=True)
print(classification_report(labels, preds, target_names=core.CLASSES, digits=3))
json.dump({"test_accuracy": acc, "best_val_accuracy": best, "epochs": args.epochs,
           "per_class": rep}, open("metrics.json", "w"), indent=2)

cm = confusion_matrix(labels, preds)
fig, ax = plt.subplots(figsize=(9, 9))
ConfusionMatrixDisplay(cm, display_labels=core.CLASSES).plot(ax=ax, xticks_rotation=45, colorbar=False)
plt.tight_layout(); plt.savefig("confusion_matrix.png", dpi=150); plt.close()

# Grad-CAM: 4 correct + up to 4 misclassified test tiles
rng = np.random.default_rng(0)
wrong = [i for i, (p, l) in enumerate(zip(preds, labels)) if p != l]
right = [i for i, (p, l) in enumerate(zip(preds, labels)) if p == l]
pick = list(rng.choice(right, 4, replace=False)) + list(rng.choice(wrong, min(4, len(wrong)), replace=False))
fig, axes = plt.subplots(2, len(pick), figsize=(2.6 * len(pick), 5.4))
for k, j in enumerate(pick):
    img, y = raw[test.indices[j]]
    cam, c, pr = core.gradcam(model, img)
    big = img.resize((128, 128))
    axes[0, k].imshow(big); axes[0, k].set_title(f"True: {core.CLASSES[y]}", fontsize=8)
    axes[1, k].imshow(big); axes[1, k].imshow(cam, cmap="jet", alpha=0.45)
    axes[1, k].set_title(f"Pred: {core.CLASSES[c]} ({pr:.0%})", fontsize=8,
                         color="green" if c == y else "red")
    axes[0, k].axis("off"); axes[1, k].axis("off")
plt.tight_layout(); plt.savefig("gradcam.png", dpi=150); plt.close()
print("Saved: eurosat_resnet18.pth, metrics.json, confusion_matrix.png, gradcam.png, split.json")
