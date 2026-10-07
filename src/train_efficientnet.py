"""
ADIM 2 — Model Eğitimi
Girdi: preprocessed/specs.npy, preprocessed/labels.npy
Model: EfficientNet-B0 (7 sınıf)
Çıktı: outputs/best_model.pth, confusion_matrix.png, training_curves.png, classification_report.txt
"""

import os
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PREPROCESSED_DIR = PROJECT_ROOT / "preprocessed"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["normal", "horizontal-misalignment", "vertical-misalignment",
               "imbalance", "ball_fault", "cage_fault", "outer_race"]
N_CLASSES   = 7

BATCH_SIZE  = 64
LR          = 1e-4
MAX_EPOCHS  = 50
PATIENCE    = 8
IMG_SIZE    = 64
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Cihaz: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ─── VERİ SETİ ─────────────────────────────────────────────────────────────────
class SpectrogramDataset(Dataset):
    def __init__(self, indices, specs_mm, labels_mm, augment=False):
        self.indices   = indices
        self.specs_mm  = specs_mm
        self.labels_mm = labels_mm
        self.augment   = augment

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx   = self.indices[i]
        spec  = torch.from_numpy(self.specs_mm[idx].copy())  # (3, H, W)

        # Resize → 64×64
        spec = torch.nn.functional.interpolate(
            spec.unsqueeze(0), size=(IMG_SIZE, IMG_SIZE),
            mode="bilinear", align_corners=False
        ).squeeze(0)

        if self.augment:
            if torch.rand(1) < 0.5:
                spec = spec + torch.randn_like(spec) * 0.02
            if torch.rand(1) < 0.3:
                shift = torch.randint(-4, 4, (1,)).item()
                spec  = torch.roll(spec, shift, dims=2)

        return spec, int(self.labels_mm[idx])

# ─── VERİ YÜKLEME ──────────────────────────────────────────────────────────────
print("\n📂 Veriler yükleniyor...")
specs_mm  = np.load(os.path.join(PREPROCESSED_DIR, "specs.npy"),  mmap_mode="r")
labels_mm = np.load(os.path.join(PREPROCESSED_DIR, "labels.npy"), mmap_mode="r")
print(f"specs  : {specs_mm.shape}")
print(f"labels : {labels_mm.shape}")

unique, counts = np.unique(labels_mm, return_counts=True)
print("\nSınıf dağılımı:")
for u, c in zip(unique, counts):
    print(f"  {CLASS_NAMES[u]:<30}: {c:>8,} pencere")

# Split
all_idx = np.arange(len(labels_mm))
train_idx, test_idx = train_test_split(all_idx, test_size=0.20,
                                        stratify=labels_mm, random_state=42)
train_idx, val_idx  = train_test_split(train_idx, test_size=0.15,
                                        stratify=labels_mm[train_idx], random_state=42)
print(f"\nTrain: {len(train_idx):,} | Val: {len(val_idx):,} | Test: {len(test_idx):,}")

# WeightedRandomSampler
train_labels  = labels_mm[train_idx]
class_counts  = np.bincount(train_labels, minlength=N_CLASSES).astype(float)
class_weights = 1.0 / (class_counts + 1e-6)
sample_weights = class_weights[train_labels]
sampler = WeightedRandomSampler(
    weights=torch.DoubleTensor(sample_weights),
    num_samples=len(train_idx),
    replacement=True
)

train_ds = SpectrogramDataset(train_idx, specs_mm, labels_mm, augment=True)
val_ds   = SpectrogramDataset(val_idx,   specs_mm, labels_mm, augment=False)
test_ds  = SpectrogramDataset(test_idx,  specs_mm, labels_mm, augment=False)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                           num_workers=0, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                           num_workers=0, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                           num_workers=0, pin_memory=True)

# ─── MODEL ─────────────────────────────────────────────────────────────────────
print("\n🧠 Model kuruluyor (EfficientNet-B0, ImageNet ağırlıkları)...")
model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, N_CLASSES)
model = model.to(DEVICE)

cw        = torch.tensor(class_weights / class_weights.sum() * N_CLASSES,
                          dtype=torch.float32).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=cw)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)

# ─── EĞİTİM ────────────────────────────────────────────────────────────────────
def run_epoch(loader, train=True):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for specs, labels in loader:
            specs, labels = specs.to(DEVICE), labels.to(DEVICE)
            if train:
                optimizer.zero_grad()
            out  = model(specs)
            loss = criterion(out, labels)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(labels)
            correct    += (out.argmax(1) == labels).sum().item()
            total      += len(labels)
    return total_loss / total, correct / total

print("\n🚀 Eğitim başlıyor...\n")
header = f"{'Epoch':>5} | {'Train Loss':>10} {'Train Acc':>9} | {'Val Loss':>8} {'Val Acc':>8} | Not"
print(header)
print("─" * len(header))

best_val_acc = 0.0
patience_cnt = 0
history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

for epoch in range(1, MAX_EPOCHS + 1):
    tr_loss, tr_acc = run_epoch(train_loader, train=True)
    vl_loss, vl_acc = run_epoch(val_loader,   train=False)
    scheduler.step()

    history["train_loss"].append(tr_loss)
    history["train_acc"].append(tr_acc)
    history["val_loss"].append(vl_loss)
    history["val_acc"].append(vl_acc)

    note = ""
    if vl_acc > best_val_acc:
        best_val_acc = vl_acc
        patience_cnt = 0
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "val_acc": vl_acc,
            "class_names": CLASS_NAMES,
            "img_size": IMG_SIZE,
        }, os.path.join(OUTPUT_DIR, "best_model.pth"))
        note = "✅ kaydedildi"
    else:
        patience_cnt += 1
        if patience_cnt >= PATIENCE:
            print(f"\n⏹️  Early stopping ({PATIENCE} epoch iyileşme yok)")
            break

    lr_now = scheduler.get_last_lr()[0]
    print(f"{epoch:>5} | {tr_loss:>10.4f} {tr_acc*100:>8.2f}% | "
          f"{vl_loss:>8.4f} {vl_acc*100:>7.2f}% | {note}")

# ─── TEST ──────────────────────────────────────────────────────────────────────
print("\n📊 Test seti değerlendiriliyor...")
ckpt = torch.load(os.path.join(OUTPUT_DIR, "best_model.pth"), map_location=DEVICE)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

all_preds, all_labels = [], []
with torch.no_grad():
    for specs, labels in tqdm(test_loader, desc="Test"):
        preds = model(specs.to(DEVICE)).argmax(1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)
test_acc   = (all_preds == all_labels).mean()

print(f"\n🎯 Test Accuracy: {test_acc*100:.2f}%")
report = classification_report(all_labels, all_preds, target_names=CLASS_NAMES)
print("\n" + report)

with open(os.path.join(OUTPUT_DIR, "classification_report.txt"), "w", encoding="utf-8") as f:
    f.write(f"Test Accuracy: {test_acc*100:.2f}%\n\n{report}")

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title(f"Confusion Matrix — Test Acc: {test_acc*100:.2f}%")
plt.ylabel("Gerçek"); plt.xlabel("Tahmin")
plt.xticks(rotation=35, ha="right"); plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=150)
plt.close()

# Eğitim eğrileri
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(history["train_loss"], label="Train"); ax1.plot(history["val_loss"], label="Val")
ax1.set_title("Loss"); ax1.legend(); ax1.set_xlabel("Epoch")
ax2.plot([a*100 for a in history["train_acc"]], label="Train")
ax2.plot([a*100 for a in history["val_acc"]],   label="Val")
ax2.set_title("Accuracy (%)"); ax2.legend(); ax2.set_xlabel("Epoch")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "training_curves.png"), dpi=150)
plt.close()

# Özet JSON
summary = {
    "test_accuracy": round(test_acc * 100, 2),
    "best_val_accuracy": round(best_val_acc * 100, 2),
    "epochs_trained": len(history["train_loss"]),
    "n_classes": N_CLASSES,
    "class_names": CLASS_NAMES,
}
with open(os.path.join(OUTPUT_DIR, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print(f"\n✅ Tüm çıktılar kaydedildi: {OUTPUT_DIR}")
print("   best_model.pth | confusion_matrix.png | training_curves.png")
print("   classification_report.txt | summary.json")
