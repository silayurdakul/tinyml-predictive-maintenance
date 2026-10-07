import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
 
PROJECT_ROOT = Path(__file__).resolve().parent.parent

PREPROCESSED_DIR = PROJECT_ROOT / "preprocessed"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
 
CLASS_NAMES = ["normal", "horizontal-misalignment", "vertical-misalignment",
               "imbalance", "ball_fault", "cage_fault", "outer_race"]
N_CLASSES   = 7
IMG_SIZE    = 32      # Küçük boyut — ESP32 için
BATCH_SIZE  = 128
LR          = 1e-3
MAX_EPOCHS  = 30
PATIENCE    = 5
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Cihaz: {DEVICE}")
 
# ─── KÜÇÜK CNN MODELİ ──────────────────────────────────────────────────────────
class TinyCNN(nn.Module):
    def __init__(self, n_classes=7):
        super().__init__()
        self.features = nn.Sequential(
            # Blok 1
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d(2),                # 16x16
            # Blok 2
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),                # 8x8
            # Blok 3
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),                # 4x4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, n_classes)
        )
 
    def forward(self, x):
        return self.classifier(self.features(x))
 
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
        idx  = self.indices[i]
        spec = torch.from_numpy(self.specs_mm[idx].copy())
        spec = torch.nn.functional.interpolate(
            spec.unsqueeze(0), size=(IMG_SIZE, IMG_SIZE),
            mode="bilinear", align_corners=False
        ).squeeze(0)
        if self.augment and torch.rand(1) < 0.5:
            spec = spec + torch.randn_like(spec) * 0.02
        return spec, int(self.labels_mm[idx])
 
# ─── VERİ YÜKLEME ──────────────────────────────────────────────────────────────
print("\n📂 Veriler yükleniyor...")
specs_mm  = np.load(os.path.join(PREPROCESSED_DIR, "specs.npy"),  mmap_mode="r")
labels_mm = np.load(os.path.join(PREPROCESSED_DIR, "labels.npy"), mmap_mode="r")
print(f"specs: {specs_mm.shape} | labels: {labels_mm.shape}")
 
all_idx = np.arange(len(labels_mm))
train_idx, test_idx = train_test_split(all_idx, test_size=0.20,
                                        stratify=labels_mm, random_state=42)
train_idx, val_idx  = train_test_split(train_idx, test_size=0.15,
                                        stratify=labels_mm[train_idx], random_state=42)
print(f"Train: {len(train_idx):,} | Val: {len(val_idx):,} | Test: {len(test_idx):,}")
 
train_labels  = labels_mm[train_idx]
class_counts  = np.bincount(train_labels, minlength=N_CLASSES).astype(float)
class_weights = 1.0 / (class_counts + 1e-6)
sample_weights = class_weights[train_labels]
sampler = WeightedRandomSampler(
    weights=torch.DoubleTensor(sample_weights),
    num_samples=len(train_idx), replacement=True
)
 
train_ds = SpectrogramDataset(train_idx, specs_mm, labels_mm, augment=True)
val_ds   = SpectrogramDataset(val_idx,   specs_mm, labels_mm, augment=False)
test_ds  = SpectrogramDataset(test_idx,  specs_mm, labels_mm, augment=False)
 
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,   num_workers=0)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,   num_workers=0)
 
# ─── MODEL ─────────────────────────────────────────────────────────────────────
model = TinyCNN(N_CLASSES).to(DEVICE)
 
# Model boyutu
total_params = sum(p.numel() for p in model.parameters())
print(f"\n🧠 TinyCNN parametresi: {total_params:,} (~{total_params*4/1024:.1f} KB)")
 
cw        = torch.tensor(class_weights / class_weights.sum() * N_CLASSES,
                          dtype=torch.float32).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=cw)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
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
                loss.backward(); optimizer.step()
            total_loss += loss.item() * len(labels)
            correct    += (out.argmax(1) == labels).sum().item()
            total      += len(labels)
    return total_loss / total, correct / total
 
print("\n🚀 TinyCNN eğitimi başlıyor...\n")
header = f"{'Epoch':>5} | {'Train Loss':>10} {'Train Acc':>9} | {'Val Loss':>8} {'Val Acc':>8} | Not"
print(header); print("─" * len(header))
 
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
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "tinyml_model.pth"))
        note = "✅ kaydedildi"
    else:
        patience_cnt += 1
        if patience_cnt >= PATIENCE:
            print(f"\n⏹️  Early stopping"); break
 
    print(f"{epoch:>5} | {tr_loss:>10.4f} {tr_acc*100:>8.2f}% | "
          f"{vl_loss:>8.4f} {vl_acc*100:>7.2f}% | {note}")
 
# ─── TEST ──────────────────────────────────────────────────────────────────────
print("\n📊 Test değerlendiriliyor...")
model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, "tinyml_model.pth"), weights_only=True))
model.eval()
 
all_preds, all_labels = [], []
with torch.no_grad():
    for specs, labels in tqdm(test_loader, desc="Test"):
        preds = model(specs.to(DEVICE)).argmax(1).cpu().numpy()
        all_preds.extend(preds); all_labels.extend(labels.numpy())
 
all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)
test_acc   = (all_preds == all_labels).mean()
print(f"\n🎯 TinyCNN Test Accuracy: {test_acc*100:.2f}%")
print("\n" + classification_report(all_labels, all_preds, target_names=CLASS_NAMES))
 
# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title(f"TinyCNN Confusion Matrix — Test Acc: {test_acc*100:.2f}%")
plt.ylabel("Gerçek"); plt.xlabel("Tahmin")
plt.xticks(rotation=35, ha="right"); plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix_tinyml.png"), dpi=150)
plt.close()
 
# ─── TFLite'A ÇEVİR ────────────────────────────────────────────────────────────
print("\n🔄 TensorFlow Lite'a çevriliyor...")
import tensorflow as tf
 
# Dummy input ile ONNX export
dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
model.cpu()
 
torch.onnx.export(
    model, dummy,
    os.path.join(OUTPUT_DIR, "tinyml_model.onnx"),
    input_names=["input"], output_names=["output"],
    opset_version=11
)
print("ONNX kaydedildi.")
 
# TFLite converter
converter = tf.lite.TFLiteConverter.from_saved_model
print("\n⚠️  TFLite dönüşümü için ONNX → TF adımı ayrı script'te yapılacak.")
print("Şimdi model boyutlarını karşılaştıralım:")
print(f"\n  EfficientNet-B0 : ~20MB")
print(f"  TinyCNN (PyTorch): ~{total_params*4/1024/1024:.2f}MB")
print(f"  TinyCNN Test Acc : {test_acc*100:.2f}%")
 
summary = {
    "model": "TinyCNN",
    "test_accuracy": round(test_acc * 100, 2),
    "parameters": total_params,
    "size_kb": round(total_params * 4 / 1024, 1),
    "img_size": IMG_SIZE,
    "comparison": {
        "EfficientNet-B0": {"accuracy": 99.99, "size_mb": 20},
        "TinyCNN": {"accuracy": round(test_acc * 100, 2), "size_kb": round(total_params * 4 / 1024, 1)}
    }
}
with open(os.path.join(OUTPUT_DIR, "tinyml_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
 
print(f"\n✅ Tamamlandı! Çıktılar: {OUTPUT_DIR}")
print("   tinyml_model.pth | tinyml_model.onnx | confusion_matrix_tinyml.png | tinyml_summary.json")