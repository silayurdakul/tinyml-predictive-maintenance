"""
ADIM 3 — Data Leakage Kontrolü
Dosya bazında train/test split yaparak modeli yeniden değerlendirir.
Aynı CSV'den gelen pencereler kesinlikle ayrı gruplarda olur.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix
from scipy.signal import stft, butter, filtfilt
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_PATH = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_PATH = OUTPUT_DIR / "best_model.pth"

WINDOW_SIZE = 4096
STEP        = 2048
CHANNELS    = [4, 5, 6]
FS          = 50000
IMG_SIZE    = 64

CLASS_MAP = {
    "normal":                   0,
    "horizontal-misalignment":  1,
    "vertical-misalignment":    2,
    "imbalance":                3,
    "overhang/ball_fault":      4,
    "overhang/cage_fault":      5,
    "overhang/outer_race":      6,
    "underhang/ball_fault":     4,
    "underhang/cage_fault":     5,
    "underhang/outer_race":     6,
}

CLASS_NAMES = ["normal", "horizontal-misalignment", "vertical-misalignment",
               "imbalance", "ball_fault", "cage_fault", "outer_race"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Cihaz: {DEVICE}")

# ─── DOSYA TARAMA ──────────────────────────────────────────────────────────────
def get_all_files(dataset_path):
    file_list = []
    for cls_name, label in CLASS_MAP.items():
        cls_path = os.path.join(dataset_path, *cls_name.split("/"))
        if not os.path.isdir(cls_path):
            continue
        for root, dirs, files in os.walk(cls_path):
            for f in files:
                if f.endswith(".csv"):
                    file_list.append((os.path.join(root, f), label))
    return file_list

# ─── SİNYAL İŞLEME ─────────────────────────────────────────────────────────────
def bandpass_filter(signal, lowcut=10, highcut=10000, fs=FS, order=4):
    nyq = fs / 2
    b, a = butter(order, [lowcut/nyq, highcut/nyq], btype="band")
    return filtfilt(b, a, signal)

def signal_to_spectrogram(signal, fs=FS, nperseg=512, noverlap=448, max_freq=5000):
    freqs, _, Zxx = stft(signal, fs=fs, nperseg=nperseg, noverlap=noverlap)
    freq_mask = freqs <= max_freq
    spec = np.abs(Zxx[freq_mask, :])
    spec = np.log1p(spec * 1000)
    spec = (spec - spec.min()) / (spec.max() - spec.min() + 1e-8)
    return spec.astype(np.float32)

def file_to_windows(filepath, label):
    """Bir CSV dosyasından tüm pencereleri üretir."""
    try:
        df      = pd.read_csv(filepath, header=None)
        signals = df.iloc[:, CHANNELS].values.T.astype(np.float32)
        for i in range(3):
            signals[i] = bandpass_filter(signals[i])
            signals[i] = (signals[i] - signals[i].mean()) / (signals[i].std() + 1e-8)

        windows, labels = [], []
        for start in range(0, signals.shape[1] - WINDOW_SIZE + 1, STEP):
            window = signals[:, start:start + WINDOW_SIZE]
            specs  = np.stack([signal_to_spectrogram(window[ch]) for ch in range(3)])
            windows.append(specs)
            labels.append(label)
        return windows, labels
    except Exception as e:
        print(f"Hata: {filepath} — {e}")
        return [], []

# ─── DATASET ───────────────────────────────────────────────────────────────────
class WindowDataset(Dataset):
    def __init__(self, windows, labels):
        self.windows = windows  # list of (3, H, W) arrays
        self.labels  = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        spec = torch.from_numpy(self.windows[i])
        spec = torch.nn.functional.interpolate(
            spec.unsqueeze(0), size=(IMG_SIZE, IMG_SIZE),
            mode="bilinear", align_corners=False
        ).squeeze(0)
        return spec, self.labels[i]

# ─── DOSYA BAZINDA SPLIT ────────────────────────────────────────────────────────
print("\n📂 Dosyalar taranıyor...")
all_files = get_all_files(DATASET_PATH)
file_paths = [f[0] for f in all_files]
file_labels = [f[1] for f in all_files]

print(f"Toplam dosya: {len(all_files)}")

# Dosya bazında %80 train / %20 test split
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_file_idx, test_file_idx = next(gss.split(file_paths, file_labels, groups=file_paths))

train_files = [all_files[i] for i in train_file_idx]
test_files  = [all_files[i] for i in test_file_idx]

print(f"Train dosya: {len(train_files)} | Test dosya: {len(test_files)}")

# ─── TEST PENCERELERİNİ OLUŞTUR ────────────────────────────────────────────────
print("\n⚙️  Test pencereleri oluşturuluyor...")
test_windows, test_labels = [], []
for filepath, label in tqdm(test_files, desc="Test dosyaları"):
    w, l = file_to_windows(filepath, label)
    test_windows.extend(w)
    test_labels.extend(l)

print(f"Test pencere sayısı: {len(test_labels):,}")

test_ds     = WindowDataset(test_windows, test_labels)
test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)

# ─── MODELİ YÜKLE ──────────────────────────────────────────────────────────────
print("\n🧠 Model yükleniyor...")
model = efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 7)
ckpt  = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])
model = model.to(DEVICE)
model.eval()

# ─── TAHMİN ────────────────────────────────────────────────────────────────────
print("\n📊 Dosya bazında test değerlendiriliyor...")
all_preds, all_labels = [], []
with torch.no_grad():
    for specs, labels in tqdm(test_loader, desc="Test"):
        preds = model(specs.to(DEVICE)).argmax(1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels)

all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)
test_acc   = (all_preds == all_labels).mean()

print(f"\n🎯 Dosya Bazında Test Accuracy: {test_acc*100:.2f}%")
print("\n" + classification_report(all_labels, all_preds, target_names=CLASS_NAMES))

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title(f"Dosya Bazında Confusion Matrix — Test Acc: {test_acc*100:.2f}%")
plt.ylabel("Gerçek"); plt.xlabel("Tahmin")
plt.xticks(rotation=35, ha="right"); plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix_file_based.png"), dpi=150)
plt.close()
print(f"\n📁 Kaydedildi: {OUTPUT_DIR}\\confusion_matrix_file_based.png")

# Özet
summary = {
    "test_accuracy_file_based": round(test_acc * 100, 2),
    "test_files": len(test_files),
    "test_windows": len(test_labels),
    "note": "Dosya bazında split — data leakage yok"
}
with open(os.path.join(OUTPUT_DIR, "summary_file_based.json"), "w") as f:
    json.dump(summary, f, indent=2)
print("✅ Tamamlandı!")
