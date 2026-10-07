"""
ADIM 1 — Preprocessing
Klasör yapısını doğru okur (3 seviyeli: sınıf/arıza_tipi/şiddet/CSV)
Çıktı: preprocessed/specs.npy ve preprocessed/labels.npy
7 sınıf:
  0: normal
  1: horizontal-misalignment
  2: vertical-misalignment
  3: imbalance
  4: ball_fault   (overhang + underhang)
  5: cage_fault   (overhang + underhang)
  6: outer_race   (overhang + underhang)
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import stft, butter, filtfilt
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_PATH = PROJECT_ROOT / "data"
SAVE_DIR = PROJECT_ROOT / "preprocessed"

SAVE_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_SIZE = 4096
STEP        = 2048
CHANNELS    = [4, 5, 6]   # 3 titreşim kanalı
FS          = 50000        # 50 kHz örnekleme frekansı

# ─── SINIF HARİTASI ────────────────────────────────────────────────────────────
# Her klasör yolu → label
# overhang ve underhang: arıza tipine göre birleştiriliyor
CLASS_MAP = {
    "normal":                   0,
    "horizontal-misalignment":  1,
    "vertical-misalignment":    2,
    "imbalance":                3,
    # overhang alt tipleri
    "overhang/ball_fault":      4,
    "overhang/cage_fault":      5,
    "overhang/outer_race":      6,
    # underhang alt tipleri — aynı label
    "underhang/ball_fault":     4,
    "underhang/cage_fault":     5,
    "underhang/outer_race":     6,
}

CLASS_NAMES = {
    0: "normal",
    1: "horizontal-misalignment",
    2: "vertical-misalignment",
    3: "imbalance",
    4: "ball_fault",
    5: "cage_fault",
    6: "outer_race",
}

# ─── DOSYA TARAMA ──────────────────────────────────────────────────────────────
def get_all_files(dataset_path):
    """
    3 seviyeli yapıyı doğru tarar:
      Düz sınıflar:     dataset/normal/*.csv
                        dataset/imbalance/10g/*.csv
      Overhang/under:   dataset/overhang/ball_fault/0g/*.csv
    """
    file_list = []

    for cls_name, label in CLASS_MAP.items():
        cls_path = os.path.join(dataset_path, *cls_name.split("/"))
        if not os.path.isdir(cls_path):
            print(f"⚠️  Klasör bulunamadı: {cls_path}")
            continue

        # Bu klasörün altındaki TÜM CSV'leri bul (derinlik fark etmez)
        for root, dirs, files in os.walk(cls_path):
            for f in files:
                if f.endswith(".csv"):
                    file_list.append((os.path.join(root, f), label))

    return file_list


# ─── SİNYAL İŞLEME ─────────────────────────────────────────────────────────────
def bandpass_filter(signal, lowcut=10, highcut=10000, fs=FS, order=4):
    nyq = fs / 2
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype="band")
    return filtfilt(b, a, signal)


def signal_to_spectrogram(signal, fs=FS, nperseg=512, noverlap=448, max_freq=5000):
    freqs, _, Zxx = stft(signal, fs=fs, nperseg=nperseg, noverlap=noverlap)
    freq_mask = freqs <= max_freq
    spec = np.abs(Zxx[freq_mask, :])
    spec = np.log1p(spec * 1000)
    spec = (spec - spec.min()) / (spec.max() - spec.min() + 1e-8)
    return spec.astype(np.float32)


# ─── TARAMA ────────────────────────────────────────────────────────────────────
print("📂 Dosyalar taranıyor...")
all_files = get_all_files(DATASET_PATH)
print(f"Toplam CSV dosyası: {len(all_files)}")

# Sınıf dağılımı özeti
from collections import Counter
label_counts = Counter(label for _, label in all_files)
print("\nDosya bazında sınıf dağılımı:")
for lbl in sorted(label_counts):
    print(f"  {CLASS_NAMES[lbl]:<30}: {label_counts[lbl]:>4} dosya")

# Spektrogram boyutunu ölç
print("\n📐 Spektrogram boyutu ölçülüyor...")
sample_sig = pd.read_csv(all_files[0][0], header=None).iloc[:WINDOW_SIZE, CHANNELS[0]].values.astype(np.float32)
sample_sig = bandpass_filter(sample_sig)
sample_sig = (sample_sig - sample_sig.mean()) / (sample_sig.std() + 1e-8)
H, W = signal_to_spectrogram(sample_sig).shape

windows_per_file = (250000 - WINDOW_SIZE) // STEP + 1
N_TOTAL = len(all_files) * windows_per_file

print(f"Spektrogram boyutu : ({H}, {W})")
print(f"Dosya başına pencere: {windows_per_file}")
print(f"Toplam pencere     : {N_TOTAL:,}")
print(f"Tahmini disk alanı : {N_TOTAL * 3 * H * W * 4 / 1e9:.1f} GB")

# ─── MEMMAP OLUŞTUR ────────────────────────────────────────────────────────────
specs_path  = os.path.join(SAVE_DIR, "specs.npy")
labels_path = os.path.join(SAVE_DIR, "labels.npy")

# Eski dosyaları sil
for p in [specs_path, labels_path]:
    if os.path.exists(p):
        os.remove(p)
        print(f"🗑️  Eski dosya silindi: {p}")

print("\n⚙️  Önişleme başlıyor...")
specs_mm  = np.lib.format.open_memmap(specs_path,  mode="w+", dtype="float32", shape=(N_TOTAL, 3, H, W))
labels_mm = np.lib.format.open_memmap(labels_path, mode="w+", dtype="int64",   shape=(N_TOTAL,))

idx = 0
errors = []

for filepath, label in tqdm(all_files, desc="Dosyalar"):
    try:
        df      = pd.read_csv(filepath, header=None)
        signals = df.iloc[:, CHANNELS].values.T.astype(np.float32)  # (3, 250000)

        for i in range(3):
            signals[i] = bandpass_filter(signals[i])
            signals[i] = (signals[i] - signals[i].mean()) / (signals[i].std() + 1e-8)

        for start in range(0, signals.shape[1] - WINDOW_SIZE + 1, STEP):
            window = signals[:, start:start + WINDOW_SIZE]
            for ch in range(3):
                specs_mm[idx, ch] = signal_to_spectrogram(window[ch])
            labels_mm[idx] = label
            idx += 1

    except Exception as e:
        errors.append((filepath, str(e)))

specs_mm.flush()
labels_mm.flush()

# ─── ÖZET ──────────────────────────────────────────────────────────────────────
print(f"\n✅ Tamamlandı!")
print(f"Yazılan toplam pencere : {idx:,}")
print(f"Hatalı dosya sayısı    : {len(errors)}")
if errors:
    print("Hatalar:")
    for fp, err in errors[:5]:
        print(f"  {fp}: {err}")

# Pencere bazında sınıf dağılımı
actual_labels = labels_mm[:idx]
label_window_counts = Counter(actual_labels.tolist())
print("\nPencere bazında sınıf dağılımı:")
for lbl in sorted(label_window_counts):
    print(f"  {CLASS_NAMES[lbl]:<30}: {label_window_counts[lbl]:>8,} pencere")

print(f"\n📁 Kayıt yeri: {SAVE_DIR}")
print("   → specs.npy")
print("   → labels.npy")
print("\nSonraki adım: train_efficientnet.py")
