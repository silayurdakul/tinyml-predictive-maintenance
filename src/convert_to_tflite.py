import os
from pathlib import Path
import numpy as np
import subprocess
import sys
 
PROJECT_ROOT = Path(__file__).resolve().parent.parent

PREPROCESSED_DIR = PROJECT_ROOT / "preprocessed"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

ONNX_PATH = OUTPUT_DIR / "tinyml_model.onnx"
TF_DIR = OUTPUT_DIR / "tinyml_tf"
TFLITE_PATH = OUTPUT_DIR / "tinyml_model.tflite"
HEADER_PATH = OUTPUT_DIR / "tinyml_model.h"
 
# ─── ADIM 1: ONNX → TF ─────────────────────────────────────────────────────────
print("ADIM 1: ONNX → TensorFlow SavedModel...")
import onnx2tf
onnx2tf.convert(
    input_onnx_file_path=ONNX_PATH,
    output_folder_path=TF_DIR,
    non_verbose=True
)
print("✅ TF SavedModel oluşturuldu")
 
# ─── ADIM 2: TF → TFLite ───────────────────────────────────────────────────────
print("\nADIM 2: TensorFlow → TFLite...")
import tensorflow as tf
 
converter = tf.lite.TFLiteConverter.from_saved_model(TF_DIR)
 
# INT8 Quantization — boyutu daha da küçültür
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type  = tf.int8
converter.inference_output_type = tf.int8
 
# Kalibrasyon verisi
specs_mm = np.load(
    PREPROCESSED_DIR / "specs.npy",
    mmap_mode="r"
)
 
def representative_dataset():
    indices = np.random.choice(len(specs_mm), 200, replace=False)
    for i in indices:
        spec = specs_mm[i].astype(np.float32)
        # Resize 52x65 → 32x32
        import torch
        t = torch.from_numpy(spec).unsqueeze(0)
        t = torch.nn.functional.interpolate(t, size=(32, 32), mode="bilinear", align_corners=False)
        yield [t.numpy()]
 
converter.representative_dataset = representative_dataset
 
tflite_model = converter.convert()
 
with open(TFLITE_PATH, "wb") as f:
    f.write(tflite_model)
 
size_kb = os.path.getsize(TFLITE_PATH) / 1024
print(f"✅ TFLite model kaydedildi: {TFLITE_PATH}")
print(f"   Boyut: {size_kb:.1f} KB")
 
# ─── ADIM 3: C Header Dosyası ──────────────────────────────────────────────────
print("\nADIM 3: Arduino için C header dosyası oluşturuluyor...")
 
with open(TFLITE_PATH, "rb") as f:
    model_data = f.read()
 
with open(HEADER_PATH, "w") as f:
    f.write("// TinyCNN TFLite Modeli — Arduino/ESP32 için\n")
    f.write("// Otomatik oluşturuldu\n\n")
    f.write("#pragma once\n\n")
    f.write(f"const unsigned int tinyml_model_len = {len(model_data)};\n\n")
    f.write("const unsigned char tinyml_model[] = {\n  ")
    for i, byte in enumerate(model_data):
        f.write(f"0x{byte:02x}")
        if i < len(model_data) - 1:
            f.write(", ")
        if (i + 1) % 12 == 0:
            f.write("\n  ")
    f.write("\n};\n")
 
print(f"✅ Header dosyası kaydedildi: {HEADER_PATH}")
print(f"   Boyut: {os.path.getsize(HEADER_PATH)/1024:.1f} KB")
 
print(f"""
╔══════════════════════════════════════════╗
║           DÖNÜŞÜM TAMAMLANDI            ║
╠══════════════════════════════════════════╣
║  TFLite model : {size_kb:>6.1f} KB               ║
║  Header dosya : tinyml_model.h          ║
║  Sonraki adım : ESP32'ye yükle          ║
╚══════════════════════════════════════════╝
""")