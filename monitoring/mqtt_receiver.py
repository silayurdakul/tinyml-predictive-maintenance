from pathlib import Path
import paho.mqtt.client as mqtt
import numpy as np
from scipy.signal import stft
from collections import deque
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0
import warnings
warnings.filterwarnings("ignore")

# ─── AYARLAR ───────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "outputs" / "best_model.pth"
CLASS_NAMES = ["normal", "horizontal-misalignment", "vertical-misalignment",
               "imbalance", "ball_fault", "cage_fault", "outer_race"]
WINDOW_SIZE = 512
IMG_SIZE    = 64
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── MODEL ─────────────────────────────────────────────────────────────────────
print("Model yükleniyor...")
model = efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 7)
ckpt  = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])
model.eval().to(DEVICE)
print("Model hazır!")

# ─── VERİ TAMPONU ──────────────────────────────────────────────────────────────
buffer_x = deque(maxlen=WINDOW_SIZE)
buffer_y = deque(maxlen=WINDOW_SIZE)
buffer_z = deque(maxlen=WINDOW_SIZE)

def signal_to_spec(signal):
    signal = np.array(signal, dtype=np.float32)
    signal = (signal - signal.mean()) / (signal.std() + 1e-8)
    _, _, Zxx = stft(signal, nperseg=64, noverlap=48)
    spec = np.abs(Zxx)
    spec = np.log1p(spec * 1000)
    spec = (spec - spec.min()) / (spec.max() - spec.min() + 1e-8)
    return spec.astype(np.float32)

def predict(x, y, z):
    specs = np.stack([
        signal_to_spec(x),
        signal_to_spec(y),
        signal_to_spec(z)
    ])  # (3, H, W)
    
    tensor = torch.from_numpy(specs).unsqueeze(0)  # (1, 3, H, W)
    tensor = torch.nn.functional.interpolate(
        tensor, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False
    ).to(DEVICE)
    
    with torch.no_grad():
        out   = model(tensor)
        probs = torch.softmax(out, dim=1)[0]
        pred  = probs.argmax().item()
        conf  = probs[pred].item()
    
    return CLASS_NAMES[pred], conf

# ─── MQTT ──────────────────────────────────────────────────────────────────────
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("MQTT bağlandı! Veri bekleniyor...\n")
        client.subscribe("vibration/data")
    else:
        print(f"Bağlantı hatası: {rc}")

counter = 0

def on_message(client, userdata, msg):
    global counter
    payload = msg.payload.decode()
    x, y, z = map(float, payload.split(","))
    
    buffer_x.append(x)
    buffer_y.append(y)
    buffer_z.append(z)
    
    counter += 1
    
    if len(buffer_x) == WINDOW_SIZE and counter % 50 == 0:
        sinif, guven = predict(
            list(buffer_x),
            list(buffer_y),
            list(buffer_z)
        )
        print(f"🔍 TAHMİN: {sinif:<30} (güven: {guven*100:.1f}%)")
        print(f"   X:{x:.2f} Y:{y:.2f} Z:{z:.2f}")
        print()

client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message
client.connect("localhost", 1883, 60)
client.loop_forever()