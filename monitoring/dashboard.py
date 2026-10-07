from pathlib import Path
from flask import Flask, render_template_string
from flask_socketio import SocketIO
import paho.mqtt.client as mqtt
import numpy as np
from scipy.signal import stft
from collections import deque
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0
import threading
import warnings
import firebase_admin
from firebase_admin import credentials, db
from datetime import datetime
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "outputs" / "best_model.pth"
FIREBASE_KEY = PROJECT_ROOT / "firebase_key.json"

FIREBASE_URL = "https://bitirmeprojesi-802a2-default-rtdb.firebaseio.com/"

CLASS_NAMES = ["normal", "horizontal-misalignment", "vertical-misalignment",
               "imbalance", "ball_fault", "cage_fault", "outer_race"]
WINDOW_SIZE = 512
IMG_SIZE    = 64
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── FİREBASE ──────────────────────────────────────────────────────────────────
print("Firebase bağlanıyor...")
cred = credentials.Certificate(FIREBASE_KEY)
firebase_admin.initialize_app(cred, {"databaseURL": FIREBASE_URL})
firebase_ref = db.reference("tahminler")
print("Firebase hazır!")

# ─── MODEL ─────────────────────────────────────────────────────────────────────
print("Model yükleniyor...")
model = efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 7)
ckpt  = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])
model.eval().to(DEVICE)
print("Model hazır!")

# ─── FLASK ─────────────────────────────────────────────────────────────────────
app  = Flask(__name__)
sock = SocketIO(app, cors_allowed_origins="*")

buffer_x = deque(maxlen=WINDOW_SIZE)
buffer_y = deque(maxlen=WINDOW_SIZE)
buffer_z = deque(maxlen=WINDOW_SIZE)
counter  = 0

HTML = """
<!DOCTYPE html>
<html lang="tr">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Rulman Arıza Tespit Sistemi</title>
  <link href="https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;600;700&display=swap" rel="stylesheet">
  <script src="https://cdn.socket.io/4.6.0/socket.io.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    :root {
      --bg:       #050a0e;
      --panel:    #0a1520;
      --border:   #1a3a4a;
      --accent:   #00d4ff;
      --accent2:  #ff6b35;
      --green:    #00ff88;
      --red:      #ff3366;
      --text:     #c8e0ea;
      --dim:      #4a7a8a;
      --mono:     'Share Tech Mono', monospace;
      --sans:     'Rajdhani', sans-serif;
    }
    * { margin: 0; padding: 0; box-sizing: border-box; }

    body {
      background: var(--bg);
      color: var(--text);
      font-family: var(--sans);
      min-height: 100vh;
      overflow-x: hidden;
    }

    /* Scanline effect */
    body::before {
      content: '';
      position: fixed;
      inset: 0;
      background: repeating-linear-gradient(
        0deg,
        transparent,
        transparent 2px,
        rgba(0,212,255,0.015) 2px,
        rgba(0,212,255,0.015) 4px
      );
      pointer-events: none;
      z-index: 1000;
    }

    header {
      border-bottom: 1px solid var(--border);
      padding: 16px 32px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      background: linear-gradient(90deg, rgba(0,212,255,0.05) 0%, transparent 100%);
    }

    .logo {
      display: flex;
      align-items: center;
      gap: 12px;
    }

    .logo-icon {
      width: 32px; height: 32px;
      border: 2px solid var(--accent);
      border-radius: 4px;
      display: flex; align-items: center; justify-content: center;
      color: var(--accent);
      font-size: 16px;
      position: relative;
    }

    .logo-icon::after {
      content: '';
      position: absolute;
      inset: 3px;
      border: 1px solid var(--accent);
      border-radius: 2px;
      opacity: 0.4;
    }

    h1 {
      font-family: var(--sans);
      font-size: 18px;
      font-weight: 700;
      letter-spacing: 3px;
      text-transform: uppercase;
      color: var(--accent);
    }

    .header-right {
      display: flex;
      align-items: center;
      gap: 24px;
      font-family: var(--mono);
      font-size: 11px;
      color: var(--dim);
    }

    .status-dot {
      width: 8px; height: 8px;
      border-radius: 50%;
      background: var(--green);
      box-shadow: 0 0 8px var(--green);
      animation: pulse 2s infinite;
      display: inline-block;
      margin-right: 6px;
    }

    @keyframes pulse {
      0%, 100% { opacity: 1; }
      50% { opacity: 0.4; }
    }

    .main {
      padding: 24px 32px;
      display: grid;
      grid-template-columns: 1fr 380px;
      grid-template-rows: auto auto;
      gap: 20px;
    }

    .panel {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 4px;
      padding: 20px;
      position: relative;
      overflow: hidden;
    }

    .panel::before {
      content: '';
      position: absolute;
      top: 0; left: 0; right: 0;
      height: 2px;
      background: linear-gradient(90deg, var(--accent), transparent);
    }

    .panel-title {
      font-family: var(--mono);
      font-size: 11px;
      letter-spacing: 2px;
      color: var(--dim);
      text-transform: uppercase;
      margin-bottom: 16px;
      display: flex;
      align-items: center;
      gap: 8px;
    }

    .panel-title::before {
      content: '//';
      color: var(--accent);
    }

    /* Tahmin paneli */
    .pred-display {
      text-align: center;
      padding: 24px 0;
    }

    .pred-label {
      font-family: var(--mono);
      font-size: 11px;
      letter-spacing: 3px;
      color: var(--dim);
      text-transform: uppercase;
      margin-bottom: 12px;
    }

    .pred-value {
      font-family: var(--sans);
      font-size: 28px;
      font-weight: 700;
      letter-spacing: 2px;
      text-transform: uppercase;
      color: var(--accent);
      text-shadow: 0 0 20px rgba(0,212,255,0.4);
      transition: all 0.3s ease;
      min-height: 40px;
    }

    .pred-value.normal { color: var(--green); text-shadow: 0 0 20px rgba(0,255,136,0.4); }
    .pred-value.fault  { color: var(--accent2); text-shadow: 0 0 20px rgba(255,107,53,0.4); }

    .confidence-bar {
      margin: 16px auto;
      max-width: 280px;
    }

    .conf-label {
      display: flex;
      justify-content: space-between;
      font-family: var(--mono);
      font-size: 11px;
      color: var(--dim);
      margin-bottom: 6px;
    }

    .conf-track {
      height: 4px;
      background: var(--border);
      border-radius: 2px;
      overflow: hidden;
    }

    .conf-fill {
      height: 100%;
      background: linear-gradient(90deg, var(--accent), var(--green));
      border-radius: 2px;
      transition: width 0.5s ease;
      width: 0%;
    }

    /* Metrikler */
    .metrics {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
      margin-top: 16px;
    }

    .metric {
      background: rgba(0,212,255,0.04);
      border: 1px solid var(--border);
      border-radius: 3px;
      padding: 12px;
    }

    .metric-label {
      font-family: var(--mono);
      font-size: 10px;
      color: var(--dim);
      letter-spacing: 1px;
      text-transform: uppercase;
      margin-bottom: 4px;
    }

    .metric-value {
      font-family: var(--mono);
      font-size: 16px;
      color: var(--text);
    }

    .metric-value span {
      font-size: 11px;
      color: var(--dim);
    }

    /* Kayıt */
    .log-panel {
      grid-column: 1 / -1;
    }

    .log-list {
      font-family: var(--mono);
      font-size: 12px;
      max-height: 140px;
      overflow-y: auto;
    }

    .log-list::-webkit-scrollbar { width: 4px; }
    .log-list::-webkit-scrollbar-track { background: var(--bg); }
    .log-list::-webkit-scrollbar-thumb { background: var(--border); }

    .log-entry {
      display: flex;
      gap: 16px;
      padding: 5px 0;
      border-bottom: 1px solid rgba(26,58,74,0.5);
      animation: fadeIn 0.3s ease;
    }

    @keyframes fadeIn {
      from { opacity: 0; transform: translateX(-8px); }
      to { opacity: 1; transform: translateX(0); }
    }

    .log-time { color: var(--dim); min-width: 70px; }
    .log-class { color: var(--accent); min-width: 200px; }
    .log-conf { color: var(--green); }
    .log-src  { color: var(--accent2); font-size: 10px; }

    /* Firebase badge */
    .firebase-badge {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      background: rgba(255,107,53,0.1);
      border: 1px solid rgba(255,107,53,0.3);
      border-radius: 3px;
      padding: 4px 10px;
      font-family: var(--mono);
      font-size: 10px;
      color: var(--accent2);
      margin-top: 12px;
    }

    .chart-wrap { position: relative; height: 200px; }
  </style>
</head>
<body>
  <header>
    <div class="logo">
      <div class="logo-icon">⚙</div>
      <h1>Rulman Arıza Tespit Sistemi</h1>
    </div>
    <div class="header-right">
      <span><span class="status-dot"></span>SİSTEM AKTİF</span>
      <span id="clock">--:--:--</span>
    </div>
  </header>

  <div class="main">
    <!-- Titreşim grafiği -->
    <div class="panel">
      <div class="panel-title">Gerçek Zamanlı Titreşim — X / Y / Z Ekseni</div>
      <div class="chart-wrap">
        <canvas id="chart"></canvas>
      </div>
      <div class="metrics" style="margin-top:12px">
        <div class="metric">
          <div class="metric-label">X Ekseni</div>
          <div class="metric-value" id="vx">—<span> m/s²</span></div>
        </div>
        <div class="metric">
          <div class="metric-label">Y Ekseni</div>
          <div class="metric-value" id="vy">—<span> m/s²</span></div>
        </div>
        <div class="metric">
          <div class="metric-label">Z Ekseni</div>
          <div class="metric-value" id="vz">—<span> m/s²</span></div>
        </div>
        <div class="metric">
          <div class="metric-label">Örnekler</div>
          <div class="metric-value" id="vcnt">0<span> pkt</span></div>
        </div>
      </div>
    </div>

    <!-- Tahmin paneli -->
    <div class="panel">
      <div class="panel-title">Model Tahmini</div>
      <div class="pred-display">
        <div class="pred-label">Tespit Edilen Durum</div>
        <div class="pred-value" id="pred">BEKLENIYOR</div>
        <div class="confidence-bar">
          <div class="conf-label">
            <span>GÜVEN</span>
            <span id="conf-pct">—</span>
          </div>
          <div class="conf-track">
            <div class="conf-fill" id="conf-fill"></div>
          </div>
        </div>
        <div class="firebase-badge" id="fb-badge">
          ☁ Firebase: —
        </div>
      </div>
    </div>

    <!-- Log -->
    <div class="panel log-panel">
      <div class="panel-title">Tahmin Kaydı</div>
      <div class="log-list" id="log"></div>
    </div>
  </div>

  <script>
    // Saat
    setInterval(() => {
      document.getElementById('clock').textContent =
        new Date().toLocaleTimeString('tr-TR');
    }, 1000);

    // Chart
    const ctx = document.getElementById('chart').getContext('2d');
    const chart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: [],
        datasets: [
          { label: 'X', data: [], borderColor: '#00d4ff', borderWidth: 1.5, pointRadius: 0, tension: 0.3 },
          { label: 'Y', data: [], borderColor: '#00ff88', borderWidth: 1.5, pointRadius: 0, tension: 0.3 },
          { label: 'Z', data: [], borderColor: '#ff6b35', borderWidth: 1.5, pointRadius: 0, tension: 0.3 }
        ]
      },
      options: {
        animation: false,
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { labels: { color: '#4a7a8a', font: { family: 'Share Tech Mono', size: 11 } } }
        },
        scales: {
          x: { ticks: { color: '#4a7a8a', maxTicksLimit: 6, font: { family: 'Share Tech Mono', size: 10 } },
               grid: { color: 'rgba(26,58,74,0.5)' } },
          y: { ticks: { color: '#4a7a8a', font: { family: 'Share Tech Mono', size: 10 } },
               grid: { color: 'rgba(26,58,74,0.5)' } }
        }
      }
    });

    const MAX = 120;
    let t = 0, cnt = 0;

    const socket = io();

    socket.on('veri', function(d) {
      cnt++;
      if (chart.data.labels.length > MAX) {
        chart.data.labels.shift();
        chart.data.datasets.forEach(ds => ds.data.shift());
      }
      chart.data.labels.push(t++);
      chart.data.datasets[0].data.push(d.x);
      chart.data.datasets[1].data.push(d.y);
      chart.data.datasets[2].data.push(d.z);
      chart.update('none');

      document.getElementById('vx').innerHTML = d.x.toFixed(2) + '<span> m/s²</span>';
      document.getElementById('vy').innerHTML = d.y.toFixed(2) + '<span> m/s²</span>';
      document.getElementById('vz').innerHTML = d.z.toFixed(2) + '<span> m/s²</span>';
      document.getElementById('vcnt').innerHTML = cnt + '<span> pkt</span>';
    });

    socket.on('tahmin', function(d) {
      const el = document.getElementById('pred');
      el.textContent = d.sinif.toUpperCase().replace(/-/g, ' ');
      el.className = 'pred-value ' + (d.sinif === 'normal' ? 'normal' : 'fault');

      const pct = Math.round(d.guven);
      document.getElementById('conf-pct').textContent = pct + '%';
      document.getElementById('conf-fill').style.width = pct + '%';

      document.getElementById('fb-badge').textContent = '☁ Firebase: ' + d.zaman;

      // Log ekle
      const log = document.getElementById('log');
      const entry = document.createElement('div');
      entry.className = 'log-entry';
      entry.innerHTML =
        '<span class="log-time">' + d.zaman + '</span>' +
        '<span class="log-class">' + d.sinif + '</span>' +
        '<span class="log-conf">' + pct + '%</span>';
      log.insertBefore(entry, log.firstChild);
      if (log.children.length > 20) log.removeChild(log.lastChild);
    });
  </script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML)
@app.route("/test/<sinif>")
def test_pred(sinif):
    from datetime import datetime
    zaman = datetime.now().strftime("%H:%M:%S")
    sock.emit("tahmin", {"sinif": sinif, "guven": 99.0, "zaman": zaman})
    return f"Gönderildi: {sinif}"

# ─── TAHMİN ────────────────────────────────────────────────────────────────────
def signal_to_spec(signal):
    signal = np.array(signal, dtype=np.float32)
    signal = (signal - signal.mean()) / (signal.std() + 1e-8)
    _, _, Zxx = stft(signal, nperseg=64, noverlap=48)
    spec = np.abs(Zxx)
    spec = np.log1p(spec * 1000)
    spec = (spec - spec.min()) / (spec.max() - spec.min() + 1e-8)
    return spec.astype(np.float32)

def predict():
    specs  = np.stack([signal_to_spec(list(buffer_x)),
                       signal_to_spec(list(buffer_y)),
                       signal_to_spec(list(buffer_z))])
    tensor = torch.from_numpy(specs).unsqueeze(0)
    tensor = torch.nn.functional.interpolate(
        tensor, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False
    ).to(DEVICE)
    with torch.no_grad():
        out   = model(tensor)
        probs = torch.softmax(out, dim=1)[0]
        pred  = probs.argmax().item()
        conf  = probs[pred].item() * 100
    return CLASS_NAMES[pred], conf

# ─── MQTT ──────────────────────────────────────────────────────────────────────
def on_connect(client, userdata, flags, rc):
    print("MQTT bağlandı!" if rc == 0 else f"MQTT hata: {rc}")
    client.subscribe("vibration/data")
    client.subscribe("vibration/tahmin")

def on_message(client, userdata, msg):
    global counter
    topic = msg.topic
    
    if topic == "vibration/tahmin":
        sinif = msg.payload.decode()
        zaman = datetime.now().strftime("%H:%M:%S")
        guven = 99.0  # TinyML güven skoru
        try:
            firebase_ref.push({
                "sinif": sinif,
                "guven": guven,
                "zaman": zaman,
                "kaynak": "TinyML_ESP32"
            })
            print(f"TAHMİN: {sinif} → Firebase ✅")
        except Exception as e:
            print(f"Firebase hata: {e}")
        sock.emit("tahmin", {"sinif": sinif, "guven": guven, "zaman": zaman})
        return

    x, y, z = map(float, msg.payload.decode().split(","))
    buffer_x.append(x); buffer_y.append(y); buffer_z.append(z)
    counter += 1
    sock.emit("veri", {"x": x, "y": y, "z": z})

    if len(buffer_x) == WINDOW_SIZE and counter % 50 == 0:
        sinif, guven = predict()
        zaman = datetime.now().strftime("%H:%M:%S")
        try:
            firebase_ref.push({
                "sinif": sinif,
                "guven": round(guven, 2),
                "zaman": zaman,
                "kaynak": "EfficientNet_PC"
            })
            print(f"TAHMİN: {sinif} ({guven:.1f}%) → Firebase ✅")
        except Exception as e:
            print(f"Firebase hata: {e}")
        sock.emit("tahmin", {"sinif": sinif, "guven": guven, "zaman": zaman})

    if len(buffer_x) == WINDOW_SIZE and counter % 50 == 0:
        sinif, guven = predict()
        zaman = datetime.now().strftime("%H:%M:%S")

        # Firebase'e yaz
        try:
            firebase_ref.push({
                "sinif": sinif,
                "guven": round(guven, 2),
                "zaman": zaman,
                "x": round(x, 3),
                "y": round(y, 3),
                "z": round(z, 3)
            })
            print(f"TAHMİN: {sinif} ({guven:.1f}%) → Firebase ✅")
        except Exception as e:
            print(f"Firebase hata: {e}")

        sock.emit("tahmin", {"sinif": sinif, "guven": guven, "zaman": zaman})

def mqtt_thread():
    c = mqtt.Client()
    c.on_connect = on_connect
    c.on_message = on_message
    c.connect("localhost", 1883, 60)
    c.loop_forever()

threading.Thread(target=mqtt_thread, daemon=True).start()

if __name__ == "__main__":
    print("Dashboard: http://localhost:5000")
    sock.run(app, host="0.0.0.0", port=5000, debug=False)