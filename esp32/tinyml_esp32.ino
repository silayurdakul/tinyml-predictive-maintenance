#include "tinyml_model.h"
#include "secrets.h"
#include <tflm_esp32.h>
#include <eloquent_tinyml.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <Wire.h>
#include <arduinoFFT.h>
#include <WiFi.h>
#include <PubSubClient.h>

#define ARENA_SIZE  120000
#define WINDOW_SIZE 512
#define IMG_SIZE    32
#define N_CLASSES   7


const char* CLASS_NAMES[] = {
  "normal", "horizontal-misalignment", "vertical-misalignment",
  "imbalance", "ball_fault", "cage_fault", "outer_race"
};

Eloquent::TF::Sequential<16, ARENA_SIZE> tf;
Adafruit_MPU6050 mpu;
ArduinoFFT<float> FFT;
WiFiClient espClient;
PubSubClient mqtt(espClient);

float buf_x[WINDOW_SIZE], buf_y[WINDOW_SIZE], buf_z[WINDOW_SIZE];
float vReal[WINDOW_SIZE], vImag[WINDOW_SIZE];
float input_data[IMG_SIZE * IMG_SIZE * 3];
float all_vals[IMG_SIZE * IMG_SIZE * 3];
int buf_idx = 0;
bool buf_full = false;
unsigned long last_sample = 0;

void normalize(float* arr, int n) {
  float m = 0, s = 0;
  for (int i = 0; i < n; i++) m += arr[i];
  m /= n;
  for (int i = 0; i < n; i++) s += (arr[i]-m)*(arr[i]-m);
  s = sqrt(s/n) + 1e-8;
  for (int i = 0; i < n; i++) arr[i] = (arr[i]-m)/s;
}

void compute_fft_row(float* signal, int start, int seg_size, float* out_bins) {
  for (int i = 0; i < WINDOW_SIZE; i++) {
    vReal[i] = (i < seg_size) ? signal[start + i] : 0.0f;
    vImag[i] = 0.0f;
  }
  FFT.windowing(vReal, WINDOW_SIZE, FFT_WIN_TYP_HAMMING, FFT_FORWARD);
  FFT.compute(vReal, vImag, WINDOW_SIZE, FFT_FORWARD);
  FFT.complexToMagnitude(vReal, vImag, WINDOW_SIZE);
  int bins = WINDOW_SIZE / 2;
  int bin_per_col = bins / IMG_SIZE;
  for (int j = 0; j < IMG_SIZE; j++) {
    float val = 0;
    for (int k = j*bin_per_col; k < (j+1)*bin_per_col && k < bins; k++)
      val += vReal[k];
    val /= bin_per_col;
    out_bins[j] = log(1 + val * 0.1);
  }
}

void build_input() {
  int seg_size = WINDOW_SIZE / IMG_SIZE;
  float min_v = 1e10, max_v = -1e10;
  float row_x[IMG_SIZE], row_y[IMG_SIZE], row_z[IMG_SIZE];

  for (int row = 0; row < IMG_SIZE; row++) {
    int start = row * seg_size;
    compute_fft_row(buf_x, start, seg_size, row_x);
    compute_fft_row(buf_y, start, seg_size, row_y);
    compute_fft_row(buf_z, start, seg_size, row_z);
    for (int col = 0; col < IMG_SIZE; col++) {
      int base = (row * IMG_SIZE + col) * 3;
      all_vals[base+0] = row_x[col];
      all_vals[base+1] = row_y[col];
      all_vals[base+2] = row_z[col];
      if (all_vals[base+0] < min_v) min_v = all_vals[base+0];
      if (all_vals[base+1] < min_v) min_v = all_vals[base+1];
      if (all_vals[base+2] < min_v) min_v = all_vals[base+2];
      if (all_vals[base+0] > max_v) max_v = all_vals[base+0];
      if (all_vals[base+1] > max_v) max_v = all_vals[base+1];
      if (all_vals[base+2] > max_v) max_v = all_vals[base+2];
    }
  }
  float range = max_v - min_v + 1e-8;
  for (int i = 0; i < IMG_SIZE * IMG_SIZE * 3; i++)
    input_data[i] = (all_vals[i] - min_v) / range;
}

void mqtt_reconnect() {
  int tries = 0;
  while (!mqtt.connected() && tries < 3) {
    if (mqtt.connect("TinyML_ESP32")) {
      Serial.println("MQTT bağlandı!");
    } else {
      tries++;
      delay(1000);
    }
  }
}

void setup() {
  Serial.begin(115200);
  delay(1000);

  pinMode(4, OUTPUT);
  digitalWrite(4, HIGH);

  Wire.begin(8, 9);
  if (!mpu.begin()) {
    Serial.println("MPU6050 bulunamadi!");
    while (1);
  }
  mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
  mpu.setFilterBandwidth(MPU6050_BAND_44_HZ);
  Serial.println("MPU6050 hazir");

  // WiFi bağlan
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  Serial.print("WiFi bağlanıyor");
  int w = 0;
  while (WiFi.status() != WL_CONNECTED && w < 20) {
    delay(500); Serial.print("."); w++;
  }
  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\nWiFi bağlandı!");
    mqtt.setServer(mqtt_server, 1883);
  } else {
    Serial.println("\nWiFi bağlanamadı, sadece serial modda devam.");
  }

  tf.setNumInputs(IMG_SIZE * IMG_SIZE * 3);
  tf.setNumOutputs(N_CLASSES);
  tf.resolver.AddConv2D();
  tf.resolver.AddMaxPool2D();
  tf.resolver.AddReshape();
  tf.resolver.AddFullyConnected();
  tf.resolver.AddSoftmax();
  tf.resolver.AddRelu();
  tf.resolver.AddDepthwiseConv2D();
  tf.resolver.AddMean();
  tf.resolver.AddTranspose();
  tf.resolver.AddPad();
  tf.resolver.AddAdd();
  tf.resolver.AddMul();
  tf.resolver.AddBatchMatMul();

  while (!tf.begin(tinyml_model).isOk())
    Serial.println(tf.exception.toString());

  Serial.println("TinyML + WiFi + MQTT hazir!");
}

void loop() {
  if (micros() - last_sample < 1000) return;
  last_sample = micros();

  sensors_event_t a, g, temp;
  mpu.getEvent(&a, &g, &temp);

  buf_x[buf_idx] = a.acceleration.x;
  buf_y[buf_idx] = a.acceleration.y;
  buf_z[buf_idx] = a.acceleration.z;
  buf_idx++;

  if (buf_idx >= WINDOW_SIZE) {
    buf_idx = 0;
    buf_full = true;
  }

  if (buf_full && buf_idx == 0) {
    normalize(buf_x, WINDOW_SIZE);
    normalize(buf_y, WINDOW_SIZE);
    normalize(buf_z, WINDOW_SIZE);
    build_input();

    if (!tf.predict(input_data).isOk()) {
      Serial.println(tf.exception.toString());
      return;
    }

    const char* sinif = CLASS_NAMES[tf.classification];
    Serial.print("TAHMIN: ");
    Serial.println(sinif);

    // MQTT'ye gönder
    if (WiFi.status() == WL_CONNECTED) {
      if (!mqtt.connected()) mqtt_reconnect();
      if (mqtt.connected()) {
        // Sensör verisi
        String payload = String(a.acceleration.x) + "," +
                         String(a.acceleration.y) + "," +
                         String(a.acceleration.z);
        mqtt.publish("vibration/data", payload.c_str());

        // Tahmin
        mqtt.publish("vibration/tahmin", sinif);
        mqtt.loop();
      }
    }
  }
}