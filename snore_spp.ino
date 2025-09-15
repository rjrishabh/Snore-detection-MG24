/*
 * Snore Detection (Improved with RMS + Confidence Margin + Adaptive Threshold)
 *
 * Features:
 *  - Adaptive silence threshold
 *  - RMS-based detection to avoid false triggers
 *  - Confidence margin filtering
 *  - Time-based smoothing
 */

#include <SilabsMicrophoneAnalog.h>
#include "SilabsTFLiteMicro.h"

// ----------------- Microphone config -----------------
#define MIC_DATA_PIN  PC9
#define MIC_PWR_PIN   PC8
#define NUM_SAMPLES   637  // matches model input
#define MIC_VALUE_MIN 735
#define MIC_VALUE_MAX 900

volatile bool data_ready_flag = false;
uint32_t mic_buffer[NUM_SAMPLES];
uint32_t mic_buffer_local[NUM_SAMPLES];

MicrophoneAnalog micAnalog(MIC_DATA_PIN, MIC_PWR_PIN);
void mic_samples_ready_cb();

// ----------------- TFLite Micro config -----------------
constexpr int kTensorArenaSize = 4*1024;
uint8_t tensor_arena[kTensorArenaSize];

tflite::MicroInterpreter* interpreter;
TfLiteTensor* model_input;
TfLiteTensor* model_output;

const float SNORE_THRESHOLD = 0.5f;   // probability threshold
const float SNORE_MARGIN    = 0.2f;   // snore > no_snore by this margin
const unsigned long MIN_SNORE_INTERVAL_MS = 500; // 0.5s between detections

// Sliding window for amplitude
#define WINDOW_SIZE 20
static uint32_t recent_amplitudes[WINDOW_SIZE] = {0};
static int amp_idx = 0;

// Adaptive silence tracking
static uint32_t baseline_amp = 0;
static bool baseline_initialized = false;

// Last snore detection time
static unsigned long last_snore_time = 0;

// ----------------- Setup -----------------
void setup() {
  Serial.begin(115200);
  Serial.println("Snore Detection - Silabs TensorFlowLite (Improved)");

  // Initialize microphone
  micAnalog.begin(mic_buffer, NUM_SAMPLES);
  micAnalog.startSampling(mic_samples_ready_cb);

  // Initialize TFLite Micro model
  sl_tflite_micro_init();
  model_input = sl_tflite_micro_get_input_tensor();
  interpreter = sl_tflite_micro_get_interpreter();
  model_output = sl_tflite_micro_get_output_tensor();

  Serial.println("Ready for snore detection");
}

// ----------------- Main loop -----------------
void loop() {
  if (data_ready_flag) {
    data_ready_flag = false;
    run_inference();
  }
}

// ----------------- Microphone callback -----------------
void mic_samples_ready_cb() {
  memcpy(mic_buffer_local, mic_buffer, NUM_SAMPLES * sizeof(uint32_t));
  data_ready_flag = true;
}

// ----------------- Inference -----------------
void run_inference() {
  micAnalog.stopSampling();

  // --- Compute RMS for more robust amplitude ---
  float sum_squares = 0;
  for (int i=0; i<NUM_SAMPLES; i++) {
    float val = (float)mic_buffer_local[i] - MIC_VALUE_MIN;
    sum_squares += val * val;
  }
  float rms = sqrt(sum_squares / NUM_SAMPLES);

  // --- HARD SILENCE FILTER ---
  if (rms < 15) { // ignore very low background noise
    micAnalog.startSampling(mic_samples_ready_cb);
    return;
  }

  // Store amplitude in sliding window
  recent_amplitudes[amp_idx] = (uint32_t)rms;
  amp_idx = (amp_idx + 1) % WINDOW_SIZE;

  // Update baseline slowly
  if (!baseline_initialized) {
    baseline_amp = (uint32_t)rms;
    baseline_initialized = true;
  }
  if (rms < baseline_amp + 10) {
    baseline_amp = (baseline_amp * 9 + (uint32_t)rms) / 10;
  }

  // Compute noise variance for adaptive threshold
  float mean = 0;
  for (int i=0; i<WINDOW_SIZE; i++) mean += recent_amplitudes[i];
  mean /= WINDOW_SIZE;

  float var = 0;
  for (int i=0; i<WINDOW_SIZE; i++) {
    float diff = recent_amplitudes[i] - mean;
    var += diff * diff;
  }
  var /= WINDOW_SIZE;
  float noise_stddev = sqrt(var);

  uint32_t adaptive_threshold = baseline_amp + (uint32_t)(5 * noise_stddev); // increased multiplier

  if ((uint32_t)rms < adaptive_threshold) {
    micAnalog.startSampling(mic_samples_ready_cb);
    return;
  }

  // --- Prepare model input ---
  int8_t* input_data = model_input->data.int8;
  for (int i = 0; i < NUM_SAMPLES; i++) {
    input_data[i] = (int8_t)((mic_buffer_local[i] - MIC_VALUE_MIN) * 255 / (MIC_VALUE_MAX - MIC_VALUE_MIN) - 128);
  }

  // --- Run inference ---
  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("Error: inference failed");
    micAnalog.startSampling(mic_samples_ready_cb);
    return;
  }

  // --- Decode output ---
  float snore_prob    = (model_output->data.int8[0] - model_output->params.zero_point) * model_output->params.scale;
  float no_snore_prob = (model_output->data.int8[1] - model_output->params.zero_point) * model_output->params.scale;

  // --- Confidence margin & time-based smoothing ---
  bool snore_detected = false;
  unsigned long now = millis();
  if (snore_prob > SNORE_THRESHOLD && (snore_prob - no_snore_prob) > SNORE_MARGIN) {
    if (now - last_snore_time > MIN_SNORE_INTERVAL_MS) {
      snore_detected = true;
      last_snore_time = now;
    }
  }

  // --- Print detection ---
  if (snore_detected) {
    Serial.println(">>> Snore detected!");
    //total_snore_events++; // increment counter for sleep scoring
  } else {
    Serial.println("No snore detected.");
  }

  micAnalog.startSampling(mic_samples_ready_cb);
}
