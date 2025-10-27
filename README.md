# 💤 SnoreSense: AI-Powered Snore Detection on the Seeed XIAO MG24 Sense

SnoreSense is a **TinyML project** that detects snoring sounds in real time using the **Seeed XIAO MG24 Sense** development board.  
It uses an onboard **microphone**, a **TensorFlow Lite (TFLite)** model, and runs completely offline — enabling **low-power, privacy-safe sleep monitoring**.

---

## 🎯 Problem Statement

Snoring is not just noisy — it can be a symptom of sleep disorders like **sleep apnea**.  
Conventional snore monitors are often expensive, power-hungry, or require cloud connectivity.  

**SnoreSense** aims to solve this by:
- Detecting snores locally on a microcontroller
- Working offline to preserve privacy
- Using affordable open-source hardware and ML tools

---

## 🧠 How It Works

1. The **XIAO MG24 Sense** continuously listens through its onboard PDM microphone.  
2. Audio samples are processed in small frames and passed through a **TinyML classifier** trained to detect snore sounds.  
3. When a snore is detected, the system can:
   - Light up an LED  
   - Log the event to memory  
   - Or send data via BLE (future update)

All this happens **entirely on-device** — no cloud required!

---

## ⚙️ Hardware Requirements

| Component | Description |
|------------|-------------|
| 🟩 Seeed XIAO MG24 Sense | Main MCU + onboard PDM microphone |
| 🔵 USB-C Cable | For programming and power |
| 🔴 Optional: LED/Buzzer | To indicate snore detection |
| 🪫 Optional: LiPo Battery | For portable operation |

---
