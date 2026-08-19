# Ensemble Activity Classifier — Arduino Nano 33 BLE Sense

**EE 446: Tiny Machine Learning for Ultra Low-Power Edge Computing | University of Washington, Spring 2026**

A tiny ensemble-learning pipeline for IMU-based human activity classification, trained on the mHealth dataset and compressed for deployment on a microcontroller.

---

## What it does

Classifies human activity (standing, sitting, lying down, walking, and other mHealth-labeled activities — 12 classes total) from a 100-sample × 6-feature IMU window (3-axis accelerometer + 3-axis gyroscope).

Three parallel branches process the same input window under different preprocessing strategies — raw, standard-scaled, and min-max-scaled — each with its own autoencoder (600 → 64 → 32) feeding a small classifier (32 → 20 → 12). The three 12-class softmax outputs are concatenated into a 36-dimensional vector and passed into a stacked meta-classifier (36 → 24 → 12) that produces the final prediction. Seven trained models work together: 3 encoders, 3 branch classifiers, 1 meta-classifier.

---

## Model compression

Each deployed model is compressed via:

1. **Magnitude-based pruning** — TensorFlow Model Optimization's PolynomialDecay schedule, sparsity ramped 10% → 80% over 500 steps
2. **Pruning-wrapper stripping** — `strip_pruning()` removes mask/bookkeeping overhead while preserving the sparse structure
3. **Quantization-aware training (QAT)** — simulates int8 arithmetic during training; a custom `MaskEnforcerCallback` keeps pruned weights at exactly zero throughout QAT (a plain prune-then-QAT pipeline lets fine-tuning silently un-prune those weights)
4. **Full int8 TFLite conversion** — calibrated with a representative dataset for accurate activation ranges

---

## Repository contents

```
TinyML_Lab8_Part_II.ipynb           ← Full pipeline: load mHealth data → train 3 branches → ensemble → prune/QAT/quantize
Lab8_EE446_Report.pdf               ← Written report (architecture, compression methodology, deployment analysis)
data/
  mHealth_subject6.log              ← mHealth Subject 6 raw sensor log (accel + gyro + activity label, 18 MB)
models/
  encoder_raw.tflite, encoder_std.tflite, encoder_minmax.tflite       ← Unpruned encoders (float)
  clf_raw.tflite, clf_std.tflite, clf_minmax.tflite                  ← Unpruned branch classifiers (float)
  stacked_meta_clf.tflite                                            ← Unpruned meta-classifier (float)
  encoder_clf_*_pruned_qat_int8.tflite / .cc                         ← Pruned + QAT + int8 branch models (deployed)
  stacked_meta_clf_pruned_qat_int8.tflite / .cc                      ← Pruned + QAT + int8 meta-classifier (deployed)
arduino/
  raw_imu_recorder/                 ← Sketch to record IMU data over Serial for building your own dataset
  tiny_ensemble_learning/           ← Inference sketch — loads all 4 compressed .cc models, runs the full ensemble
firmware/                           ← Pre-built firmware binary + flash scripts (flash without Arduino IDE)
```

---

## Quick start

### Run the notebook (train / reproduce)

```bash
pip install numpy pandas tensorflow tensorflow-model-optimization scikit-learn matplotlib
jupyter notebook TinyML_Lab8_Part_II.ipynb
```

The notebook expects `mHealth_subject6.log` in the same folder as itself — copy it in from `data/` before running, or point the notebook's data path at `data/mHealth_subject6.log`.

### Flash the Arduino sketch (inference)

1. Install the `TensorFlowLite` and `Arduino_BMI270_BMM150` libraries via Arduino Library Manager
2. Open `arduino/tiny_ensemble_learning/Tiny_Ensemble_Learning.ino` in Arduino IDE — the 4 `.cc` model files are already alongside it and load automatically
3. Upload to Nano 33 BLE Sense, open Serial Monitor at the sketch's configured baud rate
4. Place the board flat on a stable surface to observe steady-state predictions, then move it to see activity labels change

### Flash pre-built firmware (no Arduino IDE)

```bash
# Windows
firmware\flash_windows.bat

# Mac
firmware/flash_mac.command

# Linux
bash firmware/flash_linux.sh
```

### Collect your own IMU data

Open `arduino/raw_imu_recorder/Raw_IMU_Recorder.ino`, upload, and read the Serial output to capture your own 6-axis IMU recordings in the same format as `mHealth_subject6.log`.

---

## Hardware

- **Arduino Nano 33 BLE Sense** (Nordic nRF52840, 1 MB flash, 256 KB RAM, onboard IMU)

---

## Known deployment caveats

Per the lab's discussion questions: predictions on live hardware can be unstable or biased toward a single class relative to the mHealth training distribution, due to differences between the original dataset's sensor placement/orientation and the Arduino's onboard IMU axes, plus scaling mismatches between offline training data and live sensor readings. See `Lab8_EE446_Report.pdf` for the full analysis.

---

## Authors

Sparsh Dadhich — University of Washington, ECE / Neuroscience

---

## License

MIT — see [LICENSE](LICENSE). This covers the author's own code, notebooks, and documentation in this repo.
