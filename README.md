# Ensemble Activity Classifier

EE 446 Lab 8 code for classifying human activity from IMU data with a stacked ensemble. The repository includes a notebook that trains and exports the models, an Arduino inference sketch, the four deployed int8 model arrays, and one `mHealth_subject6.log` input file.

## Dataset and inputs

The notebook reads `mHealth_subject6.log`. It keeps rows whose activity-label column is greater than zero, uses columns 5–10 as three accelerometer and three gyroscope features, and changes the 1-based labels to 0-based labels. The 12 labels used by the sketch are: standing still, sitting and relaxing, lying down, walking, climbing stairs, waist bends forward, frontal elevation of arms, knees bending (crouching), cycling, jogging, running, and jump front and back.

Inputs are 100 samples by six features, flattened to 600 values. The Arduino sketch samples at 50 Hz, so it collects one such window in about two seconds.

## Ensemble

The same window is sent through three preprocessing branches: raw, standard-scaled, and min-max-scaled. Each branch has a 600 → 64 → 32 encoder and a 32 → 20 → 12 softmax classifier. The three 12-value outputs are concatenated into a 36-value vector for a 36 → 24 → 12 softmax stacked meta-classifier. For deployment, the sketch loads three combined encoder-classifier int8 models plus the int8 meta-classifier from `arduino/tiny_ensemble_learning/`.

The notebook applies pruning, quantization-aware training, and int8 TFLite conversion to the three combined branch models and the meta-classifier.

## Hardware and tools

The inference sketch uses `Arduino_BMI270_BMM150` and its comments identify the target IMU as the Arduino Nano 33 BLE Sense Rev2. It also includes TensorFlow Lite Micro headers through the Arduino `TensorFlowLite` library.

The notebook imports NumPy, pandas, Matplotlib, TensorFlow, TensorFlow Model Optimization, and scikit-learn.

## Run

### Train and export with the notebook

The notebook expects the mHealth log in the repository root. In PowerShell:

```powershell
Copy-Item data\mHealth_subject6.log .\mHealth_subject6.log
pip install numpy pandas matplotlib tensorflow tensorflow-model-optimization scikit-learn
jupyter notebook TinyML_Lab8_Part_II.ipynb
```

Run the notebook cells in order. Its optional final validation section only runs when `imu_500_rows.csv` is present in the repository root.

### Run inference on the board

1. Install the Arduino `TensorFlowLite` and `Arduino_BMI270_BMM150` libraries.
2. Open `arduino/tiny_ensemble_learning/Tiny_Ensemble_Learning.ino` in Arduino IDE. The four required `.cc` model files are in the same sketch folder.
3. Upload the sketch and open Serial Monitor at 115200 baud. The sketch prints a prediction after each 100-sample window.

`arduino/raw_imu_recorder/Raw_IMU_Recorder.ino` is a separate recorder sketch. It prints timestamped accelerometer and gyroscope readings to Serial at 50 Hz for two minutes.

## Credits

EE 446 Lab 8 repository authored by Sparsh Dadhich. Licensed under the MIT License; see [LICENSE](LICENSE).
