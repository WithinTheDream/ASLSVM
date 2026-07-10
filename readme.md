# ASL Vision Pipeline (Desktop Prototype)

This repository contains a desktop prototype of a Computer Vision system for real-time American Sign Language (ASL) recognition. The system combines spatial hand landmark extraction with a Support Vector Machine (SVM) classifier to recognize ASL hand gestures.

This prototype isolates the Machine Learning pipeline from mobile application development, allowing the model to be trained, tested, and validated independently before integration into the final application.

## System Architecture

The system employs a **Wrist-Centric Normalization** technique. Instead of feeding raw image pixels into the model, it extracts 21 three-dimensional (X, Y, Z) hand landmarks using **MediaPipe**. All landmark coordinates are normalized relative to landmark 0 (the wrist), making the model robust to variations in hand position and distance from the camera.

The prediction model is based on a **Support Vector Machine (SVM)** with an RBF kernel. This approach effectively classifies high-dimensional spatial landmark data while maintaining strong generalization performance without requiring an extremely large dataset.

## Project Structure

- `1_data_collector.py`  
  Captures hand landmarks from a webcam in real time and stores them in a CSV dataset. Includes automatic mirror calibration.

- `2_train_model.py`  
  Trains the SVM classifier, performs automatic class balancing through undersampling, and evaluates the model using a confusion matrix.

- `3_live_inference.py`  
  Runs real-time ASL recognition using the trained `.pkl` model on live webcam input. Includes "Nothing" and "Unknown Gesture" detection using probability thresholds.

- `dataset_asl_bersih.csv`  
  The extracted hand landmark dataset used for training.

- `svm_asl_model.pkl`  
  The trained SVM model ready for real-time inference.

## Getting Started

### 1. Install Python

Make sure Python 3.10 or later is installed on your system.

### 2. Create and activate a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Running the Pipeline

### Step 1 – Collect Dataset

```bash
python 1_data_collector.py
```

### Step 2 – Train the Model

```bash
python 2_train_model.py
```

### Step 3 – Run Real-Time Inference

```bash
python 3_live_inference.py
```

## Technologies Used

- Python
- OpenCV
- MediaPipe
- NumPy
- Pandas
- Scikit-learn
- Joblib

## Model Performance

The SVM classifier was evaluated using:

- Confusion Matrix
- Precision
- Recall
- F1-Score

The final model achieved an accuracy of over **90%** on the test dataset.

## License

This project is intended for academic and research purposes.