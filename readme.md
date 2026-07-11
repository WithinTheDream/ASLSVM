# 🤟 ASL AI Translator - Pro Edition

A desktop application powered by **Computer Vision** and **Machine Learning** that translates American Sign Language (ASL) into text and speech in real time. This project uses **Support Vector Machine (SVM)** for gesture classification and **MediaPipe** for hand landmark extraction.

## ✨ Key Features

- **Real-time ASL Detection:** Recognizes static ASL letters with high accuracy using a lightweight and fast SVM model.
- **FSM (Finite State Machine) for Dynamic Letters:** Implements a sweep detection algorithm to recognize dynamic ASL letters such as **'Z'**.
- **Smart Debouncer & Auto Spacing:** A smart typing algorithm that outputs a letter only after it has been held steady for several frames, and automatically inserts spaces when the hand is removed from the camera.
- **Text-to-Speech (TTS):** Uses the Windows **SAPI5** speech engine to pronounce detected letters and words asynchronously without blocking the camera thread.
- **Modern GUI:** A Cyberpunk-inspired dark mode interface built with `CustomTkinter`, featuring camera switching and mirror mode.

## 🛠️ Tech Stack

- **Language:** Python 3.11+
- **Computer Vision:** OpenCV, Google MediaPipe (Hands)
- **Machine Learning:** Scikit-learn (Support Vector Machine)
- **Desktop GUI:** CustomTkinter, Pillow (PIL)
- **Text-to-Speech:** pyttsx3 (SAPI5)
- **Packaging:** PyInstaller

## 🚀 Getting Started (Development)

### 1. Clone the repository

```bash
git clone https://github.com/WithinTheDream/ASLSVM.git
cd ASLSVM
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate
```

### 3. Install the dependencies

```bash
pip install -r requirements.txt
```

Make sure the following packages are installed:

- opencv-python
- mediapipe
- scikit-learn
- customtkinter
- pyttsx3
- Pillow
- numpy

### 4. Run the application

```bash
python desktop_app.py
```

## 📦 Building the Executable (.exe)

To package the application into a standalone Windows executable, run the following command inside your virtual environment:

```bash
pyinstaller --onefile --noconsole --clean --collect-all customtkinter --hidden-import pyttsx3.drivers.sapi5 desktop_app.py
```

> **Important:** After the build process is complete, the executable will be located in the `dist/` folder. Copy the `svm_asl_model.pkl` file into the same folder as the executable before running the application.

## 🧠 Project Structure

| File | Description |
|------|-------------|
| `1_data_collector.py` | Collects hand landmark data from the webcam and saves it as a CSV dataset. |
| `2_train_model.py` | Trains the SVM model using the collected CSV dataset. |
| `3_live_inference.py` | Prototype script for real-time ASL detection using OpenCV. |
| `desktop_app.py` | Main production application with a CustomTkinter GUI. |
| `svm_asl_model.pkl` | The trained machine learning model used for ASL recognition. |