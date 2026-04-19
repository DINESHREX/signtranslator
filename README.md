# SignTranslator 🤟

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.14-green.svg)](https://mediapipe.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**SignTranslator** is an advanced AI-powered desktop application designed to bridge the communication gap between the hearing-impaired and the hearing community. It uses computer vision and machine learning (MediaPipe + OpenCV) to translate sign language gestures into text and speech in real-time.

---

## 🌟 Key Features

- **👐 Real-Time Gesture Recognition**: Detects and translates hand signs instantly using your webcam.
- **🗣️ Bidirectional Translation**:
  - **Sign-to-Speech**: Translates gestures into audio using Text-to-Speech (TTS).
  - **Speech-to-Text**: Converts spoken words into text for easy reading.
- **📊 Custom Model Training**: Includes a built-in pipeline to collect your own gesture data and train custom models.
- **💻 Modern GUI**: A sleek, user-friendly desktop interface built with PyQt5.
- **🎙️ Voice Integration**: Built-in STT (Speech-to-Text) using Google Speech Recognition API.

---

## 🛠️ Technology Stack

- **Computer Vision**: [OpenCV](https://opencv.org/), [MediaPipe](https://mediapipe.dev/)
- **Machine Learning**: [Scikit-learn](https://scikit-learn.org/), [NumPy](https://numpy.org/), [Pandas](https://pandas.pydata.org/)
- **GUI Framework**: [PyQt5](https://www.riverbankcomputing.com/software/pyqt/)
- **Audio Processing**: [Pyttsx3](https://pypi.org/project/pyttsx3/) (TTS), [SpeechRecognition](https://pypi.org/project/SpeechRecognition/) (STT), [PyAudio](https://people.csail.mit.edu/hubert/pyaudio/)

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- A working webcam

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/DINESHREX/signtranslator.git
   cd signtranslator
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 📖 Usage

### 1. Running the Translator
Launch the main application via the GUI:
```bash
python ui/desktop/main.py  # Adjust path if necessary based on your entry point
```

### 2. Collecting Data (Optional)
To add new gestures to the system:
```bash
python src/collect_data.py
```

### 3. Training the Model
After collecting data, retrain the model:
```bash
python src/train_model.py
```

---

## 📁 Project Structure

```text
SignTranslator/
├── assets/          # Project images and media
├── data/            # Captured gesture landmarks for training
├── models/          # Saved ML models (.pkl / .h5)
├── src/             # Core logic
│   ├── capture.py       # Camera feed handling
│   ├── collect_data.py  # Data collection script
│   ├── gesture_engine.py# Landmark extraction & processing
│   ├── realtime_gesture.py# Main recognition logic
│   ├── stt.py           # Speech-to-Text module
│   ├── tts.py           # Text-to-Speech module
│   └── train_model.py   # ML training script
├── ui/              # User interface code (PyQt5)
├── requirements.txt # Project dependencies
└── README.md        # You are here!
```

---

## 🤝 Contributing

Contributions are welcome! If you have ideas for improvements or new features, feel free to:
1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

---

## ✉️ Contact

**Dinesh** - [GitHub Profile](https://github.com/DINESHREX)

Project Link: [https://github.com/DINESHREX/signtranslator](https://github.com/DINESHREX/signtranslator)
