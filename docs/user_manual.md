# User Manual

## Introduction

This project is an AI-powered ASL (American Sign Language) recognition system.

The system detects hand landmarks using MediaPipe and predicts hand signs using a Random Forest machine learning model.

---

## Requirements

- Python 3.10+
- Webcam
- Virtual Environment

Required Libraries:

- OpenCV
- MediaPipe
- NumPy
- Pandas
- Scikit-Learn

---

## How To Run

### Step 1

Activate virtual environment

Windows:

```bash
venv\Scripts\activate
```

### Step 2

Run:

```bash
python src/ui/enhanced_live_prediction.py
```

### Step 3

Show supported ASL signs:

- A
- B
- C
- D
- E
- F

### Step 4

Observe prediction on screen.

### Step 5

Press ESC to exit.

---

## Output

The system displays:

Prediction: <Detected Letter>

in real time.