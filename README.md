# 🧠 Emotion-Driven Game Agent

This project demonstrates an AI-controlled agent that dynamically navigates a game map based on real-time facial expression recognition using a webcam or screenshots. The system leverages computer vision, deep learning, and automation to simulate human-emotion-aware decision making in a grid-based environment.

---

## 🚀 Overview

An in-game agent moves within a virtual grid world. It evaluates the surrounding environment and reacts to the **player’s facial expressions** (e.g., normal vs shocked). The system avoids danger zones (visually detected with pixel colors) and backtracks when the player displays signs of fear or surprise.

---

## 🧩 Core Components

| Module               | Description |
|----------------------|-------------|
| `agent.py`           | Main execution logic: controls game agent based on emotion and environment state |
| `face_detection.py`  | Detects faces in a frame using OpenCV (Haar Cascade) |
| `model_train.py`     | Trains a deep learning model to differentiate between emotional states |
| `face_model.pt`      | Pre-trained facial embedding extractor (ResNet18-based) |
| `normal.jpg`, `shocked.jpg` | Reference face images used to compute emotion embeddings |
| `haarcascade_frontalface_default.xml` | Haar Cascade model for face detection |

---

## 🧠 Technologies Used

- **PyTorch** for feature embedding (ResNet18 backbone)
- **OpenCV** for face detection and image processing
- **PyAutoGUI** for screen interaction and automation
- **PIL / torchvision.transforms** for preprocessing face crops
- **Numpy** for vector distance comparison (emotion classification)

---

## 🎮 How It Works

1. **Game Area Detection**: System locates the game screen using color segmentation (looks for “pink” danger zones).
2. **Face Recognition**: Captures your face from a screenshot, detects facial region, and computes an embedding.
3. **Emotion Comparison**: Compares current embedding with pre-defined “normal” and “shocked” embeddings using L2 distance.
4. **Agent Behavior**:
    - If “shocked” face is detected → backtrack to previous grid
    - If danger zone detected ahead (pink color) → cancel move
    - Otherwise → move to the next unexplored direction

---

## 🛠️ Setup Instructions

### 1. Install Dependencies

```bash
pip install torch torchvision opencv-python pyautogui numpy matplotlib pillow
