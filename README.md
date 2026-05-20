# 🧠 EmoSense — AI Face Emotion Recognition

> **Built by Hamza Khan · v2.0 — Deployment Edition**

EmoSense is a real-time desktop application that watches a webcam feed (or a static image), finds every human face in view, and tells you what emotion each person is showing. It classifies faces into **7 emotion categories**: Angry, Disgust, Fear, Happy, Neutral, Sad, and Surprise.

Under the hood it runs two AI models back-to-back on every single frame — one to find faces, another to read emotions — powered by the **YOLOv8** deep learning architecture and trained on the **RAF-DB** facial expression dataset.

---

## Table of Contents

- [How It Works (The Big Picture)](#how-it-works-the-big-picture)
- [The Dataset — RAF-DB](#the-dataset--raf-db)
- [The Neural Network — YOLOv8 (CNN)](#the-neural-network--yolov8-cnn)
  - [What Is a Convolutional Neural Network?](#what-is-a-convolutional-neural-network)
  - [How YOLOv8 Is Used in EmoSense](#how-yolov8-is-used-in-emosense)
- [The Two-Stage Pipeline](#the-two-stage-pipeline)
- [Training — How the AI Learns](#training--how-the-ai-learns)
  - [Transfer Learning](#transfer-learning)
  - [Data Augmentation](#data-augmentation)
  - [The Optimizer (AdamW)](#the-optimizer-adamw)
- [Installation & Setup](#installation--setup)
- [Running the App](#running-the-app)
- [What the App Window Shows](#what-the-app-window-shows)
- [Optional Arguments](#optional-arguments)
- [Common Problems](#common-problems)

---

## How It Works (The Big Picture)

Imagine you point your webcam at a room full of people. EmoSense does the following **~33 times per second** (every 30 milliseconds):

1. **Grabs a frame** from the camera.
2. **Stage 1 — Face Detection:** A pre-trained AI model (`yolov8n-face`) scans the entire frame and draws a box around every human face it can find. It also gives each box a "confidence score" — how sure it is that the thing inside the box is actually a face.
3. **Stage 2 — Emotion Classification:** Each face box is cut out (cropped), resized to 224×224 pixels, and fed into a second AI model (`yolov8s-cls`). This model outputs 7 probabilities — one for each emotion — and picks the highest one as its answer.
4. **Drawing:** The app draws colored bounding boxes, emoji badges, and confidence percentages on the frame, then displays it in the window.
5. **Statistics:** A sidebar keeps a running tally of all emotions detected, with progress bars and a "Current Emotion" display.

This is called a **two-stage pipeline** because the data flows through two separate models in sequence — detect first, classify second.

---

## The Dataset — RAF-DB

### What Is It?

**RAF-DB** stands for **Real-world Affective Faces Database**. It is a large, publicly available academic dataset created by researchers at Dalian University of Technology (China). It contains **~30,000 real-world face images** collected from the internet — not posed lab photos, but genuine candid shots of people in everyday situations.

### Why Real-World Data Matters

Older emotion datasets (like CK+ or JAFFE) used actors making exaggerated expressions in controlled lighting. Models trained on those datasets fail in the real world because:
- Real expressions are subtle and mixed (a "sad smile," a nervous laugh).
- Lighting, angles, and image quality vary wildly.
- People of different ages, ethnicities, and genders express emotions differently.

RAF-DB solves this by using photos that capture natural, unposed human expressions in the wild.

### How It's Organized

The dataset ships with numbered folders instead of emotion names:

| Folder Number | Emotion   |
|:-------------|:----------|
| 1            | Surprise  |
| 2            | Fear      |
| 3            | Disgust   |
| 4            | Happy     |
| 5            | Sad       |
| 6            | Angry     |
| 7            | Neutral   |

It is split into two sets:
- **`train/`** — the images the model learns from (the textbook).
- **`test/`** — the images the model is tested on after training (the exam). It has never seen these during training, so performance on this set tells you how well the model generalizes.

The training script (`2_train_model.py`) automatically converts this numbered structure into named folders (`happy/`, `angry/`, etc.) because YOLOv8 expects folder names to be the class labels.

### Class Imbalance

RAF-DB has a known skew: there are far more "happy" and "neutral" images than "fear" or "disgust" images. If you don't handle this, the model learns to just guess "happy" all the time because that gives it the best accuracy on average. EmoSense combats this using **data augmentation** (described below).

---

## The Neural Network — YOLOv8 (CNN)

### What Is a Convolutional Neural Network?

A **Convolutional Neural Network (CNN)** is a type of artificial intelligence specifically designed to understand images. Here's the intuition:

1. **Pixels → Patterns:** An image is just a grid of numbers (pixel brightness values). A CNN slides small "filters" (tiny grids, e.g. 3×3 pixels) across the image. Each filter is trained to detect a specific pattern — horizontal edges, vertical edges, color blobs, curves, etc.

2. **Layers Build On Each Other:** The output of one layer becomes the input to the next. Early layers detect simple things (edges, corners). Middle layers combine those into shapes (eyes, noses, mouths). Deep layers combine shapes into concepts ("this looks like a smiling mouth" or "these are furrowed eyebrows").

3. **Pooling (Shrinking):** Between layers, the network reduces the spatial size (from 224×224 → 112×112 → 56×56 → ...) so that later layers look at bigger regions of the original image.

4. **Classification Head:** At the very end, all the spatial information is flattened into a single list of numbers and passed through "fully connected" layers that output one probability per class (in our case, 7 probabilities for 7 emotions). The class with the highest probability is the prediction.

Think of it like this: if you were teaching a child to recognize emotions, you'd start by teaching them what eyes and mouths look like, then how they change shape when someone is happy vs. sad, then how the whole face arrangement signals an emotion. A CNN does the same thing, but with math.

### How YOLOv8 Is Used in EmoSense

**YOLO** stands for **"You Only Look Once"** — it was originally designed for object detection (finding and labeling objects in images in a single pass). YOLOv8 is the 8th generation, developed by [Ultralytics](https://ultralytics.com).

EmoSense uses **two different YOLOv8 variants**:

| Model | File | Task | Architecture Variant | What It Does |
|:------|:-----|:-----|:----|:-------------|
| **Face Detector** | `yolov8n-face.pt` | Object Detection | YOLOv8-**nano** (smallest) | Finds face bounding boxes in the full frame |
| **Emotion Classifier** | `yolov8s-cls` (trained by you) | Image Classification | YOLOv8-**small** | Takes a cropped face and predicts one of 7 emotions |

**Why "small" for classification but "nano" for detection?**
- The face detector only needs to answer "where are the faces?" — a simpler task, so the tiny nano model is fast enough.
- The emotion classifier needs to distinguish between 7 subtle expressions — a harder task that benefits from the small model's extra parameters (more "brain cells").
- On a modern GPU (like an RTX 3070), the speed difference between nano and small is negligible.

### Key YOLOv8 Architecture Details

- **Backbone:** A series of convolutional blocks (called **C2f** blocks in YOLOv8) that extract features from the image. These use "bottleneck" layers with residual connections — meaning the network can learn to pass information through unchanged if a layer isn't helping, which prevents deep networks from degrading.
- **Neck (for detection model):** A **Feature Pyramid Network (FPN)** that combines features at multiple resolutions so the model can detect both large and small faces.
- **Head:** For classification, the head is a global average pooling layer followed by a fully connected layer that outputs 7 class probabilities.
- **Input Size:** 224×224 pixels (the standard size inherited from ImageNet, large enough for facial details, small enough for speed).

---

## The Two-Stage Pipeline

Here is exactly what happens to every camera frame, step by step:

```
┌──────────────┐     ┌──────────────────────┐     ┌───────────────────────┐
│  Camera      │     │  Stage 1: Face       │     │  Stage 2: Emotion     │
│  Frame       │────▶│  Detection           │────▶│  Classification       │
│  (1920×1080) │     │  (yolov8n-face)      │     │  (yolov8s-cls)        │
│              │     │  → Bounding boxes    │     │  → 7 probabilities    │
│              │     │  → Confidence scores │     │  → Best = prediction  │
└──────────────┘     └──────────────────────┘     └───────────────────────┘
```

### Stage 1 — Face Detection (in detail)

1. The frame is resized to **640×640** pixels.
2. `yolov8n-face` processes it and returns a list of bounding boxes `(x1, y1, x2, y2)` with confidence scores.
3. **Non-Maximum Suppression (NMS)** is applied: if two boxes overlap by more than 40% (measured by Intersection over Union / IoU), the one with the lower confidence is discarded. This prevents the same face from being detected twice.
4. **Bounding Box Tightening:** YOLO's face boxes often include the neck and shoulders. The app shrinks each box to focus on just the head:
   - Makes the box **square** using the center point and the shorter dimension × 0.53.
   - Trims 4% from the sides, 2% from the top, 10% from the bottom.
   - Clamps coordinates to stay inside the frame.

### Stage 2 — Emotion Classification (in detail)

1. Each tightened face region is cropped from the original frame.
2. The crop is resized to **224×224** pixels and fed to the emotion classifier.
3. The model outputs a **probability distribution** across all 7 emotions (e.g., `[0.02, 0.01, 0.03, 0.85, 0.04, 0.03, 0.02]` = 85% happy).
4. The emotion with the highest probability is picked. If that probability is below the **confidence threshold** set by the slider in the UI, the detection is silently discarded.

---

## Training — How the AI Learns

### Transfer Learning

Training a CNN from scratch requires millions of images and days of GPU time. Instead, EmoSense uses **transfer learning**:

1. Start with `yolov8s-cls.pt` — a model already trained on **ImageNet** (1.4 million images, 1000 classes). It has already learned universal visual features: edges, textures, shapes, colors, spatial relationships.
2. **Fine-tune** it on RAF-DB. The early layers (edge detectors, texture recognizers) are mostly kept. The later layers (the parts that decide "is this a dog or a cat?") are retrained to instead decide "is this person happy or sad?"
3. This means training takes **30–60 minutes** on an RTX 3070 instead of days.

### Data Augmentation

Since RAF-DB has class imbalance (way more happy/neutral images than fear/disgust), the training script applies **augmentations** — random transformations to each image before the model sees it:

| Augmentation | What It Does | Why It Helps |
|:-------------|:-------------|:-------------|
| **HSV Hue ±1.5%** | Slightly shifts colors | Makes the model ignore skin tone differences |
| **HSV Saturation ±50%** | Changes color intensity | Handles washed-out vs. vivid lighting |
| **HSV Brightness ±40%** | Makes image lighter/darker | Handles dim rooms and bright sunlight |
| **Rotation ±15°** | Tilts the face | People tilt their heads when expressing emotions |
| **Translation ±15%** | Moves the face around | Face won't always be perfectly centered |
| **Scale ±50%** | Zooms in/out | Handles close-up vs. far-away faces |
| **Horizontal Flip 50%** | Mirrors the image | Faces are roughly symmetrical |
| **Random Erasing 40%** | Hides part of the image | Forces the model to use *all* facial features, not just one |
| **Mixup 10%** | Blends two images together | Helps the model learn rare emotions by mixing them with common ones |

These make every training image slightly different each time it's shown, so the model can't just memorize the photos — it has to learn the *concept* of each emotion.

### The Optimizer (AdamW)

The optimizer is the algorithm that adjusts the model's internal numbers ("weights") after seeing each batch of images:

- **AdamW** is used — a modern variant of stochastic gradient descent that adapts the learning speed per-parameter and includes weight decay (a penalty that keeps weights small and prevents overfitting).
- **Initial learning rate:** 0.001 (how big each adjustment step is).
- **Warmup:** For the first 3 epochs, the learning rate gradually ramps up from near-zero to 0.001. This prevents the model from making wild adjustments with its random initial weights.
- **Decay:** The learning rate gradually decreases to 0.01× its initial value over the remaining epochs. As the model gets close to the optimal solution, smaller steps prevent it from overshooting.
- **Early Stopping (patience=20):** If the validation accuracy doesn't improve for 20 consecutive epochs, training stops automatically. This saves time and prevents overfitting.

### Training Output

After training, two model files are produced:
1. **`best.pt`** — The PyTorch model checkpoint from the epoch with the best validation accuracy.
2. **`best.onnx`** — The same model exported to ONNX format, which runs faster at inference time because ONNX Runtime applies graph optimizations.

Both are saved to `models/emotion_classifier/`.

---

## Installation & Setup

### Operating System Support
- **Windows**: Fully supported (Windows 10/11 recommended).
- **Linux**: Supported for **Ubuntu/Debian-based** systems (Ubuntu, Kali, Mint, etc.).
- **Other Linux Distros**: Supported, but requires manual installation of system dependencies (see [Warning](#linux-warning) below).

### Step 1 — Install Dependencies

Open your terminal or command prompt in the project folder and run:

```bash
python 1_install_requirements.py
```

**What this script does:**
- **On Windows**: Installs GPU-accelerated PyTorch (CUDA 12.1) and all UI/AI libraries.
- **On Linux (Ubuntu/Debian)**: Uses `sudo apt` to install required system libraries (`libGL`, `libglib`, etc.) then installs Python packages.
- **On other Linux distros**: Warns you and lists the libraries you need to install manually.

<a name="linux-warning"></a>
> [!WARNING]
> **For Non-Debian/Ubuntu Linux Users:**
> The installer cannot automatically install system libraries for your distro (e.g., Arch, Fedora). You MUST manually install:
> `libGL`, `libglib2.0`, `libSM`, `libXext`, and `libXrender`.
> Without these, the `cv2` (OpenCV) library will fail to import.

### Step 2 — Fix & Verify GPU

To ensure your NVIDIA GPU is being used (essential for fast training), run:

```bash
python 0_fix_gpu.py
```

This script detects your OS, checks for NVIDIA drivers, verifies PyTorch CUDA support, reinstalls PyTorch with CUDA if needed, and runs a GPU speed benchmark.

### Step 3 — Train the Emotion Model

```bash
python 2_train_model.py
```

This reads the RAF-DB dataset, converts it to the folder layout YOLO needs, trains the emotion classifier for 80 epochs, saves the finished model to `models/emotion_classifier/`, and exports a fast ONNX version.

Training on an RTX 3070 takes roughly 30 to 60 minutes. You will see a progress bar and live accuracy numbers in the console.

### Step 4 — Run the App

```bash
python 3_run_app.py
```

This opens the main window. Click "Start Camera" to begin live detection, or "Load Image" to run the detector on a photo.

---

## What the App Window Shows

The **left side** shows the camera feed with colored bounding boxes around faces. Each box has:
- An **emoji badge** in the top-left corner showing the detected emotion.
- A **confidence percentage pill** in the top-right corner.
- The **emotion name** as a text label below the emoji.

The **right side** (sidebar) shows:
- **Start/Stop Camera** and **Load Image** buttons.
- A **Confidence Threshold slider** — drag right to show only high-confidence predictions, drag left to show more (but less certain) predictions.
- **Emotion Statistics** — progress bars showing the percentage breakdown of every emotion since the camera started.
- **Current Emotion** — a large emoji and label showing the dominant emotion in the latest frame.
- **FPS counter** and **detection count** at the bottom.

---

## Optional Arguments

Override defaults when training:

```bash
python 2_train_model.py --epochs 100 --batch 32 --model yolov8n-cls.pt
```

Point the app to a specific model:

```bash
python 3_run_app.py --model path/to/your/model.pt
```

Skip the splash screen:

```bash
python 3_run_app.py --no-splash
```

---

## Common Problems

**The app says the model is not found.** Training has not been run yet, or it did not finish. Run `python 2_train_model.py` and wait for it to complete fully before opening the app.

**The camera does not open.** Make sure no other application is using the webcam. The app tries to open the first camera (device index 0).

**Predictions seem wrong.** Try adjusting the confidence threshold slider. If the model consistently misclassifies, it may need more training epochs (`--epochs 100`).

**Low FPS.** Make sure the ONNX model is being used and `torch.cuda.is_available()` returns `True`. If the GPU isn't being used, everything will be much slower.

---

## Technology Stack

| Component | Technology | Purpose |
|:----------|:-----------|:--------|
| Neural Network Framework | PyTorch | Core deep learning engine |
| Model Architecture | YOLOv8 (Ultralytics) | Face detection + emotion classification |
| Inference Engine | ONNX Runtime (GPU) | Fast model execution at runtime |
| Computer Vision | OpenCV (cv2) | Camera access, image processing, drawing |
| Desktop GUI | CustomTkinter | Modern dark-theme desktop UI |
| Image Handling | Pillow (PIL) | Emoji rendering, transparent overlays |
| Data Visualization | Matplotlib | Training loss/accuracy charts |
| Numerical Computing | NumPy | Array math for images and NMS |
| Dataset | RAF-DB | ~30,000 real-world facial expression images |

---

*Created by Hamza Khan*
