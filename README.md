# EmoSense - AI Face Emotion Recognition

EmoSense is a high-performance, real-time emotion detection system. It uses an AI pipeline to find faces in a webcam feed and classify their emotions into seven categories: **Angry, Disgust, Fear, Happy, Neutral, Sad, and Surprise**.

The project is built on the **YOLOv8** architecture and is optimized for NVIDIA RTX GPUs, though it can run on any system with a CPU as a fallback.

## How it Works

EmoSense uses a **Two-Stage Pipeline**:
1.  **Stage 1 (Face Detection)**: A YOLOv8-face model scans the image to find all human faces.
2.  **Stage 2 (Emotion Classification)**: Each face is cropped and passed to a custom-trained YOLOv8-cls model that predicts the emotion.

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
- **On Windows**: It installs the correct GPU-accelerated PyTorch and all UI/AI libraries.
- **On Linux (Ubuntu/Debian)**: It uses `sudo apt` to install required system libraries (`libGL`, `libglib`, etc.) and then installs the Python packages.
- **On other Linux distros**: It will warn you and list the libraries you need to install manually.

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

This script will detect your OS, check for NVIDIA drivers, and ensure PyTorch is correctly configured for CUDA. If anything is wrong, it will give you a direct link or command to fix it.

**Step 2 — Train the emotion model**

```
python 2_train_model.py
```

This reads the RAF-DB dataset, converts it to the format YOLO needs, trains the emotion classifier for 80 epochs, and saves the finished model to `models/emotion_classifier/`. It also exports a fast ONNX version of the model for use in the app.

Training on an RTX 3070 takes roughly 30 to 60 minutes depending on your dataset size. You will see a progress bar and live accuracy numbers in the console.

**Step 3 — Run the app**

```
python 3_run_app.py
```

This opens the main window. Click "Start Camera" to begin the live detection. You can also click "Load Image" to run the detector on a photo from your computer instead.

---

## What the App Window Shows

The left side of the window shows the camera feed with colored boxes drawn around any faces found. Each box has a small emoji badge in the corner showing the detected emotion, and a percentage in the opposite corner showing how confident the model is.

The right side of the window shows controls and statistics. The confidence threshold slider lets you control how certain the model must be before it shows a label. If you set it high, fewer but more reliable labels appear. If you set it low, more labels appear but some may be wrong.

Below the slider is a set of progress bars showing the percentage breakdown of every emotion detected since the camera started. At the very bottom is a large display showing the dominant emotion in the current frame.

---

## Optional Arguments

You can override defaults when running the training script:

```
python 2_train_model.py --epochs 100 --batch 32 --model yolov8n-cls.pt
```

You can also point the app to a specific model file if you have one saved in a different location:

```
python 3_run_app.py --model path/to/your/model.pt
```

---

## Common Problems

**The app says the model is not found.** This means training has not been run yet, or it did not finish. Run `python 2_train_model.py` and wait for it to complete fully before opening the app.

**The camera does not open.** Make sure no other application is using the webcam. The app tries to open the first camera it finds (device index 0).

**Predictions seem wrong.** Try lowering the confidence threshold slider so you can see more predictions, or raise it to filter out uncertain ones. If the model consistently misclassifies, it may need more training epochs.

---

*Created by Hamza Khan*
