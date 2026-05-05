# EmoSense - Technical Project Guide

This document explains how the entire Face Emotion Recognition system is built and how each part works. It covers every stage from raw data to the final running application.

---

## Overview

The system works in two stages. Stage 1 finds faces in an image. Stage 2 looks at each face and decides what emotion the person is showing. These two stages run back-to-back on every frame of the camera feed, which is why the system is called a two-stage pipeline.

**Stage 1** uses a model called `yolov8n-face.pt`, which is a pre-trained neural network that has already learned to find faces in photos. It draws a bounding box around each face it finds and gives a confidence score between 0 and 1 indicating how sure it is.

**Stage 2** uses `yolov8s-cls`, a classification model that you train yourself using your own dataset. It takes the cropped face from Stage 1 and outputs a probability distribution across the 7 emotion classes: angry, disgust, fear, happy, neutral, sad, and surprise.

---

## 1. Libraries and Dependencies

EmoSense relies on several powerful libraries to handle AI, Computer Vision, and UI. Below is a detailed breakdown of why they were chosen and where they are used:

### AI & Computer Vision
- **`ultralytics` (YOLOv8)**:
  - **Where**: Used in `2_train_model.py` for training and `3_run_app.py` for inference.
  - **Why**: Provides the state-of-the-art YOLOv8 architecture. It's incredibly fast, accurate, and easy to use for both object detection (finding faces) and classification (recognizing emotions).
- **`torch` (PyTorch)**:
  - **Where**: The underlying engine for `ultralytics`. Used in `0_fix_gpu.py` to verify hardware acceleration.
  - **Why**: The industry standard for deep learning. It allows the model to run on NVIDIA GPUs using CUDA, making training 10x faster than on a CPU.
- **`opencv-python` (cv2)**:
  - **Where**: Used in `3_run_app.py` and `2_train_model.py`.
  - **Why**: The "eyes" of the project. It handles webcam access, image reading/writing, resizing frames, and drawing the colorful bounding boxes and labels on the screen.
- **`onnxruntime-gpu`**:
  - **Where**: Used in `3_run_app.py`.
  - **Why**: Once a model is trained, we export it to ONNX format. ONNX Runtime is often faster than PyTorch for running the model (inference) and has fewer dependencies.

### User Interface (GUI)
- **`customtkinter`**:
  - **Where**: Used in `3_run_app.py`.
  - **Why**: Python's default UI (Tkinter) looks very dated. CustomTkinter provides a modern, "Apple-style" dark theme with rounded corners and smooth animations, making the app feel premium.
- **`pillow` (PIL)**:
  - **Where**: Used in `3_run_app.py`.
  - **Why**: Used to handle emoji rendering and complex image transformations that OpenCV doesn't support well, like placing transparent overlays.

### Utilities
- **`numpy`**:
  - **Where**: Everywhere (mostly hidden inside other libs).
  - **Why**: Handles the heavy math. Images are treated as massive grids of numbers (arrays), and NumPy is the fastest way to process them.
- **`matplotlib`**:
  - **Where**: Used in `2_train_model.py`.
  - **Why**: Generates the training charts (loss and accuracy) so you can see how the model improved over time.
- **`tqdm`**:
  - **Where**: Used in the installation and training scripts.
  - **Why**: Provides the "smart" progress bars in the terminal so you know exactly how long a task will take.

---

## 2. Environment Setup (Windows & Linux)

Before any training or running can happen, the correct environment must be set up. The project is designed to be cross-platform, supporting both **Windows 10/11** and **Linux (Ubuntu/Debian)**.

### GPU Support
The project uses NVIDIA GPUs to speed up training. This requires:
1.  **NVIDIA Drivers**: Installed on the host OS.
2.  **CUDA Toolkit**: The software that lets Python talk to the GPU.
3.  **PyTorch CUDA**: A specific version of PyTorch built for your CUDA version.

The script `0_fix_gpu.py` is provided to automate the detection and installation of these components.

---

## 2. The Dataset (RAF-DB)

The model is trained on the Real-world Affective Faces Database (RAF-DB). This dataset contains tens of thousands of face images labeled with one of 7 emotions.

The dataset uses numbered folders instead of emotion names. Each number maps to an emotion:

| Folder | Emotion  |
| :----- | :------- |
| 1      | Surprise |
| 2      | Fear     |
| 3      | Disgust  |
| 4      | Happy    |
| 5      | Sad      |
| 6      | Angry    |
| 7      | Neutral  |

The dataset is split into a `train` folder and a `test` folder, each containing the same numbered subfolders.

---

## 3. The Training Script (`2_train_model.py`)

### Step 1: Dataset Preparation

YOLO's classification trainer expects a specific folder layout where each subfolder is named after the class it contains. Since RAF-DB uses numbered folders, the script first converts the layout.

It reads every image from the numbered subfolders (e.g., `train/4/`) and copies them into named emotion folders (e.g., `train/happy/`). The output is saved at `face_emotion/data/rafdb_yolo/`. The original files are not moved or deleted. The `test` split is renamed to `val` in the output since YOLO expects that name.

YOLO then reads the folder names alphabetically and assigns class indices in that order. This means: 0=angry, 1=disgust, 2=fear, 3=happy, 4=neutral, 5=sad, 6=surprise. The app uses this same ordering when reading predictions.

### Step 2: Model Training

The script loads `yolov8s-cls.pt`, which is a pre-trained small classification model. "Pre-trained" means it already knows how to recognize general visual features like edges, textures, and shapes from training on a large general dataset. Fine-tuning it on RAF-DB emotion images takes far less time than training from scratch.

Training runs for 80 epochs by default. Each epoch is one complete pass through all the training images. During each pass, the model makes a prediction for every image, compares that prediction to the correct label, calculates an error (called the loss), and adjusts its internal weights slightly to reduce that error next time.

**Augmentation** is applied to each training image before the model sees it. This randomly modifies the image in ways that preserve its meaning but make each view slightly different. The augmentations used are: hue/saturation/brightness jitter, random rotation up to 15 degrees, small translations, random zoom, horizontal flipping (faces look the same mirrored), random erasing of small patches, and mixup (blending two images together). These techniques are especially important because RAF-DB has significantly more happy and neutral examples than fear and disgust, and augmentation helps the model learn the minority classes better.

The optimizer used is AdamW. It starts with a learning rate of 0.001, which ramps up gradually over the first 3 epochs (warmup), then decays toward 0.01 over the remaining epochs. Training stops early if accuracy does not improve for 20 consecutive epochs.

### Step 3: Saving the Model

After training, the script copies the best-performing model checkpoint from `runs/detect/emotion_classifier/weights/best.pt` to `models/emotion_classifier/best.pt`. This is where the app looks for the model by default.

### Step 4: ONNX Export

The model is then exported from PyTorch format (`.pt`) to ONNX format (`.onnx`). ONNX is a universal model format that can run on many different hardware backends without needing the full PyTorch library. The ONNX Runtime with GPU support can take advantage of the RTX 3070's Tensor Cores to run inference faster than standard PyTorch. The export uses image size 224x224 and opset 12 for broad compatibility.

### Step 5: Sample Predictions

The script generates a 3-column grid of sample images from the validation set with predicted labels drawn on them. This grid is saved to `outputs/sample_predictions.png` so you can visually inspect whether the model is learning correctly.

---

## 4. The Application (`3_run_app.py`)

### Window and Layout

The app uses `customtkinter`, a modern wrapper around Python's built-in `tkinter` UI library. The window has three sections: a title bar across the top, a camera view on the left, and a sidebar on the right.

The title bar is drawn using a raw `tkinter.Canvas` for custom gradient and text rendering. The camera panel displays the live feed or a loaded image. The sidebar contains the controls, the emotion statistics, and the current emotion display.

### Model Loading

When the app starts, it first tries to load `yolov8n-face.pt` for Stage 1. If that file is missing, it falls back to `yolov8n.pt`, which is a general object detector. If neither is found, the app uses the entire frame as the face crop instead.

For Stage 2, the app searches a list of common paths in this order:

1. `models/emotion_classifier/best.pt`
2. `models/emotion_classifier/best.onnx`
3. `runs/detect/emotion_classifier/weights/best.pt`
4. `runs/detect/emotion_classifier/weights/best.onnx`
5. Several other fallback paths
6. `best.pt` and `best.onnx` in the current directory

If no model is found, the app opens but cannot make predictions and shows a warning. You must run `2_train_model.py` first.

### The Detection Pipeline

Every 30 milliseconds (approximately 33 frames per second at maximum), the app reads a new frame from the webcam and passes it through the two-stage pipeline.

**Stage 1 — Face Detection:** The face detector runs on the full 640x640-resized frame and returns a list of bounding boxes with confidence scores. Non-Maximum Suppression (NMS) is then applied to remove duplicates. NMS works by sorting all boxes by confidence, keeping the highest one, then discarding any other box that overlaps it by more than 40% (measured as Intersection over Union). The remaining boxes are the final face detections.

**Bounding Box Tightening:** YOLO's face detector often produces boxes that include the neck and shoulders. The app shrinks these to focus on just the head. It first makes the box square by using the center point and half-size based on the shorter dimension multiplied by `SQUARE_FACTOR` (0.53). It then trims 4% from the sides, 2% from the top, and 10% from the bottom. Finally, it clamps the coordinates to stay inside the frame boundary.

**Stage 2 — Emotion Classification:** Each tightened face crop is passed to the emotion classifier. The model returns a probability for each of the 7 emotion classes. The class with the highest probability is taken as the prediction. If that probability is below the confidence threshold set by the slider, the detection is discarded and nothing is drawn for that face.

### Drawing the Results

For each detection that passes the threshold, the app draws a colored bounding box on the frame. The color is unique per emotion (e.g., red for angry, yellow for happy). An emoji badge is drawn in the top-left corner of the box, the emotion name is shown below it, and a confidence percentage pill is shown in the top-right corner.

### Statistics

The sidebar keeps a running count of every emotion that has been detected since the camera started. After each frame with at least one detection, these counts are converted to percentages and used to update the progress bars. The dominant emotion (highest confidence in the current frame) is shown as a large emoji in the "Current Emotion" box at the bottom of the sidebar.

### Image Loading

The "Load Image" button opens a file dialog. The selected image is passed through the same `_detect` function used for the camera loop, so the exact same two-stage pipeline and drawing logic applies to static images.

---

## 5. Key Design Decisions

**Why YOLOv8s-cls instead of YOLOv8n-cls?** The small variant has more parameters than the nano variant, which gives significantly better classification accuracy on a 7-class problem at only a modest increase in inference time. On an RTX 3070, the speed difference is negligible.

**Why ONNX?** ONNX Runtime with GPU support runs inference faster than standard PyTorch for a fixed-shape input because it performs graph optimizations at load time. It is also a smaller dependency to ship.

**Why 224x224?** This is the standard input size for image classification networks inherited from ImageNet training. It is large enough to capture facial detail but small enough to process quickly.

**Why a confidence threshold slider?** Emotion recognition is inherently ambiguous. Neutral expressions can look like sadness or fear depending on lighting. Allowing the user to raise the threshold means only high-confidence predictions are shown, which reduces embarrassing misclassifications at the cost of fewer labels appearing.

---

## 6. Troubleshooting

**"Model: not found" when starting the app.** The training script has not been run yet, or it did not finish successfully. Check that `models/emotion_classifier/best.pt` exists. If it does not, run `python 2_train_model.py` and let it complete fully.

**Low FPS.** Make sure the ONNX model is being used rather than the `.pt` model. ONNX Runtime is faster for inference. Also check that `torch.cuda.is_available()` returns `True` in Python; if the GPU is not being used, training and inference will be much slower.

**Training fails with a dataset error.** YOLO's classification trainer requires the folder structure to be exactly right. Each emotion class must have its own subfolder under `train/` and `val/`. The script creates this automatically, but if the copy step fails (e.g., wrong path to RAF-DB), the output folder will be empty and training will error. Check the path you are passing with `--data` and make sure it points to the folder that contains `train/` and `test/` as subfolders.

**Predictions are wrong or all the same emotion.** This usually means the model did not train for enough epochs, or the dataset has a severe class imbalance that the augmentation did not fully compensate for. Try increasing `--epochs` to 100 or 120, or check the confusion matrix saved in `runs/detect/emotion_classifier/` after training.
