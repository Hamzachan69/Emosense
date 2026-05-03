# EmoSense - Face Emotion Recognition

EmoSense is a real-time emotion detection system. You point your webcam at a person and it automatically finds their face, draws a box around it, and tells you what emotion they are showing. It can detect seven emotions: angry, disgust, fear, happy, neutral, sad, and surprise.

The system works in two steps every single frame. First, a face-detection model finds where the face is in the image. Second, an emotion-classification model looks at just the face area and decides which emotion it sees. Both steps happen fast enough to run live on a webcam.

---

## What You Need Before Starting

- Python 3.10 or newer
- An NVIDIA GPU (the system is built for an RTX 3070, but any CUDA-capable GPU works)
- The RAF-DB dataset placed in `face_emotion/data/RAF DB/DATASET/`
- The `yolov8n-face.pt` face detector file in the project root folder

If you do not have a GPU, the system will fall back to running on CPU, but training will be much slower (hours instead of minutes).

---

## How to Use It

There are three steps. You only need to do steps 1 and 2 once. After that, you just run step 3 every time you want to use the app.

**Step 1 — Install dependencies**

```
python 1_install_requirements.py
```

This installs all the required Python packages including the correct GPU-enabled version of PyTorch.

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
