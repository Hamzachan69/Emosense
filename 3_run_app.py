"""
EmoSense by Hamza Khan - v6 (RAF-DB Two-Stage Edition)

This is the main application. It opens a window with a live camera feed
and detects faces in real time. For each face found, it predicts the
person's emotion and displays it on screen with a confidence score.

The pipeline has two stages:
  Stage 1 - yolov8n-face  : finds faces in the frame and draws bounding boxes
  Stage 2 - yolov8s-cls   : classifies the emotion from each cropped face
"""

import os
import sys
import time
import argparse
from pathlib import Path

# Try to import all required libraries. If any are missing, print a helpful message and stop.
try:
    import cv2
    import numpy as np
    import customtkinter as ctk
    from PIL import Image, ImageTk, ImageDraw, ImageFont
    from ultralytics import YOLO
except ImportError as e:
    print(f"\nMissing package: {e.name}")
    print("   Run: pip install ultralytics customtkinter pillow opencv-python\n")
    sys.exit(1)

# --- CONSTANTS ---

# The 7 emotion class names in alphabetical order.
# YOLO assigns class indices based on alphabetical folder order, so this order matters.
# Index 0=angry, 1=disgust, 2=fear, 3=happy, 4=neutral, 5=sad, 6=surprise
EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# Hex color codes for each emotion, used in the GUI labels and progress bars.
EMOTION_COLORS_HEX = {
    "angry":    "#FF2222",
    "disgust":  "#00AAAA",
    "fear":     "#AA44CC",
    "happy":    "#FFD700",
    "neutral":  "#00CC44",
    "sad":      "#4488FF",
    "surprise": "#FF8800",
}

# BGR color values for the same emotions, used when drawing boxes on the camera frame with OpenCV.
# OpenCV uses BGR (Blue, Green, Red) instead of the usual RGB order.
EMOTION_COLORS_BGR = {
    "angry":    (34,  34,  255),
    "disgust":  (170, 170,   0),
    "fear":     (204,  68, 170),
    "happy":    (0,  215,  255),
    "neutral":  (0,  204,   68),
    "sad":      (255, 136,  68),
    "surprise": (0,  136,  255),
}

# Emoji characters displayed next to each emotion label on the camera feed.
EMOJI_MAP = {
    "angry":    "😠",
    "disgust":  "🤢",
    "fear":     "😨",
    "happy":    "😊",
    "neutral":  "😐",
    "sad":      "😢",
    "surprise": "😮",
}

# The pixel size of the camera display area inside the window.
CAM_W, CAM_H = 860, 484

# Minimum confidence for the face detector (Stage 1).
# Kept low (0.30) so that smaller or partially visible faces are still detected.
FACE_CONF = 0.30

# Default minimum confidence for the emotion classifier (Stage 2).
# Only predictions above this threshold are displayed on screen.
DEFAULT_CONF = 0.35

# --- BOUNDING BOX TIGHTENING PARAMETERS ---
# The face detector often draws a box that includes the neck and shoulders.
# These values shrink the box to fit more tightly around the head.

# SQUARE_FACTOR: controls how large the square crop is relative to the detected face.
# A lower value means a tighter crop. 0.53 works well for most webcam distances.
SQUARE_FACTOR = 0.53

# After squaring the box, trim additional background from each side.
PAD_SIDES  = 0.04   # trim a small amount from the left and right
PAD_TOP    = 0.02   # trim a small amount from the top (reduces extra space above head)
PAD_BOTTOM = 0.10   # trim more from the bottom (removes chin and neck area)

# Set the visual theme for the application window.
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


# --- HELPER FUNCTIONS ---

def draw_emoji(frame: np.ndarray, emoji: str,
               x: int, y: int, size: int = 26) -> np.ndarray:
    # Draws an emoji character onto an OpenCV frame at position (x, y).
    # This is done by temporarily converting the frame to a PIL image,
    # drawing the text with a system emoji font, then converting back to OpenCV.
    try:
        pil  = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil)
        font = None

        # Try to find an emoji-capable font file on the current operating system.
        for fp in [
            "C:/Windows/Fonts/seguiemj.ttf",       # Windows emoji font
            "C:/Windows/Fonts/segoeui.ttf",         # Windows fallback
            "/System/Library/Fonts/Apple Color Emoji.ttc",   # macOS emoji font
            "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf",  # Linux emoji font
        ]:
            if os.path.exists(fp):
                try:
                    font = ImageFont.truetype(fp, size)
                    break
                except Exception:
                    pass

        # If no emoji font was found, fall back to the default PIL font.
        if font is None:
            font = ImageFont.load_default()

        draw.text((x, y), emoji, font=font, embedded_color=True)
        # Convert the modified PIL image back to an OpenCV BGR frame.
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    except Exception:
        # If drawing fails for any reason, return the frame unchanged.
        return frame


def nms_boxes(boxes, iou_thresh=0.40):
    """
    Non-Maximum Suppression (NMS) -- removes duplicate face detections.

    When the face detector finds the same face multiple times with slightly
    different box positions, NMS keeps only the highest-confidence detection
    and removes any others that overlap it by more than iou_thresh.
    """
    if not boxes:
        return []

    # Extract coordinates and confidence scores into numpy arrays.
    coords = np.array([[b[0], b[1], b[2], b[3]] for b in boxes], np.float32)
    scores = np.array([b[4] for b in boxes], np.float32)
    x1, y1, x2, y2 = coords[:, 0], coords[:, 1], coords[:, 2], coords[:, 3]

    # Calculate the area of each bounding box.
    areas  = (x2 - x1) * (y2 - y1)

    # Sort detections from highest confidence to lowest.
    order  = scores.argsort()[::-1]
    keep   = []

    while order.size:
        i = order[0]   # the current best detection
        keep.append(i)

        if order.size == 1:
            break

        # Calculate the overlap (IoU) between the best box and all remaining boxes.
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou   = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)

        # Discard any boxes that overlap the current best box by more than the threshold.
        order = order[1:][iou < iou_thresh]

    return keep


def tighten_box(x1, y1, x2, y2, frame_h, frame_w):
    """
    Converts an oversized YOLO face bounding box into a tight square around the head.

    Step 1 - Square:  find the center of the box and re-expand it symmetrically
                      using SQUARE_FACTOR so the result is a square.
    Step 2 - Trim:    shave off padding fractions from each side to reduce background.
    Step 3 - Clamp:   make sure the final coordinates stay within the frame boundaries.
    """
    w  = x2 - x1
    h  = y2 - y1
    cx = (x1 + x2) // 2   # horizontal center of the original box
    cy = (y1 + y2) // 2   # vertical center of the original box

    # Step 1: Create a square box centered on (cx, cy).
    half = int(min(w, h) * SQUARE_FACTOR)
    x1, x2 = cx - half, cx + half
    y1, y2 = cy - half, cy + half

    # Step 2: Trim background from each side of the square.
    sw = x2 - x1   # current width of the square
    sh = y2 - y1   # current height of the square
    x1 += int(sw * PAD_SIDES);   x2 -= int(sw * PAD_SIDES)
    y1 += int(sh * PAD_TOP);     y2 -= int(sh * PAD_BOTTOM)

    # Step 3: Clamp to frame boundaries so we don't read outside the image.
    x1 = max(0, x1);        y1 = max(0, y1)
    x2 = min(frame_w, x2);  y2 = min(frame_h, y2)

    return x1, y1, x2, y2


# --- MAIN APPLICATION CLASS ---

class EmotionDetectorApp:

    def __init__(self, model_path=None):
        # Store the optional custom model path (can be passed via command line).
        self.face_model      = None    # Stage 1: face detector model
        self.emo_model       = None    # Stage 2: emotion classifier model
        self.cls_model_path  = model_path
        self.camera_running  = False   # whether the webcam is currently active
        self.cap             = None    # OpenCV video capture object
        self.emotion_counts  = {e: 0 for e in EMOTIONS}  # running tally of each emotion seen
        self.current_emotion = None    # the dominant emotion in the current frame

        # Build the window, lay out the UI, then load the models.
        self._build_window()
        self._build_ui()
        self._load_models()

    # -- WINDOW --
    def _build_window(self):
        # Create the main application window and center it on the screen.
        self.root = ctk.CTk()
        self.root.title("EmoSense -- AI Emotion Recognition  (by HamzaChan)")
        self.root.geometry("1300x640")
        self.root.minsize(1100, 580)
        self.root.configure(fg_color="#0d1117")
        self.root.update_idletasks()

        # Calculate screen and window dimensions to center the window.
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        ww = self.root.winfo_width()
        wh = self.root.winfo_height()
        self.root.geometry(f"{ww}x{wh}+{(sw - ww) // 2}+{(sh - wh) // 2}")

    # -- FULL UI --
    def _build_ui(self):
        # Configure the grid layout: the camera panel expands, the sidebar has a fixed width.
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, minsize=310, weight=0)

        # Build each section of the UI.
        self._build_titlebar()
        self._build_camera_panel()
        self._build_sidebar()

    # -- TITLE BAR --
    def _build_titlebar(self):
        # Creates the top bar that spans the full width of the window.
        BAR_H = 68
        bar = ctk.CTkFrame(self.root, height=BAR_H,
                           fg_color="#111827", corner_radius=0)
        bar.grid(row=0, column=0, columnspan=2, sticky="ew")
        bar.grid_propagate(False)

        import tkinter as tk

        # Use a raw tkinter Canvas inside the frame for custom gradient drawing.
        cv = tk.Canvas(bar, height=BAR_H, bg="#111827",
                       highlightthickness=0, bd=0)
        cv.pack(fill="both", expand=True)

        def _draw(event=None):
            # Redraws the title bar content. Called on window resize and at startup.
            cv.delete("all")
            W = cv.winfo_width()
            H = BAR_H

            # Draw a green accent strip on the left edge.
            cv.create_rectangle(0, 0, 7, H, fill="#00D4AA", outline="")

            # Draw a subtle vertical gradient across the rest of the bar.
            for i in range(H):
                alpha = int(255 * (0.06 * i / H))
                col   = f"#{alpha:02x}{alpha:02x}{(alpha + 8):02x}"
                try:
                    cv.create_line(7, i, W, i, fill=col)
                except Exception:
                    pass

            # Draw the app name and subtitle text.
            cv.create_text(30, H // 2, text="🧠",
                           font=("Segoe UI Emoji", 24),
                           fill="#00D4AA", anchor="w")
            cv.create_text(68, H // 2 - 10, text="  EmoSense",
                           font=("Georgia", 20, "bold"),
                           fill="#00D4AA", anchor="w")
            cv.create_text(68, H // 2 + 12, text="   by HamzaChan",
                           font=("Georgia", 11, "italic"),
                           fill="#7788AA", anchor="w")

            # Draw a small description on the right side of the bar.
            cv.create_text(W - 16, H // 2,
                           text="Real-time Facial Emotion Recognition  .  "
                                "Powered by YOLOv8  .  RAF-DB",
                           font=("Consolas", 9),
                           fill="#334455", anchor="e")

        # Redraw the title bar when the window is resized.
        cv.bind("<Configure>", _draw)
        # Also draw it shortly after startup to ensure the window width is known.
        self.root.after(50, _draw)

    # -- CAMERA PANEL --
    def _build_camera_panel(self):
        # The left panel where the camera feed or loaded image is displayed.
        frame = ctk.CTkFrame(self.root, fg_color="#161b27", corner_radius=10)
        frame.grid(row=1, column=0, sticky="nsew", padx=(10, 5), pady=10)
        frame.grid_rowconfigure(1, weight=1)
        frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(frame, text="  Camera View",
                     font=("Segoe UI", 13, "bold"),
                     text_color="#8899BB").grid(row=0, column=0, pady=(8, 2))

        # The main label that holds the camera image. Starts with a placeholder message.
        self.cam_lbl = ctk.CTkLabel(
            frame,
            text="Click  Start Camera  to begin\nor  Load Image  to analyse a photo",
            font=("Segoe UI", 12), text_color="#334455",
            width=CAM_W, height=CAM_H)
        self.cam_lbl.grid(row=1, column=0, padx=8, pady=(2, 6))

        # A small status text at the bottom of the camera panel.
        self.status_lbl = ctk.CTkLabel(frame, text="  Ready",
                                       font=("Segoe UI", 10),
                                       text_color="#334455")
        self.status_lbl.grid(row=2, column=0, pady=(0, 6))

    # -- SIDEBAR --
    def _build_sidebar(self):
        # The right-side panel containing controls, emotion stats, and the current emotion display.
        sb = ctk.CTkFrame(self.root, fg_color="#161b27",
                          corner_radius=10, width=310)
        sb.grid(row=1, column=1, sticky="nsew", padx=(5, 10), pady=10)
        sb.grid_propagate(False)
        sb.grid_columnconfigure(0, weight=1)
        sb.grid_rowconfigure(5, weight=1)  # row 5 (the stats panel) expands to fill available space

        ctk.CTkLabel(sb, text="  Controls",
                     font=("Segoe UI", 13, "bold"),
                     text_color="#AABBCC").grid(
            row=0, column=0, pady=(12, 4), padx=14, sticky="w")

        # Button row: Start/Stop Camera and Load Image side by side.
        bf = ctk.CTkFrame(sb, fg_color="transparent")
        bf.grid(row=1, column=0, sticky="ew", padx=14, pady=(0, 4))
        bf.grid_columnconfigure((0, 1), weight=1)

        self.btn_cam = ctk.CTkButton(
            bf, text="  Start Camera", command=self._toggle_cam,
            height=34, font=("Segoe UI", 11, "bold"),
            fg_color="#1A6B4A", hover_color="#22996A")
        self.btn_cam.grid(row=0, column=0, padx=(0, 3), sticky="ew")

        ctk.CTkButton(
            bf, text="  Load Image", command=self._load_image,
            height=34, font=("Segoe UI", 11, "bold"),
            fg_color="#1A3A6B", hover_color="#225A9A").grid(
            row=0, column=1, padx=(3, 0), sticky="ew")

        # Confidence threshold slider -- controls how certain the model must be before
        # showing a label. Higher values = fewer but more reliable predictions.
        sf = ctk.CTkFrame(sb, fg_color="transparent")
        sf.grid(row=2, column=0, sticky="ew", padx=14, pady=(0, 4))
        sf.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(sf, text="Confidence Threshold:",
                     font=("Segoe UI", 10),
                     text_color="#8899BB").grid(
            row=0, column=0, columnspan=2, sticky="w")

        self.conf_slider = ctk.CTkSlider(
            sf, from_=0.10, to=0.90, number_of_steps=80,
            command=self._on_conf,
            button_color="#00D4AA", button_hover_color="#00FFCC",
            progress_color="#00D4AA")
        self.conf_slider.set(DEFAULT_CONF)
        self.conf_slider.grid(row=1, column=0, sticky="ew")

        # Label that shows the current slider value as a number.
        self.conf_val = ctk.CTkLabel(sf, text=f"{DEFAULT_CONF:.2f}",
                                     font=("Consolas", 10, "bold"),
                                     text_color="#00D4AA", width=36)
        self.conf_val.grid(row=1, column=1, padx=(4, 0))

        # Horizontal divider line.
        ctk.CTkFrame(sb, height=1, fg_color="#1E2D3D").grid(
            row=3, column=0, sticky="ew", padx=14, pady=4)

        ctk.CTkLabel(sb, text="  Emotion Statistics",
                     font=("Segoe UI", 13, "bold"),
                     text_color="#AABBCC").grid(
            row=4, column=0, pady=(0, 4), padx=14, sticky="w")

        # Scrollable area containing one progress bar row per emotion.
        bars = ctk.CTkScrollableFrame(sb, fg_color="#0f1520", corner_radius=6)
        bars.grid(row=5, column=0, sticky="nsew", padx=14, pady=(0, 6))
        bars.grid_columnconfigure(0, weight=1)

        # Dictionaries to hold references to each emotion's bar and percentage label.
        self.emo_bars = {}
        self.emo_lbls = {}

        # Create one row per emotion with: emoji + name, a progress bar, and a percentage label.
        for i, emo in enumerate(EMOTIONS):
            rf = ctk.CTkFrame(bars, fg_color="transparent")
            rf.grid(row=i, column=0, sticky="ew", pady=2)
            rf.grid_columnconfigure(1, weight=1)

            ctk.CTkLabel(rf,
                         text=f"{EMOJI_MAP[emo]}  {emo}",
                         font=("Segoe UI Emoji", 11),
                         width=110, anchor="w",
                         text_color="#AABBCC").grid(row=0, column=0, sticky="w")

            bar = ctk.CTkProgressBar(rf, height=14,
                                     progress_color=EMOTION_COLORS_HEX[emo],
                                     fg_color="#1A2233")
            bar.set(0)
            bar.grid(row=0, column=1, sticky="ew", padx=5)

            lbl = ctk.CTkLabel(rf, text="0%",
                               font=("Consolas", 10, "bold"),
                               width=38, text_color="#AABBCC")
            lbl.grid(row=0, column=2)

            self.emo_bars[emo] = bar
            self.emo_lbls[emo] = lbl

        # "Current Emotion" display box -- shows a large emoji and name for the dominant emotion.
        dom = ctk.CTkFrame(sb, fg_color="#0b1018",
                           corner_radius=10, height=120)
        dom.grid(row=6, column=0, sticky="ew", padx=14, pady=(0, 6))
        dom.grid_propagate(False)
        dom.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(dom, text="Current Emotion",
                     font=("Segoe UI", 9),
                     text_color="#334455").grid(row=0, column=0, pady=(8, 0))

        # Large emoji updated in real time to show the current dominant emotion.
        self.dom_emoji = ctk.CTkLabel(dom, text="--",
                                      font=("Segoe UI Emoji", 40),
                                      text_color="#FFFFFF")
        self.dom_emoji.grid(row=1, column=0)

        # Text label below the emoji showing the emotion name.
        self.dom_name = ctk.CTkLabel(dom, text="waiting...",
                                     font=("Georgia", 13, "bold"),
                                     text_color="#445566")
        self.dom_name.grid(row=2, column=0, pady=(0, 8))

        # Bottom row showing FPS, detection count, and model status.
        ff = ctk.CTkFrame(sb, fg_color="transparent")
        ff.grid(row=7, column=0, sticky="ew", padx=14, pady=(0, 10))
        ff.grid_columnconfigure(1, weight=1)

        self.fps_lbl = ctk.CTkLabel(ff, text="FPS: --",
                                    font=("Consolas", 10),
                                    text_color="#334455")
        self.fps_lbl.grid(row=0, column=0, sticky="w")

        self.det_lbl = ctk.CTkLabel(ff, text="Detections: 0",
                                    font=("Consolas", 10),
                                    text_color="#334455")
        self.det_lbl.grid(row=0, column=1, sticky="e")

        self.mdl_lbl = ctk.CTkLabel(ff, text="Model: not loaded",
                                    font=("Consolas", 9),
                                    text_color="#223344")
        self.mdl_lbl.grid(row=1, column=0, columnspan=2, sticky="w")

    # -- MODEL LOADING --
    def _load_models(self):
        # Load Stage 1: the face detector model.
        print("\nLoading Stage-1 face detector (yolov8n-face.pt)...")
        try:
            self.face_model = YOLO("yolov8n-face.pt")
            print("  Face detector ready")
        except Exception as e:
            print(f"  Could not load yolov8n-face.pt: {e}")
            try:
                # Fall back to the general-purpose YOLOv8n if the face-specific model is missing.
                self.face_model = YOLO("yolov8n.pt")
                print("  Fallback: using yolov8n.pt (boxes may be less tight)")
            except Exception:
                self.face_model = None
                print("  No face model loaded -- will use the whole frame as the face crop")

        # Load Stage 2: the emotion classifier model.
        # Search several common locations for the trained model file.
        search_paths = [
            "models/emotion_classifier/best.pt",
            "models/emotion_classifier/best.onnx",
            "runs/detect/emotion_classifier/weights/best.pt",
            "runs/detect/emotion_classifier/weights/best.onnx",
            "models/emotion_detector/best.pt",
            "models/emotion_detector/best.onnx",
            "runs/classify/emotion_classifier/weights/best.pt",
            "best.pt",
            "best.onnx",
        ]

        cls_path = self.cls_model_path  # use the path passed on the command line if provided
        if not cls_path:
            # Search each location in order and use the first one found.
            for p in search_paths:
                if os.path.exists(p):
                    cls_path = p
                    break

        if not cls_path or not os.path.exists(cls_path):
            # No emotion model found -- the app will open but cannot make predictions.
            self.mdl_lbl.configure(text="Emotion model: not found")
            print("\nNo emotion model found.")
            print("   Expected at: models/emotion_classifier/best.pt")
            print("   Run training first:")
            print("   python 2_train_model.py\n")
            return

        try:
            # Load the emotion classifier model.
            self.emo_model = YOLO(cls_path)
            short = Path(cls_path).name
            self.mdl_lbl.configure(text=f"Model: {short}",
                                   text_color="#00CC88")
            print(f"  Emotion classifier loaded: {cls_path}")
        except Exception as e:
            self.mdl_lbl.configure(text="Emotion model: load failed")
            print(f"Error loading emotion model: {e}")

    # -- CAMERA TOGGLE --
    def _toggle_cam(self):
        # Switches the camera on or off depending on its current state.
        if self.camera_running:
            self._stop_cam()
        else:
            self._start_cam()

    def _start_cam(self):
        # Opens the default webcam and starts the detection loop.
        if self.emo_model is None:
            self._err("No emotion model loaded!\nRun: python 2_train_model.py")
            return

        self.cap = cv2.VideoCapture(0)  # 0 = the first available webcam
        if not self.cap.isOpened():
            self._err("Could not open webcam!")
            return

        # Request a higher resolution from the webcam if it supports it.
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        self.camera_running = True
        self.btn_cam.configure(text="  Stop Camera",
                               fg_color="#6B1A1A", hover_color="#9A2222")
        self.status_lbl.configure(text="  Camera running...",
                                  text_color="#00CC88")
        self._tick()  # start the frame-by-frame processing loop

    def _stop_cam(self):
        # Stops the camera and releases the hardware.
        self.camera_running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.btn_cam.configure(text="  Start Camera",
                               fg_color="#1A6B4A", hover_color="#22996A")
        self.status_lbl.configure(text="  Stopped", text_color="#445566")

    def _tick(self):
        # Called repeatedly to process one frame at a time.
        # After each frame, it schedules itself to run again after 30ms (~33 FPS max).
        if not self.camera_running:
            return

        ret, frame = self.cap.read()
        if not ret:
            # If reading a frame failed (e.g. camera disconnected), stop.
            self._stop_cam()
            return

        t0  = time.time()
        out = self._detect(frame)          # run the two-stage detection pipeline
        fps = 1.0 / max(time.time() - t0, 0.001)  # calculate frames per second
        self._show(out)                    # display the annotated frame in the window
        self.fps_lbl.configure(text=f"FPS: {fps:.1f}")
        self.root.after(30, self._tick)    # schedule the next frame in 30ms

    # -- TWO-STAGE DETECTION --
    def _detect(self, frame: np.ndarray) -> np.ndarray:
        """
        Runs the full two-stage pipeline on a single frame.

        Stage 1: Uses yolov8n-face to find face bounding boxes in the frame.
        Stage 2: Crops each face and runs the emotion classifier on the crop.

        Returns the frame with boxes, labels, and confidence scores drawn on it.
        """
        conf_thresh = self.conf_slider.get()  # get the current slider value
        fh, fw = frame.shape[:2]
        annotated = frame.copy()  # work on a copy so the original is not modified

        top_emo  = None    # emotion with the highest confidence in this frame
        top_conf = 0.0
        finals   = []      # list of (x1, y1, x2, y2, confidence, emotion_name) for drawing

        # Stage 1: Run the face detector.
        if self.face_model is not None:
            face_results = self.face_model(
                frame,
                conf    = FACE_CONF,
                iou     = 0.40,
                imgsz   = 640,   # YOLO processes images internally at 640x640
                verbose = False,
            )[0]

            # Collect all detected face boxes into a list.
            raw = []
            if face_results.boxes is not None:
                for box in face_results.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    fc = float(box.conf[0])
                    raw.append((x1, y1, x2, y2, fc))

            # Apply NMS to remove duplicate detections of the same face.
            keep  = nms_boxes(raw, 0.40)
            faces = [raw[k] for k in keep]
        else:
            # If no face model is loaded, treat the entire frame as one big face.
            faces = [(0, 0, fw, fh, 1.0)]

        # Update the detection count label in the sidebar.
        self.det_lbl.configure(text=f"Detections: {len(faces)}")

        # Stage 2: Classify the emotion for each detected face.
        for (x1, y1, x2, y2, face_conf) in faces:

            # Shrink the bounding box to focus tightly on the face.
            x1, y1, x2, y2 = tighten_box(x1, y1, x2, y2, fh, fw)

            # Crop the face region out of the frame.
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue  # skip if the crop is empty (can happen at frame edges)

            # Run the emotion classifier on the cropped face.
            emo_result = self.emo_model(crop, verbose=False)
            probs      = emo_result[0].probs
            pred_cls   = int(probs.top1)        # index of the top prediction
            pred_conf  = float(probs.top1conf)  # confidence score for that prediction

            # Skip this face if its confidence is below the threshold slider.
            if pred_conf < conf_thresh:
                continue

            # Get the emotion name from the model's class list.
            # YOLO sorts class names alphabetically (angry, disgust, fear, happy, neutral, sad, surprise).
            cls_names = self.emo_model.names
            pred_name = cls_names.get(pred_cls,
                        EMOTIONS[pred_cls] if pred_cls < len(EMOTIONS) else "?")
            pred_name = pred_name.lower()

            finals.append((x1, y1, x2, y2, pred_conf, pred_name))

            # Track which emotion has the highest confidence in this frame.
            if pred_conf > top_conf:
                top_conf = pred_conf
                top_emo  = pred_name

        # Draw the detection results onto the annotated frame.
        for (x1, y1, x2, y2, confidence, emo) in finals:
            color = EMOTION_COLORS_BGR.get(emo, (200, 200, 200))
            emoji = EMOJI_MAP.get(emo, "?")

            # Draw the main bounding box around the face.
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Draw a small square badge in the top-left corner of the box to hold the emoji.
            BADGE = 42
            bx1, by1 = x1 + 3, y1 + 3
            bx2, by2 = bx1 + BADGE, by1 + BADGE
            roi = annotated[by1:by2, bx1:bx2]
            if roi.size > 0:
                # Darken the badge background so the emoji is easier to read.
                annotated[by1:by2, bx1:bx2] = (roi * 0.35).astype(np.uint8)
            cv2.rectangle(annotated, (bx1, by1), (bx2, by2), color, 1)
            annotated = draw_emoji(annotated, emoji, bx1 + 3, by1 + 2, 26)

            # Draw the emotion name text just below the emoji badge.
            ny = by2 + 11
            (tw, th), _ = cv2.getTextSize(emo, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
            cv2.rectangle(annotated, (bx1, ny - th - 3), (bx1 + tw + 5, ny + 2),
                          (8, 12, 22), -1)  # dark background behind text for readability
            cv2.putText(annotated, emo, (bx1 + 2, ny),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, color, 1, cv2.LINE_AA)

            # Draw a small confidence percentage in the top-right corner of the bounding box.
            ct = f"{confidence:.0%}"
            (cw, ch), _ = cv2.getTextSize(ct, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            cv2.rectangle(annotated, (x2 - cw - 7, y1), (x2, y1 + ch + 5),
                          color, -1)  # filled rectangle as background for the text
            cv2.putText(annotated, ct, (x2 - cw - 3, y1 + ch + 1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1,
                        cv2.LINE_AA)

            # Increment the count for this emotion in the running statistics.
            if emo in self.emotion_counts:
                self.emotion_counts[emo] += 1

        # Update the "Current Emotion" display in the sidebar.
        self._update_dom(top_emo)

        # Update the progress bars if at least one detection was made this frame.
        if finals:
            self._update_bars()

        return annotated

    def _update_dom(self, emo):
        # Updates the large emoji and name in the "Current Emotion" box.
        if emo and emo in EMOTION_COLORS_HEX:
            c = EMOTION_COLORS_HEX[emo]
            self.dom_emoji.configure(text=EMOJI_MAP.get(emo, "?"), text_color=c)
            self.dom_name.configure(text=emo.capitalize(), text_color=c)
        else:
            # Show a dash and a dim message when no face is detected.
            self.dom_emoji.configure(text="--", text_color="#FFFFFF")
            self.dom_name.configure(text="no face detected", text_color="#334455")

    # -- DISPLAY --
    def _show(self, frame: np.ndarray):
        # Converts an OpenCV BGR frame to a CTkImage and shows it in the camera label.
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)   # OpenCV is BGR, PIL/CTk needs RGB
        h, w   = rgb.shape[:2]

        # Scale the frame to fit inside the display area while preserving aspect ratio.
        scale  = min(CAM_W / w, CAM_H / h)
        nw, nh = int(w * scale), int(h * scale)
        res    = cv2.resize(rgb, (nw, nh))

        # Create a blank black canvas of the display area size and paste the scaled frame in the center.
        canvas = np.zeros((CAM_H, CAM_W, 3), dtype=np.uint8)
        ox = (CAM_W - nw) // 2   # horizontal offset to center the image
        oy = (CAM_H - nh) // 2   # vertical offset to center the image
        canvas[oy:oy + nh, ox:ox + nw] = res

        # Convert to a CTkImage and update the label.
        pil     = Image.fromarray(canvas)
        ctk_img = ctk.CTkImage(light_image=pil, dark_image=pil,
                               size=(CAM_W, CAM_H))
        self.cam_lbl.configure(image=ctk_img, text="")
        self.cam_lbl.image = ctk_img  # keep a reference so Python doesn't garbage-collect it

    # -- LOAD IMAGE --
    def _load_image(self):
        # Opens a file picker dialog and runs the detection pipeline on a static image.
        if self.emo_model is None:
            self._err("No emotion model loaded!\nRun: python 2_train_model.py")
            return

        from tkinter import filedialog
        fp = filedialog.askopenfilename(
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp")])
        if not fp:
            return  # user cancelled the dialog

        img = cv2.imread(fp)
        if img is None:
            self._err("Could not load image.")
            return

        out = self._detect(img)    # run the same two-stage pipeline as the camera loop
        self._show(out)
        self.status_lbl.configure(text=f"  {Path(fp).name}",
                                  text_color="#AABBCC")

    # -- STATISTICS BARS --
    def _update_bars(self):
        # Recalculates the percentage for each emotion and updates the progress bars.
        total = sum(self.emotion_counts.values())
        if not total:
            return

        for emo in EMOTIONS:
            pct = self.emotion_counts[emo] / total * 100
            self.emo_bars[emo].set(pct / 100)                  # progress bar expects 0.0 to 1.0
            self.emo_lbls[emo].configure(text=f"{pct:.1f}%")   # update the percentage text label

    def _on_conf(self, val):
        # Called every time the confidence slider is moved. Updates the numeric display.
        self.conf_val.configure(text=f"{float(val):.2f}")

    # -- ERROR DIALOG --
    def _err(self, msg):
        # Shows a small error popup window with a message and an OK button.
        w = ctk.CTkToplevel(self.root)
        w.title("Error")
        w.geometry("360x130")
        w.grab_set()  # block interaction with the main window until this is dismissed
        ctk.CTkLabel(w, text=f"  {msg}",
                     font=("Segoe UI", 12), wraplength=320).pack(
            expand=True, pady=16)
        ctk.CTkButton(w, text="OK", command=w.destroy, width=70).pack(
            pady=(0, 12))

    # -- RUN --
    def run(self):
        # Prints a startup summary and launches the main window event loop.
        print("\n" + "=" * 55)
        print("  EmoSense  by HamzaChan  --  v6 (RAF-DB Edition)")
        print("  Two-stage: face detect -> emotion classify")
        print(f"  Box factor : SQUARE_FACTOR={SQUARE_FACTOR}")
        print(f"  Confidence : default {DEFAULT_CONF}")
        print("=" * 55 + "\n")
        self.root.mainloop()  # starts the GUI and keeps it running until the window is closed


# --- ENTRY POINT ---

def main():
    # Parse optional command-line arguments.
    p = argparse.ArgumentParser(description="EmoSense -- Emotion Recognition App v6")
    p.add_argument("--model", default=None,
                   help="Path to emotion classifier model (.pt or .onnx). "
                        "If omitted, the app searches common locations automatically.")
    args = p.parse_args()

    # Create the app and start the window.
    EmotionDetectorApp(model_path=args.model).run()


if __name__ == "__main__":
    main()
