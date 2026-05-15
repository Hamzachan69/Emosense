"""
EmoSense by Hamza Khan - v2.0 (Deployment Edition)

Desktop application with welcome splash screen.
Two-stage pipeline:
  Stage 1 - yolov8n-face  : finds faces in the frame and draws bounding boxes
  Stage 2 - yolov8s-cls   : classifies the emotion from each cropped face
"""

import os
import sys
import time
import argparse
from pathlib import Path

try:
    import cv2
    import numpy as np
    import customtkinter as ctk
    import tkinter as tk
    from PIL import Image, ImageTk, ImageDraw, ImageFont
    from ultralytics import YOLO
except ImportError as e:
    print(f"\nMissing package: {e.name}")
    print("   Run: pip install ultralytics customtkinter pillow opencv-python\n")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

EMOTION_COLORS_HEX = {
    "angry":    "#FF2222",
    "disgust":  "#00AAAA",
    "fear":     "#AA44CC",
    "happy":    "#FFD700",
    "neutral":  "#00CC44",
    "sad":      "#4488FF",
    "surprise": "#FF8800",
}

EMOTION_COLORS_BGR = {
    "angry":    (34,  34,  255),
    "disgust":  (170, 170,   0),
    "fear":     (204,  68, 170),
    "happy":    (0,  215,  255),
    "neutral":  (0,  204,   68),
    "sad":      (255, 136,  68),
    "surprise": (0,  136,  255),
}

EMOJI_MAP = {
    "angry":    "😠",
    "disgust":  "🤢",
    "fear":     "😨",
    "happy":    "😊",
    "neutral":  "😐",
    "sad":      "😢",
    "surprise": "😮",
}

CAM_W, CAM_H    = 860, 484
FACE_CONF       = 0.30
DEFAULT_CONF    = 0.35
SQUARE_FACTOR   = 0.53
PAD_SIDES       = 0.04
PAD_TOP         = 0.02
PAD_BOTTOM      = 0.10

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def draw_emoji(frame: np.ndarray, emoji: str,
               x: int, y: int, size: int = 26) -> np.ndarray:
    try:
        pil  = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil)
        font = None
        for fp in [
            "C:/Windows/Fonts/seguiemj.ttf",
            "C:/Windows/Fonts/segoeui.ttf",
            "/System/Library/Fonts/Apple Color Emoji.ttc",
            "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf",
        ]:
            if os.path.exists(fp):
                try:
                    font = ImageFont.truetype(fp, size)
                    break
                except Exception:
                    pass
        if font is None:
            font = ImageFont.load_default()
        draw.text((x, y), emoji, font=font, embedded_color=True)
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    except Exception:
        return frame


def nms_boxes(boxes, iou_thresh=0.40):
    if not boxes:
        return []
    coords = np.array([[b[0], b[1], b[2], b[3]] for b in boxes], np.float32)
    scores = np.array([b[4] for b in boxes], np.float32)
    x1, y1, x2, y2 = coords[:, 0], coords[:, 1], coords[:, 2], coords[:, 3]
    areas  = (x2 - x1) * (y2 - y1)
    order  = scores.argsort()[::-1]
    keep   = []
    while order.size:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou   = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[1:][iou < iou_thresh]
    return keep


def tighten_box(x1, y1, x2, y2, frame_h, frame_w):
    w  = x2 - x1
    h  = y2 - y1
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    half = int(min(w, h) * SQUARE_FACTOR)
    x1, x2 = cx - half, cx + half
    y1, y2 = cy - half, cy + half
    sw = x2 - x1
    sh = y2 - y1
    x1 += int(sw * PAD_SIDES);  x2 -= int(sw * PAD_SIDES)
    y1 += int(sh * PAD_TOP);    y2 -= int(sh * PAD_BOTTOM)
    x1 = max(0, x1);        y1 = max(0, y1)
    x2 = min(frame_w, x2);  y2 = min(frame_h, y2)
    return x1, y1, x2, y2


# ---------------------------------------------------------------------------
# Welcome / Splash Screen
# ---------------------------------------------------------------------------

class SplashScreen:
    """
    Full-screen welcome window shown before the main app loads.
    Clicking "Just get started" closes the splash and launches EmotionDetectorApp.
    """

    def __init__(self, on_start_callback):
        self.callback = on_start_callback
        self.root = ctk.CTk()
        self.root.title("EmoSense — Welcome")
        
        # Set to 1080P resolution 
        ww, wh = 1920, 1080
        
        # Center on screen
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        self.root.geometry(f"{ww}x{wh}+{(sw - ww)//2}+{(sh - wh)//2}")
        
        self.root.configure(fg_color="#0d1117")
        self.root.resizable(True, True) 

        self._build()

    def _build(self):
        root = self.root

        # Left accent bar (stays fixed width but full height)
        accent = tk.Canvas(root, width=6, bg="#00D4AA",
                           highlightthickness=0, bd=0)
        accent.place(relx=0, rely=0, relheight=1)

        # ── Header ──────────────────────────────────────────────
        header = ctk.CTkFrame(root, fg_color="transparent")
        header.place(relx=0.08, rely=0.12, relwidth=0.85)

        ctk.CTkLabel(
            header, text="🧠",
            font=("Segoe UI Emoji", 56),
            text_color="#00D4AA",
        ).pack(anchor="w")

        ctk.CTkLabel(
            header, text="EmoSense",
            font=("Georgia", 52, "bold"),
            text_color="#00D4AA",
        ).pack(anchor="w", pady=(0, 2))

        ctk.CTkLabel(
            header, text="by Hamza Khan  ·  v2.0  —  Deployment Edition",
            font=("Georgia", 14, "italic"),
            text_color="#7788AA",
        ).pack(anchor="w")

        # ── Tagline ──────────────────────────────────────────────
        self.tagline = ctk.CTkLabel(
            root,
            text="Live facial emotion recognition — detecting what faces feel, frame by frame.",
            font=("Segoe UI", 16),
            text_color="#AABBCC",
            wraplength=700,
            justify="left",
        )
        self.tagline.place(relx=0.08, rely=0.42)

        # ── Feature chips ────────────────────────────────────────
        chip_frame = ctk.CTkFrame(root, fg_color="transparent")
        chip_frame.place(relx=0.08, rely=0.54)

        chips = [
            ("🎯", "7 emotions"),
            ("📷", "Live webcam"),
            ("📊", "Real-time stats"),
            ("🖼️", "Static images"),
        ]
        for icon, label in chips:
            chip = ctk.CTkFrame(
                chip_frame,
                fg_color="#161b27",
                corner_radius=20,
                border_width=1,
                border_color="#1E2D3D"
            )
            chip.pack(side="left", padx=(0, 12))
            ctk.CTkLabel(
                chip,
                text=f"  {icon}  {label}  ",
                font=("Segoe UI Emoji", 13),
                text_color="#8899BB",
            ).pack(padx=10, pady=8)

        # ── Buttons ──────────────────────────────────────────────
        btn_row = ctk.CTkFrame(root, fg_color="transparent")
        btn_row.place(relx=0.08, rely=0.72)

        ctk.CTkButton(
            btn_row,
            text="  Just get started",
            command=self._launch,
            height=50,
            width=220,
            font=("Segoe UI", 15, "bold"),
            fg_color="#00D4AA",
            hover_color="#00FFCC",
            text_color="#0d1117",
            corner_radius=10,
        ).pack(side="left", padx=(0, 15))

        ctk.CTkButton(
            btn_row,
            text="  About",
            command=self._toggle_about,
            height=50,
            width=130,
            font=("Segoe UI", 14),
            fg_color="transparent",
            hover_color="#1a2535",
            border_color="#334455",
            border_width=1,
            text_color="#7788AA",
            corner_radius=10,
        ).pack(side="left")

        # ── About panel (Full Screen Overlay for visibility) ──────
        self.about_overlay = ctk.CTkFrame(
            root,
            fg_color="black", # Background dim
            corner_radius=0,
        )
        self.about_overlay.configure(fg_color="#0d1117") # Matches bg but will use a box
        
        self.about_visible = False

        def _close_about():
            self.about_overlay.place_forget()
            self.about_visible = False

        # Central Box
        self.about_box = ctk.CTkFrame(
            self.about_overlay,
            fg_color="#161b27",
            corner_radius=15,
            border_width=1,
            border_color="#30363d"
        )
        self.about_box.place(relx=0.5, rely=0.5, anchor="center", relwidth=0.8, relheight=0.7)

        ctk.CTkLabel(
            self.about_box,
            text="The EmoSense Story",
            font=("Georgia", 28, "bold"),
            text_color="#00D4AA"
        ).pack(pady=(30, 15))

        self.about_text = (
            "EmoSense represents a breakthrough in real-time affective computing, designed to bridge the gap between "
            "machine logic and human feeling. At its core, the application utilizes a sophisticated two-stage AI pipeline.\n\n"
            "Stage 1 — yolov8n-face detector locates human presence with pinpoint precision.\n"
            "Stage 2 — yolov8s-cls engine analyzes the subtle nuances of facial expressions across seven primary emotions.\n\n"
            "Meticulously trained on the RAF-DB (Real-world Affective Faces Database) — a repository of over 30,000 real-world images — "
            "EmoSense delivers exceptional accuracy whether tracking live video or static portraits.\n\n"
            "Version 2.0 is the production-ready edition, optimized for performance and visual clarity."
        )

        ctk.CTkLabel(
            self.about_box,
            text=self.about_text,
            font=("Segoe UI", 15),
            text_color="#AABBCC",
            wraplength=650,
            justify="center",
        ).pack(padx=40, pady=10)

        ctk.CTkButton(
            self.about_box,
            text="Close",
            command=_close_about,
            width=120,
            fg_color="#1E2D3D",
            hover_color="#2D3F50"
        ).pack(pady=30)

        # ── Footer ───────────────────────────────────────────────
        ctk.CTkLabel(
            root,
            text="Two-stage pipeline  ·  yolov8n-face  →  yolov8s-cls (RAF-DB)  ·  Summer 2025",
            font=("Consolas", 10),
            text_color="#2A3A4A",
        ).place(relx=0.08, rely=0.92)

    def _toggle_about(self):
        if self.about_visible:
            self.about_overlay.place_forget()
            self.about_visible = False
        else:
            self.about_overlay.place(relx=0, rely=0, relwidth=1, relheight=1)
            self.about_overlay.lift()
            self.about_visible = True

    def _launch(self):
        self.root.destroy()
        self.callback()

    def run(self):
        self.root.mainloop()


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

class EmotionDetectorApp:

    def __init__(self, model_path=None):
        self.face_model      = None
        self.emo_model       = None
        self.cls_model_path  = model_path
        self.camera_running  = False
        self.cap             = None
        self.emotion_counts  = {e: 0 for e in EMOTIONS}
        self.current_emotion = None

        self._build_window()
        self._build_ui()
        self._load_models()

    def _build_window(self):
        self.root = ctk.CTk()
        self.root.title("EmoSense — AI Emotion Recognition  (by HamzaChan)")
        
        # Set to 720p resolution (1280x720)
        ww, wh = 1920, 1080
        
        # Center on screen
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        self.root.geometry(f"{ww}x{wh}+{(sw - ww) // 2}+{(sh - wh) // 2}")
        
        self.root.minsize(1050, 600)
        self.root.configure(fg_color="#0d1117")
        self.root.update_idletasks()

    def _build_ui(self):
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, minsize=310, weight=0)
        self._build_titlebar()
        self._build_camera_panel()
        self._build_sidebar()

    def _build_titlebar(self):
        BAR_H = 68
        bar = ctk.CTkFrame(self.root, height=BAR_H,
                           fg_color="#111827", corner_radius=0)
        bar.grid(row=0, column=0, columnspan=2, sticky="ew")
        bar.grid_propagate(False)

        cv = tk.Canvas(bar, height=BAR_H, bg="#111827",
                       highlightthickness=0, bd=0)
        cv.pack(fill="both", expand=True)

        def _draw(event=None):
            cv.delete("all")
            W = cv.winfo_width()
            H = BAR_H
            cv.create_rectangle(0, 0, 7, H, fill="#00D4AA", outline="")
            for i in range(H):
                alpha = int(255 * (0.06 * i / H))
                col   = f"#{alpha:02x}{alpha:02x}{(alpha + 8):02x}"
                try:
                    cv.create_line(7, i, W, i, fill=col)
                except Exception:
                    pass
            cv.create_text(30, H // 2, text="🧠",
                           font=("Segoe UI Emoji", 24),
                           fill="#00D4AA", anchor="w")
            cv.create_text(68, H // 2 - 10, text="  EmoSense",
                           font=("Georgia", 20, "bold"),
                           fill="#00D4AA", anchor="w")
            cv.create_text(68, H // 2 + 12, text="   by HamzaChan  ·  v2.0",
                           font=("Georgia", 11, "italic"),
                           fill="#7788AA", anchor="w")
            cv.create_text(W - 16, H // 2,
                           text="Real-time Facial Emotion Recognition  .  "
                                "Powered by YOLOv8  .  RAF-DB",
                           font=("Consolas", 9),
                           fill="#334455", anchor="e")

        cv.bind("<Configure>", _draw)
        self.root.after(50, _draw)

    def _build_camera_panel(self):
        frame = ctk.CTkFrame(self.root, fg_color="#161b27", corner_radius=10)
        frame.grid(row=1, column=0, sticky="nsew", padx=(10, 5), pady=10)
        frame.grid_rowconfigure(1, weight=1)
        frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(frame, text="  Camera View",
                     font=("Segoe UI", 13, "bold"),
                     text_color="#8899BB").grid(row=0, column=0, pady=(8, 2))

        self.cam_lbl = ctk.CTkLabel(
            frame,
            text="Click  Start Camera  to begin\nor  Load Image  to analyse a photo",
            font=("Segoe UI", 12), text_color="#334455",
            width=CAM_W, height=CAM_H)
        self.cam_lbl.grid(row=1, column=0, padx=8, pady=(2, 6))

        self.status_lbl = ctk.CTkLabel(frame, text="  Ready",
                                       font=("Segoe UI", 10),
                                       text_color="#334455")
        self.status_lbl.grid(row=2, column=0, pady=(0, 6))

    def _build_sidebar(self):
        sb = ctk.CTkFrame(self.root, fg_color="#161b27",
                          corner_radius=10, width=310)
        sb.grid(row=1, column=1, sticky="nsew", padx=(5, 10), pady=10)
        sb.grid_propagate(False)
        sb.grid_columnconfigure(0, weight=1)
        sb.grid_rowconfigure(5, weight=1)

        ctk.CTkLabel(sb, text="  Controls",
                     font=("Segoe UI", 13, "bold"),
                     text_color="#AABBCC").grid(
            row=0, column=0, pady=(12, 4), padx=14, sticky="w")

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

        self.conf_val = ctk.CTkLabel(sf, text=f"{DEFAULT_CONF:.2f}",
                                     font=("Consolas", 10, "bold"),
                                     text_color="#00D4AA", width=36)
        self.conf_val.grid(row=1, column=1, padx=(4, 0))

        ctk.CTkFrame(sb, height=1, fg_color="#1E2D3D").grid(
            row=3, column=0, sticky="ew", padx=14, pady=4)

        ctk.CTkLabel(sb, text="  Emotion Statistics",
                     font=("Segoe UI", 13, "bold"),
                     text_color="#AABBCC").grid(
            row=4, column=0, pady=(0, 4), padx=14, sticky="w")

        bars = ctk.CTkScrollableFrame(sb, fg_color="#0f1520", corner_radius=6)
        bars.grid(row=5, column=0, sticky="nsew", padx=14, pady=(0, 6))
        bars.grid_columnconfigure(0, weight=1)

        self.emo_bars = {}
        self.emo_lbls = {}

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

        dom = ctk.CTkFrame(sb, fg_color="#0b1018",
                           corner_radius=10, height=120)
        dom.grid(row=6, column=0, sticky="ew", padx=14, pady=(0, 6))
        dom.grid_propagate(False)
        dom.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(dom, text="Current Emotion",
                     font=("Segoe UI", 9),
                     text_color="#334455").grid(row=0, column=0, pady=(8, 0))

        self.dom_emoji = ctk.CTkLabel(dom, text="--",
                                      font=("Segoe UI Emoji", 40),
                                      text_color="#FFFFFF")
        self.dom_emoji.grid(row=1, column=0)

        self.dom_name = ctk.CTkLabel(dom, text="waiting...",
                                     font=("Georgia", 13, "bold"),
                                     text_color="#445566")
        self.dom_name.grid(row=2, column=0, pady=(0, 8))

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

    def _load_models(self):
        print("\nLoading Stage-1 face detector (yolov8n-face.pt)...")
        try:
            self.face_model = YOLO("yolov8n-face.pt")
            print("  Face detector ready")
        except Exception as e:
            print(f"  Could not load yolov8n-face.pt: {e}")
            try:
                self.face_model = YOLO("yolov8n.pt")
                print("  Fallback: using yolov8n.pt")
            except Exception:
                self.face_model = None
                print("  No face model loaded -- using full frame as face crop")

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

        cls_path = self.cls_model_path
        if not cls_path:
            for p in search_paths:
                if os.path.exists(p):
                    cls_path = p
                    break

        if not cls_path or not os.path.exists(cls_path):
            self.mdl_lbl.configure(text="Emotion model: not found")
            print("\nNo emotion model found.")
            print("   Expected at: models/emotion_classifier/best.pt")
            print("   Run training first:")
            print("   python 2_train_model.py\n")
            return

        try:
            self.emo_model = YOLO(cls_path)
            short = Path(cls_path).name
            self.mdl_lbl.configure(text=f"Model: {short}",
                                   text_color="#00CC88")
            print(f"  Emotion classifier loaded: {cls_path}")
        except Exception as e:
            self.mdl_lbl.configure(text="Emotion model: load failed")
            print(f"Error loading emotion model: {e}")

    def _toggle_cam(self):
        if self.camera_running:
            self._stop_cam()
        else:
            self._start_cam()

    def _start_cam(self):
        if self.emo_model is None:
            self._err("No emotion model loaded!\nRun: python 2_train_model.py")
            return
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self._err("Could not open webcam!")
            return
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.camera_running = True
        self.btn_cam.configure(text="  Stop Camera",
                               fg_color="#6B1A1A", hover_color="#9A2222")
        self.status_lbl.configure(text="  Camera running...",
                                  text_color="#00CC88")
        self._tick()

    def _stop_cam(self):
        self.camera_running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.btn_cam.configure(text="  Start Camera",
                               fg_color="#1A6B4A", hover_color="#22996A")
        self.status_lbl.configure(text="  Stopped", text_color="#445566")

    def _tick(self):
        if not self.camera_running:
            return
        ret, frame = self.cap.read()
        if not ret:
            self._stop_cam()
            return
        t0  = time.time()
        out = self._detect(frame)
        fps = 1.0 / max(time.time() - t0, 0.001)
        self._show(out)
        self.fps_lbl.configure(text=f"FPS: {fps:.1f}")
        self.root.after(30, self._tick)

    def _detect(self, frame: np.ndarray) -> np.ndarray:
        conf_thresh = self.conf_slider.get()
        fh, fw = frame.shape[:2]
        annotated = frame.copy()

        top_emo  = None
        top_conf = 0.0
        finals   = []

        if self.face_model is not None:
            face_results = self.face_model(
                frame, conf=FACE_CONF, iou=0.40, imgsz=640, verbose=False
            )[0]
            raw = []
            if face_results.boxes is not None:
                for box in face_results.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    fc = float(box.conf[0])
                    raw.append((x1, y1, x2, y2, fc))
            keep  = nms_boxes(raw, 0.40)
            faces = [raw[k] for k in keep]
        else:
            faces = [(0, 0, fw, fh, 1.0)]

        self.det_lbl.configure(text=f"Detections: {len(faces)}")

        for (x1, y1, x2, y2, face_conf) in faces:
            x1, y1, x2, y2 = tighten_box(x1, y1, x2, y2, fh, fw)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            emo_result = self.emo_model(crop, verbose=False)
            probs      = emo_result[0].probs
            pred_cls   = int(probs.top1)
            pred_conf  = float(probs.top1conf)

            if pred_conf < conf_thresh:
                continue

            cls_names = self.emo_model.names
            pred_name = cls_names.get(pred_cls,
                        EMOTIONS[pred_cls] if pred_cls < len(EMOTIONS) else "?")
            pred_name = pred_name.lower()

            finals.append((x1, y1, x2, y2, pred_conf, pred_name))

            if pred_conf > top_conf:
                top_conf = pred_conf
                top_emo  = pred_name

        for (x1, y1, x2, y2, confidence, emo) in finals:
            color = EMOTION_COLORS_BGR.get(emo, (200, 200, 200))
            emoji = EMOJI_MAP.get(emo, "?")

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            BADGE = 42
            bx1, by1 = x1 + 3, y1 + 3
            bx2, by2 = bx1 + BADGE, by1 + BADGE
            roi = annotated[by1:by2, bx1:bx2]
            if roi.size > 0:
                annotated[by1:by2, bx1:bx2] = (roi * 0.35).astype(np.uint8)
            cv2.rectangle(annotated, (bx1, by1), (bx2, by2), color, 1)
            annotated = draw_emoji(annotated, emoji, bx1 + 3, by1 + 2, 26)

            ny = by2 + 11
            (tw, th), _ = cv2.getTextSize(emo, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
            cv2.rectangle(annotated, (bx1, ny - th - 3), (bx1 + tw + 5, ny + 2),
                          (8, 12, 22), -1)
            cv2.putText(annotated, emo, (bx1 + 2, ny),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, color, 1, cv2.LINE_AA)

            ct = f"{confidence:.0%}"
            (cw, ch), _ = cv2.getTextSize(ct, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            cv2.rectangle(annotated, (x2 - cw - 7, y1), (x2, y1 + ch + 5),
                          color, -1)
            cv2.putText(annotated, ct, (x2 - cw - 3, y1 + ch + 1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1,
                        cv2.LINE_AA)

            if emo in self.emotion_counts:
                self.emotion_counts[emo] += 1

        self._update_dom(top_emo)
        if finals:
            self._update_bars()

        return annotated

    def _update_dom(self, emo):
        if emo and emo in EMOTION_COLORS_HEX:
            c = EMOTION_COLORS_HEX[emo]
            self.dom_emoji.configure(text=EMOJI_MAP.get(emo, "?"), text_color=c)
            self.dom_name.configure(text=emo.capitalize(), text_color=c)
        else:
            self.dom_emoji.configure(text="--", text_color="#FFFFFF")
            self.dom_name.configure(text="no face detected", text_color="#334455")

    def _show(self, frame: np.ndarray):
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w   = rgb.shape[:2]
        scale  = min(CAM_W / w, CAM_H / h)
        nw, nh = int(w * scale), int(h * scale)
        res    = cv2.resize(rgb, (nw, nh))
        canvas = np.zeros((CAM_H, CAM_W, 3), dtype=np.uint8)
        ox = (CAM_W - nw) // 2
        oy = (CAM_H - nh) // 2
        canvas[oy:oy + nh, ox:ox + nw] = res
        pil     = Image.fromarray(canvas)
        ctk_img = ctk.CTkImage(light_image=pil, dark_image=pil,
                                size=(CAM_W, CAM_H))
        self.cam_lbl.configure(image=ctk_img, text="")
        self.cam_lbl.image = ctk_img

    def _load_image(self):
        if self.emo_model is None:
            self._err("No emotion model loaded!\nRun: python 2_train_model.py")
            return
        from tkinter import filedialog
        fp = filedialog.askopenfilename(
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp")])
        if not fp:
            return
        img = cv2.imread(fp)
        if img is None:
            self._err("Could not load image.")
            return
        out = self._detect(img)
        self._show(out)
        self.status_lbl.configure(text=f"  {Path(fp).name}",
                                  text_color="#AABBCC")

    def _update_bars(self):
        total = sum(self.emotion_counts.values())
        if not total:
            return
        for emo in EMOTIONS:
            pct = self.emotion_counts[emo] / total * 100
            self.emo_bars[emo].set(pct / 100)
            self.emo_lbls[emo].configure(text=f"{pct:.1f}%")

    def _on_conf(self, val):
        self.conf_val.configure(text=f"{float(val):.2f}")

    def _err(self, msg):
        w = ctk.CTkToplevel(self.root)
        w.title("Error")
        w.geometry("360x130")
        w.grab_set()
        ctk.CTkLabel(w, text=f"  {msg}",
                     font=("Segoe UI", 12), wraplength=320).pack(
            expand=True, pady=16)
        ctk.CTkButton(w, text="OK", command=w.destroy, width=70).pack(
            pady=(0, 12))

    def run(self):
        print("\n" + "=" * 55)
        print("  EmoSense  by HamzaChan  --  v2.0 (Deployment Edition)")
        print("  Two-stage: face detect -> emotion classify")
        print(f"  Box factor : SQUARE_FACTOR={SQUARE_FACTOR}")
        print(f"  Confidence : default {DEFAULT_CONF}")
        print("=" * 55 + "\n")
        self.root.mainloop()


# ---------------------------------------------------------------------------
# Entry point — splash → main app
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="EmoSense — Emotion Recognition App v2.0")
    p.add_argument("--model", default=None,
                   help="Path to emotion classifier model (.pt or .onnx). "
                        "If omitted, the app searches common locations automatically.")
    p.add_argument("--no-splash", action="store_true",
                   help="Skip the welcome splash screen and launch directly.")
    args = p.parse_args()

    def launch_main():
        EmotionDetectorApp(model_path=args.model).run()

    if args.no_splash:
        launch_main()
    else:
        SplashScreen(on_start_callback=launch_main).run()


if __name__ == "__main__":
    main()