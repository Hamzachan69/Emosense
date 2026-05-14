"""  # Start of a multi-line docstring to describe the script
EmoSense by Hamza Khan - v6 (RAF-DB Two-Stage Edition)  # Title and version
  # Empty line for readability
This is the main application. It opens a window with a live camera feed  # Purpose of the script
and detects faces in real time. For each face found, it predicts the  # Main functionality
person's emotion and displays it on screen with a confidence score.  # User feedback
  # Empty line for readability
The pipeline has two stages:  # Architecture description
  Stage 1 - yolov8n-face  : finds faces in the frame and draws bounding boxes  # Stage 1 details
  Stage 2 - yolov8s-cls   : classifies the emotion from each cropped face  # Stage 2 details
"""  # End of the docstring
  # Empty line for readability
import os  # Import 'os' to interact with the operating system (paths, files)
import sys  # Import 'sys' to access system parameters (exit, arguments)
import time  # Import 'time' for benchmarking and managing frame rates
import argparse  # Import 'argparse' to handle command-line arguments
from pathlib import Path  # Import 'Path' for modern, cross-platform file path handling
  # Empty line for readability
# Try to import all required libraries. If any are missing, print a helpful message and stop.  # Dependency check
try:  # Start of safety block
    import cv2  # Import OpenCV for camera and image processing
    import numpy as np  # Import NumPy for fast mathematical operations on image arrays
    import customtkinter as ctk  # Import CustomTkinter for a modern user interface
    from PIL import Image, ImageTk, ImageDraw, ImageFont  # Import PIL for rendering images and emojis
    from ultralytics import YOLO  # Import YOLO for loading and running the AI models
except ImportError as e:  # If a library is missing
    print(f"\nMissing package: {e.name}")  # Identify the missing package
    print("   Run: pip install ultralytics customtkinter pillow opencv-python\n")  # Provide fix
    sys.exit(1)  # Stop the script with an error code
  # Empty line for readability
# --- CONSTANTS ---  # Section for global settings that don't change
  # Empty line for readability
# The 7 emotion class names in alphabetical order.  # Defining the classification targets
# YOLO assigns class indices based on alphabetical folder order, so this order matters.  # Important technical note
# Index 0=angry, 1=disgust, 2=fear, 3=happy, 4=neutral, 5=sad, 6=surprise  # Mapping indices to names
EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]  # List of emotions
  # Empty line for readability
# Hex color codes for each emotion, used in the GUI labels and progress bars.  # UI design settings
EMOTION_COLORS_HEX = {  # Dictionary for mapping emotions to colors
    "angry":    "#FF2222",  # Red
    "disgust":  "#00AAAA",  # Teal/Cyan
    "fear":     "#AA44CC",  # Purple
    "happy":    "#FFD700",  # Gold/Yellow
    "neutral":  "#00CC44",  # Green
    "sad":      "#4488FF",  # Blue
    "surprise": "#FF8800",  # Orange
}  # End of dictionary
  # Empty line for readability
# BGR color values for the same emotions, used when drawing boxes on the camera frame with OpenCV.  # Image drawing settings
# OpenCV uses BGR (Blue, Green, Red) instead of the usual RGB order.  # Technical note on color order
EMOTION_COLORS_BGR = {  # Dictionary for mapping emotions to BGR tuples
    "angry":    (34,  34,  255),  # Red-ish
    "disgust":  (170, 170,   0),  # Cyan-ish
    "fear":     (204,  68, 170),  # Purple-ish
    "happy":    (0,  215,  255),  # Yellow-ish
    "neutral":  (0,  204,   68),  # Green-ish
    "sad":      (255, 136,  68),  # Blue-ish
    "surprise": (0,  136,  255),  # Orange-ish
}  # End of dictionary
  # Empty line for readability
# Emoji characters displayed next to each emotion label on the camera feed.  # User experience enhancement
EMOJI_MAP = {  # Dictionary for mapping emotions to emojis
    "angry":    "😠",  # Angry face
    "disgust":  "🤢",  # Nauseated face
    "fear":     "😨",  # Fearful face
    "happy":    "😊",  # Smiling face
    "neutral":  "😐",  # Neutral face
    "sad":      "😢",  # Crying face
    "surprise": "😮",  # Surprised face
}  # End of dictionary
  # Empty line for readability
# The pixel size of the camera display area inside the window.  # GUI layout settings
CAM_W, CAM_H = 860, 484  # Width and height constants
  # Empty line for readability
# Minimum confidence for the face detector (Stage 1).  # AI sensitivity for stage 1
# Kept low (0.30) so that smaller or partially visible faces are still detected.  # Rationale for low value
FACE_CONF = 0.30  # Threshold constant
  # Empty line for readability
# Default minimum confidence for the emotion classifier (Stage 2).  # AI sensitivity for stage 2
# Only predictions above this threshold are displayed on screen.  # Rationale for value
DEFAULT_CONF = 0.35  # Threshold constant
  # Empty line for readability
# --- BOUNDING BOX TIGHTENING PARAMETERS ---  # Section for refining detection areas
# The face detector often draws a box that includes the neck and shoulders.  # Problem description
# These values shrink the box to fit more tightly around the head.  # Solution description
  # Empty line for readability
# SQUARE_FACTOR: controls how large the square crop is relative to the detected face.  # Sizing parameter
# A lower value means a tighter crop. 0.53 works well for most webcam distances.  # Tip for tuning
SQUARE_FACTOR = 0.53  # Scaling constant
  # Empty line for readability
# After squaring the box, trim additional background from each side.  # Trimming description
PAD_SIDES  = 0.04   # trim a small amount (4%) from the left and right
PAD_TOP    = 0.02   # trim a tiny amount (2%) from the top (reduces extra space above head)
PAD_BOTTOM = 0.10   # trim more (10%) from the bottom (removes chin and neck area)
  # Empty line for readability
# Set the visual theme for the application window.  # GUI appearance settings
ctk.set_appearance_mode("Dark")  # Use dark mode
ctk.set_default_color_theme("blue")  # Use blue accent colors
  # Empty line for readability


# --- HELPER FUNCTIONS ---  # Section for small utility functions
  # Empty line for readability
def draw_emoji(frame: np.ndarray, emoji: str,  # Function to put emojis on image
               x: int, y: int, size: int = 26) -> np.ndarray:  # Arguments: image, emoji, position, size
    # Draws an emoji character onto an OpenCV frame at position (x, y).  # Function purpose
    # This is done by temporarily converting the frame to a PIL image,  # Conversion step
    # drawing the text with a system emoji font, then converting back to OpenCV.  # Font rendering step
    try:  # Start of safety block
        pil  = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # Convert BGR (OpenCV) to RGB (PIL)
        draw = ImageDraw.Draw(pil)  # Create a drawing handle for the PIL image
        font = None  # Initialize font variable
  # Empty line for readability
        # Try to find an emoji-capable font file on the current operating system.  # Font searching
        for fp in [  # List of common font locations
            "C:/Windows/Fonts/seguiemj.ttf",       # Windows emoji font (standard)
            "C:/Windows/Fonts/segoeui.ttf",         # Windows fallback font
            "/System/Library/Fonts/Apple Color Emoji.ttc",   # macOS emoji font
            "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf",  # Linux emoji font
        ]:  # End of font list
            if os.path.exists(fp):  # If the font file actually exists on this computer
                try:  # Try to load it
                    font = ImageFont.truetype(fp, size)  # Load the font at the requested size
                    break  # Stop looking once a font is found
                except Exception:  # If loading fails (rare)
                    pass  # Try the next one
  # Empty line for readability
        # If no emoji font was found, fall back to the default PIL font.  # Fail-safe
        if font is None:  # If the loop finished without finding a font
            font = ImageFont.load_default()  # Use the basic built-in font
  # Empty line for readability
        draw.text((x, y), emoji, font=font, embedded_color=True)  # Draw the emoji onto the PIL image
        # Convert the modified PIL image back to an OpenCV BGR frame.  # Final conversion
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)  # Return image in BGR format
    except Exception:  # If anything goes wrong during drawing
        # If drawing fails for any reason, return the frame unchanged.  # Error handling
        return frame  # Return original image to prevent app crash
  # Empty line for readability
  # Empty line for readability
def nms_boxes(boxes, iou_thresh=0.40):  # Function for Non-Maximum Suppression
    """  # Start of docstring
    Non-Maximum Suppression (NMS) -- removes duplicate face detections.  # Purpose
  # Empty line
    When the face detector finds the same face multiple times with slightly  # Scenario description
    different box positions, NMS keeps only the highest-confidence detection  # Solution logic
    and removes any others that overlap it by more than iou_thresh.  # Overlap rule
    """  # End of docstring
    if not boxes:  # If no boxes were detected at all
        return []  # Return an empty list
  # Empty line for readability
    # Extract coordinates and confidence scores into numpy arrays.  # Data preparation
    coords = np.array([[b[0], b[1], b[2], b[3]] for b in boxes], np.float32)  # Box locations
    scores = np.array([b[4] for b in boxes], np.float32)  # Detection confidence levels
    x1, y1, x2, y2 = coords[:, 0], coords[:, 1], coords[:, 2], coords[:, 3]  # Individual coordinates
  # Empty line for readability
    # Calculate the area of each bounding box.  # Math for overlap calculation
    areas  = (x2 - x1) * (y2 - y1)  # Area = width * height
  # Empty line for readability
    # Sort detections from highest confidence to lowest.  # Ordering step
    order  = scores.argsort()[::-1]  # Get indices in descending order of score
    keep   = []  # List to store the indices of boxes we want to keep
  # Empty line for readability
    while order.size:  # While there are still boxes to check
        i = order[0]   # Take the index of the highest-confidence box
        keep.append(i)  # We always keep the best one
  # Empty line for readability
        if order.size == 1:  # If that was the last box
            break  # Stop the loop
  # Empty line for readability
        # Calculate the overlap (IoU) between the best box and all remaining boxes.  # Intersection over Union
        xx1 = np.maximum(x1[i], x1[order[1:]])  # Find max of left edges
        yy1 = np.maximum(y1[i], y1[order[1:]])  # Find max of top edges
        xx2 = np.minimum(x2[i], x2[order[1:]])  # Find min of right edges
        yy2 = np.minimum(y2[i], y2[order[1:]])  # Find min of bottom edges
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)  # Area of intersection
        iou   = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)  # IoU formula
  # Empty line for readability
        # Discard any boxes that overlap the current best box by more than the threshold.  # Suppression step
        order = order[1:][iou < iou_thresh]  # Filter out highly overlapping boxes
  # Empty line for readability
    return keep  # Return the indices of the unique boxes
  # Empty line for readability
  # Empty line for readability
def tighten_box(x1, y1, x2, y2, frame_h, frame_w):  # Function to refine face boxes
    """  # Start of docstring
    Converts an oversized YOLO face bounding box into a tight square around the head.  # Goal
  # Empty line
    Step 1 - Square:  find the center of the box and re-expand it symmetrically  # Algorithm part 1
                      using SQUARE_FACTOR so the result is a square.  # Shape normalization
    Step 2 - Trim:    shave off padding fractions from each side to reduce background.  # Algorithm part 2
    Step 3 - Clamp:   make sure the final coordinates stay within the frame boundaries.  # Algorithm part 3
    """  # End of docstring
    w  = x2 - x1  # Original box width
    h  = y2 - y1  # Original box height
    cx = (x1 + x2) // 2   # Find the horizontal center point
    cy = (y1 + y2) // 2   # Find the vertical center point
  # Empty line for readability
    # Step 1: Create a square box centered on (cx, cy).  # Making it a 1:1 ratio
    half = int(min(w, h) * SQUARE_FACTOR)  # Calculate half-width for the square
    x1, x2 = cx - half, cx + half  # New left and right edges
    y1, y2 = cy - half, cy + half  # New top and bottom edges
  # Empty line for readability
    # Step 2: Trim background from each side of the square.  # Shaving edges
    sw = x2 - x1   # Width of our new square
    sh = y2 - y1   # Height of our new square
    x1 += int(sw * PAD_SIDES);   x2 -= int(sw * PAD_SIDES)  # Move left/right edges inward
    y1 += int(sh * PAD_TOP);     y2 -= int(sh * PAD_BOTTOM)  # Move top/bottom edges inward
  # Empty line for readability
    # Step 3: Clamp to frame boundaries so we don't read outside the image.  # Safety check
    x1 = max(0, x1);        y1 = max(0, y1)  # Don't go past the top or left
    x2 = min(frame_w, x2);  y2 = min(frame_h, y2)  # Don't go past the bottom or right
  # Empty line for readability
    return x1, y1, x2, y2  # Return the finalized, tight coordinates
  # Empty line for readability
  # Empty line for readability
# --- MAIN APPLICATION CLASS ---  # Section for the window logic
  # Empty line for readability

class EmotionDetectorApp:  # Define the main application class
  # Empty line for readability
    def __init__(self, model_path=None):  # Constructor method (runs on creation)
        # Store the optional custom model path (can be passed via command line).  # Setup variables
        self.face_model      = None    # Stage 1: placeholder for the face detector model
        self.emo_model       = None    # Stage 2: placeholder for the emotion classifier model
        self.cls_model_path  = model_path  # Save the custom model path if provided
        self.camera_running  = False   # Flag to track if the webcam is currently on
        self.cap             = None    # Placeholder for the OpenCV video capture object
        self.emotion_counts  = {e: 0 for e in EMOTIONS}  # Dictionary to count how many times each emotion is seen
        self.current_emotion = None    # Variable to store the most frequent emotion in the recent frame
  # Empty line for readability
        # Build the window, lay out the UI, then load the models.  # Initialization sequence
        self._build_window()  # Create the main window frame
        self._build_ui()  # Populate the window with buttons and panels
        self._load_models()  # Load the AI models into memory
  # Empty line for readability
    # -- WINDOW --  # Section for window configuration
    def _build_window(self):  # Method to create and center the window
        # Create the main application window and center it on the screen.  # Purpose
        self.root = ctk.CTk()  # Initialize the CustomTkinter root window
        self.root.title("EmoSense -- AI Emotion Recognition  (by HamzaChan)")  # Set the title
        self.root.geometry("1300x640")  # Set initial width and height
        self.root.minsize(1100, 580)  # Prevent the user from making it too small
        self.root.configure(fg_color="#0d1117")  # Set the background color to a dark theme
        self.root.update_idletasks()  # Force internal updates so dimensions are accurate
  # Empty line for readability
        # Calculate screen and window dimensions to center the window.  # Positioning logic
        sw = self.root.winfo_screenwidth()  # Get screen width
        sh = self.root.winfo_screenheight()  # Get screen height
        ww = self.root.winfo_width()  # Get window width
        wh = self.root.winfo_height()  # Get window height
        self.root.geometry(f"{ww}x{wh}+{(sw - ww) // 2}+{(sh - wh) // 2}")  # Apply centering formula
  # Empty line for readability
    # -- FULL UI --  # Section for overall layout
    def _build_ui(self):  # Method to organize the layout of components
        # Configure the grid layout: the camera panel expands, the sidebar has a fixed width.  # Responsive design
        self.root.grid_rowconfigure(1, weight=1)  # Make row 1 (main area) expandable
        self.root.grid_columnconfigure(0, weight=1)  # Make column 0 (camera) expandable
        self.root.grid_columnconfigure(1, minsize=310, weight=0)  # Column 1 (sidebar) stays at 310px
  # Empty line for readability
        # Build each section of the UI.  # Call construction methods
        self._build_titlebar()  # Create the top bar
        self._build_camera_panel()  # Create the left camera feed area
        self._build_sidebar()  # Create the right control panel
  # Empty line for readability
    # -- TITLE BAR --  # Section for the decorative top bar
    def _build_titlebar(self):  # Method to draw the custom title bar
        # Creates the top bar that spans the full width of the window.  # Purpose
        BAR_H = 68  # Define fixed height for the bar
        bar = ctk.CTkFrame(self.root, height=BAR_H,  # Create the container frame
                           fg_color="#111827", corner_radius=0)  # Set dark color and sharp corners
        bar.grid(row=0, column=0, columnspan=2, sticky="ew")  # Place it at the top, stretching across
        bar.grid_propagate(False)  # Keep the frame at BAR_H height
  # Empty line for readability
        import tkinter as tk  # Import standard tkinter for the Canvas component
  # Empty line for readability
        # Use a raw tkinter Canvas inside the frame for custom gradient drawing.  # Canvas allows custom graphics
        cv = tk.Canvas(bar, height=BAR_H, bg="#111827",  # Create canvas matching frame height
                       highlightthickness=0, bd=0)  # Remove borders and highlights
        cv.pack(fill="both", expand=True)  # Make canvas fill the frame
  # Empty line for readability
        def _draw(event=None):  # Inner function to handle the drawing of graphics
            # Redraws the title bar content. Called on window resize and at startup.  # Dynamic redraw
            cv.delete("all")  # Clear the canvas before redrawing
            W = cv.winfo_width()  # Get current canvas width
            H = BAR_H  # Use fixed height
  # Empty line for readability
            # Draw a green accent strip on the left edge.  # Brand accent
            cv.create_rectangle(0, 0, 7, H, fill="#00D4AA", outline="")  # 7-pixel wide vertical line
  # Empty line for readability
            # Draw a subtle vertical gradient across the rest of the bar.  # Aesthetic touch
            for i in range(H):  # Loop through each pixel height
                alpha = int(255 * (0.06 * i / H))  # Calculate transparency based on height
                col   = f"#{alpha:02x}{alpha:02x}{(alpha + 8):02x}"  # Build a color hex string
                try:  # Safety block
                    cv.create_line(7, i, W, i, fill=col)  # Draw a horizontal line of that color
                except Exception:  # If color calculation fails
                    pass  # Skip
  # Empty line for readability
            # Draw the app name and subtitle text.  # Visual labels
            cv.create_text(30, H // 2, text="🧠",  # Brain icon
                           font=("Segoe UI Emoji", 24),  # Use emoji font
                           fill="#00D4AA", anchor="w")  # Positioned left
            cv.create_text(68, H // 2 - 10, text="  EmoSense",  # Main title
                           font=("Georgia", 20, "bold"),  # Large serif font
                           fill="#00D4AA", anchor="w")  # Positioned left
            cv.create_text(68, H // 2 + 12, text="   by HamzaChan",  # Subtitle
                           font=("Georgia", 11, "italic"),  # Smaller italic font
                           fill="#7788AA", anchor="w")  # Positioned below title
  # Empty line for readability
            # Draw a small description on the right side of the bar.  # Metadata
            cv.create_text(W - 16, H // 2,  # Align to right edge
                           text="Real-time Facial Emotion Recognition  .  "  # Text content
                                "Powered by YOLOv8  .  RAF-DB",  # More content
                           font=("Consolas", 9),  # Techy monospace font
                           fill="#334455", anchor="e")  # Positioned right
  # Empty line for readability
        # Redraw the title bar when the window is resized.  # Ensuring responsive graphics
        cv.bind("<Configure>", _draw)  # Bind the draw function to resize events
        # Also draw it shortly after startup to ensure the window width is known.  # Startup trigger
        self.root.after(50, _draw)  # Wait 50ms then draw
  # Empty line for readability
    # -- CAMERA PANEL --  # Section for the video feed display
    def _build_camera_panel(self):  # Method to create the central viewing area
        # The left panel where the camera feed or loaded image is displayed.  # Purpose
        frame = ctk.CTkFrame(self.root, fg_color="#161b27", corner_radius=10)  # Dark inner container
        frame.grid(row=1, column=0, sticky="nsew", padx=(10, 5), pady=10)  # Place in main grid
        frame.grid_rowconfigure(1, weight=1)  # Make middle row (image) expandable
        frame.grid_columnconfigure(0, weight=1)  # Make content centerable
  # Empty line for readability
        ctk.CTkLabel(frame, text="  Camera View",  # Header text
                     font=("Segoe UI", 13, "bold"),  # Font style
                     text_color="#8899BB").grid(row=0, column=0, pady=(8, 2))  # Place at top of panel
  # Empty line for readability
        # The main label that holds the camera image. Starts with a placeholder message.  # Image placeholder
        self.cam_lbl = ctk.CTkLabel(  # Create label for image
            frame,  # Inside the panel
            text="Click  Start Camera  to begin\nor  Load Image  to analyse a photo",  # Welcome message
            font=("Segoe UI", 12), text_color="#334455",  # Style
            width=CAM_W, height=CAM_H)  # Set initial dimensions
        self.cam_lbl.grid(row=1, column=0, padx=8, pady=(2, 6))  # Place in middle of panel
  # Empty line for readability
        # A small status text at the bottom of the camera panel.  # Information display
        self.status_lbl = ctk.CTkLabel(frame, text="  Ready",  # Default status
                                       font=("Segoe UI", 10),  # Small font
                                       text_color="#334455")  # Muted color
        self.status_lbl.grid(row=2, column=0, pady=(0, 6))  # Place at bottom
  # Empty line for readability
    # -- SIDEBAR --  # Section for controls and statistics
    def _build_sidebar(self):  # Method to construct the right-side control column
        # The right-side panel containing controls, emotion stats, and the current emotion display.  # Purpose
        sb = ctk.CTkFrame(self.root, fg_color="#161b27",  # Create sidebar frame
                          corner_radius=10, width=310)  # Fixed width, dark theme
        sb.grid(row=1, column=1, sticky="nsew", padx=(5, 10), pady=10)  # Place in main grid
        sb.grid_propagate(False)  # Lock the width
        sb.grid_columnconfigure(0, weight=1)  # Center widgets
        sb.grid_rowconfigure(5, weight=1)  # row 5 (the stats panel) expands to fill available space
  # Empty line for readability
        ctk.CTkLabel(sb, text="  Controls",  # Label for the section
                     font=("Segoe UI", 13, "bold"),  # Font
                     text_color="#AABBCC").grid(  # Muted blue color
            row=0, column=0, pady=(12, 4), padx=14, sticky="w")  # Align left
  # Empty line for readability
        # Button row: Start/Stop Camera and Load Image side by side.  # Interactive controls
        bf = ctk.CTkFrame(sb, fg_color="transparent")  # Container for buttons
        bf.grid(row=1, column=0, sticky="ew", padx=14, pady=(0, 4))  # Place in sidebar
        bf.grid_columnconfigure((0, 1), weight=1)  # Equal width buttons
  # Empty line for readability
        self.btn_cam = ctk.CTkButton(  # Toggle camera button
            bf, text="  Start Camera", command=self._toggle_cam,  # Text and link to function
            height=34, font=("Segoe UI", 11, "bold"),  # Dimensions and font
            fg_color="#1A6B4A", hover_color="#22996A")  # Green colors
        self.btn_cam.grid(row=0, column=0, padx=(0, 3), sticky="ew")  # Left side
  # Empty line for readability
        ctk.CTkButton(  # Static load image button
            bf, text="  Load Image", command=self._load_image,  # Text and link to function
            height=34, font=("Segoe UI", 11, "bold"),  # Dimensions and font
            fg_color="#1A3A6B", hover_color="#225A9A").grid(  # Blue colors
            row=0, column=1, padx=(3, 0), sticky="ew")  # Right side
  # Empty line for readability
        # Confidence threshold slider -- controls how certain the model must be before  # AI adjustment
        # showing a label. Higher values = fewer but more reliable predictions.  # Explanation
        sf = ctk.CTkFrame(sb, fg_color="transparent")  # Container for slider
        sf.grid(row=2, column=0, sticky="ew", padx=14, pady=(0, 4))  # Place in sidebar
        sf.grid_columnconfigure(0, weight=1)  # Stretch slider
  # Empty line for readability
        ctk.CTkLabel(sf, text="Confidence Threshold:",  # Description
                     font=("Segoe UI", 10),  # Small font
                     text_color="#8899BB").grid(  # Muted color
            row=0, column=0, columnspan=2, sticky="w")  # Align left
  # Empty line for readability
        self.conf_slider = ctk.CTkSlider(  # Slider component
            sf, from_=0.10, to=0.90, number_of_steps=80,  # Range and granularity
            command=self._on_conf,  # Link to update function
            button_color="#00D4AA", button_hover_color="#00FFCC",  # Teal accent
            progress_color="#00D4AA")  # Active bar color
        self.conf_slider.set(DEFAULT_CONF)  # Set initial value
        self.conf_slider.grid(row=1, column=0, sticky="ew")  # Stretch across panel
  # Empty line for readability
        # Label that shows the current slider value as a number.  # Visual feedback for slider
        self.conf_val = ctk.CTkLabel(sf, text=f"{DEFAULT_CONF:.2f}",  # Initial text
                                     font=("Consolas", 10, "bold"),  # Monospace font for numerical precision
                                     text_color="#00D4AA", width=36)  # Teal color matching the slider
  # Empty line for readability
        self.conf_val.grid(row=1, column=1, padx=(4, 0))  # Place value label next to slider
  # Empty line for readability
        # Horizontal divider line.  # Visual separator in the sidebar
        ctk.CTkFrame(sb, height=1, fg_color="#1E2D3D").grid(  # Create a very thin dark frame
            row=3, column=0, sticky="ew", padx=14, pady=4)  # Place between controls and stats
  # Empty line for readability
        ctk.CTkLabel(sb, text="  Emotion Statistics",  # Section header for the bar charts
                     font=("Segoe UI", 13, "bold"),  # Bold font
                     text_color="#AABBCC").grid(  # Muted color
            row=4, column=0, pady=(0, 4), padx=14, sticky="w")  # Align left
  # Empty line for readability
        # Scrollable area containing one progress bar row per emotion.  # Data visualization container
        bars = ctk.CTkScrollableFrame(sb, fg_color="#0f1520", corner_radius=6)  # Darker inner frame
        bars.grid(row=5, column=0, sticky="nsew", padx=14, pady=(0, 6))  # Fill sidebar space
        bars.grid_columnconfigure(0, weight=1)  # Stretch content horizontally
  # Empty line for readability
        # Dictionaries to hold references to each emotion's bar and percentage label.  # State management
        self.emo_bars = {}  # References to progress bars
        self.emo_lbls = {}  # References to percentage labels
  # Empty line for readability
        # Create one row per emotion with: emoji + name, a progress bar, and a percentage label.  # Row builder
        for i, emo in enumerate(EMOTIONS):  # Loop through all 7 supported emotions
            rf = ctk.CTkFrame(bars, fg_color="transparent")  # Container for the row
            rf.grid(row=i, column=0, sticky="ew", pady=2)  # Place in scrollable list
            rf.grid_columnconfigure(1, weight=1)  # Make the bar the expandable part
  # Empty line for readability
            ctk.CTkLabel(rf,  # Name and Emoji label
                         text=f"{EMOJI_MAP[emo]}  {emo}",  # Content (e.g. "😊 happy")
                         font=("Segoe UI Emoji", 11),  # Support emojis in font
                         width=110, anchor="w",  # Fixed width, left aligned
                         text_color="#AABBCC").grid(row=0, column=0, sticky="w")  # Place in row
  # Empty line for readability
            bar = ctk.CTkProgressBar(rf, height=14,  # The actual progress bar
                                     progress_color=EMOTION_COLORS_HEX[emo],  # Color specific to emotion
                                     fg_color="#1A2233")  # Background color of the bar
            bar.set(0)  # Start at 0%
            bar.grid(row=0, column=1, sticky="ew", padx=5)  # Stretch across middle of row
  # Empty line for readability
            lbl = ctk.CTkLabel(rf, text="0%",  # The percentage text label
                               font=("Consolas", 10, "bold"),  # Monospace for alignment
                               width=38, text_color="#AABBCC")  # Fixed width
            lbl.grid(row=0, column=2)  # Place at end of row
  # Empty line for readability
            self.emo_bars[emo] = bar  # Store bar for updates
            self.emo_lbls[emo] = lbl  # Store label for updates
  # Empty line for readability
        # "Current Emotion" display box -- shows a large emoji and name for the dominant emotion.  # Primary output
        dom = ctk.CTkFrame(sb, fg_color="#0b1018",  # High-contrast dark box
                           corner_radius=10, height=120)  # Rounded corners, fixed height
        dom.grid(row=6, column=0, sticky="ew", padx=14, pady=(0, 6))  # Place above bottom panel
        dom.grid_propagate(False)  # Lock height
        dom.grid_columnconfigure(0, weight=1)  # Center everything
  # Empty line for readability
        ctk.CTkLabel(dom, text="Current Emotion",  # Metadata label
                     font=("Segoe UI", 9),  # Small font
                     text_color="#334455").grid(row=0, column=0, pady=(8, 0))  # Place at top of box
  # Empty line for readability
        # Large emoji updated in real time to show the current dominant emotion.  # Visual centerpiece
        self.dom_emoji = ctk.CTkLabel(dom, text="--",  # Initial state
                                      font=("Segoe UI Emoji", 40),  # Very large emoji font
                                      text_color="#FFFFFF")  # Default color
        self.dom_emoji.grid(row=1, column=0)  # Center in box
  # Empty line for readability
        # Text label below the emoji showing the emotion name.  # Text counterpart
        self.dom_name = ctk.CTkLabel(dom, text="waiting...",  # Initial state
                                     font=("Georgia", 13, "bold"),  # Elegant bold font
                                     text_color="#445566")  # Muted color
        self.dom_name.grid(row=2, column=0, pady=(0, 8))  # Center at bottom of box
  # Empty line for readability
        # Bottom row showing FPS, detection count, and model status.  # Utility footer
        ff = ctk.CTkFrame(sb, fg_color="transparent")  # Clean container
        ff.grid(row=7, column=0, sticky="ew", padx=14, pady=(0, 10))  # Stick to bottom of sidebar
        ff.grid_columnconfigure(1, weight=1)  # Push detections to the right
  # Empty line for readability
        self.fps_lbl = ctk.CTkLabel(ff, text="FPS: --",  # Speed indicator
                                    font=("Consolas", 10),  # Techy font
                                    text_color="#334455")  # Muted color
        self.fps_lbl.grid(row=0, column=0, sticky="w")  # Align left
  # Empty line for readability
        self.det_lbl = ctk.CTkLabel(ff, text="Detections: 0",  # Face count indicator
                                    font=("Consolas", 10),  # Monospace
                                    text_color="#334455")  # Muted color
        self.det_lbl.grid(row=0, column=1, sticky="e")  # Align right
  # Empty line for readability
        self.mdl_lbl = ctk.CTkLabel(ff, text="Model: not loaded",  # Model path indicator
                                    font=("Consolas", 9),  # Very small font
                                    text_color="#223344")  # Dark color
        self.mdl_lbl.grid(row=1, column=0, columnspan=2, sticky="w")  # Span across footer
  # Empty line for readability
    # -- MODEL LOADING --  # Section for AI initialization
    def _load_models(self):  # Method to bring models into RAM
        # Load Stage 1: the face detector model.  # Stage 1 header
        print("\nLoading Stage-1 face detector (yolov8n-face.pt)...")  # Console log
        try:  # Start of safety block
            self.face_model = YOLO("yolov8n-face.pt")  # Attempt to load face-specific YOLO
            print("  Face detector ready")  # Success log
        except Exception as e:  # If specific face model is missing
            print(f"  Could not load yolov8n-face.pt: {e}")  # Log failure
            try:  # Attempt fallback
                # Fall back to the general-purpose YOLOv8n if the face-specific model is missing.  # Rationale
                self.face_model = YOLO("yolov8n.pt")  # Load generic nano model
                print("  Fallback: using yolov8n.pt (boxes may be less tight)")  # Success log
            except Exception:  # If even fallback fails
                self.face_model = None  # Clear variable
                print("  No face model loaded -- will use the whole frame as the face crop")  # Final fallback plan
  # Empty line for readability
        # Load Stage 2: the emotion classifier model.  # Stage 2 header
        # Search several common locations for the trained model file.  # Searching logic
        search_paths = [  # List of places we might have saved 'best.pt'
            "models/emotion_classifier/best.pt",  # Primary location
            "models/emotion_classifier/best.onnx",  # Optimized location
            "runs/detect/emotion_classifier/weights/best.pt",  # YOLO training output 1
            "runs/detect/emotion_classifier/weights/best.onnx",  # YOLO training output 2
            "models/emotion_detector/best.pt",  # Alternate name
            "models/emotion_detector/best.onnx",  # Alternate name optimized
            "runs/classify/emotion_classifier/weights/best.pt",  # YOLO training output 3
            "best.pt",  # Current directory
            "best.onnx",  # Current directory optimized
        ]  # End of search list
  # Empty line for readability
        cls_path = self.cls_model_path  # use the path passed on the command line if provided
        if not cls_path:  # If no custom path was given
            # Search each location in order and use the first one found.  # Auto-discovery
            for p in search_paths:  # Loop through the list
                if os.path.exists(p):  # If file exists at this path
                    cls_path = p  # We found it!
                    break  # Stop looking
  # Empty line for readability
        if not cls_path or not os.path.exists(cls_path):  # If we still haven't found a model
            # No emotion model found -- the app will open but cannot make predictions.  # Consequence
            self.mdl_lbl.configure(text="Emotion model: not found")  # Update UI footer
            print("\nNo emotion model found.")  # Console warning
            print("   Expected at: models/emotion_classifier/best.pt")  # Help user
            print("   Run training first:")  # Solution
            print("   python 2_train_model.py\n")  # Command to run
            return  # Stop loading
  # Empty line for readability
        try:  # Final attempt to load the actual file
            # Load the emotion classifier model.  # Core AI load
            self.emo_model = YOLO(cls_path)  # Pass the path to the YOLO constructor
            short = Path(cls_path).name  # Extract just the filename (e.g. "best.pt")
            self.mdl_lbl.configure(text=f"Model: {short}",  # Show filename in UI
                                   text_color="#00CC88")  # Green success color
            print(f"  Emotion classifier loaded: {cls_path}")  # Log full path
        except Exception as e:  # If loading the file itself fails (e.g. corrupted)
            self.mdl_lbl.configure(text="Emotion model: load failed")  # UI warning
            print(f"Error loading emotion model: {e}")  # Console error

    # -- CAMERA TOGGLE --  # Section for handling the webcam state
    def _toggle_cam(self):  # Method called when the "Start/Stop Camera" button is clicked
        # Switches the camera on or off depending on its current state.  # Logic switcher
        if self.camera_running:  # If it is currently active
            self._stop_cam()  # Call the stop method
        else:  # If it is currently inactive
            self._start_cam()  # Call the start method
  # Empty line for readability
    def _start_cam(self):  # Method to initialize the webcam stream
        # Opens the default webcam and starts the detection loop.  # Hardware setup
        if self.emo_model is None:  # If the AI model failed to load earlier
            self._err("No emotion model loaded!\nRun: python 2_train_model.py")  # Show error popup
            return  # Stop here
  # Empty line for readability
        self.cap = cv2.VideoCapture(0)  # Open the first available system camera (ID 0)
        if not self.cap.isOpened():  # If the camera couldn't be accessed
            self._err("Could not open webcam!")  # Show error popup
            return  # Stop here
  # Empty line for readability
        # Request a higher resolution from the webcam if it supports it.  # Quality settings
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)  # Try to set width to 720p
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # Try to set height to 720p
  # Empty line for readability
        self.camera_running = True  # Set the flag to indicate we are live
        self.btn_cam.configure(text="  Stop Camera",  # Update button text to "Stop"
                               fg_color="#6B1A1A", hover_color="#9A2222")  # Change button to red
        self.status_lbl.configure(text="  Camera running...",  # Update status bar
                                  text_color="#00CC88")  # Use green success color
        self._tick()  # start the frame-by-frame processing loop
  # Empty line for readability
    def _stop_cam(self):  # Method to gracefully close the webcam
        # Stops the camera and releases the hardware.  # Cleanup logic
        self.camera_running = False  # Set flag to inactive
        if self.cap:  # If the capture object exists
            self.cap.release()  # Release the camera back to the system
            self.cap = None  # Clear the object
        self.btn_cam.configure(text="  Start Camera",  # Reset button text to "Start"
                               fg_color="#1A6B4A", hover_color="#22996A")  # Reset button to green
        self.status_lbl.configure(text="  Stopped", text_color="#445566")  # Update status bar
  # Empty line for readability
    def _tick(self):  # The main application loop (runs once per frame)
        # Called repeatedly to process one frame at a time.  # Recursive timing
        # After each frame, it schedules itself to run again after 30ms (~33 FPS max).  # Refresh rate
        if not self.camera_running:  # If the user clicked "Stop"
            return  # End the loop
  # Empty line for readability
        ret, frame = self.cap.read()  # Grab the latest image from the webcam
        if not ret:  # If the camera feed was lost
            self._stop_cam()  # Shut down the camera interface
            return  # Exit
  # Empty line for readability
        t0  = time.time()  # Start a timer to measure processing speed
        out = self._detect(frame)          # run the two-stage detection pipeline  # AI Inference
        fps = 1.0 / max(time.time() - t0, 0.001)  # calculate frames per second  # FPS Math
        self._show(out)                    # display the annotated frame in the window  # Render results
        self.fps_lbl.configure(text=f"FPS: {fps:.1f}")  # Update the speed indicator in the UI
        self.root.after(30, self._tick)    # schedule the next frame in 30ms (creates the loop)
  # Empty line for readability
    # -- TWO-STAGE DETECTION --  # Section for the core AI logic
    def _detect(self, frame: np.ndarray) -> np.ndarray:  # Input: raw image; Output: image with boxes
        """  # Start of docstring
        Runs the full two-stage pipeline on a single frame.  # High-level goal
  # Empty line
        Stage 1: Uses yolov8n-face to find face bounding boxes in the frame.  # Detection part
        Stage 2: Crops each face and runs the emotion classifier on the crop.  # Classification part
  # Empty line
        Returns the frame with boxes, labels, and confidence scores drawn on it.  # Result description
        """  # End of docstring
        conf_thresh = self.conf_slider.get()  # get the current minimum confidence from the UI slider
        fh, fw = frame.shape[:2]  # get original image height and width
        annotated = frame.copy()  # work on a copy so we don't ruin the original source frame
  # Empty line for readability
        top_emo  = None    # tracks which emotion has the highest confidence in the whole frame
        top_conf = 0.0  # tracks that highest confidence value
        finals   = []      # list to store successful detections (box, score, name)
  # Empty line for readability
        # Stage 1: Run the face detector.  # First pass AI
        if self.face_model is not None:  # If we successfully loaded the face detector
            face_results = self.face_model(  # Run the model on the current frame
                frame,  # The image array
                conf    = FACE_CONF,  # Use our predefined face threshold (low for sensitivity)
                iou     = 0.40,  # Overlap threshold for NMS
                imgsz   = 640,   # YOLO processes images internally at 640x640 resolution
                verbose = False,  # Don't flood the console with detection logs
            )[0]  # Get the first result object
  # Empty line for readability
            # Collect all detected face boxes into a list.  # Parsing results
            raw = []  # List for raw boxes
            if face_results.boxes is not None:  # If faces were found
                for box in face_results.boxes:  # Loop through each box found
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  # Convert coordinates to integers
                    fc = float(box.conf[0])  # Get the face detection confidence score
                    raw.append((x1, y1, x2, y2, fc))  # Add to our raw list
  # Empty line for readability
            # Apply NMS to remove duplicate detections of the same face.  # Cleaning results
            keep  = nms_boxes(raw, 0.40)  # Filter out overlaps
            faces = [raw[k] for k in keep]  # Final list of unique faces
        else:  # If no face model is loaded (fallback mode)
            # If no face model is loaded, treat the entire frame as one big face.  # Fallback logic
            faces = [(0, 0, fw, fh, 1.0)]  # Full frame box
  # Empty line for readability
        # Update the detection count label in the sidebar.  # UI Feedback
        self.det_lbl.configure(text=f"Detections: {len(faces)}")  # Show how many people are in view
  # Empty line for readability
        # Stage 2: Classify the emotion for each detected face.  # Second pass AI
        for (x1, y1, x2, y2, face_conf) in faces:  # Loop through every unique face found
  # Empty line for readability
            # Shrink the bounding box to focus tightly on the face.  # Pre-processing for Stage 2
            x1, y1, x2, y2 = tighten_box(x1, y1, x2, y2, fh, fw)  # Cut out background
  # Empty line for readability
            # Crop the face region out of the frame.  # Extraction
            crop = frame[y1:y2, x1:x2]  # Slice the image array
            if crop.size == 0:  # If the box ended up invalid (zero width/height)
                continue  # skip this one
  # Empty line for readability
            # Run the emotion classifier on the cropped face.  # Classification Inference
            emo_result = self.emo_model(crop, verbose=False)  # Predict emotion for the crop
            probs      = emo_result[0].probs  # Get probabilities for all 7 classes
            pred_cls   = int(probs.top1)        # get the index (0-6) of the most likely emotion
            pred_conf  = float(probs.top1conf)  # get the percentage score for that emotion
  # Empty line for readability
            # Skip this face if its confidence is below the threshold slider.  # User-defined filtering
            if pred_conf < conf_thresh:  # If the AI is not certain enough
                continue  # ignore this face
  # Empty line for readability
            # Get the emotion name from the model's class list.  # Mapping index to word
            # YOLO sorts class names alphabetically (angry, disgust, fear, happy, neutral, sad, surprise).
            cls_names = self.emo_model.names  # Get the list of names from the model itself
            pred_name = cls_names.get(pred_cls,  # Look up the name by index
                        EMOTIONS[pred_cls] if pred_cls < len(EMOTIONS) else "?")  # Fallback to our list
            pred_name = pred_name.lower()  # Normalize to lowercase
  # Empty line for readability
            finals.append((x1, y1, x2, y2, pred_conf, pred_name))  # Add to final rendering list
  # Empty line for readability
            # Track which emotion has the highest confidence in this frame.  # Primary face tracking
            if pred_conf > top_conf:  # If this person is more certain than the last one checked
                top_conf = pred_conf  # New top score
                top_emo  = pred_name  # New top emotion
  # Empty line for readability
        # Draw the detection results onto the annotated frame.  # Final Visualization
        for (x1, y1, x2, y2, confidence, emo) in finals:  # Loop through validated results
            color = EMOTION_COLORS_BGR.get(emo, (200, 200, 200))  # Get color for this emotion
            emoji = EMOJI_MAP.get(emo, "?")  # Get emoji for this emotion
  # Empty line for readability
            # Draw the main bounding box around the face.  # Rectangles
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)  # 2-pixel thick colored line
  # Empty line for readability
            # Draw a small square badge in the top-left corner of the box to hold the emoji.  # UI Decoration
            BADGE = 42  # Square size
            bx1, by1 = x1 + 3, y1 + 3  # Start position
            bx2, by2 = bx1 + BADGE, by1 + BADGE  # End position
            roi = annotated[by1:by2, bx1:bx2]  # Region of interest (the badge area)
            if roi.size > 0:  # If area is valid
                # Darken the badge background so the emoji is easier to read.  # Aesthetic touch
                annotated[by1:by2, bx1:bx2] = (roi * 0.35).astype(np.uint8)  # Reduce brightness to 35%
            cv2.rectangle(annotated, (bx1, by1), (bx2, by2), color, 1)  # Draw border around emoji
            annotated = draw_emoji(annotated, emoji, bx1 + 3, by1 + 2, 26)  # Put emoji in the badge
  # Empty line for readability
            # Draw the emotion name text just below the emoji badge.  # Labels
            ny = by2 + 11  # Text Y position
            (tw, th), _ = cv2.getTextSize(emo, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)  # Calculate text size
            cv2.rectangle(annotated, (bx1, ny - th - 3), (bx1 + tw + 5, ny + 2),  # Background box
                          (8, 12, 22), -1)  # Filled dark background
            cv2.putText(annotated, emo, (bx1 + 2, ny),  # The word (e.g. "happy")
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, color, 1, cv2.LINE_AA)  # Colored text
  # Empty line for readability
            # Draw a small confidence percentage in the top-right corner of the bounding box.  # Confidence Score
            ct = f"{confidence:.0%}"  # Format as "85%"
            (cw, ch), _ = cv2.getTextSize(ct, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)  # Calculate text size
            cv2.rectangle(annotated, (x2 - cw - 7, y1), (x2, y1 + ch + 5),  # Background box
                          color, -1)  # Filled with emotion color
            cv2.putText(annotated, ct, (x2 - cw - 3, y1 + ch + 1),  # The percentage text
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1,  # White text
                        cv2.LINE_AA)  # Anti-aliased font
  # Empty line for readability
            # Increment the count for this emotion in the running statistics.  # Tracking for charts
            if emo in self.emotion_counts:  # If we recognize the emotion
                self.emotion_counts[emo] += 1  # Add to historical tally
  # Empty line for readability
        # Update the "Current Emotion" display in the sidebar.  # UI update
        self._update_dom(top_emo)  # Pass the best detection to the display method
  # Empty line for readability
        # Update the progress bars if at least one detection was made this frame.  # Chart update
        if finals:  # If faces are in view
            self._update_bars()  # Refresh the bar charts
  # Empty line for readability
        return annotated  # Return the finished image with all AI overlays
  # E    def _update_dom(self, emo):  # Method to update the primary display box
        # Updates the large emoji and name in the "Current Emotion" box.  # Purpose
        if emo and emo in EMOTION_COLORS_HEX:  # If an emotion was detected
            c = EMOTION_COLORS_HEX[emo]  # Get the hex color
            self.dom_emoji.configure(text=EMOJI_MAP.get(emo, "?"), text_color=c)  # Set large emoji and color
            self.dom_name.configure(text=emo.capitalize(), text_color=c)  # Set capitalized name and color
        else:  # If no face is in the camera feed
            # Show a dash and a dim message when no face is detected.  # Idle state
            self.dom_emoji.configure(text="--", text_color="#FFFFFF")  # Reset emoji
            self.dom_name.configure(text="no face detected", text_color="#334455")  # Reset text
  # Empty line for readability
    # -- DISPLAY --  # Section for rendering the image into the GUI
    def _show(self, frame: np.ndarray):  # Input: OpenCV BGR image
        # Converts an OpenCV BGR frame to a CTkImage and shows it in the camera label.  # Conversion logic
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)   # OpenCV uses BGR, but PIL/CTk needs RGB format
        h, w   = rgb.shape[:2]  # Get height and width of the processed frame
  # Empty line for readability
        # Scale the frame to fit inside the display area while preserving aspect ratio.  # Resizing
        scale  = min(CAM_W / w, CAM_H / h)  # Calculate scaling factor to fit within CAM_W x CAM_H
        nw, nh = int(w * scale), int(h * scale)  # Calculate new dimensions
        res    = cv2.resize(rgb, (nw, nh))  # Perform the resize operation
  # Empty line for readability
        # Create a blank black canvas of the display area size and paste the scaled frame in the center.  # Padding
        canvas = np.zeros((CAM_H, CAM_W, 3), dtype=np.uint8)  # Black background image
        ox = (CAM_W - nw) // 2   # calculate horizontal offset to center the image
        oy = (CAM_H - nh) // 2   # calculate vertical offset to center the image
        canvas[oy:oy + nh, ox:ox + nw] = res  # Paste the scaled image onto the center of the canvas
  # Empty line for readability
        # Convert to a CTkImage and update the label.  # Final render
        pil     = Image.fromarray(canvas)  # Convert numpy array to PIL Image
        ctk_img = ctk.CTkImage(light_image=pil, dark_image=pil,  # Create CustomTkinter image
                                size=(CAM_W, CAM_H))  # Set display size
        self.cam_lbl.configure(image=ctk_img, text="")  # Put image in label, remove placeholder text
        self.cam_lbl.image = ctk_img  # keep a reference so Python doesn't garbage-collect the image
  # Empty line for readability
    # -- LOAD IMAGE --  # Section for static image analysis
    def _load_image(self):  # Method called when "Load Image" button is clicked
        # Opens a file picker dialog and runs the detection pipeline on a static image.  # Manual analysis
        if self.emo_model is None:  # If model is missing
            self._err("No emotion model loaded!\nRun: python 2_train_model.py")  # Show error
            return  # Stop
  # Empty line for readability
        from tkinter import filedialog  # Import file picker
        fp = filedialog.askopenfilename(  # Open the Windows file explorer dialog
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp")])  # Limit to common images
        if not fp:  # If user closed the window without selecting anything
            return  # user cancelled the dialog
  # Empty line for readability
        img = cv2.imread(fp)  # Read the image file from disk
        if img is None:  # If the file couldn't be read (e.g. invalid format)
            self._err("Could not load image.")  # Show error
            return  # Stop
  # Empty line for readability
        out = self._detect(img)    # run the exact same two-stage pipeline as the live camera loop
        self._show(out)  # Render the result in the window
        self.status_lbl.configure(text=f"  {Path(fp).name}",  # Show the filename in the status bar
                                  text_color="#AABBCC")  # Use a muted color
  # Empty line for readability
    # -- STATISTICS BARS --  # Section for data analytics visualization
    def _update_bars(self):  # Method to refresh the sidebar charts
        # Recalculates the percentage for each emotion and updates the progress bars.  # Analytics logic
        total = sum(self.emotion_counts.values())  # Get the total number of detections ever made
        if not total:  # If no detections have been made yet
            return  # Stop
  # Empty line for readability
        for emo in EMOTIONS:  # Loop through all 7 emotions
            pct = self.emotion_counts[emo] / total * 100  # Calculate share of the total
            self.emo_bars[emo].set(pct / 100)                  # progress bar expects value 0.0 to 1.0
            self.emo_lbls[emo].configure(text=f"{pct:.1f}%")   # update the percentage text label (e.g. "15.4%")
  # Empty line for readability
    def _on_conf(self, val):  # Method called when the slider moves
        # Called every time the confidence slider is moved. Updates the numeric display.  # Event handler
        self.conf_val.configure(text=f"{float(val):.2f}")  # Update the small number label
  # Empty line for readability
    # -- ERROR DIALOG --  # Section for user notifications
    def _err(self, msg):  # Method to show a custom popup
        # Shows a small error popup window with a message and an OK button.  # Alert logic
        w = ctk.CTkToplevel(self.root)  # Create a new top-level window
        w.title("Error")  # Set title
        w.geometry("360x130")  # Set size
        w.grab_set()  # block interaction with the main window until this is dismissed (modal)
        ctk.CTkLabel(w, text=f"  {msg}",  # Show the error message
                     font=("Segoe UI", 12), wraplength=320).pack(  # Wrap long text
            expand=True, pady=16)  # Padding
        ctk.CTkButton(w, text="OK", command=w.destroy, width=70).pack(  # OK button to close popup
            pady=(0, 12))  # Bottom padding
  # Empty line for readability
    # -- RUN --  # Section for app startup
    def run(self):  # Final method to launch the application
        # Prints a startup summary and launches the main window event loop.  # Entry point
        print("\n" + "=" * 55)  # Visual divider
        print("  EmoSense  by HamzaChan  --  v6 (RAF-DB Edition)")  # App branding
        print("  Two-stage: face detect -> emotion classify")  # Pipeline description
        print(f"  Box factor : SQUARE_FACTOR={SQUARE_FACTOR}")  # Config info
        print(f"  Confidence : default {DEFAULT_CONF}")  # Config info
        print("=" * 55 + "\n")  # Visual divider
        self.root.mainloop()  # starts the Tkinter event loop (blocks here until window closes)
  # Empty line for readability
  # Empty line for readability
# --- ENTRY POINT ---  # Section for the script execution logic
  # Empty line for readability
def main():  # Standard entry function
    # Parse optional command-line arguments.  # CLI handling
    p = argparse.ArgumentParser(description="EmoSense -- Emotion Recognition App v6")  # Setup parser
    p.add_argument("--model", default=None,  # Allow user to specify a model file
                   help="Path to emotion classifier model (.pt or .onnx). "  # Help text
                        "If omitted, the app searches common locations automatically.")  # Help text
    args = p.parse_args()  # Run the parser
  # Empty line for readability
    # Create the app and start the window.  # Launching sequence
    EmotionDetectorApp(model_path=args.model).run()  # Instantiate and call .run()
  # Empty line for readability
  # Empty line for readability
if __name__ == "__main__":  # Boilerplate to check if script is being run directly
    main()  # Call the main function
  # Empty line for readability
