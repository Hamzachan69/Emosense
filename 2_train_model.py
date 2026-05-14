"""  # Start of a multi-line docstring to describe the script
EmoSense by Hamza Khan - TRAINING SCRIPT v3 (RAF-DB Numbered Folders)  # Title of the script
  # Empty line for readability
PIPELINE (Two-Stage):  # Section explaining the overall AI structure
  Stage 1 -> YOLOv8n-face  : detects tight face bounding boxes  # First stage: finding the face
  Stage 2 -> YOLOv8s-cls   : classifies emotion from each cropped face  # Second stage: identifying the emotion
  # Empty line for readability
YOUR ACTUAL FOLDER STRUCTURE (what this script reads):  # Section describing expected input files
  face_emotion/data/RAF DB/DATASET/  # The main folder containing the images
      train/  # Training data folder
          1/   <- Surprise  # Subfolder '1' contains surprise images
          2/   <- Fear  # Subfolder '2' contains fear images
          3/   <- Disgust  # Subfolder '3' contains disgust images
          4/   <- Happy  # Subfolder '4' contains happy images
          5/   <- Sad  # Subfolder '5' contains sad images
          6/   <- Angry  # Subfolder '6' contains angry images
          7/   <- Neutral  # Subfolder '7' contains neutral images
      test/  # Test data folder (used for validation)
          1/ 2/ 3/ 4/ 5/ 6/ 7/   (same mapping)  # Same numbered folders as in 'train'
  # Empty line for readability
HOW TO RUN:  # Section with instructions on how to execute this script
  python 2_train_model.py  # Command to run the script
  (no arguments needed -- paths are pre-configured for your structure)  # Note about default settings
  # Empty line for readability
OPTIONAL ARGUMENTS:  # Section listing extra settings you can change
  --epochs 80          (default: 80)  # Setting to control how long the training lasts
  --batch  32          (default: 32)  # Setting to control how many images are processed at once
  --model  yolov8n-cls.pt   (default: yolov8n-cls.pt -- Nano version for faster training)  # Choice of the starting AI model
"""  # End of the multi-line docstring
  # Empty line for readability
import os  # Import the 'os' module to perform system tasks like creating or deleting folders
import sys  # Import the 'sys' module to interact with the Python interpreter (like stopping the script)
import time  # Import the 'time' module to track how long tasks take to complete
import shutil  # Import the 'shutil' module for high-level file operations (like copying files)
import argparse  # Import the 'argparse' module to handle command-line inputs/options
from pathlib import Path  # Import 'Path' from 'pathlib' for an easier way to work with file paths
  # Empty line for readability
# Try to import all required libraries. If any are missing, tell the user and stop.  # Comment explaining the next block
try:  # Start a 'try' block to catch errors if libraries aren't installed
    import torch  # Import 'torch' (PyTorch), the core library for deep learning
    import cv2  # Import 'cv2' (OpenCV), used for processing images and video
    import numpy as np  # Import 'numpy' for fast mathematical operations on image arrays
    import matplotlib.pyplot as plt  # Import 'matplotlib' to create charts and plots of training progress
    from tqdm import tqdm  # Import 'tqdm' to show progress bars in the terminal
    from ultralytics import YOLO  # Import 'YOLO' from 'ultralytics' to use the YOLOv8 AI model
    from PIL import Image  # Import 'Image' from 'Pillow' for extra image handling capabilities
except ImportError as e:  # If any of the above imports fail, execute this 'except' block
    print(f"\nMissing package: {e.name}")  # Print a message telling the user which library is missing
    print("   Run: pip install ultralytics torch torchvision opencv-python "  # Provide the command to fix it
          "matplotlib tqdm pillow\n")  # Continuation of the installation command
    sys.exit(1)  # Stop the script with an error code (1) because we can't continue without libraries
  # Empty line for readability
# --- CLASS MAPPING ---  # Section for defining how data folders relate to emotion names
# The RAF-DB dataset uses numbered folders (1 through 7) instead of emotion names.  # Explanation of data format
# This dictionary maps each folder number to the emotion it represents.  # Explanation of the next variable
RAFDB_FOLDER_TO_NAME = {  # Define a dictionary for folder number to name mapping
    "1": "surprise",  # Folder '1' is mapped to 'surprise'
    "2": "fear",  # Folder '2' is mapped to 'fear'
    "3": "disgust",  # Folder '3' is mapped to 'disgust'
    "4": "happy",  # Folder '4' is mapped to 'happy'
    "5": "sad",  # Folder '5' is mapped to 'sad'
    "6": "angry",  # Folder '6' is mapped to 'angry'
    "7": "neutral",  # Folder '7' is mapped to 'neutral'
}  # End of the dictionary
  # Empty line for readability
# YOLO automatically sorts class names alphabetically when it reads the folders.  # Explanation of YOLO's behavior
# This dictionary shows what index (number) each emotion gets assigned.  # Explanation of index mapping
# Example: 0=angry, 1=disgust, 2=fear, 3=happy, 4=neutral, 5=sad, 6=surprise  # Examples of indices
EMOTION_CLASSES = {  # Define a dictionary for index to name mapping
    0: "angry",  # Index 0 is 'angry'
    1: "disgust",  # Index 1 is 'disgust'
    2: "fear",  # Index 2 is 'fear'
    3: "happy",  # Index 3 is 'happy'
    4: "neutral",  # Index 4 is 'neutral'
    5: "sad",  # Index 5 is 'sad'
    6: "surprise",  # Index 6 is 'surprise'
}  # End of the dictionary
  # Empty line for readability
# BGR color values used to draw colored boxes on the preview images.  # Explanation of the next variable
# Each emotion gets a distinct color so they are easy to tell apart visually.  # Purpose of using different colors
EMOTION_COLORS = {  # Define a dictionary for colors (Blue, Green, Red format)
    "neutral":  (0, 255, 0),  # Neutral is Green
    "happy":    (0, 255, 255),  # Happy is Yellow
    "angry":    (0, 0, 255),  # Angry is Red
    "sad":      (255, 0, 0),  # Sad is Blue
    "fear":     (128, 0, 128),  # Fear is Purple
    "surprise": (0, 165, 255),  # Surprise is Orange
    "disgust":  (0, 128, 128),  # Disgust is Teal
}  # End of the dictionary
  # Empty line for readability
# Emoji characters used in the console output to make the class distribution easier to read.  # Purpose of emojis
EMOJIS = {  # Define a dictionary for emotion emojis
    "neutral": "😐", "happy": "😊", "angry": "😠", "sad": "😢",  # Map names to emoji characters
    "fear": "😨",   "surprise": "😮", "disgust": "🤢",  # More mappings
}  # End of the dictionary
  # Empty line for readability
# The default path where the RAF-DB dataset is expected to be located.  # Explanation of data path
DEFAULT_DATA_PATH = "face_emotion/data/RAF DB/DATASET"  # String variable for the folder path
  # Empty line for readability


# --- BANNER ---  # Section for visual output in the terminal
  # Empty line for readability
def print_banner():  # Define a function to print a decorative banner
    # Prints a startup message to the console when the script begins.  # Function description
    print("""  # Start printing a multi-line string
+-----------------------------------------------------------------+  # Top border of the banner
|                                                                 |  # Empty line inside banner
|   EmoSense -- TRAINING PIPELINE  v3  (RAF-DB Edition)           |  # Title of the tool
|                                                                 |  # Empty line
|   Stage 1 : YOLOv8n-face   -> tight face detection              |  # Description of Stage 1
|   Stage 2 : YOLOv8s-cls    -> emotion classification            |  # Description of Stage 2
|   Dataset : RAF-DB numbered folders (1-7 -> 7 emotions)         |  # Description of the dataset
|                                                                 |  # Empty line
+-----------------------------------------------------------------+  # Bottom border of the banner
""")  # End of the print statement
  # Empty line for readability
  # Empty line for readability
# --- DATASET PREPARATION ---  # Section for rearranging the data folders
  # Empty line for readability
def prepare_rafdb_numbered(  # Define a function to prepare the dataset
    data_path: str,  # Argument: the path to the original RAF-DB folder
    output_path: str = "face_emotion/data/rafdb_yolo"  # Argument: where to save the reorganized data
) -> str:  # This function returns a string (the path to the output)
    """  # Start of function docstring
    Reads the RAF-DB folder layout (numbered subfolders 1-7) and converts it  # Goal of the function
    to the folder structure that YOLO expects for classification training.  # Why we are doing this
  # Empty line
    Input layout:  # Description of how the data looks now
        data_path/  # Root folder
            train/  1/  2/  3/  4/  5/  6/  7/  # Training subfolders
            test/   1/  2/  3/  4/  5/  6/  7/  # Testing subfolders
  # Empty line
    Output layout (YOLO classification format):  # Description of how it will look after
        output_path/  # New root folder
            train/  angry/  disgust/  fear/  happy/  neutral/  sad/  surprise/  # New train subfolders
            val/    angry/  disgust/  fear/  happy/  neutral/  sad/  surprise/  # New validation subfolders
    """  # End of function docstring
    print("\nPREPARING RAF-DB DATASET (numbered folder format)...")  # Print a status message
    print("-" * 60)  # Print a separator line
  # Empty line for readability
    # Build Path objects for the main data directory and its train/test subfolders.  # Path objects are easier to use than strings
    data_dir  = Path(data_path)  # Create a Path object for the dataset root
    train_dir = data_dir / "train"  # Create a Path object for the 'train' folder
    test_dir  = data_dir / "test"  # Create a Path object for the 'test' folder
  # Empty line for readability
    # Check that all required folders exist before doing any work.  # Validation step
    # If something is missing, stop immediately with a clear error message.  # Prevents crashes later
    if not data_dir.exists():  # Check if the root folder exists
        raise FileNotFoundError(  # Throw an error if it's missing
            f"\nDataset root not found: {data_dir}\n"  # Error message
            f"   Expected: face_emotion/data/RAF DB/DATASET\n"  # Clarification
            f"   Check your folder name -- spaces and capitals matter on Windows too.\n"  # Tip for Windows users
            f"   Tip: if path has a space, wrap it in quotes when calling from terminal."  # Another tip
        )  # End of error message
    if not train_dir.exists():  # Check if 'train' folder exists
        raise FileNotFoundError(  # Throw an error if it's missing
            f"\n'train' folder not found inside: {data_dir}\n"  # Error message
            f"   Expected subfolders: train/1  train/2 ... train/7"  # Clarification
        )  # End of error message
    if not test_dir.exists():  # Check if 'test' folder exists
        raise FileNotFoundError(  # Throw an error if it's missing
            f"\n'test' folder not found inside: {data_dir}\n"  # Error message
            f"   Expected subfolders: test/1  test/2 ... test/7"  # Clarification
        )  # End of error message
  # Empty line for readability
    # Create the output directory tree.  # Preparing the new folders
    # Each split (train, val) gets one subfolder per emotion name.  # Structure requirement for YOLO
    out_dir = Path(output_path)  # Create Path object for output directory
    emotions = list(EMOTION_CLASSES.values())  # Get the list of emotion names (alphabetical)
  # Empty line for readability
    for split in ["train", "val"]:  # Loop through both training and validation splits
        for emo in emotions:  # For each emotion name
            (out_dir / split / emo).mkdir(parents=True, exist_ok=True)  # Create the folder if it doesn't exist
  # Empty line for readability
    # Counters and a list to track missing folders across both splits.  # For reporting at the end
    total_copied = 0  # Initialize total image counter
    missing_folders = []  # Initialize list of missing subfolders
  # Empty line for readability
    def _copy_numbered_split(src_split_dir: Path, dst_split_name: str):  # Inner function to handle one split
        # Copies all images from one split (train or test) into the output folders.  # Function goal
        # Numbered source folders (1-7) are renamed to emotion name folders.  # Key transformation
        nonlocal total_copied  # Allow this function to modify the variable from the outer scope
        copied_this_split = 0  # Initialize counter for this specific split
  # Empty line for readability
        for num_str, emo_name in RAFDB_FOLDER_TO_NAME.items():  # Loop through the number-to-name mapping
            src_folder = src_split_dir / num_str   # Path to source (e.g., train/4)
            dst_folder = out_dir / dst_split_name / emo_name  # Path to destination (e.g., train/happy)
  # Empty line for readability
            # If a numbered folder doesn't exist, skip it and warn the user.  # Robustness check
            if not src_folder.exists():  # Check if source subfolder exists
                missing_folders.append(str(src_folder))  # Add to missing list
                print(f"  Warning: Missing folder: {src_folder} -- skipping")  # Print warning
                continue  # Go to the next folder in the loop
  # Empty line for readability
            # Collect all image files in the numbered folder.  # Finding all images
            imgs = (list(src_folder.glob("*.jpg")) +  # Find .jpg files
                    list(src_folder.glob("*.jpeg")) +  # Find .jpeg files
                    list(src_folder.glob("*.png")))  # Find .png files
  # Empty line for readability
            # Copy each image to the output folder, skipping any that already exist.  # Doing the work
            for img_path in tqdm(imgs,  # Loop through each image with a progress bar
                                 desc=f"  {dst_split_name}/{emo_name:8} (folder {num_str})",  # Progress bar label
                                 leave=False):  # Remove progress bar when done
                dst = dst_folder / img_path.name  # Create the destination file path
                if not dst.exists():  # Only copy if the file isn't already there (saves time)
                    shutil.copy2(img_path, dst)  # Copy the file while preserving metadata
                copied_this_split += 1  # Increment counter
  # Empty line for readability
        total_copied += copied_this_split  # Add this split's count to the grand total
        print(f"  {dst_split_name}: {copied_this_split:,} images copied")  # Print summary for this split
  # Empty line for readability
    # Run the copy process for both the training and validation (test) splits.  # Execute the inner function
    _copy_numbered_split(train_dir, "train")  # Process the training set
    _copy_numbered_split(test_dir,  "val")  # Process the test set (renaming 'test' to 'val')
  # Empty line for readability
    # Print a simple bar chart showing how many images exist per emotion in train.  # Visual summary of data
    print("\n  Class Distribution (train):")  # Header
    grand_total = 0  # Reset counter for final summary
    for emo in emotions:  # For each emotion name
        n = len(list((out_dir / "train" / emo).glob("*")))  # Count how many images are in that folder
        grand_total += n  # Add to total
        bar = "#" * max(1, int(n / 80))  # Create a visual bar using '#' characters (1 per 80 images)
        print(f"    {EMOJIS.get(emo,'?')} {emo:10}  {n:5,}  {bar}")  # Print emotion, count, and bar
    print(f"  Total train images : {grand_total:,}")  # Print grand total of training images
  # Empty line for readability
    val_total = sum(  # Calculate total validation images
        len(list((out_dir / "val" / emo).glob("*"))) for emo in emotions  # Sum up counts from all val subfolders
    )  # End of sum
    print(f"  Total val   images : {val_total:,}")  # Print total validation images
  # Empty line for readability
    # If nothing was copied at all, something is wrong with the folder structure.  # Critical error check
    if grand_total == 0:  # If zero images were found
        raise RuntimeError(  # Throw an error
            "\nNo images were copied!\n"  # Error message
            "   Check that your numbered folders (1-7) actually contain .jpg/.png images.\n"  # Advice
            f"   Looked inside: {train_dir}"  # Where we looked
        )  # End of error message
  # Empty line for readability
    print(f"\n  Dataset prepared at: {out_dir}")  # Final status message
    return str(out_dir)  # Return the output path as a string
  # Empty line for readability


# --- DEVICE RESOLVER ---  # Section for determining hardware usage
  # Empty line for readability
def resolve_device(requested: str) -> str:  # Define function to choose GPU or CPU
    # Decides which hardware device to use for training (GPU or CPU).  # Function description
    # If the user asked for GPU (cuda) but none is available, fall back to CPU.  # Handling missing hardware
    if requested in ("auto", "cuda", "0", "cuda:0"):  # If the user wants an automatic choice or GPU
        if torch.cuda.is_available():  # Check if a CUDA-capable NVIDIA GPU is found
            gpu  = torch.cuda.get_device_name(0)  # Get the name of the first GPU
            vram = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3  # Calculate total VRAM in GB
            print(f"\n  GPU  : {gpu}  ({vram:.1f} GB VRAM)")  # Print GPU name and memory
            print("  Using CUDA")  # Print that we are using the GPU engine
            return "0"  # Return the index of the first GPU
        else:  # If no GPU is available
            print("\n  No CUDA GPU found -- falling back to CPU (will be slow).")  # Warn the user
            return "cpu"  # Return 'cpu' instead
    # If the user passed a specific device string (like "cpu"), use it as-is.  # Manual override
    return requested  # Return the user's requested device string
  # Empty line for readability
  # Empty line for readability
# --- TRAIN CLASSIFIER ---  # Section for the actual AI training process
  # Empty line for readability
def train_classifier(  # Define the training function
    data_dir:   str,  # Argument: path to the prepared dataset
    model_name: str = "yolov8s-cls.pt",   # Argument: starting model weights (default: small)
    epochs:     int = 80,  # Argument: number of training rounds
    batch:      int = 64,  # Argument: number of images processed at once
    imgsz:      int = 224,  # Argument: image dimensions (224x224 pixels)
    device:     str = "auto",  # Argument: hardware choice
) -> YOLO:  # This function returns the trained YOLO model object
    """  # Start of docstring
    Trains the Stage-2 emotion classifier on the prepared RAF-DB data.  # Goal
    Uses YOLOv8s-cls (small) for better accuracy than the nano variant.  # Recommendation
    An RTX 3070 handles batch=64 at imgsz=224 without running out of memory.  # Hardware tip
    """  # End of docstring
    # Resolve whether we are training on GPU or CPU.  # Get the final device string
    device = resolve_device(device)  # Call the resolve_device function defined above
  # Empty line for readability
    print("\n" + "=" * 65)  # Print header border
    print("  STAGE 2 -- EMOTION CLASSIFIER TRAINING")  # Print section title
    print("=" * 65)  # Print border
    print(f"  Model      : {model_name}")  # Print chosen model
    print(f"  Epochs     : {epochs}")  # Print number of epochs
    print(f"  Batch size : {batch}")  # Print batch size
    print(f"  Image size : {imgsz}x{imgsz}")  # Print image resolution
    print(f"  Device     : {device}")  # Print hardware being used
    print(f"  Data dir   : {data_dir}")  # Print location of training data
    print("=" * 65 + "\n")  # Print bottom border
  # Empty line for readability
    # Load the pre-trained YOLO model. It already knows how to recognize shapes  # Transfer learning concept
    # and edges -- we fine-tune it on top of that to recognize emotions.  # Our specific task
    model = YOLO(model_name)  # Initialize the model using the provided weights file
  # Empty line for readability
    model.train(  # Start the training process
        data    = data_dir,  # Tell it where the images are
        task    = "classify",  # Specify that this is a classification task
        epochs  = epochs,  # Set the duration of training
        imgsz   = imgsz,  # Set the resolution
        batch   = batch,  # Set the parallel processing count
        device  = device,  # Set the hardware device
  # Empty line for readability
        # --- Augmentation settings ---  # Techniques to make the model more robust
        # These randomly modify training images so the model doesn't just memorize  # Purpose: Generalization
        # the exact photos it has seen. This is especially important for RAF-DB  # Handling dataset issues
        # because some emotion classes have far fewer examples than others.  # Balancing the learning
        hsv_h     = 0.015,   # Randomly shift color hue by 1.5%
        hsv_s     = 0.5,     # Randomly change color saturation by 50%
        hsv_v     = 0.4,     # Randomly change brightness by 40%
        degrees   = 15.0,    # Randomly rotate images up to 15 degrees
        translate = 0.15,    # Randomly move images up/down/left/right by 15%
        scale     = 0.5,     # Randomly zoom in or out by 50%
        fliplr    = 0.5,     # Flip images horizontally 50% of the time (mirrored faces)
        flipud    = 0.0,     # Don't flip upside down (unnatural for faces)
        erasing   = 0.4,     # Randomly hide 40% of the image (forces model to look at other features)
        mixup     = 0.1,     # Blend two images together (10% of the time) to help minority classes
  # Empty line for readability
        # --- Optimizer settings ---  # Math that controls how the AI learns
        # AdamW is an optimizer that adjusts how fast the model learns over time.  # Modern standard optimizer
        optimizer     = "AdamW",  # Use the AdamW algorithm
        lr0           = 0.001,      # The initial learning speed (learning rate)
        lrf           = 0.01,       # The final learning rate (as a percentage of the start)
        warmup_epochs = 3,          # Gradually increase learning speed for the first 3 rounds
        patience      = 20,         # Stop early if no improvement for 20 rounds (saves time)
  # Empty line for readability
        # --- Output settings ---  # Where to save the results
        project  = "runs/detect",              # Main results folder
        name     = "emotion_classifier",       # Specific experiment folder
        exist_ok = True,                       # If folder exists, overwrite/reuse it
        save     = True,                       # Save the finished model files
        plots    = True,                       # Create accuracy and loss charts
        verbose  = True,                       # Show detailed logs in terminal
    )  # End of train function call
  # Empty line for readability
    print("\nSTAGE 2 TRAINING COMPLETE!")  # Notify user training finished
    return model  # Return the model object for further use (like exporting)
  # Empty line for readability
  # Empty line for readability
# --- EXPORT TO ONNX ---  # Section for converting the model format
  # Empty line for readability
def export_classifier(model: YOLO) -> str | None:  # Define function to convert to ONNX
    # Converts the trained PyTorch model to ONNX format.  # Goal of the function
    # ONNX is a universal model format that runs faster at inference time  # Performance benefit
    # and does not require the full PyTorch library to be installed.  # Dependency benefit
    print("\nEXPORTING CLASSIFIER TO ONNX...")  # Status message
    try:  # Try-except block in case export fails
        path = model.export(  # Start the export process
            format   = "onnx",  # Target format
            imgsz    = [224, 224],   # Image size for the exported model
            simplify = True,         # Remove unnecessary math steps for speed
            opset    = 12,           # Compatibility version 12
            half     = False,        # Keep standard 32-bit precision ( safer compatibility)
            dynamic  = False,        # Use fixed image size for maximum performance
            verbose  = False,        # Don't print internal export logs
        )  # End of export call
        print(f"  ONNX saved: {path}")  # Show where it was saved
        return str(path)  # Return the path to the new file
    except Exception as e:  # If something went wrong
        print(f"  ONNX export failed: {e}")  # Print the error
        print("  The PyTorch .pt model still works fine in the app.")  # Reassurance
        return None  # Return nothing
  # Empty line for readability
  # Empty line for readability
# --- SAMPLE PREDICTIONS ---  # Section for visual verification
  # Empty line for readability
def generate_samples(cls_model: YOLO, data_dir: str, num: int = 9):  # Function to test the model
    """  # Start of docstring
    Runs the full two-stage pipeline on a few validation images and saves  # What it does
    a grid of annotated results to outputs/sample_predictions.png.  # Where it saves
    This lets you quickly see how well the model is performing after training.  # Why it's useful
    """  # End of docstring
    print("\nGENERATING SAMPLE PREDICTIONS...")  # Status message
  # Empty line for readability
    val_dir = Path(data_dir) / "val"  # Path to validation images
    if not val_dir.exists():  # If it doesn't exist
        print("  No val dir found, skipping")  # Print warning
        return  # Stop function
  # Empty line for readability
    # Collect one image per emotion class from the validation folder.  # Diverse testing
    samples = []  # List to store (image_path, emotion_name)
    for cls_folder in sorted(val_dir.iterdir()):  # Loop through folders in validation directory
        if not cls_folder.is_dir():  # Skip files that aren't folders
            continue  # Go to next item
        imgs = list(cls_folder.glob("*.jpg")) + list(cls_folder.glob("*.png"))  # Find images
        if imgs:  # If images were found
            samples.append((imgs[0], cls_folder.name))  # Take the first one found
        if len(samples) >= num:  # Stop when we have enough samples
            break  # Exit loop
  # Empty line for readability
    if not samples:  # If no images were found at all
        print("  No sample images found in val/")  # Print warning
        return  # Stop function
  # Empty line for readability
    # Load the Stage-1 face detector so we can run the full two-stage pipeline.  # Need face detection for full test
    face_model = YOLO("yolov8n-face.pt")  # Initialize the face detector
  # Empty line for readability
    # Set up a matplotlib figure with one cell per sample image.  # Preparing the output image
    cols = 3  # 3 images per row
    rows = (len(samples) + cols - 1) // cols  # Calculate needed rows
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))  # Create the grid
    axes = np.array(axes).flatten()  # Flatten grid for easier looping
  # Empty line for readability
    for idx, (img_path, true_label) in enumerate(samples):  # Loop through each sample image
        img = cv2.imread(str(img_path))  # Read the image from disk
        if img is None:  # If reading failed
            continue  # Skip to next image
        annotated = img.copy()  # Create a copy to draw on
  # Empty line for readability
        # Stage 1: Run the face detector on the image.  # Find faces first
        face_results = face_model(img, conf=0.3, verbose=False)  # Detect faces with 30% confidence
        faces_found  = False  # Track if any faces were actually found
  # Empty line for readability
        if face_results[0].boxes is not None and len(face_results[0].boxes):  # If face boxes exist
            for box in face_results[0].boxes:  # Loop through each face found
                # Get the bounding box coordinates as integers.  # Pixel positions
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Convert coordinates to ints
                w, h   = x2 - x1, y2 - y1  # Calculate width and height
  # Empty line for readability
                # Crop a tight square around the center of the detected face.  # Centering the head
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2  # Horizontal and vertical center
                half   = int(min(w, h) * 0.45)  # Calculate half-size for square crop
                x1 = max(0, cx - half);  x2 = min(img.shape[1], cx + half)  # Calculate left/right edges
                y1 = max(0, cy - half);  y2 = min(img.shape[0], cy + half)  # Calculate top/bottom edges
  # Empty line for readability
                # Cut out just the face region.  # Slicing the image array
                crop = img[y1:y2, x1:x2]  # Create the crop
                if crop.size == 0:  # If crop is invalid
                    continue  # Skip
  # Empty line for readability
                # Stage 2: Run the emotion classifier on the cropped face.  # The main AI task
                emo_result = cls_model(crop, verbose=False)  # Predict emotion
                probs      = emo_result[0].probs  # Get probabilities
                pred_cls   = int(probs.top1)         # Get index of best guess
                pred_name  = EMOTION_CLASSES.get(pred_cls, "?")  # Map index to name
                conf       = float(probs.top1conf)   # Get confidence percentage
                color      = EMOTION_COLORS.get(pred_name, (255, 255, 255))  # Get color for box
  # Empty line for readability
                # Draw the bounding box and the predicted label on the image.  # Visualization
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)  # Draw box
                cv2.putText(annotated, f"{pred_name} {conf:.0%}",  # Draw text label
                            (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX,  # Positioned above box
                            0.6, color, 2, cv2.LINE_AA)  # Style settings
                faces_found = True  # Mark that we found and classified a face
  # Empty line for readability

        # If no face was detected, run the emotion model on the whole image instead.  # Fallback logic
        if not faces_found:  # If the face detector found zero faces
            emo_result = cls_model(img, verbose=False)  # Run the classifier on the entire original image
            probs      = emo_result[0].probs  # Get prediction probabilities
            pred_name  = EMOTION_CLASSES.get(int(probs.top1), "?")  # Get the top predicted emotion name
            conf       = float(probs.top1conf)  # Get the confidence score
            cv2.putText(annotated, f"{pred_name} {conf:.0%}",  # Draw the result on the image
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,  # Position: top-left corner
                        0.8, (0, 255, 0), 2, cv2.LINE_AA)  # Green color, larger font
  # Empty line for readability
        # Convert from OpenCV's BGR to RGB so matplotlib displays colors correctly.  # OpenCV uses BGR, Matplotlib uses RGB
        rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)  # Perform the color space conversion
        axes[idx].imshow(rgb)  # Add the image to the current grid cell
        axes[idx].set_title(f"True: {true_label}", fontsize=9)  # Set cell title to the correct emotion name
        axes[idx].axis("off")  # Hide the X and Y axis for a cleaner look
  # Empty line for readability
    # Turn off any unused subplot cells (if num samples < rows * cols).  # Cleanup step
    for i in range(len(samples), len(axes)):  # For every cell that didn't get an image
        axes[i].axis("off")  # Disable that cell
  # Empty line for readability
    plt.suptitle("Sample Predictions -- Two-Stage Pipeline (face detect -> classify)",  # Main title for the whole grid
                 fontsize=13, fontweight="bold")  # Style settings for the title
    plt.tight_layout()  # Automatically adjust spacing between subplots
  # Empty line for readability
    # Save the grid image to the outputs folder.  # Persistence
    os.makedirs("outputs/snapshots", exist_ok=True)  # Create the output folder if it's missing
    out = "outputs/sample_predictions.png"  # Set the final filename
    plt.savefig(out, dpi=130, bbox_inches="tight")  # Save the figure to disk with high quality
    plt.close()  # Close the plot to free up memory
    print(f"  Saved: {out}")  # Notify the user where the image is
  # Empty line for readability
  # Empty line for readability
# --- MAIN ---  # The starting point of the script
  # Empty line for readability
def main():  # Define the main function
    print_banner()  # Show the decorative banner defined earlier
  # Empty line for readability
    # Set up command-line argument parsing so the user can override defaults.  # Handling terminal options
    parser = argparse.ArgumentParser(description="EmoSense -- Train on RAF-DB (numbered folders)")  # Initialize parser
    parser.add_argument("--data",         type=str, default=DEFAULT_DATA_PATH,  # Add option for data path
                        help="Path to RAF DB/DATASET folder (contains train/ and test/)")  # Help text
    parser.add_argument("--model",        type=str, default="yolov8s-cls.pt",  # Add option for model choice
                        help="yolov8s-cls.pt (recommended) or yolov8n-cls.pt (faster, less accurate)")  # Help text
    parser.add_argument("--epochs",       type=int, default=80,  # Add option for training length
                        help="Number of training epochs")  # Help text
    parser.add_argument("--batch",        type=int, default=64,  # Add option for batch size
                        help="Batch size (64 fits an RTX 3070 8 GB)")  # Help text
    parser.add_argument("--img-size",     type=int, default=224)  # Add option for resolution
    parser.add_argument("--device",       type=str, default="auto")  # Add option for hardware
    parser.add_argument("--skip-export",  action="store_true",  # Add option to skip ONNX conversion
                        help="Skip the ONNX export step")  # Help text
    parser.add_argument("--skip-samples", action="store_true",  # Add option to skip sample generation
                        help="Skip generating sample prediction images")  # Help text
    args = parser.parse_args()  # Actually read the arguments from the terminal
  # Empty line for readability
    # Print environment info so the user can confirm their GPU is being used.  # Diagnostic summary
    print("\nSYSTEM INFO:")  # Header
    print(f"  Python  : {sys.version.split()[0]}")  # Show Python version
    print(f"  PyTorch : {torch.__version__}")  # Show PyTorch version
    print(f"  CUDA    : {'Available' if torch.cuda.is_available() else 'Not available'}")  # Show GPU availability
    if torch.cuda.is_available():  # If GPU is found
        print(f"  GPU     : {torch.cuda.get_device_name(0)}")  # Show its specific name
  # Empty line for readability
    print(f"\n  Dataset path : {args.data}")  # Confirm data location
    print(f"  Model        : {args.model}")  # Confirm starting weights
    print(f"  Epochs       : {args.epochs}")  # Confirm epoch count
    print(f"  Batch size   : {args.batch}")  # Confirm batch size
  # Empty line for readability
    # Record the start time so we can report total duration at the end.  # Timing
    start_time = time.time()  # Get current time
  # Empty line for readability
    try:  # Start main execution block
        # Step 1: Convert the RAF-DB numbered folders to YOLO classification format.  # Reorganizing files
        prepared_dir = prepare_rafdb_numbered(  # Call the preparation function
            data_path   = args.data,  # Pass the data path
            output_path = "face_emotion/data/rafdb_yolo",  # Pass the output path
        )  # End of call
  # Empty line for readability
        # Step 2: Train the emotion classifier on the prepared dataset.  # The core task
        model = train_classifier(  # Call the training function
            data_dir   = prepared_dir,  # Pass the prepared directory
            model_name = args.model,  # Pass chosen model
            epochs     = args.epochs,  # Pass epoch count
            batch      = args.batch,  # Pass batch size
            imgsz      = args.img_size,  # Pass image size
            device     = args.device,  # Pass hardware choice
        )  # End of call
  # Empty line for readability
        # Step 3: Copy the best model weights to the models/ folder so the app can find them.  # Deployment
        final_dir = Path("models/emotion_classifier")  # Path to the final model folder
        final_dir.mkdir(parents=True, exist_ok=True)  # Create it if it doesn't exist
  # Empty line for readability
        best_pt = Path(model.trainer.save_dir) / "weights" / "best.pt"  # Locate the best .pt file
        if best_pt.exists():  # If it was found
            shutil.copy2(best_pt, final_dir / "best.pt")  # Copy it to our 'models' folder
            print(f"\n  Best .pt  -> {final_dir}/best.pt")  # Notify user
  # Empty line for readability
        # Step 4: Export the trained model to ONNX format (unless skipped).  # Conversion
        if not args.skip_export:  # If the user didn't ask to skip
            export_classifier(model)  # Call the export function
            best_onnx = Path(model.trainer.save_dir) / "weights" / "best.onnx"  # Locate best .onnx file
            if best_onnx.exists():  # If it exists
                shutil.copy2(best_onnx, final_dir / "best.onnx")  # Copy it to 'models' folder
                print(f"  Best .onnx -> {final_dir}/best.onnx")  # Notify user
  # Empty line for readability
        # Step 5: Generate a sample prediction grid image (unless skipped).  # Visualization
        if not args.skip_samples:  # If user didn't ask to skip
            generate_samples(model, prepared_dir)  # Call the sample generation function
  # Empty line for readability
        print("\n" + "=" * 70)  # Final header
        print("  ALL DONE!")  # Success message
        print("=" * 70)  # Header
        print(f"  Model saved at  : {final_dir}/best.pt")  # Recap location
        print(f"  Training logs   : {model.trainer.save_dir}")  # Recap logs location
        print(f"  Sample grid     : outputs/sample_predictions.png")  # Recap sample location
        print(f"\n  Next step: python 3_run_app.py")  # Advice on what to do next
        print("=" * 70 + "\n")  # Footer
  # Empty line for readability
    except (FileNotFoundError, RuntimeError) as e:  # Handle expected errors
        # Known errors (wrong paths, empty dataset) -- print message and exit cleanly.  # Explanation
        print(f"\n{e}")  # Print the error message
        sys.exit(1)  # Exit script with error
    except Exception as e:  # Handle unexpected errors
        # Unexpected errors -- print the full traceback so the user knows exactly what happened.  # Debugging help
        import traceback  # Need traceback for detailed error info
        print(f"\nUNEXPECTED ERROR: {e}")  # Print general error
        traceback.print_exc()  # Print full technical details
        sys.exit(1)  # Exit script with error
    finally:  # This block runs no matter what (success or failure)
        # Always print the total time taken, even if an error occurred.  # Stats
        total_time = time.time() - start_time  # Calculate duration
        hours, rem = divmod(total_time, 3600)  # Extract hours
        minutes, seconds = divmod(rem, 60)  # Extract minutes and seconds
  # Empty line for readability
        print("\n" + "-" * 70)  # Footer separator
        print(f"Total Execution Time: {int(hours):02d}h {int(minutes):02d}m {int(seconds):02d}s")  # Print formatted time
        print("-" * 70 + "\n")  # Bottom border
  # Empty line for readability
  # Empty line for readability
if __name__ == "__main__":  # Check if this script is being run directly (not imported)
    main()  # Run the main function
  # Final empty line
