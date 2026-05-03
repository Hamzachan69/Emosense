"""
EmoSense by Hamza Khan - TRAINING SCRIPT v3 (RAF-DB Numbered Folders)

PIPELINE (Two-Stage):
  Stage 1 -> YOLOv8n-face  : detects tight face bounding boxes
  Stage 2 -> YOLOv8s-cls   : classifies emotion from each cropped face

YOUR ACTUAL FOLDER STRUCTURE (what this script reads):
  face_emotion/data/RAF DB/DATASET/
      train/
          1/   <- Surprise
          2/   <- Fear
          3/   <- Disgust
          4/   <- Happy
          5/   <- Sad
          6/   <- Angry
          7/   <- Neutral
      test/
          1/ 2/ 3/ 4/ 5/ 6/ 7/   (same mapping)

HOW TO RUN:
  python 2_train_model.py
  (no arguments needed -- paths are pre-configured for your structure)

OPTIONAL ARGUMENTS:
  --epochs 80          (default: 80)
  --batch  64          (default: 64)
  --model  yolov8s-cls.pt   (default: yolov8s-cls.pt -- better than nano)
"""

import os
import sys
import time
import shutil
import argparse
from pathlib import Path

# Try to import all required libraries. If any are missing, tell the user and stop.
try:
    import torch
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    from ultralytics import YOLO
    from PIL import Image
except ImportError as e:
    print(f"\nMissing package: {e.name}")
    print("   Run: pip install ultralytics torch torchvision opencv-python "
          "matplotlib tqdm pillow\n")
    sys.exit(1)

# --- CLASS MAPPING ---
# The RAF-DB dataset uses numbered folders (1 through 7) instead of emotion names.
# This dictionary maps each folder number to the emotion it represents.
RAFDB_FOLDER_TO_NAME = {
    "1": "surprise",
    "2": "fear",
    "3": "disgust",
    "4": "happy",
    "5": "sad",
    "6": "angry",
    "7": "neutral",
}

# YOLO automatically sorts class names alphabetically when it reads the folders.
# This dictionary shows what index (number) each emotion gets assigned.
# Example: 0=angry, 1=disgust, 2=fear, 3=happy, 4=neutral, 5=sad, 6=surprise
EMOTION_CLASSES = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "neutral",
    5: "sad",
    6: "surprise",
}

# BGR color values used to draw colored boxes on the preview images.
# Each emotion gets a distinct color so they are easy to tell apart visually.
EMOTION_COLORS = {
    "neutral":  (0, 255, 0),
    "happy":    (0, 255, 255),
    "angry":    (0, 0, 255),
    "sad":      (255, 0, 0),
    "fear":     (128, 0, 128),
    "surprise": (0, 165, 255),
    "disgust":  (0, 128, 128),
}

# Emoji characters used in the console output to make the class distribution easier to read.
EMOJIS = {
    "neutral": "😐", "happy": "😊", "angry": "😠", "sad": "😢",
    "fear": "😨",   "surprise": "😮", "disgust": "🤢",
}

# The default path where the RAF-DB dataset is expected to be located.
DEFAULT_DATA_PATH = "face_emotion/data/RAF DB/DATASET"


# --- BANNER ---

def print_banner():
    # Prints a startup message to the console when the script begins.
    print("""
+---------------------------------------------------------------+
|                                                               |
|   EmoSense -- TRAINING PIPELINE  v3  (RAF-DB Edition)        |
|                                                               |
|   Stage 1 : YOLOv8n-face   -> tight face detection           |
|   Stage 2 : YOLOv8s-cls    -> emotion classification         |
|   Dataset : RAF-DB numbered folders (1-7 -> 7 emotions)      |
|                                                               |
+---------------------------------------------------------------+
""")


# --- DATASET PREPARATION ---

def prepare_rafdb_numbered(
    data_path: str,
    output_path: str = "face_emotion/data/rafdb_yolo"
) -> str:
    """
    Reads the RAF-DB folder layout (numbered subfolders 1-7) and converts it
    to the folder structure that YOLO expects for classification training.

    Input layout:
        data_path/
            train/  1/  2/  3/  4/  5/  6/  7/
            test/   1/  2/  3/  4/  5/  6/  7/

    Output layout (YOLO classification format):
        output_path/
            train/  angry/  disgust/  fear/  happy/  neutral/  sad/  surprise/
            val/    angry/  disgust/  fear/  happy/  neutral/  sad/  surprise/
    """
    print("\nPREPARING RAF-DB DATASET (numbered folder format)...")
    print("-" * 60)

    # Build Path objects for the main data directory and its train/test subfolders.
    data_dir  = Path(data_path)
    train_dir = data_dir / "train"
    test_dir  = data_dir / "test"

    # Check that all required folders exist before doing any work.
    # If something is missing, stop immediately with a clear error message.
    if not data_dir.exists():
        raise FileNotFoundError(
            f"\nDataset root not found: {data_dir}\n"
            f"   Expected: face_emotion/data/RAF DB/DATASET\n"
            f"   Check your folder name -- spaces and capitals matter on Windows too.\n"
            f"   Tip: if path has a space, wrap it in quotes when calling from terminal."
        )
    if not train_dir.exists():
        raise FileNotFoundError(
            f"\n'train' folder not found inside: {data_dir}\n"
            f"   Expected subfolders: train/1  train/2 ... train/7"
        )
    if not test_dir.exists():
        raise FileNotFoundError(
            f"\n'test' folder not found inside: {data_dir}\n"
            f"   Expected subfolders: test/1  test/2 ... test/7"
        )

    # Create the output directory tree.
    # Each split (train, val) gets one subfolder per emotion name.
    out_dir = Path(output_path)
    emotions = list(EMOTION_CLASSES.values())  # alphabetical order

    for split in ["train", "val"]:
        for emo in emotions:
            (out_dir / split / emo).mkdir(parents=True, exist_ok=True)

    # Counters and a list to track missing folders across both splits.
    total_copied = 0
    missing_folders = []

    def _copy_numbered_split(src_split_dir: Path, dst_split_name: str):
        # Copies all images from one split (train or test) into the output folders.
        # Numbered source folders (1-7) are renamed to emotion name folders.
        nonlocal total_copied
        copied_this_split = 0

        for num_str, emo_name in RAFDB_FOLDER_TO_NAME.items():
            src_folder = src_split_dir / num_str   # e.g. train/4
            dst_folder = out_dir / dst_split_name / emo_name  # e.g. train/happy

            # If a numbered folder doesn't exist, skip it and warn the user.
            if not src_folder.exists():
                missing_folders.append(str(src_folder))
                print(f"  Warning: Missing folder: {src_folder} -- skipping")
                continue

            # Collect all image files in the numbered folder.
            imgs = (list(src_folder.glob("*.jpg")) +
                    list(src_folder.glob("*.jpeg")) +
                    list(src_folder.glob("*.png")))

            # Copy each image to the output folder, skipping any that already exist.
            for img_path in tqdm(imgs,
                                 desc=f"  {dst_split_name}/{emo_name:8} (folder {num_str})",
                                 leave=False):
                dst = dst_folder / img_path.name
                if not dst.exists():
                    shutil.copy2(img_path, dst)
                copied_this_split += 1

        total_copied += copied_this_split
        print(f"  {dst_split_name}: {copied_this_split:,} images copied")

    # Run the copy process for both the training and validation (test) splits.
    _copy_numbered_split(train_dir, "train")
    _copy_numbered_split(test_dir,  "val")

    # Print a simple bar chart showing how many images exist per emotion in train.
    print("\n  Class Distribution (train):")
    grand_total = 0
    for emo in emotions:
        n = len(list((out_dir / "train" / emo).glob("*")))
        grand_total += n
        bar = "#" * max(1, int(n / 80))  # scale bar width to image count
        print(f"    {EMOJIS.get(emo,'?')} {emo:10}  {n:5,}  {bar}")
    print(f"  Total train images : {grand_total:,}")

    val_total = sum(
        len(list((out_dir / "val" / emo).glob("*"))) for emo in emotions
    )
    print(f"  Total val   images : {val_total:,}")

    # If nothing was copied at all, something is wrong with the folder structure.
    if grand_total == 0:
        raise RuntimeError(
            "\nNo images were copied!\n"
            "   Check that your numbered folders (1-7) actually contain .jpg/.png images.\n"
            f"   Looked inside: {train_dir}"
        )

    print(f"\n  Dataset prepared at: {out_dir}")
    return str(out_dir)


# --- DEVICE RESOLVER ---

def resolve_device(requested: str) -> str:
    # Decides which hardware device to use for training (GPU or CPU).
    # If the user asked for GPU (cuda) but none is available, fall back to CPU.
    if requested in ("auto", "cuda", "0", "cuda:0"):
        if torch.cuda.is_available():
            gpu  = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
            print(f"\n  GPU  : {gpu}  ({vram:.1f} GB VRAM)")
            print("  Using CUDA")
            return "0"
        else:
            print("\n  No CUDA GPU found -- falling back to CPU (will be slow).")
            return "cpu"
    # If the user passed a specific device string (like "cpu"), use it as-is.
    return requested


# --- TRAIN CLASSIFIER ---

def train_classifier(
    data_dir:   str,
    model_name: str = "yolov8s-cls.pt",   # small model, more accurate than nano
    epochs:     int = 80,
    batch:      int = 64,
    imgsz:      int = 224,
    device:     str = "auto",
) -> YOLO:
    """
    Trains the Stage-2 emotion classifier on the prepared RAF-DB data.
    Uses YOLOv8s-cls (small) for better accuracy than the nano variant.
    An RTX 3070 handles batch=64 at imgsz=224 without running out of memory.
    """
    # Resolve whether we are training on GPU or CPU.
    device = resolve_device(device)

    print("\n" + "=" * 65)
    print("  STAGE 2 -- EMOTION CLASSIFIER TRAINING")
    print("=" * 65)
    print(f"  Model      : {model_name}")
    print(f"  Epochs     : {epochs}")
    print(f"  Batch size : {batch}")
    print(f"  Image size : {imgsz}x{imgsz}")
    print(f"  Device     : {device}")
    print(f"  Data dir   : {data_dir}")
    print("=" * 65 + "\n")

    # Load the pre-trained YOLO model. It already knows how to recognize shapes
    # and edges -- we fine-tune it on top of that to recognize emotions.
    model = YOLO(model_name)

    model.train(
        data    = data_dir,
        task    = "classify",
        epochs  = epochs,
        imgsz   = imgsz,
        batch   = batch,
        device  = device,

        # --- Augmentation settings ---
        # These randomly modify training images so the model doesn't just memorize
        # the exact photos it has seen. This is especially important for RAF-DB
        # because some emotion classes have far fewer examples than others.
        hsv_h     = 0.015,   # slightly shift the hue (color tone) of each image
        hsv_s     = 0.5,     # randomly increase or decrease color saturation
        hsv_v     = 0.4,     # randomly brighten or darken the image
        degrees   = 15.0,    # rotate the image randomly up to 15 degrees
        translate = 0.15,    # shift the image slightly left, right, up, or down
        scale     = 0.5,     # zoom in or out randomly
        fliplr    = 0.5,     # flip 50% of images horizontally (faces look the same mirrored)
        flipud    = 0.0,     # never flip upside down -- upside-down faces would confuse the model
        erasing   = 0.4,     # randomly erase small patches -- helps with minority classes like fear and disgust
        mixup     = 0.1,     # blend two training images together -- boosts minority class learning

        # --- Optimizer settings ---
        # AdamW is an optimizer that adjusts how fast the model learns over time.
        optimizer     = "AdamW",
        lr0           = 0.001,      # starting learning rate
        lrf           = 0.01,       # final learning rate (as a fraction of lr0)
        warmup_epochs = 3,          # ramp up the learning rate slowly for the first 3 epochs
        patience      = 20,         # stop training early if accuracy doesn't improve for 20 epochs

        # --- Output settings ---
        project  = "runs/detect",              # folder where training results are saved
        name     = "emotion_classifier",       # subfolder name inside runs/detect/
        exist_ok = True,                       # allow reusing the folder without crashing
        save     = True,                       # save the best and last model checkpoints
        plots    = True,                       # save training curve graphs (loss, accuracy)
        verbose  = True,                       # print progress to the console
    )

    print("\nSTAGE 2 TRAINING COMPLETE!")
    return model


# --- EXPORT TO ONNX ---

def export_classifier(model: YOLO) -> str | None:
    # Converts the trained PyTorch model to ONNX format.
    # ONNX is a universal model format that runs faster at inference time
    # and does not require the full PyTorch library to be installed.
    print("\nEXPORTING CLASSIFIER TO ONNX...")
    try:
        path = model.export(
            format   = "onnx",
            imgsz    = [224, 224],   # must match the image size used during training
            simplify = True,         # simplify the computation graph for faster inference
            opset    = 12,           # ONNX opset version -- 12 has broad hardware support
            half     = False,        # do not use 16-bit floats (keeps compatibility)
            dynamic  = False,        # fixed input shape is faster than dynamic shapes
            verbose  = False,
        )
        print(f"  ONNX saved: {path}")
        return str(path)
    except Exception as e:
        print(f"  ONNX export failed: {e}")
        print("  The PyTorch .pt model still works fine in the app.")
        return None


# --- SAMPLE PREDICTIONS ---

def generate_samples(cls_model: YOLO, data_dir: str, num: int = 9):
    """
    Runs the full two-stage pipeline on a few validation images and saves
    a grid of annotated results to outputs/sample_predictions.png.
    This lets you quickly see how well the model is performing after training.
    """
    print("\nGENERATING SAMPLE PREDICTIONS...")

    val_dir = Path(data_dir) / "val"
    if not val_dir.exists():
        print("  No val dir found, skipping")
        return

    # Collect one image per emotion class from the validation folder.
    samples = []
    for cls_folder in sorted(val_dir.iterdir()):
        if not cls_folder.is_dir():
            continue
        imgs = list(cls_folder.glob("*.jpg")) + list(cls_folder.glob("*.png"))
        if imgs:
            samples.append((imgs[0], cls_folder.name))
        if len(samples) >= num:
            break

    if not samples:
        print("  No sample images found in val/")
        return

    # Load the Stage-1 face detector so we can run the full two-stage pipeline.
    face_model = YOLO("yolov8n-face.pt")

    # Set up a matplotlib figure with one cell per sample image.
    cols = 3
    rows = (len(samples) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = np.array(axes).flatten()

    for idx, (img_path, true_label) in enumerate(samples):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        annotated = img.copy()

        # Stage 1: Run the face detector on the image.
        face_results = face_model(img, conf=0.3, verbose=False)
        faces_found  = False

        if face_results[0].boxes is not None and len(face_results[0].boxes):
            for box in face_results[0].boxes:
                # Get the bounding box coordinates as integers.
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w, h   = x2 - x1, y2 - y1

                # Crop a tight square around the center of the detected face.
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                half   = int(min(w, h) * 0.45)
                x1 = max(0, cx - half);  x2 = min(img.shape[1], cx + half)
                y1 = max(0, cy - half);  y2 = min(img.shape[0], cy + half)

                # Cut out just the face region.
                crop = img[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                # Stage 2: Run the emotion classifier on the cropped face.
                emo_result = cls_model(crop, verbose=False)
                probs      = emo_result[0].probs
                pred_cls   = int(probs.top1)         # index of the top predicted class
                pred_name  = EMOTION_CLASSES.get(pred_cls, "?")
                conf       = float(probs.top1conf)   # confidence score for that prediction
                color      = EMOTION_COLORS.get(pred_name, (255, 255, 255))

                # Draw the bounding box and the predicted label on the image.
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated, f"{pred_name} {conf:.0%}",
                            (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, color, 2, cv2.LINE_AA)
                faces_found = True

        # If no face was detected, run the emotion model on the whole image instead.
        if not faces_found:
            emo_result = cls_model(img, verbose=False)
            probs      = emo_result[0].probs
            pred_name  = EMOTION_CLASSES.get(int(probs.top1), "?")
            conf       = float(probs.top1conf)
            cv2.putText(annotated, f"{pred_name} {conf:.0%}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 2, cv2.LINE_AA)

        # Convert from OpenCV's BGR to RGB so matplotlib displays colors correctly.
        rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        axes[idx].imshow(rgb)
        axes[idx].set_title(f"True: {true_label}", fontsize=9)
        axes[idx].axis("off")

    # Turn off any unused subplot cells (if num samples < rows * cols).
    for i in range(len(samples), len(axes)):
        axes[i].axis("off")

    plt.suptitle("Sample Predictions -- Two-Stage Pipeline (face detect -> classify)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()

    # Save the grid image to the outputs folder.
    os.makedirs("outputs/snapshots", exist_ok=True)
    out = "outputs/sample_predictions.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# --- MAIN ---

def main():
    print_banner()

    # Set up command-line argument parsing so the user can override defaults.
    parser = argparse.ArgumentParser(description="EmoSense -- Train on RAF-DB (numbered folders)")
    parser.add_argument("--data",         type=str, default=DEFAULT_DATA_PATH,
                        help="Path to RAF DB/DATASET folder (contains train/ and test/)")
    parser.add_argument("--model",        type=str, default="yolov8s-cls.pt",
                        help="yolov8s-cls.pt (recommended) or yolov8n-cls.pt (faster, less accurate)")
    parser.add_argument("--epochs",       type=int, default=80,
                        help="Number of training epochs")
    parser.add_argument("--batch",        type=int, default=64,
                        help="Batch size (64 fits an RTX 3070 8 GB)")
    parser.add_argument("--img-size",     type=int, default=224)
    parser.add_argument("--device",       type=str, default="auto")
    parser.add_argument("--skip-export",  action="store_true",
                        help="Skip the ONNX export step")
    parser.add_argument("--skip-samples", action="store_true",
                        help="Skip generating sample prediction images")
    args = parser.parse_args()

    # Print environment info so the user can confirm their GPU is being used.
    print("\nSYSTEM INFO:")
    print(f"  Python  : {sys.version.split()[0]}")
    print(f"  PyTorch : {torch.__version__}")
    print(f"  CUDA    : {'Available' if torch.cuda.is_available() else 'Not available'}")
    if torch.cuda.is_available():
        print(f"  GPU     : {torch.cuda.get_device_name(0)}")

    print(f"\n  Dataset path : {args.data}")
    print(f"  Model        : {args.model}")
    print(f"  Epochs       : {args.epochs}")
    print(f"  Batch size   : {args.batch}")

    # Record the start time so we can report total duration at the end.
    start_time = time.time()

    try:
        # Step 1: Convert the RAF-DB numbered folders to YOLO classification format.
        prepared_dir = prepare_rafdb_numbered(
            data_path   = args.data,
            output_path = "face_emotion/data/rafdb_yolo",
        )

        # Step 2: Train the emotion classifier on the prepared dataset.
        model = train_classifier(
            data_dir   = prepared_dir,
            model_name = args.model,
            epochs     = args.epochs,
            batch      = args.batch,
            imgsz      = args.img_size,
            device     = args.device,
        )

        # Step 3: Copy the best model weights to the models/ folder so the app can find them.
        final_dir = Path("models/emotion_classifier")
        final_dir.mkdir(parents=True, exist_ok=True)

        best_pt = Path(model.trainer.save_dir) / "weights" / "best.pt"
        if best_pt.exists():
            shutil.copy2(best_pt, final_dir / "best.pt")
            print(f"\n  Best .pt  -> {final_dir}/best.pt")

        # Step 4: Export the trained model to ONNX format (unless skipped).
        if not args.skip_export:
            export_classifier(model)
            best_onnx = Path(model.trainer.save_dir) / "weights" / "best.onnx"
            if best_onnx.exists():
                shutil.copy2(best_onnx, final_dir / "best.onnx")
                print(f"  Best .onnx -> {final_dir}/best.onnx")

        # Step 5: Generate a sample prediction grid image (unless skipped).
        if not args.skip_samples:
            generate_samples(model, prepared_dir)

        print("\n" + "=" * 70)
        print("  ALL DONE!")
        print("=" * 70)
        print(f"  Model saved at  : {final_dir}/best.pt")
        print(f"  Training logs   : {model.trainer.save_dir}")
        print(f"  Sample grid     : outputs/sample_predictions.png")
        print(f"\n  Next step: python 3_run_app.py")
        print("=" * 70 + "\n")

    except (FileNotFoundError, RuntimeError) as e:
        # Known errors (wrong paths, empty dataset) -- print message and exit cleanly.
        print(f"\n{e}")
        sys.exit(1)
    except Exception as e:
        # Unexpected errors -- print the full traceback so the user knows exactly what happened.
        import traceback
        print(f"\nUNEXPECTED ERROR: {e}")
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Always print the total time taken, even if an error occurred.
        total_time = time.time() - start_time
        hours, rem = divmod(total_time, 3600)
        minutes, seconds = divmod(rem, 60)

        print("\n" + "-" * 70)
        print(f"Total Execution Time: {int(hours):02d}h {int(minutes):02d}m {int(seconds):02d}s")
        print("-" * 70 + "\n")


if __name__ == "__main__":
    main()
