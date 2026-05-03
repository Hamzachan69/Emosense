import os
import sys
import subprocess
import platform

def print_banner():
    print("""
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║   📦  EmoSense — UNIVERSAL REQUIREMENTS INSTALLER             ║
║       (Windows / Linux / MacOS / Kali Linux)                  ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
""")

def run_command(command):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + command)
        return True
    except subprocess.CalledProcessError:
        return False

def main():
    print_banner()
    
    os_name = platform.system()
    print(f"🖥️  Detected OS: {os_name}")
    
    # ── 1. CHECK FOR NVIDIA GPU (CUDA) ──────────────────────────────────────
    has_gpu = False
    try:
        # Check if nvidia-smi exists (standard way to check for NVIDIA drivers)
        subprocess.check_output('nvidia-smi', shell=True)
        has_gpu = True
        print("🎮 NVIDIA GPU Detected! Preparing CUDA installation...")
    except:
        print("🐌 No NVIDIA GPU found or drivers missing. Preparing CPU installation...")

    # ── 2. INSTALL PYTORCH (The most important part) ────────────────────────
    print("\n🚀 Installing PyTorch (Core Brain)...")
    
    if os_name == "Windows":
        if has_gpu:
            # High-performance CUDA 12.1 for Windows
            run_command(["torch", "torchvision", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cu121"])
        else:
            # Standard CPU version for Windows
            run_command(["torch", "torchvision", "torchaudio"])
            
    elif os_name == "Linux":
        # Linux (including Kali) usually handles CUDA better via default pip or specific wheels
        if has_gpu:
            print("💡 On Linux, ensuring CUDA compatibility...")
            run_command(["torch", "torchvision", "torchaudio"]) # Default Linux torch usually includes CUDA
        else:
            # Force CPU version for Linux if no GPU
            run_command(["torch", "torchvision", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cpu"])
    else:
        # MacOS or others
        run_command(["torch", "torchvision", "torchaudio"])

    # ── 3. INSTALL OTHER DEPENDENCIES ───────────────────────────────────────
    print("\n🎨 Installing UI and AI Libraries...")
    
    dependencies = [
        "ultralytics",     # YOLOv8
        "customtkinter",   # Professional GUI
        "opencv-python",   # Camera & Image processing
        "pillow",          # Emoji rendering
        "matplotlib",      # Training charts
        "tqdm",            # Progress bars
        "pandas",          # Data handling
        "requests",        # Network (if needed)
        "psutil"           # System monitoring
    ]
    
    for dep in dependencies:
        print(f"  → Installing {dep}...")
        if not run_command([dep]):
            print(f"  ❌ Failed to install {dep}")

    # ── 4. FINAL VERIFICATION ───────────────────────────────────────────────
    print("\n" + "═"*60)
    print("✅ INSTALLATION COMPLETE!")
    print("═"*60)
    
    try:
        import torch
        print(f"🔹 PyTorch Version: {torch.__version__}")
        print(f"🔹 CUDA Available:  {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"🔹 GPU Device:      {torch.cuda.get_device_name(0)}")
    except:
        print("❌ Verification failed. Please restart your terminal and try again.")

    print("\n🚀 NEXT STEPS:")
    print("1. Run 'python 0_fix_gpu.py' to verify your drivers.")
    print("2. Run 'python 3_run_app.py' to launch EmoSense!")
    print("═"*60 + "\n")

if __name__ == "__main__":
    main()