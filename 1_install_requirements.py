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

def run_command(command, use_shell=False):
    try:
        subprocess.check_call(command, shell=use_shell)
        return True
    except subprocess.CalledProcessError:
        return False

def pip_install(package_list):
    return run_command([sys.executable, "-m", "pip", "install"] + package_list)

def check_linux_system_deps():
    """Check and install system dependencies on Debian/Ubuntu-based systems."""
    print("\n🐧 Checking Linux System Dependencies...")
    
    # Check if we are on a Debian-based system
    is_debian = os.path.exists("/etc/debian_version")
    
    if is_debian:
        print("  Detected Debian/Ubuntu-based system. Checking for required libraries...")
        # Common libraries needed for OpenCV and other ML tools
        deps = ["libgl1-mesa-glx", "libglib2.0-0", "libsm6", "libxext6", "libxrender1"]
        
        print("  Installing/Updating: " + ", ".join(deps))
        # Note: This requires sudo, so we warn the user
        print("  ⚠️  This step requires sudo privileges to install system libraries.")
        cmd = "sudo apt-get update && sudo apt-get install -y " + " ".join(deps)
        if not run_command(cmd, use_shell=True):
            print("  ❌ Failed to install system dependencies. You might need to install them manually.")
    else:
        print("\n  ⚠️  WARNING: You are using a non-Debian/Ubuntu Linux distribution.")
        print("  Please ensure you have the following system libraries installed manually:")
        print("  - libGL (mesa)")
        print("  - libglib-2.0")
        print("  - libSM, libXext, libXrender")
        print("  Failure to install these may cause 'ImportError' when running OpenCV.")

def main():
    print_banner()
    
    os_name = platform.system()
    print(f"🖥️  Detected OS: {os_name}")
    
    if os_name == "Linux":
        check_linux_system_deps()

    # ── 1. CHECK FOR NVIDIA GPU (CUDA) ──────────────────────────────────────
    has_gpu = False
    try:
        # Check if nvidia-smi exists
        subprocess.check_output('nvidia-smi', shell=(os_name == "Windows"))
        has_gpu = True
        print("\n🎮 NVIDIA GPU Detected! Preparing CUDA installation...")
    except:
        print("\n🐌 No NVIDIA GPU found or drivers missing. Preparing CPU installation...")

    # ── 2. INSTALL PYTORCH ──────────────────────────────────────────────────
    print("🚀 Installing PyTorch (Core Brain)...")
    
    if os_name == "Windows":
        if has_gpu:
            # High-performance CUDA 12.1 for Windows
            pip_install(["torch", "torchvision", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cu121"])
        else:
            # Standard CPU version for Windows
            pip_install(["torch", "torchvision", "torchaudio"])
            
    elif os_name == "Linux":
        if has_gpu:
            print("💡 On Linux, ensuring CUDA compatibility...")
            # Default Linux torch usually includes CUDA, but we can specify the index to be sure
            pip_install(["torch", "torchvision", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cu121"])
        else:
            # Force CPU version for Linux if no GPU
            pip_install(["torch", "torchvision", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cpu"])
    else:
        # MacOS or others
        pip_install(["torch", "torchvision", "torchaudio"])

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
        "psutil",          # System monitoring
        "onnxruntime-gpu" if has_gpu else "onnxruntime" # Fast inference
    ]
    
    for dep in dependencies:
        print(f"  → Installing {dep}...")
        if not pip_install([dep]):
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