"""
GPU FIX & DIAGNOSTIC SCRIPT
Run this BEFORE training to verify your RTX GPU is working.

What this does:
  1. Uninstalls CPU-only PyTorch completely
  2. Installs the correct CUDA version
  3. Runs a full GPU test
  4. Tells you exactly what to do next

Usage:
    python 0_fix_gpu.py
"""

import subprocess
import sys
import os


def run(cmd, label="", capture=True):
    print(f"  ⏳ {label}..." if label else "", flush=True)
    result = subprocess.run(cmd, capture_output=capture, text=True)
    return result


def pip(args, label=""):
    return run([sys.executable, "-m", "pip"] + args, label)


def banner(text):
    print("\n" + "="*65)
    print(f"  {text}")
    print("="*65)


# ─────────────────────────────────────────────────────────────────────────────
def step1_check_nvidia():
    """Check if Windows can see your NVIDIA GPU at all."""
    banner("STEP 1: Checking NVIDIA GPU visibility")

    result = run(["nvidia-smi"], "Running nvidia-smi")

    if result.returncode != 0:
        print("""
  ❌ nvidia-smi failed. This means one of:
     a) NVIDIA driver is NOT installed
     b) Driver is too old

  👉 Fix: Download & install the latest driver:
     https://www.nvidia.com/Download/index.aspx
     → GeForce → RTX 30 Series → RTX 3050 or 3070 → Windows 11
     → Install → RESTART your PC → re-run this script
""")
        return False

    # Print the important line from nvidia-smi
    for line in result.stdout.split('\n'):
        if 'RTX' in line or 'Driver' in line or 'CUDA' in line or 'MiB' in line:
            print(f"  {line.strip()}")

    print("\n  ✅ NVIDIA GPU found and driver is working!")
    return True


# ─────────────────────────────────────────────────────────────────────────────
def step2_check_current_torch():
    """Check what PyTorch is currently installed."""
    banner("STEP 2: Checking current PyTorch installation")

    result = run([sys.executable, "-c",
                  "import torch; print(torch.__version__); print(torch.cuda.is_available())"])

    if result.returncode != 0:
        print("  ⚠️  PyTorch not installed at all.")
        return "none", False

    lines = result.stdout.strip().split('\n')
    version = lines[0] if lines else "unknown"
    cuda_ok = lines[1].strip().lower() == 'true' if len(lines) > 1 else False

    print(f"  PyTorch version: {version}")
    print(f"  CUDA available:  {'✅ YES' if cuda_ok else '❌ NO'}")

    if '+cpu' in version:
        print("\n  ⚠️  You have the CPU-ONLY version installed.")
        print("  This is the problem — we need to replace it with the CUDA version.")
        return version, False

    if cuda_ok:
        print("\n  ✅ PyTorch already has CUDA support!")
        return version, True

    print("\n  ⚠️  PyTorch has CUDA build but CUDA is still not available.")
    print("  This usually means CUDA Toolkit is not installed.")
    return version, False


# ─────────────────────────────────────────────────────────────────────────────
def step3_reinstall_torch():
    """Force-reinstall the CUDA version of PyTorch."""
    banner("STEP 3: Force-installing PyTorch with CUDA 12.1")
    print("  (Uninstalling CPU version first, then installing CUDA version)")
    print("  This download is ~2.5 GB — please wait...\n")

    # Uninstall existing torch completely
    pip(["uninstall", "torch", "torchvision", "torchaudio", "-y"], "Uninstalling old PyTorch")

    # Install CUDA 12.1 build (compatible with RTX 30xx)
    result = pip([
        "install",
        "torch", "torchvision",
        "--index-url", "https://download.pytorch.org/whl/cu121",
        "--force-reinstall",
        "--no-cache-dir"
    ], "Installing PyTorch CUDA 12.1 (big download ~2.5 GB)")

    if result.returncode != 0:
        print(f"\n  ❌ Install failed!")
        print(f"  Error: {result.stderr[-500:]}")
        return False

    print("\n  ✅ PyTorch CUDA installed successfully!")
    return True


# ─────────────────────────────────────────────────────────────────────────────
def step4_full_gpu_test():
    """Run a comprehensive GPU test."""
    banner("STEP 4: Full GPU Diagnostic Test")

    test_script = """
import torch
import sys

print("=" * 55)
print("  PYTORCH GPU DIAGNOSTIC")
print("=" * 55)
print(f"  PyTorch version : {torch.__version__}")
print(f"  Python version  : {sys.version.split()[0]}")
print(f"  CUDA available  : {torch.cuda.is_available()}")
print(f"  CUDA version    : {torch.version.cuda}")

if not torch.cuda.is_available():
    print()
    print("  ❌ CUDA still not available after reinstall.")
    print("  👉 You need to install CUDA Toolkit 12.1:")
    print("     https://developer.nvidia.com/cuda-12-1-0-download-archive")
    print("     Windows → x86_64 → 11 → exe (local)")
    print("     Install → RESTART PC → re-run this script")
    sys.exit(1)

# GPU details
n = torch.cuda.device_count()
print(f"  GPU count       : {n}")
for i in range(n):
    props = torch.cuda.get_device_properties(i)
    vram = props.total_memory / 1024**3
    print(f"  GPU {i}           : {props.name}")
    print(f"  VRAM            : {vram:.1f} GB")
    print(f"  Compute cap.    : {props.major}.{props.minor}")

print()
print("  Running speed test (matrix multiply on GPU)...")
import time
device = torch.device('cuda:0')

# Warm up
a = torch.randn(1000, 1000, device=device)
b = torch.randn(1000, 1000, device=device)
_ = torch.mm(a, b)
torch.cuda.synchronize()

# Benchmark
start = time.time()
for _ in range(100):
    c = torch.mm(a, b)
torch.cuda.synchronize()
elapsed = time.time() - start

print(f"  Speed test      : 100x (1000x1000) matmul in {elapsed:.2f}s")
print(f"  GPU is          : {'🔥 Fast!' if elapsed < 0.5 else '✅ Working'}")
print()
print("=" * 55)
print("  ✅ ALL TESTS PASSED — GPU IS READY FOR TRAINING!")
print("=" * 55)
print()
print("  👉 Now run:")
print("     python 2_train_model.py --data .\\\\face_emotion\\\\data\\\\affectnet-yolo-format")
"""

    result = run([sys.executable, "-c", test_script], capture=False)
    return result.returncode == 0


# ─────────────────────────────────────────────────────────────────────────────
def step5_check_cuda_toolkit():
    """Check if CUDA Toolkit is installed."""
    banner("STEP 5: Checking CUDA Toolkit installation")

    # Check nvcc (CUDA compiler)
    result = run(["nvcc", "--version"], "Checking nvcc")

    if result.returncode == 0:
        for line in result.stdout.split('\n'):
            if 'release' in line.lower():
                print(f"  ✅ CUDA Toolkit: {line.strip()}")
        return True
    else:
        print("  ⚠️  CUDA Toolkit (nvcc) not found in PATH.")
        print("  This may or may not be a problem — PyTorch bundles its own CUDA libs.")
        print("  If Step 4 passed, you're fine. If not, install CUDA Toolkit 12.1:")
        print("  → https://developer.nvidia.com/cuda-12-1-0-download-archive")
        return False


# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("""
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║   🔧 GPU FIX & DIAGNOSTIC — RTX 3050 / 3070 Edition          ║
║                                                               ║
║   This script will:                                           ║
║   1. Check your NVIDIA driver                                 ║
║   2. Check current PyTorch (detect CPU-only version)          ║
║   3. Force-reinstall PyTorch with CUDA 12.1                   ║
║   4. Run a full GPU test with speed benchmark                 ║
║   5. Tell you exactly what to do next                         ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
""")

    # Step 1: Check nvidia-smi
    driver_ok = step1_check_nvidia()
    if not driver_ok:
        print("\n❌ Cannot proceed without NVIDIA driver. Install driver first, then re-run.")
        sys.exit(1)

    # Step 2: Check current PyTorch
    torch_version, cuda_already_ok = step2_check_current_torch()

    # Step 3: Reinstall if needed
    if not cuda_already_ok:
        reinstalled = step3_reinstall_torch()
        if not reinstalled:
            print("\n❌ Reinstall failed. Check your internet connection and try again.")
            sys.exit(1)
    else:
        print("\n  ⏭️  Skipping reinstall — CUDA already working.")

    # Step 4: Full GPU test
    step5_check_cuda_toolkit()
    gpu_ok = step4_full_gpu_test()

    # Final verdict
    banner("FINAL RESULT")
    if gpu_ok:
        print("""
  🎉 YOUR GPU IS READY!

  Run training now:
     python 2_train_model.py --data .\\face_emotion\\data\\affectnet-yolo-format

  Expected training time with RTX GPU: ~30-45 minutes for 50 epochs
  (vs ~3 hours on CPU)
""")
    else:
        print("""
  ❌ GPU NOT READY YET

  Most likely cause: CUDA Toolkit not installed.

  Fix steps:
  1. Go to: https://developer.nvidia.com/cuda-12-1-0-download-archive
  2. Select: Windows → x86_64 → 11 → exe (local)
  3. Download and install (~3 GB)
  4. RESTART your PC
  5. Re-run: python 0_fix_gpu.py
""")


if __name__ == "__main__":
    main()
