"""  # Start of a multi-line docstring to describe the script
GPU FIX & DIAGNOSTIC SCRIPT  # Title of the script
Run this BEFORE training to verify your NVIDIA GPU is working.  # Purpose of the script
  # Empty line for readability
What this does:  # Section listing the script's functions
  1. Uninstalls CPU-only PyTorch completely  # Task 1: Clean up wrong versions
  2. Installs the correct CUDA version  # Task 2: Install the GPU-ready version
  3. Runs a full GPU test  # Task 3: Verify it works with a speed test
  4. Tells you exactly what to do next  # Task 4: Provide instructions
  # Empty line for readability
Usage:  # How to use the script
    python 0_fix_gpu.py  # The command to type in the terminal
"""  # End of the docstring
  # Empty line for readability
import subprocess  # Import 'subprocess' to run terminal commands from within Python
import sys  # Import 'sys' to access system-specific parameters and functions
import os  # Import 'os' to interact with the operating system
import platform  # Import 'platform' to detect which OS is being used (Windows/Linux)
  # Empty line for readability
def run(cmd, label="", capture=True):  # Define a function to run terminal commands
    print(f"  ⏳ {label}..." if label else "", flush=True)  # Print a status message with a timer icon
    try:  # Start a 'try' block to catch execution errors
        # Run the command and capture the output if 'capture' is True
        result = subprocess.run(cmd, capture_output=capture, text=True, shell=(platform.system() == "Windows"))
        return result  # Return the result of the command (output, exit code, etc.)
    except FileNotFoundError:  # If the command itself wasn't found (e.g., 'nvidia-smi' missing)
        class DummyResult:  # Create a simple object to mimic a real result
            def __init__(self):  # Initialize the dummy object
                self.returncode = 127  # Set error code to 127 (command not found)
                self.stdout = ""  # No output
                self.stderr = "Command not found"  # Set the error message
        return DummyResult()  # Return the dummy object
  # Empty line for readability
def pip(args, label=""):  # Define a function specifically for 'pip' commands
    # Calls the 'run' function to execute a pip command using the current Python interpreter
    return run([sys.executable, "-m", "pip"] + args, label)
  # Empty line for readability
def banner(text):  # Define a function to print a decorative header
    print("\n" + "="*65)  # Print a top border
    print(f"  {text}")  # Print the header text
    print("="*65)  # Print a bottom border
  # Empty line for readability
# ─────────────────────────────────────────────────────────────────────────────
def step1_check_nvidia():  # Define the first diagnostic step
    """Check if the OS can see your NVIDIA GPU at all."""  # Docstring describing the step
    banner("STEP 1: Checking NVIDIA GPU visibility")  # Print the step header
    os_name = platform.system()  # Get the current OS name
  # Empty line for readability
    result = run(["nvidia-smi"], "Running nvidia-smi")  # Try to run the NVIDIA diagnostic tool
  # Empty line for readability
    if result.returncode != 0:  # If the command failed (GPU or driver not found)
        if os_name == "Windows":  # If we are on Windows
            print("""  # Start of instructions for Windows users
  ❌ nvidia-smi failed. This means one of:
     a) NVIDIA driver is NOT installed
     b) Driver is too old

  👉 Fix: Download & install the latest driver:
     https://www.nvidia.com/Download/index.aspx
     → GeForce → RTX 30 Series → RTX 3050 or 3070 → Windows 11
     → Install → RESTART your PC → re-run this script
""")  # End of instructions
        else:  # If we are on Linux
            print("""  # Start of instructions for Linux users
  ❌ nvidia-smi failed. This means one of:
     a) NVIDIA driver is NOT installed
     b) Driver is not properly configured

  👉 Fix (Ubuntu/Debian):
     sudo apt update
     sudo ubuntu-drivers autoinstall
     → RESTART your system → re-run this script
""")  # End of instructions
        return False  # Return False because the check failed
  # Empty line for readability
    # Print the important line from nvidia-smi  # Summary of the tool's output
    for line in result.stdout.split('\n'):  # Loop through each line of the tool's output
        # Only print lines containing relevant information
        if 'RTX' in line or 'Driver' in line or 'CUDA' in line or 'MiB' in line:
            print(f"  {line.strip()}")  # Print the filtered line
  # Empty line for readability
    print(f"\n  ✅ NVIDIA GPU found and driver is working on {os_name}!")  # Success message
    return True  # Return True because the check passed
  # Empty line for readability
# ─────────────────────────────────────────────────────────────────────────────
def step2_check_current_torch():  # Define the second diagnostic step
    """Check what PyTorch is currently installed."""  # Docstring describing the step
    banner("STEP 2: Checking current PyTorch installation")  # Print the step header
  # Empty line for readability
    # Run a small Python script inside terminal to check torch version and CUDA status
    result = run([sys.executable, "-c",
                  "import torch; print(torch.__version__); print(torch.cuda.is_available())"])
  # Empty line for readability
    if result.returncode != 0:  # If PyTorch isn't installed at all
        print("  ⚠️  PyTorch not installed at all.")  # Notify the user
        return "none", False  # Return 'none' as version and False for CUDA status
  # Empty line for readability
    lines = result.stdout.strip().split('\n')  # Split the output into lines
    version = lines[0] if lines else "unknown"  # Get the version from the first line
    # Get the True/False status from the second line
    cuda_ok = lines[1].strip().lower() == 'true' if len(lines) > 1 else False
  # Empty line for readability
    print(f"  PyTorch version: {version}")  # Show the current version
    print(f"  CUDA available:  {'✅ YES' if cuda_ok else '❌ NO'}")  # Show if it can use the GPU
  # Empty line for readability
    if '+cpu' in version:  # If the version name includes '+cpu', it's the wrong version
        print("\n  ⚠️  You have the CPU-ONLY version installed.")  # Warn the user
        print("  This is the problem — we need to replace it with the CUDA version.")  # Explain the issue
        return version, False  # Return current version and False (not ready)
  # Empty line for readability
    if cuda_ok:  # If it's already working
        print("\n  ✅ PyTorch already has CUDA support!")  # Notify the user
        return version, True  # Return current version and True (already ready)
  # Empty line for readability
    print("\n  ⚠️  PyTorch has CUDA build but CUDA is still not available.")  # If version is right but it still fails
    print("  This usually means CUDA Toolkit or drivers are not properly set up.")  # Explain potential cause
    return version, False  # Return current version and False
  # Empty line for readability

# ─────────────────────────────────────────────────────────────────────────────
def step3_reinstall_torch():  # Define the third step: fixing the installation
    """Force-reinstall the CUDA version of PyTorch."""  # Docstring
    banner("STEP 3: Force-installing PyTorch with CUDA 12.1")  # Print the step header
    print("  (Uninstalling CPU version first, then installing CUDA version)")  # Explain the plan
    print("  This download is ~2.5 GB — please wait...\n")  # Warn about the large download size
  # Empty line for readability
    # Uninstall existing torch completely to avoid version conflicts
    pip(["uninstall", "torch", "torchvision", "torchaudio", "-y"], "Uninstalling old PyTorch")
  # Empty line for readability
    # Install the specific CUDA 12.1 build from the official PyTorch repository
    result = pip([
        "install",
        "torch", "torchvision", "torchaudio",
        "--index-url", "https://download.pytorch.org/whl/cu121",  # Specify the GPU-enabled server
        "--force-reinstall",  # Overwrite any existing files
        "--no-cache-dir"  # Don't use saved files; download fresh ones
    ], "Installing PyTorch CUDA 12.1 (big download ~2.5 GB)")
  # Empty line for readability
    if result.returncode != 0:  # If the installation failed
        print(f"\n  ❌ Install failed!")  # Notify user
        # Print the last part of the error message so the user can see what happened
        print(f"  Error: {result.stderr[-500:] if result.stderr else 'Unknown error'}")
        return False  # Return False
  # Empty line for readability
    print("\n  ✅ PyTorch CUDA installed successfully!")  # Success message
    return True  # Return True
  # Empty line for readability
# ─────────────────────────────────────────────────────────────────────────────
def step4_full_gpu_test():  # Define the final test step
    """Run a comprehensive GPU test."""  # Docstring
    banner("STEP 4: Full GPU Diagnostic Test")  # Print the step header
  # Empty line for readability
    # Define a small Python script that will actually run ON THE GPU to test it
    test_script = """
import torch  # Import torch inside the test script
import sys  # Import sys inside the test script

print("=" * 55)  # Header border
print("  PYTORCH GPU DIAGNOSTIC")  # Title
print("=" * 55)  # Header border
print(f"  PyTorch version : {torch.__version__}")  # Show version
print(f"  Python version  : {sys.version.split()[0]}")  # Show python version
print(f"  CUDA available  : {torch.cuda.is_available()}")  # Show if CUDA is found
print(f"  CUDA version    : {torch.version.cuda}")  # Show CUDA library version

if not torch.cuda.is_available():  # If it still isn't working after reinstall
    print()
    print("  ❌ CUDA still not available after reinstall.")  # Print error
    print("  👉 Manual check required:")  # Offer help
    if sys.platform == 'win32':  # Instructions for Windows
        print("     1. Install CUDA Toolkit 12.1: https://developer.nvidia.com/cuda-12-1-0-download-archive")
        print("     2. Restart PC and re-run this script.")
    else:  # Instructions for Linux
        print("     1. Install CUDA Toolkit: sudo apt install nvidia-cuda-toolkit")
        print("     2. Ensure drivers match toolkit version.")
    sys.exit(1)  # Stop with error

# GPU details
n = torch.cuda.device_count()  # Count how many GPUs are in the system
print(f"  GPU count       : {n}")  # Print the count
for i in range(n):  # Loop through each GPU
    props = torch.cuda.get_device_properties(i)  # Get physical properties (name, memory)
    vram = props.total_memory / 1024**3  # Convert bytes to Gigabytes
    print(f"  GPU {i}           : {props.name}")  # Print name
    print(f"  VRAM            : {vram:.1f} GB")  # Print memory
    print(f"  Compute cap.    : {props.major}.{props.minor}")  # Print compute capability

print()
print("  Running speed test (matrix multiply on GPU)...")  # Start the stress test
import time  # Import time for benchmarking
device = torch.device('cuda:0')  # Select the first GPU

# Warm up  # This is the "worm" / "warm" part the user might be referring to
# It runs the math once to wake up the GPU and initialize the libraries
a = torch.randn(1000, 1000, device=device)  # Create a large random grid on GPU
b = torch.randn(1000, 1000, device=device)  # Create another random grid on GPU
_ = torch.mm(a, b)  # Multiply them together
torch.cuda.synchronize()  # Wait for the GPU to finish before starting the timer

# Benchmark  # The actual timed test
start = time.time()  # Start the clock
for _ in range(100):  # Repeat 100 times to get a stable average
    c = torch.mm(a, b)  # Matrix multiplication
torch.cuda.synchronize()  # Wait for all operations to finish
elapsed = time.time() - start  # Calculate time taken

print(f"  Speed test      : 100x (1000x1000) matmul in {elapsed:.2f}s")  # Print result
print(f"  GPU is          : {'🔥 Fast!' if elapsed < 0.5 else '✅ Working'}")  # Judge performance
print()
print("=" * 55)
print("  ✅ ALL TESTS PASSED — GPU IS READY FOR TRAINING!")  # Final success message
print("=" * 55)
print()
print("  👉 Now run:")  # Instructions for next step
print("     python 2_train_model.py --data ./" + ("face_emotion/data/affectnet-yolo-format" if sys.platform != "win32" else "face_emotion\\\\data\\\\affectnet-yolo-format"))
"""  # End of the test script string
  # Empty line for readability
    # Execute the test script string as a separate Python process
    result = run([sys.executable, "-c", test_script], capture=False)
    return result.returncode == 0  # Return True if the test script finished successfully
  # Empty line for readability
# ─────────────────────────────────────────────────────────────────────────────
def step5_check_cuda_toolkit():  # Define step 5: checking for the compiler
    """Check if CUDA Toolkit is installed."""  # Docstring
    banner("STEP 5: Checking CUDA Toolkit installation")  # Header
    os_name = platform.system()  # Get OS
  # Empty line for readability
    # Check if 'nvcc' (the NVIDIA compiler) is available in the terminal
    result = run(["nvcc", "--version"], "Checking nvcc")
  # Empty line for readability
    if result.returncode == 0:  # If nvcc was found
        for line in result.stdout.split('\n'):  # Loop through output
            if 'release' in line.lower():  # Find the release version line
                print(f"  ✅ CUDA Toolkit: {line.strip()}")  # Print the version
        return True  # Return True
    else:  # If nvcc was NOT found
        print("  ⚠️  CUDA Toolkit (nvcc) not found in PATH.")  # Warn the user
        print("  This may or may not be a problem — PyTorch bundles its own CUDA libs.")  # Reassurance
        print("  If Step 4 passed, you're fine. If not, follow these steps:")  # Help
        if os_name == "Windows":  # Link for Windows
            print("  → https://developer.nvidia.com/cuda-12-1-0-download-archive")
        else:  # Command for Linux
            print("  → sudo apt install nvidia-cuda-toolkit")
        return False  # Return False
  # Empty line for readability
# ─────────────────────────────────────────────────────────────────────────────
def main():  # Main execution flow
    os_name = platform.system()  # Get OS name
    print(f"""  # Start big decorative block
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║   🔧 GPU FIX & DIAGNOSTIC — {os_name.upper()} EDITION               ║
║                                                               ║
║   This script will:                                           ║
║   1. Check your NVIDIA driver                                 ║
║   2. Check current PyTorch (detect CPU-only version)          ║
║   3. Force-reinstall PyTorch with CUDA 12.1                   ║
║   4. Run a full GPU test with speed benchmark                 ║
║   5. Tell you exactly what to do next                         ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
""")  # End decorative block
  # Empty line for readability
    # Step 1: Check nvidia-smi visibility
    driver_ok = step1_check_nvidia()
    if not driver_ok:  # If no driver
        print(f"\n❌ Cannot proceed without NVIDIA driver. Install driver first, then re-run.")
        sys.exit(1)  # Stop script
  # Empty line for readability
    # Step 2: Check the currently installed PyTorch
    torch_version, cuda_already_ok = step2_check_current_torch()
  # Empty line for readability
    # Step 3: Reinstall ONLY IF NEEDED (saves 2.5GB download)
    if not cuda_already_ok:
        reinstalled = step3_reinstall_torch()  # Try to fix it
        if not reinstalled:  # If fix failed
            print("\n❌ Reinstall failed. Check your internet connection and try again.")
            sys.exit(1)  # Stop script
    else:  # If it was already working
        print("\n  ⏭️  Skipping reinstall — CUDA already working.")
  # Empty line for readability
    # Step 4: Full GPU test and check toolkit
    step5_check_cuda_toolkit()
    gpu_ok = step4_full_gpu_test()
  # Empty line for readability
    # Final verdict printed to terminal
    banner("FINAL RESULT")
    if gpu_ok:  # If everything works!
        # Set the example command path based on OS
        data_path = ".\\face_emotion\\data\\affectnet-yolo-format" if os_name == "Windows" else "./face_emotion/data/affectnet-yolo-format"
        print(f"""  # Print success instructions
  🎉 YOUR GPU IS READY!

  Run training now:
     python 2_train_model.py --data {data_path}

  Expected training time with NVIDIA GPU: ~30-45 minutes for 50 epochs
  (vs ~3 hours on CPU)
""")
    else:  # If it still isn't ready
        print("\n  ❌ GPU NOT READY YET")
        if os_name == "Windows":  # Windows fix steps
            print("""
  Fix steps:
  1. Go to: https://developer.nvidia.com/cuda-12-1-0-download-archive
  2. Select: Windows → x86_64 → 11 → exe (local)
  3. Download and install (~3 GB)
  4. RESTART your PC
  5. Re-run: python 0_fix_gpu.py
""")
        else:  # Linux fix steps
            print("""
  Fix steps:
  1. Run: sudo apt install nvidia-cuda-toolkit
  2. RESTART your system
  3. Re-run: python 0_fix_gpu.py
""")
  # Empty line for readability
if __name__ == "__main__":  # Standard Python entry point
    main()  # Run the main function
  # Final empty line
