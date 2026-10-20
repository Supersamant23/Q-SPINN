"""
device.py -- Shared GPU/CPU detection and reporting utility.

Both experiment scripts import this at startup. It selects the best
available device, prints a clear summary, and returns a torch.device
so the rest of the code never has to think about it.
"""

import torch


def get_device(verbose: bool = True) -> torch.device:
    """
    Detect the best available device and return it.

    Prints a clear banner on startup so the user always knows what
    hardware the experiment is running on — no silent CPU fallbacks.

    Returns:
        torch.device  — either cuda:0 or cpu
    """
    sep = "=" * 58

    if torch.cuda.is_available():
        dev = torch.device("cuda:0")
        props = torch.cuda.get_device_properties(dev)
        vram_gb = props.total_memory / 1e9

        if verbose:
            print(sep)
            print("  ✅  GPU DETECTED — training on CUDA")
            print(f"      Device  : {props.name}")
            print(f"      VRAM    : {vram_gb:.1f} GB")
            print(f"      CUDA    : {torch.version.cuda}")
            print(f"      PyTorch : {torch.__version__}")
            print(sep)

        return dev

    else:
        if verbose:
            print(sep)
            print("  ⚠️   NO CUDA GPU DETECTED — falling back to CPU")
            print("      Training will be significantly slower.")
            print(f"      PyTorch : {torch.__version__}")
            print()
            print("  To enable GPU acceleration:")
            print("  1. Confirm NVIDIA drivers are installed (run nvidia-smi)")
            print("  2. Reinstall PyTorch with CUDA support:")
            print("     pip install torch==2.14.0+cu126 \\")
            print("       --index-url https://download.pytorch.org/whl/cu126")
            print(sep)

        return torch.device("cpu")


def move(obj, device: torch.device):
    """Convenience: move a tensor or nn.Module to device."""
    return obj.to(device)
