import sys
import torch
from pathlib import Path

ROOT = Path('c:/Users/adity/Documents/College/Projects/College/Q-SPINN')
sys.path.insert(0, str(ROOT / 'experiments'))

from exp_B_hybrid_fi2010.train import QSPINN, SNNGate, delta_encode

def validate():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Initialize the upgraded model
    model = QSPINN().to(device)
    
    # Create dummy batch of FI-2010 data: (Batch=32, TimeSteps=20, Features=144)
    B, T, F = 32, 20, 144
    x = torch.randn(B, T, F, dtype=torch.float32, device=device)
    
    # Generate mock spikes (delta encoded)
    spikes = delta_encode(x)
    
    print("Testing forward pass (Heavy Mode)...")
    # Mode="dense" forces the heavy PINN path to activate fully
    heavy, g, rate, L_pde, H, gl = model(spikes, x, mode="dense")
    
    print(f"Prediction Output Shape: {heavy.shape}")
    print(f"Schrödinger PDE Loss (L_pde): {L_pde.item():.6f}")
    print(f"Hamiltonian Matrix Shape: {H.shape}")
    print("\n✅ Mathematical Upgrade (torch.matrix_exp) validated successfully! No dimensionality or runtime errors.")

if __name__ == "__main__":
    validate()
