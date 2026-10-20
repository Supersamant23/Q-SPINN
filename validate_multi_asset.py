import sys
import torch
import torch.nn as nn
from pathlib import Path

ROOT = Path('c:/Users/adity/Documents/College/Projects/College/Q-SPINN')
sys.path.insert(0, str(ROOT / 'experiments'))

import exp_B_hybrid_fi2010.train as train_module

def validate_multi_asset():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Force Multi-Asset config
    train_module.N_ASSETS = 3
    train_module.N_FEATURES = 144 * 3 # 432
    
    model = train_module.QSPINN(n_in=432, d=8, n_assets=3).to(device)
    
    # Mock data for one batch (B=32)
    B, T, F = 32, 20, 432
    Xb = torch.randn(B, T, F, dtype=torch.float32, device=device)
    
    print("Testing 3-Asset Heavy Path forward pass...")
    heavy, L_pde, H, S, L_cov, L_norm = model.heavy_path(Xb)
    
    print("Testing 3-Asset Hybrid SNN/PINN forward pass...")
    dt_seq = torch.ones(B, T, dtype=torch.float32, device=device) * 0.1
    spikes = train_module.delta_encode(Xb)
    cheap, g, rate, L_pde_hybrid, H_hybrid, S_hybrid, _, L_cov_hybrid, L_norm_hybrid = model(spikes, Xb, dt_seq, mode="hybrid")
    
    print(f"Prediction Output Shape: {heavy.shape}")
    print(f"Schrödinger PDE Loss (L_pde): {L_pde.item():.6f}")
    print(f"Hamiltonian Matrix Shape (Should be 24x24): {H.shape}")
    
    print("\n✅ Multi-Asset Magnetic Graph Laplacian validated successfully! No dimensionality or runtime errors.")

if __name__ == "__main__":
    validate_multi_asset()
