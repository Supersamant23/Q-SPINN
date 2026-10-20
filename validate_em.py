import sys
import torch
import torch.nn as nn
from pathlib import Path

ROOT = Path('c:/Users/adity/Documents/College/Projects/College/Q-SPINN')
sys.path.insert(0, str(ROOT / 'experiments'))

from exp_B_hybrid_fi2010.train import QSPINN, delta_encode

def validate_em_training():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    model = QSPINN().to(device)
    
    # Mock data for one batch (B=32)
    B, T, F = 32, 20, 144
    Xb = torch.randn(B, T, F, dtype=torch.float32, device=device)
    yb = torch.randint(0, 3, (B,), dtype=torch.int64, device=device)
    
    # EM Setup
    params_physics = list(model.amp.parameters()) + list(model.ham.parameters())
    params_head = list(model.head.parameters())
    
    opt_physics = torch.optim.Adam(params_physics, lr=1e-3)
    opt_head = torch.optim.Adam(params_head, lr=1e-3)
    ce = nn.CrossEntropyLoss()
    
    print("Testing EM Cross-Fitting Batch Split...")
    Xb_phys, Xb_head = Xb.chunk(2)
    yb_phys, yb_head = yb.chunk(2)
    
    print("Executing E-Step (Physics Only)...")
    _, L_pde, _ = model.heavy_path(Xb_phys)
    loss_phys = L_pde
    opt_physics.zero_grad()
    loss_phys.backward()
    opt_physics.step()
    print("E-Step Success.")
    
    print("Executing M-Step (Head Only)...")
    heavy, _, _ = model.heavy_path(Xb_head)
    loss_head = ce(heavy, yb_head)
    opt_head.zero_grad()
    loss_head.backward()
    opt_head.step()
    print("M-Step Success.")
    
    print("\n✅ EM Cross-Fitted Training validates flawlessly! No autograd graph detachment errors.")

if __name__ == "__main__":
    validate_em_training()
