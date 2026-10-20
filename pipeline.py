import sys
import time
import json
import asyncio
import torch
import numpy as np
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'experiments'))

from ingestor import stream_live
from exp_B_hybrid_fi2010.train import QSPINN, N_ASSETS

from lob_extractor import FI2010Extractor

# Initialize the Trained Q-SPINN (Exp B - FI-2010)
# Forced to CPU for live inference to prevent background CUDA context crashes
device = 'cpu'
model = QSPINN().to(device)
ckpt_path = ROOT / "experiments" / "exp_B_hybrid_fi2010" / "qspinn_fi2010.pt"
if ckpt_path.exists():
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    print("✅ Successfully loaded trained FI-2010 weights!")
else:
    print("⚠️ Weights not found, using random initialization")
model.eval()

LOG_FILE = ROOT / "live_inference.jsonl"
last_write = time.time()

# Store last seen prices and order books
prices = {"btc": 64000.0, "eth": 3500.0, "sol": 150.0}
extractors = {"btc": FI2010Extractor(), "eth": FI2010Extractor(), "sol": FI2010Extractor()}
latest_features = {"btc": np.zeros(144), "eth": np.zeros(144), "sol": np.zeros(144)}

from collections import deque
omega_history = deque(maxlen=600)  # 5 minutes at 2Hz (dt=0.5s)

async def process_tick(stream_name, payload):
    global last_write
    
    symbol = ""
    if 'btc' in stream_name: symbol = "btc"
    elif 'eth' in stream_name: symbol = "eth"
    elif 'sol' in stream_name: symbol = "sol"
    
    if not symbol: return
    
    # Extract real prices from the live trade stream
    if 'trade' in stream_name:
        p = float(payload.get('p', 0))
        prices[symbol] = p
        
    # Reconstruct FI-2010 144 features from Depth stream
    elif 'depth' in stream_name:
        bids = payload.get('b', [])
        asks = payload.get('a', [])
        feats = extractors[symbol].update_and_extract(bids, asks)
        if feats is not None:
            latest_features[symbol] = feats
        
    # Throttle inference and logging to 2 Hz so we don't overwhelm the dashboard file
    if time.time() - last_write < 0.5:
        return
    last_write = time.time()
    
    # Use the 144 features for BTC to feed into our FI-2010 trained weights
    stacked_features = latest_features["btc"]
    
    # Keep a rolling window of 2 ticks so the HamiltonianNet can compute velocities (seq_len > 0)
    if not hasattr(process_tick, "feature_history"):
        process_tick.feature_history = deque([stacked_features, stacked_features], maxlen=2)
    else:
        process_tick.feature_history.append(stacked_features)
        
    features_seq = np.stack(process_tick.feature_history)
    features = torch.tensor(features_seq, dtype=torch.float32, device=device).unsqueeze(0)
    
    # LIVE INFERENCE: Pass through Heavy Path to get the Hamiltonian
    with torch.no_grad():
        heavy, L_pde, H, S, L_cov, L_norm = model.heavy_path(features)
        
        # Calculate Numerical Abscissa (omega) from the Symmetric generator component S
        omega = torch.linalg.eigvalsh(S)[:, -1].item()
        omega_history.append(omega)
        
        # V3-06: Time-Integrated Numerical Abscissa (Riemann sum integral)
        Omega_T = sum(omega_history) * 0.5
        
        # V3-07: Phase-Space Divergence Diagnostic
        d_J = torch.trace(S[0]).item()
        
    data = {
        "time": datetime.now().strftime("%H:%M:%S"),
        "omega": omega,
        "Omega_T": Omega_T,
        "d_J": d_J,
        "BTC": prices["btc"],
        "ETH": prices["eth"],
        "SOL": prices["sol"]
    }
    
    # Write to log for dashboard to read
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(data) + "\n")
        
import argparse

async def main():
    parser = argparse.ArgumentParser(description="Q-SPINN Live Pipeline")
    parser.add_argument("--backfill", action="store_true", help="Run historical data backfill before streaming")
    args = parser.parse_args()
    
    print("=== Launching Q-SPINN Production Pipeline ===")
    
    if args.backfill:
        print("Running historical data backfill...")
        from ingestor import backfill_historical
        await backfill_historical()
        print("Backfill complete.")
        
    print("Wiring ingestor WebSocket to PyTorch Q-SPINN Engine...")
    print(f"Outputting inference stream to {LOG_FILE}")
    
    # Clear previous log
    if LOG_FILE.exists():
        LOG_FILE.unlink()
        
    await stream_live(callback=process_tick)

if __name__ == "__main__":
    asyncio.run(main())
