"""
common.py -- Synthetic Limit Order Book (LOB) generator + shared utilities.

WHAT THIS IS
------------
A stand-in for the real Q-SPINN data layer. It fabricates a multi-asset LOB
stream whose *shape and semantics* match FI-2010 exactly, so that replacing it
with the real dataset is a one-function change and nothing downstream moves.

FI-2010 feature convention reproduced here:
    per asset, 10 book levels x 4 quantities = 40 features
    ordered as [p_ask_1, v_ask_1, p_bid_1, v_bid_1, ..., p_bid_10, v_bid_10]
FI-2010 label convention reproduced here:
    3-class mid-price direction over horizon k, with a dead-zone alpha
    0 = DOWN, 1 = STATIONARY, 2 = UP

The generator also injects *regime shifts* (jump + volatility burst) and returns
a ground-truth mask for them. Real markets do not ship you that mask -- it exists
here only so the SNN gate can be sanity-checked during the toy phase.
"""

import math
import numpy as np
import torch

# ----------------------------------------------------------------------------
# Scope constants. Phase 1 of the proposal fixes a three-asset basket.
# ----------------------------------------------------------------------------
N_ASSETS = 3                    # AAPL / MSFT / NVDA stand-ins
LEVELS = 10                     # book depth, matches FI-2010
FEATS_PER_ASSET = LEVELS * 4    # 40
N_FEATURES = N_ASSETS * FEATS_PER_ASSET   # 120
SEQ_LEN = 20                    # lookback window in ticks
HORIZON = 5                     # prediction horizon k
ALPHA = 0.0015                  # dead-zone for the STATIONARY class


# === SWAP-01 ================================================================
# TOY  : synthetic correlated GBM with injected regime shifts.
# FINAL: load FI-2010 (NoAuction_DecPre / DecTrain files) and Binance depth
#        snapshots. Keep the return signature identical -- everything
#        downstream (SNN, complex layer, PINN, ablation) is dataset-agnostic
#        because it only ever sees (features, labels, mask).
# OWNER: Abi (dataset preprocessing).
# ============================================================================
def generate_lob_stream(n_ticks=20000, seed=0):
    """Return (features, labels, regime_mask) for one continuous tick stream.

    features    : float32 (n_ticks, N_FEATURES)   raw LOB snapshot per tick
    labels      : int64   (n_ticks,)              3-class direction at horizon k
    regime_mask : float32 (n_ticks,)              1.0 inside an injected shift

    The stream is not pure Brownian motion. Mid-price returns are driven by a
    persistent ORDER-FLOW IMBALANCE process, which is also what shapes the
    visible book depth. That is deliberate: order-flow imbalance -> short-horizon
    price impact is the best-documented predictable effect in real microstructure,
    and it is the effect FI-2010 models actually exploit. Without it the toy would
    have nothing to learn and every model would score at the majority-class
    baseline, which teaches you nothing about whether the architecture works.
    """
    rng = np.random.default_rng(seed)

    dt = 1.0 / 252 / 390
    base_vol = np.array([0.25, 0.20, 0.45])       # NVDA is the noisy one
    base_price = np.array([190.0, 410.0, 120.0])
    KAPPA = 6e-4                                  # price impact per unit imbalance

    corr = np.array([[1.00, 0.62, 0.55],
                     [0.62, 1.00, 0.48],
                     [0.55, 0.48, 1.00]])
    chol = np.linalg.cholesky(corr)

    # --- regime shift schedule ---------------------------------------------
    regime = np.zeros(n_ticks, dtype=np.float32)
    for _ in range(max(4, n_ticks // 600)):
        start = rng.integers(SEQ_LEN, n_ticks - HORIZON - 60)
        regime[start:start + rng.integers(20, 60)] = 1.0

    # --- persistent order-flow imbalance, one AR(1) per asset --------------
    PHI = 0.93                                    # imbalance autocorrelation
    imb = np.zeros((n_ticks, N_ASSETS))
    for t in range(1, n_ticks):
        shock = rng.standard_normal(N_ASSETS) * (2.5 if regime[t] else 1.0)
        imb[t] = PHI * imb[t - 1] + math.sqrt(1 - PHI ** 2) * shock
    imb = np.tanh(imb * 0.8)                      # squash into (-1, 1)

    # --- mid-price paths: impact term + correlated diffusion + jumps -------
    mids = np.zeros((n_ticks, N_ASSETS))
    mids[0] = base_price
    for t in range(1, n_ticks):
        z = chol @ rng.standard_normal(N_ASSETS)
        vol = base_vol * (4.0 if regime[t] else 1.0)
        impact = KAPPA * imb[t - 1]               # yesterday's imbalance moves today
        jump = 0.0
        if regime[t] and not regime[t - 1]:
            jump = rng.choice([-1, 1]) * 0.012
        mids[t] = mids[t - 1] * np.exp(impact + vol * math.sqrt(dt) * z + jump)

    # --- expand each mid into a 10-level book ------------------------------
    # Depth is asymmetric in proportion to the imbalance, so the imbalance is
    # OBSERVABLE from the features -- the model has to discover it, not be told.
    feats = np.zeros((n_ticks, N_FEATURES), dtype=np.float32)
    for a in range(N_ASSETS):
        mid = mids[:, a]
        spread = mid * 0.0002 * (3.0 if base_vol[a] > 0.3 else 1.0)
        for lvl in range(LEVELS):
            off = spread * (lvl + 1)
            depth = rng.lognormal(mean=6.0 - 0.25 * lvl, sigma=0.35, size=n_ticks)
            depth *= np.where(regime > 0, 0.35, 1.0)      # book thins in a shift
            base = a * FEATS_PER_ASSET + lvl * 4
            feats[:, base + 0] = mid + off                          # p_ask
            feats[:, base + 1] = depth * (1 - 0.6 * imb[:, a])      # v_ask
            feats[:, base + 2] = mid - off                          # p_bid
            feats[:, base + 3] = depth * (1 + 0.6 * imb[:, a])      # v_bid

    # --- FI-2010 style labels on asset 0's mid ------------------------------
    target_mid = mids[:, 0]
    fut = np.convolve(target_mid, np.ones(HORIZON) / HORIZON, mode="full")[HORIZON - 1:]
    fut = fut[:n_ticks]
    shifted = np.concatenate([fut[HORIZON:], np.repeat(fut[-1], HORIZON)])
    pct = (shifted - target_mid) / target_mid
    labels = np.ones(n_ticks, dtype=np.int64)         # STATIONARY
    labels[pct > ALPHA] = 2                           # UP
    labels[pct < -ALPHA] = 0                          # DOWN

    return feats, labels, regime


# === SWAP-02 ================================================================
# TOY  : z-score normalisation fitted on the whole stream.
# FINAL: fit the scaler on the TRAIN window only and apply it forward, or use
#        FI-2010's pre-normalised DecPre files directly. Fitting on the full
#        stream leaks future statistics -- harmless for a demo, fatal in the
#        walk-forward backtest of Phase 3.
# ============================================================================
def make_windows(feats, labels, regime, seq_len=SEQ_LEN, train_frac=0.7):
    """Slice the stream into overlapping windows and split it chronologically."""
    mu, sd = feats.mean(0, keepdims=True), feats.std(0, keepdims=True) + 1e-8
    x = (feats - mu) / sd

    n = len(x) - seq_len - HORIZON
    idx = np.arange(n)
    X = np.stack([x[i:i + seq_len] for i in idx]).astype(np.float32)
    y = labels[idx + seq_len - 1]
    r = np.stack([regime[i:i + seq_len].max() for i in idx]).astype(np.float32)

    cut = int(n * train_frac)     # chronological split -- never shuffle first
    return (torch.tensor(X[:cut]), torch.tensor(y[:cut]), torch.tensor(r[:cut]),
            torch.tensor(X[cut:]), torch.tensor(y[cut:]), torch.tensor(r[cut:]))


# === SWAP-03 ================================================================
# TOY  : delta (change-based) spike encoding with a fixed threshold.
# FINAL: this is a genuine research choice, not boilerplate. Candidates are
#        delta modulation (here), rate coding, latency/time-to-first-spike
#        coding, or feeding raw order-flow events straight through as native
#        spikes. Phase 2 should ablate at least two of these.
# ============================================================================
def delta_encode(x, threshold=0.20):
    """(B, T, F) real-valued -> (B, T, 2F) ternary spikes as {ON, OFF} channels.

    An SNN does not consume floats; it consumes events. A delta encoder emits a
    spike only when a feature moves by more than `threshold`, which is exactly
    the "99% of ticks carry no signal" claim made concrete.
    """
    d = torch.zeros_like(x)
    d[:, 1:] = x[:, 1:] - x[:, :-1]
    on = (d > threshold).float()
    off = (d < -threshold).float()
    return torch.cat([on, off], dim=-1)


# ----------------------------------------------------------------------------
# Metrics -- implemented by hand so the toy has zero dependency beyond torch.
# ----------------------------------------------------------------------------
def macro_f1(pred, true, n_classes=3):
    f1s = []
    for c in range(n_classes):
        tp = ((pred == c) & (true == c)).sum().item()
        fp = ((pred == c) & (true != c)).sum().item()
        fn = ((pred != c) & (true == c)).sum().item()
        denom = 2 * tp + fp + fn
        f1s.append(0.0 if denom == 0 else 2 * tp / denom)
    return sum(f1s) / len(f1s)


def set_seed(s=0):
    torch.manual_seed(s)
    np.random.seed(s)
