"""
snn_toy.py -- Component 1 of 3: the Spiking Neural Network, standalone.

WHAT IT DEMONSTRATES
--------------------
A Leaky Integrate-and-Fire (LIF) network trained by backpropagation-through-time
with a surrogate gradient, classifying LOB mid-price direction from spike trains.

Everything about the neuron is written out longhand rather than imported, so the
membrane-potential recurrence is visible on screen:

    U[t] = beta * U[t-1] + I[t] - theta * S[t-1]        (leak, integrate, reset)
    S[t] = 1 if U[t] > theta else 0                     (fire)

The Heaviside step has zero gradient everywhere it is defined, which is why SNNs
cannot be trained by plain autograd. The standard fix -- used here -- is a
surrogate gradient: keep the hard step in the forward pass, substitute a smooth
derivative in the backward pass.

Run:  python snn_toy.py
"""

import math
import torch
import torch.nn as nn
from common import (generate_lob_stream, make_windows, delta_encode,
                    macro_f1, set_seed, N_FEATURES)


# ----------------------------------------------------------------------------
# The surrogate gradient. This is the single trick that makes SNNs trainable.
# ----------------------------------------------------------------------------
class SurrogateSpike(torch.autograd.Function):
    """Forward: hard Heaviside. Backward: arctan surrogate derivative."""

    @staticmethod
    def forward(ctx, v):                 # v = U - theta
        ctx.save_for_backward(v)
        return (v > 0).float()

    @staticmethod
    def backward(ctx, grad_out):
        (v,) = ctx.saved_tensors
        # === SWAP-04 ====================================================
        # TOY  : arctan surrogate, slope fixed at 1.
        # FINAL: snnTorch exposes fast-sigmoid, triangular, and sigmoid
        #        surrogates with a tunable slope. The choice measurably
        #        affects convergence; treat it as a hyperparameter in the
        #        Phase 3 ablation, not a constant.
        # ================================================================
        return grad_out / (1.0 + (math.pi * v) ** 2)


spike_fn = SurrogateSpike.apply


class LIFLayer(nn.Module):
    """A layer of Leaky Integrate-and-Fire neurons with reset-by-subtraction."""

    def __init__(self, n_in, n_out, beta=0.9, theta=1.0):
        super().__init__()
        self.fc = nn.Linear(n_in, n_out)
        # beta is learnable and squashed to (0,1) -- a per-neuron time constant.
        self.beta_raw = nn.Parameter(torch.logit(torch.full((n_out,), beta)))
        self.theta = theta

    def forward(self, x_seq):
        """x_seq: (B, T, n_in) -> spikes (B, T, n_out), final membrane (B, n_out)"""
        B, T, _ = x_seq.shape
        beta = torch.sigmoid(self.beta_raw)
        U = torch.zeros(B, self.fc.out_features, device=x_seq.device)
        S = torch.zeros_like(U)
        out = []
        for t in range(T):
            I = self.fc(x_seq[:, t])
            # === SWAP-05 ================================================
            # TOY  : reset by subtraction, soft threshold, no refractory
            #        period, no synaptic (second-order) current.
            # FINAL: snnTorch's Leaky / Synaptic / Alpha neurons give you
            #        reset-to-zero, learnable thresholds, and refractory
            #        dynamics for free. Swap this loop for
            #        snn.Leaky(beta=..., spike_grad=...) and the rest of
            #        the file is unchanged.
            # OWNER: SNN layer owner (Phase 2).
            # ============================================================
            U = beta * U + I - self.theta * S
            S = spike_fn(U - self.theta)
            out.append(S)
        return torch.stack(out, dim=1), U


class LeakyReadout(nn.Module):
    """Non-spiking leaky integrator. Accumulates evidence, never fires."""

    def __init__(self, n_in, n_out, beta=0.9):
        super().__init__()
        self.fc = nn.Linear(n_in, n_out)
        self.beta = beta

    def forward(self, s_seq):
        U = torch.zeros(s_seq.shape[0], self.fc.out_features, device=s_seq.device)
        for t in range(s_seq.shape[1]):
            U = self.beta * U + self.fc(s_seq[:, t])
        return U                                   # logits


class ToySNN(nn.Module):
    def __init__(self, n_in=2 * N_FEATURES, hidden=128, n_classes=3):
        super().__init__()
        self.lif1 = LIFLayer(n_in, hidden)
        self.lif2 = LIFLayer(hidden, hidden)
        self.readout = LeakyReadout(hidden, n_classes)

    def forward(self, spikes):
        s1, _ = self.lif1(spikes)
        s2, _ = self.lif2(s1)
        logits = self.readout(s2)
        # Spike rate is the currency of an SNN: it is proportional to energy.
        rate = 0.5 * (s1.mean() + s2.mean())
        return logits, rate


def main():
    set_seed(0)
    feats, labels, regime = generate_lob_stream(n_ticks=14000, seed=0)
    Xtr, ytr, _, Xte, yte, _ = make_windows(feats, labels, regime)
    Str, Ste = delta_encode(Xtr), delta_encode(Xte)

    # === SWAP-06 ====================================================
    # TOY  : validation = the last 15% of the training window; test =
    #        one contiguous later block. Selection is on macro-F1.
    # FINAL: Phase 3 mandates PURGED, WALK-FORWARD cross-validation --
    #        multiple rolling (train, gap, test) folds with an embargo
    #        gap so that overlapping label windows cannot leak across
    #        the boundary. Report the Deflated Sharpe Ratio alongside
    #        macro-F1 so the number survives multiple-testing scrutiny.
    # OWNER: Abi (experimental evaluation and baselines).
    # ================================================================
    vcut = int(0.85 * len(Str))
    Sva, yva = Str[vcut:], ytr[vcut:]
    Str, ytr = Str[:vcut], ytr[:vcut]

    print(f"train {tuple(Str.shape)}  val {tuple(Sva.shape)}  test {tuple(Ste.shape)}")
    print(f"input spike density: {Str.mean():.3f} "
          f"(fraction of (tick, feature) slots carrying an event)")
    print(f"majority-class baseline on test: "
          f"{(yte == yte.mode().values).float().mean():.3f}\n")

    model = ToySNN()
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    ce = nn.CrossEntropyLoss()

    LAMBDA_RATE = 3e-3      # sparsity pressure -- fewer spikes, less energy
    n, bs = len(Str), 128
    best_f1, best_state = -1.0, None
    for ep in range(20):
        model.train()
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            logits, rate = model(Str[b])
            loss = ce(logits, ytr[b]) + LAMBDA_RATE * rate
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(b)

        model.eval()
        with torch.no_grad():
            vf1 = macro_f1(model(Sva)[0].argmax(1), yva)
        if vf1 > best_f1:
            best_f1 = vf1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if (ep + 1) % 4 == 0 or ep == 0:
            print(f"epoch {ep+1:2d} | train loss {tot/n:.4f} | val macro-F1 {vf1:.3f}")

    model.load_state_dict(best_state)          # early stopping, done properly
    model.eval()
    with torch.no_grad():
        lg, rt = model(Ste)
        pred = lg.argmax(1)
    print("\n--- SNN-only result (best-on-validation checkpoint) ---")
    print(f"accuracy   {(pred == yte).float().mean():.3f}")
    print(f"macro-F1   {macro_f1(pred, yte):.3f}")
    print(f"spike rate {rt:.4f}  ->  ~{100*(1-rt):.1f}% of hidden neurons "
          f"stay silent per tick; those synapses cost nothing on "
          f"event-driven hardware.")
    torch.save(model.state_dict(), "snn_toy.pt")
    print("saved snn_toy.pt")


if __name__ == "__main__":
    main()
