# Q-SPINN — Toy Model / Implementation Template

A runnable, deliberately small version of the Q-SPINN architecture from the
proposal (*Spiking-Gated, Quantum-Inspired Physics-Informed Neural Network for
Regime-Aware Limit Order Book Prediction*, Dr. Biswajit Das, July 2026).

**What this is:** a skeleton that runs end-to-end today, with every place that
must change before it becomes the final-year deliverable marked in the source as
`SWAP-NN` and indexed in the table below.

**What this is not:** a result. The data is synthetic and the numbers below are
from one seed on one split. Nothing here is evidence that Q-SPINN works on real
markets. It is evidence that the architecture is buildable and that we know
exactly what is left to do.

---

## Files

| File | What it contains |
|---|---|
| `common.py` | Synthetic LOB stream generator, FI-2010-shaped features and labels, spike encoder, metrics |
| `snn_toy.py` | Component 1 — LIF spiking network, standalone |
| `pinn_toy.py` | Component 2 — Black–Scholes PINN, standalone |
| `hybrid_toy.py` | Component 3 — the full Q-SPINN: SNN gate → complex amplitude layer → Hamiltonian → PINN head |

```bash
pip install torch
python snn_toy.py          # the SNN alone
python pinn_toy.py         # the PINN alone
python hybrid_toy.py       # the hybrid, plus a compute/accuracy Pareto sweep
python hybrid_toy.py --ablation   # Q-SPINN vs SNN-only vs PINN-only vs LSTM
```

No GPU needed. Each script finishes in a few minutes on a laptop CPU.

---

## What each component demonstrates

### 1. `snn_toy.py` — the SNN is real, not a wrapper

The Leaky Integrate-and-Fire neuron is written out longhand so the recurrence is
visible on screen:

```
U[t] = beta * U[t-1] + I[t] - theta * S[t-1]      leak, integrate, reset
S[t] = 1 if U[t] > theta else 0                   fire
```

The Heaviside step has zero gradient everywhere it is defined, so plain autograd
cannot train it. The fix — a **surrogate gradient** — is implemented explicitly:
hard step forward, smooth arctan derivative backward. This is the one mechanism
that makes SNNs trainable, and it is worth being able to explain from memory.

Result: **macro-F1 0.52** against a 0.36 majority-class baseline, with only
**3.7% of hidden neurons firing per tick**.

### 2. `pinn_toy.py` — the physics term is real, not a regulariser in name

`V(S, t)` is trained on the Black–Scholes PDE **with no price labels at all**.
The derivatives come from `torch.autograd.grad`, not finite differences — that is
what makes it a PINN rather than a neural surrogate for a numerical solver.

Because European calls have a closed form, we can score it against exact truth:
**mean absolute error ≈ 0.05 on option values spanning 0–100**. That number is
the proof the PDE machinery converges. The same autograd scaffolding carries the
quantum-inspired constraint later.

### 3. `hybrid_toy.py` — the two, connected

```
LOB window (B, 20, 120)
   → delta encoder                 events, not floats
   → [1] SNN GATE (LIF)            spike train + binary wake-up decision g
        g = 0 → DORMANT: answer from the SNN's own cheap readout, heavy path never runs
        g = 1 → WAKE:
              → [2] COMPLEX AMPLITUDE LAYER   psi in C^8, ||psi|| = 1, rho = |psi><psi|
              → [3] HAMILTONIAN NET           H(order-flow velocity), Hermitian by construction
              → [4] SCHRODINGER CONSTRAINT    residual of i d|psi>/dt = H|psi>
              → [5] HEAD                      3-class direction
```

Three details worth defending out loud:

- **`H = (A + A†)/2` makes Hermiticity structural.** If `H` were not Hermitian
  the evolution would not preserve norm, `|psi|²` would stop being a probability
  distribution, and the entire quantum-probability framing collapses. The run
  prints `||H − H†||` and it is exactly `0`. Verified, not asserted.
- **`H` is emitted by a small network reading order-flow velocity.** That is
  Risk 3 of the proposal ("defining the Hamiltonian is subjective") mitigated
  exactly as the proposal promised.
- **The Hamiltonian does work, not decoration.** The head classifies partly from
  `psi_next = psi_T − i·H·psi_T·dt` — a state the physics predicted, not only the
  state that was observed.

---

## Results from the current toy (one seed, synthetic data)

```
model                              acc  macro-F1    gate      MACs/win
----------------------------------------------------------------------
LSTM baseline (dense)            0.522     0.515     n/a       942,080
SNN only (never wakes)           0.517     0.520   0.000        63,155
PINN only (always awake)         0.466     0.455   1.000       420,467
Q-SPINN (full hybrid)            0.509     0.511   0.040        77,387
----------------------------------------------------------------------
majority-class baseline          0.361
```

Gate selectivity: fires on **65% of regime-shift windows** and **0% of quiet
windows**. Compute: **≈90% of MACs avoided** versus an always-dense equivalent.

**The honest reading, which is what to say to the guide:**

1. Every arm clears the majority baseline, so the pipeline learns the
   order-flow signal. The plumbing works.
2. No arm dominates on accuracy, and the spread is too small on one seed to be a
   real difference. **The hybrid's demonstrated advantage is compute, not
   accuracy.** That is the only claim currently supported.
3. The heavy path scored 0.58 on validation and 0.46 on the later test window.
   That gap *is* Risk 1 (non-stationarity), appearing on synthetic data that
   was not even designed to be adversarial. On real LOB data it will be worse.
4. The gate is selective **only because the toy generator hands it a
   ground-truth regime mask**. Real data has no such label. See SWAP-15.

---

## The SWAP index — every change needed for the real deliverable

| # | File | Currently | Must become | Phase | Owner |
|---|---|---|---|---|---|
| 01 | `common.py` | Synthetic GBM + order-flow imbalance | FI-2010 (`NoAuction_DecPre`) + Binance depth snapshots. Return signature stays identical, so nothing downstream changes | 1 | Abi |
| 02 | `common.py` | Scaler fitted on the whole stream | Fit on the train window only, or use FI-2010's pre-normalised files. Current version leaks future statistics — fatal for walk-forward | 1 | Abi |
| 03 | `common.py` | Delta spike encoding, fixed threshold | A real research choice: delta vs rate vs latency coding vs native order events. Ablate at least two | 2 | SNN owner |
| 04 | `snn_toy.py` | arctan surrogate, slope 1 | snnTorch surrogates (fast-sigmoid / triangular), slope treated as a hyperparameter | 2 | SNN owner |
| 05 | `snn_toy.py` | Hand-written LIF, reset by subtraction, no refractory period | `snn.Leaky` / `snn.Synaptic`, learnable thresholds, refractory dynamics. Everything else in the file is unchanged by this swap | 2 | SNN owner |
| 06 | `snn_toy.py` | Val = last 15% of train, simple early stopping | Purged, walk-forward CV with an embargo gap, plus Deflated Sharpe Ratio | 3 | Abi |
| 07 | `pinn_toy.py` | Real-valued Black–Scholes operator | The complex Schrödinger-analogue residual (already implemented in `hybrid_toy.py`). Keep Black–Scholes alive as the PINN-only ablation arm | 2 | PINN owner |
| 08 | `pinn_toy.py` | Loss weights fixed at 1/1/1 | Adaptive weighting (gradient annealing or NTK-based). PINN loss terms differ by orders of magnitude and the balance decides convergence | 2 | PINN owner |
| 09 | `hybrid_toy.py` | One gate decision per window | Per-tick streaming gate, with a refractory period and hysteresis so it does not chatter or wake on 40 consecutive ticks | 2 | SNN owner |
| 10 | `hybrid_toy.py` | `H` is a free learned matrix | A **formulated** Hamiltonian (Objective 1): `H = T(momentum) + V(distance from fundamental value)`, each term separately parameterised and inspectable | 1–2 | PINN / maths owner |
| 11 | `hybrid_toy.py` | `lambda_pde`, `lambda_gate` hand-picked; `DT = 1` tick | Sweep `lambda_pde` (it decides whether this is a PINN or a classifier with a decorative loss term — the guide *will* ask); sweep `lambda_gate` for a Pareto curve; give `DT` real time units | 3 | Abi |
| 12 | `hybrid_toy.py` | One seed, one split | ≥5 seeds, mean ± std, Deflated Sharpe Ratio once the metric becomes P&L | 3 | Abi |
| 13 | `hybrid_toy.py` | Analytic MAC count | Measured wall-clock and joules on target hardware, or a Loihi cycle model. Until then say "MACs avoided", never "energy saved" — easiest claim for an examiner to attack | 3 | Abi |
| 14 | `hybrid_toy.py` | Three fixed training stages | Try joint fine-tuning at low LR after staging; try Gumbel-Softmax or REINFORCE instead of straight-through. "We tried end-to-end and it collapsed, here is the evidence" is a legitimate result | 2–3 | Team |
| 15 | `hybrid_toy.py` | **Gate supervised by ground-truth regime mask** | **See below — highest-risk item in the project** | 1 | Whole team |

---

## SWAP-15 — the open question to take to the guide

The central claim of the proposal is that *the SNN wakes the PINN upon detection
of a statistically significant regime-shift event*. That claim is unfalsifiable
until "regime shift" has a definition, and real LOB data does not come with one.

In this toy, the gate is selective only because the generator hands it a
ground-truth mask. Removing that supervision was tried first: the gate reached
intermediate firing rates but showed **no preference at all for regime windows**
— it learned to save compute, and learned nothing about regimes.

Three defensible paths, and the team has to pick one:

**(a) Construct a proxy label.** Realised-volatility jump, order-flow-imbalance
z-score above *k*σ, or an HMM / change-point detector run offline over the
training window. Keeps the "regime-aware" title honest. Most work.

**(b) Drop the supervision entirely.** Let the gate learn purely from the
accuracy-versus-compute trade-off. Already tested here; produces a working
conditional-computation gate with no regime semantics.

**(c) Reframe the claim.** Stop calling it regime detection, call it a learned
conditional-computation gate. That is what (b) actually produces, and it is still
a publishable contribution — it just is not the title currently on the proposal.

**(a) is the honest path if the "regime-aware" title stays.** This is a scoping
decision, not an implementation detail, which is why it belongs in the meeting.

---

## Two things that broke while building this, worth mentioning

Both are load-bearing engineering lessons, not trivia — they will recur.

**1. The gate collapses in two opposite directions under end-to-end training.**
Start it shut and the heavy path never receives gradient, so it never becomes
worth waking, so the gate stays shut forever. Start it open and the heavy path
does all the work, the SNN receives no error signal, its neurons stop firing
entirely, and the gate becomes a constant. The fix is **staged training**: train
each branch alone, then train the gate to arbitrate between two already-competent
branches (see SWAP-14 for what to try beyond this).

**2. An unbounded gate logit silently kills the surrogate gradient.** The arctan
surrogate is `1/(1 + (πv)²)`. If `v` grows large the gradient is ~0 and the gate
freezes wherever it initialised — with no error, no warning, and a perfectly
normal-looking loss curve. Squashing the logit with `tanh(x/4)` before
thresholding fixes it. This is a single line in `SNNGate.forward` and it is the
difference between a gate that learns and a gate that is a constant.
