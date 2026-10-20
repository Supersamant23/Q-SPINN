"""
pinn_toy.py -- Component 2 of 3: the Physics-Informed Neural Network, standalone.

WHAT IT DEMONSTRATES
--------------------
A network V(S, t) trained WITHOUT a single ground-truth price label, using only
the Black-Scholes PDE as its supervision signal:

    dV/dt + 0.5 * sigma^2 * S^2 * d2V/dS2 + r * S * dV/dS - r * V = 0

The three ingredients of any PINN are separated out below so they are easy to
point at:

    L_pde   -- the PDE residual at randomly sampled interior collocation points,
               computed with torch.autograd.grad (this IS the "physics")
    L_bc    -- boundary conditions at S = 0 and S = S_max
    L_ic    -- the terminal payoff max(S - K, 0) at t = T
    L_total = L_pde + L_bc + L_ic

Because European call options have a closed-form solution, we can score the
network against exact truth. That number is the honest proof the PDE machinery
works, and it is the thing to show the guide first -- if this converges, the
same autograd scaffolding will carry the quantum-inspired constraint later.

Run:  python pinn_toy.py
"""

import math
import torch
import torch.nn as nn
from common import set_seed

# --- Market parameters for the toy contract ---------------------------------
K, T, R, SIGMA, S_MAX = 100.0, 1.0, 0.05, 0.20, 200.0


def bs_call_closed_form(S, t):
    """Analytic Black-Scholes European call. Ground truth for scoring only."""
    tau = torch.clamp(T - t, min=1e-8)
    d1 = (torch.log(torch.clamp(S, min=1e-8) / K) +
          (R + 0.5 * SIGMA ** 2) * tau) / (SIGMA * torch.sqrt(tau))
    d2 = d1 - SIGMA * torch.sqrt(tau)
    N = lambda x: 0.5 * (1 + torch.erf(x / math.sqrt(2)))
    return S * N(d1) - K * torch.exp(-R * tau) * N(d2)


class PINN(nn.Module):
    """V(S, t). Inputs are rescaled so both live roughly in [0, 1]."""

    def __init__(self, width=64, depth=4):
        super().__init__()
        layers, d = [], 2
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.Tanh()]   # tanh: needs 2nd derivatives
            d = width
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, S, t):
        return self.net(torch.cat([S / S_MAX, t / T], dim=1))


def pde_residual(model, S, t):
    """The physics term. Every derivative here comes from autograd, not finite
    differences -- that is precisely what makes it a PINN and not a numerical
    PDE solver with a neural surrogate bolted on."""
    S = S.requires_grad_(True)
    t = t.requires_grad_(True)
    V = model(S, t)

    g = lambda y, x: torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]
    V_t = g(V, t)
    V_S = g(V, S)
    V_SS = g(V_S, S)

    # === SWAP-07 ============================================================
    # TOY  : the real-valued Black-Scholes operator. Constant sigma, constant r,
    #        one underlying, no dividends.
    # FINAL: this single line is where Q-SPINN's actual contribution goes.
    #        Replace it with the Schrodinger-analogue residual on a
    #        complex-valued amplitude psi:
    #
    #            i * d|psi>/dt = H_hat |psi>
    #        =>  residual = d(psi)/dt + i * H_hat @ psi
    #
    #        H_hat must be Hermitian (H = H^dagger) or the evolution is not
    #        norm-preserving and the "probability amplitude" reading collapses.
    #        See hybrid_toy.py, which already implements exactly this.
    #        Keep the Black-Scholes version alive as the PINN-only ablation arm.
    # OWNER: PINN layer owner (Phase 2), with Abi benchmarking both arms.
    # ========================================================================
    return V_t + 0.5 * SIGMA ** 2 * S ** 2 * V_SS + R * S * V_S - R * V


def main():
    set_seed(0)
    model = PINN()
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1500, gamma=0.5)

    # === SWAP-08 ============================================================
    # TOY  : fixed loss weights of 1 / 1 / 1.
    # FINAL: PINN loss terms routinely differ by orders of magnitude and the
    #        balance decides whether it converges at all. Use adaptive weighting
    #        (learning-rate annealing on the gradients, or NTK-based weights)
    #        and report the weights you used -- reviewers ask.
    # ========================================================================
    W_PDE, W_BC, W_TC = 1.0, 1.0, 1.0
    N_COLLOC = 2000

    print("Training on the PDE alone -- no option-price labels are ever used.\n")
    for step in range(6000):
        # interior collocation points, resampled every step
        S = torch.rand(N_COLLOC, 1) * S_MAX
        t = torch.rand(N_COLLOC, 1) * T
        L_pde = pde_residual(model, S, t).pow(2).mean()

        # boundary conditions
        tb = torch.rand(256, 1) * T
        V0 = model(torch.zeros_like(tb), tb)                       # V(0, t) = 0
        Vmax = model(torch.full_like(tb, S_MAX), tb)               # V(Smax, t)
        target_max = S_MAX - K * torch.exp(-R * (T - tb))
        L_bc = V0.pow(2).mean() + (Vmax - target_max).pow(2).mean()

        # terminal condition (the payoff)
        St = torch.rand(512, 1) * S_MAX
        VT = model(St, torch.full_like(St, T))
        L_tc = (VT - torch.clamp(St - K, min=0.0)).pow(2).mean()

        loss = W_PDE * L_pde + W_BC * L_bc + W_TC * L_tc
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()

        if step % 1000 == 0 or step == 5999:
            with torch.no_grad():
                Sg = torch.linspace(1, S_MAX, 400).unsqueeze(1)
                tg = torch.zeros_like(Sg)
                mae = (model(Sg, tg) - bs_call_closed_form(Sg, tg)).abs().mean()
            print(f"step {step:5d} | L_pde {L_pde.item():.3e} "
                  f"| L_bc {L_bc.item():.3e} | L_tc {L_tc.item():.3e} "
                  f"| MAE vs analytic @ t=0: {mae.item():.3f}")

    # --- final scoring against the closed-form solution ---------------------
    with torch.no_grad():
        Sg = torch.linspace(1, S_MAX, 400).unsqueeze(1)
        print("\n--- PINN-only result: learned surface vs analytic Black-Scholes ---")
        print(f"{'t':>6} {'MAE':>10} {'max err':>10}")
        for tv in [0.0, 0.25, 0.5, 0.75]:
            tg = torch.full_like(Sg, tv)
            err = (model(Sg, tg) - bs_call_closed_form(Sg, tg)).abs()
            print(f"{tv:6.2f} {err.mean().item():10.4f} {err.max().item():10.4f}")

        print(f"\n{'spot':>6} {'PINN':>10} {'analytic':>10}")
        for s in [60.0, 90.0, 100.0, 110.0, 150.0]:
            st = torch.tensor([[s]])
            zt = torch.tensor([[0.0]])
            print(f"{s:6.0f} {model(st, zt).item():10.4f} "
                  f"{bs_call_closed_form(st, zt).item():10.4f}")

    torch.save(model.state_dict(), "pinn_toy.pt")
    print("\nsaved pinn_toy.pt")


if __name__ == "__main__":
    main()
