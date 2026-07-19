"""
potassium_conductance.py

MIT OCW 9.40 - Lecture 4: Hodgkin-Huxley Model, Part 1 (potassium)

Closes out the potassium half of the model: simulate the n gate's step
response to a voltage-clamp depolarization (piecewise-constant voltage,
exact exponential-relaxation solution within each segment - see
simulate_gating_variable in utils.py), then raise it to the
4th power to get G_K(t) = G_K_bar * n^4, and confirm it reproduces the
textbook "delayed rectifier" shape: n starts small, jumps toward a large
n_inf on depolarization, and relaxes up slowly (tau_n is comparatively
large) - giving a conductance that ramps up and then STAYS ON for as
long as the depolarization holds. No inactivation gate needed here;
that's the whole point of the contrast with sodium in sodium_conductance.py.

Run interactively in VS Code with the Jupyter extension using the '# %%' cell markers,
or run top-to-bottom as a normal script.
"""

# %%
import os
import numpy as np
import matplotlib.pyplot as plt
from utils import get_figures_dir, alpha_n, beta_n, steady_state_and_tau, simulate_gating_variable, G_K_BAR, E_K

FIGURES_DIR = get_figures_dir(__file__)

# %%
# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
# Depolarization-from-rest convention throughout (see utils.py).
V_HYPERPOLARIZED = -10.0   # holding potential before the step (mV, dep-from-rest)
V_DEPOLARIZED = 50.0       # stepped-to potential (mV, dep-from-rest)

T_STEP = 2.0     # time of the voltage step (ms)
T_MAX = 20.0     # total simulated duration (ms)
N_STEPS = 2000
TIME = np.linspace(0, T_MAX, N_STEPS)

# a single voltage step: hyperpolarized until T_STEP, then depolarized
VOLTAGE_SEQUENCE = [
    (0.0, T_STEP, V_HYPERPOLARIZED),
    (T_STEP, T_MAX, V_DEPOLARIZED),
]


# %%
# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_n_and_conductance(t, n_trace, G_K_trace, t_step, V_hyperpolarized, V_depolarized, save_path=None):
    """Two stacked subplots: n(t) itself, and the resulting G_K(t) = G_K_bar * n^4."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    ax1.plot(t, n_trace, color="tab:blue", linewidth=2)
    ax1.axvline(t_step, color="black", linestyle=":", linewidth=1)
    ax1.set_ylabel("n (activation gate)")
    ax1.set_title(f"Potassium Activation: Step from {V_hyperpolarized:.0f} to {V_depolarized:.0f} mV (dep. from rest)")

    ax2.plot(t, G_K_trace, color="tab:purple", linewidth=2)
    ax2.axvline(t_step, color="black", linestyle=":", linewidth=1, label="Voltage step")
    ax2.set_xlabel("Time (ms)")
    ax2.set_ylabel("G_K = G_K_bar * n^4 (mS/cm^2)")
    ax2.legend()

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150)
    plt.show()


# %%
# ---------------------------------------------------------------------------
# Demo: step response, n^4 shape confirms "turns on and stays on"
# ---------------------------------------------------------------------------
def run_step_response_demo():
    """Simulates n's step response, derives G_K = G_K_bar*n^4, checks it plateaus rather than decaying."""
    n_inf_hyper, tau_n_hyper = steady_state_and_tau(alpha_n(V_HYPERPOLARIZED), beta_n(V_HYPERPOLARIZED))
    n_inf_depol, tau_n_depol = steady_state_and_tau(alpha_n(V_DEPOLARIZED), beta_n(V_DEPOLARIZED))

    print(f"[n step response] at V={V_HYPERPOLARIZED:.0f} mV: n_inf={n_inf_hyper:.4f}, tau_n={tau_n_hyper:.4f} ms")
    print(f"  at V={V_DEPOLARIZED:.0f} mV: n_inf={n_inf_depol:.4f}, tau_n={tau_n_depol:.4f} ms")
    print(f"  n_inf rises with depolarization: {n_inf_depol > n_inf_hyper} (should be True)")

    n_trace = simulate_gating_variable("n", VOLTAGE_SEQUENCE, TIME)
    G_K_trace = G_K_BAR * n_trace ** 4

    # sanity check: G_K should be near its plateau value by the end of the
    # simulation (well after the step, since T_MAX >> tau_n at V_DEPOLARIZED),
    # not decaying back down - the defining "stays on" behavior
    G_K_plateau = G_K_BAR * n_inf_depol ** 4
    G_K_final = G_K_trace[-1]
    print(f"  G_K at end of sim = {G_K_final:.4f}, theoretical plateau = {G_K_plateau:.4f} "
          f"(should be close - confirms K+ conductance turns on and STAYS on)")

    # sanity check: G_K should be monotonically non-decreasing after the step
    # (no dip, unlike what an inactivation gate would produce)
    post_step_mask = TIME >= T_STEP
    G_K_post = G_K_trace[post_step_mask]
    is_monotonic = np.all(np.diff(G_K_post) >= -1e-6)
    print(f"  G_K monotonically rising after step: {is_monotonic} "
          f"(should be True - no inactivation dip for potassium)")

    plot_n_and_conductance(
        TIME, n_trace, G_K_trace, T_STEP, V_HYPERPOLARIZED, V_DEPOLARIZED,
        save_path=os.path.join(FIGURES_DIR, "n_activation_step_response.png"),
    )


# %%
if __name__ == "__main__":
    run_step_response_demo()

# %%
