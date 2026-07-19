"""
gating_kinetics.py

MIT OCW 9.40 - Lecture 4/5: Hodgkin-Huxley Model (Part 0 - shared gating math)

Before building any single conductance, this script lays out the piece of
math all three gates (n, m, h) share: given alpha(V) and beta(V), every
gate reduces to the same two numbers at any voltage -
    x_inf(V) = alpha(V) / (alpha(V) + beta(V))
    tau_x(V) = 1 / (alpha(V) + beta(V))
and the same relaxation-toward-x_inf differential equation. What makes n,
m, and h behave so differently isn't the math - it's the shape of their
individual alpha/beta functions.

This is also the plot that makes the Lecture 5 quiz point (h is a
mirror-image sigmoid of m and n) directly visible: n_inf and m_inf both
rise from 0 to 1 with depolarization; h_inf falls from 1 to 0 - same
underlying machinery, opposite voltage dependence, which is exactly
why sodium's m and h together can produce a transient (rise-then-fall)
current while potassium's n alone can only produce a sustained one.

Run interactively in VS Code with the Jupyter extension using the '# %%' cell markers,
or run top-to-bottom as a normal script.
"""

# %%
import os
import numpy as np
import matplotlib.pyplot as plt
from utils import (
    get_figures_dir, alpha_n, beta_n, alpha_m, beta_m, alpha_h, beta_h,
    steady_state_and_tau, V_REST_ABSOLUTE,
)

FIGURES_DIR = get_figures_dir(__file__)

# %%
# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
# V here is depolarization-from-rest (mV), matching the alpha/beta
# functions' classic HH-1952 form - V=0 is rest, V=100 is a strongly
# depolarized clamp. Sweep comfortably past the range any of the three
# gates actually move across.
V_SWEEP = np.linspace(-40, 120, 400)


# %%
# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_steady_states(V, n_inf, m_inf, h_inf, save_path=None):
    """
    All three x_inf curves overlaid - the "n and m rise, h falls" picture
    in one plot. This is the figure that makes h's opposite-direction
    voltage dependence impossible to miss.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(V, n_inf, color="tab:blue", linewidth=2, label="n_inf (K+ activation)")
    plt.plot(V, m_inf, color="tab:green", linewidth=2, label="m_inf (Na+ activation)")
    plt.plot(V, h_inf, color="tab:red", linewidth=2, label="h_inf (Na+ inactivation)")
    plt.axhline(0.5, color="black", linewidth=0.5, linestyle="--")
    plt.xlabel("Depolarization from rest, V (mV)")
    plt.ylabel("Steady-state gating value (x_inf)")
    plt.title("Steady-State Gating Variables vs. Voltage")
    plt.legend()
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_time_constants(V, tau_n, tau_m, tau_h, save_path=None):
    """
    tau_x for all three gates overlaid - shows the key quantitative fact
    used throughout Lecture 5: tau_m is much smaller than tau_n or tau_h
    across the whole range, which is *why* sodium activates fast while
    both potassium activation and sodium inactivation are comparatively slow.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(V, tau_n, color="tab:blue", linewidth=2, label="tau_n (K+ activation)")
    plt.plot(V, tau_m, color="tab:green", linewidth=2, label="tau_m (Na+ activation)")
    plt.plot(V, tau_h, color="tab:red", linewidth=2, label="tau_h (Na+ inactivation)")
    plt.xlabel("Depolarization from rest, V (mV)")
    plt.ylabel("Time constant tau_x (ms)")
    plt.title("Gating Time Constants vs. Voltage - m Is Much Faster Than n or h")
    plt.legend()
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150)
    plt.show()


# %%
# ---------------------------------------------------------------------------
# Demo: compute and check all three gates' kinetics across the sweep
# ---------------------------------------------------------------------------
def run_gating_kinetics_overview():
    """Computes x_inf and tau_x for n, m, h across V_SWEEP, checks the qualitative claims, plots both."""
    n_inf, tau_n = steady_state_and_tau(alpha_n(V_SWEEP), beta_n(V_SWEEP))
    m_inf, tau_m = steady_state_and_tau(alpha_m(V_SWEEP), beta_m(V_SWEEP))
    h_inf, tau_h = steady_state_and_tau(alpha_h(V_SWEEP), beta_h(V_SWEEP))

    print(f"[gating overview] V range: {V_SWEEP.min():.0f} to {V_SWEEP.max():.0f} mV "
          f"(depolarization from rest; rest itself is V=0, "
          f"absolute rest ~= {V_REST_ABSOLUTE:.1f} mV)")

    # sanity check: n_inf and m_inf should rise (low -> high) with V
    print(f"  n_inf: {n_inf[0]:.4f} -> {n_inf[-1]:.4f} (should rise)")
    print(f"  m_inf: {m_inf[0]:.4f} -> {m_inf[-1]:.4f} (should rise, and faster than n_inf)")
    # sanity check: h_inf should FALL (high -> low) with V - the mirror-image behavior
    print(f"  h_inf: {h_inf[0]:.4f} -> {h_inf[-1]:.4f} (should FALL - opposite of n_inf/m_inf)")

    # sanity check: tau_m should be smaller than tau_n and tau_h across most of the range
    mean_tau_m = np.mean(tau_m)
    mean_tau_n = np.mean(tau_n)
    mean_tau_h = np.mean(tau_h)
    print(f"  Mean tau_m = {mean_tau_m:.4f} ms, mean tau_n = {mean_tau_n:.4f} ms, "
          f"mean tau_h = {mean_tau_h:.4f} ms")
    print(f"  tau_m < tau_n: {mean_tau_m < mean_tau_n}, "
          f"tau_m < tau_h: {mean_tau_m < mean_tau_h} (both should be True - "
          f"this is why sodium activation is the fast gate)")

    plot_steady_states(
        V_SWEEP, n_inf, m_inf, h_inf,
        save_path=os.path.join(FIGURES_DIR, "gating_variables_vs_voltage.png"),
    )
    plot_time_constants(
        V_SWEEP, tau_n, tau_m, tau_h,
        save_path=os.path.join(FIGURES_DIR, "gating_time_constants_vs_voltage.png"),
    )


# %%
if __name__ == "__main__":
    run_gating_kinetics_overview()

# %%
