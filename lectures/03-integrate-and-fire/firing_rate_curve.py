"""
firing_rate_curve.py

MIT OCW 9.40 - Lecture 3: Ion-Specific Conductances and IF Models (Part 3 - f-I curve)

Sweeps injected current I_e across a range spanning below and above
rheobase, and compares three things on one plot:

    1. Exact leaky-IF firing rate:
         delta_t = -tau * ln((V_inf - V_threshold) / (V_inf - V_reset))
         f = 1 / delta_t
    2. Large-I_e linear approximation:
         f = (I_e - I_threshold) / (C * delta_V),  I_e > I_threshold
    3. Empirical firing rate, measured by actually running simulate_lif
       (from integrate_and_fire.py's machinery) at several current values
       and computing 1/mean(ISI) - this is the real sanity check: the
       "exact" formula above is itself only exact in the sense that it
       correctly solves the ODE for delta_t, but it's still worth
       confirming it matches an actual simulated spike train rather than
       just trusting the algebra end-to-end.

Expected shape: firing rate is 0 below rheobase, then rises - the exact
curve jumps up right at threshold and curves toward the linear
approximation as I_e grows large, with the two curves converging (not
touching) at high current.

Run interactively in VS Code with the Jupyter extension using the '# %%' cell markers,
or run top-to-bottom as a normal script.
"""

# %%
import os
import numpy as np
import matplotlib.pyplot as plt
from utils import (
    get_figures_dir, step_current, rheobase_current, simulate_lif,
    firing_rate_leaky_exact, firing_rate_leaky_linear_approx, firing_rate_no_leak,
)

FIGURES_DIR = get_figures_dir(__file__)

# %%
# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
# Same R_L, C, E_leak, V_threshold, V_reset as integrate_and_fire.py, for
# direct continuity between the two scripts' numbers.
R_L = 2.0
C = 0.5
G_LEAK = 1.0 / R_L

E_LEAK = -75.0
V_THRESHOLD = -55.0
V_RESET = -70.0

# rheobase = G_leak*(V_threshold-E_leak) = 0.5*20 = 10.0 - sweep needs to
# comfortably span both sides of this
I_MIN, I_MAX = 0.0, 60.0
N_CURRENTS = 300
I_SWEEP = np.linspace(I_MIN, I_MAX, N_CURRENTS)

# empirical validation: only run full simulate_lif at a handful of currents
# (it's much more expensive than the closed-form formulas), spanning below,
# near, and well above rheobase
EMPIRICAL_CURRENTS = np.array([5.0, 10.0, 12.0, 15.0, 20.0, 30.0, 45.0, 60.0])
EMPIRICAL_T_MAX = 8.0       # long enough to collect several ISIs even for
                             # currents just above rheobase (slow firing)
EMPIRICAL_N_STEPS = 8000
EMPIRICAL_STEP_T_ON = 0.5


# %%
# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_fi_curve(I_sweep, f_exact, f_linear, I_threshold,
                   empirical_I=None, empirical_f=None, save_path=None):
    """
    f-I curve: exact (solid) vs. linear approximation (dashed), rheobase
    marked as a vertical line, and empirical simulate_lif measurements
    overlaid as scatter points if provided.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(I_sweep, f_exact, color="tab:blue", linewidth=2, label="Exact (log formula)")
    plt.plot(I_sweep, f_linear, color="black", linestyle="--", linewidth=1.5,
              label="Linear approximation (large I_e)")
    plt.axvline(I_threshold, color="tab:red", linestyle=":", linewidth=1,
                label=f"Rheobase I_threshold = {I_threshold:.2f}")

    if empirical_I is not None and empirical_f is not None:
        plt.scatter(empirical_I, empirical_f, color="tab:orange", zorder=3,
                    label="Empirical (simulate_lif)", marker="o", s=40)

    plt.xlabel("Injected current I_e")
    plt.ylabel("Firing rate f (Hz)")
    plt.title("Leaky IF f-I Curve: Exact vs. Linear Approximation vs. Simulated")
    plt.legend()
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_fi_curve_with_no_leak(I_sweep, f_exact, f_no_leak, save_path=None):
    """
    Side-by-side comparison: leaky IF (0 below rheobase, then rising) vs.
    no-leak IF (linear through the origin) - makes visually obvious that
    adding a leak is what introduces the rheobase / minimum-current
    requirement in the first place.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(I_sweep, f_exact, color="tab:blue", linewidth=2, label="Leaky IF (exact)")
    plt.plot(I_sweep, f_no_leak, color="tab:green", linewidth=2, linestyle="--",
              label="No-leak IF (for comparison)")
    plt.xlabel("Injected current I_e")
    plt.ylabel("Firing rate f (Hz)")
    plt.title("Leak Introduces a Minimum-Current Requirement (Rheobase)")
    plt.legend()
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150)
    plt.show()


# %%
# ---------------------------------------------------------------------------
# Empirical firing rate via actual simulation
# ---------------------------------------------------------------------------
def measure_empirical_firing_rate(I_amplitude, t_max, n_steps, t_on):
    """
    Run simulate_lif at a single constant current amplitude, return the
    empirical firing rate as 1/mean(steady-state ISI). Drops the first ISI
    (includes the initial transient from V0=V_reset's ramp-up, not a
    steady-state interval) same as integrate_and_fire.py's demos.

    Returns np.nan if fewer than 2 spikes were produced (rate can't be
    estimated, or the neuron didn't spike at all - e.g. below rheobase).
    """
    t = np.linspace(0, t_max, n_steps)
    I_e_func = lambda tt: step_current(np.array([tt]), t_on, I_amplitude)[0]

    _, spike_times = simulate_lif(
        t, I_e_func, C, V_reset=V_RESET, V_threshold=V_THRESHOLD,
        R_L=R_L, E_leak=E_LEAK
    )

    if len(spike_times) < 3:
        # need at least 2 ISIs after dropping the first, i.e. >= 3 spikes,
        # to get a stable steady-state estimate
        return np.nan

    isis = np.diff(spike_times)
    steady_isis = isis[1:]
    return 1.0 / np.mean(steady_isis)


def run_empirical_validation():
    """Measures empirical firing rate at each current in EMPIRICAL_CURRENTS, prints a comparison table."""
    print("[empirical validation] comparing exact formula vs. simulate_lif:")
    print(f"  {'I_e':>8} {'f_exact':>10} {'f_empirical':>12} {'abs diff':>10}")

    empirical_f = np.empty_like(EMPIRICAL_CURRENTS)
    for i, I_amp in enumerate(EMPIRICAL_CURRENTS):
        f_emp = measure_empirical_firing_rate(
            I_amp, EMPIRICAL_T_MAX, EMPIRICAL_N_STEPS, EMPIRICAL_STEP_T_ON
        )
        empirical_f[i] = f_emp

        f_exact_single = firing_rate_leaky_exact(
            np.array([I_amp]), G_LEAK, C, E_LEAK, V_THRESHOLD, V_RESET
        )[0]

        diff_str = f"{abs(f_exact_single - f_emp):.4f}" if not np.isnan(f_emp) else "n/a"
        emp_str = f"{f_emp:.4f}" if not np.isnan(f_emp) else "no spikes"
        print(f"  {I_amp:>8.2f} {f_exact_single:>10.4f} {emp_str:>12} {diff_str:>10}")

    return empirical_f


# %%
# ---------------------------------------------------------------------------
# Run: full f-I sweep
# ---------------------------------------------------------------------------
def run_fi_sweep():
    """Computes exact and linear-approx firing rate across I_SWEEP, validates against simulation, plots both."""
    I_threshold = rheobase_current(G_LEAK, V_THRESHOLD, E_LEAK)
    print(f"[f-I sweep] rheobase I_threshold = {I_threshold:.4f}")

    f_exact = firing_rate_leaky_exact(I_SWEEP, G_LEAK, C, E_LEAK, V_THRESHOLD, V_RESET)
    f_linear = firing_rate_leaky_linear_approx(I_SWEEP, G_LEAK, C, E_LEAK, V_THRESHOLD, V_RESET)

    # sanity check: f_exact should be exactly 0 below rheobase
    below = I_SWEEP < I_threshold
    print(f"  f_exact below rheobase: max = {f_exact[below].max():.6f} (should be 0)")

    # sanity check: as I_e -> I_MAX (large), exact and linear should converge
    print(f"  At I_e={I_SWEEP[-1]:.1f}: f_exact={f_exact[-1]:.4f}, "
          f"f_linear={f_linear[-1]:.4f}, "
          f"relative diff={abs(f_exact[-1]-f_linear[-1])/f_linear[-1]*100:.2f}%")

    empirical_f = run_empirical_validation()

    plot_fi_curve(
        I_SWEEP, f_exact, f_linear, I_threshold,
        empirical_I=EMPIRICAL_CURRENTS, empirical_f=empirical_f,
        save_path=os.path.join(FIGURES_DIR, "fi_curve_exact_vs_linear.png"),
    )

    # no-leak comparison: use the same delta_V so the two are on equal footing
    delta_V = V_THRESHOLD - V_RESET
    f_no_leak = firing_rate_no_leak(I_SWEEP, C, delta_V)
    plot_fi_curve_with_no_leak(
        I_SWEEP, f_exact, f_no_leak,
        save_path=os.path.join(FIGURES_DIR, "fi_curve_leaky_vs_no_leak.png"),
    )


# %%
if __name__ == "__main__":
    run_fi_sweep()

# %%