"""
integrate_and_fire.py

MIT OCW 9.40 - Lecture 3: Ion-Specific Conductances and IF Models (Part 2 - IF neuron)

Integrate-and-fire (IF) spiking model, in two flavors:
    1. No-leak IF: dV/dt = I_e / C, hard threshold + reset
    2. Leaky IF:   tau*dV/dt = -(V - V_inf), V_inf = E_leak + R_L*I_e, hard threshold + reset

Key results demonstrated:
    - No-leak: regular periodic firing for ANY I_e > 0, firing rate scales
      linearly with I_e (f = I_e / (C * delta_V))
    - Leaky: firing requires I_e above rheobase (V_inf must exceed
      V_threshold), otherwise V just settles at V_inf and never spikes

Run interactively in VS Code with the Jupyter extension using the '# %%' cell markers,
or run top-to-bottom as a normal script.
"""

# %%
import os
import numpy as np
import matplotlib.pyplot as plt
from utils import (
    get_figures_dir, step_current, tau_from_RC, V_inf_from_current,
    rheobase_current, simulate_lif,
)

FIGURES_DIR = get_figures_dir(__file__)

# %%
# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
# Same R_L, C as conductance_battery.py (tau = R_L * C = 1.0 s exactly).
R_L = 2.0
C = 0.5

E_LEAK = -75.0        # leak reversal potential (mV)
V_THRESHOLD = -55.0   # spike threshold (mV)
V_RESET = -70.0       # post-spike reset voltage (mV)

T_MAX = 5.0
N_STEPS = 5000        # fine relative to tau=1.0s, so simulate_lif's
                       # sample-resolution spike detection doesn't blur timing
TIME = np.linspace(0, T_MAX, N_STEPS)

# no-leak demo: no rheobase concept here, any I_e > 0 eventually spikes
NO_LEAK_STEP_T_ON = 0.5
NO_LEAK_STEP_AMPLITUDE = 20.0

# leaky demo, above rheobase: G_leak*(V_threshold - E_leak) = 0.5*20 = 10.0,
# so 20.0 is comfortably above it
LEAKY_STEP_T_ON = 0.5
LEAKY_STEP_AMPLITUDE = 20.0

# leaky demo, below rheobase: V_inf = E_leak + R_L*I_e = -75 + 2*5 = -65 mV,
# which is below V_threshold=-55mV -> should never spike
BELOW_RHEOBASE_STEP_T_ON = 0.5
BELOW_RHEOBASE_STEP_AMPLITUDE = 5.0


# %%
# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_if_response(t, I_e, V, spike_times, V_threshold, V_reset, E_leak=None,
                      title="Integrate-and-fire", save_path=None):
    """
    Two stacked subplots (current, voltage). V_threshold and V_reset are
    marked as reference lines; each spike is marked with a scatter dot at
    (spike_time, V_threshold), since simulate_lif's V trace only shows the
    reset value at the spike sample, not the spike itself.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    ax1.plot(t, I_e, color="tab:orange")
    ax1.set_ylabel("Injected current I_e")
    ax1.set_title(title)
    ax1.axhline(0, color="black", linewidth=0.5, linestyle="--")

    ax2.plot(t, V, color="tab:blue", label="V(t)", zorder=1)
    ax2.axhline(V_threshold, color="tab:red", linestyle=":", linewidth=1,
                label=f"V_threshold = {V_threshold:.1f} mV")
    ax2.axhline(V_reset, color="tab:purple", linestyle=":", linewidth=1,
                label=f"V_reset = {V_reset:.1f} mV")
    if E_leak is not None:
        ax2.axhline(E_leak, color="tab:gray", linestyle=":", linewidth=1,
                    label=f"E_leak = {E_leak:.1f} mV")
    if len(spike_times) > 0:
        ax2.scatter(spike_times, [V_threshold] * len(spike_times),
                    color="black", marker="v", zorder=2, label="Spikes")

    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Membrane voltage V (mV)")
    ax2.legend(loc="lower right", fontsize=8)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_interspike_intervals(spike_times, title="Inter-spike intervals", save_path=None):
    """Sequence of intervals between consecutive spikes - should flatten out to a constant under constant I_e."""
    if len(spike_times) < 2:
        print("  [plot_interspike_intervals] fewer than 2 spikes - nothing to plot.")
        return

    isis = np.diff(spike_times)
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(isis) + 1), isis, marker="o", color="tab:blue")
    plt.xlabel("Spike interval index")
    plt.ylabel("Inter-spike interval (s)")
    plt.title(title)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150)
    plt.show()


# %%
# ---------------------------------------------------------------------------
# Demo: no-leak IF under constant current
# ---------------------------------------------------------------------------
def run_no_leak_demo():
    """
    Constant current into a no-leak IF neuron -> V ramps linearly, hits
    V_threshold, resets to V_reset, repeats. Checks empirical firing rate
    against the analytic f = I_e / (C * delta_V).
    """
    I_e_func = lambda t: step_current(np.array([t]), NO_LEAK_STEP_T_ON, NO_LEAK_STEP_AMPLITUDE)[0]

    V_sim, spike_times = simulate_lif(
        TIME, I_e_func, C, V_reset=V_RESET, V_threshold=V_THRESHOLD, R_L=None
    )

    delta_V = V_THRESHOLD - V_RESET
    analytic_f = NO_LEAK_STEP_AMPLITUDE / (C * delta_V)

    print(f"[no-leak demo] {len(spike_times)} spikes over {T_MAX:.1f}s")
    if len(spike_times) >= 2:
        isis = np.diff(spike_times)
        # drop the first ISI - it includes the initial ramp from V0=V_reset
        # under the step's t_on delay, not a steady-state interval
        steady_isis = isis[1:] if len(isis) > 1 else isis
        empirical_f = 1.0 / np.mean(steady_isis)
        print(f"  Mean steady-state ISI = {np.mean(steady_isis):.4f} s "
              f"(std = {np.std(steady_isis):.6f}, should be ~0 for regular firing)")
        print(f"  Empirical firing rate = {empirical_f:.4f} Hz, "
              f"analytic f = I_e/(C*delta_V) = {analytic_f:.4f} Hz")
    else:
        print("  Not enough spikes to compute firing rate - check step amplitude.")

    I_e_trace = step_current(TIME, NO_LEAK_STEP_T_ON, NO_LEAK_STEP_AMPLITUDE)
    plot_if_response(
        TIME, I_e_trace, V_sim, spike_times, V_THRESHOLD, V_RESET,
        title="No-Leak IF: Regular Periodic Firing Under Constant Current",
        save_path=os.path.join(FIGURES_DIR, "no_leak_if_response.png"),
    )
    plot_interspike_intervals(
        spike_times, title="No-Leak IF: Inter-Spike Intervals",
        save_path=os.path.join(FIGURES_DIR, "no_leak_isi.png"),
    )


# %%
# ---------------------------------------------------------------------------
# Demo: leaky IF under constant current (above rheobase)
# ---------------------------------------------------------------------------
def run_leaky_demo():
    """Leaky IF neuron with current above rheobase -> fires regularly, but not linearly-from-zero like the no-leak case."""
    G_leak = 1.0 / R_L
    I_threshold = rheobase_current(G_leak, V_THRESHOLD, E_LEAK)
    print(f"[leaky demo] rheobase I_threshold = {I_threshold:.4f} "
          f"(injecting {LEAKY_STEP_AMPLITUDE:.4f}, which is "
          f"{'above' if LEAKY_STEP_AMPLITUDE > I_threshold else 'NOT above'} rheobase)")

    I_e_func = lambda t: step_current(np.array([t]), LEAKY_STEP_T_ON, LEAKY_STEP_AMPLITUDE)[0]

    V_sim, spike_times = simulate_lif(
        TIME, I_e_func, C, V_reset=V_RESET, V_threshold=V_THRESHOLD,
        R_L=R_L, E_leak=E_LEAK
    )

    print(f"  {len(spike_times)} spikes over {T_MAX:.1f}s")
    if len(spike_times) >= 2:
        isis = np.diff(spike_times)
        steady_isis = isis[1:] if len(isis) > 1 else isis
        empirical_f = 1.0 / np.mean(steady_isis)
        print(f"  Mean steady-state ISI = {np.mean(steady_isis):.4f} s "
              f"(std = {np.std(steady_isis):.6f})")
        print(f"  Empirical firing rate = {empirical_f:.4f} Hz")

    I_e_trace = step_current(TIME, LEAKY_STEP_T_ON, LEAKY_STEP_AMPLITUDE)
    plot_if_response(
        TIME, I_e_trace, V_sim, spike_times, V_THRESHOLD, V_RESET, E_leak=E_LEAK,
        title="Leaky IF: Firing Above Rheobase",
        save_path=os.path.join(FIGURES_DIR, "leaky_if_response.png"),
    )
    plot_interspike_intervals(
        spike_times, title="Leaky IF: Inter-Spike Intervals",
        save_path=os.path.join(FIGURES_DIR, "leaky_isi.png"),
    )


# %%
# ---------------------------------------------------------------------------
# Demo: leaky IF under constant current (below rheobase) - should NOT spike
# ---------------------------------------------------------------------------
def run_below_rheobase_demo():
    """
    Current whose V_inf sits below V_threshold -> the neuron should never
    spike, no matter how long the simulation runs. This is the qualitatively
    new leaky-IF behavior the lecture introduces.
    """
    V_inf = V_inf_from_current(R_L, BELOW_RHEOBASE_STEP_AMPLITUDE, E=E_LEAK)
    print(f"[below-rheobase demo] V_inf = {V_inf:.4f} mV "
          f"(V_threshold = {V_THRESHOLD:.1f} mV, "
          f"{'below - should NOT spike' if V_inf < V_THRESHOLD else 'WARNING: not below threshold'})")

    I_e_func = lambda t: step_current(np.array([t]), BELOW_RHEOBASE_STEP_T_ON, BELOW_RHEOBASE_STEP_AMPLITUDE)[0]

    V_sim, spike_times = simulate_lif(
        TIME, I_e_func, C, V_reset=V_RESET, V_threshold=V_THRESHOLD,
        R_L=R_L, E_leak=E_LEAK
    )

    print(f"  Number of spikes = {len(spike_times)} "
          f"({'PASS' if len(spike_times) == 0 else 'FAIL'} - expected 0)")
    print(f"  Final V = {V_sim[-1]:.4f} mV (should be close to V_inf = {V_inf:.4f} mV)")

    I_e_trace = step_current(TIME, BELOW_RHEOBASE_STEP_T_ON, BELOW_RHEOBASE_STEP_AMPLITUDE)
    plot_if_response(
        TIME, I_e_trace, V_sim, spike_times, V_THRESHOLD, V_RESET, E_leak=E_LEAK,
        title="Leaky IF: Below Rheobase - V Settles at V_inf, Never Spikes",
        save_path=os.path.join(FIGURES_DIR, "below_rheobase_if_response.png"),
    )


# %%
if __name__ == "__main__":
    run_no_leak_demo()
    run_leaky_demo()
    run_below_rheobase_demo()

# %%