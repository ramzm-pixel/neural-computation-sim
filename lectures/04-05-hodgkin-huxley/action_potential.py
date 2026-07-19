"""
action_potential.py

MIT OCW 9.40 - Lecture 5: Hodgkin-Huxley Model (Part 3 - the full spike)

The centerpiece file: integrates all four coupled state variables (V, n, m,
h) together via utils.py's simulate_hh_neuron, and confirms a
brief current injection alone - with no separate "spike-generating rule" -
produces a full action potential purely from the feedback loop described
in Lecture 5 Section 5:

    depolarize -> m rises fast -> I_Na in -> depolarizes further (runaway)
    -> V approaches E_Na -> n (slower) has been rising the whole time,
    G_K turns on and pulls V back down -> h (also slower) has been
    falling, sodium inactivates and shuts off I_Na -> K+ efflux finishes
    repolarizing the cell

Also separately confirms the "no leak = no spike" contrast implied by the
whole Lecture 3-5 arc: without functioning sodium (G_Na forced to 0),
the same current injection produces only a passive RC-like bump, never
a spike - underscoring that the spike is genuinely an emergent property
of the sodium/potassium feedback, not just "current in, big voltage out."

Run interactively in VS Code with the Jupyter extension using the '# %%' cell markers,
or run top-to-bottom as a normal script.
"""

# %%
import os
import numpy as np
import matplotlib.pyplot as plt
from utils import (
    get_figures_dir, simulate_hh_neuron, detect_spikes,
    G_K_BAR, G_NA_BAR, G_LEAK, E_K, E_NA, E_LEAK, C_M, V_REST_ABSOLUTE,
)

FIGURES_DIR = get_figures_dir(__file__)

# %%
# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
T_MAX = 30.0       # ms
N_STEPS = 6000
TIME = np.linspace(0, T_MAX, N_STEPS)

PULSE_T_ON = 5.0
PULSE_T_OFF = 6.0    # brief 1 ms pulse is enough to trigger a full spike
PULSE_AMPLITUDE = 20.0  # uA/cm^2, comfortably suprathreshold for these classic HH parameters

SPIKE_THRESHOLD = 50.0  # mV, depolarization-from-rest, for detect_spikes


# %%
# ---------------------------------------------------------------------------
# Current waveform
# ---------------------------------------------------------------------------
def pulse_current_func(t_on, t_off, amplitude):
    """Returns a callable I_e(t): amplitude between t_on and t_off, else 0 - matches simulate_hh_neuron's I_e_func signature."""
    def I_e(t):
        return amplitude if t_on <= t < t_off else 0.0
    return I_e


# %%
# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_full_action_potential(t, V, n, m, h, t_on, t_off, save_path=None):
    """
    Two stacked subplots: V(t) (the actual spike), and all three gating
    variables together underneath - the picture that ties the whole
    lecture together: watch m spike up first, n climb slower, h fall
    with a lag, all driving the single V trace above.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    ax1.plot(t, V, color="tab:blue", linewidth=2)
    ax1.axvspan(t_on, t_off, color="tab:orange", alpha=0.2, label="Current pulse")
    ax1.set_ylabel("V (mV, depolarization from rest)")
    ax1.set_title("Full Hodgkin-Huxley Action Potential")
    ax1.legend(loc="upper right")

    ax2.plot(t, n, color="tab:blue", linewidth=1.5, label="n (K+ activation)")
    ax2.plot(t, m, color="tab:green", linewidth=1.5, label="m (Na+ activation)")
    ax2.plot(t, h, color="tab:red", linewidth=1.5, label="h (Na+ inactivation)")
    ax2.axvspan(t_on, t_off, color="tab:orange", alpha=0.2)
    ax2.set_xlabel("Time (ms)")
    ax2.set_ylabel("Gating variable value")
    ax2.legend(loc="upper right")

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_currents(t, V, n, m, h, save_path=None):
    """
    Individual I_K, I_Na, I_leak traces overlaid - makes the "fast sodium
    in, then slower potassium out" sequence from the original voltage-clamp
    data (Lectures 4-5) directly visible in the free-running spike, not
    just in the clamped experiments that originally motivated the model.
    """
    I_K = G_K_BAR * (n ** 4) * (V - E_K)
    I_Na = G_NA_BAR * (m ** 3) * h * (V - E_NA)
    I_L = G_LEAK * (V - E_LEAK)

    plt.figure(figsize=(9, 5))
    plt.plot(t, I_Na, color="tab:green", linewidth=1.5, label="I_Na")
    plt.plot(t, I_K, color="tab:blue", linewidth=1.5, label="I_K")
    plt.plot(t, I_L, color="tab:gray", linewidth=1, label="I_leak")
    plt.axhline(0, color="black", linewidth=0.5, linestyle="--")
    plt.xlabel("Time (ms)")
    plt.ylabel("Current (uA/cm^2)")
    plt.title("Individual Ionic Currents During the Spike")
    plt.legend()
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_spike_vs_no_sodium(t, V_full, V_no_sodium, t_on, t_off, save_path=None):
    """Side-by-side: full HH model spikes, sodium-disabled model only shows a passive RC-like bump."""
    plt.figure(figsize=(9, 5))
    plt.plot(t, V_full, color="tab:blue", linewidth=2, label="Full HH model (spikes)")
    plt.plot(t, V_no_sodium, color="tab:gray", linewidth=2, linestyle="--",
              label="G_Na forced to 0 (passive bump only)")
    plt.axvspan(t_on, t_off, color="tab:orange", alpha=0.2, label="Current pulse")
    plt.xlabel("Time (ms)")
    plt.ylabel("V (mV, depolarization from rest)")
    plt.title("The Spike Is an Emergent Property of the Na+/K+ Feedback Loop")
    plt.legend()
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150)
    plt.show()


# %%
# ---------------------------------------------------------------------------
# Demo: full action potential from a brief current pulse
# ---------------------------------------------------------------------------
def run_action_potential_demo():
    """Integrates the full coupled system, confirms a spike is produced, plots V and all three gates."""
    I_e_func = pulse_current_func(PULSE_T_ON, PULSE_T_OFF, PULSE_AMPLITUDE)

    V, n, m, h = simulate_hh_neuron(TIME, I_e_func)

    spike_times = detect_spikes(V, TIME, threshold=SPIKE_THRESHOLD)
    print(f"[action potential demo] {len(spike_times)} spike(s) detected "
          f"(threshold={SPIKE_THRESHOLD:.0f} mV dep-from-rest)")
    if spike_times:
        print(f"  First spike at t={spike_times[0]:.4f} ms "
              f"(pulse was {PULSE_T_ON:.1f}-{PULSE_T_OFF:.1f} ms)")

    print(f"  Peak V = {V.max():.4f} mV (dep-from-rest), "
          f"absolute peak ~= {V.max() + V_REST_ABSOLUTE:.4f} mV")
    print(f"  Peak m = {m.max():.4f} (should approach ~1 - nearly full sodium activation)")
    print(f"  Min h after spike = {h.min():.4f} (should drop well below its resting value - inactivation)")
    print(f"  Peak n = {n.max():.4f} (reached AFTER the m/h peak - slower potassium gate lagging behind)")

    # sanity check on ordering: m should peak before n (fast activation vs
    # slow activation), and h's minimum should occur close to n's peak
    # (inactivation finishing around when repolarization is underway)
    m_peak_t = TIME[np.argmax(m)]
    n_peak_t = TIME[np.argmax(n)]
    print(f"  m peaks at t={m_peak_t:.4f} ms, n peaks at t={n_peak_t:.4f} ms "
          f"(m should peak first: {m_peak_t < n_peak_t})")

    plot_full_action_potential(
        TIME, V, n, m, h, PULSE_T_ON, PULSE_T_OFF,
        save_path=os.path.join(FIGURES_DIR, "action_potential_full.png"),
    )
    plot_currents(
        TIME, V, n, m, h,
        save_path=os.path.join(FIGURES_DIR, "action_potential_currents.png"),
    )

    return V, n, m, h


# %%
# ---------------------------------------------------------------------------
# Demo: disable sodium entirely - the spike disappears
# ---------------------------------------------------------------------------
def run_no_sodium_control(V_full):
    """
    Re-runs the identical current pulse with G_Na forced to 0, to show the
    spike is not just "current in => big voltage swing out" - without a
    functioning sodium conductance, the same stimulus produces only a
    small passive RC-like bump (governed by C and G_leak alone), confirming
    the spike genuinely depends on the sodium/potassium feedback loop.
    """
    I_e_func = pulse_current_func(PULSE_T_ON, PULSE_T_OFF, PULSE_AMPLITUDE)

    V_no_sodium, n_ns, m_ns, h_ns = simulate_hh_neuron(TIME, I_e_func, G_Na=0.0)

    print(f"\n[no-sodium control] peak V with G_Na=0: {V_no_sodium.max():.4f} mV "
          f"(dep-from-rest), vs. full-model peak: {V_full.max():.4f} mV")
    print(f"  Spike suppressed: {V_no_sodium.max() < 0.3 * V_full.max()} "
          f"(should be True - a huge reduction, not just a smaller spike)")

    plot_spike_vs_no_sodium(
        TIME, V_full, V_no_sodium, PULSE_T_ON, PULSE_T_OFF,
        save_path=os.path.join(FIGURES_DIR, "spike_vs_no_sodium_control.png"),
    )


# %%
if __name__ == "__main__":
    V_full, n_full, m_full, h_full = run_action_potential_demo()
    run_no_sodium_control(V_full)

# %%
