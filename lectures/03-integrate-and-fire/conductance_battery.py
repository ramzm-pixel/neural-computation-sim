"""
conductance_battery.py

MIT OCW 9.40 - Lecture 3: Ion-Specific Conductances and IF Models (Part 1 - the battery)

Models an ion-selective conductance as a resistor IN SERIES WITH A BATTERY,
rather than the plain leak resistor from Lecture 2:

    I_ion = G_ion * (V - E_ion)      (driving-potential form)

Summing voltage drops across the battery and resistor in series gives the
same relation: V_m = E_ion + I_ion/G_ion => I_ion = G_ion*(V_m - E_ion).

This changes Lecture 2's governing RC equation from:
    tau * dV/dt = -(V - V_inf),   V_inf = R_L * I_e
to:
    tau * dV/dt = -(V - V_inf),   V_inf = E_leak + R_L * I_e

i.e. V_inf is now offset by the battery voltage - this is what finally
fixes the "dead neuron" problem from Lecture 2's capacitor_model.py: with
zero injected current, V no longer sits wherever it happened to stop, it
relaxes to a genuine resting potential E_leak.

Run interactively in VS Code with the Jupyter extension using the '# %%' cell markers,
or run top-to-bottom as a normal script.
"""

# %%
import os
import numpy as np
import matplotlib.pyplot as plt
from utils import (
    get_figures_dir, step_current, pulse_current, tau_from_RC,
    V_inf_from_current, simulate_rc_response, driving_potential, ionic_current,
)

FIGURES_DIR = get_figures_dir(__file__)

# %%
# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
# Same R_L, C as Lecture 2 (tau = R_L * C = 1.0 s exactly, for convenient
# demo plots - real neurons have tau ~10-100 ms).
R_L = 2.0        # leak resistance (Ohm)
C = 0.5          # membrane capacitance (F)  ->  tau = 1.0 s

E_LEAK = -75.0   # leak reversal potential (mV) - using the K+-like leak
                 # battery from the lecture, in mV rather than Lecture 2's
                 # arbitrary volt-scale demo units, since the whole point
                 # here is comparing against a real resting potential

T_MAX = 6.0
N_STEPS = 2000
TIME = np.linspace(0, T_MAX, N_STEPS)

# step demo: injected current pushes V_inf above E_leak
STEP_T_ON = 1.0
STEP_AMPLITUDE = 20.0   # -> V_inf = E_leak + R_L*I_0 = -75 + 40 = -35 mV

# I-V curve demo: sweep clamped voltage, look at ionic current through G_leak
G_LEAK = 1.0 / R_L
V_SWEEP = np.linspace(-120, 20, 200)  # mV, spans well past E_leak both directions


# %%
# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_iv_curve(V, I_ion, E_ion, G_ion, save_path=None):
    """
    I-V curve for an ion conductance: straight line through E_ion (not
    through the origin) - the "reversal potential" is exactly where the
    line crosses zero current.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(V, I_ion, color="tab:blue", linewidth=2)
    plt.axhline(0, color="black", linewidth=0.5, linestyle="--")
    plt.axvline(E_ion, color="tab:red", linestyle=":", linewidth=1.5,
                label=f"E_ion = {E_ion:.1f} mV (reversal potential)")
    plt.xlabel("Clamped membrane voltage V (mV)")
    plt.ylabel("Ionic current I_ion (arbitrary units)")
    plt.title(f"I-V Curve for a Battery-Backed Conductance (G={G_ion:.2f})")
    plt.legend()
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_rc_battery_response(t, I_e, V, V_inf=None, E_leak=None, title="RC model with battery", save_path=None):
    """Same layout as Lecture 2's plot_rc_response, with E_leak marked as the true resting line."""
    _, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    ax1.plot(t, I_e, color="tab:orange")
    ax1.set_ylabel("Injected current I_e")
    ax1.set_title(title)
    ax1.axhline(0, color="black", linewidth=0.5, linestyle="--")

    ax2.plot(t, V, color="tab:blue", label="V(t)")
    if V_inf is not None:
        ax2.axhline(V_inf, color="tab:green", linewidth=1, linestyle=":",
                    label=f"V_inf = {V_inf:.2f} mV")
    if E_leak is not None:
        ax2.axhline(E_leak, color="tab:red", linewidth=1, linestyle=":",
                    label=f"E_leak = {E_leak:.2f} mV (true resting potential)")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Membrane voltage V (mV)")
    ax2.legend()

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150)
    plt.show()


# %%
# ---------------------------------------------------------------------------
# Demo: I-V curve for the battery-backed leak conductance
# ---------------------------------------------------------------------------
def run_iv_curve_demo():
    """
    Sweeps clamped voltage across E_leak, plots I_leak = G_leak*(V-E_leak).
    Confirms the line crosses zero exactly at E_leak (the defining property
    of the reversal potential) rather than at V=0 like Lecture 2's plain
    resistor.
    """
    I_leak = ionic_current(V_SWEEP, G_LEAK, E_LEAK)

    # sanity check: current should be ~0 right at E_leak
    idx_near_E = np.argmin(np.abs(V_SWEEP - E_LEAK))
    print(f"[I-V curve demo] G_leak = {G_LEAK:.4f}, E_leak = {E_LEAK:.1f} mV")
    print(f"  I_leak at V closest to E_leak ({V_SWEEP[idx_near_E]:.2f} mV) = "
          f"{I_leak[idx_near_E]:.6f} (should be ~0)")
    print(f"  I_leak at V=0 mV = {ionic_current(0.0, G_LEAK, E_LEAK):.4f} "
          f"(nonzero now, unlike Lecture 2's plain-resistor I-V curve)")

    plot_iv_curve(
        V_SWEEP, I_leak, E_LEAK, G_LEAK,
        save_path=os.path.join(FIGURES_DIR, "iv_curve_battery_conductance.png"),
    )


# %%
# ---------------------------------------------------------------------------
# Demo: step current -> V relaxes toward a battery-shifted V_inf
# ---------------------------------------------------------------------------
def run_battery_step_demo():
    """
    Same step-current setup as Lecture 2's rc_model.py, but now V_inf and
    the resting state are both offset by E_leak instead of starting at 0 -
    this is the direct fix for the "dead neuron" behavior: with I_e=0, V
    now relaxes to a genuine resting potential (E_leak) instead of
    freezing wherever it last was.
    """
    tau = tau_from_RC(R_L, C)
    V_inf = V_inf_from_current(R_L, STEP_AMPLITUDE, E=E_LEAK)

    I_e_func = lambda t: step_current(np.array([t]), STEP_T_ON, STEP_AMPLITUDE)[0]
    V_sim = simulate_rc_response(TIME, I_e_func, R_L, C, V0=E_LEAK, E=E_LEAK,
                                  discontinuities=[STEP_T_ON])

    print(f"[battery step demo] tau = {tau:.4f} s, E_leak = {E_LEAK:.1f} mV, "
          f"V_inf = {V_inf:.4f} mV")
    print(f"  V at t=0 (before current turns on) = {V_sim[0]:.4f} mV "
          f"(should equal E_leak = {E_LEAK:.1f} mV)")
    print(f"  V at t={T_MAX:.1f}s (long after step) = {V_sim[-1]:.4f} mV "
          f"(should be close to V_inf = {V_inf:.4f} mV)")

    I_e_trace = step_current(TIME, STEP_T_ON, STEP_AMPLITUDE)
    plot_rc_battery_response(
        TIME, I_e_trace, V_sim, V_inf=V_inf, E_leak=E_LEAK,
        title="RC Model with Battery: Step Current -> Relaxation to Battery-Shifted V_inf",
        save_path=os.path.join(FIGURES_DIR, "battery_step_response.png"),
    )


# %%
# ---------------------------------------------------------------------------
# Demo: pulse current -> V returns to E_leak, not to 0
# ---------------------------------------------------------------------------
def run_battery_pulse_demo():
    """
    Pulse current -> V rises toward V_inf during the pulse, then relaxes
    back down to E_leak (not 0) once the pulse ends - this is the direct
    side-by-side contrast with Lecture 2's plain RC model, where the same
    pulse would relax back to 0 since there was no battery.
    """
    pulse_t_on, pulse_t_off = 1.0, 3.5
    V_inf_on = V_inf_from_current(R_L, STEP_AMPLITUDE, E=E_LEAK)

    I_e_func = lambda t: pulse_current(np.array([t]), pulse_t_on, pulse_t_off, STEP_AMPLITUDE)[0]
    V_sim = simulate_rc_response(TIME, I_e_func, R_L, C, V0=E_LEAK, E=E_LEAK,
                                  discontinuities=[pulse_t_on, pulse_t_off])

    print(f"[battery pulse demo] V_inf during pulse = {V_inf_on:.4f} mV")
    print(f"  V well after pulse ends ({TIME[-1]:.1f}s) = {V_sim[-1]:.4f} mV "
          f"(relaxes back to E_leak = {E_LEAK:.1f} mV, not to 0)")

    I_e_trace = pulse_current(TIME, pulse_t_on, pulse_t_off, STEP_AMPLITUDE)
    plot_rc_battery_response(
        TIME, I_e_trace, V_sim, V_inf=V_inf_on, E_leak=E_LEAK,
        title="RC Model with Battery: Pulse Current -> Returns to E_leak, Not 0",
        save_path=os.path.join(FIGURES_DIR, "battery_pulse_response.png"),
    )


# %%
if __name__ == "__main__":
    run_iv_curve_demo()
    run_battery_step_demo()
    run_battery_pulse_demo()

# %%
