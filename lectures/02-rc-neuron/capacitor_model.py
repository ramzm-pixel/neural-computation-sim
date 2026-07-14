"""
capacitor_model.py

MIT OCW 9.40 - Lecture 2: The RC Model of a Neuron (Part 1 - pure capacitor)

Pure capacitor model of a neuron (zeroth-order model, no leak/resistor yet).

Governing equation:
    I_e = C * dV/dt
    =>  V(t) = V(0) + (1/C) * integral(I_e dt')

This is the "dead neuron" model: a pure integrator with no resistor,
so V never returns to baseline on its own once current stops.

Run interactively in VS Code with the Jupyter extension using the '# %%' cell markers,
or run top-to-bottom as a normal script.
"""

# %%
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid
from utils import get_figures_dir, step_current, pulse_current

FIGURES_DIR = get_figures_dir(__file__)

# %%
# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
C = 1.0              # membrane capacitance (F) - arbitrary units for demo clarity

T_MAX = 5.0           # total simulated duration (s)
N_STEPS = 1000        # number of time samples
TIME = np.linspace(0, T_MAX, N_STEPS)

# step demo
STEP_T_ON = 1.0
STEP_AMPLITUDE = 2.0

# ramp demo
RAMP_T_ON = 1.0
RAMP_SLOPE = 1.5

# pulse demo
PULSE_T_ON = 1.0
PULSE_T_OFF = 3.0
PULSE_AMPLITUDE = 2.0


# %%
# ---------------------------------------------------------------------------
# Current waveform generators
# ---------------------------------------------------------------------------
def ramp_current(t, t_on, slope):
    """Linearly ramping current starting at t_on: I(t) = slope * (t - t_on)."""
    r_c = np.where(t > t_on, slope * (t - t_on), 0)
    return r_c


# %%
# ---------------------------------------------------------------------------
# Integration: current -> voltage
# ---------------------------------------------------------------------------
def integrate_capacitor_voltage(t, I_e, C, V0=0.0):
    """
    V(t) = V(0) + (1/C) * integral(I_e dt'), via cumulative trapezoidal
    integration (more accurate than a plain cumsum, which ignores dt spacing).
    """
    Vt = V0 + cumulative_trapezoid(I_e / C, t, initial=0.0)
    return Vt


# %%
# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_current_and_voltage(t, I_e, V, title="Capacitor model", save_path=None):
    """Plot injected current and resulting voltage in two stacked subplots sharing the time axis."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    ax1.plot(t, I_e, color="tab:orange")
    ax1.set_ylabel("Injected current I_e (A)")
    ax1.set_title(title)
    ax1.axhline(0, color="black", linewidth=0.5, linestyle="--")

    ax2.plot(t, V, color="tab:blue")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Membrane voltage V (V)")
    ax2.axhline(0, color="black", linewidth=0.5, linestyle="--")

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150)
    plt.show()


# %%
# ---------------------------------------------------------------------------
# Demo: step current -> linear voltage ramp
# ---------------------------------------------------------------------------
def run_step_demo():
    """Constant current turned on at t=STEP_T_ON -> linear voltage ramp. Checks slope = I_0 / C."""
    I_e = step_current(TIME, STEP_T_ON, STEP_AMPLITUDE)
    V = integrate_capacitor_voltage(TIME, I_e, C)

    # analytic check: slope should be STEP_AMPLITUDE / C after t_on
    post_on = TIME > STEP_T_ON
    fitted_slope = np.polyfit(TIME[post_on], V[post_on], 1)[0]
    analytic_slope = STEP_AMPLITUDE / C
    print(f"[step demo] fitted slope = {fitted_slope:.4f}, "
          f"analytic I_0/C = {analytic_slope:.4f}")

    plot_current_and_voltage(
        TIME, I_e, V,
        title="Step Current -> Linear Voltage Ramp",
        save_path=os.path.join(FIGURES_DIR, "step_current_response.png"),
    )


# %%
# ---------------------------------------------------------------------------
# Demo: ramp current -> parabolic voltage profile
# ---------------------------------------------------------------------------
def run_ramp_demo():
    """Linearly increasing current -> parabolic voltage profile (since V is the integral of a ramp)."""
    I_e = ramp_current(TIME, RAMP_T_ON, RAMP_SLOPE)
    V = integrate_capacitor_voltage(TIME, I_e, C)

    # analytic check: V(t) = slope * (t - t_on)^2 / (2*C) for t > t_on
    post_on = TIME > RAMP_T_ON
    t_shifted = TIME[post_on] - RAMP_T_ON
    V_analytic = RAMP_SLOPE * t_shifted**2 / (2 * C)
    max_error = np.max(np.abs(V[post_on] - V_analytic))
    print(f"[ramp demo] max deviation from analytic parabola: {max_error:.4f}")

    plot_current_and_voltage(
        TIME, I_e, V,
        title="Ramp Current -> Parabolic Voltage Profile",
        save_path=os.path.join(FIGURES_DIR, "ramp_current_response.png"),
    )


# %%
# ---------------------------------------------------------------------------
# Demo: pulse current -> ramp up, then flat (dead neuron limitation)
# ---------------------------------------------------------------------------
def run_pulse_demo():
    """
    Pulse current -> V ramps up during the pulse, then stays flat forever
    after - the "dead neuron" limitation: no resistor means no way for V
    to relax back to baseline once current stops.
    """
    I_e = pulse_current(TIME, PULSE_T_ON, PULSE_T_OFF, PULSE_AMPLITUDE)
    V = integrate_capacitor_voltage(TIME, I_e, C)

    # analytic check: V should be flat (constant) after t_off
    post_off = TIME > PULSE_T_OFF
    V_after_pulse = V[post_off]
    flatness = np.max(V_after_pulse) - np.min(V_after_pulse)
    print(f"[pulse demo] V variation after pulse ends: {flatness:.6f} "
          f"(should be ~0 - confirms 'dead neuron' behavior)")

    plot_current_and_voltage(
        TIME, I_e, V,
        title="Pulse Current -> Ramp Then Flat (No Return to Baseline)",
        save_path=os.path.join(FIGURES_DIR, "pulse_current_response.png"),
    )


# %%
if __name__ == "__main__":
    run_step_demo()
    run_ramp_demo()
    run_pulse_demo()

# %%