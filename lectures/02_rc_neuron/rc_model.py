"""
rc_model.py

MIT OCW 9.40 - Lecture 2: The RC Model of a Neuron (Part 2 - RC circuit)

Full RC model of a neuron: capacitor + leak resistor.

Governing equation (Kirchhoff's current law):
    I_leak + I_C = I_e
    V/R_L + C*dV/dt = I_e

Rearranged into canonical first-order linear ODE form:
    tau * dV/dt = -(V - V_inf)
    where tau = R_L * C  and  V_inf = R_L * I_e

Analytic solution for constant I_e (and therefore constant V_inf):
    V(t) = V_inf + (V(0) - V_inf) * exp(-t/tau)

This models the neuron actually "getting tired" and relaxing toward a
steady state, instead of the pure capacitor's endless integration -
this is what turns the RC circuit into a low-pass filter (see
low_pass_filter.py) and gives it a taste of "dead neuron" recovery,
though a real resting potential still needs the battery from
nernst_potential.py.

Run interactively in VS Code with the Jupyter extension using the '# %%' cell markers,
or run top-to-bottom as a normal script.
"""

# %%
import os
import numpy as np
import matplotlib.pyplot as plt
from utils import get_figures_dir, step_current, pulse_current, tau_from_RC, V_inf_from_current, simulate_rc_response

FIGURES_DIR = get_figures_dir(__file__)

# %%
# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
# Chosen so tau = R_L * C = 1.0 s exactly - convenient round number for
# demo plots. Real neurons have tau ~ 10-100 ms; that rescaling doesn't
# change any of the qualitative behavior, only the time axis.
R_L = 2.0    # leak resistance (Ohm)
C = 0.5      # membrane capacitance (F)  ->  tau = R_L * C = 1.0 s

T_MAX = 6.0
N_STEPS = 2000
TIME = np.linspace(0, T_MAX, N_STEPS)

# step demo
STEP_T_ON = 1.0
STEP_AMPLITUDE = 2.0   # -> V_inf = R_L * I_0 = 4.0 V

# pulse demo (on -> hold -> off, to show relaxation both directions)
PULSE_T_ON = 1.0
PULSE_T_OFF = 3.5
PULSE_AMPLITUDE = 2.0


# %%
# ---------------------------------------------------------------------------
# Core RC dynamics
# ---------------------------------------------------------------------------
def analytic_rc_response(t, V0, V_inf, tau, t_start=0.0):
    """Closed-form solution: V(t) = V_inf + (V0 - V_inf) * exp(-(t-t_start)/tau), for t >= t_start."""
    return V_inf + (V0 - V_inf) * np.exp(-(t - t_start) / tau)


# %%
# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_rc_response(t, I_e, V, V_inf=None, tau=None, title="RC model", save_path=None):
    """Same layout as plot_current_and_voltage() in capacitor_model.py, with V_inf overlaid if given."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    ax1.plot(t, I_e, color="tab:orange")
    ax1.set_ylabel("Injected current I_e (A)")
    ax1.set_title(title)
    ax1.axhline(0, color="black", linewidth=0.5, linestyle="--")

    ax2.plot(t, V, color="tab:blue", label="V(t)")
    if V_inf is not None:
        ax2.axhline(V_inf, color="tab:green", linewidth=1, linestyle=":",
                    label=f"V_inf = {V_inf:.2f} V")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Membrane voltage V (V)")
    ax2.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax2.legend()

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_analytic_vs_simulated(t, V_simulated, V_analytic, title="Analytic vs. simulated", save_path=None):
    """Overlay simulated (odeint) vs. closed-form V(t) on the same axes, to confirm they match."""
    plt.figure(figsize=(8, 5))
    plt.plot(t, V_simulated, color="tab:blue", linewidth=2, label="Simulated (odeint)")
    plt.plot(t, V_analytic, color="black", linestyle="--", linewidth=1.5, label="Analytic")
    plt.xlabel("Time (s)")
    plt.ylabel("Membrane voltage V (V)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150)
    plt.show()


# %%
# ---------------------------------------------------------------------------
# Demo: step current -> exponential relaxation to V_inf
# ---------------------------------------------------------------------------
def run_step_demo():
    """Constant current at t=STEP_T_ON -> V relaxes exponentially toward V_inf, time constant tau=R_L*C."""
    tau = tau_from_RC(R_L, C)
    V_inf = V_inf_from_current(R_L, STEP_AMPLITUDE)

    I_e_func = lambda t: step_current(np.array([t]), STEP_T_ON, STEP_AMPLITUDE)[0]
    V_sim = simulate_rc_response(TIME, I_e_func, R_L, C, V0=0.0,
                                  discontinuities=[STEP_T_ON])

    # analytic check on the post-t_on segment (V0=0 at t_on, relaxing to V_inf)
    post_on_mask = TIME >= STEP_T_ON
    t_post = TIME[post_on_mask]
    V_analytic_post = analytic_rc_response(t_post, V0=0.0, V_inf=V_inf, tau=tau, t_start=STEP_T_ON)

    max_error = np.max(np.abs(V_sim[post_on_mask] - V_analytic_post))
    print(f"[step demo] tau = {tau:.4f} s, V_inf = {V_inf:.4f} V, "
          f"max sim-vs-analytic error = {max_error:.6f}")

    I_e_trace = step_current(TIME, STEP_T_ON, STEP_AMPLITUDE)
    plot_rc_response(
        TIME, I_e_trace, V_sim, V_inf=V_inf, tau=tau,
        title="RC Model: Step Current -> Exponential Relaxation to V_inf",
        save_path=os.path.join(FIGURES_DIR, "rc_step_response.png"),
    )

    # build a full-length analytic trace (0 before t_on, relaxation after) for the overlay plot
    V_analytic_full = np.where(
        TIME < STEP_T_ON, 0.0,
        analytic_rc_response(TIME, V0=0.0, V_inf=V_inf, tau=tau, t_start=STEP_T_ON)
    )
    plot_analytic_vs_simulated(
        TIME, V_sim, V_analytic_full,
        title="RC Step Response: Analytic vs. Simulated",
        save_path=os.path.join(FIGURES_DIR, "rc_step_analytic_vs_simulated.png"),
    )


# %%
# ---------------------------------------------------------------------------
# Demo: pulse current -> relax toward V_inf, then relax back toward 0
# ---------------------------------------------------------------------------
def run_pulse_demo():
    """
    Pulse current -> V relaxes toward V_inf during the pulse, then relaxes
    back to 0 once it ends - unlike capacitor_model.py's pulse demo, this
    one actually returns to baseline. That's the key thing the resistor adds.
    """
    tau = tau_from_RC(R_L, C)
    V_inf_on = V_inf_from_current(R_L, PULSE_AMPLITUDE)

    I_e_func = lambda t: pulse_current(np.array([t]), PULSE_T_ON, PULSE_T_OFF, PULSE_AMPLITUDE)[0]
    V_sim = simulate_rc_response(TIME, I_e_func, R_L, C, V0=0.0,
                                  discontinuities=[PULSE_T_ON, PULSE_T_OFF])

    # analytic solution, piecewise: flat at 0, then relax to V_inf_on, then relax back to 0
    on_mask = (TIME >= PULSE_T_ON) & (TIME < PULSE_T_OFF)
    off_mask = TIME >= PULSE_T_OFF

    V_analytic = np.zeros_like(TIME)
    V_analytic[on_mask] = analytic_rc_response(
        TIME[on_mask], V0=0.0, V_inf=V_inf_on, tau=tau, t_start=PULSE_T_ON
    )

    # voltage at t_off is needed as the starting point for the "off" segment
    V_at_t_off = analytic_rc_response(
        np.array([PULSE_T_OFF]), V0=0.0, V_inf=V_inf_on, tau=tau, t_start=PULSE_T_ON
    )[0]
    V_analytic[off_mask] = analytic_rc_response(
        TIME[off_mask], V0=V_at_t_off, V_inf=0.0, tau=tau, t_start=PULSE_T_OFF
    )

    max_error = np.max(np.abs(V_sim - V_analytic))
    print(f"[pulse demo] tau = {tau:.4f} s, V_inf (during pulse) = {V_inf_on:.4f} V, "
          f"V at t_off = {V_at_t_off:.4f} V, max sim-vs-analytic error = {max_error:.6f}")

    I_e_trace = pulse_current(TIME, PULSE_T_ON, PULSE_T_OFF, PULSE_AMPLITUDE)
    plot_rc_response(
        TIME, I_e_trace, V_sim, V_inf=V_inf_on, tau=tau,
        title="RC Model: Pulse Current -> Relax Up, Then Relax Back to Baseline",
        save_path=os.path.join(FIGURES_DIR, "rc_pulse_response.png"),
    )
    plot_analytic_vs_simulated(
        TIME, V_sim, V_analytic,
        title="RC Pulse Response: Analytic vs. Simulated",
        save_path=os.path.join(FIGURES_DIR, "rc_pulse_analytic_vs_simulated.png"),
    )


# %%
# ---------------------------------------------------------------------------
# Demo: verify tau numerically (1/e decay check)
# ---------------------------------------------------------------------------
def run_tau_verification():
    """Checks that at t = t_on + tau, V has closed 1 - 1/e (~63%) of the gap to V_inf."""
    tau = tau_from_RC(R_L, C)
    V_inf = V_inf_from_current(R_L, STEP_AMPLITUDE)

    I_e_func = lambda t: step_current(np.array([t]), STEP_T_ON, STEP_AMPLITUDE)[0]

    # simulate finely enough to land close to exactly t_on + tau
    fine_time = np.linspace(0, STEP_T_ON + 3 * tau, 5000)
    V_sim = simulate_rc_response(fine_time, I_e_func, R_L, C, V0=0.0,
                                  discontinuities=[STEP_T_ON])

    idx_at_tau = np.argmin(np.abs(fine_time - (STEP_T_ON + tau)))
    V_at_tau = V_sim[idx_at_tau]

    fraction_of_gap_closed = (V_at_tau - 0.0) / (V_inf - 0.0)
    theoretical_fraction = 1 - np.exp(-1)  # ~0.632

    print(f"[tau verification] at t = t_on + tau ({fine_time[idx_at_tau]:.4f} s): "
          f"V = {V_at_tau:.4f} V")
    print(f"  Fraction of gap to V_inf closed: {fraction_of_gap_closed:.4f} "
          f"(theoretical 1 - 1/e = {theoretical_fraction:.4f})")


# %%
if __name__ == "__main__":
    run_step_demo()
    run_pulse_demo()
    run_tau_verification()

# %%