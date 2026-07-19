"""
sodium_conductance.py

MIT OCW 9.40 - Lecture 5: Hodgkin-Huxley Model, Part 2 (sodium)

The centerpiece contrast with potassium: simulate m and h SEPARATELY under
the same voltage step used in potassium_conductance.py, then multiply
G_Na(t) = G_Na_bar * m^3 * h and show the product reproduces sodium's
transient (rise-then-decay) current, even though the voltage stays
depolarized the whole time - something m (or n) alone can never produce,
since a single relaxing-toward-x_inf gate can only rise-and-plateau or
fall-and-plateau, never both in sequence at one fixed voltage.

Also reproduces the "why does I_Na shrink near E_Na" effect from the
Lecture 5 quiz: even with m^3*h at its plateau value, I_Na = G_Na*(V-E_Na)
still collapses as V approaches E_Na, purely from the driving-force term -
nothing to do with h. That's checked explicitly at the end.

Run interactively in VS Code with the Jupyter extension using the '# %%' cell markers,
or run top-to-bottom as a normal script.
"""

# %%
import os
import numpy as np
import matplotlib.pyplot as plt
from utils import (
    get_figures_dir, alpha_m, beta_m, alpha_h, beta_h, steady_state_and_tau,
    simulate_gating_variable, G_NA_BAR, E_NA,
)

FIGURES_DIR = get_figures_dir(__file__)

# %%
# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
# Same hyperpolarized/depolarized voltages and step time as
# potassium_conductance.py, for direct side-by-side comparison of the two
# conductances' shapes under identical conditions.
V_HYPERPOLARIZED = -10.0
V_DEPOLARIZED = 50.0

T_STEP = 2.0
T_MAX = 20.0
N_STEPS = 2000
TIME = np.linspace(0, T_MAX, N_STEPS)

VOLTAGE_SEQUENCE = [
    (0.0, T_STEP, V_HYPERPOLARIZED),
    (T_STEP, T_MAX, V_DEPOLARIZED),
]

# driving-force demo: sweep the depolarized clamp voltage itself, holding
# m and h at their (V-dependent) plateau values, to show I_Na shrinking
# near E_NA purely from (V - E_Na) - independent of any inactivation effect
V_CLAMP_SWEEP = np.linspace(-20, E_NA + 40, 200)


# %%
# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_m_h_and_conductance(t, m_trace, h_trace, G_Na_trace, t_step,
                              V_hyperpolarized, V_depolarized, save_path=None):
    """Three stacked subplots: m(t), h(t), and the product G_Na(t) = G_Na_bar*m^3*h."""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 8), sharex=True)

    ax1.plot(t, m_trace, color="tab:green", linewidth=2)
    ax1.axvline(t_step, color="black", linestyle=":", linewidth=1)
    ax1.set_ylabel("m (activation)")
    ax1.set_title(f"Sodium Activation & Inactivation: Step from "
                   f"{V_hyperpolarized:.0f} to {V_depolarized:.0f} mV (dep. from rest)")

    ax2.plot(t, h_trace, color="tab:red", linewidth=2)
    ax2.axvline(t_step, color="black", linestyle=":", linewidth=1)
    ax2.set_ylabel("h (inactivation)")

    ax3.plot(t, G_Na_trace, color="tab:purple", linewidth=2)
    ax3.axvline(t_step, color="black", linestyle=":", linewidth=1, label="Voltage step")
    ax3.set_xlabel("Time (ms)")
    ax3.set_ylabel("G_Na = G_Na_bar * m^3 * h")
    ax3.legend()

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_driving_force_effect(V_sweep, I_Na_at_plateau, E_Na_val, save_path=None):
    """
    I_Na at m^3*h's plateau value, swept across clamp voltage - shows the
    current shrinking back toward 0 as V approaches E_Na, confirming this
    is a driving-force effect (V - E_Na -> 0) independent of h dropping.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(V_sweep, I_Na_at_plateau, color="tab:orange", linewidth=2)
    plt.axhline(0, color="black", linewidth=0.5, linestyle="--")
    plt.axvline(E_Na_val, color="tab:red", linestyle=":", linewidth=1.5,
                label=f"E_Na = {E_Na_val:.1f} mV (dep. from rest)")
    plt.xlabel("Clamped voltage V (mV, depolarization from rest)")
    plt.ylabel("I_Na at m^3*h plateau (arbitrary units)")
    plt.title("Peak Sodium Current Collapses Near E_Na - Driving Force, Not Inactivation")
    plt.legend()
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150)
    plt.show()


# %%
# ---------------------------------------------------------------------------
# Demo: step response - m and h together produce a transient conductance
# ---------------------------------------------------------------------------
def run_transient_conductance_demo():
    """
    Simulates m and h separately under the same step, multiplies them into
    G_Na = G_Na_bar*m^3*h, and confirms the product RISES then FALLS - the
    signature transient shape - unlike potassium_conductance.py's n^4,
    which only rises and plateaus.
    """
    m_inf_hyper, tau_m_hyper = steady_state_and_tau(alpha_m(V_HYPERPOLARIZED), beta_m(V_HYPERPOLARIZED))
    m_inf_depol, tau_m_depol = steady_state_and_tau(alpha_m(V_DEPOLARIZED), beta_m(V_DEPOLARIZED))
    h_inf_hyper, tau_h_hyper = steady_state_and_tau(alpha_h(V_HYPERPOLARIZED), beta_h(V_HYPERPOLARIZED))
    h_inf_depol, tau_h_depol = steady_state_and_tau(alpha_h(V_DEPOLARIZED), beta_h(V_DEPOLARIZED))

    print(f"[sodium step response] at V={V_HYPERPOLARIZED:.0f} mV: "
          f"m_inf={m_inf_hyper:.4f} (tau={tau_m_hyper:.4f} ms), "
          f"h_inf={h_inf_hyper:.4f} (tau={tau_h_hyper:.4f} ms)")
    print(f"  at V={V_DEPOLARIZED:.0f} mV: "
          f"m_inf={m_inf_depol:.4f} (tau={tau_m_depol:.4f} ms), "
          f"h_inf={h_inf_depol:.4f} (tau={tau_h_depol:.4f} ms)")
    print(f"  m_inf rises with depolarization: {m_inf_depol > m_inf_hyper} (should be True)")
    print(f"  h_inf FALLS with depolarization: {h_inf_depol < h_inf_hyper} (should be True - mirror image of m)")
    print(f"  tau_m << tau_h at depolarized V: {tau_m_depol:.4f} ms vs {tau_h_depol:.4f} ms "
          f"(m must be much faster than h for a transient to appear at all)")

    m_trace = simulate_gating_variable("m", VOLTAGE_SEQUENCE, TIME)
    h_trace = simulate_gating_variable("h", VOLTAGE_SEQUENCE, TIME)
    G_Na_trace = G_NA_BAR * (m_trace ** 3) * h_trace

    # sanity check: G_Na should RISE after the step, reach a peak, then FALL -
    # the transient shape. Find the peak within the post-step window and
    # confirm the trace afterward drops well below it.
    post_step_mask = TIME >= T_STEP
    G_Na_post = G_Na_trace[post_step_mask]
    peak_idx = np.argmax(G_Na_post)
    peak_value = G_Na_post[peak_idx]
    final_value = G_Na_post[-1]

    print(f"  G_Na peak after step = {peak_value:.4f} (at t={TIME[post_step_mask][peak_idx]:.4f} ms)")
    print(f"  G_Na at end of sim = {final_value:.4f}")
    print(f"  Conductance falls to less than half its peak: {final_value < 0.5 * peak_value} "
          f"(should be True - this is inactivation shutting the channel back off, "
          f"unlike potassium which stays near its plateau)")

    plot_m_h_and_conductance(
        TIME, m_trace, h_trace, G_Na_trace, T_STEP, V_HYPERPOLARIZED, V_DEPOLARIZED,
        save_path=os.path.join(FIGURES_DIR, "m_h_activation_inactivation.png"),
    )


# %%
# ---------------------------------------------------------------------------
# Demo: driving-force collapse near E_Na, independent of h
# ---------------------------------------------------------------------------
def run_driving_force_demo():
    """
    Holds m and h at their (V-dependent) PLATEAU values (i.e. h hasn't had
    time to inactivate yet - simulating the initial peak of a real clamp
    response) and sweeps clamp voltage, to isolate the driving-force effect:
    I_Na should still shrink toward 0 as V -> E_Na, purely from (V - E_Na),
    confirming this isn't an inactivation artifact.
    """
    m_inf_sweep, _ = steady_state_and_tau(alpha_m(V_CLAMP_SWEEP), beta_m(V_CLAMP_SWEEP))
    h_inf_sweep, _ = steady_state_and_tau(alpha_h(V_CLAMP_SWEEP), beta_h(V_CLAMP_SWEEP))

    I_Na_plateau = G_NA_BAR * (m_inf_sweep ** 3) * h_inf_sweep * (V_CLAMP_SWEEP - E_NA)

    # sanity check: current should cross zero right at E_Na (within one grid step)
    idx_near_E = np.argmin(np.abs(V_CLAMP_SWEEP - E_NA))
    print(f"[driving-force demo] I_Na near V=E_Na ({V_CLAMP_SWEEP[idx_near_E]:.2f} mV) "
          f"= {I_Na_plateau[idx_near_E]:.6f} (should be ~0)")

    # sanity check: current well above E_Na should be small in magnitude
    # relative to the peak seen at moderate depolarization - even though
    # m_inf*h_inf's product isn't shrinking there, (V-E_Na) is
    idx_far_above = np.argmin(np.abs(V_CLAMP_SWEEP - (E_NA + 40)))
    idx_moderate = np.argmin(np.abs(V_CLAMP_SWEEP - 40))
    print(f"  I_Na at V=E_Na+40mV = {I_Na_plateau[idx_far_above]:.4f}, "
          f"I_Na at V=40mV = {I_Na_plateau[idx_moderate]:.4f} "
          f"(both driven by the same conductance regime, but the driving "
          f"force term (V-E_Na) differs sharply)")

    plot_driving_force_effect(
        V_CLAMP_SWEEP, I_Na_plateau, E_NA,
        save_path=os.path.join(FIGURES_DIR, "sodium_driving_force_collapse.png"),
    )


# %%
if __name__ == "__main__":
    run_transient_conductance_demo()
    run_driving_force_demo()

# %%
