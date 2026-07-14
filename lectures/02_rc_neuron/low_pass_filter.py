"""
low_pass_filter.py

MIT OCW 9.40 - Lecture 2: The RC Model of a Neuron (Part 3 - low-pass filtering)

Demonstrates the RC circuit's low-pass filter behavior: sweep current
pulse width across a wide range (some >> tau, some << tau), record the
PEAK voltage reached during each pulse, and plot peak response vs. pulse
width.

Expected shape:
    - Long pulses (width >> tau): V has time to reach close to V_inf -> strong response
    - Short pulses (width << tau): V barely moves before current shuts off -> weak response

This is the classic low-pass filter signature: neurons respond well to
slowly-changing inputs and poorly to rapidly-changing ones.

Run interactively in VS Code with the Jupyter extension using the '# %%' cell markers,
or run top-to-bottom as a normal script.
"""

# %%
import os
import numpy as np
import matplotlib.pyplot as plt
from utils import get_figures_dir, pulse_current, tau_from_RC, V_inf_from_current, simulate_rc_response

FIGURES_DIR = get_figures_dir(__file__)

# %%
# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
R_L = 2.0    # leak resistance (Ohm)
C = 0.5      # membrane capacitance (F)  ->  tau = R_L * C = 1.0 s

PULSE_T_ON = 1.0
PULSE_AMPLITUDE = 2.0   # -> V_inf = R_L * I_0 = 4.0 V during any pulse

# Sweep pulse widths from far below tau to far above tau (log-spaced,
# since the interesting transition happens over orders of magnitude)
N_WIDTHS = 40
PULSE_WIDTHS = np.logspace(-2, 1.5, N_WIDTHS)  # ~0.01 s to ~30 s, tau = 1 s sits in the middle

# each simulation needs to run long enough after the pulse ends to be sure
# we've captured the true peak (a long pulse may still be rising at t_off)
SIM_PADDING_FACTOR = 1.5  # extra sim time beyond t_off, as a multiple of the widest pulse
N_STEPS_PER_SIM = 3000


# %%
# ---------------------------------------------------------------------------
# Core sweep logic
# ---------------------------------------------------------------------------
def peak_response_for_pulse_width(width, R_L, C, amplitude, t_on=PULSE_T_ON):
    """
    Simulate the RC response to a pulse of the given width, return the peak
    voltage reached. Peak always lands at t_off for a single positive pulse
    (V rises while I_e is on, falls once it's off) - we scan the full trace's
    max anyway as a robustness check rather than assuming that a priori.
    """
    t_off = t_on + width
    t_end = t_off + SIM_PADDING_FACTOR * width + 0.1  # small floor so very short pulses still get simulated a bit past t_off
    t = np.linspace(0, t_end, N_STEPS_PER_SIM)

    I_e_func = lambda tt: pulse_current(np.array([tt]), t_on, t_off, amplitude)[0]
    V = simulate_rc_response(t, I_e_func, R_L, C, V0=0.0,
                              discontinuities=[t_on, t_off])

    return np.max(V)


def sweep_pulse_widths(widths, R_L, C, amplitude):
    """Run peak_response_for_pulse_width() across an array of pulse widths."""
    peaks = np.array([
        peak_response_for_pulse_width(w, R_L, C, amplitude)
        for w in widths
    ])
    return peaks


# %%
# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_low_pass_curve(widths, peaks, tau, V_inf, save_path=None):
    """Peak voltage vs. pulse width, log-x axis, with V_inf and tau marked as reference lines."""
    plt.figure(figsize=(8, 5))
    plt.semilogx(widths, peaks, marker="o", markersize=3, color="tab:blue",
                 label="Peak V response")
    plt.axhline(V_inf, color="tab:green", linestyle=":", linewidth=1,
                label=f"V_inf = {V_inf:.2f} V (ceiling)")
    plt.axvline(tau, color="tab:red", linestyle="--", linewidth=1,
                label=f"tau = {tau:.2f} s")
    plt.xlabel("Pulse width (s, log scale)")
    plt.ylabel("Peak membrane voltage (V)")
    plt.title("RC Circuit as a Low-Pass Filter: Peak Response vs. Pulse Width")
    plt.legend()
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_example_traces(widths_to_show, R_L, C, amplitude, t_on=PULSE_T_ON, save_path=None):
    """
    Overlay a few individual V(t) traces (short, near-tau, long pulse
    widths) so the "barely moves" vs. "reaches ceiling" behavior is visible
    directly, not just in the peak-vs-width summary curve.

    All traces share ONE time array, sized for the LONGEST pulse's decay -
    giving each width its own short window would cut shorter traces off
    before they've decayed back to baseline, making them look incomplete
    next to the longest one when they aren't, they just weren't drawn far
    enough. (Bit me once - see the Lecture 2 README.)
    """
    plt.figure(figsize=(8, 5))
    tau = tau_from_RC(R_L, C)

    widest = max(widths_to_show)
    t_end = t_on + widest + SIM_PADDING_FACTOR * widest + 0.1
    t = np.linspace(0, t_end, N_STEPS_PER_SIM)

    for width in widths_to_show:
        t_off = t_on + width
        I_e_func = lambda tt: pulse_current(np.array([tt]), t_on, t_off, amplitude)[0]
        V = simulate_rc_response(t, I_e_func, R_L, C, V0=0.0,
                                  discontinuities=[t_on, t_off])

        plt.plot(t, V, label=f"width = {width:.3f} s ({width/tau:.2f} * tau)")

    plt.xlabel("Time (s)")
    plt.ylabel("Membrane voltage V (V)")
    plt.title("Example Voltage Traces at Different Pulse Widths")
    plt.legend()
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150)
    plt.show()


# %%
# ---------------------------------------------------------------------------
# Run: full sweep
# ---------------------------------------------------------------------------
def run_low_pass_sweep():
    """Run the full pulse-width sweep and produce the low-pass curve plus a few example traces."""
    tau = tau_from_RC(R_L, C)
    V_inf = V_inf_from_current(R_L, PULSE_AMPLITUDE)

    print(f"[low-pass sweep] tau = {tau:.4f} s, V_inf = {V_inf:.4f} V")
    print(f"  Sweeping {N_WIDTHS} pulse widths from {PULSE_WIDTHS.min():.4f} s "
          f"to {PULSE_WIDTHS.max():.4f} s")

    peaks = sweep_pulse_widths(PULSE_WIDTHS, R_L, C, PULSE_AMPLITUDE)

    # sanity checks: shortest pulse should be well below V_inf,
    # longest pulse should be very close to V_inf
    shortest_peak_fraction = peaks[0] / V_inf
    longest_peak_fraction = peaks[-1] / V_inf
    print(f"  Shortest pulse ({PULSE_WIDTHS[0]:.4f} s): reaches "
          f"{shortest_peak_fraction:.4f} of V_inf")
    print(f"  Longest pulse ({PULSE_WIDTHS[-1]:.4f} s): reaches "
          f"{longest_peak_fraction:.4f} of V_inf")

    plot_low_pass_curve(
        PULSE_WIDTHS, peaks, tau, V_inf,
        save_path=os.path.join(FIGURES_DIR, "low_pass_filter_curve.png"),
    )

    # pick a few representative widths for the example-traces plot:
    # well below tau, near tau, well above tau
    example_widths = [tau * 0.05, tau * 0.5, tau * 1.0, tau * 3.0, tau * 10.0]
    plot_example_traces(
        example_widths, R_L, C, PULSE_AMPLITUDE,
        save_path=os.path.join(FIGURES_DIR, "example_traces_by_width.png"),
    )


# %%
if __name__ == "__main__":
    run_low_pass_sweep()

# %%