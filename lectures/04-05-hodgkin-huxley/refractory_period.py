"""
refractory_period.py

MIT OCW 9.40 - Lecture 5: Hodgkin-Huxley Model (Part 4 - refractory period)

Two-pulse protocol: inject a first (suprathreshold) pulse to trigger a
spike, wait a variable delay, then inject a SECOND pulse of the same
amplitude and ask whether it triggers a second spike.

Reproduces the two claims from lecture:
    1. A second pulse shortly after the first does NOT trigger a spike -
       h has not yet recovered (deinactivated) back to its resting value,
       so m^3*h can't reach the level needed for the runaway upstroke even
       with an identical suprathreshold stimulus.
    2. There's no hard cutoff - sweeping the inter-pulse delay and instead
       asking "how much current is needed to trigger a second spike at
       this delay" traces out a GRADED curve: right after a spike it takes
       much more current, and that required current gradually decreases
       back toward the normal threshold as h recovers.

Run interactively in VS Code with the Jupyter extension using the '# %%' cell markers,
or run top-to-bottom as a normal script.
"""

# %%
import os
import numpy as np
import matplotlib.pyplot as plt
from utils import get_figures_dir, simulate_hh_neuron, detect_spikes

FIGURES_DIR = get_figures_dir(__file__)

# %%
# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
T_MAX = 40.0
N_STEPS = 8000
TIME = np.linspace(0, T_MAX, N_STEPS)

FIRST_PULSE_T_ON = 5.0
FIRST_PULSE_T_OFF = 6.0
FIRST_PULSE_AMPLITUDE = 20.0   # same suprathreshold amplitude as action_potential.py

SECOND_PULSE_DURATION = 1.0
SPIKE_THRESHOLD = 50.0

# demonstration 1: fixed short delay, same amplitude as the first pulse -
# should fail to produce a second spike
SHORT_DELAY = 3.0    # ms after first pulse ends

# demonstration 2: sweep delay, binary-search the minimum amplitude that
# still triggers a second spike at each delay
DELAY_SWEEP = np.linspace(1.0, 20.0, 15)   # ms after first pulse ends
AMPLITUDE_SEARCH_MIN = 5.0
AMPLITUDE_SEARCH_MAX = 200.0
AMPLITUDE_SEARCH_TOLERANCE = 1.0
MAX_SEARCH_ITERS = 12


# %%
# ---------------------------------------------------------------------------
# Current waveform: two pulses
# ---------------------------------------------------------------------------
def two_pulse_current_func(t1_on, t1_off, amp1, t2_on, t2_off, amp2):
    """Callable I_e(t): amp1 during [t1_on, t1_off), amp2 during [t2_on, t2_off), else 0."""
    def I_e(t):
        if t1_on <= t < t1_off:
            return amp1
        if t2_on <= t < t2_off:
            return amp2
        return 0.0
    return I_e


# %%
# ---------------------------------------------------------------------------
# Core: does a given second-pulse amplitude/delay produce a second spike?
# ---------------------------------------------------------------------------
def second_pulse_triggers_spike(delay, amplitude):
    """
    Runs the full two-pulse simulation, returns True if a spike is detected
    starting at or after the second pulse's onset (i.e. a genuinely second,
    distinct spike - not just the tail of the first one).
    """
    second_on = FIRST_PULSE_T_OFF + delay
    second_off = second_on + SECOND_PULSE_DURATION

    I_e_func = two_pulse_current_func(
        FIRST_PULSE_T_ON, FIRST_PULSE_T_OFF, FIRST_PULSE_AMPLITUDE,
        second_on, second_off, amplitude,
    )

    V, n, m, h = simulate_hh_neuron(TIME, I_e_func)
    spike_times = detect_spikes(V, TIME, threshold=SPIKE_THRESHOLD)

    second_spikes = [s for s in spike_times if s >= second_on]
    return len(second_spikes) > 0, V, n, m, h


def find_threshold_amplitude(delay, amp_min=AMPLITUDE_SEARCH_MIN, amp_max=AMPLITUDE_SEARCH_MAX,
                              tolerance=AMPLITUDE_SEARCH_TOLERANCE, max_iters=MAX_SEARCH_ITERS):
    """
    Binary search for the minimum second-pulse amplitude that triggers a
    second spike at a given delay. Assumes monotonicity (larger amplitude
    is more likely to spike) - true for this model across the tested range.

    Returns np.inf if even amp_max fails to trigger a spike (delay too
    short for any reasonable pulse to work).
    """
    triggers_at_max, _, _, _, _ = second_pulse_triggers_spike(delay, amp_max)
    if not triggers_at_max:
        return np.inf

    lo, hi = amp_min, amp_max
    for _ in range(max_iters):
        if hi - lo < tolerance:
            break
        mid = 0.5 * (lo + hi)
        triggers, _, _, _, _ = second_pulse_triggers_spike(delay, mid)
        if triggers:
            hi = mid
        else:
            lo = mid

    return hi


# %%
# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_two_pulse_trace(t, V, h, t1_on, t1_off, t2_on, t2_off, triggered, save_path=None):
    """V(t) and h(t) together, with both pulses marked - shows h still depressed when the second pulse fires."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

    ax1.plot(t, V, color="tab:blue", linewidth=2)
    ax1.axvspan(t1_on, t1_off, color="tab:orange", alpha=0.2, label="1st pulse")
    ax1.axvspan(t2_on, t2_off, color="tab:red", alpha=0.2, label="2nd pulse")
    ax1.set_ylabel("V (mV, dep. from rest)")
    ax1.set_title(f"Two-Pulse Protocol: Second Pulse {'DOES' if triggered else 'does NOT'} Trigger a Spike")
    ax1.legend()

    ax2.plot(t, h, color="tab:red", linewidth=2)
    ax2.axvspan(t1_on, t1_off, color="tab:orange", alpha=0.2)
    ax2.axvspan(t2_on, t2_off, color="tab:red", alpha=0.2)
    ax2.set_xlabel("Time (ms)")
    ax2.set_ylabel("h (Na+ inactivation)")

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_refractory_curve(delays, thresholds, baseline_amplitude, save_path=None):
    """
    Threshold amplitude needed to trigger a second spike, vs. delay since
    the first spike - the graded refractory curve. Should start high (or
    infinite, at very short delays) and decrease toward the normal
    suprathreshold amplitude as delay grows.
    """
    finite_mask = np.isfinite(thresholds)

    plt.figure(figsize=(8, 5))
    plt.plot(delays[finite_mask], thresholds[finite_mask], marker="o", color="tab:blue",
              label="Minimum 2nd-pulse amplitude to spike")
    if np.any(~finite_mask):
        # mark delays where even the max searched amplitude failed
        plt.scatter(delays[~finite_mask], [AMPLITUDE_SEARCH_MAX] * np.sum(~finite_mask),
                    marker="x", color="tab:red", s=60,
                    label=f"No spike even at {AMPLITUDE_SEARCH_MAX:.0f} (absolute refractory)")
    plt.axhline(baseline_amplitude, color="tab:green", linestyle=":", linewidth=1.5,
                label=f"Baseline suprathreshold amplitude ({baseline_amplitude:.0f})")
    plt.xlabel("Delay after first pulse ends (ms)")
    plt.ylabel("Threshold amplitude for 2nd spike")
    plt.title("Graded Refractory Period: No Hard Cutoff")
    plt.legend()
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150)
    plt.show()


# %%
# ---------------------------------------------------------------------------
# Demo 1: fixed short delay, same amplitude - should fail
# ---------------------------------------------------------------------------
def run_short_delay_demo():
    """Confirms a second pulse at the same amplitude, shortly after the first spike, fails to trigger a second spike."""
    second_on = FIRST_PULSE_T_OFF + SHORT_DELAY
    second_off = second_on + SECOND_PULSE_DURATION

    triggered, V, n, m, h = second_pulse_triggers_spike(SHORT_DELAY, FIRST_PULSE_AMPLITUDE)

    print(f"[short delay demo] delay={SHORT_DELAY:.1f} ms, "
          f"same amplitude as first pulse ({FIRST_PULSE_AMPLITUDE:.1f})")
    print(f"  Second pulse triggered a spike: {triggered} (should be False - refractory)")

    h_at_second_pulse = h[np.argmin(np.abs(TIME - second_on))]
    print(f"  h at second pulse onset = {h_at_second_pulse:.4f} "
          f"(should be well below its resting value - still inactivated)")

    plot_two_pulse_trace(
        TIME, V, h, FIRST_PULSE_T_ON, FIRST_PULSE_T_OFF, second_on, second_off, triggered,
        save_path=os.path.join(FIGURES_DIR, "two_pulse_refractory_trace.png"),
    )


# %%
# ---------------------------------------------------------------------------
# Demo 2: sweep delay, find threshold amplitude at each - the graded curve
# ---------------------------------------------------------------------------
def run_refractory_curve_demo():
    """Binary-searches the minimum second-pulse amplitude needed at each delay, plots the resulting graded curve."""
    print(f"\n[refractory curve demo] sweeping {len(DELAY_SWEEP)} delays "
          f"from {DELAY_SWEEP.min():.1f} to {DELAY_SWEEP.max():.1f} ms")

    thresholds = np.array([find_threshold_amplitude(delay) for delay in DELAY_SWEEP])

    for delay, thresh in zip(DELAY_SWEEP, thresholds):
        thresh_str = f"{thresh:.2f}" if np.isfinite(thresh) else "no spike (searched up to max)"
        print(f"  delay={delay:5.2f} ms  ->  threshold amplitude = {thresh_str}")

    # sanity check: threshold should generally decrease as delay increases
    # (later delays need less current) - check the trend over the finite
    # portion rather than requiring strict monotonicity at every point,
    # since the binary search has some numerical tolerance
    finite_thresholds = thresholds[np.isfinite(thresholds)]
    if len(finite_thresholds) >= 2:
        overall_decreasing = finite_thresholds[-1] < finite_thresholds[0]
        print(f"  Overall trend decreasing (later delay = less current needed): "
              f"{overall_decreasing} (should be True)")

    plot_refractory_curve(
        DELAY_SWEEP, thresholds, FIRST_PULSE_AMPLITUDE,
        save_path=os.path.join(FIGURES_DIR, "refractory_period_curve.png"),
    )


# %%
if __name__ == "__main__":
    run_short_delay_demo()
    run_refractory_curve_demo()

# %%
