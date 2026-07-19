"""
myotonia_model.py

MIT OCW 9.40 - Lecture 5: Hodgkin-Huxley Model (Part 5 - the disease capstone)

The word-model-to-math-model case study from the second half of Lecture 5:
extends the plain HH neuron (action_potential.py) with a t-tubule
potassium-accumulation compartment (utils.py's
simulate_hh_myotonia), and sweeps ONE parameter - the fraction of sodium
channels that fail to inactivate - to reproduce BOTH clinical phenotypes
described in lecture from a single underlying mechanism:

    0% failure          -> normal, single action potential per stimulus
    small failure (~2%) -> myotonic run: many extra spikes, continuing
                            even after the stimulus is removed, driven by
                            t-tubule K+ buildup raising E_K and further
                            depolarizing the fiber (runaway feedback)
    larger failure      -> depolarization block: voltage locks at a high
                            plateau instead of oscillating - not enough
                            functioning sodium channels to support spiking,
                            but enough non-inactivating ones to hold V high

This directly encodes the corrected version of the Q6 causal chain from
the lecture discussion: extra spikes -> K+ piles up in the t-tubule
faster than diffusion clears it -> E_K rises -> further depolarization
-> more spikes (not "more K+ leaving causes more Na+ to enter", which
gets the loop backwards).

Run interactively in VS Code with the Jupyter extension using the '# %%' cell markers,
or run top-to-bottom as a normal script.
"""

# %%
import os
import numpy as np
import matplotlib.pyplot as plt
from utils import get_figures_dir, simulate_hh_myotonia, detect_spikes

FIGURES_DIR = get_figures_dir(__file__)

# %%
# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
T_MAX = 60.0     # ms - long enough to see spiking continue well after the stimulus ends
N_STEPS = 12000
TIME = np.linspace(0, T_MAX, N_STEPS)

STIMULUS_T_ON = 5.0
STIMULUS_T_OFF = 8.0     # slightly longer than action_potential.py's pulse,
                          # to reliably kick off multiple spikes in the
                          # myotonic case before the stimulus is removed
STIMULUS_AMPLITUDE = 20.0

SPIKE_THRESHOLD = 50.0

# t-tubule model parameters - see utils.py's simulate_hh_myotonia
# docstring for the reasoning behind these defaults. TIME is in ms
# throughout this script, so TAU_DIFF=350 directly matches the lecture's
# 300-400 ms diffusion-time estimate for a ~25 micron t-tubule (an earlier
# pass at this used TAU_DIFF=0.35, a 1000x unit bug - clearing that fast,
# K+ never accumulates enough to matter; see the module README for the
# tuning story).
K_OUT = 5.0
TAU_DIFF = 350.0
VOLUME = 800.0       # tuned so a single extra spike's worth of K+ influx
                      # shifts E_K by a small, physiologically-plausible
                      # amount rather than immediately swamping it -
                      # smaller volume overshoots straight to depolarization
                      # block with no oscillating "run" in between
FARADAY = 1.0
NERNST_SLOPE = 26.7  # mV, ~kT/q at ~37C for a monovalent ion (linearized
                      # Nernst - see hh_myotonia_derivatives docstring)

# the three fractions used for the headline comparison figure - found by
# sweeping frac_non_inactivating at fixed VOLUME/TAU_DIFF above and
# checking which regime each value lands in (see module README)
FRACTION_NORMAL = 0.0
FRACTION_MYOTONIA = 0.02      # -> myotonic run: several extra spikes after the stimulus
FRACTION_PARALYSIS = 0.15     # -> depolarization block: locks high and flat

# sweep for the "find the transition points" scan
FRACTION_SWEEP = np.linspace(0.0, 0.20, 25)


# %%
# ---------------------------------------------------------------------------
# Current waveform
# ---------------------------------------------------------------------------
def stimulus_current_func(t_on, t_off, amplitude):
    """Single stimulus pulse - identical shape across all three phenotype runs, only frac_non_inactivating differs."""
    def I_e(t):
        return amplitude if t_on <= t < t_off else 0.0
    return I_e


# %%
# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_three_phenotypes(t, results, t_on, t_off, save_path=None):
    """
    Three stacked V(t) subplots - normal, myotonia, paralysis - sharing a
    time axis, so the qualitative difference (single spike vs. runaway
    train vs. locked-high plateau) is directly comparable at a glance.
    """
    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    titles = [
        f"Normal (0% non-inactivating)",
        f"Myotonia ({FRACTION_MYOTONIA*100:.0f}% non-inactivating) - Runaway Spike Train",
        f"Periodic Paralysis ({FRACTION_PARALYSIS*100:.0f}% non-inactivating) - Depolarization Block",
    ]
    colors = ["tab:green", "tab:orange", "tab:red"]

    for ax, (V, _, _, _, _), title, color in zip(axes, results, titles, colors):
        ax.plot(t, V, color=color, linewidth=1.5)
        ax.axvspan(t_on, t_off, color="tab:blue", alpha=0.15, label="Stimulus")
        ax.set_ylabel("V (mV, dep. from rest)")
        ax.set_title(title)
        ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("Time (ms)")
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_ttubule_potassium(t, results, t_on, t_off, save_path=None):
    """t-tubule [K+] over time for all three cases - shows the accumulation driving the myotonic and paralysis phenotypes."""
    plt.figure(figsize=(9, 5))
    labels = ["Normal", "Myotonia", "Paralysis"]
    colors = ["tab:green", "tab:orange", "tab:red"]

    for (_, _, _, _, K_t), label, color in zip(results, labels, colors):
        plt.plot(t, K_t, color=color, linewidth=1.5, label=label)

    plt.axvspan(t_on, t_off, color="tab:blue", alpha=0.15, label="Stimulus")
    plt.axhline(K_OUT, color="black", linestyle=":", linewidth=1, label=f"Resting [K+]_out = {K_OUT:.1f}")
    plt.xlabel("Time (ms)")
    plt.ylabel("[K+] in t-tubule (arbitrary conc. units)")
    plt.title("T-Tubule Potassium Accumulation Across Phenotypes")
    plt.legend()
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_spike_count_vs_fraction(fractions, spike_counts, save_path=None):
    """Number of spikes produced (during + after the stimulus) vs. frac_non_inactivating - shows the rise, then the drop-off into depolarization block."""
    plt.figure(figsize=(8, 5))
    plt.plot(fractions * 100, spike_counts, marker="o", color="tab:purple")
    plt.xlabel("% of sodium channels failing to inactivate")
    plt.ylabel("Total spike count")
    plt.title("One Parameter, Two Phenotypes: Spike Count vs. Non-Inactivating Fraction")
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150)
    plt.show()


# %%
# ---------------------------------------------------------------------------
# Demo: the headline three-way comparison
# ---------------------------------------------------------------------------
def run_three_phenotype_comparison():
    """Runs the normal, myotonia, and paralysis cases with identical stimuli, differing only in frac_non_inactivating."""
    I_e_func = stimulus_current_func(STIMULUS_T_ON, STIMULUS_T_OFF, STIMULUS_AMPLITUDE)

    results = []
    for frac, label in [(FRACTION_NORMAL, "normal"),
                          (FRACTION_MYOTONIA, "myotonia"),
                          (FRACTION_PARALYSIS, "paralysis")]:
        V, n, m, h, K_t = simulate_hh_myotonia(
            TIME, I_e_func, frac_non_inactivating=frac,
            K_out=K_OUT, tau_diff=TAU_DIFF, volume=VOLUME, faraday=FARADAY,
            nernst_slope=NERNST_SLOPE,
        )
        spike_times = detect_spikes(V, TIME, threshold=SPIKE_THRESHOLD)
        spikes_after_stimulus = [s for s in spike_times if s > STIMULUS_T_OFF]

        print(f"[{label}] frac_non_inactivating={frac:.3f}: "
              f"{len(spike_times)} total spike(s), "
              f"{len(spikes_after_stimulus)} occurring AFTER stimulus ends, "
              f"peak [K+]_t = {K_t.max():.4f}, final V = {V[-1]:.4f} mV")

        results.append((V, n, m, h, K_t))

    # sanity checks tying back to the lecture's description
    normal_spikes = len(detect_spikes(results[0][0], TIME, threshold=SPIKE_THRESHOLD))
    myotonia_spike_times = detect_spikes(results[1][0], TIME, threshold=SPIKE_THRESHOLD)
    myotonia_spikes_after = [s for s in myotonia_spike_times if s > STIMULUS_T_OFF]

    print(f"\n  Normal case fires ~1 spike: {normal_spikes <= 2} (should be True)")
    print(f"  Myotonia continues firing after stimulus removed: "
          f"{len(myotonia_spikes_after) > 0} (should be True - this IS the myotonic run)")

    # depolarization block check: paralysis case's late-time voltage should
    # sit high and roughly flat, not returned to rest and not still oscillating
    V_paralysis = results[2][0]
    late_window = TIME > (T_MAX - 5.0)
    late_V = V_paralysis[late_window]
    late_V_range = late_V.max() - late_V.min()
    print(f"  Paralysis case late-time V: mean={late_V.mean():.4f} mV, "
          f"range over last 5ms={late_V_range:.4f} mV "
          f"(should be HIGH and roughly FLAT - locked in depolarization block, "
          f"not oscillating and not at rest)")

    plot_three_phenotypes(
        TIME, results, STIMULUS_T_ON, STIMULUS_T_OFF,
        save_path=os.path.join(FIGURES_DIR, "myotonia_vs_paralysis.png"),
    )
    plot_ttubule_potassium(
        TIME, results, STIMULUS_T_ON, STIMULUS_T_OFF,
        save_path=os.path.join(FIGURES_DIR, "ttubule_potassium_accumulation.png"),
    )

    return results


# %%
# ---------------------------------------------------------------------------
# Demo: sweep the fraction, find where each phenotype transition happens
# ---------------------------------------------------------------------------
def run_fraction_sweep():
    """Sweeps frac_non_inactivating finely, counts spikes at each value, to see the rise-then-collapse shape."""
    I_e_func = stimulus_current_func(STIMULUS_T_ON, STIMULUS_T_OFF, STIMULUS_AMPLITUDE)

    spike_counts = []
    print(f"\n[fraction sweep] scanning {len(FRACTION_SWEEP)} values from "
          f"{FRACTION_SWEEP.min():.3f} to {FRACTION_SWEEP.max():.3f}")

    for frac in FRACTION_SWEEP:
        V, n, m, h, K_t = simulate_hh_myotonia(
            TIME, I_e_func, frac_non_inactivating=frac,
            K_out=K_OUT, tau_diff=TAU_DIFF, volume=VOLUME, faraday=FARADAY,
            nernst_slope=NERNST_SLOPE,
        )
        spike_times = detect_spikes(V, TIME, threshold=SPIKE_THRESHOLD)
        spike_counts.append(len(spike_times))
        print(f"  frac={frac:.4f}  ->  {len(spike_times)} spikes")

    spike_counts = np.array(spike_counts)

    # sanity check: spike count should rise above the normal single-spike
    # baseline somewhere in the sweep (the myotonic regime existing at all)
    print(f"\n  Max spike count in sweep: {spike_counts.max()} "
          f"(at frac={FRACTION_SWEEP[np.argmax(spike_counts)]:.4f}) "
          f"- should be well above 1-2 (normal baseline)")

    plot_spike_count_vs_fraction(
        FRACTION_SWEEP, spike_counts,
        save_path=os.path.join(FIGURES_DIR, "spike_count_vs_fraction.png"),
    )


# %%
if __name__ == "__main__":
    run_three_phenotype_comparison()
    run_fraction_sweep()

# %%
