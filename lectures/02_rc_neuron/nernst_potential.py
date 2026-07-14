"""
nernst_potential.py

MIT OCW 9.40 - Lecture 2: The RC Model of a Neuron (Part 4 - the battery)

Derives and computes the equilibrium (Nernst) potential for an ion species,
via the Boltzmann-distribution route from the lecture:

    P1/P2 = exp(-dE / kT)          (Boltzmann distribution over two states)
    dE = Q * dV                     (energy difference for a charged particle)
    =>  dV = (kT/Q) * ln(P1/P2)
    =>  dV = (kT/Q) * ln([ion]_in / [ion]_out)   (probability ratio = concentration ratio)

This is the voltage a membrane settles at if it's permeable to ONLY one ion
species - the "battery" that a pure RC model is missing, since a bare
capacitor+resistor has no way to generate its own resting potential.

Also includes a from-scratch particle simulation (many charged particles
diffusing under a concentration gradient AND an opposing electric field) to
connect the Nernst equation back to the same diffusion-vs-drift competition
covered in Lecture 1's drift_ohms_law.py, rather than treating it as an
unrelated new formula.

Run interactively in VS Code with the Jupyter extension using the '# %%' cell markers,
or run top-to-bottom as a normal script.
"""

# %%
import os
import numpy as np
import matplotlib.pyplot as plt
from utils import get_figures_dir

FIGURES_DIR = get_figures_dir(__file__)

# %%
# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
K_B = 1.380649e-23        # Boltzmann constant (J/K)
Q_ELEMENTARY = 1.602176634e-19  # elementary charge (C)
T_ROOM = 293.15            # room temperature (K), ~20 C

# Squid giant axon numbers from the lecture
K_IN = 400.0    # intracellular [K+] (mM)
K_OUT = 20.0    # extracellular [K+] (mM)

# A second ion for comparison/generalization - typical Na+ concentrations
# (approximate, illustrative - real values vary by preparation)
NA_IN = 50.0
NA_OUT = 440.0


# %%
# ---------------------------------------------------------------------------
# Core Nernst equation
# ---------------------------------------------------------------------------
def kT_over_Q(T=T_ROOM, valence=1):
    """
    kT/Q in volts (valence=1 for K+/Na+, -1 for Cl-, 2 for Ca2+, etc).
    Comes out to ~25 mV for a monovalent ion at room temp, matching the
    "25 mV" shortcut used throughout electrophysiology.
    """
    return (K_B * T) / (valence * Q_ELEMENTARY)


def nernst_potential(conc_in, conc_out, T=T_ROOM, valence=1):
    """
    Nernst equilibrium potential (V_in - V_out): dV = (kT/Q) * ln(conc_out / conc_in).

    Sign check to keep in mind: a positive ion diffusing OUT of the cell
    (conc_in > conc_out) carries positive charge away from the inside,
    leaving it more NEGATIVE - so conc_in > conc_out must give a negative
    potential. That's what flipping the log ratio to out/in guarantees.
    Got this backwards on the first pass - see the Lecture 2 README.
    """
    return kT_over_Q(T, valence) * np.log(conc_out / conc_in)


# %%
# ---------------------------------------------------------------------------
# Particle-based verification (diffusion + drift under a field)
# ---------------------------------------------------------------------------
def simulate_ion_equilibrium(n_particles, n_steps, step_size, E_field_strength,
                              conc_ratio_bias, rng=None):
    """
    Diffusion (unbiased random walk) competing against drift (a directional
    bias from an electric field) - reuses Lecture 1's simulate_drift_particles
    mechanics. E_field_strength is a probability shift added to 0.5 (0.1 ->
    p_right=0.6), standing in for a real field's effect on step probability.

    This is qualitative, not a first-principles derivation of the Nernst
    potential from particle mechanics (that would need the field to update
    self-consistently as charge separates - a much bigger simulation). The
    point is just to connect it back to Lecture 1's diffusion-vs-drift race.

    conc_ratio_bias isn't used yet - kept as a placeholder for a future
    version where the starting particle density is non-uniform, to actually
    reflect a real concentration gradient.
    """
    if rng is None:
        rng = np.random.default_rng()

    p_right = 0.5 + E_field_strength
    p_right = np.clip(p_right, 0.0, 1.0)

    steps = rng.choice([-1, 1], size=(n_particles, n_steps),
                        p=[1 - p_right, p_right]) * step_size
    positions = np.cumsum(steps, axis=1)
    positions = np.hstack([np.zeros((n_particles, 1)), positions[:, :-1]])

    return positions


# %%
# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_nernst_vs_concentration_ratio(ratios, T=T_ROOM, valence=1, save_path=None):
    """
    Nernst potential vs. [ion]_in/[ion]_out ratio - shows how compressed the
    relationship is on a log scale (a 20x concentration difference only
    produces ~75 mV, not something huge).
    """
    # ratios represents conc_in/conc_out; with conc_out fixed at 1,
    # conc_in = ratio directly
    voltages_mV = nernst_potential(ratios, 1.0, T=T, valence=valence) * 1000

    plt.figure(figsize=(8, 5))
    plt.semilogx(ratios, voltages_mV, color="tab:purple")
    plt.axhline(0, color="black", linewidth=0.5, linestyle="--")
    plt.axvline(1, color="black", linewidth=0.5, linestyle="--")
    plt.xlabel("[ion]_in / [ion]_out (log scale)")
    plt.ylabel("Equilibrium potential (mV)")
    plt.title(f"Nernst Potential vs. Concentration Ratio (valence={valence}, T={T:.1f} K)")
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_particle_drift_demo(positions_no_field, positions_with_field, save_path=None):
    """Mean particle displacement over time, with vs. without a drift field - links back to Lecture 1."""
    mean_no_field = positions_no_field.mean(axis=0)
    mean_with_field = positions_with_field.mean(axis=0)
    time = np.arange(positions_no_field.shape[1])

    plt.figure(figsize=(8, 5))
    plt.plot(time, mean_no_field, color="tab:blue", label="No field (pure diffusion)")
    plt.plot(time, mean_with_field, color="tab:red", label="With field (drift)")
    plt.axhline(0, color="black", linewidth=0.5, linestyle="--")
    plt.xlabel("Time (steps)")
    plt.ylabel("<x> (mean displacement)")
    plt.title("Diffusion vs. Drift Under a Field - Same Mechanism Behind the Nernst Potential")
    plt.legend()
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150)
    plt.show()


# %%
# ---------------------------------------------------------------------------
# Demo: squid giant axon K+ equilibrium potential
# ---------------------------------------------------------------------------
def run_squid_axon_demo():
    """Computes E_K for the classic squid axon numbers, checks it lands near -75 mV."""
    kT_Q = kT_over_Q()
    print(f"[squid axon demo] kT/Q at room temp = {kT_Q * 1000:.4f} mV "
          f"(lecture shortcut: ~25 mV)")

    E_K = nernst_potential(K_IN, K_OUT)
    print(f"  E_K = (kT/Q) * ln([K+]_in / [K+]_out) "
          f"= {kT_Q*1000:.2f} mV * ln({K_IN}/{K_OUT}) = {E_K * 1000:.2f} mV")
    print(f"  Expected from lecture: approx -75 mV")

    # sign check: K+ is more concentrated inside, so it diffuses OUT,
    # leaving the inside net negative -> E_K should be negative. Confirm:
    if E_K < 0:
        print("  Sign check OK: E_K is negative, consistent with K+ leaving "
              "a net positive charge deficit inside the cell.")
    else:
        print("  WARNING: E_K came out positive - check the sign convention "
              "against the physical picture (K+ diffusing out should make "
              "the inside more negative).")

    E_Na = nernst_potential(NA_IN, NA_OUT)
    print(f"\n  For comparison, E_Na (illustrative Na+ concentrations) = {E_Na * 1000:.2f} mV")
    print("  Na+ is more concentrated OUTSIDE, so it diffuses IN, making the "
          "inside more positive -> E_Na should be positive, consistent with "
          "typical values (~+55 to +60 mV in real neurons).")


# %%
# ---------------------------------------------------------------------------
# Demo: Nernst potential across a range of concentration ratios
# ---------------------------------------------------------------------------
def run_concentration_sweep_demo():
    """Sweeps concentration ratio over several orders of magnitude, plots the resulting potential."""
    ratios = np.logspace(-2, 2, 100)  # 0.01 to 100
    plot_nernst_vs_concentration_ratio(
        ratios,
        save_path=os.path.join(FIGURES_DIR, "nernst_vs_concentration_ratio.png"),
    )


# %%
# ---------------------------------------------------------------------------
# Demo: particle-level diffusion vs. drift (qualitative link to Lecture 1)
# ---------------------------------------------------------------------------
def run_particle_drift_demo():
    """Runs the particle sim with and without a drift field, to link back to Lecture 1's random walks."""
    rng = np.random.default_rng(seed=42)

    n_particles, n_steps, step_size = 1000, 500, 1.0

    positions_no_field = simulate_ion_equilibrium(
        n_particles, n_steps, step_size, E_field_strength=0.0,
        conc_ratio_bias=1.0, rng=rng
    )
    positions_with_field = simulate_ion_equilibrium(
        n_particles, n_steps, step_size, E_field_strength=0.1,
        conc_ratio_bias=1.0, rng=rng
    )

    plot_particle_drift_demo(
        positions_no_field, positions_with_field,
        save_path=os.path.join(FIGURES_DIR, "particle_diffusion_vs_drift.png"),
    )


# %%
if __name__ == "__main__":
    run_squid_axon_demo()
    run_concentration_sweep_demo()
    run_particle_drift_demo()

# %%