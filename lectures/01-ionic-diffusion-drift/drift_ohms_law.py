"""
drift_ohms_law.py

MIT OCW 9.40 - Lecture 1: Drift and Ohm's Law in Solution

Two things happen in this file:
    1. Deterministic exploration of R = rho*L/A and I = delta_V/R
       (how resistance scales with geometry - no simulation involved)
    2. A simulated comparison of pure diffusion vs. biased "drift" random
       walks, to visually demonstrate the key distinction from Lecture 1:
       diffusion's mean position stays near 0 (spread grows as sqrt(t)),
       while drift's mean position grows LINEARLY with t.
"""

# %%
import os
import numpy as np
import matplotlib.pyplot as plt
from utils import (
    get_figures_dir,
    simulate_random_walks,
    compute_mean_displacement,
)

FIGURES_DIR = get_figures_dir(__file__)

# %%
# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
RHO = 60.0          # resistivity of mammalian saline (Ohm*cm, approximate)
L_FIXED = 1.0        # fixed length for the area sweep (cm)
A_FIXED = 1.0        # fixed area for the length sweep (cm^2)
DELTA_V = 1.0        # voltage for current calculations (V)

N_PARTICLES = 1000
N_STEPS = 500
STEP_SIZE = 1.0
DRIFT_BIAS = 0.6      # probability of stepping right (see simulate_drift_particles)
DT = 1.0
TIME = np.arange(N_STEPS) * DT


# %%
# ---------------------------------------------------------------------------
# Ohm's Law in solution (deterministic - no simulation)
# ---------------------------------------------------------------------------
def resistance(rho, L, A):
    """R = rho * L / A"""
    R = rho * L / A
    return R


def current(delta_V, rho, L, A):
    """I = delta_V / R"""
    R = resistance(rho, L, A)
    I = delta_V / R
    return I


def sweep_resistance_vs_length(rho, A, L_values):
    """R vs L for fixed A - should be linear."""
    R_values = L_values * rho / A
    return R_values


def sweep_resistance_vs_area(rho, L, A_values):
    """R vs A for fixed L - should be a 1/A curve."""
    R_values = L * rho / A_values
    return R_values


# %%
# ---------------------------------------------------------------------------
# Drift simulation (biased random walk)
# ---------------------------------------------------------------------------
def simulate_drift_particles(n_particles, n_steps, step_size, drift_bias, rng=None):
    """
    Same mechanics as simulate_random_walks(), but with an added directional
    bias at every step - models an electric field pushing ions in one
    direction, on top of their random thermal motion.

    drift_bias = probability of stepping RIGHT (+step_size) at each step.
    drift_bias=0.5 reduces to pure unbiased diffusion.
    drift_bias > 0.5 biases motion rightward (net positive drift).
    """
    if rng is None:
        rng = np.random.default_rng()

    # p lines up with [-1, 1]: probability of -1 is (1-drift_bias),
    # probability of +1 is drift_bias
    steps = rng.choice([-1, 1], size=(n_particles, n_steps),
                        p=[1 - drift_bias, drift_bias]) * step_size

    positions = np.cumsum(steps, axis=1)
    positions = np.hstack([np.zeros((n_particles, 1)), positions[:, :-1]])

    return positions


# %%
# ---------------------------------------------------------------------------
# Run: Ohm's Law sweeps
# ---------------------------------------------------------------------------
L_values = np.linspace(0.1, 10, 50)
A_values = np.linspace(0.1, 10, 50)

R_vs_L = sweep_resistance_vs_length(RHO, A_FIXED, L_values)
R_vs_A = sweep_resistance_vs_area(RHO, L_FIXED, A_values)

I_example = current(DELTA_V, RHO, L_FIXED, A_FIXED)
print(f"Example: I = delta_V / R = {I_example:.4f} A "
      f"(rho={RHO}, L={L_FIXED}, A={A_FIXED}, delta_V={DELTA_V})")


# %%
# ---------------------------------------------------------------------------
# Run: diffusion vs. drift simulation
# ---------------------------------------------------------------------------
rng = np.random.default_rng(seed=42)

positions_diffusion = simulate_random_walks(N_PARTICLES, N_STEPS, STEP_SIZE, rng=rng)
positions_drift = simulate_drift_particles(N_PARTICLES, N_STEPS, STEP_SIZE, DRIFT_BIAS, rng=rng)

mean_x_diffusion = compute_mean_displacement(positions_diffusion)
mean_x_drift = compute_mean_displacement(positions_drift)


# %%
# ---------------------------------------------------------------------------
# Plot 1: R vs L (linear)
# ---------------------------------------------------------------------------
plt.figure(figsize=(8, 5))
plt.plot(L_values, R_vs_L, color="tab:blue")
plt.xlabel("Length L (cm)")
plt.ylabel("Resistance R (Ohm)")
plt.title(f"Resistance vs. Length (fixed A={A_FIXED} cm^2)  —  R = rho*L/A, linear in L")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "resistance_vs_length.png"), dpi=150)
plt.show()


# %%
# ---------------------------------------------------------------------------
# Plot 2: R vs A (inverse, 1/A shape)
# ---------------------------------------------------------------------------
plt.figure(figsize=(8, 5))
plt.plot(A_values, R_vs_A, color="tab:green")
plt.xlabel("Cross-sectional Area A (cm^2)")
plt.ylabel("Resistance R (Ohm)")
plt.title(f"Resistance vs. Area (fixed L={L_FIXED} cm)  —  R = rho*L/A, inverse in A")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "resistance_vs_area.png"), dpi=150)
plt.show()


# %%
# ---------------------------------------------------------------------------
# Plot 3: diffusion vs. drift - mean position over time (the "money plot")
# ---------------------------------------------------------------------------
plt.figure(figsize=(8, 5))
plt.plot(TIME, mean_x_diffusion, color="tab:blue",
         label="Pure diffusion <x> (should stay ~0)")
plt.plot(TIME, mean_x_drift, color="tab:red",
         label=f"Drift <x> (bias={DRIFT_BIAS}, should grow linearly with t)")
plt.axhline(0, color="black", linewidth=0.5, linestyle="--")
plt.xlabel("Time")
plt.ylabel("<x> (mean displacement)")
plt.title("Diffusion vs. Drift: Signature of Random Walk vs. Directed Motion")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "diffusion_vs_drift.png"), dpi=150)
plt.show()

# %%