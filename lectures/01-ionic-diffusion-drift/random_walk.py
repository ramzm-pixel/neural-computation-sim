"""
random_walk.py

MIT OCW 9.40 - Lecture 1: Diffusion & Random Walk

Simulates many particles undergoing independent 1D random walks in solution.
Empirically verifies:
    - <x>  ~ 0                (no net drift; diffusion is unbiased)
    - <x^2> = 2*D*t            (mean squared displacement grows linearly with time)
    - D = delta^2 / (2*tau)    (macroscopic D recovered from microscopic step params)

Run interactively in VS Code with the Jupyter extension using the '# %%' cell markers,
or run top-to-bottom as a normal script.
"""

# %%
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from utils import simulate_random_walks, get_figures_dir, compute_mean_displacement, compute_mean_squared_displacement

FIGURES_DIR = get_figures_dir(__file__)

# %%
# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
N_PARTICLES = 1000      # number of particles in the simulated "crowd"
N_STEPS = 500            # number of random steps per particle
STEP_SIZE = 1.0          # delta - distance traveled per step (arbitrary units)
TAU = 1.0                # tau - time duration of a single step (arbitrary units)

DT = TAU                 # each step takes TAU seconds, so timestep = TAU
TIME = np.arange(N_STEPS) * DT   # time axis, shape (N_STEPS,)


# %%
# ---------------------------------------------------------------------------
# Analysis functions (compute_mean_displacement, compute_mean_squared_displacement,
# and simulate_random_walks all live in utils.py now, since diffusion_fick.py
# and drift_ohms_law.py reuse them too)
# ---------------------------------------------------------------------------
def fit_diffusion_coefficient(msd, time):
    """
    Linear fit of <x^2> vs t.
    Since <x^2> = 2*D*t, slope = 2*D  ->  D = slope / 2.

    Returns fitted D and the full linregress result (for R^2, etc).
    """
    slope, _, rvalue, _, _ = linregress(time, msd)
    D_fit = slope / 2 # type: ignore
    return D_fit, rvalue


def theoretical_diffusion_coefficient(step_size, tau):
    """D = delta^2 / (2*tau) - theoretical D from microscopic step parameters."""
    return (step_size ** 2) / (2 * tau)


# %%
# ---------------------------------------------------------------------------
# Run simulation
# ---------------------------------------------------------------------------
rng = np.random.default_rng(seed=42)  # seeded for reproducibility

positions = simulate_random_walks(N_PARTICLES, N_STEPS, STEP_SIZE, rng=rng)

mean_x = compute_mean_displacement(positions)
msd = compute_mean_squared_displacement(positions)

D_fit, rvalue = fit_diffusion_coefficient(msd, TIME)
D_theory = theoretical_diffusion_coefficient(STEP_SIZE, TAU)

print(f"Theoretical D (from delta^2 / 2*tau): {D_theory:.4f}")
print(f"Fitted D (from <x^2> vs t slope):     {D_fit:.4f}")
print(f"R^2 of linear fit:                    {rvalue**2:.4f}") # type: ignore


# %%
# ---------------------------------------------------------------------------
# Plot 1: individual particle trajectories
# ---------------------------------------------------------------------------
plt.figure(figsize=(8, 5))
n_to_plot = 15
for i in range(n_to_plot):
    plt.plot(TIME, positions[i], alpha=0.7, linewidth=1)
plt.xlabel("Time")
plt.ylabel("Position (x)")
plt.title(f"{n_to_plot} Individual Random Walk Trajectories")
plt.axhline(0, color="black", linewidth=0.5, linestyle="--")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "trajectories.png"), dpi=150)
plt.show()


# %%
# ---------------------------------------------------------------------------
# Plot 2: mean displacement <x> vs t  (should hover near zero)
# ---------------------------------------------------------------------------
plt.figure(figsize=(8, 5))
plt.plot(TIME, mean_x, color="tab:blue", linewidth=1)
plt.axhline(0, color="black", linewidth=0.8, linestyle="--", label="Expected: <x> = 0")
plt.xlabel("Time")
plt.ylabel("<x> (mean displacement)")
plt.title(f"Mean Displacement Across {N_PARTICLES} Particles")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "mean_displacement.png"), dpi=150)
plt.show()


# %%
# ---------------------------------------------------------------------------
# Plot 3: mean squared displacement <x^2> vs t, with theoretical overlay
# ---------------------------------------------------------------------------
plt.figure(figsize=(8, 5))
plt.plot(TIME, msd, color="tab:red", linewidth=1.5, label="Simulated <x^2>")
plt.plot(TIME, 2 * D_theory * TIME, color="black", linestyle="--",
         label=f"Theoretical: 2*D*t  (D={D_theory:.3f})")
plt.xlabel("Time")
plt.ylabel("<x^2> (mean squared displacement)")
plt.title("Mean Squared Displacement Grows Linearly With Time")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "msd_vs_time.png"), dpi=150)
plt.show()

# %%