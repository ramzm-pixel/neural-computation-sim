"""
diffusion_fick.py

MIT OCW 9.40 - Lecture 1: Fick's First Law

Simulates diffusion from a clustered "drop of dye" starting point (Option B:
particle-based diffusion), then derives concentration and flux from particle
positions -- demonstrating that Fick's Law emerges from many random walkers.

Empirically shows:
    - A concentrated cluster of particles flattens/spreads out over time
    - Flux J = -D * dC/dx is largest where the concentration gradient is steepest,
      and near zero where the profile is flat
"""

# %%
import os
import numpy as np
import matplotlib.pyplot as plt
from utils import simulate_random_walks, get_figures_dir

FIGURES_DIR = get_figures_dir(__file__)

# %%
# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
N_PARTICLES = 2000
N_STEPS = 500
STEP_SIZE = 1.0
CLUSTERED_POSITION = 5.0   # all particles start clustered near this position
N_BINS = 40                 # spatial resolution for the concentration histogram

# theoretical D, same formula as random_walk.py (tau=1 assumed here for simplicity)
D_THEORY = (STEP_SIZE ** 2) / 2


# %%
# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------
def initialize_clustered_particles(n_particles, clustered_position, rng):
    """Jitter n_particles randomly around a starting position (a 'drop of dye')."""
    starting_positions = clustered_position + rng.uniform(-0.5, 0.5, size=n_particles)
    return starting_positions


def simulate_diffusion_from_cluster(n_particles, n_steps, step_size, clustered_position, rng=None):
    """
    Reuses simulate_random_walks() (which starts everyone at 0), then shifts
    each particle by its own clustered starting offset.
    """
    if rng is None:
        rng = np.random.default_rng()

    starting_positions = initialize_clustered_particles(n_particles, clustered_position, rng)

    positions = simulate_random_walks(n_particles, n_steps, step_size, rng=rng)
    positions = positions + starting_positions[:, np.newaxis]

    return positions


# %%
# ---------------------------------------------------------------------------
# Concentration & flux from particle snapshots
# ---------------------------------------------------------------------------
def positions_to_concentration(positions, timestep, bins):
    """
    Take a snapshot of all particle positions AT ONE TIMESTEP,
    and turn it into a concentration profile via histogram.

    Parameters
    ----------
    positions : the full (n_particles, n_steps) array
    timestep : which column (point in time) to snapshot
    bins : number of bins (or bin edges) for the histogram

    Returns
    -------
    concentration : np.ndarray - count of particles in each bin
    bin_centers : np.ndarray - x-position representing each bin
    """
    # positions[:, timestep] -> all particles (rows), one point in time (one column)
    hist, bin_edges = np.histogram(positions[:, timestep], bins)

    # midpoint of each bin: average of its left and right edge
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    return hist, bin_centers


def compute_flux_from_concentration(concentration, bin_centers, D):
    """
    Apply Fick's First Law: J = -D * dC/dx

    np.gradient estimates the derivative (slope) of an array using
    finite differences - here it gives us dC and dx separately, so we
    can divide to get dC/dx.
    """
    dC = np.gradient(concentration)
    dx = np.gradient(bin_centers)

    J = -D * dC / dx

    return J


def get_concentration_over_time(positions, timesteps_to_snapshot, bins):
    """
    Loop over several chosen timesteps, computing a concentration
    snapshot at each one (using positions_to_concentration).

    A plain Python loop is fine here (not a vectorization concern like
    looping over particles would be) - np.histogram doesn't have a simple
    vectorized "many timesteps at once" equivalent, and we're only looping
    over a handful of snapshots, not thousands of particles.

    IMPORTANT: `bins` must be the SAME bin edges (an array from
    compute_shared_bin_edges) for every snapshot - not just a bin count.
    Otherwise np.histogram auto-picks a different range for each snapshot
    (based on that snapshot's min/max), making bin widths -- and therefore
    particle counts -- incomparable across timesteps. A tightly clustered
    snapshot (like t=0) would get artificially narrow bins and look
    "shorter" than it should, even though it's actually the most concentrated.

    Returns
    -------
    list of (concentration, bin_centers) tuples, one per snapshot
    """
    concentrations = []

    for timestep in timesteps_to_snapshot:
        concentration, bin_centers = positions_to_concentration(positions, timestep, bins)
        concentrations.append((concentration, bin_centers))

    return concentrations


def compute_shared_bin_edges(positions, n_bins):
    """
    Compute ONE set of bin edges spanning the full range particles reach
    across ALL timesteps (not just one snapshot).

    Using these same edges for every snapshot ensures bin width stays
    constant across timesteps, so concentration values are genuinely
    comparable between an early (tightly clustered) and late (spread out)
    snapshot -- instead of each snapshot silently getting its own bin width.
    """
    global_min = positions.min()
    global_max = positions.max()
    return np.linspace(global_min, global_max, n_bins + 1)


# %%
# ---------------------------------------------------------------------------
# Run simulation
# ---------------------------------------------------------------------------
rng = np.random.default_rng(seed=42)  # seeded for reproducibility

positions = simulate_diffusion_from_cluster(
    N_PARTICLES, N_STEPS, STEP_SIZE, CLUSTERED_POSITION, rng=rng
)

# snapshot the concentration profile at a few points in time, to show it
# flattening/spreading out as time goes on
snapshot_timesteps = [0, 25, 100, 250, 499]

# use the SAME bin edges for every snapshot (based on the full range particles
# reach across all of time), so bin widths -- and therefore particle counts --
# are directly comparable between an early, tightly clustered snapshot and a
# later, spread-out one
shared_bin_edges = compute_shared_bin_edges(positions, n_bins=N_BINS)

concentrations = get_concentration_over_time(positions, snapshot_timesteps, bins=shared_bin_edges)


# %%
# ---------------------------------------------------------------------------
# Plot 1: concentration profile flattening/spreading over time
# ---------------------------------------------------------------------------
plt.figure(figsize=(8, 5))
for timestep, (concentration, bin_centers) in zip(snapshot_timesteps, concentrations):
    plt.plot(bin_centers, concentration, label=f"t = {timestep}", alpha=0.8)
plt.xlabel("Position (x)")
plt.ylabel("Concentration (particle count per bin)")
plt.title("Concentration Profile Spreading Over Time")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "concentration_over_time.png"), dpi=150)
plt.show()


# %%
# ---------------------------------------------------------------------------
# Plot 2: flux vs position, at one snapshot in time
# ---------------------------------------------------------------------------
snapshot_for_flux = 100  # pick one timestep to inspect flux at
concentration, bin_centers = positions_to_concentration(positions, snapshot_for_flux, bins=shared_bin_edges)
flux = compute_flux_from_concentration(concentration, bin_centers, D=D_THEORY)

fig, ax1 = plt.subplots(figsize=(8, 5))

# concentration on the left axis
ax1.plot(bin_centers, concentration, color="tab:blue", label="Concentration")
ax1.set_xlabel("Position (x)")
ax1.set_ylabel("Concentration", color="tab:blue")
ax1.tick_params(axis="y", labelcolor="tab:blue")

# flux on a second, overlaid y-axis (different units/scale than concentration)
ax2 = ax1.twinx()
ax2.plot(bin_centers, flux, color="tab:red", label="Flux J = -D dC/dx")
ax2.set_ylabel("Flux (J)", color="tab:red")
ax2.tick_params(axis="y", labelcolor="tab:red")
ax2.axhline(0, color="black", linewidth=0.5, linestyle="--")

plt.title(f"Concentration & Flux at t = {snapshot_for_flux}")
fig.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "flux_vs_position.png"), dpi=150)
plt.show()

# %%