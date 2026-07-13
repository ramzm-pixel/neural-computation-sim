"""
utils.py

Shared simulation utilities for Lecture 1 (Diffusion & Drift).

Functions here are reused across:
    - random_walk.py       (pure diffusion)
    - diffusion_fick.py     (diffusion from a clustered starting point -> concentration/flux)
    - drift_ohms_law.py     (diffusion + directional drift, for comparison)
"""

import os
import numpy as np


def get_figures_dir(script_file):
    """
    Given a script's __file__, return (and create) a 'figures' folder
    next to that script.

    This makes savefig() paths resolve relative to the SCRIPT's own
    location, not whatever directory the script happens to be run from
    (e.g. some Code Runner extensions run from a workspace root instead
    of the script's own folder, which silently breaks plain relative
    paths like "figures/plot.png").

    Call this from each script as:
        FIGURES_DIR = get_figures_dir(__file__)

    Note: __file__ always refers to whichever script is CALLING this
    function (passed in explicitly), not to utils.py itself - that's
    why script_file has to be passed in rather than hardcoded here.
    """
    script_dir = os.path.dirname(os.path.abspath(script_file))
    figures_dir = os.path.join(script_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    return figures_dir


def simulate_random_walks(n_particles, n_steps, step_size, rng=None):
    """
    Simulate independent 1D random walks for many particles.

    Each particle takes a step of +step_size or -step_size at each timestep,
    chosen with equal probability (unbiased diffusion). All particles start
    at position 0.

    Parameters
    ----------
    n_particles : int
        number of particles to simulate
    n_steps : int
        number of timesteps to simulate
    step_size : float
        distance traveled per step (delta)
    rng : np.random.Generator, optional
        random number generator (for reproducibility). If None, a fresh one is created.

    Returns
    -------
    positions : np.ndarray, shape (n_particles, n_steps)
        positions[i, t] = position of particle i at timestep t
    """
    if rng is None:
        rng = np.random.default_rng()

    # +1 or -1 at each step, for every particle
    steps = rng.choice([-1, 1], size=(n_particles, n_steps)) * step_size

    # position at time t = cumulative sum of steps up to t
    positions = np.cumsum(steps, axis=1)

    # shift so every particle starts at x = 0 at timestep 0
    positions = np.hstack([np.zeros((n_particles, 1)), positions[:, :-1]])

    return positions


def compute_mean_displacement(positions):
    """<x> at each timestep, averaged across particles. Should hover near 0 for
    pure diffusion, but grows linearly with t for drift."""
    return positions.mean(axis=0)


def compute_mean_squared_displacement(positions):
    """<x^2> at each timestep, averaged across particles."""
    return (positions ** 2).mean(axis=0)