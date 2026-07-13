# Lecture 1 — Ionic Diffusion & Drift

This is my first project for the course — simulating the two mechanisms
that move ions in solution (random diffusion and electrically-driven drift)
and checking the equations from Lecture 1 against an actual simulation,
instead of just taking them on faith from the slides.

## Files

### `utils.py`
Shared code used by all three scripts below, so I'm not copy-pasting the
same random walk logic into every file:
- `simulate_random_walks` — simulates many particles taking independent
  ±step_size random steps, starting at x=0
- `compute_mean_displacement` / `compute_mean_squared_displacement` — ⟨x⟩
  and ⟨x²⟩ across particles at each timestep
- `get_figures_dir` — makes sure the `figures/` output folder is found
  relative to the script itself, not whatever folder I happened to run it
  from (this bit me once — see notes below)

### `random_walk.py`
The core diffusion simulation. This is where I checked:
- **⟨x⟩ ≈ 0** over time — since diffusion is unbiased, positive and
  negative steps should cancel out on average
- **⟨x²⟩ = 2Dt** — mean squared displacement should grow linearly with
  time. I fit a line to the simulated ⟨x²⟩ vs t data (using
  `scipy.stats.linregress`) and recovered an empirical D that matches the
  theoretical D = δ²/(2τ) from the step size and step duration I chose

Produces: `trajectories.png`, `mean_displacement.png`, `msd_vs_time.png`

### `diffusion_fick.py`
Simulates diffusion starting from a clustered "drop of dye" (all particles
jittered around one starting position), then builds a concentration profile
and flux directly from where the particles ended up:
- Concentration profile via `np.histogram` (particle counts per spatial bin)
- Flux via `np.gradient`, applying Fick's First Law: **J = -D·∂C/∂x**

The point of doing it this way (instead of just solving the diffusion
equation directly) was to see for myself that Fick's Law actually falls out
of a bunch of individual random walkers doing their own thing — it's not a
separate assumption, it's a consequence of the same random walk from
`random_walk.py`.

Produces: `concentration_over_time.png`, `flux_vs_position.png`

**Bugs I ran into and fixed:**
- My first version let `np.histogram` auto-pick bin edges separately for
  each timestep snapshot. That's fine for one snapshot, but it meant an
  early, tightly-clustered snapshot got much narrower bins than a
  spread-out later one — so the particle counts weren't actually
  comparable across timesteps, even though each plot looked fine on its
  own. Fixed by computing one shared set of bin edges up front and reusing
  it for every snapshot.
- The flux plot shows some left-right asymmetry around the concentration
  peak. I was worried this was a bug at first, but re-running with
  different random seeds showed the asymmetry flips direction randomly —
  so it's just sampling noise from a finite number of particles, not
  something wrong in the code. `np.gradient` seems to make this noise more
  visible since it's amplifying small bin-to-bin differences.

### `drift_ohms_law.py`
Two separate things in this file:
1. **Ohm's Law in solution** (no simulation, just the formulas):
   R = ρL/A and I = ΔV/R. I swept R vs. L (comes out linear) and R vs. A
   (comes out as a 1/A curve) just to see the shape of each relationship.
2. **Diffusion vs. drift comparison**: I simulated a biased random walk
   (`simulate_drift_particles`, using unequal step probabilities via
   `rng.choice(..., p=[...])`) next to plain diffusion, and plotted ⟨x⟩ vs
   t for both. This was the plot I most wanted to see: pure diffusion's
   mean position stays near zero (only the spread grows, and only as √t),
   while drift's mean position grows in a straight line with t. That
   difference is apparently how you can tell, just from watching a
   particle cloud, whether it's diffusing or being pushed by a field.

Produces: `resistance_vs_length.png`, `resistance_vs_area.png`,
`diffusion_vs_drift.png`

## Running

From this folder:

```bash
pip install -r ../../requirements.txt
python random_walk.py
python diffusion_fick.py
python drift_ohms_law.py
```

I wrote each script with `# %%` cell markers so I could run it cell-by-cell
in VS Code (with the Jupyter extension) instead of re-running the whole
simulation every time I wanted to tweak a parameter like particle count or
drift bias.

## Notes to self / things I'm unsure about

- I averaged flux from a single simulation run, which is why it's noisy —
  averaging over multiple seeds would probably clean it up, but I haven't
  done that yet.
- I know D (diffusion) and R (resistance) are related somehow — something
  called the Einstein relation apparently connects them — but that hasn't
  come up in the course yet, so for now I'm just treating diffusion and
  drift as two separate mechanisms and comparing them visually rather than
  mathematically.

