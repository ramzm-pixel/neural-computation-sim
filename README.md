# Neural Computation — From Ions to Spikes

I'm working through [MIT OCW 9.40 (Introduction to Neural Computation)](https://ocw.mit.edu/)
on my own, and building a small simulation project for each lecture as I go,
to make sure I actually understand the material instead of just recognizing
the formulas.

The eventual goal is to end up with a simulated spiking neuron
(Hodgkin-Huxley model) built from scratch, but I'm taking it one lecture at
a time and only building each piece once I've actually covered it in the
course.

## Roadmap

- [x] **Lecture 1 — Ionic Diffusion & Drift**: random walk simulation, Fick's
  First Law, Ohm's Law in solution →
  [`lectures/01-ionic-diffusion-drift`](lectures/01-ionic-diffusion-drift)
- [x] **Lecture 2 — The RC Model of a Neuron**: passive membrane, membrane
  time constant, low-pass filtering, and the Nernst (equilibrium) potential →
  [`lectures/02-rc-neuron-model`](lectures/02-rc-neuron-model)
- [ ] **Lecture 3 — Ion-Specific Conductances and Integrate-and-Fire Models**:
  reversal/driving-potential framework (each ion conductance modeled as a
  resistor in series with a battery), then the Integrate-and-Fire spiking
  model and its firing-rate-vs-injected-current derivation, including the
  rheobase (minimum current needed to spike at all)
- [ ] **Lecture 4/5 — Hodgkin-Huxley**: voltage-gated channels, simulated
  action potentials — this is the last lecture this repo covers. After
  that I'll keep watching 9.40 on its own, without turning every lecture
  into a coding project (see note at the bottom).

I'll update this as I go through the course — each lecture gets its own
folder and its own README with more detail.

## Project structure

```
neural-computation-sim/
├── shared/
│   └── plotting_utils.py       # get_figures_dir() - lecture-agnostic helper
└── lectures/
    ├── 01-ionic-diffusion-drift/
    └── 02-rc-neuron-model/
```

`shared/` holds the one piece of code every lecture folder needs regardless
of topic (a helper that makes sure figures save next to the script that
made them, not wherever the terminal happened to be `cd`'d into). It's
installed as an editable package (see Setup below) so every lecture folder
can import it with a plain `from shared.plotting_utils import get_figures_dir`,
no matter which directory a script is actually run from.

Everything else — simulation logic, plotting, demos — is lecture-specific
and lives in that lecture's own folder. Each lecture folder has its own
`utils.py` for helpers that get reused across that lecture's scripts (e.g.
`simulate_random_walks` in Lecture 1, `simulate_rc_response` in Lecture 2).
Lecture-specific `utils.py` files re-export `get_figures_dir` from `shared`,
so within a lecture folder everything can still be pulled from one local
`utils` import.

## Setup

```bash
pip install -r requirements.txt
pip install -e .              # installs the shared/ package in editable mode
```

The second command is what makes `from shared.plotting_utils import
get_figures_dir` resolve correctly from inside any lecture folder.

## Module 1: Ionic Diffusion & Drift

My first project, covering the two mechanisms that move ions in solution —
random diffusion and electrically-driven drift. I simulated each one from
scratch (particle-by-particle) instead of just plugging numbers into the
formulas from the slides, to actually see the behavior the equations are
describing.

**`random_walk.py`** — simulates many particles doing independent 1D random
walks, then checks that:
- ⟨x⟩ ≈ 0 over time (diffusion is unbiased — no net drift)
- ⟨x²⟩ = 2Dt (mean squared displacement grows linearly with time)
- The diffusion coefficient D I get from fitting the simulated data matches
  the theoretical D = δ²/(2τ) from the step size and step duration I chose

![Mean squared displacement grows linearly with time](lectures/01-ionic-diffusion-drift/figures/msd_vs_time.png)

**`diffusion_fick.py`** — simulates diffusion starting from a clustered
"drop of dye," then builds a concentration profile and flux straight from
where the particles ended up (histogram + finite differences), to see
Fick's First Law (J = -D·∂C/∂x) show up on its own from a bunch of random
walkers, instead of assuming the continuum equation from the start.

![Concentration profile spreading over time](lectures/01-ionic-diffusion-drift/figures/concentration_over_time.png)

**`drift_ohms_law.py`** — looks at how resistance scales with geometry
(R = ρL/A), and compares plain diffusion against a biased "drift" random
walk side by side. This was the plot I most wanted to see for myself:
diffusion's mean position stays near zero while its spread grows as √t,
but drift's mean position grows in a straight line with t — apparently
that's the signature that tells you whether particles are just diffusing
or being pushed by a field.

![Diffusion vs drift: signature of random walk vs directed motion](lectures/01-ionic-diffusion-drift/figures/diffusion_vs_drift.png)

**`utils.py`** — the simulation code shared across all three scripts above,
so I wasn't copy-pasting the same random walk logic everywhere.

More detail (including bugs I ran into and things I'm still unsure about)
is in the [Lecture 1 README](lectures/01-ionic-diffusion-drift/README.md).

## Module 2: The RC Model of a Neuron

Second project — building up a neuron's passive electrical model from
scratch, piece by piece: a bare capacitor first (and seeing why that alone
is a "dead neuron"), then adding a leak resistor, then using that RC
circuit to demonstrate low-pass filtering, and finally deriving the
Nernst equilibrium potential — the "battery" a neuron needs to generate
its own resting potential.

**`capacitor_model.py`** — the zeroth-order model: just a capacitor, no
resistor. Confirms that a step current produces a linear voltage ramp, a
ramp current produces a parabolic voltage profile, and — most importantly
— that voltage never returns to baseline once current stops. That's the
"dead neuron" limitation that motivates adding a resistor next.

**`rc_model.py`** — the real passive membrane model: capacitor + leak
resistor, giving the governing equation τ·dV/dt = -(V - V∞). Numerically
integrates this with `scipy.integrate.odeint` and checks it against the
closed-form exponential solution. Also explicitly verifies that at t=τ,
the voltage has closed ~63% (1 - 1/e) of the gap to V∞ — the defining
property of the time constant.

![RC step response: analytic vs simulated](lectures/02-rc-neuron/figures/rc_step_analytic_vs_simulated.png)

**`low_pass_filter.py`** — sweeps current-pulse width from far below τ to
far above τ, and plots peak voltage response vs. pulse width. This is the
plot that actually shows *why* an RC neuron behaves like a low-pass filter:
short pulses barely move the voltage before they end, long pulses have
time to approach V∞.

![Low-pass filter curve](lectures/02-rc-neuron/figures/low_pass_filter_curve.png)

**`nernst_potential.py`** — derives the equilibrium (Nernst) potential via
the Boltzmann-distribution route from the lecture, and checks it against
the classic squid giant axon numbers (E_K ≈ -75 mV). Also includes a
particle-level diffusion-vs-drift simulation reusing Lecture 1's random
walk mechanics, to connect the Nernst potential back to the same
diffusion/drift competition rather than treating it as an unrelated new
formula.

**`utils.py`** — shared helpers across the four scripts above:
`step_current`/`pulse_current` generators, `tau_from_RC`,
`V_inf_from_current`, and `simulate_rc_response` (the `odeint` wrapper).

More detail — including a real bug I hit (a sign error in the Nernst
equation that flipped E_K positive) and a subtler one (`odeint` silently
skipping over a current pulse entirely for certain pulse widths) — is in
the [Lecture 2 README](lectures/02-rc-neuron/README.md).

### Running it

```bash
pip install -r requirements.txt
pip install -e .

cd lectures/01-ionic-diffusion-drift
python random_walk.py
python diffusion_fick.py
python drift_ohms_law.py

cd ../02-rc-neuron-model
python capacitor_model.py
python rc_model.py
python low_pass_filter.py
python nernst_potential.py
```

Each script has `# %%` cell markers so I could run them cell-by-cell in
VS Code with the Jupyter extension, instead of re-running the whole
simulation every time I wanted to tweak a parameter.

## A note on scope

I'm deliberately capping this repo at Lecture 5 (Hodgkin-Huxley model,
finished across lectures 4-5). Coding up every lecture from here on would
turn into busywork rather than something that's actually useful to show —
reproducing textbook derivations in code is good for learning, but past a
certain point it stops being differentiated portfolio work. I'll keep
watching 9.40 after that, just without a matching folder for every lecture.
The next project after this one will likely involve real EEG data and
reproducing a published BCI decoding paper, which is a better use of that
kind of time.