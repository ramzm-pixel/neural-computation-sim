# Lecture 2 — The RC Model of a Neuron

Second project for the course. Lecture 2 builds up a neuron's passive
electrical behavior piece by piece — starting from a bare capacitor,
showing why that alone isn't enough, adding a leak resistor to get the
full RC model, using that model to demonstrate low-pass filtering, and
finally explaining where a neuron's own "battery" (its resting/equilibrium
potential) actually comes from.

I built one script per stage, in the order the lecture develops them.

## Files

### `utils.py`
Shared code across all four scripts below:
- `step_current` / `pulse_current` — current waveform generators
- `tau_from_RC`, `V_inf_from_current` — the two derived quantities that
  characterize an RC circuit's response
- `rc_derivative`, `simulate_rc_response` — the ODE right-hand side and a
  wrapper around `scipy.integrate.odeint` that numerically integrates it

`get_figures_dir` is re-exported here from `shared/plotting_utils.py` at
the repo root (see the top-level README), so `from utils import
get_figures_dir, ...` still works the same way it does in Lecture 1's
folder.

### `capacitor_model.py`
The zeroth-order model — just a capacitor, no resistor yet:

    I_e = C * dV/dt

Three demos: a step current gives a linear voltage ramp, a ramp current
gives a parabolic voltage profile, and a pulse current ramps the voltage
up then leaves it flat forever once the current stops. That last one is
the point of this script — a pure capacitor is a "dead neuron," because
there's no resistor to let charge leak back out, so voltage never returns
to baseline on its own once you stop injecting current.

Produces: `step_current_response.png`, `ramp_current_response.png`,
`pulse_current_response.png`

### `rc_model.py`
Adds the leak resistor. Kirchhoff's current law gives:

    V/R_L + C*dV/dt = I_e

which rearranges into the canonical form:

    tau * dV/dt = -(V - V_inf)      where tau = R_L*C, V_inf = R_L*I_e

I numerically integrate this with `scipy.integrate.odeint` and check it
against the closed-form exponential solution
`V(t) = V_inf + (V0 - V_inf)*exp(-(t-t_start)/tau)` — the two should
overlay exactly, and they do (max error ~0 in the printed sanity checks).

Also explicitly checks that at `t = t_on + tau`, the voltage has closed
`1 - 1/e ≈ 63.2%` of the gap to V_inf — that's what "tau" actually means,
and it's worth checking numerically rather than just trusting the algebra.

Produces: `rc_step_response.png`, `rc_step_analytic_vs_simulated.png`,
`rc_pulse_response.png`, `rc_pulse_analytic_vs_simulated.png`

### `low_pass_filter.py`
Sweeps current pulse width from far below tau to far above tau (log-spaced,
since the interesting transition happens over orders of magnitude), and
records the peak voltage reached for each width. Plots peak response vs.
pulse width — this is the actual demonstration of *why* an RC neuron is a
low-pass filter: short pulses don't leave enough time for V to rise very
far before the current shuts off again; long pulses have time to approach
V_inf.

Produces: `low_pass_filter_curve.png`, `example_traces_by_width.png`

### `nernst_potential.py`
The last piece — where does a neuron's own resting voltage (its "battery")
actually come from, with no external electrode? Derives the Nernst
equilibrium potential via the Boltzmann-distribution route the lecture
uses:

    P1/P2 = exp(-dE/kT)   ->   dV = (kT/Q) * ln([ion]_out / [ion]_in)

Checked against the classic squid giant axon numbers ([K+]_in=400mM,
[K+]_out=20mM), which should give E_K ≈ -75 mV.

Also includes a particle-level simulation reusing Lecture 1's random walk
mechanics (diffusion competing against a directional drift bias), to
connect the Nernst potential back to the same diffusion-vs-drift mechanism
from Lecture 1, rather than treating it as an unrelated new formula.

Produces: `nernst_vs_concentration_ratio.png`, `particle_diffusion_vs_drift.png`

## Running

From the repo root, one-time setup:

```bash
pip install -r requirements.txt
pip install -e .
```

Then, from this folder:

```bash
cd lectures/02-rc-neuron-model
python capacitor_model.py
python rc_model.py
python low_pass_filter.py
python nernst_potential.py
```

Same as Lecture 1, every script has `# %%` cell markers for running
cell-by-cell in VS Code with the Jupyter extension.

## Bugs I ran into and fixed

**Sign error in the Nernst equation.** My first version of
`nernst_potential()` used `ln(conc_in/conc_out)`, which gave E_K = **+75.68
mV** for the squid axon numbers — backwards from the expected -75 mV. The
fix was flipping it to `ln(conc_out/conc_in)`. I'd built a sign check
directly into `run_squid_axon_demo()` (K+ diffusing out of a cell should
leave the inside net negative, so E_K must come out negative) specifically
because this is a place the lecture calls out as easy to get backwards, and
the check caught it immediately rather than me just trusting the formula
because it ran without errors.

**`odeint` silently stepping over a current pulse.** This one was much
subtler and showed up as a strange dip in the low-pass filter curve at
pulse widths around 1.1-2.2s (right around tau). It turned out `odeint`
takes its own adaptive internal steps between the output points I asked
for, sized assuming the right-hand side is smooth. A step/pulse current has
a hard jump, and for certain pulse widths `odeint`'s adaptive step size
happened to be wide enough to jump clean over the entire pulse — it never
"saw" the current turn on, and returned a flat zero trace while still
reporting `"Integration successful"`. No error, no NaN, nothing that a
basic sanity check on the output would catch — it just silently produced
wrong-but-plausible-looking data for specific pulse widths only.

Fixed by passing `discontinuities=[t_on, t_off]` into
`simulate_rc_response`, which forwards them to `odeint`'s `tcrit`
parameter — this forces the solver to place a step exactly at those times
instead of potentially skipping over them. Every call site that drives the
ODE with a step or pulse current now passes this in.

**Takeaway for later lectures:** any time an ODE solver is driven by a
discontinuous input (a synaptic current pulse, a step stimulus, etc. — this
will matter again for the Hodgkin-Huxley model), the solver needs to be
told explicitly where the discontinuities are. It won't find them on its
own, and it won't necessarily error out if it misses one.

## Notes to self / things I'm unsure about

- `R_L` and `C` in these scripts are chosen so `tau = 1.0s` exactly, purely
  for convenient demo plots — real neurons have tau in the 10-100ms range.
  The rescaling doesn't change any qualitative behavior, just the time
  axis.
- The particle simulation in `nernst_potential.py` is explicitly
  qualitative — it uses a fixed illustrative drift bias rather than
  self-consistently updating the electric field as charge separates (which
  would be a much bigger simulation). It's there to build intuition for
  *why* diffusion and drift reach a balance, not to numerically derive the
  -75mV figure from particle mechanics.