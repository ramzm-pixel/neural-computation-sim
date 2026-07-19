# Lecture 4/5 — The Hodgkin-Huxley Model

Fourth and final project for this repo. Lectures 4 and 5 finish what
Lecture 3 left as an assumed rule: a real spike isn't a threshold-and-reset
event bolted onto a passive membrane, it's an emergent property of two
voltage- and time-dependent conductances (sodium and potassium) built from
gating variables, coupled back into the membrane equation they themselves
depend on. Lecture 4 works out potassium's gating variable (`n`); Lecture 5
works out sodium's — which needs *two* gates, not one — assembles the full
coupled system, and then spends its second half on a genuinely striking
case study: a single sodium-channel defect, modeled all the way through,
produces two clinically opposite-looking diseases (myotonia and periodic
paralysis) as different severities of the same mechanism.

I built this as one module covering both lectures rather than splitting
them, since Lecture 4's potassium work and Lecture 5's sodium work are
really one derivation split across two class sessions — the point only
lands once both halves are in place and multiplied together.

## Files

### `utils.py`
Shared code across all six scripts below. Unlike Lecture 2/3's `utils.py`,
none of this is re-exported from `shared/` — it's all defined directly
here, since nothing in it has a second consumer outside this folder.
`shared/` is for code reused across lecture folders (that's why
`circuit_utils.py` lives there — Lecture 2 needed it, then Lecture 3
needed the same functions too); the Hodgkin-Huxley rate functions and
integrator below are specific to this lecture, so they stay local.

- `alpha_n`/`beta_n`, `alpha_m`/`beta_m`, `alpha_h`/`beta_h` — the classic
  Hodgkin-Huxley 1952 rate functions, in their original depolarization-
  from-rest convention (`V=0` at rest, not absolute mV)
- `steady_state_and_tau` — the one piece of math all three gates share:
  `alpha`, `beta` in, `(x_inf, tau_x)` out
- `simulate_gating_variable` — steps a single gate through a
  piecewise-constant voltage sequence using the closed-form exponential
  relaxation (exact, no integration error)
- `simulate_hh_neuron` — integrates the full 4-variable coupled system
  (`V`, `n`, `m`, `h`) via `odeint`, given an injected-current function
- `detect_spikes` — simple threshold-crossing spike counter/timer
- `ttubule_k_derivative`, `hh_myotonia_derivatives`, `simulate_hh_myotonia`
  — the Lecture 5 disease-model extension: a 5th state variable
  (t-tubule `[K+]`) and a `frac_non_inactivating` parameter that lets a
  fraction of sodium channels skip the `h` gate entirely

`get_figures_dir` is still re-exported from `shared/plotting_utils.py`,
same as every other lecture folder.

### `gating_kinetics.py`
Before building either conductance, this lays out the shared math: given
`alpha(V)` and `beta(V)`, every gate reduces to the same two numbers —
`x_inf(V)` and `tau_x(V)` — and the same relaxation-toward-`x_inf`
differential equation. What makes `n`, `m`, and `h` behave so differently
isn't the math, it's the shape of their individual rate functions.

Confirms the two facts the rest of the module depends on: `n_inf` and
`m_inf` both rise with depolarization while `h_inf` **falls** (mirror-image
sigmoids — this is why sodium's `m` and `h` together can produce a
transient current while potassium's `n` alone can only produce a sustained
one), and `tau_m` is much smaller than `tau_n` or `tau_h` across the whole
voltage range (mean ~0.23 ms vs. ~3.1 ms and ~2.9 ms) — the quantitative
reason sodium activates fast while both potassium activation and sodium
inactivation are comparatively slow.

Produces: `gating_variables_vs_voltage.png`, `gating_time_constants_vs_voltage.png`

### `potassium_conductance.py`
Closes out the Lecture 4 half: simulates `n`'s step response to a
voltage-clamp depolarization, then raises it to the 4th power to get
`G_K(t) = G_K_bar * n^4`. Confirms the textbook "delayed rectifier" shape —
`n` jumps toward a large `n_inf` on depolarization and relaxes up slowly,
giving a conductance that ramps up and then **stays on** for as long as
the depolarization holds, with no dip. That "stays on" behavior is the
direct contrast the next script is built around.

Produces: `n_activation_step_response.png`

### `sodium_conductance.py`
The centerpiece contrast with potassium. Simulates `m` and `h`
**separately** under the identical voltage step used in
`potassium_conductance.py`, then multiplies `G_Na(t) = G_Na_bar * m^3 * h`
and confirms the product reproduces sodium's transient (rise-then-decay)
shape — something a single relaxing gate can never produce on its own,
since it can only rise-and-plateau or fall-and-plateau, never both in
sequence at one fixed voltage.

Also isolates a separate effect that's easy to conflate with inactivation:
even holding `m` and `h` at their plateau values, `I_Na = G_Na*(V-E_Na)`
still collapses toward zero as the clamp voltage approaches `E_Na` — pure
driving-force collapse (`V - E_Na -> 0`, Ohm's law with no push left),
nothing to do with `h` dropping. Checked explicitly by sweeping clamp
voltage with the gates frozen at plateau.

Produces: `m_h_activation_inactivation.png`, `sodium_driving_force_collapse.png`

### `action_potential.py`
The centerpiece of the whole module: integrates all four coupled state
variables together via `simulate_hh_neuron`, and confirms a single brief
current pulse — no separate "spike rule" — produces a full action
potential purely from the feedback loop: `m` rises fast, `I_Na` runs away
positive feedback, `V` approaches `E_Na` and the sodium current itself
collapses (driving-force effect, see above), `n` (slower) has been
climbing the whole time and pulls `V` back down, `h` (also slower) has
been falling and shuts `I_Na` off for good.

Peak voltage comes out to ~105.5 mV depolarization-from-rest (~+40.5 mV
absolute, matching the classic HH-model textbook peak), `m` peaks well
before `n` (7.07 ms vs. 8.13 ms), and `h` bottoms out at ~0.077 — a large
majority of sodium channels inactivated at the peak of the spike.

Also runs a control with `G_Na` forced to 0: the identical current pulse
then produces only a tiny passive RC-style bump (peak ~13 mV vs. ~105.5 mV
with sodium intact) — confirming the spike is genuinely an emergent
property of the sodium/potassium feedback loop, not just "current in,
big voltage out."

Produces: `action_potential_full.png`, `action_potential_currents.png`,
`spike_vs_no_sodium_control.png`

### `refractory_period.py`
Two-pulse protocol: trigger a spike, wait a variable delay, then ask how
much current a second pulse needs to trigger a *second* spike at that
delay. Reproduces both lecture claims:

1. A second pulse at the *same* amplitude as the first, only 3 ms later,
   fails to trigger a spike (`h` is still down at ~0.108, well below its
   resting value — the channels haven't deinactivated yet).
2. There's no hard cutoff. Binary-searching the minimum second-pulse
   amplitude across a range of delays traces out a genuinely graded curve:
   delays under ~4 ms need more current than the search ceiling even finds
   (absolute refractory), then the threshold drops steeply — from ~101 at
   5.1 ms down to ~6.5 by ~16 ms — and flattens out toward the normal
   suprathreshold baseline (20) as `h` fully recovers.

Produces: `two_pulse_refractory_trace.png`, `refractory_period_curve.png`

### `myotonia_model.py`
The disease capstone — the word-model-to-math-model case study from the
second half of Lecture 5. Extends the plain HH neuron with a t-tubule
potassium-accumulation compartment (`simulate_hh_myotonia`), and sweeps
one parameter — the fraction of sodium channels that fail to inactivate
(`frac_non_inactivating`) — to reproduce both clinical phenotypes from a
single mechanism:

- **0% failure** (normal): 1 spike, returns to rest.
- **2% failure** (myotonia): 4 total spikes, 3 of them occurring *after*
  the stimulus is removed — a genuine runaway myotonic run, driven by
  t-tubule K+ piling up faster than it can diffuse away, raising `E_K`,
  further depolarizing the fiber, triggering more spikes.
- **15% failure** (periodic paralysis): locks into a flat, elevated
  plateau (final V ~76 mV depolarization-from-rest) instead of
  oscillating — depolarization block. Not enough functioning sodium
  channels left to support spiking, but enough non-inactivating ones to
  hold the voltage high.

The fraction sweep (0 to 0.20, 25 steps) shows the full arc directly: spike
count rises from 1 at `frac=0` to a peak of 4 around `frac=0.02-0.05`, then
falls back to 1 by `frac~0.13` and stays there — the myotonia-to-paralysis
transition, all from one number.

Produces: `myotonia_vs_paralysis.png`, `ttubule_potassium_accumulation.png`,
`spike_count_vs_fraction.png`

## Running

From the repo root, one-time setup:

```bash
pip install -r requirements.txt
pip install -e .
```

Then, from this folder:

```bash
cd lectures/04-05-hodgkin-huxley
python gating_kinetics.py
python potassium_conductance.py
python sodium_conductance.py
python action_potential.py
python refractory_period.py
python myotonia_model.py
```

Same as every earlier lecture, every script has `# %%` cell markers for
running cell-by-cell in VS Code with the Jupyter extension.
`refractory_period.py` and `myotonia_model.py` are noticeably slower than
the others (binary search over several `odeint` runs, and a 25-point
fraction sweep respectively) — a few minutes rather than a few seconds.

## Bugs I ran into and fixed

**The one that actually mattered: a 1000x unit bug in the t-tubule model.**
`myotonia_model.py`'s time array is in milliseconds throughout (matching
every other script in this folder), but my first pass at `TAU_DIFF` used
`0.35` — intending "350 ms," Lecture 5's own diffusion-time estimate for
the t-tubule, but actually specifying 0.35 ms in the units the simulation
was running in, a thousand times faster than intended. At that clearance
rate, potassium never accumulated enough to matter, and every
`frac_non_inactivating` value just produced 1 spike, no myotonic run, no
paralysis — a flat, boring result that didn't match the lecture at all.

I found it by tracing `V`, `K_t`, and the dynamic `E_K` through a "stuck"
case sample-by-sample rather than just staring at the final plot — the
trace showed the cell settling into a *stable* elevated plateau (`n` stuck
high, `h` stuck low) rather than oscillating, which pointed at the
potassium-clearance timescale being wrong relative to the spike dynamics,
not a sign error or a wrong formula. Fixed by setting `TAU_DIFF = 350.0`
directly in the script's own units.

**A second, subtler tuning problem once the units were fixed.** Even with
the right timescale, my first `VOLUME` (the t-tubule's effective size in
the K+-concentration ODE) was still too small — the coupling between K+
buildup and `E_K` was so strong that any non-inactivating fraction jumped
straight from "no effect" to full depolarization block, with no
in-between regime that actually oscillated into a myotonic run. Fixed by
sweeping `VOLUME` and `frac_non_inactivating` together until a genuine
multi-spike run showed up — `VOLUME = 800.0` with `frac_non_inactivating`
in roughly the 0.02-0.05 range is the regime that actually reproduces
the lecture's qualitative picture. Both constants are convenience units
rather than literal electrophysiology values (see `ttubule_k_derivative`'s
docstring) precisely so they could be tuned this way without redefining
the model.

**Removable singularities in `alpha_n` and `alpha_m`.** Both classic HH
rate functions have a `0/0` at one specific voltage (`V=10` for `alpha_n`,
`V=25` for `alpha_m`) where the analytic limit is well-defined (0.1 and
1.0 respectively) but a literal implementation divides by zero right at
that point. Guarded both with `np.where` on the denominator rather than
letting `V` sweeps that happen to land near those points produce a NaN
that would silently poison an entire downstream `odeint` run.

## Notes to self / things I'm unsure about

- The myotonia model's `VOLUME`/`TAU_DIFF`/`NERNST_SLOPE` constants are
  tuned to reproduce the *qualitative* shape Lecture 5 describes (myotonic
  run at a small failure fraction, depolarization block at a larger one),
  not fit to any specific published parameter set for real t-tubule
  geometry or intracellular volume. If I ever wanted to make quantitative
  claims from this model (e.g. "this predicts an attack threshold at X%
  channel failure"), those constants would need to come from the actual
  Cannon/Corey papers rather than from tuning against "does it visibly
  oscillate."
- The linearized Nernst relation used for `E_K(K_t)` (`E_K_rest +
  nernst_slope * ln(K_t/K_rest)`) is the real Nernst equation, not an
  approximation of it — but it assumes `[K+]` inside the t-tubule
  compartment is the only thing changing; a fully first-principles version
  would also need the intracellular concentration to move (conservation
  of the K+ that left the cell), which this model doesn't track. Fine for
  reproducing the lecture's point about the extracellular side, but it's
  a simplification worth knowing about.
- `simulate_gating_variable`'s piecewise-constant-voltage approach (used
  in `potassium_conductance.py` and `sodium_conductance.py`) is exact for
  a single isolated gate under a voltage clamp, but `action_potential.py`,
  `refractory_period.py`, and `myotonia_model.py` all use the full
  `odeint`-based `simulate_hh_neuron`/`simulate_hh_myotonia` instead, since
  once `V` is a dynamic variable rather than an experimenter-controlled
  clamp, there's no piecewise-constant voltage sequence to exploit — the
  whole point of those scripts is that `V` isn't fixed anymore.
- Never got around to implementing the GHK equation, which Lecture 3's
  README already flagged as open and Lecture 5 didn't revisit either —
  closing that out here since this is the last lecture in the repo: it
  stays unimplemented. The gating-variable/conductance framework used
  throughout this repo (each ion treated independently, driving potentials
  summed) is the one Hodgkin and Huxley actually built the spike model on,
  and it's what every script in this folder relies on — GHK would only
  have mattered for scenarios with multiple simultaneously-permeant ions
  interacting nonlinearly, which never came up as a concrete equation to
  check code against.
- `hh_derivatives`/`hh_myotonia_derivatives` assume the three gates are
  fully independent of the membrane's own history beyond their current
  values (standard HH assumption) — the lecture itself notes Hodgkin and
  Huxley's independence assumption between `m` and `h` isn't quite exact
  biologically, just "not bad." Nothing here tries to correct for that;
  it's the same simplification the original 1952 model makes.

## This is the last lecture in this repo

Per the top-level README's note on scope: this folder closes out the
project. A few threads that came up along the way and were deliberately
left open in earlier READMEs, closed out here since there won't be a
Lecture 6 folder to revisit them in:

- **Lecture 3's open GHK-equation question** — see the note above. Staying
  unimplemented; never became a concrete equation to check code against
  in either Lecture 4 or 5.
- **Lecture 3's "spike generation is still an assumed rule, not emergent"
  note** — resolved here. `action_potential.py` is the payoff: spiking
  falls directly out of the coupled `V`/`n`/`m`/`h` system with no
  threshold-and-reset rule anywhere in it.
- **Lecture 1's Einstein-relation aside** (the D-and-R connection that
  hadn't come up in the course yet as of that README) — never came up in
  Lectures 2 through 5 either. Leaving it exactly as flagged in the
  Lecture 1 README rather than guessing at a connection the course itself
  never made explicit.

No further updates planned for this repo. Next project (real EEG data,
reproducing a published BCI decoding paper) starts fresh elsewhere, per
the top-level README.
