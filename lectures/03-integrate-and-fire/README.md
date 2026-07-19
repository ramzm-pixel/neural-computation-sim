# Lecture 3 — Ion-Specific Conductances and Integrate-and-Fire Models

Third project for the course. Lecture 3 does two mostly-separate things:
first it fixes the last piece missing from Lecture 2's RC model (an ion
conductance isn't just a resistor — it's a resistor *in series with a
battery*, and that's what actually gives a neuron a resting potential),
and then it sets the passive membrane model aside entirely in favor of a
much simpler spiking model, integrate-and-fire, building up its
firing-rate-vs-injected-current relationship from scratch.

## Files

### `utils.py`
Shared code across all three scripts below:
- `driving_potential`, `ionic_current` — the `V - E_ion` / `G_ion*(V-E_ion)`
  building blocks for a battery-backed conductance
- `rheobase_current` — the minimum constant current needed for `V_inf` to
  reach `V_threshold` at all
- `simulate_lif` — the integrate-and-fire simulator. `odeint` alone can't
  do a reset event mid-integration, so this steps the ODE forward
  chunk-by-chunk: integrate until `V` crosses `V_threshold` (detected
  between two consecutive time samples), record the spike, snap `V` back
  to `V_reset`, and continue from there. Handles both the no-leak case
  (`R_L=None`) and the leaky case.
- `firing_rate_no_leak`, `firing_rate_leaky_exact`,
  `firing_rate_leaky_linear_approx` — the three firing-rate formulas from
  the lecture

`step_current`, `pulse_current`, `tau_from_RC`, `V_inf_from_current`, and
`simulate_rc_response` are re-exported from `shared/circuit_utils.py` —
same RC machinery Lecture 2 uses, just with the reversal-potential
parameter (`E`) now doing real work instead of defaulting to 0.
`get_figures_dir` is re-exported the same way, from
`shared/plotting_utils.py`. This folder only ever imports from `shared/` —
never from Lecture 2's folder directly — so it stays self-sufficient as
long as `shared/` is present.

### `conductance_battery.py`
Models an ion-selective conductance as a resistor in series with a
battery:

    I_ion = G_ion * (V - E_ion)

which comes from summing voltage drops across the two elements in series
(`V_m = E_ion + I_ion/G_ion`, solved for `I_ion`). This changes Lecture 2's
governing equation from `V_inf = R_L*I_e` to:

    V_inf = E_leak + R_L*I_e

i.e. `V_inf` is now offset by the battery voltage. This is the direct fix
for the "dead neuron" problem from `capacitor_model.py` — with zero
injected current, `V` no longer freezes wherever it happened to stop, it
relaxes to a genuine resting potential (`E_leak`).

Three demos: an I-V curve confirming the line crosses zero current exactly
at `E_ion` (not at V=0, unlike Lecture 2's plain resistor), a step-current
response showing `V` resting at `E_leak` and relaxing toward the new
battery-shifted `V_inf`, and a pulse-current response showing `V` return
to `E_leak` after the pulse ends rather than to 0.

Produces: `iv_curve_battery_conductance.png`, `battery_step_response.png`,
`battery_pulse_response.png`

### `integrate_and_fire.py`
The integrate-and-fire (IF) spiking model: integrate input into `V`, and
the instant `V` reaches `V_threshold`, register a spike and reset `V` to
`V_reset`. Two flavors:

- **No-leak** (`dV/dt = I_e/C`): any constant `I_e > 0` eventually spikes,
  and fires regularly with rate `f = I_e / (C * delta_V)`.
- **Leaky** (`tau*dV/dt = -(V - V_inf)`, `V_inf = E_leak + R_L*I_e`): now
  `V_inf` itself has to exceed `V_threshold` for the neuron to ever spike.
  Below that current (the **rheobase**), `V` just relaxes toward `V_inf`
  and sits there — no matter how long the simulation runs.

Three demos, matching those two cases plus the qualitatively new one: a
no-leak neuron firing regularly, a leaky neuron firing regularly above
rheobase, and a leaky neuron *not* firing at all below rheobase (with a
pass/fail check on `len(spike_times) == 0`). Each demo also checks that
inter-spike intervals are constant under constant current (regular
firing), and the no-leak case additionally checks the empirical firing
rate against the analytic `f = I_e/(C*delta_V)`.

Produces: `no_leak_if_response.png`, `no_leak_isi.png`,
`leaky_if_response.png`, `leaky_isi.png`, `below_rheobase_if_response.png`

### `firing_rate_curve.py`
Sweeps injected current across a range spanning below and above rheobase,
and compares three things on one plot:

1. The **exact** leaky-IF firing rate, from solving the ODE for the
   inter-spike interval:
   `delta_t = -tau * ln((V_inf - V_threshold)/(V_inf - V_reset))`, `f = 1/delta_t`
2. The **large-I_e linear approximation** from the lecture:
   `f = (I_e - I_threshold) / (C*delta_V)` for `I_e > I_threshold`
3. **Empirical** firing rate, from actually running `simulate_lif` at
   several current values and measuring `1/mean(ISI)` directly — this is
   the real check, since it confirms the exact closed-form formula matches
   a genuine simulated spike train and not just self-consistent algebra.
   All 6 empirical points (I ran currents from just above rheobase up to
   6x rheobase) matched the exact formula to within ~0.02 Hz.

Also plots the leaky vs. no-leak firing rate side by side, to make the
rheobase's origin visually obvious: it's specifically the leak that
introduces the minimum-current requirement, since the no-leak curve is
just a straight line through the origin with no threshold.

Produces: `fi_curve_exact_vs_linear.png`, `fi_curve_leaky_vs_no_leak.png`

## Running

From the repo root, one-time setup:

```bash
pip install -r requirements.txt
pip install -e .
```

Then, from this folder:

```bash
cd lectures/03-integrate-and-fire
python conductance_battery.py
python integrate_and_fire.py
python firing_rate_curve.py
```

Same as Lectures 1 and 2, every script has `# %%` cell markers for running
cell-by-cell in VS Code with the Jupyter extension.

## Bugs I ran into and fixed

**None specific to this lecture's own math** — the RC-with-battery
equation is a direct generalization of Lecture 2's (E=0 recovers it
exactly, which `shared/circuit_utils.py`'s functions confirm by taking `E`
as an optional parameter defaulting to 0), and I re-ran all of Lecture 2's
scripts against the refactored `shared/circuit_utils.py` to make sure
pulling those functions out didn't change any of that lecture's output.

The one real gotcha was in `simulate_lif`: `odeint` has no concept of a
mid-integration reset event, so the naive approach (just integrate the
whole time array in one call, like Lecture 2's `simulate_rc_response`
does) can't produce a spike train at all — it'll just keep integrating
straight through `V_threshold` with no reset. `simulate_lif` works around
this by running `odeint` in chunks: integrate until a threshold crossing
is detected in the output samples, snap `V` back to `V_reset`, then start
a fresh `odeint` call from there for the rest of the time array. This
means spike timing is only as precise as the sample spacing of `t` — it
finds the crossing at whichever time sample first exceeds threshold, it
doesn't interpolate to the exact sub-sample crossing time. Fine for these
demos since `t` is sampled finely relative to `tau`, but worth knowing
about if firing rates ever look slightly off at coarser time resolution.

## Notes to self / things I'm unsure about

- `simulate_lif`'s chunk-by-chunk `odeint` approach means it's meaningfully
  slower than a single `odeint` call over the whole time array — that's
  why `firing_rate_curve.py` only runs the full simulator at a handful of
  spot-check currents (`EMPIRICAL_CURRENTS`) rather than at all 300 points
  in the main sweep, which use the closed-form formulas instead.
- Right at `I_e == I_threshold` (exactly at rheobase), the exact firing
  rate formula is mathematically 0 (the neuron never actually reaches
  threshold in finite time, even though `V_inf == V_threshold`) — I
  guard for this in `firing_rate_leaky_exact` by only computing the log
  ratio where `V_inf > V_threshold` strictly, and leaving `f=0` everywhere
  else, rather than risking a `log(0)` or `log(negative)` from float
  rounding landing exactly on the boundary.
- I didn't implement the Goldman-Hodgkin-Katz (GHK) equation this lecture
  — the lecture only introduces it qualitatively (what happens if you open
  more than one conductance at once), without deriving the actual formula,
  so there wasn't a concrete equation to check code against yet. Flagged
  as open for whenever it actually gets derived.
- This is also the last lecture whose spike generator is a simplification
  (a threshold-and-reset rule bolted on from outside the RC model, rather
  than something that falls out of the equations themselves). Lecture 4/5
  replace this with the actual Hodgkin-Huxley gating variables
  (`m`, `h`, `n`), where spiking is an emergent property of voltage- and
  time-dependent conductances rather than an assumed rule — see the
  top-level README's roadmap.
- **Final update (repo closes at Lecture 5):** both threads above are
  resolved now that the repo is finished. The GHK equation stayed
  unimplemented through Lecture 5 as well — it never came up as a concrete
  formula in the actual lecture material, only as the same qualitative
  aside, so there was nothing to check code against by the end either.
  The spike-rule simplification did get replaced as anticipated:
  `lectures/04-05-hodgkin-huxley/action_potential.py` integrates the full
  coupled `V`/`n`/`m`/`h` system and produces a real action potential with
  no threshold-and-reset rule anywhere in the code — `simulate_lif`'s
  chunk-by-chunk reset logic in this folder turned out to be the last time
  this repo needed that trick.