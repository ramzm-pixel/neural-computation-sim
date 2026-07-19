"""
utils.py

Shared simulation utilities for Lecture 4/5 (Hodgkin-Huxley Model, Parts 1 and 2).

Functions here are reused across:
    - gating_kinetics.py       (n_inf/m_inf/h_inf overview, generic kinetics)
    - potassium_conductance.py (n gate step response, G_K = G_K_bar * n^4)
    - sodium_conductance.py    (m and h gates together, G_Na = G_Na_bar * m^3 * h)
    - action_potential.py      (full coupled V/n/m/h loop -> a real spike)
    - refractory_period.py     (two-pulse protocol using the full HH integrator)
    - myotonia_model.py        (t-tubule K+ extension - both disease phenotypes)

Unlike Lecture 2/3's utils.py, none of this is re-exported from shared/ -
it's all defined directly here. shared/ is for code reused ACROSS lecture
folders (that's the whole reason circuit_utils.py exists: Lecture 2 needed
it, then Lecture 3 needed the same thing too). Nothing here has a second
consumer outside this folder - it's Hodgkin-Huxley-specific from the
alpha/beta rate functions on down - so it stays local.

get_figures_dir is the one thing still re-exported from shared/ - same as
every other lecture folder in this repo, since it's the one truly
lecture-agnostic helper.

Classic Hodgkin-Huxley squid axon parameters (alpha/beta rate functions,
conductances, reversal potentials) are used throughout, in the ORIGINAL
Hodgkin-Huxley 1952 convention: V here is DEPOLARIZATION FROM REST
(V = V_m - V_rest), not absolute membrane voltage. This matches the
alpha/beta functions' classic algebraic form. Scripts that want an
absolute mV figure shift by V_REST_ABSOLUTE at the plotting boundary
rather than rederiving the rate functions in absolute-voltage form.

This folder only ever imports from shared/ - never from another lecture's
folder - so it stays self-sufficient as long as shared/ is present.
"""

import os
import sys
import numpy as np
from scipy.integrate import odeint

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from shared.plotting_utils import get_figures_dir


# ---------------------------------------------------------------------------
# Classic squid giant axon parameters (Hodgkin & Huxley 1952)
# ---------------------------------------------------------------------------
G_K_BAR = 36.0      # max potassium conductance (mS/cm^2)
G_NA_BAR = 120.0    # max sodium conductance (mS/cm^2)
G_LEAK = 0.3        # leak conductance (mS/cm^2)

E_K = -12.0         # potassium reversal potential, relative to rest (mV)
E_NA = 115.0        # sodium reversal potential, relative to rest (mV)
E_LEAK = 10.6       # leak reversal potential, relative to rest (mV)

C_M = 1.0           # membrane capacitance (uF/cm^2)

V_REST_ABSOLUTE = -65.0  # for converting to/from absolute mV, illustrative


# ---------------------------------------------------------------------------
# alpha/beta rate functions (classic HH 1952 form, V = depolarization from rest)
# ---------------------------------------------------------------------------
def alpha_n(V):
    """Potassium activation (n) opening rate."""
    V = np.asarray(V, dtype=float)
    denom = np.exp((10.0 - V) / 10.0) - 1.0
    # standard removable-singularity handling at V=10 (denom -> 0):
    # the analytic limit is 0.1, so guard the divide rather than let it NaN
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(np.abs(denom) < 1e-7, 0.1, 0.01 * (10.0 - V) / denom)
    return result


def beta_n(V):
    """Potassium activation (n) closing rate."""
    V = np.asarray(V, dtype=float)
    return 0.125 * np.exp(-V / 80.0)


def alpha_m(V):
    """Sodium activation (m) opening rate."""
    V = np.asarray(V, dtype=float)
    denom = np.exp((25.0 - V) / 10.0) - 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(np.abs(denom) < 1e-7, 1.0, 0.1 * (25.0 - V) / denom)
    return result


def beta_m(V):
    """Sodium activation (m) closing rate (deactivation)."""
    V = np.asarray(V, dtype=float)
    return 4.0 * np.exp(-V / 18.0)


def alpha_h(V):
    """Sodium inactivation (h) closing rate - note h_inf DECREASES with V, so alpha_h decreases with V."""
    V = np.asarray(V, dtype=float)
    return 0.07 * np.exp(-V / 20.0)


def beta_h(V):
    """Sodium inactivation (h) opening rate (deinactivation)."""
    V = np.asarray(V, dtype=float)
    return 1.0 / (np.exp((30.0 - V) / 10.0) + 1.0)


# ---------------------------------------------------------------------------
# Generic gating-variable kinetics: alpha/beta -> x_inf, tau_x
# ---------------------------------------------------------------------------
def steady_state_and_tau(alpha, beta):
    """
    Given alpha(V), beta(V) values (already evaluated at some V), return
    (x_inf, tau_x) via the standard two-state kinetic scheme:
        x_inf = alpha / (alpha + beta)
        tau_x = 1 / (alpha + beta)
    Works for n, m, or h alike - this is the one piece of math shared by
    all three gates, exactly the generic "relax exponentially toward x_inf"
    picture from lecture.
    """
    x_inf = alpha / (alpha + beta)
    tau_x = 1.0 / (alpha + beta)
    return x_inf, tau_x


def gate_derivative(x, alpha, beta):
    """dx/dt = alpha*(1-x) - beta*x, the standard two-state gating ODE (equivalent to (x_inf - x)/tau_x)."""
    return alpha * (1.0 - x) - beta * x


def simulate_gating_variable(gate, V_sequence, t, V0_gate=None):
    """
    Step a single gating variable (n, m, or h) through a piecewise-constant
    voltage sequence, using the closed-form exponential relaxation:
        x(t) = x_inf + (x(t0) - x_inf) * exp(-(t-t0)/tau_x)
    within each constant-voltage segment (exact solution of the linear ODE,
    no numerical integration error - matches how the lecture describes
    "piecewise constant changes in voltage").

    Parameters
    ----------
    gate : one of "n", "m", "h"
    V_sequence : list of (t_start, t_end, V) tuples, V given as
                 depolarization-from-rest (mV)
    t : full time array to evaluate the gate on (must span V_sequence)
    V0_gate : initial value of the gate; defaults to x_inf at the first
              segment's voltage (i.e. assumes the cell has equilibrated
              there before t=0)

    Returns
    -------
    x_trace : gate value at each time in t
    """
    alpha_fn, beta_fn = {"n": (alpha_n, beta_n), "m": (alpha_m, beta_m), "h": (alpha_h, beta_h)}[gate]

    x_trace = np.empty_like(t, dtype=float)

    if V0_gate is None:
        V_first = V_sequence[0][2]
        x_inf0, _ = steady_state_and_tau(alpha_fn(V_first), beta_fn(V_first))
        x_current = float(x_inf0)
    else:
        x_current = V0_gate

    for t_start, t_end, V_seg in V_sequence:
        x_inf, tau_x = steady_state_and_tau(alpha_fn(V_seg), beta_fn(V_seg))
        mask = (t >= t_start) & (t < t_end)
        x_trace[mask] = x_inf + (x_current - x_inf) * np.exp(-(t[mask] - t_start) / tau_x)

        # carry the value at t_end forward as the next segment's starting point
        x_current = x_inf + (x_current - x_inf) * np.exp(-(t_end - t_start) / tau_x)

    # fill any trailing samples exactly at/after the last segment's t_end
    last_t_end = V_sequence[-1][1]
    tail_mask = t >= last_t_end
    if np.any(tail_mask):
        x_trace[tail_mask] = x_current

    return x_trace


# ---------------------------------------------------------------------------
# Full coupled Hodgkin-Huxley neuron (n, m, h, V all integrated together)
# ---------------------------------------------------------------------------
def hh_derivatives(state, t, I_e_func, G_K=G_K_BAR, G_Na=G_NA_BAR, G_L=G_LEAK,
                    E_K_=E_K, E_Na_=E_NA, E_L=E_LEAK, C=C_M):
    """
    Right-hand side for odeint: state = [V, n, m, h], all in HH's
    depolarization-from-rest convention. Implements the full loop from
    lecture: currents from each gate's present value, summed, driving dV/dt,
    while each gate simultaneously relaxes toward its own (V-dependent)
    steady state.
    """
    V, n, m, h = state

    I_K = G_K * (n ** 4) * (V - E_K_)
    I_Na = G_Na * (m ** 3) * h * (V - E_Na_)
    I_L = G_L * (V - E_L)

    I_e = I_e_func(t)
    dVdt = (I_e - I_K - I_Na - I_L) / C

    dndt = gate_derivative(n, alpha_n(V), beta_n(V))
    dmdt = gate_derivative(m, alpha_m(V), beta_m(V))
    dhdt = gate_derivative(h, alpha_h(V), beta_h(V))

    return [dVdt, dndt, dmdt, dhdt]


def resting_gate_values(V0=0.0):
    """
    Steady-state (n, m, h) at a given resting depolarization V0 (default 0,
    i.e. exactly at rest) - the natural initial condition for simulations
    that start with the neuron sitting quietly before any current is injected.
    """
    n0, _ = steady_state_and_tau(alpha_n(V0), beta_n(V0))
    m0, _ = steady_state_and_tau(alpha_m(V0), beta_m(V0))
    h0, _ = steady_state_and_tau(alpha_h(V0), beta_h(V0))
    return float(n0), float(m0), float(h0)


def simulate_hh_neuron(t, I_e_func, G_K=G_K_BAR, G_Na=G_NA_BAR, G_L=G_LEAK,
                        E_K_=E_K, E_Na_=E_NA, E_L=E_LEAK, C=C_M,
                        V0=0.0, gate_state0=None):
    """
    Integrate the full 4-variable Hodgkin-Huxley system (V, n, m, h) over
    time array t, given an injected-current function I_e_func(t).

    V0 and the returned V trace are in depolarization-from-rest convention
    (V=0 at rest), matching the alpha/beta functions above - shift by
    V_REST_ABSOLUTE at the call site if absolute mV is wanted for a plot.

    gate_state0 defaults to the steady-state (n, m, h) at V0, i.e. assumes
    the neuron has equilibrated at V0 before t[0].

    Returns
    -------
    V, n, m, h : each an array matching t's shape
    """
    if gate_state0 is None:
        n0, m0, h0 = resting_gate_values(V0)
    else:
        n0, m0, h0 = gate_state0

    state0 = [V0, n0, m0, h0]
    result = odeint(hh_derivatives, state0, t,
                     args=(I_e_func, G_K, G_Na, G_L, E_K_, E_Na_, E_L, C))

    V_trace = result[:, 0]
    n_trace = result[:, 1]
    m_trace = result[:, 2]
    h_trace = result[:, 3]

    return V_trace, n_trace, m_trace, h_trace


def detect_spikes(V, t, threshold=50.0):
    """
    Simple upward-threshold-crossing spike detector on a V(t) trace (V in
    depolarization-from-rest convention, so threshold=50 mV corresponds to
    a genuine fast upstroke, well above any passive fluctuation).

    Returns a list of approximate spike times (time of the sample at which
    V first exceeds threshold on each crossing, not sub-sample interpolated -
    fine for counting spikes and rough ISI statistics).
    """
    above = V > threshold
    crossings = np.where(np.diff(above.astype(int)) == 1)[0] + 1
    return t[crossings].tolist()


# ---------------------------------------------------------------------------
# T-tubule potassium accumulation extension (Lecture 5 myotonia model)
# ---------------------------------------------------------------------------
def ttubule_k_derivative(K_t, I_K, K_out, tau_diff, volume, faraday=96485.0):
    """
    d[K+]_t/dt for the t-tubule compartment (Lecture 5 Section 7.5):
        diffusion term:  -(K_t - K_out) / tau_diff
        current term:     I_K / (volume * Faraday)

    I_K here is the potassium current FLOWING OUT of the main compartment
    (i.e. into the t-tubule) at this instant - same G_K*(n^4)*(V-E_K) term
    used in hh_derivatives, just fed in already-computed by the caller so
    this function stays a pure one-line ODE piece, not a re-implementation
    of the conductance model.

    Units are intentionally left in "convenient sim units" rather than
    literal SI/electrophysiology units (real Faraday's constant is
    96485 C/mol) - volume and faraday are both parameters precisely so a
    calling script can rescale them until the accumulation timescale and
    coupling strength line up with the lecture's qualitative picture,
    rather than silently baking in a specific unit system here. See
    myotonia_model.py's own comments for the tuning story - the naive
    "physically literal" constants overshoot straight to depolarization
    block with no oscillating run in between; a much weaker coupling
    (large `volume`) is what actually reproduces a myotonic run.
    """
    diffusion_term = -(K_t - K_out) / tau_diff
    current_term = I_K / (volume * faraday)
    return diffusion_term + current_term


def hh_myotonia_derivatives(state, t, I_e_func, frac_non_inactivating,
                             K_out, tau_diff, volume, faraday,
                             G_K=G_K_BAR, G_Na=G_NA_BAR, G_L=G_LEAK,
                             E_Na_=E_NA, E_L=E_LEAK, C=C_M,
                             nernst_slope=26.7, E_K_rest=E_K, K_rest=None):
    """
    Extended HH system for the Lecture 5 myotonia/periodic-paralysis model:
    state = [V, n, m, h, K_t] where K_t is t-tubule potassium concentration.

    Two changes from the plain hh_derivatives model:
      1. A fraction `frac_non_inactivating` of the sodium conductance does
         NOT carry the h gate (models channels that fail to inactivate) -
         so total I_Na = G_Na*(m^3)*[ (1-frac)*h + frac ]*(V-E_Na).
      2. E_K is no longer fixed - it's recomputed each step from the
         current t-tubule potassium concentration via a linearized Nernst
         relation: E_K(K_t) = E_K_rest + nernst_slope*ln(K_t/K_rest)
         (nernst_slope defaults to ~26.7 mV, i.e. kT/q at ~37C, so this
         reduces to the textbook Nernst equation for a monovalent ion).

    K_rest defaults to K_out (t-tubule at rest sits at the bulk
    extracellular concentration, before any spiking has occurred).
    """
    V, n, m, h, K_t = state

    if K_rest is None:
        K_rest = K_out

    E_K_dynamic = E_K_rest + nernst_slope * np.log(K_t / K_rest)

    I_K = G_K * (n ** 4) * (V - E_K_dynamic)

    h_effective = (1.0 - frac_non_inactivating) * h + frac_non_inactivating
    I_Na = G_Na * (m ** 3) * h_effective * (V - E_Na_)

    I_L = G_L * (V - E_L)

    I_e = I_e_func(t)
    dVdt = (I_e - I_K - I_Na - I_L) / C

    dndt = gate_derivative(n, alpha_n(V), beta_n(V))
    dmdt = gate_derivative(m, alpha_m(V), beta_m(V))
    dhdt = gate_derivative(h, alpha_h(V), beta_h(V))

    dKdt = ttubule_k_derivative(K_t, I_K, K_out, tau_diff, volume, faraday)

    return [dVdt, dndt, dmdt, dhdt, dKdt]


def simulate_hh_myotonia(t, I_e_func, frac_non_inactivating,
                          K_out=5.0, tau_diff=350.0, volume=800.0, faraday=1.0,
                          G_K=G_K_BAR, G_Na=G_NA_BAR, G_L=G_LEAK,
                          E_Na_=E_NA, E_L=E_LEAK, C=C_M,
                          nernst_slope=26.7, E_K_rest=E_K,
                          V0=0.0, gate_state0=None, K_t0=None):
    """
    Integrate the extended (V, n, m, h, K_t) myotonia system over t.

    Defaults: tau_diff=350.0 ms, matching the lecture's own 300-400 ms
    diffusion-time estimate for a ~25 micron t-tubule (t is in ms
    throughout myotonia_model.py - an earlier pass at this used
    tau_diff=0.35, a 1000x unit bug that cleared K+ far too fast to ever
    accumulate). volume=800.0 and faraday=1.0 are convenient sim units
    (see ttubule_k_derivative's docstring), tuned by sweeping against
    frac_non_inactivating until the model actually produces an
    oscillating myotonic run rather than jumping straight from "nothing
    happens" to "depolarization block" - see myotonia_model.py for the
    full tuning story.

    K_t0 defaults to K_out (t-tubule starts equilibrated with the bulk
    extracellular space, before any spiking).

    Returns
    -------
    V, n, m, h, K_t : each an array matching t's shape
    """
    if gate_state0 is None:
        n0, m0, h0 = resting_gate_values(V0)
    else:
        n0, m0, h0 = gate_state0

    if K_t0 is None:
        K_t0 = K_out

    state0 = [V0, n0, m0, h0, K_t0]
    result = odeint(
        hh_myotonia_derivatives, state0, t,
        args=(I_e_func, frac_non_inactivating, K_out, tau_diff, volume, faraday,
              G_K, G_Na, G_L, E_Na_, E_L, C, nernst_slope, E_K_rest, None),
    )

    V_trace = result[:, 0]
    n_trace = result[:, 1]
    m_trace = result[:, 2]
    h_trace = result[:, 3]
    K_trace = result[:, 4]

    return V_trace, n_trace, m_trace, h_trace, K_trace
