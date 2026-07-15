"""
utils.py

Shared simulation utilities for Lecture 3 (Ion-Specific Conductances and
Integrate-and-Fire Models).

Functions here are reused across:
    - conductance_battery.py   (ion conductance as resistor+battery, updated V_inf)
    - integrate_and_fire.py    (no-leak and leaky IF models, threshold/reset)
    - firing_rate_curve.py     (rheobase, exact vs. linear-approx f-I curve)

step_current, pulse_current, tau_from_RC, V_inf_from_current, and
simulate_rc_response are re-exported from shared/circuit_utils.py - same
RC machinery Lecture 2 uses (with the E parameter now doing real work,
since this lecture's V_inf is battery-shifted rather than starting at 0).
get_figures_dir is re-exported the same way, from shared/plotting_utils.py.

This folder only ever imports from shared/ - never from another lecture's
folder - so it stays self-sufficient as long as shared/ is present.
"""

import os
import sys
import numpy as np
from scipy.integrate import odeint

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from shared.plotting_utils import get_figures_dir
from shared.circuit_utils import (
    step_current,
    pulse_current,
    tau_from_RC,
    V_inf_from_current,
    simulate_rc_response,
)


# ---------------------------------------------------------------------------
# Ion conductance as a resistor-in-series-with-a-battery
# ---------------------------------------------------------------------------
def driving_potential(V, E_ion):
    """V - E_ion - how far the membrane voltage is from this ion's reversal potential."""
    return V - E_ion


def ionic_current(V, G_ion, E_ion):
    """I_ion = G_ion * (V - E_ion) - conductance times driving potential."""
    return G_ion * driving_potential(V, E_ion)


# ---------------------------------------------------------------------------
# Integrate-and-fire core
# ---------------------------------------------------------------------------
def rheobase_current(G_leak, V_threshold, E_leak):
    """
    I_threshold = G_leak * (V_threshold - E_leak) - the minimum constant
    current needed for V_inf to reach V_threshold at all. Below this, the
    leaky IF neuron never spikes, no matter how long current is injected.
    """
    return G_leak * (V_threshold - E_leak)


def simulate_lif(t, I_e_func, C, V_reset, V_threshold, R_L=None, E_leak=0.0, V0=None):
    """
    Simulate a leaky (or leak-free, if R_L is None) integrate-and-fire
    neuron over the time array `t`, with hard threshold-and-reset.

    odeint alone can't do a reset event mid-integration, so this steps the
    ODE forward chunk-by-chunk: integrate until V crosses V_threshold
    (detected between two consecutive samples), record a spike at that
    crossing time, snap V back to V_reset, and continue from there.

    Returns:
        V            - voltage trace over t (shows V_reset at spike sample)
        spike_times  - list of times at which V crossed V_threshold

    Note: this detects the crossing at whatever time-sample resolution `t`
    provides - it doesn't interpolate to the exact sub-sample crossing
    time. Fine for the demos here since t is sampled finely relative to
    tau, but see the Lecture 3 README for the caveat.
    """
    if V0 is None:
        V0 = V_reset

    V_trace = np.empty_like(t, dtype=float)
    spike_times = []

    idx_start = 0
    V_current = V0

    while idx_start < len(t) - 1:
        t_chunk = t[idx_start:]

        if R_L is None:
            def deriv_no_leak(V, tt, I_e_func=I_e_func, C=C):
                return I_e_func(tt) / C
            V_chunk = odeint(deriv_no_leak, V_current, t_chunk).flatten()
        else:
            def deriv_leaky(V, tt, R_L=R_L, C=C, E_leak=E_leak, I_e_func=I_e_func):
                I_e = I_e_func(tt)
                I_leak = (V - E_leak) / R_L
                return (I_e - I_leak) / C
            V_chunk = odeint(deriv_leaky, V_current, t_chunk).flatten()

        above = V_chunk >= V_threshold
        crossing_local_idx = np.argmax(above) if np.any(above) else -1

        if crossing_local_idx <= 0:
            # no spike in this chunk (or already >= threshold at the very
            # first sample, which we don't re-trigger on) - fill in the
            # rest of the trace and stop
            V_trace[idx_start:] = V_chunk
            break

        global_crossing_idx = idx_start + crossing_local_idx
        spike_times.append(t[global_crossing_idx])

        V_trace[idx_start:global_crossing_idx] = V_chunk[:crossing_local_idx]
        V_trace[global_crossing_idx] = V_reset

        idx_start = global_crossing_idx
        V_current = V_reset

    return V_trace, spike_times


def firing_rate_no_leak(I_e, C, delta_V):
    """f = I_e / (C * delta_V) for I_e > 0, else 0 - no-leak IF firing rate."""
    I_e = np.asarray(I_e, dtype=float)
    return np.where(I_e > 0, I_e / (C * delta_V), 0.0)


def firing_rate_leaky_exact(I_e, G_leak, C, E_leak, V_threshold, V_reset):
    """
    Exact leaky-IF firing rate via the full log expression:
        delta_t = -tau * ln((V_inf - V_threshold) / (V_inf - V_reset))
        f = 1 / delta_t
    Returns 0 wherever V_inf <= V_threshold (never reaches threshold).
    """
    I_e = np.asarray(I_e, dtype=float)
    R_L = 1.0 / G_leak
    tau = tau_from_RC(R_L, C)
    V_inf = V_inf_from_current(R_L, I_e, E=E_leak)

    f = np.zeros_like(I_e, dtype=float)
    valid = V_inf > V_threshold

    ratio = np.full_like(I_e, np.nan, dtype=float)
    ratio[valid] = (V_inf[valid] - V_threshold) / (V_inf[valid] - V_reset)

    # ratio should land in (0, 1) for every physically valid spiking case;
    # guard anyway so a bad parameter combo fails quietly to f=0 instead of
    # raising from log(<=0)
    safe = valid & (ratio > 0) & (ratio < 1)
    delta_t = -tau * np.log(ratio[safe])
    f[safe] = 1.0 / delta_t

    return f


def firing_rate_leaky_linear_approx(I_e, G_leak, C, E_leak, V_threshold, V_reset):
    """
    Large-I_e linear approximation: f = (I_e - I_threshold) / (C * delta_V)
    for I_e > I_threshold, else 0. Same slope as the no-leak case, just
    shifted right by the rheobase current.
    """
    I_e = np.asarray(I_e, dtype=float)
    I_threshold = rheobase_current(G_leak, V_threshold, E_leak)
    delta_V = V_threshold - V_reset
    return np.where(I_e > I_threshold, (I_e - I_threshold) / (C * delta_V), 0.0)