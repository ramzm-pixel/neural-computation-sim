"""
shared/circuit_utils.py

RC circuit building blocks that are genuinely lecture-agnostic: current
waveform generators, the tau/V_inf relations, and an odeint-based RC
simulator. Originally lived only in Lecture 2's utils.py; pulled out here
once Lecture 3 needed the same machinery (now with an added battery/E_leak
term) so neither lecture folder has to reach into the other's directory.

Each lecture folder's own utils.py re-exports whatever subset of this it
uses, same pattern as get_figures_dir in plotting_utils.py - so within a
lecture folder everything still comes from one local `from utils import
...`, and each folder stays self-sufficient assuming shared/ is present.
"""

import numpy as np
from scipy.integrate import odeint


def step_current(t, t_on, amplitude):
    """Step current: 0 before t_on, constant `amplitude` after."""
    s_c = np.where(t > t_on, amplitude, 0)
    return s_c


def pulse_current(t, t_on, t_off, amplitude):
    """Rectangular current pulse: amplitude between t_on and t_off, else 0."""
    p_c = np.where((t > t_on) & (t < t_off), amplitude, 0)
    return p_c


def tau_from_RC(R, C):
    """tau = R * C - the membrane time constant."""
    return R * C


def V_inf_from_current(R, I_e, E=0.0):
    """
    V_inf = E + R * I_e - the RC steady-state voltage.

    E defaults to 0, matching Lecture 2's leak-only model (no battery).
    Lecture 3 passes E=E_leak (or E=E_ion generally) to get the
    battery-shifted V_inf: V_inf = E_ion + R_ion * I_e.
    """
    return E + R * I_e


def rc_derivative(V, t, R, C, I_e_func, E=0.0):
    """
    dV/dt = (I_e_func(t) - (V - E)/R) / C - the general RC-with-battery ODE,
    in the form odeint expects. E=0 recovers Lecture 2's plain leak-resistor
    equation exactly.
    """
    I_e = I_e_func(t)
    I_leak = (V - E) / R
    return (I_e - I_leak) / C


def simulate_rc_response(t, I_e_func, R, C, V0=0.0, E=0.0, discontinuities=None):
    """
    Numerically integrate the RC (or RC-with-battery) ODE with odeint,
    given a current waveform function I_e_func(t).

    Always pass every t_on/t_off used by I_e_func as `discontinuities` - if
    I_e_func has a hard jump (step/pulse), odeint can silently step right
    over it and return a flat wrong trace with no error. See the "Bugs I
    ran into" section of the Lecture 2 README for the full story on this
    one, it's worth reading.

    E=0 (default) recovers Lecture 2's plain leak-only behavior exactly.
    Lecture 3 passes E=E_ion for the battery-shifted version.
    """
    V = odeint(rc_derivative, V0, t, args=(R, C, I_e_func, E), tcrit=discontinuities)
    return V.flatten()
