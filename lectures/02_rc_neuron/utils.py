"""
utils.py

Shared simulation utilities for Lecture 2 (RC Model of a Neuron).

Functions here are reused across:
    - capacitor_model.py    (pure capacitor, no resistor)
    - rc_model.py            (full RC model: capacitor + leak resistor)
    - low_pass_filter.py     (pulse-width sweep, RC as a low-pass filter)
    - nernst_potential.py    (imports get_figures_dir only)
"""

import os
import sys
import numpy as np
from scipy.integrate import odeint

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from shared.plotting_utils import get_figures_dir

def step_current(t, t_on, amplitude):
    """Step current: 0 before t_on, constant `amplitude` after."""
    s_c = np.where(t > t_on, amplitude, 0)
    return s_c

def pulse_current(t, t_on, t_off, amplitude):
    """Rectangular current pulse: amplitude between t_on and t_off, else 0."""
    p_c = np.where((t > t_on) & (t < t_off), amplitude, 0)
    return p_c

def tau_from_RC(R_L, C):
    """tau = R_L * C - the membrane time constant."""
    return R_L * C

def V_inf_from_current(R_L, I_e):
    """V_inf = R_L * I_e - steady-state voltage for a given constant current (Ohm's law)."""
    return R_L * I_e

def rc_derivative(V, t, R_L, C, I_e_func):
    """dV/dt = (I_e_func(t) - V/R_L) / C - the RC ODE, in the form odeint expects."""
    I_e = I_e_func(t)
    return (I_e - V / R_L) / C

def simulate_rc_response(t, I_e_func, R_L, C, V0=0.0, discontinuities=None):
    """
    Numerically integrate the RC ODE with odeint, given a current waveform
    function I_e_func(t).

    Always pass every t_on/t_off used by I_e_func as `discontinuities` - if
    I_e_func has a hard jump (step/pulse), odeint can silently step right
    over it and return a flat wrong trace with no error. See the "Bugs I
    ran into" section of the Lecture 2 README for the full story on this
    one, it's worth reading.
    """
    V = odeint(rc_derivative, V0, t, args=(R_L, C, I_e_func), tcrit=discontinuities)
    return V.flatten()