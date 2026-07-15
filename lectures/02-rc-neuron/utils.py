"""
utils.py

Shared simulation utilities for Lecture 2 (RC Model of a Neuron).

Functions here are reused across:
    - capacitor_model.py    (pure capacitor, no resistor)
    - rc_model.py            (full RC model: capacitor + leak resistor)
    - low_pass_filter.py     (pulse-width sweep, RC as a low-pass filter)
    - nernst_potential.py    (imports get_figures_dir only)

step_current, pulse_current, tau_from_RC, V_inf_from_current, and
simulate_rc_response are re-exported from shared/circuit_utils.py, since
Lecture 3 needs the same RC machinery (with an added battery term) and
duplicating it per-lecture would mean fixing the same bug (see the
"odeint silently stepping over a current pulse" story in the README) in
two places. get_figures_dir is re-exported the same way, from
shared/plotting_utils.py.

This folder's rc_model.py/capacitor_model.py/low_pass_filter.py all still
just do `from utils import get_figures_dir, step_current, ...` exactly as
before - only where these functions are defined has changed, not their
names or signatures.
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from shared.plotting_utils import get_figures_dir
from shared.circuit_utils import (
    step_current,
    pulse_current,
    tau_from_RC,
    V_inf_from_current,
    simulate_rc_response,
)
