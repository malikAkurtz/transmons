"""
RCSJ-based generation of a realistic SFQ voltage waveform.

This standalone script integrates the Resistively and Capacitively Shunted
Junction (RCSJ) equation for a Josephson junction biased near its critical
current and pulsed by a train of rectangular current excursions.  Each
excursion drives the phase across one ``2 \pi`` slip, producing a
single-flux-quantum (SFQ) voltage pulse of area exactly ``\Phi_0``.

The simulated voltage waveform is downsampled onto a uniform time grid and
written to ``sfq_V_lookup.csv`` so that :mod:`main` can load it as the drive
waveform instead of the analytic Gaussian train from
:func:`utils.create_gaussian_sfq_pulses`.

Run as::

    python riyas_pulses.py

The script prints diagnostics about each pulse (the area should be very
close to ``\Phi_0`` and the ratio close to ``1.0``) and writes the lookup
table next to itself.
"""

import os
import numpy as np
from scipy.integrate import solve_ivp

# ---------------------------------------------------------------------
# Physical constants and reduced-flux factor.
# ---------------------------------------------------------------------
q      = 1.6e-19          # Elementary charge [C]
h_bar  = 1.05e-34         # Reduced Planck constant [J s]
factor = h_bar / (2 * q)  # Reduced flux quantum Phi_0/(2 pi) [V s]

# ---------------------------------------------------------------------
# Junction parameters (RCSJ model).
# ---------------------------------------------------------------------
Ic  = 1e-03   # Critical current        [A]
C   = 5.7e-14 # Junction capacitance    [F]
Rn  = 4       # Normal-state resistance [Ohm]
Rs  = 2       # Shunt resistance        [Ohm]
res = 1/Rn + 1/Rs  # Combined parallel conductance [S]

# ---------------------------------------------------------------------
# Drive parameters.
# ---------------------------------------------------------------------
Ibias      = 0.85 * Ic   # DC bias current
Ip         = 0.20 * Ic   # Rectangular pulse amplitude on top of Ibias
Npulses    = 105         # Number of SFQ pulses to emit
t_rec      = 1 / 5e9     # Recovery time between pulses (5 GHz repetition)
rect_width = 5e-12       # Rectangular pulse width [s]

Phi0         = 2 * np.pi * factor  # Flux quantum [V s]
Ihigh        = Ibias + Ip
# Estimated phase-slip duration; used to bound the integrator's max step
# so that the slip is well-resolved.
tau_slip_est = Phi0 / (res * np.sqrt(Ihigh**2 - Ic**2))
max_step     = tau_slip_est / 50


def rectangular_pulse(t, t0):
    """Bias-plus-rectangular drive current at time ``t``.

    Returns ``Ibias + Ip`` while ``t`` lies inside ``[t0, t0 + rect_width]``
    and ``Ibias`` everywhere else.
    """
    return Ibias + Ip * float(t0 <= t <= t0 + rect_width)


def rcsj(t, y, I_of_t):
    r"""Right-hand side of the RCSJ equation in dimensionless form.

    State vector ``y = [phi, phi_dot]`` where ``phi`` is the
    superconducting phase across the junction.  The equation is

    .. math::

        \ddot \phi = \frac{I(t)}{C\,\Phi_0/(2\pi)}
                   - \frac{I_c}{C\,\Phi_0/(2\pi)} \sin\phi
                   - \frac{1/R}{C}\dot \phi
    """
    I = I_of_t(t)
    return [y[1],
            I / (C * factor) - (Ic / (C * factor)) * np.sin(y[0]) - (res / C) * y[1]]


def phase_event_2pi(phi_ref):
    """Build a SciPy event that triggers when the phase has slipped by 2π.

    The event fires (and terminates integration of the current segment) the
    first time ``phi(t) - phi_ref == 2 pi`` with positive direction —
    i.e. exactly one SFQ slip.
    """
    def event(t, y):
        return (y[0] - phi_ref) - 2 * np.pi
    event.terminal  = True
    event.direction = +1
    return event


# ---------------------------------------------------------------------
# Initial conditions: phase parked at the static-bias fixed point.
# ---------------------------------------------------------------------
y0        = np.array([np.arcsin(Ibias / Ic), 0.0])
t_pointer = 0.0

t_segs = []   # Time segments produced by each pulse + recovery.
y_segs = []   # Corresponding state segments.
I_segs = []   # Drive current samples on each segment.
t_events = [] # Times at which each 2π slip occurred.

for k in range(Npulses):
    t0      = t_pointer
    I_pulse = lambda t, t0=t0: rectangular_pulse(t, t0)

    # Integrate the RCSJ equation forwards until the phase slips by 2π,
    # at which point the event fires and integration stops.
    sol = solve_ivp(
        lambda t, y: rcsj(t, y, I_pulse),
        [t0, t0 + rect_width + 10e-12],
        y0,
        method="RK45",
        events=phase_event_2pi(y0[0]),
        rtol=1e-10, atol=1e-13,
        max_step=max_step,
    )

    if not sol.t_events[0].size:
        raise RuntimeError(f"Pulse {k+1}: No 2π slip detected.")

    te = sol.t_events[0][0]
    ye = sol.y_events[0][0]
    t_events.append(te)

    # Append the active part of the trajectory.
    t1 = sol.t; y1 = sol.y.T
    if t_segs:
        # Drop the duplicate seam point shared with the previous segment.
        t1 = t1[1:]; y1 = y1[1:]
    t_segs.append(t1); y_segs.append(y1)
    I_segs.append(np.array([I_pulse(t) for t in t1]))

    # Recovery: fast jump back to the static fixed point.  We do this
    # synthetically (two-point segment) rather than re-integrating to keep
    # the script fast.
    t2 = np.array([te, te + t_rec])
    y2 = np.array([[ye[0], ye[1]], [np.arcsin(Ibias / Ic), 0.0]])

    t_segs.append(t2[1:]); y_segs.append(y2[1:])
    I_segs.append(np.array([Ibias]))

    t_pointer = t2[-1]
    y0 = np.array([np.arcsin(Ibias / Ic), 0.0])

# Concatenate all segments into one timeline and convert phi_dot -> voltage
# using the Josephson voltage relation V = (hbar / 2e) * d phi / d t.
t_full = np.concatenate(t_segs)
y_full = np.concatenate(y_segs)
I_full = np.concatenate(I_segs)
V      = factor * y_full[:, 1]

# Sanity-check: integrated voltage per pulse should equal Phi_0.
area = np.trapezoid(V, t_full)
print(f"Area: {area:.4e} V·s  |  Phi0: {factor*2*np.pi:.4e} V·s  |  ratio: {area/(factor*2*np.pi):.6f}")
print(f"Raw solver points: {len(t_full)}")
print(f"2π events at: {[f'{te:.4e} s' for te in t_events]}")

# Resample onto a 100k-point uniform grid for fast lookup at solve-time.
t_grid = np.linspace(t_full[0], t_full[-1], 100_000)
V_grid = np.interp(t_grid, t_full, V)
I_grid = np.interp(t_grid, t_full, I_full)

# Persist next to this script.
out_path = os.path.join(os.path.dirname(__file__), "sfq_V_lookup.csv")
header = "time_s,voltage_V,current_A"
np.savetxt(out_path, np.column_stack([t_grid, V_grid, I_grid]),
           delimiter=",", header=header, comments="", fmt="%.6e")

print(f"Lookup table written: {out_path}  ({len(t_grid)} rows)")
