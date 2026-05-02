import os
import numpy as np
from scipy.integrate import solve_ivp

# Physical constants
q      = 1.6e-19
h_bar  = 1.05e-34
factor = h_bar / (2 * q)   # = Phi0 / (2*pi)

# Junction parameters
Ic  = 1e-03
C   = 5.7e-14
Rn  = 4
Rs  = 2
res = 1/Rn + 1/Rs

# Drive parameters
Ibias      = 0.85 * Ic
Ip         = 0.20 * Ic
Npulses    = 105
t_rec      = 1 / 5e9
rect_width = 5e-12

Phi0         = 2 * np.pi * factor
Ihigh        = Ibias + Ip
tau_slip_est = Phi0 / (res * np.sqrt(Ihigh**2 - Ic**2))
max_step     = tau_slip_est / 50


def rectangular_pulse(t, t0):
    return Ibias + Ip * float(t0 <= t <= t0 + rect_width)


def rcsj(t, y, I_of_t):
    I = I_of_t(t)
    return [y[1],
            I / (C * factor) - (Ic / (C * factor)) * np.sin(y[0]) - (res / C) * y[1]]


def phase_event_2pi(phi_ref):
    def event(t, y):
        return (y[0] - phi_ref) - 2 * np.pi
    event.terminal  = True
    event.direction = +1
    return event


y0        = np.array([np.arcsin(Ibias / Ic), 0.0])
t_pointer = 0.0

t_segs = []
y_segs = []
I_segs = []
t_events = []

for k in range(Npulses):
    t0      = t_pointer
    I_pulse = lambda t, t0=t0: rectangular_pulse(t, t0)

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

    t1 = sol.t; y1 = sol.y.T
    if t_segs:
        t1 = t1[1:]; y1 = y1[1:]
    t_segs.append(t1); y_segs.append(y1)
    I_segs.append(np.array([I_pulse(t) for t in t1]))

    # Recovery
    t2 = np.array([te, te + t_rec])
    y2 = np.array([[ye[0], ye[1]], [np.arcsin(Ibias / Ic), 0.0]])

    t_segs.append(t2[1:]); y_segs.append(y2[1:])
    I_segs.append(np.array([Ibias]))

    t_pointer = t2[-1]
    y0 = np.array([np.arcsin(Ibias / Ic), 0.0])

t_full = np.concatenate(t_segs)
y_full = np.concatenate(y_segs)
I_full = np.concatenate(I_segs)
V      = factor * y_full[:, 1]

area = np.trapezoid(V, t_full)
print(f"Area: {area:.4e} V·s  |  Phi0: {factor*2*np.pi:.4e} V·s  |  ratio: {area/(factor*2*np.pi):.6f}")
print(f"Raw solver points: {len(t_full)}")
print(f"2π events at: {[f'{te:.4e} s' for te in t_events]}")

# Downsample to a uniform 10 000-point grid for the lookup table
t_grid = np.linspace(t_full[0], t_full[-1], 100_000)
V_grid = np.interp(t_grid, t_full, V)
I_grid = np.interp(t_grid, t_full, I_full)

out_path = os.path.join(os.path.dirname(__file__), "sfq_V_lookup.csv")
header = "time_s,voltage_V,current_A"
np.savetxt(out_path, np.column_stack([t_grid, V_grid, I_grid]),
           delimiter=",", header=header, comments="", fmt="%.6e")

print(f"Lookup table written: {out_path}  ({len(t_grid)} rows)")
