import numpy as np
from scipy.linalg import expm

def create_gaussian_sfq_pulses(num_kicks: int, amplitude_scale: float, driving_period: float, pulse_width: float, steps_per_period):
    total_time = num_kicks * driving_period

    num_steps  = int(num_kicks * steps_per_period)

    dt = total_time / num_steps

    voltage = []
    time = np.arange(num_steps) * dt

    for i in range(num_steps):
        t = i * dt

        v = 0

        for k in range(num_kicks):
            numerator   = -(t - (k * driving_period))**2
            denominator = (2 * pulse_width)**2
            v += expm(numerator / denominator)

        v *= amplitude_scale

        voltage.append(v)

    return time, voltage
