"""
Default drive parameters for the Gaussian SFQ pulse train.

These constants control the train of Gaussian pulses produced by
:func:`utils.create_gaussian_sfq_pulses` and consumed by
:class:`CrankNicolson.CrankNicolsonSolver` when called from :mod:`main`.

The amplitude is normalised so that the time-integral of a single Gaussian
voltage pulse equals exactly the reduced flux quantum, mimicking an ideal
Single-Flux-Quantum (SFQ) kick.
"""

import numpy as np
from constants import REDUCED_FLUX_QUANTUM


# Number of SFQ kicks in the pulse train.
NUM_KICKS = 150

# Standard deviation of each Gaussian pulse, in seconds.
SIGMA = 1.7e-11

# Amplitude scaling chosen so that  \int V(t) dt  per pulse equals the
# reduced flux quantum  Phi_0^* = hbar/(2e).  For a unit Gaussian of width
# SIGMA this requires V_peak = Phi_0^* / (SIGMA * sqrt(2 pi)).
AMPLITUDE_SCALE = REDUCED_FLUX_QUANTUM / (SIGMA * np.sqrt(2 * np.pi))

# Detuning between the pulse repetition period and the qubit period 1/f_01.
# Set to zero for the resonant case.
DETUNING = 0

# Time-grid resolution: number of integration steps per qubit period.
STEPS_PER_PERIOD = 200