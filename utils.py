"""
Helpers for synthesising drive waveforms.

Currently provides a Gaussian SFQ pulse-train generator used by the main
simulation entry point.  An alternative, physically-realistic SFQ waveform
generator based on the RCSJ model is provided in :mod:`riyas_pulses`, which
writes a CSV lookup table that can be loaded in :mod:`main` instead.
"""

import numpy as np
from scipy.linalg import expm
from Matrices import PAULI_MATRICES
from itertools import product

def create_gaussian_sfq_pulses(num_kicks: int, amplitude_scale: float, driving_period: float, pulse_width: float, steps_per_period):
    r"""Generate a uniformly-sampled train of Gaussian SFQ-like pulses.

    The resulting waveform is the sum of ``num_kicks`` Gaussians spaced by
    ``driving_period`` seconds, each with standard deviation
    ``pulse_width`` and peak amplitude ``amplitude_scale``:

    .. math::

        V(t) = A \sum_{k=0}^{N-1}
                \exp\!\Big( -\tfrac{(t - k\,T_d)^2}{(2\,\sigma)^2} \Big)

    Parameters
    ----------
    num_kicks : int
        Number of Gaussian kicks ``N`` in the train.
    amplitude_scale : float
        Peak amplitude ``A`` of each Gaussian (Volts).  Typically chosen
        so that the time integral of each pulse equals the reduced flux
        quantum (see :data:`config.AMPLITUDE_SCALE`).
    driving_period : float
        Spacing ``T_d`` between successive pulses, in seconds.
    pulse_width : float
        Gaussian standard deviation ``\sigma``, in seconds.
    steps_per_period : int
        Number of time samples used to discretise each ``driving_period``.

    Returns
    -------
    time : np.ndarray
        Uniformly spaced time grid of length ``num_kicks * steps_per_period``.
    voltage : list[float]
        Voltage samples on that grid.

    Notes
    -----
    The function uses :func:`scipy.linalg.expm` on a scalar exponent rather
    than :func:`numpy.exp`; for scalar inputs the two are mathematically
    equivalent and the result is a 0-d array that matplotlib accepts.
    """
    total_time = num_kicks * driving_period
    num_steps  = int(num_kicks * steps_per_period)
    dt = total_time / num_steps

    voltage = []
    time = np.arange(num_steps) * dt

    for i in range(num_steps):
        t = i * dt

        # Sum of Gaussians centred at t = k * T_d for k = 0, ..., N-1.
        v = 0
        for k in range(num_kicks):
            numerator   = -(t - (k * driving_period))**2
            denominator = (2 * pulse_width)**2
            v += np.exp(numerator / denominator)

        v *= amplitude_scale
        voltage.append(v)

    return np.array(time), np.array(voltage)

def get_multi_qubit_paulis(num_qubits: int):
    
    # Create all possible tensor product combinations
    full_basis = []
    for p_combo in product(PAULI_MATRICES, repeat=num_qubits):
        # Start with the first matrix in the combination
        res = p_combo[0]
        # Tensor it with the rest
        for next_p in p_combo[1:]:
            res = np.kron(res, next_p)
        full_basis.append(res)
        
    return full_basis

def get_pauli_coefs(U_q: np.ndarray):
    """
    U_q: The projected logical unitary matrix (2x2, 4x4, etc.)
    """
    d = U_q.shape[0]
    num_qubits = int(np.log2(d))
    
    if 2**num_qubits != d:
        raise ValueError(f"Matrix dimension {d} is not a power of 2.")

    # Generate the 4^N basis matrices
    pauli_basis = get_multi_qubit_paulis(num_qubits)
    
    coefs = np.zeros(len(pauli_basis), dtype=complex)

    for idx, matrix in enumerate(pauli_basis):
        coefs[idx] = np.trace(matrix @ U_q) / d
        
    return coefs