"""
Bundled quantum description of a circuit.

A :class:`System` packages everything the time-evolution stage needs:

* the underlying :class:`~Circuit.Circuit` (used for accessing the
  capacitance matrix and coupling capacitance during driving),
* the **charge operator** ``n``, in both the charge and energy bases, and
* the **bare Hamiltonian** ``H0``, again in both bases.

It is the output of :func:`Quantize.quantize_transmon` and the input to
:class:`CrankNicolson.CrankNicolsonSolver`.
"""

import numpy as np

from Circuit import Circuit
from Operator import Operator


class System():
    """Quantised circuit ready for time evolution.

    Parameters
    ----------
    circuit : Circuit
        The lumped-element circuit (typically a :class:`~Transmon.Transmon`).
    charge_operator : Operator
        The dimensionless Cooper-pair number operator ``n`` in the charge
        basis and (after :func:`Quantize.quantize_transmon` runs) in the
        energy eigenbasis.
    unperturbed_hamiltonian : Operator
        The bare (drive-free) Hamiltonian ``H_0``.  Diagonal in the energy
        basis by construction.

    Attributes
    ----------
    state : Wavefunction or None
        Optional slot for the current quantum state during evolution.
        Initialised to ``None`` and left to callers to populate.
    """

    def __init__(self, circuit: Circuit, charge_operator: Operator, unperturbed_hamiltonian: Operator):
        self.circuit  = circuit
        self.n        = charge_operator
        self.H0       = unperturbed_hamiltonian
        self.state    = None
