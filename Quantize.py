"""
Canonical quantisation of the transmon circuit.

Given a built :class:`~Transmon.Transmon` (with its capacitance matrix) and
an external flux through the SQUID loop, :func:`quantize_transmon`:

1. Computes the charging energy ``E_C = e^2 / (2 C_\Sigma)``.
2. Computes the flux-tunable effective Josephson energy ``E_J(\Phi_{ext})``.
3. Builds the charge-basis Hamiltonian

   .. math::

       H_0 = 4 E_C \hat n^2 - \frac{E_J}{2} \sum_{n} (|n\rangle\langle n+1| + h.c.)

4. Diagonalises ``H_0`` and constructs the **energy-basis** representation
   of ``H_0`` (diagonal of eigenvalues) and of the charge operator ``\hat n``.

The result is bundled into a :class:`~System.System` for downstream use by
:class:`~CrankNicolson.CrankNicolsonSolver`.
"""

import numpy as np

from System import System
from Circuit import Circuit
from Operator import Operator
from constants import e
from Expression import *


def quantize(circuit: Circuit, external_flux: float, n_charge: int):
    r"""Quantise a transmon and return a ready-to-evolve :class:`System`.

    Parameters
    ----------
    transmon : Transmon
        Transmon circuit; :meth:`Circuit.build` must already have been
        called so that ``transmon.capacitance_matrix`` is populated.
    external_flux : float
        External flux through the SQUID loop, in Webers.
    n_charge : int
        Number of charge basis states; must be **odd** so that the basis
        ranges symmetrically from ``-N`` to ``+N`` with ``N = (n_charge-1)/2``.
        Larger values give better convergence but quadratic memory cost.

    Returns
    -------
    System
        Bundled circuit, charge operator, and Hamiltonian, each available
        in both the ``"charge"`` and ``"energy"`` bases.
    """

    # Symmetric integer charge ladder n in {-N, ..., +N}.
    N = int((n_charge - 1) / 2)
    charge_values = np.array([k for k in range(-N, N + 1)])

    # Cooper-pair number operator (diagonal in the charge basis).
    n = Operator(
        basis_to_matrix={"charge": np.diag(charge_values)}
    )

    # Bare Hamiltonian in the charge basis.
    H0 = Operator(
        basis_to_matrix={"charge":  Expression.realize(
            expression=circuit.hamiltonian(external_flux),
            n_charge=n_charge
            )
        }
    )

    # Diagonalise H0; columns of `energy_states` are the energy eigenvectors
    # expressed in the charge basis.  `eigh` is used because H0 is Hermitian.
    energies, energy_states = np.linalg.eigh(H0["charge"])

    # Energy-basis representation: H0 is diagonal of eigenvalues, and the
    # charge operator is rotated by the eigenvector matrix.
    H0["energy"] = np.diag(energies)
    n["energy"] = energy_states.conj().T @ n["charge"] @ energy_states

    system = System(
        circuit=circuit,
        charge_operator=n,
        unperturbed_hamiltonian=H0
    )

    return system
