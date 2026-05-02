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
from Transmon import Transmon
from DCSQUID import DCSQUID
from Operator import Operator
from constants import e


def quantize_transmon(transmon: Transmon, external_flux: float, n_charge: int):
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
    # Total island capacitance to ground; for the single-node transmon
    # this is just the [0,0] entry of the reduced C matrix.
    total_capacitance = transmon.capacitance_matrix[0][0]

    # Charging energy E_C = e^2 / (2 C_\Sigma).
    charging_energy = e**2 / (2 * total_capacitance)

    # Flux-tunable effective Josephson energy of the SQUID.
    josephson_energy = DCSQUID.calculate_effective_josephson_energy(
        left_josephson_energy=transmon.dcsquid.left_josephson_energy,
        right_josephson_energy=transmon.dcsquid.right_josephson_energy,
        external_flux=external_flux
    )

    # Symmetric integer charge ladder n in {-N, ..., +N}.
    N = int((n_charge - 1) / 2)
    charge_values = np.array([k for k in range(-N, N + 1)])

    # Cooper-pair number operator (diagonal in the charge basis).
    n = Operator(
        basis_to_matrix={"charge": np.diag(charge_values)}
    )

    # Kinetic term K = 4 E_C n^2.
    K = Operator(
        basis_to_matrix={"charge": 4 * charging_energy * (n["charge"] @ n["charge"])}
    )

    # Build the cosine potential -E_J cos(\hat\phi) in the charge basis.
    # In this basis cos(\hat\phi) = 1/2 (S + S^†) where S is the unit-shift
    # operator |n><n+1|.  We assemble the symmetric off-diagonal matrix
    # (S + S^†) here as `upper_lower_matrix`.
    upper_lower_matrix = np.zeros((n_charge, n_charge))
    for i in range(-N, N+1):
        upper_lower_matrix += np.outer(np.eye(n_charge)[i], np.eye(n_charge)[i+1]) + np.outer(np.eye(n_charge)[i+1], np.eye(n_charge)[i])

    # Potential term U = -(E_J / 2) * (S + S^†).
    U = Operator(
        basis_to_matrix={"charge": -(josephson_energy / 2) * upper_lower_matrix}
    )

    # Bare Hamiltonian in the charge basis.
    H0 = Operator(
        basis_to_matrix={"charge":  K["charge"] + U["charge"]}
    )

    # Diagonalise H0; columns of `energy_states` are the energy eigenvectors
    # expressed in the charge basis.  `eigh` is used because H0 is Hermitian.
    energies, energy_states = np.linalg.eigh(H0["charge"])

    # Energy-basis representation: H0 is diagonal of eigenvalues, and the
    # charge operator is rotated by the eigenvector matrix.
    H0["energy"] = np.diag(energies)
    n["energy"] = energy_states.conj().T @ n["charge"] @ energy_states

    system = System(
        circuit=transmon,
        charge_operator=n,
        unperturbed_hamiltonian=H0
    )

    return system
