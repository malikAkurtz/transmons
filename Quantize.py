import numpy as np

from System import System
from Transmon import Transmon
from DCSQUID import DCSQUID
from Operator import Operator
from constants import e

def quantize_transmon(transmon: Transmon, external_flux: float, n_charge: int):
    total_capacitance = transmon.capacitance_matrix[0][0]

    charging_energy = e**2 / (2 * total_capacitance)

    josephson_energy = DCSQUID.calculate_effective_josephson_energy(
        left_josephson_energy=transmon.dcsquid.left_josephson_energy,
        right_josephson_energy=transmon.dcsquid.right_josephson_energy,
        external_flux=external_flux
    )

    N = int((n_charge - 1) / 2)
    charge_values = np.array([k for k in range(-N, N + 1)])

    n = Operator(
        basis_to_matrix={"charge": np.diag(charge_values)}
    )

    K = Operator(
        basis_to_matrix={"charge": 4 * charging_energy * (n["charge"] @ n["charge"])}
    )

    upper_lower_matrix = np.zeros((n_charge, n_charge))

    for i in range(-N, N+1):
        upper_lower_matrix += np.outer(np.eye(n_charge)[i], np.eye(n_charge)[i+1]) + np.outer(np.eye(n_charge)[i+1], np.eye(n_charge)[i])

    U = Operator(
        basis_to_matrix={"charge": -(josephson_energy / 2) * upper_lower_matrix}
    )

    H0 = Operator(
        basis_to_matrix={"charge":  K["charge"] + U["charge"]}
    )

    energies, energy_states = np.linalg.eigh(H0["charge"])

    H0["energy"] = np.diag(energies)

    n["energy"] = energy_states.conj().T @ n["charge"] @ energy_states

    system = System(
        circuit=transmon,
        charge_operator=n,
        unperturbed_hamiltonian=H0
    )

    return system
